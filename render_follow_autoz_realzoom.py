import pandas as pd, numpy as np, cv2, os, subprocess
from scipy.signal import savgol_filter

def letterbox_to_size(img, out_w, out_h):
    """
    Resize to fit inside (out_w,out_h) without warping.
    Pads equally on all sides using BORDER_REPLICATE so the edges don't go black.
    """
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)
    scale = min(out_w / w, out_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    pad_left  = (out_w - new_w) // 2
    pad_right = out_w - new_w - pad_left
    pad_top   = (out_h - new_h) // 2
    pad_bot   = out_h - new_h - pad_top
    return cv2.copyMakeBorder(resized, pad_top, pad_bot, pad_left, pad_right, cv2.BORDER_REPLICATE)


# ---- Inputs / outputs ----
clip       = r'.\out\atomic_clips\004__GOAL__t266.50-t283.10.mp4'
track_csv  = r'.\out\autoframe_work\ball_track.csv'
out_mp4    = r'.\out\reels\tiktok\004__directed_SINGLECHAIN_BALLTRACK_AUTOZ_REAL.mp4'

# ---- Output geometry ----
W_out, H_out = 608, 1080
aspect = W_out / H_out

# ---- Behavior knobs ----
tau = 0.24              # s of look-ahead (pan lead)
# Pan limits (px/s and px/s^2) — apply to both x and y
slew_max_pan   = 300.0
accel_max_pan  = 1400.0
# Zoom: min/max scale (relative to widest same-aspect crop)
min_scale      = 0.60   # smaller = tighter zoom-in allowed (e.g., 0.60 ≈ 1.67x in)
# Speed that counts as “fast” play (px/s). Faster => zoom out
speed_fast_px  = 900.0
# Zoom responsiveness (scale change per second, and its accel)
max_zoom_rate   = 1.5    # scale units / s
max_zoom_accel  = 5.0    # scale units / s^2
zoom_hysteresis = 0.035  # scale units; must exceed this to change target
# Ball placement inside the crop (fractions of crop size)
ball_left_frac = 0.45   # a little look-ahead to the right
ball_top_frac  = 0.55   # a tad below center so we see space ahead

# ---- Read source video ----
cap = cv2.VideoCapture(clip)
fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

# Max same-aspect crop (widest view without letterboxing)
max_w = int(np.floor(H * aspect / 2) * 2)
max_w = min(max_w, W)
max_h = int(np.floor(max_w / aspect / 2) * 2)
# the above ensures (max_w / max_h) == aspect and fits inside the frame

# Min crop (tightest zoom)
min_w = int(np.floor(max_w * min_scale / 2) * 2)
min_h = int(np.floor(max_h * min_scale / 2) * 2)
min_w = max(192, min_w)
min_h = max(192, min_h)

# ---- Load and smooth track ----
df = pd.read_csv(track_csv)
def smooth_series(s_vals, fps):
    n = len(s_vals)
    if n < 5: return s_vals
    win = max(5, int(round(fps*0.5)))
    if win % 2 == 0: win += 1
    if win > n: win = (n if n % 2 == 1 else n-1)
    if win < 5: return s_vals
    return savgol_filter(s_vals, window_length=win, polyorder=2)

for col in ['cx','cy']:
    s = pd.to_numeric(df[col], errors='coerce').ffill(limit=10).bfill(limit=10)
    df[col] = smooth_series(s.to_numpy(), fps)

cx = pd.to_numeric(df['cx'], errors='coerce').to_numpy()
cy = pd.to_numeric(df['cy'], errors='coerce').to_numpy()
# ensure no NaNs
if np.isnan(cx).any():
    idx = np.where(~np.isnan(cx))[0]
    cx = np.interp(np.arange(len(cx)), idx, cx[idx])
if np.isnan(cy).any():
    idx = np.where(~np.isnan(cy))[0]
    cy = np.interp(np.arange(len(cy)), idx, cy[idx])

# Velocities (px/s)
vx = np.gradient(cx) * fps
vy = np.gradient(cy) * fps

# Lead the ball (more lead horizontally than vertically)
lead_x = cx + tau*vx
lead_y = cy + (tau*0.5)*vy

# ---- Zoom target from speed (with hysteresis) ----
speed = np.hypot(vx, vy)
speed_smooth = smooth_series(speed, fps)
speed_norm = np.clip(speed_smooth / max(1e-6, speed_fast_px), 0.0, 1.0)
# scale_target: 1.0 = max view (zoomed out), min_scale = tight zoom-in
scale_raw = min_scale + (1.0 - min_scale) * speed_norm
scale_raw = np.clip(scale_raw, min_scale, 1.0)

scale_target = np.zeros_like(scale_raw)
scale_target[0] = float(scale_raw[0])
for i in range(1, len(scale_raw)):
    prev = scale_target[i-1]
    raw  = scale_raw[i]
    if raw > prev + zoom_hysteresis:
        scale_target[i] = raw
    elif raw < prev - zoom_hysteresis:
        scale_target[i] = raw
    else:
        scale_target[i] = prev

# ---- Apply slew/accel limits to x, y, and scale ----
N  = len(cx)
dt = 1.0 / fps

x  = np.zeros(N); y  = np.zeros(N); s = np.zeros(N)
vx_c = vy_c = vs = 0.0

# initialize at first targets
s[0] = float(scale_target[0])
w0, h0 = s[0]*max_w, s[0]*max_h
x_des0 = np.clip(lead_x[0] - ball_left_frac*w0, 0, W - w0)
y_des0 = np.clip(lead_y[0] - ball_top_frac*h0, 0, H - h0)
x[0] = float(x_des0)
y[0] = float(y_des0)

for i in range(1, N):
    # --- scale (zoom) dynamics in scale-units ---
    err_s   = scale_target[i] - s[i-1]
    v_des_s = np.clip(err_s/dt, -max_zoom_rate, max_zoom_rate)
    dv_s    = np.clip(v_des_s - vs, -max_zoom_accel*dt, max_zoom_accel*dt)
    vs     += dv_s
    vs      = np.clip(vs, -max_zoom_rate, max_zoom_rate)
    s[i]    = s[i-1] + vs*dt
    s[i]    = float(np.clip(s[i], min_scale, 1.0))

    # derive current crop size
    wi, hi = s[i]*max_w, s[i]*max_h

    # --- x pan ---
    desired_left = np.clip(lead_x[i] - ball_left_frac*wi, 0, W - wi)
    err_x   = desired_left - x[i-1]
    v_des_x = np.clip(err_x/dt, -slew_max_pan, slew_max_pan)
    dv_x    = np.clip(v_des_x - vx_c, -accel_max_pan*dt, accel_max_pan*dt)
    vx_c   += dv_x
    vx_c    = np.clip(vx_c, -slew_max_pan, slew_max_pan)
    x[i]    = x[i-1] + vx_c*dt
    x[i]    = float(np.clip(x[i], 0, W - wi))

    # --- y pan ---
    desired_top  = np.clip(lead_y[i] - ball_top_frac*hi, 0, H - hi)
    err_y   = desired_top - y[i-1]
    v_des_y = np.clip(err_y/dt, -slew_max_pan, slew_max_pan)
    dv_y    = np.clip(v_des_y - vy_c, -accel_max_pan*dt, accel_max_pan*dt)
    vy_c   += dv_y
    vy_c    = np.clip(vy_c, -slew_max_pan, slew_max_pan)
    y[i]    = y[i-1] + vy_c*dt
    y[i]    = float(np.clip(y[i], 0, H - hi))

# ---- Optional debug dump
dbg_dir = r'.\out\autoframe_work'
os.makedirs(dbg_dir, exist_ok=True)
pd.DataFrame({
    'frame': df['frame'], 't': df['time'],
    'cx': cx, 'cy': cy, 'vx': vx, 'vy': vy,
    'scale_target': scale_target,
    'scale': s, 'x': x, 'y': y
}).to_csv(os.path.join(dbg_dir, 'virtual_cam_autoz_real.csv'), index=False)

# ---- Render per frame (crop WxH box, then resize to 608x1080)
os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
tmp = os.path.join(dbg_dir, '_temp_frames_autoz_real')
os.makedirs(tmp, exist_ok=True)

cap = cv2.VideoCapture(clip)
i = 0
while True:
    ok, bgr = cap.read()
    if not ok: break
    si = float(s[i] if i < len(s) else s[-1])
    wi = int(round(si * max_w))
    hi = int(round(si * max_h))
    xi = int(round(x[i] if i < len(x) else x[-1]))
    yi = int(round(y[i] if i < len(y) else y[-1]))
    xi = max(0, min(xi, W - wi))
    yi = max(0, min(yi, H - hi))
    crop = bgr[yi:yi+hi, xi:xi+wi]
    # edge safety
    if crop.shape[1] != wi or crop.shape[0] != hi:
        pad_r = max(0, wi - crop.shape[1])
        pad_b = max(0, hi - crop.shape[0])
        if pad_r or pad_b:
            crop = cv2.copyMakeBorder(crop, 0, pad_b, 0, pad_r, cv2.BORDER_REPLICATE)
    crop = cv2.resize(crop, (W_out, H_out), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(os.path.join(tmp, f'f_{i:06d}.jpg'), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
    i += 1
cap.release()

# ---- Encode with source audio
subprocess.run([
    'ffmpeg','-y',
    '-framerate', str(int(round(fps))),
    '-i', os.path.join(tmp, 'f_%06d.jpg'),
    '-i', clip,
    '-filter_complex','[0:v]colorspace=iall=bt470bg:all=bt709:irange=pc:range=tv,format=yuv420p[v]',
    '-map','[v]','-map','1:a:0?',
    '-c:v','libx264','-preset','veryfast','-crf','19',
    '-x264-params','keyint=120:min-keyint=120:scenecut=0',
    '-pix_fmt','yuv420p','-profile:v','high','-level','4.0',
    '-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709',
    '-shortest','-movflags','+faststart',
    '-c:a','aac','-b:a','128k',
    out_mp4
], check=True)

print('Wrote', out_mp4)

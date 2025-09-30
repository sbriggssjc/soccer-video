import pandas as pd, numpy as np, cv2, os, subprocess
from scipy.signal import savgol_filter

clip       = r'.\out\atomic_clips\004__GOAL__t266.50-t283.10.mp4'
track_csv  = r'.\out\autoframe_work\ball_track.csv'
out_mp4    = r'.\out\reels\tiktok\004__directed_SINGLECHAIN_BALLTRACK_AUTOZ_GENTLE.mp4'

W_out, H_out = 608, 1080
aspect = W_out / H_out

# ---------------- gentler behavior knobs ----------------
tau = 0.18                 # less lead so pan & zoom feel calmer
slew_max_pan   = 300.0     # keep pan snappy enough
accel_max_pan  = 1400.0

min_scale      = 0.85      # only mild punch-in (0.85..1.00)
speed_fast_px  = 1000.0    # treat play as 'fast' a bit later
# zoom slew/accel are intentionally tiny
slew_max_zoom  = 0.35      # scale units / s (slow!)
accel_max_zoom = 1.0       # scale units / s^2 (slow!)
per_frame_scale_cap = 0.0025  # **max 0.25% scale change per frame**

# deadband + long smoothing to avoid zoom 'breathing'
zoom_deadband = 0.02       # ignore target changes < 2% scale
ema_seconds   = 1.5        # long EMA over ~1.5 s for scale target

ball_left_frac = 0.46
ball_top_frac  = 0.54

# --------------------------------------------------------
cap = cv2.VideoCapture(clip)
fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

max_w = int(np.floor(H * aspect / 2) * 2)
max_w = min(max_w, W)
max_h = int(np.floor(max_w / aspect / 2) * 2)

min_w = int(np.floor(max_w * min_scale / 2) * 2)
min_h = int(np.floor(max_h * min_scale / 2) * 2)
min_w = max(192, min_w); min_h = max(192, min_h)

df = pd.read_csv(track_csv)

def smooth_series(arr, fps):
    n = len(arr)
    if n < 5: return arr
    win = max(5, int(round(fps*0.5)))
    if win % 2 == 0: win += 1
    if win > n: win = (n if n % 2 == 1 else n-1)
    if win < 5: return arr
    return savgol_filter(arr, window_length=win, polyorder=2)

for col in ['cx','cy']:
    s = pd.to_numeric(df[col], errors='coerce').ffill(limit=10).bfill(limit=10)
    df[col] = smooth_series(s.to_numpy(), fps)

cx = pd.to_numeric(df['cx'], errors='coerce').to_numpy()
cy = pd.to_numeric(df['cy'], errors='coerce').to_numpy()
if np.isnan(cx).any():
    idx = np.where(~np.isnan(cx))[0]
    cx = np.interp(np.arange(len(cx)), idx, cx[idx])
if np.isnan(cy).any():
    idx = np.where(~np.isnan(cy))[0]
    cy = np.interp(np.arange(len(cy)), idx, cy[idx])

vx = np.gradient(cx) * fps
vy = np.gradient(cy) * fps
lead_x = cx + tau*vx
lead_y = cy + (tau*0.6)*vy

# ---------- GENTLE ZOOM TARGET ----------
speed = np.hypot(vx, vy)
speed_norm = np.clip(speed / max(1e-6, speed_fast_px), 0.0, 1.0)
# slow -> min_scale, fast -> 1.0 (zoom out)
scale_target = 1.0 - (1.0 - min_scale)*(1.0 - speed_norm)

# long smoothing + hysteresis (deadband) + per-frame cap
scale_target = smooth_series(scale_target, fps)
alpha = 2.0 / (fps*ema_seconds + 1.0)           # EMA over ~ema_seconds
zt = np.zeros_like(scale_target)
zt[0] = scale_target[0]
for i in range(1, len(scale_target)):
    desired = scale_target[i]
    # deadband around previous target
    if abs(desired - zt[i-1]) < zoom_deadband:
        desired = zt[i-1]
    # EMA towards desired
    ema = zt[i-1] + alpha*(desired - zt[i-1])
    # per-frame absolute cap
    cap_step = per_frame_scale_cap
    zt[i] = zt[i-1] + np.clip(ema - zt[i-1], -cap_step, cap_step)
scale_target = np.clip(zt, min_scale, 1.0)

# crop size targets
w_t = scale_target * max_w
h_t = scale_target * max_h
x_t = np.clip(lead_x - ball_left_frac * w_t, 0, W - w_t)
y_t = np.clip(lead_y - ball_top_frac  * h_t, 0, H - h_t)

# ---------- DYNAMICS ----------
N  = len(cx); dt = 1.0 / fps
x  = np.zeros(N); y  = np.zeros(N); s = np.zeros(N)
vx_c = vy_c = vs = 0.0

s[0] = float(scale_target[0])
w0, h0 = s[0]*max_w, s[0]*max_h
x[0] = float(np.clip(x_t[0], 0, W - w0))
y[0] = float(np.clip(y_t[0], 0, H - h0))

for i in range(1, N):
    # scale (very slow)
    err_s   = scale_target[i] - s[i-1]
    v_des_s = np.clip(err_s/dt, -slew_max_zoom, slew_max_zoom)
    dv_s    = np.clip(v_des_s - vs, -accel_max_zoom*dt, accel_max_zoom*dt)
    vs      = np.clip(vs + dv_s, -slew_max_zoom, slew_max_zoom)
    s[i]    = float(np.clip(s[i-1] + vs*dt, min_scale, 1.0))

    wi, hi = s[i]*max_w, s[i]*max_h

    # x pan
    err_x   = (lead_x[i] - ball_left_frac*wi) - x[i-1]
    v_des_x = np.clip(err_x/dt, -slew_max_pan, slew_max_pan)
    dv_x    = np.clip(v_des_x - vx_c, -accel_max_pan*dt, accel_max_pan*dt)
    vx_c    = np.clip(vx_c + dv_x, -slew_max_pan, slew_max_pan)
    x[i]    = float(np.clip(x[i-1] + vx_c*dt, 0, W - wi))

    # y pan
    err_y   = (lead_y[i] - ball_top_frac*hi) - y[i-1]
    v_des_y = np.clip(err_y/dt, -slew_max_pan, slew_max_pan)
    dv_y    = np.clip(v_des_y - vy_c, -accel_max_pan*dt, accel_max_pan*dt)
    vy_c    = np.clip(vy_c + dv_y, -slew_max_pan, slew_max_pan)
    y[i]    = float(np.clip(y[i-1] + vy_c*dt, 0, H - hi))

dbg_dir = r'.\out\autoframe_work'
os.makedirs(dbg_dir, exist_ok=True)
pd.DataFrame({'frame':df['frame'], 't':df['time'], 'cx':cx, 'cy':cy,
              'scale':s, 'x':x, 'y':y}).to_csv(
    os.path.join(dbg_dir, 'virtual_cam_autoz_gentle.csv'), index=False)

os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
tmp = os.path.join(dbg_dir, '_temp_frames_autoz_gentle')
os.makedirs(tmp, exist_ok=True)

cap = cv2.VideoCapture(clip)
i = 0
while True:
    ok, bgr = cap.read()
    if not ok: break
    si = float(s[i] if i < len(s) else s[-1])
    wi = int(round(si * max_w)); hi = int(round(si * max_h))
    xi = int(round(x[i] if i < len(x) else x[-1]))
    yi = int(round(y[i] if i < len(y) else y[-1]))
    xi = max(0, min(xi, W - wi)); yi = max(0, min(yi, H - hi))
    crop = bgr[yi:yi+hi, xi:xi+wi]
    if crop.shape[1] != wi or crop.shape[0] != hi:
        pad_r = max(0, wi - crop.shape[1]); pad_b = max(0, hi - crop.shape[0])
        if pad_r or pad_b:
            crop = cv2.copyMakeBorder(crop, 0, pad_b, 0, pad_r, cv2.BORDER_REPLICATE)
    crop = cv2.resize(crop, (W_out, H_out), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(os.path.join(tmp, f'f_{i:06d}.jpg'), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
    i += 1
cap.release()

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

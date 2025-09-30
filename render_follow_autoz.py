import pandas as pd, numpy as np, cv2, os, subprocess
from scipy.signal import savgol_filter

# ---- Inputs / outputs ----
clip       = r'.\out\atomic_clips\004__GOAL__t266.50-t283.10.mp4'
track_csv  = r'.\out\autoframe_work\ball_track.csv'   # from track_ball.py
out_mp4    = r'.\out\reels\tiktok\004__directed_SINGLECHAIN_BALLTRACK_AUTOZ.mp4'

# ---- Output geometry ----
W_out, H_out = 608, 1080

# ---- Camera behavior knobs (tweak here) ----
# Lead time (s) so we don't arrive late to the cross
tau = 0.24
# Pan limits
slew_max_pan   = 280.0   # px/s
accel_max_pan  = 1200.0  # px/s^2
# Zoom limits (min/max crop width in source pixels).
# Max width is fixed by aspect (no true "zoom out" beyond that).
min_w_ratio    = 0.62    # 0.62*max_w ~= 1.6x zoom-in; raise to zoom less, lower to zoom more
# Zoom responsiveness driven by ball speed (px/s)
speed_fast_px  = 900.0   # speed where we treat play as "full fast" (zoomed out)
# Zoom limits for how quickly zoom can change
slew_max_zoom  = 900.0   # px/s change in crop width
accel_max_zoom = 3000.0  # px/s^2 change in crop width
# Place ball horizontally within the crop
ball_left_frac = 0.45    # 0=left edge, 0.5=center; 0.45 gives look-ahead to the right

# ---- Read source video info ----
cap = cv2.VideoCapture(clip)
fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

# Base crop width to preserve 608x1080 aspect (this is the widest view)
max_w = int(np.floor(H*W_out/H_out/2)*2)
max_w = min(max_w, W)  # safety
min_w = int(np.floor(max_w*min_w_ratio/2)*2)
min_w = max(192, min_w)  # hard lower bound for safety

# ---- Load and prep track ----
df = pd.read_csv(track_csv)

# Interpolate short gaps (<=10 frames), smooth with ~0.5s SavGol
def smooth_series(s_vals, fps):
    n = len(s_vals)
    if n < 5: return s_vals
    win = max(5, int(round(fps*0.5)))
    if win % 2 == 0: win += 1
    if win > n: win = (n if n % 2 == 1 else n-1)
    if win < 5: return s_vals
    return savgol_filter(s_vals, window_length=win, polyorder=2)

for col in ['cx','cy']:
    s = pd.to_numeric(df[col], errors='coerce')
    s = s.ffill(limit=10).bfill(limit=10)
    vals = s.to_numpy()
    vals = smooth_series(vals, fps)
    df[col] = vals

cx = pd.to_numeric(df['cx'], errors='coerce').to_numpy()
if np.isnan(cx).any():
    idx = np.where(~np.isnan(cx))[0]
    cx = np.interp(np.arange(len(cx)), idx, cx[idx])

# Velocity in px/s
vx   = np.gradient(cx) * fps
lead = cx + tau*vx

# ---- Desired position & zoom per frame (targets) ----
# speed_norm: 0=standing still, 1=full fast
speed = np.abs(vx)
speed_norm = np.clip(speed / max(1e-6, speed_fast_px), 0.0, 1.0)

# width_target: zoom out (w->max_w) when fast; zoom in (w->min_w) when slow
width_target = max_w - (1.0 - speed_norm) * (max_w - min_w)
# smooth width target slightly to avoid noise
width_target = smooth_series(width_target, fps)

# left-edge target makes the ball sit at ball_left_frac across the crop
x_target = np.clip(lead - ball_left_frac*width_target, 0, W - width_target)

# ---- Apply slew/accel limits to both pan (x) and zoom (w) ----
N    = len(cx)
dt   = 1.0 / fps

x    = np.zeros(N); x[0] = float(x_target[0])
v_x  = 0.0
w    = np.zeros(N); w[0] = float(width_target[0])
v_w  = 0.0

for i in range(1, N):
    # Pan
    err_x   = x_target[i] - x[i-1]
    v_des_x = np.clip(err_x/dt, -slew_max_pan, slew_max_pan)
    dv_x    = np.clip(v_des_x - v_x, -accel_max_pan*dt, accel_max_pan*dt)
    v_x    += dv_x
    v_x     = np.clip(v_x, -slew_max_pan, slew_max_pan)
    x[i]    = x[i-1] + v_x*dt

    # Zoom width
    err_w   = width_target[i] - w[i-1]
    v_des_w = np.clip(err_w/dt, -slew_max_zoom, slew_max_zoom)
    dv_w    = np.clip(v_des_w - v_w, -accel_max_zoom*dt, accel_max_zoom*dt)
    v_w    += dv_w
    v_w     = np.clip(v_w, -slew_max_zoom, slew_max_zoom)
    w[i]    = w[i-1] + v_w*dt

    # Bounds: clamp width first, then re-clamp x to keep crop inside frame
    w[i] = float(np.clip(w[i], min_w, max_w))
    x[i] = float(np.clip(x[i], 0, W - w[i]))

# ---- Debug path (optional)
dbg_dir = r'.\out\autoframe_work'
os.makedirs(dbg_dir, exist_ok=True)
pd.DataFrame({
    'frame': df['frame'],
    't':     df['time'],
    'cx':    cx,
    'x':     x,
    'w':     w,
    'speed': speed
}).to_csv(os.path.join(dbg_dir, 'virtual_cam_autoz.csv'), index=False)

# ---- Render per frame
os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
tmp = os.path.join(dbg_dir, '_temp_frames_autoz')
os.makedirs(tmp, exist_ok=True)

cap = cv2.VideoCapture(clip)
i = 0
while True:
    ok, bgr = cap.read()
    if not ok: break
    wi = int(round(w[i] if i < len(w) else w[-1]))
    xi = int(round(x[i] if i < len(x) else x[-1]))
    xi = max(0, min(xi, W - wi))
    crop = bgr[:, xi:xi+wi]
    # guard against edge rounding
    if crop.shape[1] != wi:
        pad = wi - crop.shape[1]
        if pad > 0:
            crop = cv2.copyMakeBorder(crop, 0, 0, 0, pad, cv2.BORDER_REPLICATE)
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

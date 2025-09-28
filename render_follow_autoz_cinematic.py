import pandas as pd, numpy as np, cv2, os, subprocess
from scipy.signal import savgol_filter

clip       = r'.\out\atomic_clips\004__GOAL__t266.50-t283.10.mp4'
track_csv  = r'.\out\autoframe_work\ball_track.csv'
out_mp4    = r'.\out\reels\tiktok\004__directed_SINGLECHAIN_BALLTRACK_AUTOZ_CINEMATIC.mp4'

W_out, H_out = 608, 1080
aspect = W_out / H_out

# ---------------- CINEMATIC TUNING ----------------
# Pan lead and dynamics
tau = 0.15                    # a hair less lead than before
slew_max_pan   = 260.0        # px/s (lower = calmer)
accel_max_pan  = 900.0        # px/s^2
jerk_max_pan   = 2400.0       # px/s^3 cap on accel change per second

# Zoom tiers (gentle) — actual zoom-in is small:
S_WIDE   = 1.00               # full frame
S_MED    = 0.93               # mild punch-in
S_TIGHT  = 0.87               # modest punch-in
tiers    = np.array([S_TIGHT, S_MED, S_WIDE])  # we’ll pick by index 0..2

# Predictive window & thresholds
peek_sec = 0.8                # how far ahead we 'look' to pick zoom
speed_fast_px = 950.0         # ≈ when play is moving quickly (px/s)
speed_slow_px = 420.0         # ≈ when play is slow/stationary

# Hysteresis + dwell
zoom_deadband = 0.015         # ignore small target shifts
min_hold_sec  = 1.25          # don't leave a tier before this time passes
ema_seconds   = 2.2           # very slow EMA on scale target
per_frame_scale_cap = 0.0015  # **max 0.15% scale change per frame**

# Composition (ball offset inside crop)
ball_left_frac = 0.46
ball_top_frac  = 0.54
# ---------------------------------------------------

cap = cv2.VideoCapture(clip)
fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

# Base (max) crop size for 608x1080 pillarbox
max_w = int(np.floor(H * aspect / 2) * 2)
max_w = min(max_w, W)
max_h = int(np.floor(max_w / aspect / 2) * 2)

df = pd.read_csv(track_csv)

def smooth_series(arr, fps):
    n = len(arr)
    if n < 5: return arr
    win = max(5, int(round(fps*0.5)))  # ~0.5 s window
    if win % 2 == 0: win += 1
    if win > n: win = (n if n % 2 == 1 else n-1)
    if win < 5: return arr
    return savgol_filter(arr, window_length=win, polyorder=2)

# Clean & smooth detections
for col in ['cx','cy']:
    s = pd.to_numeric(df[col], errors='coerce').ffill(limit=10).bfill(limit=10)
    df[col] = smooth_series(s.to_numpy(), fps)

cx = pd.to_numeric(df['cx'], errors='coerce').to_numpy()
cy = pd.to_numeric(df['cy'], errors='coerce').to_numpy()
# Fill any remaining NaNs
if np.isnan(cx).any():
    idx = np.where(~np.isnan(cx))[0]
    cx = np.interp(np.arange(len(cx)), idx, cx[idx])
if np.isnan(cy).any():
    idx = np.where(~np.isnan(cy))[0]
    cy = np.interp(np.arange(len(cy)), idx, cy[idx])

# Velocity
vx = np.gradient(cx) * fps
vy = np.gradient(cy) * fps
speed = np.hypot(vx, vy)

# Predictive 'peek' using future max speed in a window
N = len(cx)
peek = int(round(peek_sec * fps))
future_speed = speed.copy()
for i in range(N):
    j2 = min(N, i + peek)
    future_speed[i] = np.max(speed[i:j2]) if i < j2 else speed[i]

# Decide zoom tier with hysteresis + dwell
tier_idx = np.zeros(N, dtype=int)  # 0=tight, 1=med, 2=wide
current = 2  # start wide
hold_left = 0.0

# Thresholds with hysteresis: to go tighter we require slower future speed
to_med_fast  = speed_fast_px * 0.85
to_tight_slow= speed_slow_px * 0.90
to_wide_fast = speed_fast_px * 1.05
to_med_slow  = speed_slow_px * 1.10

for i in range(N):
    fs = future_speed[i]
    # enforce dwell
    if hold_left > 0:
        tier_idx[i] = current
        hold_left -= 1.0/fps
        continue
    # propose transition
    nxt = current
    if current == 2:  # wide -> med if future not too fast
        if fs < to_med_fast:
            nxt = 1
    elif current == 1:
        if fs < to_tight_slow:
            nxt = 0   # med -> tight when future is really calm
        elif fs > to_wide_fast:
            nxt = 2   # med -> wide if future ramps fast
    else:  # tight
        if fs > to_med_slow:
            nxt = 1   # tight -> med if action building

    if nxt != current:
        current = nxt
        hold_left = min_hold_sec
    tier_idx[i] = current

# Smooth tier changes to a target scale (then EMA + per-frame cap)
scale_target_discrete = tiers[tier_idx]

# Very slow EMA on target + deadband + tiny cap
alpha = 2.0 / (fps*ema_seconds + 1.0)
s_target = np.zeros_like(scale_target_discrete, dtype=float)
s_target[0] = float(scale_target_discrete[0])
for i in range(1, N):
    desired = scale_target_discrete[i]
    if abs(desired - s_target[i-1]) < zoom_deadband:
        desired = s_target[i-1]
    ema = s_target[i-1] + alpha*(desired - s_target[i-1])
    step_cap = per_frame_scale_cap
    s_target[i] = s_target[i-1] + np.clip(ema - s_target[i-1], -step_cap, step_cap)

s_target = np.clip(s_target, S_TIGHT, S_WIDE)

# Lead the ball a bit for pan
lead_x = cx + tau*vx
lead_y = cy + (tau*0.6)*vy

# Compute pan targets based on scale
w_t = s_target * max_w
h_t = s_target * max_h
x_t = np.clip(lead_x - ball_left_frac * w_t, 0, W - w_t)
y_t = np.clip(lead_y - ball_top_frac  * h_t, 0, H - h_t)

# Pan dynamics with accel + jerk limiting
x = np.zeros(N); y = np.zeros(N); s = np.zeros(N)
vx_c = vy_c = vs = 0.0
ax_c = ay_c = 0.0
dt = 1.0 / fps

s[0] = float(s_target[0])
wi, hi = s[0]*max_w, s[0]*max_h
x[0] = float(np.clip(x_t[0], 0, W - wi))
y[0] = float(np.clip(y_t[0], 0, H - hi))

for i in range(1, N):
    # scale (already slow-targeted; just drift to it softly)
    err_s   = s_target[i] - s[i-1]
    v_des_s = np.clip(err_s/dt, -0.25, 0.25)  # even softer per-second slew
    dv_s    = np.clip(v_des_s - vs, -0.6*dt, 0.6*dt)
    vs      = np.clip(vs + dv_s, -0.35, 0.35)
    s[i]    = float(np.clip(s[i-1] + vs*dt, S_TIGHT, S_WIDE))

    wi, hi = s[i]*max_w, s[i]*max_h

    # X pan with jerk cap
    err_x   = (lead_x[i] - ball_left_frac*wi) - x[i-1]
    v_des_x = np.clip(err_x/dt, -slew_max_pan, slew_max_pan)
    a_des_x = np.clip((v_des_x - vx_c)/dt, -accel_max_pan, accel_max_pan)
    # jerk limit
    da_x    = np.clip(a_des_x - ax_c, -jerk_max_pan*dt, jerk_max_pan*dt)
    ax_c    = np.clip(ax_c + da_x, -accel_max_pan, accel_max_pan)
    vx_c    = np.clip(vx_c + ax_c*dt, -slew_max_pan, slew_max_pan)
    x[i]    = float(np.clip(x[i-1] + vx_c*dt, 0, W - wi))

    # Y pan with jerk cap
    err_y   = (lead_y[i] - ball_top_frac*hi) - y[i-1]
    v_des_y = np.clip(err_y/dt, -slew_max_pan, slew_max_pan)
    a_des_y = np.clip((v_des_y - vy_c)/dt, -accel_max_pan, accel_max_pan)
    da_y    = np.clip(a_des_y - ay_c, -jerk_max_pan*dt, jerk_max_pan*dt)
    ay_c    = np.clip(ay_c + da_y, -accel_max_pan, accel_max_pan)
    vy_c    = np.clip(vy_c + ay_c*dt, -slew_max_pan, slew_max_pan)
    y[i]    = float(np.clip(y[i-1] + vy_c*dt, 0, H - hi))

# Debug dump (optional)
dbg_dir = r'.\out\autoframe_work'
os.makedirs(dbg_dir, exist_ok=True)
pd.DataFrame({
    'frame':df['frame'], 't':df['time'], 'cx':cx, 'cy':cy, 'speed':speed,
    'future_speed':future_speed, 'tier':tier_idx, 's_target':s_target,
    'x':x, 'y':y, 'scale':s
}).to_csv(os.path.join(dbg_dir, 'virtual_cam_autoz_cinematic.csv'), index=False)

# Render frames
os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
tmp = os.path.join(dbg_dir, '_temp_frames_autoz_cinematic')
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

# Encode with source audio
subprocess.run([
    'ffmpeg','-y',
    '-framerate', str(int(round(fps))),
    '-i', os.path.join(tmp, 'f_%06d.jpg'),
    '-i', clip,
    '-map','0:v','-map','1:a:0?',
    '-c:v','libx264','-preset','veryfast','-crf','19',
    '-x264-params','keyint=120:min-keyint=120:scenecut=0',
    '-pix_fmt','yuv420p','-profile:v','high','-level','4.0',
    '-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709',
    '-shortest','-movflags','+faststart',
    '-c:a','aac','-b:a','128k',
    out_mp4
], check=True)

print('Wrote', out_mp4)

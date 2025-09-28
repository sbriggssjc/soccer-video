import pandas as pd, numpy as np, cv2, os, subprocess
from scipy.signal import savgol_filter

clip       = r'.\out\atomic_clips\004__GOAL__t266.50-t283.10.mp4'
track_csv  = r'.\out\autoframe_work\ball_track.csv'
out_mp4    = r'.\out\reels\tiktok\004__directed_SINGLECHAIN_BALLTRACK_AUTOZ_CINEMATIC_v3.mp4'

W_out, H_out = 608, 1080
aspect = W_out / H_out

# ===================== TUNING =====================
attack_dir = 'right'  # 'right' or 'left' (set correct!)
goalward_sign = 1.0 if attack_dir == 'right' else -1.0

# Zoom tiers (gentle; TIGHTER for dribbles)
S_TIGHTER = 0.88
S_TIGHT   = 0.90
S_MED     = 0.95
S_WIDE    = 1.00
tiers     = np.array([S_TIGHT, S_MED, S_WIDE])  # default state mapping

# Zoom smoothing & dwell (very steady)
zoom_deadband       = 0.025
min_hold_sec        = 1.90
ema_seconds         = 3.40
per_frame_scale_cap = 0.0009

# Pan lead & damping (calmer)
tau = 0.14
slew_max_pan   = 200.0   # px/s
accel_max_pan  = 600.0   # px/s^2
jerk_max_pan   = 1400.0  # px/s^3

# Predictive window
peek_sec = 0.9

# Dribble (tighter & more certain)
dribble_speed_lo = 200.0
dribble_speed_hi = 650.0
dribble_dir_std  = 40.0   # deg

# Cross detection & behavior
cross_speed_min     = 800.0
cross_lateral_ratio = 1.6      # |vx| dominates |vy|
flank_frac          = 0.55     # ball beyond this side is 'flank' toward attacking goal
cross_lock_sec      = 0.55     # hold MED zoom & anticipate landing
ball_left_frac_base = 0.46
ball_top_frac       = 0.54
ball_left_frac_cross= 0.38     # give more look-ahead during cross

# Shot trigger (after cross lock, allow tighter)
shot_speed_min  = 1000.0
attack_third_fr = 0.62
# ==================================================

cap = cv2.VideoCapture(clip)
fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

max_w = int(np.floor(H * aspect / 2) * 2); max_w = min(max_w, W)
max_h = int(np.floor(max_w / aspect / 2) * 2)

df = pd.read_csv(track_csv)

def smooth_series(arr, fps):
    n = len(arr)
    if n < 5: return arr
    win = max(5, int(round(fps*0.5)))  # ~0.5s
    if win % 2 == 0: win += 1
    if win > n: win = (n if n % 2 == 1 else n-1)
    if win < 5: return arr
    return savgol_filter(arr, window_length=win, polyorder=2)

# fill short gaps + smooth
for col in ['cx','cy']:
    s = pd.to_numeric(df[col], errors='coerce').ffill(limit=10).bfill(limit=10)
    df[col] = smooth_series(s.to_numpy(), fps)

cx = pd.to_numeric(df['cx'], errors='coerce').to_numpy()
cy = pd.to_numeric(df['cy'], errors='coerce').to_numpy()
# fill any remaining NaNs
if np.isnan(cx).any():
    idx = np.where(~np.isnan(cx))[0]
    cx = np.interp(np.arange(len(cx)), idx, cx[idx])
if np.isnan(cy).any():
    idx = np.where(~np.isnan(cy))[0]
    cy = np.interp(np.arange(len(cy)), idx, cy[idx])

vx = np.gradient(cx) * fps
vy = np.gradient(cy) * fps
speed = np.hypot(vx, vy)
N = len(cx)

peek = int(round(peek_sec * fps))
future_speed_max = speed.copy()
future_vx_mean = np.zeros(N)
future_vy_mean = np.zeros(N)
future_dir_std = np.zeros(N)

for i in range(N):
    j2 = min(N, i + max(1, peek))
    seg_vx = vx[i:j2]; seg_vy = vy[i:j2]; seg_sp = speed[i:j2]
    if seg_sp.size: future_speed_max[i] = np.max(seg_sp)
    if seg_vx.size:
        future_vx_mean[i] = np.mean(seg_vx)
        future_vy_mean[i] = np.mean(seg_vy)
        ang = np.degrees(np.arctan2(seg_vy, seg_vx + 1e-6))
        future_dir_std[i] = np.std(ang)
    else:
        future_vx_mean[i] = vx[i]
        future_vy_mean[i] = vy[i]
        future_dir_std[i] = 180.0

# regions & signs
if attack_dir == 'right':
    flank_ok = cx > W*flank_frac
    in_final_third = cx > W*attack_third_fr
    landing_x_center = W*0.88
else:
    flank_ok = cx < W*(1.0-flank_frac)
    in_final_third = cx < W*(1.0-attack_third_fr)
    landing_x_center = W*0.12

# --- State logic: DRIBBLE / CROSS_LOCK / FREE ---
tier = np.full(N, 2, dtype=int)  # start WIDE
s_disc = np.zeros(N)
cross_lock_left = 0.0
current = 2
hold_left = 0.0

for i in range(N):
    fs   = future_speed_max[i]
    gw   = future_vx_mean[i] * goalward_sign > 0
    lat_ratio = abs(vx[i])/(abs(vy[i])+1e-6)
    is_cross_like = (fs >= cross_speed_min) and (lat_ratio >= cross_lateral_ratio) and gw and flank_ok[i]
    is_dribble    = (dribble_speed_lo <= fs <= dribble_speed_hi) and (future_dir_std[i] <= dribble_dir_std)
    is_shot_like  = (fs >= shot_speed_min) and gw and in_final_third[i]

    # Enter CROSS_LOCK if cross-like; reset timer
    if is_cross_like:
        cross_lock_left = cross_lock_sec

    # While CROSS_LOCK: force MED zoom tier
    if cross_lock_left > 0:
        desired_state = 1  # MED
        cross_lock_left -= 1.0/fps
    else:
        # After cross window, allow shot/dribble decisions
        if is_shot_like:
            desired_state = 0  # TIGHT
        elif is_dribble:
            desired_state = 0  # TIGHT
        else:
            # calm default: MED unless extremely fast chaos
            desired_state = 1 if fs < shot_speed_min*0.85 else 2

    if hold_left <= 0 and desired_state != current:
        current = desired_state
        hold_left = min_hold_sec

    tier[i] = current
    if hold_left > 0: hold_left -= 1.0/fps

# map tier -> base discrete scale, but allow TIGHTER specifically for dribble frames
base_scale = tiers[tier]
for i in range(N):
    if (dribble_speed_lo <= future_speed_max[i] <= dribble_speed_hi) and (future_dir_std[i] <= dribble_dir_std):
        base_scale[i] = min(base_scale[i], S_TIGHTER)

# slow EMA + deadband + per-frame cap
alpha = 2.0 / (fps*ema_seconds + 1.0)
s_target = np.zeros(N)
s_target[0] = float(base_scale[0])
for i in range(1, N):
    desired = base_scale[i]
    if abs(desired - s_target[i-1]) < zoom_deadband:
        desired = s_target[i-1]
    ema = s_target[i-1] + alpha*(desired - s_target[i-1])
    step = np.clip(ema - s_target[i-1], -per_frame_scale_cap, per_frame_scale_cap)
    s_target[i] = float(np.clip(s_target[i-1] + step, S_TIGHTER, S_WIDE))

# Lead
lead_x = cx + tau*vx
lead_y = cy + (tau*0.6)*vy

# Compose; during cross lock blend toward landing zone and keep ball further left (more look-ahead)
x = np.zeros(N); y = np.zeros(N); s = np.zeros(N)
vx_c = vy_c = vs = 0.0
ax_c = ay_c = 0.0
dt = 1.0/fps

s[0] = float(s_target[0])
wi = s[0]*max_w; hi = s[0]*max_h
ball_left_frac = ball_left_frac_base
x[0] = float(np.clip(lead_x[0] - ball_left_frac*wi, 0, W-wi))
y[0] = float(np.clip(lead_y[0] - ball_top_frac*hi, 0, H-hi))

cross_lock_left = 0.0  # recompute on the fly for composition too
for i in range(1, N):
    fs   = future_speed_max[i]
    gw   = future_vx_mean[i] * goalward_sign > 0
    lat_ratio = abs(vx[i])/(abs(vy[i])+1e-6)
    is_cross_like = (fs >= cross_speed_min) and (lat_ratio >= cross_lateral_ratio) and gw and flank_ok[i]
    if is_cross_like:
        cross_lock_left = cross_lock_sec
    cross_active = cross_lock_left > 0
    if cross_active: cross_lock_left -= dt

    # SCALE (very deliberate)
    err_s   = s_target[i] - s[i-1]
    v_des_s = np.clip(err_s/dt, -0.18, 0.18)
    dv_s    = np.clip(v_des_s - vs, -0.40*dt, 0.40*dt)
    vs      = np.clip(vs + dv_s, -0.25, 0.25)
    s[i]    = float(np.clip(s[i-1] + vs*dt, S_TIGHTER, S_WIDE))

    wi = s[i]*max_w; hi = s[i]*max_h

    # During cross: shift composition and anticipate landing zone
    if cross_active:
        ball_left_frac_now = ball_left_frac_cross
        # blend the left edge toward a landing corridor
        desired_left_from_ball = lead_x[i] - ball_left_frac_now*wi
        desired_left_from_land = landing_x_center - 0.50*wi
        x_tgt = 0.6*desired_left_from_ball + 0.4*desired_left_from_land
    else:
        x_tgt = lead_x[i] - ball_left_frac_base*wi

    y_tgt = lead_y[i] - ball_top_frac*hi

    # PAN with accel + jerk limits (calm)
    # X
    err_x   = np.clip(x_tgt, 0, W-wi) - x[i-1]
    v_des_x = np.clip(err_x/dt, -slew_max_pan, slew_max_pan)
    a_des_x = np.clip((v_des_x - vx_c)/dt, -accel_max_pan, accel_max_pan)
    da_x    = np.clip(a_des_x - ax_c, -jerk_max_pan*dt, jerk_max_pan*dt)
    ax_c    = np.clip(ax_c + da_x, -accel_max_pan, accel_max_pan)
    vx_c    = np.clip(vx_c + ax_c*dt, -slew_max_pan, slew_max_pan)
    x[i]    = float(np.clip(x[i-1] + vx_c*dt, 0, W-wi))

    # Y
    err_y   = np.clip(y_tgt, 0, H-hi) - y[i-1]
    v_des_y = np.clip(err_y/dt, -slew_max_pan, slew_max_pan)
    a_des_y = np.clip((v_des_y - vy_c)/dt, -accel_max_pan, accel_max_pan)
    da_y    = np.clip(a_des_y - ay_c, -jerk_max_pan*dt, jerk_max_pan*dt)
    ay_c    = np.clip(ay_c + da_y, -accel_max_pan, accel_max_pan)
    vy_c    = np.clip(vy_c + ay_c*dt, -slew_max_pan, slew_max_pan)
    y[i]    = float(np.clip(y[i-1] + vy_c*dt, 0, H-hi))

# (optional) debug
dbg_dir = r'.\out\autoframe_work'
os.makedirs(dbg_dir, exist_ok=True)
pd.DataFrame({
    'frame':df['frame'],'t':df['time'],
    'cx':cx,'cy':cy,'speed':speed,
    'future_speed_max':future_speed_max,'future_dir_std':future_dir_std,
    'tier':tier,'s_target':s_target,'scale':s,'x':x,'y':y
}).to_csv(os.path.join(dbg_dir, 'virtual_cam_autoz_cinematic_v3.csv'), index=False)

# Render
os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
tmp = os.path.join(dbg_dir, '_temp_frames_autoz_cinematic_v3')
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

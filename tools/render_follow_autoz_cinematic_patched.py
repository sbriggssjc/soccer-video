import argparse
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patched cinematic auto-zoom renderer")
    parser.add_argument("--in", "--src", dest="input", required=True,
                        help="Input MP4 clip")
    parser.add_argument("--out", dest="output",
                        help="Output MP4 path (defaults to <clip>.__CINEMATIC.mp4)")
    parser.add_argument("--track", dest="track_csv",
                        help="Ball track CSV (defaults to work dir / ball_track.csv)")
    parser.add_argument("--work-dir", dest="work_dir",
                        help="Working directory root (defaults to out/autoframe_work/cinematic/<clip_stem>)")
    parser.add_argument("--width", type=int, default=608,
                        help="Output width")
    parser.add_argument("--height", type=int, default=1080,
                        help="Output height")
    parser.add_argument("--attack-dir", choices=["left", "right"], default="right",
                        help="Team attack direction for heuristics")
    parser.add_argument("--clean-temp", action="store_true",
                        help="Clean working directory before rendering")
    return parser.parse_args()


args = parse_args()
input_path = Path(args.input).expanduser().resolve()
if not input_path.exists():
    raise SystemExit(f"Input clip not found: {input_path}")

work_root = Path(args.work_dir).expanduser().resolve() if args.work_dir else (
    Path("out") / "autoframe_work" / "cinematic" / input_path.stem
)
if args.clean_temp and work_root.exists():
    shutil.rmtree(work_root)
work_root.mkdir(parents=True, exist_ok=True)

clip = str(input_path)
track_csv_path = Path(args.track_csv).expanduser().resolve() if args.track_csv else (work_root / "ball_track.csv")
track_csv_path.parent.mkdir(parents=True, exist_ok=True)
track_csv = str(track_csv_path)

output_path = Path(args.output).expanduser().resolve() if args.output else (
    input_path.with_name(input_path.name + ".__CINEMATIC.mp4")
)
output_path.parent.mkdir(parents=True, exist_ok=True)
out_mp4 = str(output_path)

W_out, H_out = int(args.width), int(args.height)
if W_out <= 0 or H_out <= 0:
    raise SystemExit("Output dimensions must be positive")
aspect = W_out / H_out

attack_dir = args.attack_dir

clip       = r'.\out\atomic_clips\004__GOAL__t266.50-t283.10.mp4'
track_csv  = r'.\out\autoframe_work\ball_track.csv'
out_mp4    = r'.\out\reels\tiktok\004__directed_SINGLECHAIN_BALLTRACK_AUTOZ_CINEMATIC.mp4'

W_out, H_out = 608, 1080
aspect = W_out / H_out

# ===================== TUNING =====================
# FIELD / TEAM DIRECTION (set to 'right' if attacking right-to-left on the screen feels wrong)
# attack_dir already configured via CLI (args.attack_dir)

# Zoom tiers (gentle)
S_WIDE  = 1.00
S_MED   = 0.95
S_TIGHT = 0.90
tiers   = np.array([S_TIGHT, S_MED, S_WIDE])  # 0..2

# Zoom smoothing & dwell (steadier)
zoom_deadband         = 0.020    # ignore micro target changes
min_hold_sec          = 1.70     # minimum time to stay in a tier
ema_seconds           = 3.00     # slower EMA -> steadier zoom
per_frame_scale_cap   = 0.0010   # <= 0.10% change per frame

# Pan lead & damping
tau = 0.14                       # lead seconds (slightly shorter for tighter comp)
slew_max_pan   = 220.0           # px/s (lower = calmer)
accel_max_pan  = 700.0           # px/s^2
jerk_max_pan   = 1600.0          # px/s^3

# Predictive window
peek_sec = 0.9                   # look ahead

# Dribble/shot logic thresholds
# - Dribble: moderate speed & stable direction -> go TIGHT
# - Shot: fast, goal-ward & in attacking third -> go TIGHT
dribble_speed_lo  = 250.0
dribble_speed_hi  = 800.0
dribble_dir_std   = 55.0         # direction stability (deg) over peek window

shot_speed_min    = 1050.0
attack_third_frac = 0.62         # x beyond this is 'final third' for attacks

# Composition (ball placement inside crop)
ball_left_frac = 0.46
ball_top_frac  = 0.54
# ==================================================

cap = cv2.VideoCapture(clip)
fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

# Base (max) crop size for pillarbox
max_w = int(np.floor(H * aspect / 2) * 2)
max_w = min(max_w, W)
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

# forward-looking features
future_speed_max = speed.copy()
future_vx_mean = np.zeros(N)
future_vy_mean = np.zeros(N)
future_dir_std = np.zeros(N)

for i in range(N):
    j2 = min(N, i + peek) if peek > 1 else i+1
    seg_vx = vx[i:j2]
    seg_vy = vy[i:j2]
    seg_sp = speed[i:j2]
    future_speed_max[i] = np.max(seg_sp) if seg_sp.size else speed[i]
    if seg_vx.size:
        future_vx_mean[i] = np.mean(seg_vx)
        future_vy_mean[i] = np.mean(seg_vy)
        # direction stability in degrees
        ang = np.degrees(np.arctan2(seg_vy, seg_vx+1e-6))
        future_dir_std[i] = np.std(ang)
    else:
        future_vx_mean[i] = vx[i]
        future_vy_mean[i] = vy[i]
        future_dir_std[i] = 180.0

# field direction & goal-ward sign
goalward_sign = 1.0 if attack_dir == 'right' else -1.0
x_goal_th = W * (attack_third_frac if attack_dir == 'right' else (1.0 - attack_third_frac))

# pick zoom tier with hysteresis + dwell + action cues
tier = np.full(N, 2, dtype=int)  # start WIDE
current = 2
hold_left = 0.0

for i in range(N):
    if hold_left > 0:
        tier[i] = current
        hold_left -= 1.0/fps
        continue

    fs   = future_speed_max[i]
    gw   = future_vx_mean[i] * goalward_sign > 0  # moving toward goal?
    in3  = (cx[i] > x_goal_th) if attack_dir == 'right' else (cx[i] < x_goal_th)
    dir_stable = (future_dir_std[i] <= dribble_dir_std)

    next_state = current

    # SHOT: very fast, goal-ward, in final third -> TIGHT
    if fs >= shot_speed_min and gw and in3:
        next_state = 0
    else:
        # DRIBBLE: moderate speed and stable direction -> TIGHT
        if (dribble_speed_lo <= fs <= dribble_speed_hi) and dir_stable:
            next_state = 0
        else:
            # otherwise MED for middling action; WIDE for chaos
            if fs > shot_speed_min * 0.85:
                next_state = 2
            else:
                next_state = 1

    if next_state != current:
        current = next_state
        hold_left = min_hold_sec

    tier[i] = current

# map to discrete scale, then slow EMA with deadband + per-frame cap
scale_discrete = tiers[tier]
alpha = 2.0 / (fps*ema_seconds + 1.0)
s_target = np.zeros_like(scale_discrete, dtype=float)
s_target[0] = float(scale_discrete[0])
for i in range(1, N):
    desired = scale_discrete[i]
    if abs(desired - s_target[i-1]) < zoom_deadband:
        desired = s_target[i-1]
    ema = s_target[i-1] + alpha*(desired - s_target[i-1])
    step = np.clip(ema - s_target[i-1], -per_frame_scale_cap, per_frame_scale_cap)
    s_target[i] = s_target[i-1] + step
s_target = np.clip(s_target, S_TIGHT, S_WIDE)

# lead & compose
lead_x = cx + tau*vx
lead_y = cy + (tau*0.6)*vy

w_t = s_target * max_w
h_t = s_target * max_h
x_t = np.clip(lead_x - ball_left_frac * w_t, 0, W - w_t)
y_t = np.clip(lead_y - ball_top_frac  * h_t, 0, H - h_t)

# damped pan with accel + jerk limit
x = np.zeros(N); y = np.zeros(N); s = np.zeros(N)
vx_c = vy_c = vs = 0.0
ax_c = ay_c = 0.0
dt = 1.0 / fps

s[0] = float(s_target[0])
wi, hi = s[0]*max_w, s[0]*max_h
x[0] = float(np.clip(x_t[0], 0, W - wi))
y[0] = float(np.clip(y_t[0], 0, H - hi))

for i in range(1, N):
    # scale (very deliberate)
    err_s   = s_target[i] - s[i-1]
    v_des_s = np.clip(err_s/dt, -0.20, 0.20)  # softer zoom velocity
    dv_s    = np.clip(v_des_s - vs, -0.45*dt, 0.45*dt)
    vs      = np.clip(vs + dv_s, -0.28, 0.28)
    s[i]    = float(np.clip(s[i-1] + vs*dt, S_TIGHT, S_WIDE))

    wi, hi = s[i]*max_w, s[i]*max_h

    # X
    err_x   = (lead_x[i] - ball_left_frac*wi) - x[i-1]
    v_des_x = np.clip(err_x/dt, -slew_max_pan, slew_max_pan)
    a_des_x = np.clip((v_des_x - vx_c)/dt, -accel_max_pan, accel_max_pan)
    da_x    = np.clip(a_des_x - ax_c, -jerk_max_pan*dt, jerk_max_pan*dt)
    ax_c    = np.clip(ax_c + da_x, -accel_max_pan, accel_max_pan)
    vx_c    = np.clip(vx_c + ax_c*dt, -slew_max_pan, slew_max_pan)
    x[i]    = float(np.clip(x[i-1] + vx_c*dt, 0, W - wi))

    # Y
    err_y   = (lead_y[i] - ball_top_frac*hi) - y[i-1]
    v_des_y = np.clip(err_y/dt, -slew_max_pan, slew_max_pan)
    a_des_y = np.clip((v_des_y - vy_c)/dt, -accel_max_pan, accel_max_pan)
    da_y    = np.clip(a_des_y - ay_c, -jerk_max_pan*dt, jerk_max_pan*dt)
    ay_c    = np.clip(ay_c + da_y, -accel_max_pan, accel_max_pan)
    vy_c    = np.clip(vy_c + ay_c*dt, -slew_max_pan, slew_max_pan)
    y[i]    = float(np.clip(y[i-1] + vy_c*dt, 0, H - hi))

# (optional) debug dump
dbg_dir = str(work_root)
os.makedirs(dbg_dir, exist_ok=True)
pd.DataFrame({
    'frame':df['frame'], 't':df['time'],
    'cx':cx,'cy':cy, 'speed':speed,
    'future_speed_max':future_speed_max,
    'future_dir_std':future_dir_std,
    'tier':tier, 's_target':s_target,
    'x':x,'y':y,'scale':s
}).to_csv(os.path.join(dbg_dir, 'virtual_cam_autoz_cinematic_v2.csv'), index=False)

# Render
os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
frames_dir = Path(dbg_dir) / 'frames'
frames_dir.mkdir(parents=True, exist_ok=True)
tmp = str(frames_dir)

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

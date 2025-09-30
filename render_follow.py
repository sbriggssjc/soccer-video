import pandas as pd, numpy as np, cv2, os, subprocess
from scipy.signal import savgol_filter

clip = r'.\out\atomic_clips\004__GOAL__t266.50-t283.10.mp4'
track_csv = r'.\out\autoframe_work\ball_track.csv'
out_mp4 = r'.\out\reels\tiktok\004__directed_SINGLECHAIN_BALLTRACK.mp4'

W_out, H_out = 608, 1080
cap = cv2.VideoCapture(clip)
fps = cap.get(cv2.CAP_PROP_FPS)
W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

# target crop width to preserve 608x1080 aspect, even number
crop_w = int(np.floor(H*W_out/H_out/2)*2)
crop_w = min(crop_w, W)  # safety
half   = crop_w/2

df = pd.read_csv(track_csv)

# interpolate short gaps (<=10 frames) only
for col in ['cx','cy']:
    s = pd.to_numeric(df[col], errors='coerce')
    s = s.ffill(limit=10).bfill(limit=10)
    df[col] = s

# Savitzky-Golay smoothing (about 0.5 s window, must be odd and <= series length)
def smooth_series(arr, fps):
    n = len(arr)
    if n < 5:
        return arr
    win = max(5, int(round(fps*0.5)))   # ~0.5 s
    if win % 2 == 0: win += 1           # must be odd
    if win > n: win = (n if n % 2 == 1 else n-1)
    if win < 5:  return arr
    return savgol_filter(arr, window_length=win, polyorder=2)

for col in ['cx','cy']:
    vals = pd.to_numeric(df[col], errors='coerce').to_numpy()
    df[col] = smooth_series(vals, fps)

# Lead the camera with velocity so we don't arrive late to the cross
tau = 0.24  # seconds of look-ahead; raise to follow faster play
cx = pd.to_numeric(df['cx'], errors='coerce').to_numpy()
# if any NaNs remain, fill with nearest valid to keep shapes sane
if np.isnan(cx).any():
    idx = np.where(~np.isnan(cx))[0]
    cx = np.interp(np.arange(len(cx)), idx, cx[idx])
vx = np.gradient(cx) * fps
lead_cx = cx + tau*vx

# Desired left edge: keep ball ~45% from left (look-ahead room)
target = np.clip(lead_cx - 0.45*crop_w, 0, W - crop_w)

# Slew/accel limits - bump these to ensure the cam can keep up with the cross
slew_max  = 280.0   # px/s  (try 220-320)
accel_max = 1200.0  # px/s^2 (try 800-1500)

x = np.zeros_like(target)
x[0] = target[0]
v = 0.0
dt = 1.0 / max(fps, 1e-6)

for i in range(1, len(target)):
    err   = target[i] - x[i-1]
    v_des = np.clip(err/dt, -slew_max, slew_max)
    dv    = np.clip(v_des - v, -accel_max*dt, accel_max*dt)
    v    += dv
    v     = np.clip(v, -slew_max, slew_max)
    x[i]  = x[i-1] + v*dt
    x[i]  = float(np.clip(x[i], 0, W - crop_w))

# Optional: write debug path
dbg_dir = r'.\out\autoframe_work'
os.makedirs(dbg_dir, exist_ok=True)
pd.DataFrame({'frame':df['frame'], 't':df['time'], 'cx':cx, 'x':x}).to_csv(
    os.path.join(dbg_dir, 'virtual_cam.csv'), index=False)

# Render per frame
os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
tmp = os.path.join(dbg_dir, '_temp_frames')
os.makedirs(tmp, exist_ok=True)

cap = cv2.VideoCapture(clip)
i = 0
while True:
    ok, bgr = cap.read()
    if not ok: break
    xi = int(round(x[i] if i < len(x) else x[-1]))
    xi = max(0, min(xi, W - crop_w))
    crop = bgr[:, xi:xi+int(crop_w)]
    if crop.shape[1] != int(crop_w):
        # last-ditch safety if indexing ever goes weird
        pad = int(crop_w) - crop.shape[1]
        crop = cv2.copyMakeBorder(crop, 0, 0, 0, pad, cv2.BORDER_REPLICATE)
    crop = cv2.resize(crop, (W_out, H_out), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(os.path.join(tmp, f'f_{i:06d}.jpg'), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
    i += 1
cap.release()

# Encode with source audio (if present)
subprocess.run([
    'ffmpeg','-y',
    '-framerate', str(int(round(fps if fps and fps>0 else 24))),
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


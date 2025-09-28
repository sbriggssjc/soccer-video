import argparse, os, subprocess, math
import numpy as np, pandas as pd, cv2
from scipy.signal import savgol_filter

def smooth_series(arr, fps):
    n = len(arr)
    if n < 5:
        return arr
    win = max(5, int(round(fps * 0.5)))  # ~0.5s window
    if win % 2 == 0: win += 1
    if win > n: win = (n if n % 2 == 1 else n-1)
    if win < 5: return arr
    return savgol_filter(arr, window_length=win, polyorder=2)

def lerp(a,b,t): return a + (b-a)*t
def clamp(x,a,b): return a if x < a else b if x > b else x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clip', required=True)
    ap.add_argument('--track_csv', required=True)
    ap.add_argument('--out_mp4', required=True)
    ap.add_argument('--tau', type=float, default=0.26)
    ap.add_argument('--slew', type=float, default=210.0)
    ap.add_argument('--accel', type=float, default=750.0)
    ap.add_argument('--left_frac', type=float, default=0.44)
    ap.add_argument('--W_out', type=int, default=608)
    ap.add_argument('--H_out', type=int, default=1080)
    # zoom params
    ap.add_argument('--zoom_min', type=float, default=1.00)  # wide
    ap.add_argument('--zoom_max', type=float, default=1.85)  # tight
    ap.add_argument('--zoom_rate', type=float, default=0.45) # EMA rate (per second)
    ap.add_argument('--zoom_accel', type=float, default=1.2) # max zoom change per second
    ap.add_argument('--speed_tight', type=float, default=60) # px/s -> prefer tight at/below
    ap.add_argument('--speed_wide',  type=float, default=260) # px/s -> prefer wide at/above
    ap.add_argument('--hyst', type=float, default=35)        # px/s hysteresis
    args = ap.parse_args()

    # Probe input
    cap = cv2.VideoCapture(args.clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Base crop width to preserve 608x1080 aspect (even number)
    crop_w_base = int(np.floor(H * args.W_out / args.H_out / 2) * 2)
    crop_w_base = max(16, min(crop_w_base, W))

    # Load tracking
    df = pd.read_csv(args.track_csv)
    cx = pd.to_numeric(df['cx'], errors='coerce').to_numpy()
    # fraction of valid samples
    valid_mask = ~np.isnan(cx)
    valid_frac = float(valid_mask.mean())

    # If tracking is mostly missing, fall back to steady wide center render
    if valid_frac < 0.2:
        render_fallback(args.clip, args.out_mp4, fps, W, H, args.W_out, args.H_out, crop_w_base, args.left_frac)
        print('Wrote', args.out_mp4, '(fallback: low track coverage)')
        return

    # Interp short gaps only: up to 10 frames either side
    idx = np.where(valid_mask)[0]
    cx = np.interp(np.arange(len(cx)), idx, cx[idx])
    # Smooth and velocity
    cx_sm = smooth_series(cx, fps)
    vx = np.gradient(cx_sm) * fps

    # lead with velocity
    lead = cx_sm + args.tau * vx

    # camera pan (damped)
    target_left = np.clip(lead - args.left_frac * crop_w_base, 0, W - crop_w_base)
    x = np.zeros_like(target_left, dtype=float)
    v = 0.0
    x[0] = target_left[0]
    dt = 1.0 / fps
    slew = abs(args.slew)
    acc  = abs(args.accel)

    for i in range(1, len(target_left)):
        err   = target_left[i] - x[i-1]
        v_des = clamp(err / dt, -slew, slew)
        dv    = clamp(v_des - v, -acc * dt, acc * dt)
        v     = clamp(v + dv, -slew, slew)
        x[i]  = clamp(x[i-1] + v * dt, 0, W - crop_w_base)

    # speed-based zoom request: slow => tighter, fast => wider
    speed = np.abs(vx)  # px/s of ball
    tight_thr = max(1.0, args.speed_tight)
    wide_thr  = max(tight_thr + 1.0, args.speed_wide)

    # two thresholds with hysteresis band
    band_lo = max(1.0, tight_thr - args.hyst*0.5)
    band_hi = wide_thr + args.hyst*0.5

    # map speed -> [0..1] where 1 = tight, 0 = wide
    s_norm = np.clip((band_hi - speed) / max(1e-6, (band_hi - band_lo)), 0.0, 1.0)
    zoom_req = lerp(args.zoom_min, args.zoom_max, s_norm)  # slow->zoom_max, fast->zoom_min

    # smooth zoom with EMA and accel clamp
    alpha = 1.0 - math.exp(-abs(args.zoom_rate) * dt)   # stable [0..1)
    zoom = max(args.zoom_min, min(args.zoom_max, args.zoom_min))  # init safely at zoom_min

    zoom_series = np.zeros_like(zoom_req, dtype=float)
    for i in range(len(zoom_req)):
        z_des = clamp(float(zoom_req[i]), args.zoom_min, args.zoom_max)
        # accel limit in zoom-units/sec
        dz_max = abs(args.zoom_accel) * dt
        z_step = clamp(z_des - zoom, -dz_max, dz_max)
        zoom   = clamp(zoom + alpha * z_step, args.zoom_min, args.zoom_max)
        # belt-and-suspenders: never zero
        if zoom < 1e-6: zoom = args.zoom_min
        zoom_series[i] = zoom

    # Render frames
    tmp_dir = os.path.join(os.path.dirname(args.out_mp4), '_temp_frames')
    os.makedirs(tmp_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.clip)
    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok: break

        z = float(zoom_series[i] if i < len(zoom_series) else zoom_series[-1])
        if z < 1e-6: z = args.zoom_min
        eff_w = int(round(crop_w_base / z))
        eff_w = int(np.clip(eff_w, 16, W))        # bounds & even
        if eff_w % 2 == 1: eff_w -= 1
        # recalc left edge for current zoom target
        xi = int(round(cx_sm[i] - args.left_frac * eff_w))
        xi = int(np.clip(xi, 0, W - eff_w))

        crop = bgr[:, xi:xi+eff_w]
        if crop.shape[1] != eff_w:
            pad = eff_w - crop.shape[1]
            crop = cv2.copyMakeBorder(crop, 0, 0, 0, pad, cv2.BORDER_REPLICATE)

        crop = cv2.resize(crop, (args.W_out, args.H_out), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(os.path.join(tmp_dir, f'f_{i:06d}.jpg'), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
        i += 1
    cap.release()

    # Encode with source audio if present
    os.makedirs(os.path.dirname(args.out_mp4), exist_ok=True)
    subprocess.run([
        'ffmpeg','-y',
        '-framerate', str(int(round(fps))),
        '-i', os.path.join(tmp_dir, 'f_%06d.jpg'),
        '-i', args.clip,
        '-map','0:v','-map','1:a:0?',
        '-c:v','libx264','-preset','veryfast','-crf','19',
        '-x264-params','keyint=120:min-keyint=120:scenecut=0',
        '-pix_fmt','yuv420p','-profile:v','high','-level','4.0',
        '-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709',
        '-shortest','-movflags','+faststart',
        '-c:a','aac','-b:a','128k',
        args.out_mp4
    ], check=True)

    print('Wrote', args.out_mp4)

def render_fallback(clip, out_mp4, fps, W, H, W_out, H_out, crop_w_base, left_frac):
    tmp_dir = os.path.join(os.path.dirname(out_mp4), '_temp_frames')
    os.makedirs(tmp_dir, exist_ok=True)
    cap = cv2.VideoCapture(clip)
    xi = int(round((W - crop_w_base) * (left_frac)))  # center-ish with same left_frac
    xi = int(np.clip(xi, 0, W - crop_w_base))
    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok: break
        crop = bgr[:, xi:xi+crop_w_base]
        crop = cv2.resize(crop, (W_out, H_out), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(os.path.join(tmp_dir, f'f_{i:06d}.jpg'), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
        i += 1
    cap.release()
    subprocess.run([
        'ffmpeg','-y',
        '-framerate', str(int(round(fps))),
        '-i', os.path.join(tmp_dir, 'f_%06d.jpg'),
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

if __name__ == '__main__':
    main()

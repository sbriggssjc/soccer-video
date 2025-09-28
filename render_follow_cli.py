import argparse, os, subprocess
import cv2, numpy as np, pandas as pd
from scipy.signal import savgol_filter

def parse_args():
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
    # zoom
    ap.add_argument('--zoom_min', type=float, default=1.0)
    ap.add_argument('--zoom_max', type=float, default=1.85)
    ap.add_argument('--zoom_rate', type=float, default=0.35)
    ap.add_argument('--zoom_accel', type=float, default=0.90)
    ap.add_argument('--speed_tight', type=float, default=50.0)
    ap.add_argument('--speed_wide', type=float, default=280.0)
    ap.add_argument('--hyst', type=float, default=35.0)
    # new stability knobs
    ap.add_argument('--deadband', type=float, default=18.0, help='px error ignored for pan')
    ap.add_argument('--dir_hold', type=int, default=6, help='frames before accepting direction reversal')
    ap.add_argument('--jerk', type=float, default=2500.0, help='px/s^3 max change of acceleration')
    ap.add_argument('--zoom_dwell', type=float, default=0.7, help='seconds min dwell between zoom state changes')
    return ap.parse_args()

def smooth_series(arr, fps):
    n = len(arr)
    if n < 5: return arr
    win = max(5, int(round(fps*0.5)))
    if win % 2 == 0: win += 1
    win = min(win, n if n%2==1 else n-1)
    if win < 5: return arr
    return savgol_filter(arr, window_length=win, polyorder=2)

def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    crop_w_base = int(np.floor(H*args.W_out/args.H_out/2)*2)
    crop_w_base = min(crop_w_base, W)
    half = crop_w_base/2

    df = pd.read_csv(args.track_csv)
    cx = pd.to_numeric(df['cx'], errors='coerce').to_numpy()
    # fill short gaps (<=10)
    s = pd.Series(cx).ffill(limit=10).bfill(limit=10).to_numpy()
    # if any NaNs remain, interp globally
    if np.isnan(s).any():
        idx = np.where(~np.isnan(s))[0]
        s = np.interp(np.arange(len(s)), idx, s[idx])
    cx = smooth_series(s, fps)

    # lead with velocity
    vx = np.gradient(cx) * fps
    lead_cx = cx + args.tau * vx
    target = np.clip(lead_cx - args.left_frac*crop_w_base, 0, W - crop_w_base)

    # pan controller with deadband + direction hysteresis + jerk limit
    dt = 1.0/fps
    x   = np.zeros_like(target)
    v   = 0.0
    a   = 0.0
    x[0] = target[0]
    slew = args.slew
    accel = args.accel
    jerk = args.jerk
    dead = args.deadband
    dir_hold = args.dir_hold
    hold_cnt = 0
    last_sign = 0

    for i in range(1, len(target)):
        err = target[i] - x[i-1]
        # deadband
        if abs(err) < dead: err = 0.0
        sign = 0 if err==0 else (1 if err>0 else -1)
        # direction-change hysteresis
        if sign != 0 and sign != last_sign and last_sign != 0:
            if hold_cnt < dir_hold:
                # pretend error is zero until we believe the reversal
                err = 0.0
                sign = last_sign
                hold_cnt += 1
            else:
                last_sign = sign
                hold_cnt = 0
        else:
            if sign != 0: last_sign = sign
            hold_cnt = 0

        v_des = np.clip(err/dt, -slew, slew)
        a_des = np.clip((v_des - v)/dt, -accel, accel)
        # jerk clamp
        j = (a_des - a)/dt
        if j >  jerk: a_des = a + jerk*dt
        if j < -jerk: a_des = a - jerk*dt
        a = a_des
        v = np.clip(v + a*dt, -slew, slew)
        x[i] = float(np.clip(x[i-1] + v*dt, 0, W - crop_w_base))

    # zoom controller (calm)
    speed = np.abs(vx)
    desir_zoom = np.where(speed < args.speed_tight, args.zoom_max,
                    np.where(speed > args.speed_wide, args.zoom_min,
                             # linear map between thresholds
                             args.zoom_max - (speed-args.speed_tight)*
                             (args.zoom_max-args.zoom_min)/max(1.0, (args.speed_wide-args.speed_tight))))
    # hysteresis around thresholds
    desir_zoom = smooth_series(desir_zoom, fps)
    zoom = np.zeros_like(desir_zoom, dtype=float)
    zoom[0] = np.clip(desir_zoom[0], args.zoom_min, args.zoom_max)
    vz = 0.0
    az = 0.0
    z_last_switch = 0.0
    min_dwell_frames = int(round(args.zoom_dwell*fps))

    for i in range(1, len(desir_zoom)):
        # dwell guard
        if i - z_last_switch < min_dwell_frames:
            z_target = zoom[i-1]  # hold
        else:
            z_target = desir_zoom[i]
            if (z_target > zoom[i-1] and desir_zoom[i-1] <= zoom[i-1]) or \
               (z_target < zoom[i-1] and desir_zoom[i-1] >= zoom[i-1]):
                z_last_switch = i

        # smooth S-curve to z_target
        errz = z_target - zoom[i-1]
        vz_des = np.clip(errz/dt, -args.zoom_rate, args.zoom_rate)
        az_des = np.clip((vz_des - vz)/dt, -args.zoom_accel, args.zoom_accel)
        vz = np.clip(vz + az_des*dt, -args.zoom_rate, args.zoom_rate)
        zoom[i] = float(np.clip(zoom[i-1] + vz*dt, args.zoom_min, args.zoom_max))

    # render frames
    dbg_dir = os.path.join(os.path.dirname(args.out_mp4), '_temp_frames')
    os.makedirs(dbg_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.clip)
    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok: break
        z = max(zoom[i], 1e-6)
        eff_w = int(round(crop_w_base / z))
        eff_w = max(16, min(eff_w, crop_w_base))  # clamp

        # compute matched-height crop to preserve aspect before resizing
        xi_base = x[i] if i < len(x) else x[-1]
        xi_center = int(round(xi_base + half))
        eff_w_i = int(eff_w)
        eff_h_i = int(np.floor((eff_w_i * args.H_out / args.W_out) / 2) * 2)

        # clamp height to source
        eff_h_i = min(eff_h_i, H - (H % 2))  # keep even and ≤ H

        # vertical framing: center (or later, follow df['cy'] if you like)
        yi_center = H // 2
        yi = int(np.clip(yi_center - eff_h_i // 2, 0, H - eff_h_i))

        # clamp x as well
        xi = int(np.clip(xi_center - eff_w_i // 2, 0, W - eff_w_i))

        # final crop with correct aspect
        crop = bgr[yi:yi+eff_h_i, xi:xi+eff_w_i]

        # safety pad if edge rounding ever bites
        if crop.shape[0] != eff_h_i or crop.shape[1] != eff_w_i:
            pad_h = max(0, eff_h_i - crop.shape[0])
            pad_w = max(0, eff_w_i - crop.shape[1])
            crop = cv2.copyMakeBorder(crop, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)

        # now resize — no warp since crop already matches 608:1080
        crop = cv2.resize(crop, (args.W_out, args.H_out), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(os.path.join(dbg_dir, f'f_{i:06d}.jpg'), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
        i += 1
    cap.release()

    # encode
    subprocess.run([
        'ffmpeg','-y',
        '-framerate', str(int(round(fps))),
        '-i', os.path.join(dbg_dir, 'f_%06d.jpg'),
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

if __name__=='__main__':
    main()

# plan_render_cli.py
import argparse, os
import numpy as np
import pandas as pd
import cv2, subprocess
from scipy.signal import savgol_filter


def even(n):
    n = int(np.floor(n / 2) * 2)
    return n


def savgol(arr, fps, seconds=0.3):
    n = len(arr)
    if n < 5:
        return arr
    win = max(5, int(round(fps * seconds)))
    if win % 2 == 0:
        win += 1
    if win > n:
        win = n - (1 - n % 2)
    if win < 5:
        return arr
    return savgol_filter(arr, window_length=win, polyorder=2, mode='interp')


def slew_accel_limit(sig, fps, slew_per_s, accel_per_s2):
    if len(sig) == 0:
        return sig
    dt = 1.0 / fps
    y = np.zeros_like(sig, dtype=float)
    v = 0.0
    y[0] = float(sig[0])
    for i in range(1, len(sig)):
        err = sig[i] - y[i - 1]
        v_des = np.clip(err / dt, -slew_per_s, slew_per_s)
        dv = np.clip(v_des - v, -accel_per_s2 * dt, accel_per_s2 * dt)
        v += dv
        v = np.clip(v, -slew_per_s, slew_per_s)
        y[i] = y[i - 1] + v * dt
    return y


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--track_csv", required=True)
    ap.add_argument("--out_mp4", required=True)
    ap.add_argument("--W_out", type=int, default=608)
    ap.add_argument("--H_out", type=int, default=1080)

    # camera dynamics
    ap.add_argument("--slew", type=float, default=110.0)      # px/s
    ap.add_argument("--accel", type=float, default=320.0)     # px/s^2
    ap.add_argument("--zoom_rate", type=float, default=0.14)  # 1/s
    ap.add_argument("--zoom_accel", type=float, default=0.45)  # 1/s^2
    ap.add_argument("--left_frac", type=float, default=0.44)
    ap.add_argument("--ball_margin", type=float, default=0.25)
    ap.add_argument("--conf_min", type=float, default=0.35)
    ap.add_argument("--miss_jump", type=float, default=160.0)

    # composition / look-ahead
    ap.add_argument("--zoom_min", type=float, default=1.00)
    ap.add_argument("--zoom_max", type=float, default=1.60)
    ap.add_argument("--ctx_radius", type=int, default=420)
    ap.add_argument("--k_near", type=int, default=4)
    ap.add_argument("--ctx_pad", type=float, default=0.30)

    # “fit both ball + player” zoom policy
    ap.add_argument("--player_fit_margin", type=int, default=80)   # px margin around union
    ap.add_argument("--keep_margin", type=int, default=40)         # hard keep-in-frame margin

    # safety
    ap.add_argument("--speed_tight", type=float, default=60)
    ap.add_argument("--speed_wide", type=float, default=220)
    ap.add_argument("--hyst", type=float, default=35)

    args = ap.parse_args()

    # video props
    cap=cv2.VideoCapture(args.clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    crop_w_base = int(np.floor(H * args.W_out / args.H_out / 2) * 2)
    crop_w_base = min(crop_w_base, even(W))

    df = pd.read_csv(args.track_csv, engine='python', on_bad_lines='skip')

    cx_series = pd.to_numeric(df.get('cx', pd.Series(np.nan, index=df.index)), errors='coerce')
    cy_series = pd.to_numeric(df.get('cy', pd.Series(np.nan, index=df.index)), errors='coerce')
    conf_series = pd.to_numeric(df.get('conf', pd.Series(1.0, index=df.index)), errors='coerce')

    cx = cx_series.interpolate(limit=15).ffill().bfill().to_numpy(dtype=float)
    cy = cy_series.interpolate(limit=15).ffill().bfill().to_numpy(dtype=float)
    conf = conf_series.fillna(1.0).to_numpy(dtype=float)

    lead_frac = args.left_frac
    ball_margin = getattr(args, 'ball_margin', 0.25)
    conf_min = getattr(args, 'conf_min', 0.35)
    miss_jump = getattr(args, 'miss_jump', 160.0)

    crop_w_base = max(2, crop_w_base)
    min_w = max(2, even(crop_w_base / max(args.zoom_max, 1e-6)))

    if len(cx) == 0:
        left_path = np.array([0.0], dtype=float)
        zoom_path = np.array([args.zoom_min], dtype=float)
        cy = np.array([H / 2.0], dtype=float)
    else:
        cx = savgol(cx, fps, seconds=0.35)
        cy = savgol(cy, fps, seconds=0.35)

        spd = np.abs(np.gradient(cx)) * fps
        z_tgt = np.where(
            (conf >= conf_min) & (spd <= getattr(args, 'speed_wide', 220)),
            args.zoom_max,
            np.maximum(args.zoom_min, args.zoom_max - 0.50)
        )

        zoom = slew_accel_limit(z_tgt, fps, args.zoom_rate, args.zoom_accel)
        zoom = np.clip(zoom, args.zoom_min, args.zoom_max)

        eff_w = np.maximum(2, (crop_w_base / np.clip(zoom, 1e-6, None))).astype(int)
        eff_w -= eff_w % 2
        eff_w = np.clip(eff_w, min_w, crop_w_base).astype(int)

        Lmin = cx - (1.0 - ball_margin) * eff_w
        Lmax = cx - ball_margin * eff_w
        Lmin = np.clip(Lmin, 0, W - eff_w)
        Lmax = np.clip(Lmax, 0, W - eff_w)

        left = np.zeros_like(cx, dtype=float)
        v = 0.0
        dt = 1.0 / float(fps)
        desired0 = np.clip(cx[0] - lead_frac * eff_w[0], Lmin[0], Lmax[0])
        left[0] = desired0

        for i in range(1, len(cx)):
            good = (conf[i] >= conf_min) and (abs(cx[i] - cx[i - 1]) <= miss_jump)
            if not good:
                desired = left[i - 1]
            else:
                desired = np.clip(cx[i] - lead_frac * eff_w[i], Lmin[i], Lmax[i])

            err = desired - left[i - 1]
            v_des = np.clip(err / dt, -args.slew, args.slew)
            dv = np.clip(v_des - v, -args.accel * dt, args.accel * dt)
            v = np.clip(v + dv, -args.slew, args.slew)
            left[i] = left[i - 1] + v * dt

            if left[i] < Lmin[i]:
                left[i] = Lmin[i]
                v = (left[i] - left[i - 1]) / dt
            elif left[i] > Lmax[i]:
                left[i] = Lmax[i]
                v = (left[i] - left[i - 1]) / dt

        left = savgol(left, fps, seconds=0.28)
        zoom = savgol(zoom, fps, seconds=0.28)

        left = pd.Series(left).ffill().bfill().to_numpy(dtype=float)
        zoom = pd.Series(zoom).ffill().bfill().to_numpy(dtype=float)

        zoom = np.clip(zoom, args.zoom_min, args.zoom_max)
        eff_w = np.maximum(2, (crop_w_base / np.clip(zoom, 1e-6, None))).astype(int)
        eff_w -= eff_w % 2
        eff_w = np.clip(eff_w, min_w, crop_w_base).astype(int)
        left = np.clip(left, 0, W - eff_w)

        left_path = left
        zoom_path = zoom

    # Render frames (pure crop+scale, no aspect warp)
    out_dir = os.path.dirname(args.out_mp4)
    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, "_temp_frames")
    os.makedirs(tmp, exist_ok=True)

    cap = cv2.VideoCapture(args.clip)
    i=0
    while True:
        ok, bgr = cap.read()
        if not ok: break
        i_clamped = min(i, len(left_path)-1)
        left = int(round(left_path[i_clamped]))
        zoom = float(zoom_path[i_clamped])
        eff_w = int(round(crop_w_base / max(zoom,1e-6)))
        eff_w -= eff_w % 2
        eff_h = int(round(eff_w * args.H_out / args.W_out))
        eff_h -= eff_h % 2
        # center vertical on ball Y with safety (never warp)
        by = int(round(cy[min(i,len(cy)-1)]))
        top = clamp(by - eff_h//2, 0, H - eff_h)
        left = clamp(left, 0, W - eff_w)

        crop = bgr[top:top+eff_h, left:left+eff_w]
        frame = cv2.resize(crop, (args.W_out, args.H_out), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(os.path.join(tmp, f'f_{i:06d}.jpg'), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
        i+=1
    cap.release()

    # Encode with source audio
    fr = int(round(fps))
    subprocess.run([
        'ffmpeg','-y',
        '-framerate', str(fr),
        '-i', os.path.join(tmp, 'f_%06d.jpg'),
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

    print("Wrote", args.out_mp4)

if __name__ == "__main__":
    main()

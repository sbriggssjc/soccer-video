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


def initial_zoom_from_speed(cx, fps, args):
    if len(cx) == 0:
        return np.array([], dtype=float)
    vx = np.gradient(cx) * fps
    speed = np.abs(vx)
    tight = float(getattr(args, 'speed_tight', 0.0))
    wide = float(getattr(args, 'speed_wide', tight + 1.0))
    zoom = np.empty_like(speed, dtype=float)
    if wide <= tight:
        zoom.fill(args.zoom_min)
        return zoom
    for i, s in enumerate(speed):
        if s <= tight:
            zoom[i] = args.zoom_min
        elif s >= wide:
            zoom[i] = args.zoom_max
        else:
            t = (s - tight) / (wide - tight)
            zoom[i] = args.zoom_min * (1.0 - t) + args.zoom_max * t
    return zoom


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
    ap.add_argument("--ball_margin", type=float, default=0.18)
    ap.add_argument("--conf_min", type=float, default=0.25)
    ap.add_argument("--miss_jump", type=float, default=220.0)

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
    ap.add_argument("--speed_wide", type=float, default=240)
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
    for col in ['cx','cy']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df[['cx','cy']] = df[['cx','cy']].interpolate(limit=15).ffill().bfill()

    cx = df['cx'].to_numpy(dtype=float)
    cy = df['cy'].to_numpy(dtype=float)
    conf = pd.to_numeric(df.get('conf', pd.Series(1.0, index=df.index)), errors='coerce').fillna(1.0).to_numpy()

    cx = savgol(cx, fps, seconds=0.35)
    cy = savgol(cy, fps, seconds=0.35)

    lead_frac = args.left_frac
    ball_margin = getattr(args, 'ball_margin', 0.18)
    conf_min = getattr(args, 'conf_min', 0.25)
    miss_jump = getattr(args, 'miss_jump', 220.0)

    crop_w_base = max(2, crop_w_base)
    min_w = max(2, even(crop_w_base / max(args.zoom_max, 1e-6)))

    zoom = np.clip(initial_zoom_from_speed(cx, fps, args), args.zoom_min, args.zoom_max)
    eff_w = crop_w_base / np.clip(zoom, 1e-6, None)
    eff_w = np.clip(eff_w, min_w, crop_w_base)
    eff_w = np.floor(eff_w / 2.0) * 2.0
    eff_w = np.clip(eff_w, min_w, crop_w_base)
    eff_w = eff_w.astype(int)

    target_left = cx - lead_frac * eff_w

    min_left = cx - (1.0 - ball_margin) * eff_w
    max_left = cx - ball_margin * eff_w
    target_left = np.clip(target_left, min_left, max_left)
    target_left = np.clip(target_left, 0, W - eff_w)

    frozen_left = target_left.copy()
    last_good = None
    miss_active = False
    blend_frames = max(1, int(round(fps * 0.4)))
    blend_step = 0
    recover_start = target_left[0] if len(target_left) else 0.0
    zoom_hold = zoom[0] if len(zoom) else args.zoom_min
    dt = 1.0 / float(fps)

    for i in range(len(target_left)):
        if i > 0:
            zoom_hold = zoom[i - 1]

        jump_ok = True
        if last_good is not None and abs(cx[i] - cx[last_good]) > miss_jump:
            jump_ok = False
        good = (conf[i] >= conf_min) and jump_ok

        if good:
            zoom_hold = zoom[i]
            zoom[i] = zoom_hold
            if miss_active:
                if blend_step == 0:
                    if i > 0:
                        recover_start = frozen_left[i - 1]
                    elif last_good is not None:
                        recover_start = frozen_left[last_good]
                    else:
                        recover_start = target_left[i]
                blend_step += 1
                t = min(1.0, blend_step / float(blend_frames))
                frozen_left[i] = recover_start + (target_left[i] - recover_start) * t
                if t >= 1.0:
                    miss_active = False
                    blend_step = 0
            else:
                frozen_left[i] = target_left[i]
            last_good = i
        else:
            if last_good is not None:
                frozen_left[i] = frozen_left[last_good]
            elif i > 0:
                frozen_left[i] = frozen_left[i - 1]
            else:
                frozen_left[i] = target_left[i]
            zoom_hold = min(args.zoom_max, zoom_hold + args.zoom_rate * dt)
            zoom[i] = zoom_hold
            miss_active = True
            blend_step = 0

    left = slew_accel_limit(frozen_left, fps, args.slew, args.accel)
    zoom = slew_accel_limit(zoom, fps, args.zoom_rate, args.zoom_accel)

    left = savgol(left, fps, seconds=0.30)
    zoom = savgol(zoom, fps, seconds=0.30)

    left = pd.Series(left).ffill().bfill().to_numpy()
    zoom = pd.Series(zoom).ffill().bfill().to_numpy()
    zoom = np.clip(zoom, args.zoom_min, args.zoom_max)

    eff_w = crop_w_base / np.clip(zoom, 1e-6, None)
    eff_w = np.clip(eff_w, min_w, crop_w_base)
    eff_w = np.floor(eff_w / 2.0) * 2.0
    eff_w = np.clip(eff_w, min_w, crop_w_base)
    eff_w = eff_w.astype(int)
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

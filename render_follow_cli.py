#!/usr/bin/env python3
import argparse, os, subprocess
import numpy as np
import pandas as pd
import cv2
from scipy.signal import savgol_filter

def smooth_series(arr, fps):
    n = len(arr)
    if n < 5: return arr
    win = max(5, int(round(fps*0.5)))  # ~0.5 s
    if win % 2 == 0: win += 1
    if win > n: win = (n if n % 2 == 1 else n-1)
    if win < 5: return arr
    return savgol_filter(arr, window_length=win, polyorder=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--track_csv", required=True)
    ap.add_argument("--out_mp4", required=True)

    ap.add_argument("--tau", type=float, default=0.26)
    ap.add_argument("--slew", type=float, default=230.0)
    ap.add_argument("--accel", type=float, default=850.0)
    ap.add_argument("--left_frac", type=float, default=0.44)

    ap.add_argument("--W_out", type=int, default=608)
    ap.add_argument("--H_out", type=int, default=1080)

    ap.add_argument("--zoom_min", type=float, default=1.00)
    ap.add_argument("--zoom_max", type=float, default=1.85)
    ap.add_argument("--zoom_rate", type=float, default=0.65)   # EMA factor (0–1), lower = smoother
    ap.add_argument("--zoom_accel", type=float, default=1.6)   # max zoom change per second
    ap.add_argument("--speed_tight", type=float, default=60.0) # px/s
    ap.add_argument("--speed_wide",  type=float, default=250.0)# px/s
    ap.add_argument("--hyst", type=float, default=25.0)        # hysteresis width (px/s)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.clip)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.clip}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # base crop width for 608x1080 aspect (even)
    crop_w_base = int(np.floor(H * args.W_out / args.H_out / 2) * 2)
    crop_w_base = min(crop_w_base, W)
    dt = 1.0 / float(fps)

    # load track
    df = pd.read_csv(args.track_csv)
    for col in ["cx","cy"]:
        s = pd.to_numeric(df[col], errors="coerce")
        s = s.ffill(limit=10).bfill(limit=10)
        df[col] = s

    for col in ["cx","cy"]:
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy()
        if np.isnan(vals).any():
            idx = np.where(~np.isnan(vals))[0]
            vals = np.interp(np.arange(len(vals)), idx, vals[idx])
        df[col] = smooth_series(vals, fps)

    cx = df["cx"].to_numpy()
    vx = np.gradient(cx) * fps
    lead_cx = cx + args.tau * vx

    # pan target (left edge) with look-ahead
    target = np.clip(lead_cx - args.left_frac * crop_w_base, 0, W - crop_w_base)

    # pan dynamics
    x = np.zeros_like(target, dtype=np.float32)
    x[0] = target[0]
    v = 0.0
    slew_max = args.slew
    accel_max = args.accel

    # zoom dynamics with hysteresis
    # three bands: WIDE (fast play), MID, TIGHT (slow/settled/dribble/shot)
    # thresholds with hysteresis
    wide_on  = args.speed_wide + args.hyst
    wide_off = args.speed_wide - args.hyst
    tight_on = max(0.0, args.speed_tight - args.hyst)
    tight_off= args.speed_tight + args.hyst

    state = "MID"
    z = np.zeros_like(target, dtype=np.float32)
    z_cur = 1.0  # 1.0 => crop_w = crop_w_base;  >1 => tighter (smaller crop)

    def target_zoom(speed):
        nonlocal state
        # hysteretic state machine
        if state == "MID":
            if speed <= tight_on: state = "TIGHT"
            elif speed >= wide_on: state = "WIDE"
        elif state == "TIGHT":
            if speed >= tight_off: state = "MID"
        elif state == "WIDE":
            if speed <= wide_off: state = "MID"

        if state == "TIGHT":
            return args.zoom_max
        elif state == "WIDE":
            return args.zoom_min
        else:
            # map speed MID-> zoom between min/max (gentle)
            s0, s1 = tight_off, wide_off
            if s1 <= s0: 
                alpha = 0.0
            else:
                alpha = np.clip((speed - s0)/(s1 - s0), 0.0, 1.0)
            return args.zoom_max*(1.0 - alpha) + args.zoom_min*alpha

    for i in range(1, len(target)):
        # pan step
        err   = float(target[i] - x[i-1])
        v_des = np.clip(err/dt, -slew_max, slew_max)
        dv    = np.clip(v_des - v, -accel_max*dt, accel_max*dt)
        v    += dv
        v     = np.clip(v, -slew_max, slew_max)
        x[i]  = float(np.clip(x[i-1] + v*dt, 0, W - crop_w_base))

        # zoom step
        spd = abs(v)  # px/s
        z_tgt = float(target_zoom(spd))
        # EMA smoothing
        z_cur = (1.0 - args.zoom_rate)*z_cur + args.zoom_rate*z_tgt
        # acceleration clamp on zoom
        z_prev = z[i-1] if i>0 else z_cur
        z_step_max = args.zoom_accel * dt
        z_cur = float(np.clip(z_cur, z_prev - z_step_max, z_prev + z_step_max))
        z[i] = float(np.clip(z_cur, args.zoom_min, args.zoom_max))

    # render
    os.makedirs(os.path.dirname(args.out_mp4), exist_ok=True)
    tmp_dir = os.path.join(os.path.dirname(args.track_csv), "_temp_frames")
    os.makedirs(tmp_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.clip)
    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok: break

        zoom = float(z[i]) if i < len(z) else float(z[-1])
        # effective crop width shrinks as zoom grows
        eff_w = int(round(crop_w_base / zoom))
        if eff_w % 2 == 1: eff_w += 1
        eff_w = max(16, min(eff_w, W))

        # center the target window around the desired left edge + look-ahead fraction of eff_w
        # keep the ball near args.left_frac position in the visible window at current zoom
        left = float(x[i])  # previously solved on base width
        # adjust to keep consistent positioning under zoom
        left_adj = np.clip(lead_cx[i] - args.left_frac * eff_w, 0, W - eff_w)
        xi = int(round(left_adj))

        crop = bgr[:, xi:xi+eff_w]
        if crop.shape[1] != eff_w:
            pad = eff_w - crop.shape[1]
            crop = cv2.copyMakeBorder(crop, 0, 0, 0, pad, cv2.BORDER_REPLICATE)

        # resize to output
        crop = cv2.resize(crop, (args.W_out, args.H_out), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(os.path.join(tmp_dir, f"f_{i:06d}.jpg"), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
        i += 1

    cap.release()

    # encode with source audio if present
    fr = int(round(fps if fps and fps>0 else 24))
    cmd = [
        "ffmpeg","-y",
        "-framerate", str(fr),
        "-i", os.path.join(tmp_dir, "f_%06d.jpg"),
        "-i", args.clip,
        "-map","0:v","-map","1:a:0?",
        "-c:v","libx264","-preset","veryfast","-crf","19",
        "-x264-params","keyint=120:min-keyint=120:scenecut=0",
        "-pix_fmt","yuv420p","-profile:v","high","-level","4.0",
        "-colorspace","bt709","-color_primaries","bt709","-color_trc","bt709",
        "-shortest","-movflags","+faststart",
        "-c:a","aac","-b:a","128k",
        args.out_mp4
    ]
    subprocess.run(cmd, check=True)
    print("Wrote", args.out_mp4)

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
import argparse, os, math, subprocess
import numpy as np
import pandas as pd
import cv2


def parse_ppl(s):
    if not isinstance(s, str) or not s:
        return []
    out = []
    for tok in s.split("|"):
        try:
            x,y = tok.split(":")
            out.append((float(x), float(y)))
        except Exception:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--track_csv", required=True)
    ap.add_argument("--out_mp4", required=True)

    # motion parameters (same spirit as before)
    ap.add_argument("--tau", type=float, default=0.26)      # lookahead (s) for ball
    ap.add_argument("--slew", type=float, default=210.0)    # px/s for x,y
    ap.add_argument("--accel", type=float, default=750.0)   # px/s^2 for x,y
    ap.add_argument("--left_frac", type=float, default=0.44)

    # output size
    ap.add_argument("--W_out", type=int, default=608)
    ap.add_argument("--H_out", type=int, default=1080)

    # zoom parameters
    ap.add_argument("--zoom_min", type=float, default=1.00)
    ap.add_argument("--zoom_max", type=float, default=1.85)
    ap.add_argument("--zoom_rate", type=float, default=0.45)   # zoom speed (units/s)
    ap.add_argument("--zoom_accel", type=float, default=1.2)   # zoom accel (units/s^2)

    # context box settings
    ap.add_argument("--k_near", type=int, default=4)       # how many nearest players to include
    ap.add_argument("--ctx_pad", type=float, default=0.30) # extra padding on bbox (ratio of width)
    ap.add_argument("--ctx_radius", type=float, default=420.0) # max distance (px) to consider "near"

    # behavior based on motion
    ap.add_argument("--speed_tight", type=float, default=60)   # px/s ball => tighter zoom
    ap.add_argument("--speed_wide",  type=float, default=260)  # px/s ball => wider zoom
    ap.add_argument("--hyst", type=float, default=35.0)        # deadband on speed switch

    args = ap.parse_args()

    cap = cv2.VideoCapture(args.clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # base crop width that preserves aspect before resize
    crop_w_base = int(np.floor(H * (args.W_out/args.H_out) / 2) * 2)
    crop_w_base = min(crop_w_base, W - (W % 2))
    aspect = args.W_out / args.H_out

    df = pd.read_csv(args.track_csv)
    cx = pd.to_numeric(df["cx"], errors="coerce").to_numpy()
    cy = pd.to_numeric(df["cy"], errors="coerce").to_numpy()
    # fill remaining gaps safely
    if np.isnan(cx).any():
        idx = np.where(~np.isnan(cx))[0]
        if len(idx) >= 2:
            cx = np.interp(np.arange(len(cx)), idx, cx[idx])
        else:
            cx = np.nan_to_num(cx, nan=W/2)
    if np.isnan(cy).any():
        idx = np.where(~np.isnan(cy))[0]
        if len(idx) >= 2:
            cy = np.interp(np.arange(len(cy)), idx, cy[idx])
        else:
            cy = np.nan_to_num(cy, nan=H/2)

    ppl = [parse_ppl(s) for s in df.get("ppl", [])]

    # lookahead on ball x
    vx = np.gradient(cx) * fps
    cx_lead = cx + args.tau * vx

    # state
    dt = 1.0 / fps
    x, y = 0.0, H/2.0
    vx_cam = vy_cam = 0.0
    zoom = 1.0
    vzoom = 0.0

    # helpers
    def clamp(val, lo, hi): return lo if val < lo else hi if val > hi else val

    # temp frames folder
    tmp_dir = os.path.join(os.path.dirname(args.out_mp4) or ".", "_temp_frames")
    os.makedirs(tmp_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.clip)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok: break

        # --- build a context box around the ball + nearest players ---
        cx_i = float(cx_lead[i] if i < len(cx_lead) else cx[-1])
        cy_i = float(cy[i]       if i < len(cy)       else H/2)

        # gather nearby players
        near = []
        if i < len(ppl) and ppl[i]:
            for (px,py) in ppl[i]:
                d = math.hypot(px - cx_i, py - cy_i)
                if d <= args.ctx_radius:
                    near.append((d, px, py))
            near.sort(key=lambda t: t[0])
            near = near[:args.k_near]

        # initial bbox: at least the ball point
        xs = [cx_i]; ys = [cy_i]
        for _,px,py in near:
            xs.append(px); ys.append(py)

        # pad bbox horizontally (context), then fit to output aspect
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        bw = max(40.0, maxx - minx)
        bh = max(40.0, maxy - miny)

        # expand horizontally by ctx_pad
        padw = bw * args.ctx_pad
        box_w = bw + 2*padw
        box_h = bh + 2*padw/ aspect  # expand vertically in proportion to keep context

        # center of box (weighted a bit toward the ball)
        cx_box = 0.7*cx_i + 0.3*((minx+maxx)/2.0)
        cy_box = 0.7*cy_i + 0.3*((miny+maxy)/2.0)

        # make box match output aspect
        if box_w / box_h > aspect:
            # too wide → grow height
            box_h = box_w / aspect
        else:
            # too tall → grow width
            box_w = box_h * aspect

        # translate to left edge (x target) so ball sits slightly forward (left_frac)
        left_target = cx_box - args.left_frac * box_w
        top_target  = cy_box - 0.5 * box_h

        # --- turn desired box into pan/zoom targets ---
        # zoom target from desired width
        target_zoom = clamp(crop_w_base / max(1.0, box_w),
                            args.zoom_min, args.zoom_max)

        # pan targets (x,y for box left/top), clamped to frame
        x_target = clamp(left_target, 0, W - box_w)
        y_target = clamp(top_target,  0, H - box_h)

        # slew/accel limit x
        ex = x_target - x
        v_des_x = clamp(ex/dt, -args.slew, args.slew)
        dvx = clamp(v_des_x - vx_cam, -args.accel*dt, args.accel*dt)
        vx_cam = clamp(vx_cam + dvx, -args.slew, args.slew)
        x = clamp(x + vx_cam*dt, 0, W - box_w)

        # slew/accel limit y
        ey = y_target - y
        v_des_y = clamp(ey/dt, -args.slew, args.slew)
        dvy = clamp(v_des_y - vy_cam, -args.accel*dt, args.accel*dt)
        vy_cam = clamp(vy_cam + dvy, -args.slew, args.slew)
        y = clamp(y + vy_cam*dt, 0, H - box_h)

        # zoom rate/accel limit
        ez = target_zoom - zoom
        v_des_z = clamp(ez/dt, -args.zoom_rate, args.zoom_rate)
        dvz = clamp(v_des_z - vzoom, -args.zoom_accel*dt, args.zoom_accel*dt)
        vzoom = clamp(vzoom + dvz, -args.zoom_rate, args.zoom_rate)
        zoom = clamp(zoom + vzoom*dt, args.zoom_min, args.zoom_max)

        # derive crop width/height from zoom, keep aspect
        eff_w = max(2, int(round(crop_w_base / max(1e-6, zoom))))
        eff_h = int(np.floor((eff_w / aspect) / 2) * 2)
        eff_w = int(np.floor(eff_w / 2) * 2)

        # fix if height spills
        if eff_h > H: 
            eff_h = H - (H % 2)
            eff_w = int(np.floor(eff_h * aspect / 2) * 2)

        # convert (x,y) (left/top) to ints and clamp
        xi = int(clamp(round(x), 0, max(0, W - eff_w)))
        yi = int(clamp(round(y), 0, max(0, H - eff_h)))

        crop = frame[yi:yi+eff_h, xi:xi+eff_w]
        if crop.shape[0] != eff_h or crop.shape[1] != eff_w:
            # rare safety pad
            pad_h = max(0, eff_h - crop.shape[0])
            pad_w = max(0, eff_w - crop.shape[1])
            crop = cv2.copyMakeBorder(crop, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)

        crop = cv2.resize(crop, (args.W_out, args.H_out), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(os.path.join(tmp_dir, f"f_{i:06d}.jpg"), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
        i += 1

    cap.release()

    os.makedirs(os.path.dirname(args.out_mp4) or ".", exist_ok=True)
    subprocess.run([
        "ffmpeg","-y",
        "-framerate", str(int(round(fps))),
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
    ], check=True)

    print("Wrote", args.out_mp4)


if __name__ == "__main__":
    main()

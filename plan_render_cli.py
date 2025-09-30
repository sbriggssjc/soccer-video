# plan_render_cli.py
import argparse
import os
import subprocess

import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def letterbox_to_size(img, out_w, out_h):
    """Resize to fit inside (out_w, out_h) without changing aspect."""

    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)

    scale = min(out_w / max(1, w), out_h / max(1, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    pad_left = (out_w - new_w) // 2
    pad_right = out_w - new_w - pad_left
    pad_top = (out_h - new_h) // 2
    pad_bottom = out_h - new_h - pad_top

    return cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_REPLICATE,
    )


def ema(series, alpha):
    out = np.empty_like(series, dtype=np.float32)
    m = np.isnan(series)
    idx = np.where(~m)[0]
    if len(idx) == 0:
        return np.zeros_like(series, dtype=np.float32)
    first = idx[0]
    out[: first + 1] = series[first]
    prev = series[first]
    for i in range(first + 1, len(series)):
        x = series[i]
        if np.isnan(x):
            out[i] = prev
        else:
            prev = alpha * x + (1 - alpha) * prev
            out[i] = prev
    return out


def kalman_1d(z, q=0.08, r=4.0):
    """
    Minimal constant-velocity Kalman for 1D with NaN handling.
    q: process noise, r: measurement noise (tune per dataset)
    Returns (x, v) filtered states.
    """

    n = len(z)
    x = np.zeros(n, dtype=np.float32)
    v = np.zeros(n, dtype=np.float32)
    P = np.eye(2, dtype=np.float32) * 1e3
    F = np.array([[1, 1], [0, 1]], dtype=np.float32)
    Q = np.array([[q, 0], [0, q]], dtype=np.float32)
    H = np.array([[1, 0]], dtype=np.float32)
    R = np.array([[r]], dtype=np.float32)

    idx = np.where(~np.isnan(z))[0]
    if len(idx) == 0:
        return x, v
    k0 = idx[0]
    x[k0] = z[k0]
    v[k0] = 0.0
    for t in range(k0 + 1, n):
        xv = np.array([x[t - 1], v[t - 1]], dtype=np.float32)
        xv = F @ xv
        P[:] = F @ P @ F.T + Q

        if np.isnan(z[t]):
            x[t], v[t] = xv
            continue

        y = z[t] - (H @ xv)[0]
        S = H @ P @ H.T + R
        K = (P @ H.T) / S
        xv = xv + (K.flatten() * y)
        P[:] = (np.eye(2, dtype=np.float32) - K @ H) @ P
        x[t], v[t] = xv
    for t in range(k0 - 1, -1, -1):
        x[t] = x[t + 1] - v[t + 1]
        v[t] = v[t + 1]
    return x, v


def reject_outliers(x, thresh_px=160.0):
    x = x.copy()
    m = np.isnan(x)
    idx = np.where(~m)[0]
    if len(idx) < 3:
        return x
    prev = x[idx[0]]
    for i in idx[1:]:
        if abs(x[i] - prev) > thresh_px:
            x[i] = np.nan
        else:
            prev = x[i]
    return x


def smooth_xy(cx, cy, ema_alpha=0.25, k_q=0.08, k_r=4.0):
    kx, vx = kalman_1d(cx, q=k_q, r=k_r)
    ky, vy = kalman_1d(cy, q=k_q, r=k_r)
    sx = ema(kx, ema_alpha)
    sy = ema(ky, ema_alpha)
    return sx, sy, vx, vy


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def clamp_vec(center, half_w, half_h, src_w, src_h):
    cx, cy = center
    cx = clamp(cx, half_w, src_w - half_w)
    cy = clamp(cy, half_h, src_h - half_h)
    return cx, cy


def limit_step(prev, target, max_step):
    delta = target - prev
    if np.abs(delta) <= max_step:
        return target
    return prev + np.sign(delta) * max_step


def odd_window(fps, seconds, minimum=5):
    n = max(minimum, int(round(fps * seconds)))
    if n % 2 == 0:
        n += 1
    return n


def smooth_series(arr, fps, sec=0.7, poly=2):
    arr = np.asarray(arr, dtype=np.float32)
    if len(arr) < 7:
        return arr
    win = odd_window(fps, sec, 7)
    win = min(win, len(arr) - (1 - len(arr) % 2))
    if win < 3:
        return arr
    return savgol_filter(arr, win, poly)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--track_csv", required=True)
    ap.add_argument("--out_mp4", required=True)
    ap.add_argument("--W_out", type=int, default=608)
    ap.add_argument("--H_out", type=int, default=1080)

    # Motion/zoom controls
    ap.add_argument("--slew", type=float, default=80.0)
    ap.add_argument("--accel", type=float, default=260.0)
    ap.add_argument("--max_jerk", type=float, default=500.0)
    ap.add_argument("--zoom_rate", type=float, default=0.10)
    ap.add_argument("--zoom_accel", type=float, default=0.30)
    ap.add_argument("--zoom_jerk", type=float, default=0.60)

    # Composition controls
    ap.add_argument("--left_frac", type=float, default=0.48)
    ap.add_argument("--keep_margin", type=float, default=220.0)
    ap.add_argument("--zoom_min", type=float, default=1.00)
    ap.add_argument("--zoom_max", type=float, default=1.45)
    ap.add_argument("--start_wide_s", type=float, default=1.6)
    ap.add_argument("--min_streak", type=int, default=16)
    ap.add_argument("--loss_streak", type=int, default=4)
    ap.add_argument("--prewiden_factor", type=float, default=1.30)
    ap.add_argument("--hyst", type=int, default=90)

    # Predictive tracking controls
    ap.add_argument("--lookahead_s", type=float, default=1.0)
    ap.add_argument("--pass_speed", type=float, default=360.0)
    ap.add_argument("--pass_lookahead_s", type=float, default=0.7)

    args = ap.parse_args()

    cap = cv2.VideoCapture(args.clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    W_src = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_src = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Robust CSV loading
    df = pd.read_csv(args.track_csv, engine="python", on_bad_lines="skip")
    for col in ["frame", "time", "cx", "cy", "visible", "conf", "miss_streak"]:
        if col not in df.columns:
            df[col] = np.nan

    df["frame"] = pd.to_numeric(df["frame"], errors="coerce")
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["cx"] = pd.to_numeric(df["cx"], errors="coerce")
    df["cy"] = pd.to_numeric(df["cy"], errors="coerce")
    df["visible"] = df["visible"].fillna(0).astype(int)
    df["conf"] = pd.to_numeric(df["conf"], errors="coerce").fillna(0.0)
    df["miss_streak"] = pd.to_numeric(df["miss_streak"], errors="coerce").fillna(0).astype(int)

    df = df.sort_values("frame").reset_index(drop=True)

    if df.empty:
        raise RuntimeError("Track CSV is empty")

    fps = max(1.0, float(fps))
    dt = 1.0 / fps
    N = len(df)

    cx = df["cx"].to_numpy(np.float32)
    cy = df["cy"].to_numpy(np.float32)
    cx_series = pd.Series(cx)
    cy_series = pd.Series(cy)
    cx_interp = (
        cx_series.interpolate(limit=12, limit_direction="both")
        .ffill()
        .bfill()
        .to_numpy(np.float32)
    )
    cy_interp = (
        cy_series.interpolate(limit=12, limit_direction="both")
        .ffill()
        .bfill()
        .to_numpy(np.float32)
    )

    cx_np = cx_series.to_numpy(dtype=float)
    cy_np = cy_series.to_numpy(dtype=float)

    cx_np = reject_outliers(cx_np, 160.0)
    cy_np = reject_outliers(cy_np, 160.0)

    # Updated pan/zoom planner
    SRC_W, SRC_H = W_src, H_src
    W_out = int(args.W_out)
    H_out = int(args.H_out)
    OUT_W, OUT_H = W_out, H_out

    sx_raw, sy_raw, vx, vy = smooth_xy(cx_np, cy_np, ema_alpha=0.25, k_q=0.06, k_r=6.0)

    # Preserve NaNs where detections are missing so the planner can react accordingly.
    valid_mask = ~np.isnan(cx_np) & ~np.isnan(cy_np)
    sx = sx_raw.copy()
    sy = sy_raw.copy()
    sx[~valid_mask] = np.nan
    sy[~valid_mask] = np.nan

    speed_arr = np.hypot(vx, vy)
    lead_scale = np.clip(speed_arr / 18.0, 0.0, 1.0)
    lead_px_arr = 24.0 * lead_scale
    tx = sx + vx * 0.25 * lead_scale
    ty = sy + vy * 0.25 * lead_scale

    AR = OUT_W / max(OUT_H, 1)
    vw_min = 820.0
    vw_max = 1400.0
    vw = vw_max - (vw_max - vw_min) * np.clip(speed_arr / 30.0, 0.0, 1.0)
    vh = vw / AR

    safe_margin = 0.12

    centers_x = np.zeros_like(sx, dtype=np.float32)
    centers_y = np.zeros_like(sy, dtype=np.float32)
    view_w = np.zeros_like(sx, dtype=np.float32)
    view_h = np.zeros_like(sy, dtype=np.float32)

    first = np.where(~np.isnan(sx) & ~np.isnan(sy))[0]
    if len(first) == 0:
        centers_x[:] = SRC_W * 0.5
        centers_y[:] = SRC_H * 0.5
        view_w[:] = vw_max
        view_h[:] = vw_max / AR
        k0 = 0
    else:
        k0 = first[0]
        centers_x[k0] = clamp(tx[k0], 0, SRC_W)
        centers_y[k0] = clamp(ty[k0], 0, SRC_H)
        view_w[k0] = vw[k0]
        view_h[k0] = vh[k0]

    for t in range(max(1, k0 + 1), len(sx)):
        cx_tgt = tx[t] if not np.isnan(tx[t]) else centers_x[t - 1]
        cy_tgt = ty[t] if not np.isnan(ty[t]) else centers_y[t - 1]
        vw_tgt = vw[t] if vw[t] > 0 else view_w[t - 1]
        vh_tgt = vw_tgt / AR

        spd = float(speed_arr[t - 1] if t > 0 else 0.0)
        max_pan_px = float(np.clip(6.0 + 1.15 * spd, 12.0, 85.0))
        max_zoom_px = float(np.clip(5.0 + 0.70 * spd, 8.0, 36.0))

        cx_soft = limit_step(centers_x[t - 1], cx_tgt, max_pan_px)
        cy_soft = limit_step(centers_y[t - 1], cy_tgt, max_pan_px)

        vw_soft = limit_step(view_w[t - 1], vw_tgt, max_zoom_px)
        vh_soft = vw_soft / AR

        if not np.isnan(sx[t]):
            bx = sx[t]
        elif t > 0:
            bx = sx[t - 1]
        else:
            bx = SRC_W / 2

        if not np.isnan(sy[t]):
            by = sy[t]
        elif t > 0:
            by = sy[t - 1]
        else:
            by = SRC_H / 2

        half_w = vw_soft * 0.5
        half_h = vh_soft * 0.5
        left = cx_soft - half_w
        right = cx_soft + half_w
        top = cy_soft - half_h
        bottom = cy_soft + half_h

        safe_l = left + vw_soft * safe_margin
        safe_r = right - vw_soft * safe_margin
        safe_t = top + vh_soft * safe_margin
        safe_b = bottom - vh_soft * safe_margin

        shift_x = 0.0
        shift_y = 0.0
        if bx < safe_l:
            shift_x = bx - safe_l
        elif bx > safe_r:
            shift_x = bx - safe_r
        if by < safe_t:
            shift_y = by - safe_t
        elif by > safe_b:
            shift_y = by - safe_b

        cx_soft += shift_x
        cy_soft += shift_y

        left = cx_soft - half_w
        right = cx_soft + half_w
        top = cy_soft - half_h
        bottom = cy_soft + half_h

        edge = max(18.0, min(36.0, 0.06 * vw_soft))
        if bx < left + edge:
            cx_soft = bx + half_w - edge
        elif bx > right - edge:
            cx_soft = bx - half_w + edge

        if by < top + edge:
            cy_soft = by + half_h - edge
        elif by > bottom - edge:
            cy_soft = by - half_h + edge

        cx_soft, cy_soft = clamp_vec((cx_soft, cy_soft), half_w, half_h, SRC_W, SRC_H)

        centers_x[t] = cx_soft
        centers_y[t] = cy_soft
        view_w[t] = vw_soft
        view_h[t] = vh_soft

    if len(first) > 0:
        for t in range(k0 - 1, -1, -1):
            centers_x[t] = centers_x[t + 1]
            centers_y[t] = centers_y[t + 1]
            view_w[t] = view_w[t + 1]
            view_h[t] = view_h[t + 1]

    centers_x = ema(centers_x, 0.08)
    centers_y = ema(centers_y, 0.08)

    df["center_x"] = centers_x
    df["center_y"] = centers_y
    df["view_w"] = view_w
    df["view_h"] = view_h

    cx_filled = np.where(np.isfinite(sx_raw), sx_raw, cx_interp)
    cy_filled = np.where(np.isfinite(sy_raw), sy_raw, cy_interp)

    out_dir = os.path.dirname(args.out_mp4)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir or ".", "_temp_frames")
    os.makedirs(tmp, exist_ok=True)

    cap = cv2.VideoCapture(args.clip)
    frame_count = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        idx = min(frame_count, N - 1)

        # --- ACTION-AWARE ZOOM/FRAMING ---------------------------------------
        # Current smoothed center & view size
        cx = centers_x[idx]
        cy = centers_y[idx]
        vw = view_w[idx]
        vh = view_h[idx]

        # Estimate per-frame ball speed vector (px/frame)
        if idx > 0:
            vx = centers_x[idx] - centers_x[idx - 1]
            vy = centers_y[idx] - centers_y[idx - 1]
        else:
            vx = vy = 0.0
        ball_speed = float(np.hypot(vx, vy))

        # Heuristics:
        # - Dribbling: slow => slightly tighter view, modest lead
        # - Pass: medium => wider view, lead more in ball direction
        # - Shot: fast & near goal line => widest view, extra headroom toward goal
        vw_scale = 1.00
        vh_scale = 1.00
        lead_px = 0.0

        # thresholds (tune if needed)
        slow_thr = 4.0  # px/frame
        fast_thr = 12.0  # px/frame
        goal_band = 0.15  # within 15% of either side horizontally counts as "near goal"

        A_out = float(W_out) / float(H_out)

        # classify
        near_left = cx <= goal_band * W_src
        near_right = cx >= (1.0 - goal_band) * W_src
        near_goal_side = near_left or near_right

        if ball_speed <= slow_thr:
            # Likely dribbling: tighter zoom, a little forward lead
            vw_scale = 0.90
            vh_scale = 0.90
            lead_px = np.clip(ball_speed * 6.0, 0, 120)
        elif ball_speed <= fast_thr:
            # Likely a pass: widen to include passer/receiver context; lead more
            vw_scale = 1.20
            vh_scale = 1.10
            lead_px = np.clip(ball_speed * 10.0, 40, 200)
        else:
            # Fast: shot or long pass. If near a side, bias toward that "goal" and widen more
            vw_scale = 1.30 if near_goal_side else 1.20
            vh_scale = 1.15
            lead_px = np.clip(ball_speed * 12.0, 80, 240)

        # Apply scaling to requested window (clamped later)
        eff_w = int(round(np.clip(vw * vw_scale, 64, W_src)))
        eff_h = int(round(np.clip(vh * vh_scale, 64, H_src)))

        # Keep the internal camera window aspect flexible (we'll letterbox at the very end)
        # but try not to get insanely skinny/tall:
        eff_w = int(np.clip(eff_w, 64, W_src))
        eff_h = int(np.clip(eff_h, 64, H_src))

        # Directional lead: look slightly ahead of motion
        cx_lead = cx + np.sign(vx) * lead_px

        # Initial placement using requested left fraction
        left_val = cx_lead - args.left_frac * eff_w
        top_val = cy - 0.5 * eff_h

        # --- BALL-IN-FRAME GUARANTEE -----------------------------------------
        # The ball must be at least keep_margin inside all sides. If not possible, zoom out.
        margin = float(args.keep_margin)

        def fits_ball_with_margin(lv, tv, ew, eh):
            return (
                (cx - lv) >= margin
                and (lv + ew - cx) >= margin
                and (cy - tv) >= margin
                and (tv + eh - cy) >= margin
            )

        # Try up to a few expansions if the ball is too close to an edge
        for _ in range(6):
            if fits_ball_with_margin(left_val, top_val, eff_w, eff_h):
                break
            # Zoom out a notch (expand window equally)
            eff_w = int(min(W_src, eff_w * 1.12))
            eff_h = int(min(H_src, eff_h * 1.12))
            left_val = cx_lead - args.left_frac * eff_w
            top_val = cy - 0.5 * eff_h

        # Clamp to source bounds
        eff_w = max(2, min(eff_w, W_src))
        eff_h = max(2, min(eff_h, H_src))
        left_int = int(round(clamp(left_val, 0, max(W_src - eff_w, 0))))
        top_int = int(round(clamp(top_val, 0, max(H_src - eff_h, 0))))
        right_int = min(W_src, left_int + eff_w)
        bottom_int = min(H_src, top_int + eff_h)

        # --- CROP & PAD TO INTERNAL WINDOW -----------------------------------
        crop = bgr[top_int:bottom_int, left_int:right_int]
        pad_bottom = max(eff_h - crop.shape[0], 0)
        pad_right = max(eff_w - crop.shape[1], 0)
        if pad_bottom > 0 or pad_right > 0:
            crop = cv2.copyMakeBorder(crop, 0, pad_bottom, 0, pad_right, borderType=cv2.BORDER_REPLICATE)

        # --- *NO WARPING*: FINAL RESIZE WITH LETTERBOX TO (W_out,H_out) -------
        frame = letterbox_to_size(crop, W_out, H_out)
        cv2.imwrite(os.path.join(tmp, f"f_{frame_count:06d}.jpg"), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
        frame_count += 1
    cap.release()

    frames_out = df["frame"].ffill().bfill().astype(int).to_numpy()
    times_out = df["time"].ffill().bfill().to_numpy(dtype=float)

    debug_csv = pd.DataFrame(
        {
            "frame": frames_out[:frame_count],
            "time": times_out[:frame_count],
            "center_x": centers_x[:frame_count],
            "center_y": centers_y[:frame_count],
            "view_w": view_w[:frame_count],
            "view_h": view_h[:frame_count],
            "ball_x": cx_filled[:frame_count],
            "ball_y": cy_filled[:frame_count],
            "speed": speed_arr[:frame_count],
            "lead_px": lead_px_arr[:frame_count],
            "visible": df["visible"].to_numpy(dtype=int)[:frame_count],
            "conf": df["conf"].to_numpy(dtype=float)[:frame_count],
            "miss_streak": df["miss_streak"].to_numpy(dtype=int)[:frame_count],
        }
    )
    debug_csv_path = os.path.join(out_dir or ".", "virtual_cam.csv")
    debug_csv.to_csv(debug_csv_path, index=False)

    fr_exact = f"{fps:.3f}"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fr_exact),        # use exact fps for the image sequence
            "-i",
            os.path.join(tmp, "f_%06d.jpg"),
            "-i",
            args.clip,
            "-filter_complex",
            "[0:v]colorspace=iall=bt470bg:all=bt709:irange=pc:range=tv,format=yuv420p[v]",
            "-map",
            "[v]",
            "-map",
            "1:a:0?",
            "-r",
            str(fr_exact),                 # force CFR on the output stream too
            "-vsync",
            "cfr",                     # avoid ffmpeg inserting/duplicating frames
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "19",
            "-x264-params",
            "keyint=120:min-keyint=120:scenecut=0",
            "-pix_fmt",
            "yuv420p",
            "-profile:v",
            "high",
            "-level",
            "4.0",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-shortest",
            "-movflags",
            "+faststart",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            args.out_mp4,
        ],
        check=True,
    )

    print("Wrote", args.out_mp4)


if __name__ == "__main__":
    main()


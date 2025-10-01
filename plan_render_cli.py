# plan_render_cli.py
import argparse
import os
import shutil
import subprocess
from pathlib import Path

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
    return sx, sy, vx, vy, kx, ky


def apply_dead_reckoning(
    sx,
    sy,
    kx,
    ky,
    vx,
    vy,
    visible,
    fps,
    max_seconds=2.0,
):
    n = len(sx)
    out_x = np.asarray(sx, dtype=np.float32).copy()
    out_y = np.asarray(sy, dtype=np.float32).copy()
    out_vx = np.asarray(vx, dtype=np.float32).copy()
    out_vy = np.asarray(vy, dtype=np.float32).copy()
    occ = np.zeros(n, dtype=np.int32)

    max_frames = max(1, int(round(float(fps) * float(max_seconds))))

    pred_x = np.nan
    pred_y = np.nan
    pred_vx = 0.0
    pred_vy = 0.0
    occlusion_run = 0

    for i in range(n):
        vis = False
        if i < len(visible):
            vis = bool(visible[i])

        kx_i = float(kx[i]) if i < len(kx) else np.nan
        ky_i = float(ky[i]) if i < len(ky) else np.nan
        vx_i = float(vx[i]) if i < len(vx) else 0.0
        vy_i = float(vy[i]) if i < len(vy) else 0.0

        if vis and np.isfinite(out_x[i]) and np.isfinite(out_y[i]):
            pred_x = kx_i if np.isfinite(kx_i) else out_x[i]
            pred_y = ky_i if np.isfinite(ky_i) else out_y[i]
            pred_vx = vx_i
            pred_vy = vy_i
            occlusion_run = 0
            occ[i] = 0
        elif vis:
            # Visible but smoothed value is missing; fall back to raw Kalman state.
            if np.isfinite(kx_i) and np.isfinite(ky_i):
                out_x[i] = kx_i
                out_y[i] = ky_i
                pred_x = kx_i
                pred_y = ky_i
            elif np.isfinite(pred_x) and np.isfinite(pred_y):
                out_x[i] = pred_x
                out_y[i] = pred_y
            pred_vx = vx_i
            pred_vy = vy_i
            occlusion_run = 0
            occ[i] = 0
        else:
            occlusion_run += 1
            occ[i] = occlusion_run
            if occlusion_run <= max_frames and np.isfinite(pred_x) and np.isfinite(pred_y):
                pred_x += pred_vx
                pred_y += pred_vy
                out_x[i] = pred_x
                out_y[i] = pred_y
                out_vx[i] = pred_vx
                out_vy[i] = pred_vy
            else:
                out_x[i] = np.nan
                out_y[i] = np.nan
                out_vx[i] = 0.0
                out_vy[i] = 0.0

    return out_x, out_y, out_vx, out_vy, occ, max_frames


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
    ap.add_argument("--keep_margin", type=float, default=250.0)
    ap.add_argument("--zoom_min", type=float, default=1.00)
    ap.add_argument("--zoom_max", type=float, default=1.55)
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

    df = df.sort_values("frame").drop_duplicates(subset="frame", keep="first").reset_index(drop=True)

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

    (
        sx_raw,
        sy_raw,
        vx_raw,
        vy_raw,
        kx_raw,
        ky_raw,
    ) = smooth_xy(cx_np, cy_np, ema_alpha=0.25, k_q=0.06, k_r=6.0)

    visible_arr = df["visible"].to_numpy(dtype=int)

    (
        sx,
        sy,
        vx,
        vy,
        occlusion_frames,
        deadreckon_limit,
    ) = apply_dead_reckoning(
        sx_raw,
        sy_raw,
        kx_raw,
        ky_raw,
        vx_raw,
        vy_raw,
        visible_arr,
        fps,
        max_seconds=2.0,
    )

    speed_arr = np.hypot(vx, vy)
    lead_scale = np.clip(speed_arr / 18.0, 0.0, 1.0)
    lead_px_arr = 24.0 * lead_scale
    tx = sx + vx * 0.25 * lead_scale
    ty = sy + vy * 0.25 * lead_scale

    # --- Viewport planning with hard aspect-ratio crop --------------------
    out_ar = float(OUT_W) / float(OUT_H)
    src_ar = float(SRC_W) / float(SRC_H if SRC_H else 1)

    if src_ar >= out_ar:
        base_crop_h = float(SRC_H)
        base_crop_w = base_crop_h * out_ar
    else:
        base_crop_w = float(SRC_W)
        base_crop_h = base_crop_w / out_ar

    base_crop_w = min(float(SRC_W), max(2.0, base_crop_w))
    base_crop_h = min(float(SRC_H), max(2.0, base_crop_h))

    zoom_min_factor = max(1.0, float(args.zoom_min))
    zoom_max_factor = max(zoom_min_factor, float(args.zoom_max))

    max_pan_per_frame = 25.0
    max_pan_accel = 12.0
    max_zoom_step = 0.03
    max_zoom_accel = 0.015

    safe_box_frac = 0.60

    target_x = np.where(np.isfinite(tx), tx, cx_interp)
    target_y = np.where(np.isfinite(ty), ty, cy_interp)

    stage_a_x = ema(target_x, alpha=0.30)
    stage_a_y = ema(target_y, alpha=0.30)

    centers_x = np.zeros_like(stage_a_x, dtype=np.float32)
    centers_y = np.zeros_like(stage_a_y, dtype=np.float32)
    zoom_series = np.zeros_like(stage_a_x, dtype=np.float32)
    view_w = np.zeros_like(stage_a_x, dtype=np.float32)
    view_h = np.zeros_like(stage_a_y, dtype=np.float32)
    pan_delta_x = np.zeros_like(stage_a_x, dtype=np.float32)
    pan_delta_y = np.zeros_like(stage_a_x, dtype=np.float32)
    zoom_delta_series = np.zeros_like(stage_a_x, dtype=np.float32)

    def compute_zoom_target(speed_px, occ_ratio):
        if np.isnan(speed_px):
            base = zoom_min_factor
        elif speed_px <= 4.0:
            base = min(zoom_max_factor, zoom_min_factor + 0.20)
        elif speed_px <= 12.0:
            base = min(zoom_max_factor, zoom_min_factor + 0.08)
        else:
            base = zoom_min_factor
        if occ_ratio > 0.0:
            base = min(zoom_max_factor, base + 0.25 * occ_ratio)
        return base

    half_w0 = base_crop_w * 0.5 / zoom_min_factor
    half_h0 = base_crop_h * 0.5 / zoom_min_factor

    centers_x[0] = clamp(stage_a_x[0], half_w0, SRC_W - half_w0)
    centers_y[0] = clamp(stage_a_y[0], half_h0, SRC_H - half_h0)
    zoom_series[0] = compute_zoom_target(speed_arr[0], 0.0)
    zoom_series[0] = clamp(zoom_series[0], zoom_min_factor, zoom_max_factor)
    view_w[0] = base_crop_w / zoom_series[0]
    view_h[0] = base_crop_h / zoom_series[0]

    for t in range(1, len(stage_a_x)):
        occ_frames = int(occlusion_frames[t]) if t < len(occlusion_frames) else 0
        occ_ratio = min(1.0, occ_frames / float(max(1, deadreckon_limit))) if occ_frames > 0 else 0.0

        desired_zoom = compute_zoom_target(speed_arr[t], occ_ratio)
        prev_zoom = zoom_series[t - 1]
        zoom_step = clamp(desired_zoom - prev_zoom, -max_zoom_step, max_zoom_step)
        zoom_step = clamp(zoom_step, zoom_delta_series[t - 1] - max_zoom_accel, zoom_delta_series[t - 1] + max_zoom_accel)
        zoom_new = clamp(prev_zoom + zoom_step, zoom_min_factor, zoom_max_factor)
        zoom_series[t] = zoom_new
        zoom_delta_series[t] = zoom_step

        vw = base_crop_w / zoom_new
        vh = base_crop_h / zoom_new
        view_w[t] = vw
        view_h[t] = vh

        prev_cx = centers_x[t - 1]
        prev_cy = centers_y[t - 1]

        safe_half_w = 0.5 * vw * safe_box_frac
        safe_half_h = 0.5 * vh * safe_box_frac

        cx_tgt = stage_a_x[t]
        cy_tgt = stage_a_y[t]

        desired_cx = prev_cx
        if cx_tgt < prev_cx - safe_half_w:
            desired_cx = cx_tgt + safe_half_w
        elif cx_tgt > prev_cx + safe_half_w:
            desired_cx = cx_tgt - safe_half_w

        desired_cy = prev_cy
        if cy_tgt < prev_cy - safe_half_h:
            desired_cy = cy_tgt + safe_half_h
        elif cy_tgt > prev_cy + safe_half_h:
            desired_cy = cy_tgt - safe_half_h

        step_x = clamp(desired_cx - prev_cx, -max_pan_per_frame, max_pan_per_frame)
        step_x = clamp(step_x, pan_delta_x[t - 1] - max_pan_accel, pan_delta_x[t - 1] + max_pan_accel)

        step_y = clamp(desired_cy - prev_cy, -max_pan_per_frame, max_pan_per_frame)
        step_y = clamp(step_y, pan_delta_y[t - 1] - max_pan_accel, pan_delta_y[t - 1] + max_pan_accel)

        new_cx = prev_cx + step_x
        new_cy = prev_cy + step_y

        half_w = vw * 0.5
        half_h = vh * 0.5

        new_cx, new_cy = clamp_vec((new_cx, new_cy), half_w, half_h, SRC_W, SRC_H)

        centers_x[t] = new_cx
        centers_y[t] = new_cy
        pan_delta_x[t] = step_x
        pan_delta_y[t] = step_y

    x0_raw = centers_x - view_w * 0.5
    y0_raw = centers_y - view_h * 0.5

    def smooth_axis(arr):
        arr = np.asarray(arr, dtype=np.float32)
        if len(arr) < 5:
            return arr
        if len(arr) >= 9:
            window = 9
        else:
            window = len(arr) if len(arr) % 2 == 1 else max(3, len(arr) - 1)
        if window < 3 or window > len(arr):
            return arr
        return savgol_filter(arr, window_length=window, polyorder=2, mode="interp")

    x0_smooth = smooth_axis(x0_raw)
    y0_smooth = smooth_axis(y0_raw)

    x0_clamped = np.clip(x0_smooth, 0.0, np.maximum(0.0, SRC_W - view_w))
    y0_clamped = np.clip(y0_smooth, 0.0, np.maximum(0.0, SRC_H - view_h))

    centers_x = (x0_clamped + view_w * 0.5).astype(np.float32)
    centers_y = (y0_clamped + view_h * 0.5).astype(np.float32)

    view_w = view_w.astype(np.float32)
    view_h = view_h.astype(np.float32)
    crop_x = x0_clamped.astype(np.float32)
    crop_y = y0_clamped.astype(np.float32)
    crop_h_arr = np.clip(np.round(view_h)).astype(int)
    crop_h_arr = np.clip(crop_h_arr, 2, SRC_H)
    crop_w_arr = np.clip(np.round(out_ar * crop_h_arr)).astype(int)
    crop_w_arr = np.clip(crop_w_arr, 2, SRC_W)

    df["center_x"] = centers_x
    df["center_y"] = centers_y
    df["view_w"] = view_w
    df["view_h"] = view_h
    df["crop_x"] = crop_x
    df["crop_y"] = crop_y
    df["crop_w"] = crop_w_arr
    df["crop_h"] = crop_h_arr

    cx_filled = np.where(
        np.isfinite(sx),
        sx,
        np.where(np.isfinite(sx_raw), sx_raw, cx_interp),
    )
    cy_filled = np.where(
        np.isfinite(sy),
        sy,
        np.where(np.isfinite(sy_raw), sy_raw, cy_interp),
    )

    out_dir = os.path.dirname(args.out_mp4) or "."
    os.makedirs(out_dir, exist_ok=True)
    out_frames_dir = Path(out_dir) / "_temp_frames"
    if out_frames_dir.exists():
        shutil.rmtree(out_frames_dir)
    out_frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.clip)
    frame_count = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        idx = min(frame_count, N - 1)

        # --- AR-PRESERVING CROP ----------------------------------------------
        cx = centers_x[idx]
        cy = centers_y[idx]
        crop_w = int(crop_w_arr[idx]) if idx < len(crop_w_arr) else int(out_ar * SRC_H)
        crop_h = int(crop_h_arr[idx]) if idx < len(crop_h_arr) else SRC_H
        crop_w = max(2, min(crop_w, SRC_W))
        crop_h = max(2, min(crop_h, SRC_H))

        left = int(round(np.clip(crop_x[idx], 0.0, SRC_W - crop_w)))
        top = int(round(np.clip(crop_y[idx], 0.0, SRC_H - crop_h)))
        right = min(SRC_W, left + crop_w)
        bottom = min(SRC_H, top + crop_h)

        crop = bgr[top:bottom, left:right]

        if crop.shape[0] <= 0 or crop.shape[1] <= 0:
            frame = np.zeros((H_out, W_out, 3), dtype=np.uint8)
        else:
            res = cv2.resize(crop, (W_out, H_out), interpolation=cv2.INTER_CUBIC)
            frame = res
        cv2.imwrite(
            str(out_frames_dir / f"f_{frame_count:06d}.jpg"),
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 96],
        )
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
            "crop_x": crop_x[:frame_count],
            "crop_y": crop_y[:frame_count],
            "crop_w": crop_w_arr[:frame_count],
            "crop_h": crop_h_arr[:frame_count],
            "ball_x": cx_filled[:frame_count],
            "ball_y": cy_filled[:frame_count],
            "speed": speed_arr[:frame_count],
            "lead_px": lead_px_arr[:frame_count],
            "visible": df["visible"].to_numpy(dtype=int)[:frame_count],
            "conf": df["conf"].to_numpy(dtype=float)[:frame_count],
            "miss_streak": df["miss_streak"].to_numpy(dtype=int)[:frame_count],
            "occlusion_frames": occlusion_frames[:frame_count],
        }
    )
    debug_csv_path = os.path.join(out_dir, "virtual_cam.csv")
    debug_csv.to_csv(debug_csv_path, index=False)

    fr_exact = f"{fps:.3f}"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fr_exact),        # use exact fps for the image sequence
            "-i",
            str(out_frames_dir / "f_%06d.jpg"),
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


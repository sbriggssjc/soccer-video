# plan_render_cli.py
import argparse
import math
import os
import subprocess
from typing import Iterable, List, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def even(n: int) -> int:
    n = int(math.floor(n / 2) * 2)
    return max(n, 0)


def clamp(v: float, lo: float, hi: float) -> float:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def savgol(arr: Iterable[float], fps: float, seconds: float = 0.5) -> np.ndarray:
    arr_np = np.asarray(arr, dtype=float)
    n = len(arr_np)
    if n < 5:
        return arr_np
    win = max(5, int(round(fps * seconds)))
    if win % 2 == 0:
        win += 1
    if win > n:
        win = n - (1 - n % 2)
    if win < 5:
        return arr_np
    return savgol_filter(arr_np, window_length=win, polyorder=2, mode="interp")


def fill_small_gaps(series: pd.Series, limit: int = 10) -> pd.Series:
    filled = series.copy()
    filled = filled.interpolate(method="nearest", limit=limit, limit_direction="both")
    filled = filled.fillna(method="ffill", limit=limit)
    filled = filled.fillna(method="bfill", limit=limit)
    return filled


def map_speed_to_zoom(speed: float, speed_tight: float, speed_wide: float, zoom_min: float, zoom_max: float) -> float:
    if math.isclose(speed_tight, speed_wide):
        return (zoom_min + zoom_max) * 0.5
    if speed_wide < speed_tight:
        ratio = clamp((speed - speed_wide) / (speed_tight - speed_wide), 0.0, 1.0)
        return zoom_min + (zoom_max - zoom_min) * ratio
    ratio = clamp((speed - speed_tight) / (speed_wide - speed_tight), 0.0, 1.0)
    return zoom_max + (zoom_min - zoom_max) * ratio


def ensure_even(value: float) -> int:
    v = int(round(value))
    if v % 2 != 0:
        v -= 1
    return max(v, 2)


def compute_velocity(values: np.ndarray, fps: float) -> np.ndarray:
    if len(values) == 0:
        return np.zeros(0, dtype=float)
    grad = np.gradient(values, edge_order=2) * fps
    grad = np.nan_to_num(grad, nan=0.0)
    return grad


def load_track(args, fps: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(args.track_csv, engine="python", on_bad_lines="skip")

    frame_series = pd.to_numeric(df.get("frame"), errors="coerce")
    if frame_series.isna().all():
        frame_series = pd.Series(np.arange(len(df), dtype=float))
    frame_series = frame_series.fillna(method="ffill").fillna(method="bfill")
    frames = frame_series.to_numpy(dtype=int)

    time_series = pd.to_numeric(df.get("time"), errors="coerce")
    if time_series.isna().all():
        time_series = pd.Series(frames, dtype=float) / max(fps, 1e-6)
    times = time_series.fillna(method="ffill").fillna(method="bfill").to_numpy(dtype=float)

    cx_series = pd.to_numeric(df.get("cx"), errors="coerce")
    cy_series = pd.to_numeric(df.get("cy"), errors="coerce")

    cx_filled = fill_small_gaps(cx_series, limit=10)
    cy_filled = fill_small_gaps(cy_series, limit=10)

    cx_interp = cx_filled.fillna(method="ffill").fillna(method="bfill")
    cy_interp = cy_filled.fillna(method="ffill").fillna(method="bfill")

    cx = savgol(cx_interp.to_numpy(dtype=float), fps, seconds=0.5)
    cy = savgol(cy_interp.to_numpy(dtype=float), fps, seconds=0.5)

    vx_series = pd.to_numeric(df.get("vx"), errors="coerce") if "vx" in df else None
    vy_series = pd.to_numeric(df.get("vy"), errors="coerce") if "vy" in df else None

    if vx_series is None or vx_series.isna().all():
        vx = compute_velocity(cx, fps)
    else:
        vx = fill_small_gaps(vx_series, limit=10).fillna(method="ffill").fillna(method="bfill").to_numpy(dtype=float)
    if vy_series is None or vy_series.isna().all():
        vy = compute_velocity(cy, fps)
    else:
        vy = fill_small_gaps(vy_series, limit=10).fillna(method="ffill").fillna(method="bfill").to_numpy(dtype=float)

    conf_series = pd.to_numeric(df.get("conf"), errors="coerce") if "conf" in df else pd.Series(np.nan, index=df.index)
    conf = fill_small_gaps(conf_series, limit=5).fillna(0.0).to_numpy(dtype=float)

    visible_series = pd.to_numeric(df.get("visible"), errors="coerce") if "visible" in df else None
    finite_mask = np.isfinite(cx_series.to_numpy(dtype=float)) & np.isfinite(cy_series.to_numpy(dtype=float))
    if visible_series is None or visible_series.isna().all():
        visible = finite_mask.astype(bool)
    else:
        visible = (visible_series.fillna(0.0).to_numpy(dtype=float) > 0.5) & finite_mask

    return frames, times, cx, cy, vx, vy, visible.astype(bool), conf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--track_csv", required=True)
    ap.add_argument("--out_mp4", required=True)
    ap.add_argument("--W_out", type=int, default=608)
    ap.add_argument("--H_out", type=int, default=1080)

    # dynamics
    ap.add_argument("--slew", type=float, default=110.0)
    ap.add_argument("--accel", type=float, default=320.0)
    ap.add_argument("--zoom_rate", type=float, default=0.14)
    ap.add_argument("--zoom_accel", type=float, default=0.45)
    ap.add_argument("--left_frac", type=float, default=0.44)
    ap.add_argument("--speed_tight", type=float, default=60.0)
    ap.add_argument("--speed_wide", type=float, default=220.0)
    ap.add_argument("--hyst", type=float, default=35.0)
    ap.add_argument("--zoom_min", type=float, default=1.00)
    ap.add_argument("--zoom_max", type=float, default=1.60)

    # new planning controls
    ap.add_argument("--lookahead_s", type=float, default=0.60)
    ap.add_argument("--keep_margin", type=int, default=140)
    ap.add_argument("--start_wide_s", type=float, default=0.80)
    ap.add_argument("--min_streak", type=int, default=18)
    ap.add_argument("--loss_streak", type=int, default=10)
    ap.add_argument("--prewiden_factor", type=float, default=1.15)
    ap.add_argument("--pass_speed", type=float, default=380.0)
    ap.add_argument("--pass_lookahead_s", type=float, default=0.45)
    ap.add_argument("--max_jerk", type=float, default=900.0)
    ap.add_argument("--zoom_jerk", type=float, default=1.0)

    args = ap.parse_args()

    cap = cv2.VideoCapture(args.clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    frames, times, cx, cy, vx, vy, visible, conf = load_track(args, fps)
    N = len(cx)
    if N == 0:
        raise RuntimeError("Track CSV is empty")

    crop_w_base = even(int(math.floor(H * args.W_out / max(args.H_out, 1)) ))
    crop_w_base = min(crop_w_base, even(W))
    crop_w_base = max(2, crop_w_base)

    dt = 1.0 / max(fps, 1e-6)
    lookahead = max(1, int(round(args.lookahead_s * fps)))
    pass_lookahead = max(1, int(round(args.pass_lookahead_s * fps)))

    speed = np.abs(vx)
    speed_filtered = np.zeros_like(speed)
    speed_state = speed[0] if len(speed) else 0.0
    for i, s in enumerate(speed):
        if i == 0:
            speed_state = s
        else:
            if s > speed_state + args.hyst:
                speed_state = s - args.hyst
            elif s < speed_state - args.hyst:
                speed_state = s + args.hyst
        speed_filtered[i] = max(speed_state, 0.0)

    left_des = np.zeros(N, dtype=float)
    zoom_des = np.zeros(N, dtype=float)
    states: List[str] = []

    visible_streak = 0
    invisible_streak = 0
    state = "acquire"

    cx_last_valid = W / 2.0

    for i in range(N):
        vis = bool(visible[i])
        if vis:
            visible_streak += 1
            invisible_streak = 0
        else:
            invisible_streak += 1
            visible_streak = 0

        t_cur = times[i] if i < len(times) else i * dt

        if state == "acquire":
            if t_cur >= args.start_wide_s and visible_streak >= args.min_streak:
                state = "track"
        elif state == "track":
            if invisible_streak >= args.loss_streak:
                state = "reacquire"
        elif state == "reacquire":
            if visible_streak >= args.min_streak:
                state = "track"

        states.append(state)

        cx_val = cx[i]
        if np.isfinite(cx_val):
            cx_last_valid = cx_val
        cx_use = cx_last_valid

        window_end = min(N, i + lookahead + 1)
        cx_window = cx[i:window_end]
        vx_window = vx[i:window_end]
        cxp = np.nanmedian(cx_window) if np.isfinite(cx_window).any() else cx_use
        vxp = np.nanmedian(vx_window) if np.isfinite(vx_window).any() else vx[i]
        speed_window = np.abs(vx_window)
        speedp = np.nanmedian(speed_window) if np.isfinite(speed_window).any() else speed_filtered[i]

        pass_active = False
        if speedp >= args.pass_speed:
            window_end = min(N, i + lookahead + pass_lookahead + 1)
            cx_window = cx[i:window_end]
            vx_window = vx[i:window_end]
            if np.isfinite(cx_window).any():
                cxp = np.nanmedian(cx_window)
            if np.isfinite(vx_window).any():
                vxp = np.nanmedian(vx_window)
            speed_window = np.abs(vx_window)
            if np.isfinite(speed_window).any():
                speedp = np.nanmedian(speed_window)
            pass_active = True

        base_zoom = map_speed_to_zoom(speed_filtered[i], args.speed_tight, args.speed_wide, args.zoom_min, args.zoom_max)

        if state == "acquire":
            zoom_goal = args.zoom_min
        elif state == "reacquire":
            zoom_goal = clamp(base_zoom / args.prewiden_factor, args.zoom_min, args.zoom_max)
        else:
            zoom_goal = base_zoom

        if pass_active:
            zoom_goal = clamp(max(args.zoom_min, zoom_goal - 0.08), args.zoom_min, args.zoom_max)

        zoom_des[i] = clamp(zoom_goal, args.zoom_min, args.zoom_max)

        crop_w = crop_w_base / max(zoom_des[i], 1e-6)
        crop_w = clamp(crop_w, 2.0, float(W))

        if state == "acquire":
            left_target = cx_use - 0.5 * crop_w
        else:
            left_target = cxp - args.left_frac * crop_w

        left_target = clamp(left_target, 0.0, W - crop_w)
        offset = cx_use - left_target
        if offset < args.keep_margin:
            left_target -= (args.keep_margin - offset)
        elif offset > crop_w - args.keep_margin:
            left_target += offset - (crop_w - args.keep_margin)
        left_target = clamp(left_target, 0.0, W - crop_w)
        left_des[i] = left_target

    # Dynamics smoothing
    zoom_path = np.zeros(N, dtype=float)
    left_path = np.zeros(N, dtype=float)

    z = clamp(zoom_des[0], args.zoom_min, args.zoom_max)
    if z <= 0:
        z = args.zoom_min
    r_prev = 0.0
    q_prev = 0.0

    L = clamp(left_des[0], 0.0, W - crop_w_base / max(z, 1e-6))
    v = 0.0
    a_prev = 0.0

    for i in range(N):
        zoom_target = clamp(zoom_des[i], args.zoom_min, args.zoom_max)
        rate_limit = args.zoom_rate
        accel_limit = args.zoom_accel
        jerk_limit = args.zoom_jerk * dt * dt

        r_des = clamp(zoom_target - z, -rate_limit * dt, rate_limit * dt)
        accel_step = clamp(r_des - r_prev, -accel_limit * dt, accel_limit * dt)
        jerk = clamp(accel_step - q_prev, -jerk_limit, jerk_limit)
        accel_step = q_prev + jerk
        r_prev = clamp(r_prev + accel_step, -rate_limit * dt, rate_limit * dt)
        z += r_prev
        if z <= 0:
            z = args.zoom_min
        z = clamp(z, args.zoom_min, args.zoom_max)
        q_prev = accel_step
        zoom_path[i] = z

        crop_w = crop_w_base / max(z, 1e-6)
        crop_w = clamp(crop_w, 2.0, float(W))

        L_target = clamp(left_des[i], 0.0, W - crop_w)
        slew_limit = args.slew
        accel_limit_pan = args.accel
        if states[i] == "reacquire":
            slew_limit *= 0.85
            accel_limit_pan *= 0.8

        err = L_target - L
        v_des = clamp(err / dt, -slew_limit, slew_limit)
        a_step = clamp(v_des - v, -accel_limit_pan * dt, accel_limit_pan * dt)
        jerk_pan_limit = args.max_jerk * dt * dt
        jerk_pan = clamp(a_step - a_prev, -jerk_pan_limit, jerk_pan_limit)
        a_step = a_prev + jerk_pan
        v = clamp(v + a_step, -slew_limit, slew_limit)
        L += v * dt
        L = clamp(L, 0.0, W - crop_w)
        a_prev = a_step

        cx_cur = cx[i]
        if np.isfinite(cx_cur):
            offset = cx_cur - L
            margin = args.keep_margin
            crop_w = crop_w_base / max(z, 1e-6)
            crop_w = clamp(crop_w, 2.0, float(W))
            min_zoom = args.zoom_min
            if offset < margin or offset > crop_w - margin:
                delta = 0.1 * (args.zoom_max - args.zoom_min)
                new_z = clamp(z - delta, min_zoom, args.zoom_max)
                if new_z != z:
                    z = new_z
                    crop_w = crop_w_base / max(z, 1e-6)
                    crop_w = clamp(crop_w, 2.0, float(W))
                    r_prev = 0.0
                    q_prev = 0.0
                if offset < margin:
                    L = clamp(cx_cur - margin, 0.0, W - crop_w)
                elif offset > crop_w - margin:
                    L = clamp(cx_cur - (crop_w - margin), 0.0, W - crop_w)
                v = 0.0
                a_prev = 0.0
        zoom_path[i] = z
        left_path[i] = L

    out_dir = os.path.dirname(args.out_mp4)
    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, "_temp_frames")
    os.makedirs(tmp, exist_ok=True)

    cap = cv2.VideoCapture(args.clip)
    frame_count = 0
    last_good_cy = H / 2.0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        idx = min(frame_count, N - 1)
        left_val = float(left_path[idx])
        zoom_val = float(zoom_path[idx])
        if zoom_val <= 0:
            zoom_val = args.zoom_min

        crop_w_float = crop_w_base / max(zoom_val, 1e-6)
        crop_w_float = clamp(crop_w_float, 2.0, float(W))
        crop_w_int = ensure_even(crop_w_float)
        crop_w_int = min(crop_w_int, even(W))

        left_int = int(round(left_val))
        left_int = int(clamp(left_int, 0, max(W - crop_w_int, 0)))
        x2 = left_int + crop_w_int

        cy_val = cy[idx]
        if np.isfinite(cy_val):
            last_good_cy = float(cy_val)
        eff_h_float = crop_w_int * args.H_out / max(args.W_out, 1e-6)
        eff_h_int = ensure_even(eff_h_float)
        eff_h_int = min(eff_h_int, even(H))
        top = int(round(last_good_cy - eff_h_int / 2))
        top = int(clamp(top, 0, max(H - eff_h_int, 0)))
        y2 = top + eff_h_int

        crop = bgr[top:y2, left_int:x2]
        if crop.shape[1] != crop_w_int or crop.shape[0] != eff_h_int:
            pad_left = 0
            pad_right = max(crop_w_int - crop.shape[1], 0)
            pad_top = 0
            pad_bottom = max(eff_h_int - crop.shape[0], 0)
            crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, borderType=cv2.BORDER_REPLICATE)

        frame = cv2.resize(crop, (args.W_out, args.H_out), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(os.path.join(tmp, f"f_{frame_count:06d}.jpg"), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
        frame_count += 1
    cap.release()

    times_out = times[:frame_count] if len(times) >= frame_count else np.arange(frame_count, dtype=float) * dt
    frames_out = frames[:frame_count] if len(frames) >= frame_count else np.arange(frame_count, dtype=int)
    visible_out = visible[:frame_count] if len(visible) >= frame_count else np.zeros(frame_count, dtype=bool)
    cx_out = cx[:frame_count] if len(cx) >= frame_count else np.pad(cx, (0, frame_count - len(cx)), constant_values=W / 2.0)
    cy_out = cy[:frame_count] if len(cy) >= frame_count else np.pad(cy, (0, frame_count - len(cy)), constant_values=H / 2.0)
    vx_out = vx[:frame_count] if len(vx) >= frame_count else np.zeros(frame_count, dtype=float)

    debug_csv = pd.DataFrame(
        {
            "frame": frames_out,
            "t": times_out,
            "left": left_path[:frame_count],
            "zoom": zoom_path[:frame_count],
            "state": states[:frame_count],
            "visible": visible_out,
            "cx": cx_out,
            "cy": cy_out,
            "speed": np.abs(vx_out),
        }
    )
    debug_csv_path = os.path.join(out_dir, "virtual_cam.csv")
    debug_csv.to_csv(debug_csv_path, index=False)

    fr = int(round(fps))
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fr),
            "-i",
            os.path.join(tmp, "f_%06d.jpg"),
            "-i",
            args.clip,
            "-map",
            "0:v",
            "-map",
            "1:a:0?",
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


# plan_render_cli.py
import argparse
import math
import os
import subprocess
from typing import Iterable, List

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


def fill_small_gaps(values: pd.Series, limit: int = 10) -> pd.Series:
    filled = values.copy()
    filled = filled.interpolate(limit=limit, limit_direction="both")
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


def main() -> None:
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
    ap.add_argument("--lookahead", type=int, default=None)
    ap.add_argument("--widen_step", type=float, default=0.04)

    # acquisition / reacquisition
    ap.add_argument("--acquire_secs", type=float, default=0.80)
    ap.add_argument("--acquire_zoom", type=float, default=1.00)
    ap.add_argument("--acquire_margin", type=int, default=160)
    ap.add_argument("--acquire_min_streak", type=int, default=18)
    ap.add_argument("--reacquire_margin", type=int, default=200)
    ap.add_argument("--reacquire_zoom", type=float, default=1.00)
    ap.add_argument("--reacquire_streak", type=int, default=10)
    ap.add_argument("--edge_margin", type=int, default=110)
    ap.add_argument("--max_jerk", type=float, default=1200.0)
    ap.add_argument("--zoom_jerk", type=float, default=1.2)

    args = ap.parse_args()

    cap = cv2.VideoCapture(args.clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    crop_w_base = even(int(math.floor(H * args.W_out / args.H_out)))
    crop_w_base = min(crop_w_base, even(W))
    crop_w_base = max(2, crop_w_base)

    df = pd.read_csv(args.track_csv, engine="python", on_bad_lines="skip")

    default_index = pd.RangeIndex(start=0, stop=len(df), step=1)
    frame_idx = pd.to_numeric(
        df.get("frame", pd.Series(default_index, index=df.index)),
        errors="coerce",
    ).fillna(method="ffill").fillna(method="bfill")
    if len(frame_idx) == 0:
        frame_idx = pd.Series(default_index, dtype=float)
    frame_idx = frame_idx.fillna(0).astype(int)
    time_series = pd.to_numeric(
        df.get("time", pd.Series(np.nan, index=df.index)),
        errors="coerce",
    )
    if time_series.isna().all():
        time_series = frame_idx / max(fps, 1e-6)

    cx_series = pd.to_numeric(df.get("cx", pd.Series(np.nan, index=df.index)), errors="coerce")
    cy_series = pd.to_numeric(df.get("cy", pd.Series(np.nan, index=df.index)), errors="coerce")
    conf_series = pd.to_numeric(df.get("conf"), errors="coerce") if "conf" in df else pd.Series(np.nan, index=df.index)

    visible_series = df.get("visible") if "visible" in df else None
    if visible_series is not None:
        visible = pd.Series(visible_series).astype(float).fillna(0.0) > 0.5
    else:
        visible = pd.Series(np.ones(len(df), dtype=bool))

    cx_filled = fill_small_gaps(cx_series, limit=10)
    cy_filled = fill_small_gaps(cy_series, limit=10)

    cx_interp = cx_filled.fillna(method="ffill").fillna(method="bfill")
    cy_interp = cy_filled.fillna(method="ffill").fillna(method="bfill")

    cx_smoothed = savgol(cx_interp.to_numpy(dtype=float), fps, seconds=0.5)
    cy_smoothed = savgol(cy_interp.to_numpy(dtype=float), fps, seconds=0.5)

    finite_mask = np.isfinite(cx_series.to_numpy(dtype=float)) & np.isfinite(cy_series.to_numpy(dtype=float))
    visible = visible.to_numpy(dtype=bool) & finite_mask

    N = len(cx_smoothed)
    times = time_series.to_numpy(dtype=float)
    frames = frame_idx.to_numpy(dtype=int)
    states: List[str] = []

    if N == 0:
        left_path = np.array([0.0], dtype=float)
        zoom_path = np.array([args.zoom_min], dtype=float)
        cy_smoothed = np.array([H / 2.0], dtype=float)
        states = ["acquire"]
    else:
        vx = np.gradient(cx_smoothed, edge_order=2) * fps
        vx = np.nan_to_num(vx, nan=0.0)
        speed = np.abs(vx)

        speed_filtered = np.zeros_like(speed)
        prev_speed = 0.0
        for i, s in enumerate(speed):
            if i == 0:
                prev_speed = s
            else:
                if s > prev_speed + args.hyst:
                    prev_speed = s - args.hyst
                elif s < prev_speed - args.hyst:
                    prev_speed = s + args.hyst
                else:
                    prev_speed = prev_speed
            speed_filtered[i] = max(prev_speed, 0.0)

        left_des = np.zeros(N, dtype=float)
        zoom_des = np.zeros(N, dtype=float)
        cx_targets = np.zeros(N, dtype=float)

        visible_streak = 0
        invisible_streak = 0
        state = "acquire"
        last_good_cx = None

        for i in range(N):
            vis = bool(visible[i])
            if vis:
                visible_streak += 1
                invisible_streak = 0
            else:
                invisible_streak += 1
                visible_streak = 0

            current_time = times[i] if i < len(times) else i / max(fps, 1e-6)

            if state == "acquire":
                if visible_streak >= args.acquire_min_streak:
                    state = "track"
                elif current_time >= args.acquire_secs:
                    state = "reacquire"
            elif state == "track":
                if invisible_streak >= args.reacquire_streak:
                    state = "reacquire"
            elif state == "reacquire":
                if visible_streak >= args.reacquire_streak:
                    state = "track"

            states.append(state)

            cx_val = cx_smoothed[i]
            if np.isfinite(cx_val):
                last_good_cx = cx_val
            elif last_good_cx is None:
                last_good_cx = W / 2.0
            cx_use = last_good_cx if last_good_cx is not None else W / 2.0
            cx_targets[i] = cx_use

            if state == "track":
                zoom_goal = map_speed_to_zoom(speed_filtered[i], args.speed_tight, args.speed_wide, args.zoom_min, args.zoom_max)
            elif state == "reacquire":
                zoom_goal = args.reacquire_zoom
            else:
                zoom_goal = args.acquire_zoom

            zoom_goal = clamp(zoom_goal, args.zoom_min, args.zoom_max)
            zoom_des[i] = zoom_goal

            crop_w = crop_w_base / max(zoom_goal, 1e-6)
            crop_w = float(clamp(crop_w, 2.0, float(W)))

            Xball = clamp(cx_use, args.edge_margin, W - args.edge_margin)
            Ldes = clamp(Xball - args.left_frac * crop_w, 0.0, W - crop_w)

            if state in ("acquire", "reacquire"):
                margin = args.acquire_margin if state == "acquire" else args.reacquire_margin
                Ldes = clamp(cx_use - 0.5 * crop_w, 0.0, W - crop_w)
                offset = cx_use - Ldes
                if offset < margin:
                    Ldes = clamp(cx_use - margin, 0.0, W - crop_w)
                elif offset > crop_w - margin:
                    Ldes = clamp(cx_use - (crop_w - margin), 0.0, W - crop_w)

            offset = cx_use - Ldes
            if offset < args.edge_margin:
                Ldes = clamp(cx_use - args.edge_margin, 0.0, W - crop_w)
            elif offset > crop_w - args.edge_margin:
                Ldes = clamp(cx_use - (crop_w - args.edge_margin), 0.0, W - crop_w)

            left_des[i] = Ldes

        dt = 1.0 / max(fps, 1e-6)

        zoom_path = np.zeros(N, dtype=float)
        left_path = np.zeros(N, dtype=float)
        zoom = clamp(zoom_des[0], args.zoom_min, args.zoom_max)
        if zoom <= 0:
            zoom = args.zoom_min
        zoom_rate = 0.0
        zoom_acc_prev = 0.0

        left = clamp(left_des[0], 0.0, W - crop_w_base / max(zoom, 1e-6))
        v = 0.0
        a_prev = 0.0

        for i in range(N):
            z_target = clamp(zoom_des[i], args.zoom_min, args.zoom_max)
            if zoom <= 0:
                zoom = args.zoom_min
            z_err = z_target - zoom
            rate_des = clamp(z_err / dt, -args.zoom_rate, args.zoom_rate)
            rate_delta = clamp(rate_des - zoom_rate, -args.zoom_accel * dt, args.zoom_accel * dt)
            jerk_limit = args.zoom_jerk * dt * dt
            rate_delta = clamp(rate_delta - zoom_acc_prev, -jerk_limit, jerk_limit) + zoom_acc_prev
            zoom_rate = clamp(zoom_rate + rate_delta, -args.zoom_rate, args.zoom_rate)
            zoom += zoom_rate * dt
            if zoom <= 0:
                zoom = args.zoom_min
            zoom = clamp(zoom, args.zoom_min, args.zoom_max)
            zoom_acc_prev = rate_delta
            zoom_path[i] = zoom

            crop_w = crop_w_base / max(zoom, 1e-6)
            crop_w = clamp(crop_w, 2.0, float(W))

            L_target = clamp(left_des[i], 0.0, W - crop_w)
            err = L_target - left
            v_des = clamp(err / dt, -args.slew, args.slew)
            accel_step = clamp(v_des - v, -args.accel * dt, args.accel * dt)
            jerk_limit_pan = args.max_jerk * dt * dt
            accel_step = clamp(accel_step - a_prev, -jerk_limit_pan, jerk_limit_pan) + a_prev
            v = clamp(v + accel_step, -args.slew, args.slew)
            left_prev = left
            left += v * dt
            left = clamp(left, 0.0, W - crop_w)
            a_prev = accel_step

            cx_target = cx_targets[i]
            if np.isfinite(cx_target):
                offset = cx_target - left
                if offset < args.edge_margin:
                    new_left = clamp(cx_target - args.edge_margin, 0.0, W - crop_w)
                    if new_left != left:
                        left = new_left
                        if i > 0:
                            v = (left - left_path[i - 1]) / dt
                            a_prev = 0.0
                elif offset > crop_w - args.edge_margin:
                    new_left = clamp(cx_target - (crop_w - args.edge_margin), 0.0, W - crop_w)
                    if new_left != left:
                        left = new_left
                        if i > 0:
                            v = (left - left_path[i - 1]) / dt
                            a_prev = 0.0

            left_path[i] = left

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
        idx = min(frame_count, len(left_path) - 1)
        left_val = float(left_path[idx])
        zoom_val = float(zoom_path[idx]) if len(zoom_path) else args.zoom_min
        if zoom_val <= 0:
            zoom_val = args.zoom_min

        crop_w_float = crop_w_base / max(zoom_val, 1e-6)
        crop_w_float = clamp(crop_w_float, 2.0, float(W))
        crop_w_int = ensure_even(crop_w_float)
        crop_w_int = min(crop_w_int, even(W))

        left_int = int(round(left_val))
        left_int = int(clamp(left_int, 0, max(W - crop_w_int, 0)))
        x2 = left_int + crop_w_int

        cy_val = cy_smoothed[min(frame_count, len(cy_smoothed) - 1)] if len(cy_smoothed) else H / 2.0
        if np.isfinite(cy_val):
            last_good_cy = cy_val
        cy_use = last_good_cy

        eff_h_float = crop_w_int * args.H_out / max(args.W_out, 1e-6)
        eff_h_int = ensure_even(eff_h_float)
        eff_h_int = min(eff_h_int, even(H))
        top = int(round(cy_use - eff_h_int / 2))
        top = int(clamp(top, 0, max(H - eff_h_int, 0)))
        y2 = top + eff_h_int

        crop = bgr[top:y2, left_int:x2]
        if crop.shape[1] != crop_w_int or crop.shape[0] != eff_h_int:
            pad_left = 0
            pad_right = crop_w_int - crop.shape[1]
            pad_top = 0
            pad_bottom = eff_h_int - crop.shape[0]
            pad_right = max(pad_right, 0)
            pad_bottom = max(pad_bottom, 0)
            crop = cv2.copyMakeBorder(
                crop,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                borderType=cv2.BORDER_REPLICATE,
            )

        frame = cv2.resize(crop, (args.W_out, args.H_out), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(os.path.join(tmp, f"f_{frame_count:06d}.jpg"), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
        frame_count += 1
    cap.release()

    if len(left_path) > 0:
        frames_out = frames[:len(left_path)] if len(frames) >= len(left_path) else np.arange(len(left_path), dtype=int)
        times_out = times[:len(left_path)] if len(times) >= len(left_path) else np.arange(len(left_path), dtype=float) / max(fps, 1e-6)
        if len(states) >= len(left_path):
            states_out = states[:len(left_path)]
        else:
            tail_state = states[-1] if states else "acquire"
            states_out = states + [tail_state] * (len(left_path) - len(states))
        visible_out = visible[:len(left_path)] if len(visible) >= len(left_path) else np.zeros(len(left_path), dtype=bool)
        cx_out = cx_smoothed[:len(left_path)] if len(cx_smoothed) >= len(left_path) else np.pad(cx_smoothed, (0, len(left_path) - len(cx_smoothed)), constant_values=W / 2.0)
        cy_out = cy_smoothed[:len(left_path)] if len(cy_smoothed) >= len(left_path) else np.pad(cy_smoothed, (0, len(left_path) - len(cy_smoothed)), constant_values=H / 2.0)

        debug_csv = pd.DataFrame({
            "frame": frames_out,
            "t": times_out,
            "left": left_path,
            "zoom": zoom_path,
            "state": states_out,
            "visible": visible_out,
            "cx": cx_out,
            "cy": cy_out,
        })
        debug_csv_path = os.path.join(out_dir, "virtual_cam.csv")
        debug_csv.to_csv(debug_csv_path, index=False)

    fr = int(round(fps))
    subprocess.run([
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
    ], check=True)

    print("Wrote", args.out_mp4)


if __name__ == "__main__":
    main()

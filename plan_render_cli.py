# plan_render_cli.py
import argparse
import os
import subprocess

import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


# === ADD imports ===
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


def clamp(v, lo, hi):
    return float(max(lo, min(hi, v)))


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
    N = len(df)

    cx = df["cx"].to_numpy(np.float32)
    cx_series = pd.Series(cx)
    cx_filled = cx_series.interpolate(limit=12, limit_direction="both").fillna(method="ffill").fillna(method="bfill").to_numpy(np.float32)
    vx = np.gradient(cx_filled) * fps

    base_tau = getattr(args, "lookahead_s", 1.0)
    tau = base_tau * (1.0 + 0.6 * np.clip(np.abs(vx) / 400.0, 0, 1))
    lead_cx = cx_filled + tau * vx

    pass_speed = getattr(args, "pass_speed", 360.0)
    pass_mask = (np.abs(vx) > pass_speed).astype(np.float32)
    pass_look = getattr(args, "pass_lookahead_s", 0.7)
    lead_pass = cx_filled + (pass_look * vx)

    W_out = int(args.W_out)
    H_out = int(args.H_out)
    target_aspect = W_out / max(H_out, 1)
    crop_w_base = int(np.floor(H_src * target_aspect / 2.0) * 2)
    crop_w_base = min(crop_w_base, W_src)
    crop_w_base = max(2, crop_w_base)

    left_frac = float(getattr(args, "left_frac", 0.48))
    keep_margin = float(getattr(args, "keep_margin", 220.0))
    zoom_min = float(getattr(args, "zoom_min", 1.0))
    zoom_max = float(getattr(args, "zoom_max", 1.45))
    start_wide_s = float(getattr(args, "start_wide_s", 1.6))
    min_streak = int(getattr(args, "min_streak", 16))
    loss_streak = int(getattr(args, "loss_streak", 4))
    prewiden_factor = float(getattr(args, "prewiden_factor", 1.30))

    left_path = np.zeros(N, dtype=np.float32)
    zoom_path = np.ones(N, dtype=np.float32) * zoom_min

    initial_center = cx_filled[0] if np.isfinite(cx_filled[0]) else (lead_cx[0] if np.isfinite(lead_cx[0]) else W_src * 0.5)
    zoom = zoom_min
    eff_w = crop_w_base / max(zoom, 1e-6)
    left = clamp(initial_center - left_frac * eff_w, 0, W_src - eff_w)

    hyst = int(getattr(args, "hyst", 90))
    z_dead = 0.03
    stable_count = 0

    for i in range(N):
        vis = int(df.at[i, "visible"]) if i < len(df) else 0
        miss = int(df.at[i, "miss_streak"]) if i < len(df) else 0
        conf = float(df.at[i, "conf"]) if i < len(df) else 0.0
        cx_i = float(cx_filled[i]) if np.isfinite(cx_filled[i]) else float(lead_cx[i])
        lead_i = float(lead_cx[i]) if np.isfinite(lead_cx[i]) else cx_i

        t_val = df.at[i, "time"] if np.isfinite(df.at[i, "time"]) else float(i) / fps
        want_wide = (t_val < start_wide_s) or (miss >= loss_streak) or (vis == 0) or (conf < 0.15)
        if want_wide:
            zoom = max(zoom_min, zoom - 0.02 * prewiden_factor)
            stable_count = 0
        else:
            stable_count = stable_count + 1 if vis == 1 else 0
            if stable_count > min_streak:
                zoom = min(zoom_max, zoom + 0.015)

        eff_w = crop_w_base / max(zoom, 1e-6)

        alpha = float(np.clip(pass_mask[i] * 0.6, 0, 0.6))
        pred_center = (1.0 - alpha) * lead_i + alpha * lead_pass[i]

        desired_left = pred_center - left_frac * eff_w
        desired_left = clamp(desired_left, 0, W_src - eff_w)

        inner_loops = 0
        cx_target = cx_i if np.isfinite(cx_i) else pred_center
        while inner_loops < 3:
            eff_w = crop_w_base / max(zoom, 1e-6)
            left = clamp(desired_left, 0, W_src - eff_w)
            left_edge = left + keep_margin
            right_edge = left + eff_w - keep_margin
            if cx_target < left_edge - 4:
                desired_left = clamp(cx_target - keep_margin, 0, W_src - eff_w)
                if desired_left == left and zoom > zoom_min:
                    zoom = max(zoom_min, zoom - 0.04)
            elif cx_target > right_edge + 4:
                desired_left = clamp(cx_target + keep_margin - eff_w, 0, W_src - eff_w)
                if desired_left == left and zoom > zoom_min:
                    zoom = max(zoom_min, zoom - 0.04)
            else:
                break
            inner_loops += 1

        if i > 0 and abs(zoom - zoom_path[i - 1]) < z_dead:
            zoom = zoom_path[i - 1]

        zoom = clamp(zoom, zoom_min, zoom_max)
        zoom_path[i] = float(zoom)
        eff_w = crop_w_base / max(zoom, 1e-6)
        left = clamp(desired_left, 0, W_src - eff_w)
        left = clamp(left, 0, W_src - eff_w)
        left_path[i] = float(left)

    slew = float(getattr(args, "slew", 80.0))
    accel = float(getattr(args, "accel", 260.0))
    max_jerk = float(getattr(args, "max_jerk", 500.0))
    z_rate = float(getattr(args, "zoom_rate", 0.10))
    z_accel = float(getattr(args, "zoom_accel", 0.30))
    z_jerk = float(getattr(args, "zoom_jerk", 0.60))

    dt = 1.0 / fps
    L = left_path.copy()
    Z = zoom_path.copy()

    L_s = smooth_series(L, fps, sec=0.7)
    Z_s = smooth_series(Z, fps, sec=0.7)

    L_clamp = np.zeros_like(L_s)
    Z_clamp = np.zeros_like(Z_s)
    v = a = 0.0
    vz = az = 0.0

    L_clamp[0] = float(L_s[0])
    Z_clamp[0] = float(clamp(Z_s[0], zoom_min, zoom_max))

    for i in range(1, N):
        v_des = (L_s[i] - L_clamp[i - 1]) / dt
        a_des = (v_des - v) / dt
        da = np.clip(a_des - a, -max_jerk * dt, max_jerk * dt)
        a = np.clip(a + da, -accel, accel)
        v = np.clip(v + a * dt, -slew, slew)
        L_clamp[i] = L_clamp[i - 1] + v * dt

        vz_des = (Z_s[i] - Z_clamp[i - 1]) / dt
        az_des = (vz_des - vz) / dt
        daz = np.clip(az_des - az, -z_jerk * dt, z_jerk * dt)
        az = np.clip(az + daz, -z_accel, z_accel)
        vz = np.clip(vz + az * dt, -z_rate, z_rate)
        Z_clamp[i] = float(clamp(Z_clamp[i - 1] + vz * dt, zoom_min, zoom_max))

        eff_w = crop_w_base / max(Z_clamp[i], 1e-6)
        left = clamp(L_clamp[i], 0, W_src - eff_w)
        left_edge = left + keep_margin
        right_edge = left + eff_w - keep_margin
        cx_i = cx_filled[i]
        if np.isfinite(cx_i):
            if cx_i < left_edge:
                left = clamp(cx_i - keep_margin, 0, W_src - eff_w)
            elif cx_i > right_edge:
                left = clamp(cx_i + keep_margin - eff_w, 0, W_src - eff_w)
        L_clamp[i] = left

    left_path = L_clamp.astype(np.float32)
    zoom_path = Z_clamp.astype(np.float32)

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
        zoom_val = float(max(zoom_min, zoom_path[idx]))
        eff_w = int(round(crop_w_base / max(zoom_val, 1e-6)))
        eff_w = max(2, min(eff_w, W_src))
        left_val = clamp(left_path[idx], 0, W_src - eff_w)
        left_int = int(round(left_val))
        left_int = int(clamp(left_int, 0, max(W_src - eff_w, 0)))
        right_int = min(W_src, left_int + eff_w)
        crop = bgr[:, left_int:right_int]
        if crop.shape[1] != eff_w:
            pad_right = max(eff_w - crop.shape[1], 0)
            crop = cv2.copyMakeBorder(crop, 0, 0, 0, pad_right, borderType=cv2.BORDER_REPLICATE)

        frame = cv2.resize(crop, (W_out, H_out), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(os.path.join(tmp, f"f_{frame_count:06d}.jpg"), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
        frame_count += 1
    cap.release()

    frames_out = df["frame"].fillna(method="ffill").fillna(method="bfill").astype(int).to_numpy()
    times_out = df["time"].fillna(method="ffill").fillna(method="bfill").to_numpy(dtype=float)

    debug_csv = pd.DataFrame(
        {
            "frame": frames_out[:frame_count],
            "time": times_out[:frame_count],
            "left": left_path[:frame_count],
            "zoom": zoom_path[:frame_count],
            "cx": cx_filled[:frame_count],
            "lead": lead_cx[:frame_count],
            "visible": df["visible"].to_numpy(dtype=int)[:frame_count],
            "conf": df["conf"].to_numpy(dtype=float)[:frame_count],
            "miss_streak": df["miss_streak"].to_numpy(dtype=int)[:frame_count],
        }
    )
    debug_csv_path = os.path.join(out_dir or ".", "virtual_cam.csv")
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


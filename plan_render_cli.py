# plan_render_cli.py
import argparse
import math
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


# --- AR + safety helpers ------------------------------------------------------


def _target_out_ar(W_out: int, H_out: int) -> float:
    return float(W_out) / float(H_out)


def _ensure_ball_inside_crop(
    cx,
    cy,
    bx,
    by,
    view_w,
    view_h,
    margin,
    W_in,
    H_in,
):
    """
    (cx, cy) = current crop center;
    (bx, by) = ball position;
    view_w, view_h = crop size BEFORE AR lock;
    margin = safety pixels that must remain between ball and crop edge.
    Returns adjusted (cx, cy) so ball+margin is inside the crop.
    """

    half_w = view_w * 0.5
    half_h = view_h * 0.5

    # Left/right edges where the ball+margin must fit
    left = cx - half_w + margin
    right = cx + half_w - margin
    top = cy - half_h + margin
    bottom = cy + half_h - margin

    dx = 0.0
    dy = 0.0
    if bx < left:
        dx = bx - left
    if bx > right:
        dx = bx - right
    if by < top:
        dy = by - top
    if by > bottom:
        dy = by - bottom

    cx += dx
    cy += dy

    # Clamp center so the crop rectangle stays within frame (best effort before padding)
    cx = max(half_w, min(W_in - half_w, cx))
    cy = max(half_h, min(H_in - half_h, cy))
    return cx, cy


def _speed_aware_lead(
    bx_prev,
    by_prev,
    bx_now,
    by_now,
    fps,
    lead_min_s,
    lead_max_s,
    v_low=120.0,
    v_high=720.0,
):
    """
    Compute dynamic lead seconds based on pixel speed per second.
    v_low..v_high are rough thresholds for slow vs very fast ball motion.
    Returns lead seconds in [lead_min_s, lead_max_s] and a unit direction vector.
    """

    vx = (bx_now - bx_prev) * fps
    vy = (by_now - by_prev) * fps
    speed = (vx * vx + vy * vy) ** 0.5
    # Map speed -> lead seconds
    if speed <= v_low:
        lead_s = lead_min_s
    elif speed >= v_high:
        lead_s = lead_max_s
    else:
        t = (speed - v_low) / max(1e-6, (v_high - v_low))
        lead_s = (1.0 - t) * lead_min_s + t * lead_max_s

    # Direction (unit) from current to predicted
    mag = ((bx_now - bx_prev) ** 2 + (by_now - by_prev) ** 2) ** 0.5
    if mag < 1e-6:
        dirx, diry = 0.0, 0.0
    else:
        dirx, diry = (bx_now - bx_prev) / mag, (by_now - by_prev) / mag

    return lead_s, dirx, diry


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def smooth_path(vals, max_step=None, win=9, poly=2):
    vals = np.asarray(vals, dtype=np.float32)
    if max_step is not None and len(vals) > 1:
        for i in range(1, len(vals)):
            d = vals[i] - vals[i - 1]
            if d > max_step:
                vals[i] = vals[i - 1] + max_step
            elif d < -max_step:
                vals[i] = vals[i - 1] - max_step
    w = min(win if win % 2 else win - 1, len(vals) - (1 - len(vals) % 2))
    if w >= 5 and w % 2 == 1:
        vals = savgol_filter(vals, w, poly, mode="interp")
    return vals


def speed_px_per_s(xs, ys, fps, i, k=3):
    i0 = max(0, i - k)
    i1 = min(len(xs) - 1, i + k)
    dt = max(1, i1 - i0) / fps
    dx = float(xs[i1] - xs[i0])
    dy = float(ys[i1] - ys[i0])
    return np.hypot(dx, dy) / dt


def choose_lead_frames(v, fps, lead_s_min, lead_s_max, v_lo=300, v_hi=1200):
    if v <= v_lo:
        lead_s = lead_s_min
    elif v >= v_hi:
        lead_s = lead_s_max
    else:
        t = (v - v_lo) / (v_hi - v_lo)
        lead_s = (1 - t) * lead_s_min + t * lead_s_max
    return int(round(lead_s * fps))

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
    max_seconds=2.0,  # kept for back-compat (unused in new logic)
    long_max_seconds=12.0,
    vel_decay_tau_s=3.0,
):
    """
    Predicts through occlusions with exponential velocity decay and *never* emits NaNs;
    instead it switches to predicted (decayed) states. Returns:
      out_x, out_y, out_vx, out_vy, occ (occlusion run length in frames), long_limit_frames
    """
    n = len(sx)
    out_x = np.asarray(sx, dtype=np.float32).copy()
    out_y = np.asarray(sy, dtype=np.float32).copy()
    out_vx = np.asarray(vx, dtype=np.float32).copy()
    out_vy = np.asarray(vy, dtype=np.float32).copy()
    occ = np.zeros(n, dtype=np.int32)

    # state used while hidden
    pred_x = np.nan
    pred_y = np.nan
    pred_vx = 0.0
    pred_vy = 0.0
    occlusion_run = 0
    long_limit_frames = max(1, int(round(float(fps) * float(long_max_seconds))))
    dt = 1.0 / max(1e-6, float(fps))
    # per-frame decay factor so v -> v*exp(-dt/τ)
    decay = math.exp(-dt / max(1e-6, float(vel_decay_tau_s)))

    for i in range(n):
        vis = bool(visible[i]) if i < len(visible) else False

        kx_i = float(kx[i]) if i < len(kx) else np.nan
        ky_i = float(ky[i]) if i < len(ky) else np.nan
        vx_i = float(vx[i]) if i < len(vx) else 0.0
        vy_i = float(vy[i]) if i < len(vy) else 0.0

        if vis and np.isfinite(sx[i]) and np.isfinite(sy[i]):
            # lock to filtered measurement, reset predictors
            out_x[i] = sx[i]
            out_y[i] = sy[i]
            pred_x = kx_i if np.isfinite(kx_i) else sx[i]
            pred_y = ky_i if np.isfinite(ky_i) else sy[i]
            pred_vx = vx_i
            pred_vy = vy_i
            occlusion_run = 0
            occ[i] = 0
        else:
            # occluded OR missing smooth position
            occlusion_run += 1
            occ[i] = occlusion_run

            if np.isfinite(kx_i) and np.isfinite(ky_i) and vis:
                # visible but smoothed missing – fallback to Kalman
                out_x[i] = kx_i
                out_y[i] = ky_i
                pred_x, pred_y = kx_i, ky_i
                pred_vx, pred_vy = vx_i, vy_i
                occlusion_run = 0
                occ[i] = 0
            else:
                # fully hidden: integrate decaying velocity
                if not np.isfinite(pred_x) or not np.isfinite(pred_y):
                    # initialize from last finite (smooth or kalman or raw)
                    last_x = (
                        sx[i - 1]
                        if i > 0 and np.isfinite(sx[i - 1])
                        else kx[i - 1]
                        if i > 0 and np.isfinite(kx[i - 1])
                        else out_x[i - 1]
                        if i > 0
                        else 0.0
                    )
                    last_y = (
                        sy[i - 1]
                        if i > 0 and np.isfinite(sy[i - 1])
                        else ky[i - 1]
                        if i > 0 and np.isfinite(ky[i - 1])
                        else out_y[i - 1]
                        if i > 0
                        else 0.0
                    )
                    last_vx = vx[i - 1] if i > 0 else 0.0
                    last_vy = vy[i - 1] if i > 0 else 0.0
                    pred_x, pred_y = float(last_x), float(last_y)
                    pred_vx, pred_vy = float(last_vx), float(last_vy)

                # decay velocity then step position
                pred_vx *= decay
                pred_vy *= decay
                pred_x += pred_vx
                pred_y += pred_vy

                out_x[i] = pred_x
                out_y[i] = pred_y
                out_vx[i] = pred_vx
                out_vy[i] = pred_vy

    return out_x, out_y, out_vx, out_vy, occ, long_limit_frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--track_csv", required=True)
    ap.add_argument("--out_mp4", required=True)
    ap.add_argument("--W_out", type=int, default=608)
    ap.add_argument("--H_out", type=int, default=1080)

    ap.add_argument("--lead_seconds_min", type=float, default=0.4)
    ap.add_argument("--lead_seconds_max", type=float, default=0.9)
    ap.add_argument("--safe_margin_px", type=int, default=120)
    ap.add_argument("--zoom_min_w", type=int, default=520)
    ap.add_argument("--zoom_max_w", type=int, default=1280)
    ap.add_argument("--zoom_step_max", type=float, default=0.05)
    ap.add_argument("--pan_step_max", type=int, default=36)
    ap.add_argument("--envelope_seconds", type=float, default=1.2)
    ap.add_argument(
        "--dead_reckon_s",
        type=float,
        default=12.0,
        help="Max seconds to predict during occlusion (with decay).",
    )
    ap.add_argument(
        "--pred_decay_tau",
        type=float,
        default=3.0,
        help="Seconds time-constant for velocity decay during occlusion.",
    )
    ap.add_argument(
        "--occl_zoom_gain",
        type=float,
        default=0.035,
        help="Per-frame zoom widening factor while occluded.",
    )
    ap.add_argument(
        "--occl_zoom_cap",
        type=float,
        default=1.00,
        help="Max extra zoom factor (e.g., 1.0 == up to zoom_max_w).",
    )
    ap.add_argument(
        "--loss_pan_boost",
        type=float,
        default=1.75,
        help="Multiply pan_step_max while occluded.",
    )
    ap.add_argument(
        "--loss_zoom_boost",
        type=float,
        default=1.35,
        help="Multiply zoom_step_max while occluded.",
    )
    ap.add_argument(
        "--lead_on_loss_s",
        type=float,
        default=0.9,
        help="Extra lead seconds while occluded.",
    )
    ap.add_argument(
        "--envelope_seconds_loss",
        type=float,
        default=2.5,
        help="Future envelope sizing window while occluded.",
    )

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
    OUT_W = int(args.W_out)
    OUT_H = int(args.H_out)
    OUT_AR = OUT_W / OUT_H if OUT_H else 1.0

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
        long_max_seconds=args.dead_reckon_s,
        vel_decay_tau_s=args.pred_decay_tau,
    )

    speed_arr = np.hypot(vx, vy)

    ball_x = np.where(
        np.isfinite(sx),
        sx,
        np.where(np.isfinite(sx_raw), sx_raw, cx_interp),
    )
    ball_y = np.where(
        np.isfinite(sy),
        sy,
        np.where(np.isfinite(sy_raw), sy_raw, cy_interp),
    )
    ball_x = np.clip(ball_x, 0.0, float(max(1, SRC_W) - 1))
    ball_y = np.clip(ball_y, 0.0, float(max(1, SRC_H) - 1))

    vx_pred = np.zeros_like(ball_x, dtype=np.float32)
    if len(ball_x) > 1:
        vx_pred[1:] = (ball_x[1:] - ball_x[:-1]) / dt

    x0_series = []
    y0_series = []
    cw_series = []

    min_w = int(args.zoom_min_w)
    max_w = int(min(args.zoom_max_w, SRC_W))
    curr_w = clamp(min_w, int(OUT_AR * 200), max_w)
    curr_w = clamp(curr_w, min_w, max_w)
    curr_h = int(round(curr_w / OUT_AR)) if OUT_AR else SRC_H
    if curr_h > SRC_H:
        curr_h = SRC_H
        curr_w = int(round(curr_h * OUT_AR)) if OUT_AR else max_w
    curr_x0 = clamp(int(ball_x[0] - curr_w / 2), 0, max(0, SRC_W - curr_w))
    curr_y0 = clamp(int(ball_y[0] - curr_h / 2), 0, max(0, SRC_H - curr_h))

    for i in range(len(ball_x)):
        bx = float(ball_x[i])
        by = float(ball_y[i])

        occ = int(occlusion_frames[i]) if i < len(occlusion_frames) else 0
        loss_mode = occ > 0

        v = speed_px_per_s(ball_x, ball_y, fps, i, k=3)
        lead = choose_lead_frames(
            v,
            fps,
            args.lead_seconds_min,
            args.lead_seconds_max,
        )
        j = min(len(ball_x) - 1, i + lead)
        bx_lead = float(ball_x[j])
        by_lead = float(ball_y[j])

        if v < 300:
            alpha = 0.15
        elif v < 800:
            alpha = 0.35
        else:
            alpha = 0.55
        tx = (1.0 - alpha) * bx + alpha * bx_lead
        ty = (1.0 - alpha) * by + alpha * by_lead

        k1 = i
        env_N = int(
            round(
                (args.envelope_seconds_loss if loss_mode else args.envelope_seconds)
                * fps
            )
        )
        k2 = min(len(ball_x) - 1, i + env_N)
        seg_x = ball_x[k1 : k2 + 1]
        seg_y = ball_y[k1 : k2 + 1]
        x_min = float(np.min(seg_x))
        x_max = float(np.max(seg_x))
        y_min = float(np.min(seg_y))
        y_max = float(np.max(seg_y))

        need_w = (x_max - x_min) + 2 * args.safe_margin_px
        need_h = (y_max - y_min) + 2 * args.safe_margin_px

        w_from_h = int(np.ceil(need_h * OUT_AR)) if OUT_AR else max_w
        w_needed = max(int(np.ceil(need_w)), w_from_h)

        w_target = clamp(int(w_needed), min_w, max_w)

        w_change_max = int(np.ceil(curr_w * args.zoom_step_max))
        if w_change_max < 1:
            w_change_max = 1
        if w_target > curr_w + w_change_max:
            curr_w += w_change_max
        elif w_target < curr_w - w_change_max:
            curr_w -= w_change_max
        else:
            curr_w = w_target

        curr_h = int(round(curr_w / OUT_AR)) if OUT_AR else SRC_H
        if curr_h > SRC_H:
            curr_h = SRC_H
            curr_w = int(round(curr_h * OUT_AR)) if OUT_AR else max_w

        cx = clamp(int(round(tx)), 0, SRC_W)
        cy = clamp(int(round(ty)), 0, SRC_H)

        x0 = cx - curr_w // 2
        y0 = cy - curr_h // 2

        x0_min = int(np.floor(x_max - curr_w + args.safe_margin_px))
        x0_max = int(np.ceil(x_min - args.safe_margin_px))
        if x0 < x0_min:
            x0 = x0_min
        if x0 > x0_max:
            x0 = x0_max

        y0_min = int(np.floor(y_max - curr_h + args.safe_margin_px))
        y0_max = int(np.ceil(y_min - args.safe_margin_px))
        if y0 < y0_min:
            y0 = y0_min
        if y0 > y0_max:
            y0 = y0_max

        x0 = clamp(x0, 0, max(0, SRC_W - curr_w))
        y0 = clamp(y0, 0, max(0, SRC_H - curr_h))

        dx = x0 - curr_x0
        dy = y0 - curr_y0
        pan_step_max = int(args.pan_step_max)
        if abs(dx) > pan_step_max:
            x0 = curr_x0 + int(np.sign(dx)) * pan_step_max
        if abs(dy) > pan_step_max:
            y0 = curr_y0 + int(np.sign(dy)) * pan_step_max

        curr_x0 = int(x0)
        curr_y0 = int(y0)

        x0_series.append(curr_x0)
        y0_series.append(curr_y0)
        cw_series.append(int(curr_w))

    pan_step_max = int(args.pan_step_max)
    x0_smooth = smooth_path(x0_series, max_step=pan_step_max, win=9, poly=2)
    y0_smooth = smooth_path(y0_series, max_step=pan_step_max, win=9, poly=2)
    if cw_series:
        zoom_smooth_limit = int(max(1, round(args.zoom_step_max * max(cw_series))))
    else:
        zoom_smooth_limit = 1
    cw_smooth = smooth_path(cw_series, max_step=zoom_smooth_limit, win=9, poly=2)

    x0_final = []
    y0_final = []
    cw_final = []
    ch_final = []
    for x0_val, y0_val, w_val in zip(x0_smooth, y0_smooth, cw_smooth):
        w = int(clamp(int(round(w_val)), min_w, max_w))
        h = int(round(w / OUT_AR)) if OUT_AR else SRC_H
        if h > SRC_H:
            h = SRC_H
            w = int(round(h * OUT_AR)) if OUT_AR else max_w
        w = max(2, min(w, SRC_W))
        h = max(2, min(h, SRC_H))
        max_x0 = max(0, SRC_W - w)
        max_y0 = max(0, SRC_H - h)
        x0 = int(clamp(int(round(x0_val)), 0, max_x0))
        y0 = int(clamp(int(round(y0_val)), 0, max_y0))
        x0_final.append(x0)
        y0_final.append(y0)
        cw_final.append(w)
        ch_final.append(h)

    x0_arr = np.asarray(x0_final, dtype=np.float32)
    y0_arr = np.asarray(y0_final, dtype=np.float32)
    cw_arr = np.asarray(cw_final, dtype=np.float32)
    ch_arr = np.asarray(ch_final, dtype=np.float32)

    margin_px = int(args.safe_margin_px) if hasattr(args, "safe_margin_px") else 160
    lead_seconds = getattr(args, "lead_seconds_max", 1.2)
    zoom_max_w = int(getattr(args, "zoom_max_w", SRC_W))
    for i in range(len(x0_arr)):
        w = float(cw_arr[i])
        h = float(ch_arr[i])
        left = float(x0_arr[i])
        right = left + w
        cx_curr = float(ball_x[i]) if i < len(ball_x) else left + w * 0.5
        vx_i = float(vx_pred[i]) if i < len(vx_pred) and i > 0 else 0.0
        if not np.isfinite(vx_i):
            vx_i = 0.0
        cx_pred = cx_curr + vx_i * lead_seconds

        if cx_pred < left + margin_px:
            left = clamp(cx_pred - margin_px, 0.0, float(max(0.0, SRC_W - w)))
        elif cx_pred > right - margin_px:
            left = clamp(cx_pred + margin_px - w, 0.0, float(max(0.0, SRC_W - w)))

        if not np.isfinite(cx_np[i]):
            widened_w = int(round(w * 1.12))
            widened_w = min(widened_w, zoom_max_w)
            widened_w = int(clamp(widened_w, min_w, max_w))
            if widened_w > w:
                w = float(widened_w)
                h = float(int(round(w / OUT_AR)) if OUT_AR else SRC_H)
                if h > SRC_H:
                    h = float(SRC_H)
                    w = float(int(round(h * OUT_AR)) if OUT_AR else zoom_max_w)
            j = i - 1
            last_cx = None
            while j >= 0:
                if np.isfinite(cx_np[j]):
                    last_cx = float(cx_np[j])
                    break
                j -= 1
            if last_cx is not None:
                left = clamp(last_cx - 0.5 * w, 0.0, float(max(0.0, SRC_W - w)))

        w = float(clamp(w, 2.0, float(SRC_W)))
        h = float(clamp(h, 2.0, float(SRC_H)))
        left = clamp(left, 0.0, float(max(0.0, SRC_W - w)))

        x0_arr[i] = float(left)
        cw_arr[i] = float(w)
        ch_arr[i] = float(h)

    centers_x = x0_arr + cw_arr * 0.5
    centers_y = y0_arr + ch_arr * 0.5

    df["center_x"] = centers_x
    df["center_y"] = centers_y
    df["view_w"] = cw_arr
    df["view_h"] = ch_arr
    df["crop_x"] = x0_arr
    df["crop_y"] = y0_arr
    df["crop_w"] = cw_arr.astype(int)
    df["crop_h"] = ch_arr.astype(int)

    cx_filled = ball_x
    cy_filled = ball_y

    view_w = cw_arr
    view_h = ch_arr
    crop_x = x0_arr
    crop_y = y0_arr
    crop_w_arr = cw_arr.astype(int)
    crop_h_arr = ch_arr.astype(int)

    out_dir = os.path.dirname(args.out_mp4) or "."
    os.makedirs(out_dir, exist_ok=True)
    out_frames_dir = Path(out_dir) / "_temp_frames"
    if out_frames_dir.exists():
        shutil.rmtree(out_frames_dir)
    out_frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.clip)

    prev_ball_x = None
    prev_ball_y = None
    base_pan_step = float(args.pan_step_max)
    base_zoom_step = float(args.zoom_step_max)
    zoom_min_w = int(args.zoom_min_w)
    zoom_max_w = int(getattr(args, "zoom_max_w", SRC_W))
    OUT_AR = _target_out_ar(OUT_W, OUT_H) if OUT_H else 1.0

    if len(cw_arr) > 0:
        view_w_curr = float(cw_arr[0])
    else:
        view_w_curr = float(zoom_min_w)
    view_w_curr = max(float(zoom_min_w), min(float(zoom_max_w), view_w_curr))
    if len(x0_arr) > 0 and len(ch_arr) > 0:
        cx = float(x0_arr[0] + cw_arr[0] * 0.5)
        cy = float(y0_arr[0] + ch_arr[0] * 0.5)
    else:
        cx = float(SRC_W * 0.5)
        cy = float(SRC_H * 0.5)

    used_center_x = []
    used_center_y = []
    used_view_w = []
    used_view_h = []
    used_crop_x = []
    used_crop_y = []

    frame_count = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        idx = min(frame_count, N - 1)

        # --- figure target output AR ---
        OUT_AR = _target_out_ar(OUT_W, OUT_H) if OUT_H else 1.0

        # ball positions
        ball_x_now = float(ball_x[idx]) if idx < len(ball_x) else cx
        ball_y_now = float(ball_y[idx]) if idx < len(ball_y) else cy
        if not np.isfinite(ball_x_now) or not np.isfinite(ball_y_now):
            ball_x_now = cx
            ball_y_now = cy

        occ = int(occlusion_frames[idx]) if idx < len(occlusion_frames) else 0
        loss_mode = occ > 0

        # Allow faster catch-up during loss
        pan_step_now = base_pan_step * (args.loss_pan_boost if loss_mode else 1.0)
        zoom_step_now = base_zoom_step * (args.loss_zoom_boost if loss_mode else 1.0)

        # 1) Base zoom window coming from smoothing state
        target_view_w = float(cw_arr[idx]) if idx < len(cw_arr) else view_w_curr
        target_view_w = max(float(zoom_min_w), min(float(zoom_max_w), target_view_w))
        if not np.isfinite(target_view_w):
            target_view_w = view_w_curr

        if loss_mode:
            # widen proportional to occlusion run, capped
            widen_factor = min(
                1.0 + args.occl_zoom_gain * occ, 1.0 + args.occl_zoom_cap
            )
            target_view_w = min(
                float(zoom_max_w), float(target_view_w) * widen_factor
            )

        zoom_delta = target_view_w - view_w_curr
        max_zoom_delta = max(1.0, abs(view_w_curr) * zoom_step_now)
        if abs(zoom_delta) > max_zoom_delta:
            zoom_delta = math.copysign(max_zoom_delta, zoom_delta)
        view_w_curr += zoom_delta
        view_w_curr = max(float(zoom_min_w), min(float(zoom_max_w), view_w_curr))
        view_w_curr = max(2.0, min(float(SRC_W), view_w_curr))

        if OUT_AR > 0:
            view_h_curr = int(round(view_w_curr / OUT_AR))
        else:
            view_h_curr = SRC_H
        view_h_curr = max(1, min(view_h_curr, SRC_H))
        if OUT_AR > 0:
            view_w_curr = float(int(round(view_h_curr * OUT_AR)))
        view_w_curr = max(2.0, min(float(SRC_W), view_w_curr))

        # 2) Predict/lead target center using speed-aware lead
        bx_prev = prev_ball_x if prev_ball_x is not None else ball_x_now
        by_prev = prev_ball_y if prev_ball_y is not None else ball_y_now
        if not np.isfinite(bx_prev):
            bx_prev = ball_x_now
        if not np.isfinite(by_prev):
            by_prev = ball_y_now

        lead_s, dirx, diry = _speed_aware_lead(
            bx_prev,
            by_prev,
            ball_x_now,
            ball_y_now,
            fps=fps,
            lead_min_s=args.lead_seconds_min,
            lead_max_s=args.lead_seconds_max,
            v_low=120.0,
            v_high=720.0,
        )
        if loss_mode:
            lead_s = max(lead_s, args.lead_on_loss_s)
        lead_px = lead_s * fps * pan_step_now
        lead_cx = ball_x_now + dirx * lead_px
        lead_cy = ball_y_now + diry * lead_px

        dt_frame = 1.0 / fps if fps else 0.0
        t_env = min(1.0, max(0.0, 1.0 - math.exp(-dt_frame / max(1e-6, args.envelope_seconds))))
        target_cx = (1.0 - t_env) * ball_x_now + t_env * lead_cx
        target_cy = (1.0 - t_env) * ball_y_now + t_env * lead_cy

        dx = target_cx - cx
        dy = target_cy - cy
        max_dx = math.copysign(min(abs(dx), pan_step_now), dx)
        max_dy = math.copysign(min(abs(dy), pan_step_now), dy)
        cx += max_dx
        cy += max_dy

        # ensure both current and predicted ball x fit with margin
        cx_pred = ball_x_now + dirx * (lead_s * fps * pan_step_now)

        need_left = min(ball_x_now, cx_pred) - args.safe_margin_px
        need_right = max(ball_x_now, cx_pred) + args.safe_margin_px
        need_w = max(need_right - need_left, view_w_curr)

        # expand if needed
        if need_w > view_w_curr:
            view_w_curr = min(float(zoom_max_w), need_w)
            view_h_curr = int(round(view_w_curr / OUT_AR)) if OUT_AR else SRC_H
            view_h_curr = min(view_h_curr, SRC_H)
            if OUT_AR:
                # re-quantize width to AR
                view_w_curr = float(int(round(view_h_curr * OUT_AR)))

        # 3) Hard guarantee: ball must be inside the crop with a margin every frame
        cx, cy = _ensure_ball_inside_crop(
            cx,
            cy,
            ball_x_now,
            ball_y_now,
            view_w=view_w_curr,
            view_h=view_h_curr,
            margin=args.safe_margin_px,
            W_in=SRC_W,
            H_in=SRC_H,
        )
        # also pull toward predicted center when occluded
        if loss_mode:
            cx, cy = _ensure_ball_inside_crop(
                cx,
                cy,
                cx_pred,
                ball_y_now,
                view_w=view_w_curr,
                view_h=view_h_curr,
                margin=args.safe_margin_px,
                W_in=SRC_W,
                H_in=SRC_H,
            )

        # 4) Compute final integer crop rectangle (AR-locked), clamp to frame
        half_w = int(round(view_w_curr * 0.5))
        half_h = int(round(view_h_curr * 0.5))
        cx_int = int(round(cx))
        cy_int = int(round(cy))
        x0 = cx_int - half_w
        y0 = cy_int - half_h
        x1 = x0 + int(round(view_w_curr))
        y1 = y0 + int(round(view_h_curr))

        pad_left = max(0, -x0)
        pad_top = max(0, -y0)
        pad_right = max(0, x1 - SRC_W)
        pad_bottom = max(0, y1 - SRC_H)

        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(SRC_W, x1)
        y1 = min(SRC_H, y1)

        crop = bgr[y0:y1, x0:x1]

        if any([pad_left, pad_top, pad_right, pad_bottom]):
            crop = cv2.copyMakeBorder(
                crop,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                borderType=cv2.BORDER_REPLICATE,
            )

        crop_resized = cv2.resize(crop, (OUT_W, OUT_H), interpolation=cv2.INTER_LANCZOS4)

        used_center_x.append(cx)
        used_center_y.append(cy)
        used_view_w.append(int(round(view_w_curr)))
        used_view_h.append(int(round(view_h_curr)))
        used_crop_x.append(x0)
        used_crop_y.append(y0)

        prev_ball_x, prev_ball_y = float(ball_x_now), float(ball_y_now)

        cv2.imwrite(
            str(out_frames_dir / f"f_{frame_count:06d}.jpg"),
            crop_resized,
            [int(cv2.IMWRITE_JPEG_QUALITY), 96],
        )
        frame_count += 1
    cap.release()

    frames_out = df["frame"].ffill().bfill().astype(int).to_numpy()
    times_out = df["time"].ffill().bfill().to_numpy(dtype=float)

    if used_center_x:
        debug_center_x = np.asarray(used_center_x, dtype=float)
        debug_center_y = np.asarray(used_center_y, dtype=float)
        debug_view_w = np.asarray(used_view_w, dtype=int)
        debug_view_h = np.asarray(used_view_h, dtype=int)
        debug_crop_x = np.asarray(used_crop_x, dtype=int)
        debug_crop_y = np.asarray(used_crop_y, dtype=int)
    else:
        debug_center_x = centers_x[:frame_count]
        debug_center_y = centers_y[:frame_count]
        debug_view_w = view_w[:frame_count]
        debug_view_h = view_h[:frame_count]
        debug_crop_x = crop_x[:frame_count]
        debug_crop_y = crop_y[:frame_count]

    debug_csv = pd.DataFrame(
        {
            "frame": frames_out[:frame_count],
            "time": times_out[:frame_count],
            "center_x": debug_center_x,
            "center_y": debug_center_y,
            "view_w": debug_view_w,
            "view_h": debug_view_h,
            "crop_x": debug_crop_x,
            "crop_y": debug_crop_y,
            "crop_w": np.asarray(used_view_w if used_view_w else crop_w_arr[:frame_count], dtype=int),
            "crop_h": np.asarray(used_view_h if used_view_h else crop_h_arr[:frame_count], dtype=int),
            "ball_x": cx_filled[:frame_count],
            "ball_y": cy_filled[:frame_count],
            "speed": speed_arr[:frame_count],
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


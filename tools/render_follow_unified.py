"""Unified ball-lock renderer.

This script consolidates the historical ``render_follow_*`` variants into a single
implementation that reproduces the behaviour of the "good tester clip" while
remaining configurable through presets and CLI overrides.

The module is intentionally self-contained so that the calibration and debug
helpers can import and reuse the building blocks (label loading, camera
planning, etc.).  The implementation is optimised for clarity and predictable
Windows behaviour rather than raw performance.
"""

from __future__ import annotations


import argparse
import os, sys

# Ensure repo root is importable even when running python tools\...`r
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


from tools.path_naming import build_output_name
from bisect import bisect_left
import glob
import hashlib
import json
import logging
import math
import re
import shutil
import subprocess

# --- REQUIRED IMPORTS RESTORED AFTER TRY/EXCEPT CLEANUP ---
from pathlib import Path

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Mapping, Optional, Sequence, TextIO, Tuple, Union

from math import hypot
from statistics import median

import cv2
import numpy as np

from tools.path_naming import build_output_name, normalize_tags_in_stem

logger = logging.getLogger(__name__)


def build_fallback_pan_plan(
    ball_samples: List[Dict[str, float]],
    *,
    src_w: int,
    crop_w: int,
    fps: float = 24.0,
    anticipation_s: float = 0.35,
    edge_margin: int = 40,
    max_speed_px_per_s: float = 900.0,
    smooth_alpha: float = 0.18,
) -> List[float]:
    """
    Returns a list of crop-center X positions in *source pixel* coords,
    one per frame, for the entire clip.

    This is used only when the segment-predict planner fails to build any
    segments, so we always produce *some* reasonable follow behavior.
    """

    if not ball_samples:
        # Last resort: static center
        center = src_w * 0.5
        return [center]

    # Extract time-ordered ball timeline
    ball_samples_sorted = sorted(ball_samples, key=lambda r: r["t"])
    ts = [r["t"] for r in ball_samples_sorted]
    xs = [r["x"] for r in ball_samples_sorted]

    # Clamp usable crop centers so that the portrait crop never leaves frame
    half_crop = crop_w * 0.5
    min_center = half_crop + edge_margin
    max_center = src_w - half_crop - edge_margin

    # Helper: clamp within field bounds
    def clamp_center(x: float) -> float:
        return max(min_center, min(max_center, x))

    # Estimate per-sample velocity (px/sec)
    vxs = [0.0] * len(xs)
    for i in range(1, len(xs)):
        dt = ts[i] - ts[i - 1]
        if dt <= 1e-6:
            vxs[i] = vxs[i - 1]
        else:
            vxs[i] = (xs[i] - xs[i - 1]) / dt

    # Simple median smoothing on position + velocity to remove spikes
    def median_filter(vals: List[float], window: int = 5) -> List[float]:
        n = len(vals)
        if n == 0 or window <= 1:
            return vals[:]
        half = window // 2
        out = []
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            window_vals = sorted(vals[lo:hi])
            out.append(window_vals[len(window_vals) // 2])
        return out

    xs_smooth = median_filter(xs, window=5)
    vxs_smooth = median_filter(vxs, window=5)

    # Convert the irregular telemetry times into per-frame targets.
    # We assume the first sample is near frame 0.
    duration_s = ts[-1] - ts[0]
    num_frames_est = max(1, int(math.ceil(duration_s * fps)) + 1)

    frame_centers: List[float] = []
    max_step_px = max_speed_px_per_s / fps

    # Start from smoothed ball X around first sample
    prev_center = clamp_center(xs_smooth[0])

    for f in range(num_frames_est):
        t = ts[0] + f / fps

        # Find nearest sample index for this time
        # (linear scan is fine for short clips; could binary-search if desired)
        j = min(range(len(ts)), key=lambda k: abs(ts[k] - t))

        x_ball = xs_smooth[j]
        vx_ball = vxs_smooth[j]

        # Forward anticipation: project ahead in time
        x_proj = x_ball + vx_ball * anticipation_s

        # Adaptive bias (Option C):
        # - if vx > 0 (moving right) -> keep ball slightly left of center
        # - if vx < 0 (moving left)  -> keep ball slightly right of center
        # We implement this by nudging the projected center in motion direction
        # but only a fraction of the remaining room.
        motion_sign = 1.0 if vx_ball > 0 else (-1.0 if vx_ball < 0 else 0.0)

        # How far could we move toward that side within bounds?
        if motion_sign > 0:
            room_right = max_center - x_proj
            bias = 0.35 * room_right
        elif motion_sign < 0:
            room_left = x_proj - min_center
            bias = -0.35 * room_left
        else:
            bias = 0.0

        target_center = clamp_center(x_proj + bias)

        # Limit pan speed so we don't whip across the field
        delta = target_center - prev_center
        if abs(delta) > max_step_px:
            delta = max_step_px if delta > 0 else -max_step_px
            target_center = prev_center + delta

        # Low-pass smoothing to keep motion buttery
        smoothed_center = prev_center + smooth_alpha * (target_center - prev_center)
        smoothed_center = clamp_center(smoothed_center)

        frame_centers.append(smoothed_center)
        prev_center = smoothed_center

    return frame_centers


def _gaussian1d(values: Sequence[float], sigma: float, mode: str = "nearest") -> np.ndarray:
    """Lightweight 1D Gaussian smooth without requiring SciPy.

    Parameters
    ----------
    values : Sequence[float]
        Input samples to smooth.
    sigma : float
        Standard deviation of the Gaussian kernel.
    mode : str
        Border handling; only "nearest" is supported.
    """

    arr = np.asarray(values, dtype=float)
    if sigma <= 0 or arr.size == 0:
        return arr.astype(float)

    # 3-sigma kernel radius for a good approximation.
    radius = max(1, int(round(float(sigma) * 3)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / float(sigma)) ** 2)
    kernel_sum = kernel.sum()
    if not np.isfinite(kernel_sum) or kernel_sum == 0:
        return arr.astype(float)
    kernel /= kernel_sum

    pad_mode = "edge" if mode == "nearest" else mode
    padded = np.pad(arr, radius, mode=pad_mode)
    smoothed = np.convolve(padded, kernel, mode="same")
    return smoothed[radius:-radius]


def compute_predictive_follow_centers(
    ball_x: "np.ndarray",
    t: "np.ndarray",
    src_width: int,
    crop_width: int,
    fps: float,
    *,
    window_frac: float = 0.25,
    horizon_s: float = 0.35,
    alpha: float = 0.90,
    max_cam_speed_px_per_s: float = 850.0,
) -> "np.ndarray":
    """
    Compute smooth camera centers for a horizontal follow that:
    - Tracks the *path* of the ball, not its exact position.
    - Lets the ball drift inside a central window of the crop.
    - Pans only when the ball exits that window.
    - Limits camera speed so motion is smooth and never jerky.

    Parameters
    ----------
    ball_x : np.ndarray
        Ball x positions in source pixels, one per frame.
    t : np.ndarray
        Timestamps in seconds, same length as ball_x.
    src_width : int
        Width of the source video in pixels.
    crop_width : int
        Width of the portrait crop (the vertical slice) in pixels.
    fps : float
        Output frame rate in frames per second.

    Tunables (keyword-only)
    -----------------------
    window_frac : float
        Half-width of the safe window as a fraction of crop width.
        E.g. 0.25 => ball can drift ±25% of crop width around center
        before we pan.
    horizon_s : float
        Look-ahead horizon for prediction, in seconds.
    alpha : float
        EMA smoothing factor over the predicted path (0.9 = heavy smoothing).
    max_cam_speed_px_per_s : float
        Max horizontal speed of the camera center, in pixels/sec.
    """
    import numpy as np

    if len(ball_x) == 0:
        return np.zeros(0, dtype=float)

    ball_x = np.asarray(ball_x, dtype=float)
    t = np.asarray(t, dtype=float)

    # Basic temporal spacing; fall back to 1/fps if timestamps are weird.
    if len(t) > 1:
        dt = float(np.median(np.diff(t)))
        if not np.isfinite(dt) or dt <= 0:
            dt = 1.0 / float(fps or 24.0)
    else:
        dt = 1.0 / float(fps or 24.0)

    # --- 2.1: estimate velocity and accel along x -------------------------
    # Use central differences for interior points; forward/backward at edges.
    v = np.zeros_like(ball_x)
    a = np.zeros_like(ball_x)

    if len(ball_x) >= 2:
        v[1:-1] = (ball_x[2:] - ball_x[:-2]) / (t[2:] - t[:-2])
        v[0] = (ball_x[1] - ball_x[0]) / max(t[1] - t[0], dt)
        v[-1] = (ball_x[-1] - ball_x[-2]) / max(t[-1] - t[-2], dt)

    if len(ball_x) >= 3:
        a[1:-1] = (v[2:] - v[:-2]) / (t[2:] - t[:-2])
        a[0] = 0.0
        a[-1] = 0.0

    # --- 2.2: predict a little into the future ----------------------------
    H = float(horizon_s)
    predicted = ball_x + v * H + 0.5 * a * H * H

    # --- 2.3: Bidirectional EMA over the predicted path -------------------
    # Forward + backward EMA averaged together so the end of the clip is as
    # smooth as the start.  A forward-only EMA lags at the end, causing the
    # camera to chase ball jumps reactively during goal celebrations etc.
    fwd = np.empty_like(predicted)
    fwd[0] = predicted[0]
    for i in range(1, len(predicted)):
        fwd[i] = alpha * fwd[i - 1] + (1.0 - alpha) * predicted[i]

    bwd = np.empty_like(predicted)
    bwd[-1] = predicted[-1]
    for i in range(len(predicted) - 2, -1, -1):
        bwd[i] = alpha * bwd[i + 1] + (1.0 - alpha) * predicted[i]

    smooth = 0.5 * (fwd + bwd)

    # --- 2.4: convert to camera centers with window + max speed ----------
    centers = np.empty_like(smooth)
    # Start centered on first smoothed point, but clamped to valid crop range.
    half_crop = crop_width / 2.0
    min_center = half_crop
    max_center = src_width - half_crop

    def clamp_center(x):
        return float(min(max(x, min_center), max_center))

    centers[0] = clamp_center(smooth[0])

    safe_half_window = window_frac * crop_width
    # Max camera step per frame in pixels.
    max_step = max_cam_speed_px_per_s * dt

    for i in range(1, len(smooth)):
        cam_prev = centers[i - 1]
        bx = ball_x[i]

        # If the ball is within the safe window of the previous camera center,
        # keep the camera where it is. This avoids jittery micro-motions.
        if abs(bx - cam_prev) <= safe_half_window:
            target = cam_prev  # no pan; optionally nudge slightly toward smooth[i]
        else:
            # Ball exiting the window => pan toward the *smoothed, predicted* path.
            target = smooth[i]

        # Move toward target with a max step to enforce smoothness.
        delta = target - cam_prev
        if abs(delta) > max_step:
            delta = max_step if delta > 0 else -max_step

        centers[i] = clamp_center(cam_prev + delta)

    return centers


def load_any_telemetry(path):
    path = str(path)
    """
    Unified loader for ball telemetry (JSONL) and follow telemetry
    (JSON or JSONL). Returns a list of dicts with keys:
        t, cx, cy, zoom, valid
    No other fields required.
    """

    import json, os

    rows = []

    # JSON follow telemetry (single JSON containing "keyframes")
    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            root = json.load(f)
        kfs = root.get("keyframes", [])
        for kf in kfs:
            rows.append({
                "t": kf.get("t"),
                "cx": kf.get("cx"),
                "cy": kf.get("cy"),
                "zoom": kf.get("zoom", 1.0),
                "valid": True,
            })
        return rows

    # JSONL (either ball telemetry or flattened follow telemetry)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except Exception:
                continue

            rows.append(row)

    if not rows:
        return rows

    # Inspect first row to infer schema
    first = rows[0]

    if isinstance(first, dict) and "cx" in first and "cy" in first:
        # Already in camera-plan / ball-plan format
        return rows

    # Otherwise assume raw telemetry rows
    return rows


def _safe_float(v):
    try:
        f = float(v)
    except Exception:
        return None
    if not math.isfinite(f):
        return None
    return f


def safe_float(val, default):
    try:
        if val is None:
            return default
        x = float(val)
        if not math.isfinite(x):
            return default
        return x
    except Exception:
        return default


def _interpolate_series(series: list[Optional[float]], default_value: float) -> list[float]:
    if not series:
        return []

    filled = list(series)
    valid_indices = [i for i, v in enumerate(filled) if v is not None and math.isfinite(v)]
    if not valid_indices:
        return [default_value for _ in filled]

    first_idx, last_idx = valid_indices[0], valid_indices[-1]
    for idx in range(0, first_idx):
        filled[idx] = filled[first_idx]
    for idx in range(last_idx + 1, len(filled)):
        filled[idx] = filled[last_idx]

    for start, end in zip(valid_indices, valid_indices[1:]):
        start_val = float(filled[start])
        end_val = float(filled[end])
        gap = end - start
        if gap > 1:
            step = (end_val - start_val) / gap
            for offset in range(1, gap):
                filled[start + offset] = start_val + step * offset

    return [default_value if v is None else float(v) for v in filled]


def _smooth_track(xs: list[float], window: int = 9) -> list[float]:
    """
    Simple centered moving-average smoother for camera centers.
    Balanced preset:
      - default window=9 frames (~0.37s @ 24fps)
      - handles short clips by shrinking the window
    """
    if not xs:
        return xs
    n = len(xs)
    # Ensure an odd window and cap by length
    w = max(3, min(window, n))
    if w % 2 == 0:
        w += 1
    half = w // 2

    out: list[float] = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        segment = xs[lo:hi]
        out.append(sum(segment) / len(segment))
    return out


def _clamp_velocity(xs: list[float], max_delta: float = 45.0) -> list[float]:
    """
    Limit frame-to-frame movement of the camera center.
    max_delta is in pixels per frame (Balanced preset).
    """
    if not xs:
        return xs
    out = [float(xs[0])]
    for x in xs[1:]:
        prev = out[-1]
        delta = float(x) - prev
        if delta > max_delta:
            delta = max_delta
        elif delta < -max_delta:
            delta = -max_delta
        out.append(prev + delta)
    return out


def smooth_pan_track(xs: list[float]) -> list[float]:
    """
    Balanced smoothing preset for camera centers:
      1) centered moving average
      2) velocity clamp
    """
    if not xs:
        return xs
    xs = _smooth_track(xs, window=9)      # Balanced smoothness
    xs = _clamp_velocity(xs, 45.0)        # Balanced responsiveness
    return xs


def _normalize_follow_override_map(
    override_map: Mapping[int, Mapping[str, float]], total_frames: int, fps_in: float
) -> Mapping[int, Mapping[str, float]]:
    if not override_map:
        return override_map

    target_len = max(total_frames, max(override_map.keys()) + 1)
    cx_series: list[Optional[float]] = [None] * target_len
    cy_series: list[Optional[float]] = [None] * target_len
    zoom_series: list[Optional[float]] = [None] * target_len
    t_series: list[Optional[float]] = [None] * target_len

    for frame_idx, data in override_map.items():
        if frame_idx < 0 or frame_idx >= target_len:
            continue
        cx_series[frame_idx] = safe_float(data.get("cx"), None)
        cy_series[frame_idx] = safe_float(data.get("cy"), None)
        zoom_series[frame_idx] = safe_float(data.get("zoom", 1.0), 1.0)
        t_series[frame_idx] = safe_float(data.get("t"), None)

    cx_interp = _interpolate_series(cx_series, 0.0)
    cy_interp = _interpolate_series(cy_series, 0.0)
    zoom_interp = _interpolate_series(zoom_series, 1.0)

    for idx in range(target_len):
        if t_series[idx] is None:
            t_series[idx] = (idx / fps_in) if fps_in > 0 else 0.0

    normalized: dict[int, Mapping[str, float]] = {}
    for frame_idx in range(target_len):
        normalized[frame_idx] = {
            "t": float(t_series[frame_idx]) if t_series[frame_idx] is not None else 0.0,
            "cx": float(cx_interp[frame_idx]),
            "cy": float(cy_interp[frame_idx]),
            "zoom": float(zoom_interp[frame_idx]),
        }

    return normalized


def _interp_at_time(samples, t):
    """
    Linear interpolate x at time t from a list of (t, x) pairs.
    If t is outside the range, clamp to the nearest endpoint.
    samples must be non-empty and sorted by t.
    """
    if not samples:
        return None
    if t <= samples[0][0]:
        return samples[0][1]
    if t >= samples[-1][0]:
        return samples[-1][1]
    for i in range(1, len(samples)):
        t0, x0 = samples[i - 1]
        t1, x1 = samples[i]
        if t0 <= t <= t1:
            if t1 == t0:
                return x0
            alpha = (t - t0) / (t1 - t0)
            return x0 + alpha * (x1 - x0)
    return samples[-1][1]


def _ball_xy_at_time(
    samples_x: Sequence[tuple[float, float]],
    samples_y: Sequence[tuple[float, float]],
    t: float,
    times: Optional[Sequence[float]] = None,
) -> tuple[Optional[float], Optional[float]]:
    if not samples_x or not samples_y:
        return None, None
    if times is None:
        times = [row[0] for row in samples_x]
    if not times:
        return None, None
    last_idx = min(len(samples_x), len(samples_y)) - 1
    if last_idx < 0:
        return None, None
    idx = bisect_left(times, t)
    if idx <= 0:
        nearest_idx = 0
    elif idx >= len(times):
        nearest_idx = last_idx
    else:
        before = times[idx - 1]
        after = times[idx]
        nearest_idx = idx if abs(after - t) < abs(t - before) else idx - 1
    nearest_idx = min(nearest_idx, last_idx)
    return samples_x[nearest_idx][1], samples_y[nearest_idx][1]


def _ball_tx_pairs(ball_samples) -> list[tuple[float, float]]:
    samples_tx: list[tuple[float, float]] = []
    for s in ball_samples or []:
        t_val = None
        x_val = None
        if isinstance(s, dict):
            t_val = s.get("t") or s.get("time") or s.get("ts")
            x_val = s.get("x") or s.get("cx") or s.get("ball_x")
        else:
            if hasattr(s, "t") or hasattr(s, "x"):
                t_val = getattr(s, "t", None)
                x_val = getattr(s, "x", None)
            elif hasattr(s, "time") or hasattr(s, "cx"):
                t_val = getattr(s, "time", None)
                x_val = getattr(s, "cx", None)
            else:
                seq = tuple(s) if hasattr(s, "__len__") else None
                if seq is not None and len(seq) >= 2:
                    t_val, x_val = seq[0], seq[1]
        if t_val is None or x_val is None:
            continue
        try:
            samples_tx.append((float(t_val), float(x_val)))
        except Exception:
            continue

    samples_tx.sort(key=lambda p: p[0])
    return samples_tx


def build_segment_predict_center_path(
    ball_samples,
    fps,
    src_w,
    portrait_w,
    duration_s,
    window_s=0.6,
    lead_s=0.3,
    tol_frac=0.35,
    smooth_alpha=0.88,
):
    """
    Return a list center_x_per_frame of length N_frames, where each entry is
    the desired center X (in source pixel coords) for that frame, using a
    segment-level prediction of the ball path.

    - ball_samples: list of dicts or tuples with fields/time keys:
        * 't' or index 0 => time in seconds
        * 'x' or index 1 => x in source pixel coords
    - fps: frames per second
    - src_w: width of the source video
    - portrait_w: width of the portrait crop
    - duration_s: total duration of the clip in seconds
    """
    if fps <= 0 or src_w <= 0 or portrait_w <= 0 or duration_s <= 0:
        return []

    samples_tx = _ball_tx_pairs(ball_samples)
    samples_tx.sort(key=lambda p: p[0])

    def build_plan_segment_predict(ball_xy, fps, crop_w, src_width):
        # 1. Soft filtering: keep almost everything, only drop impossible values
        filtered = [(t, x) for (t, x) in ball_xy if 0 <= x <= src_width]

        # If extremely few usable samples, fall back to raw ball_xy
        if len(filtered) < 3:
            filtered = list(ball_xy)

        if not filtered:
            return [], []

        # 2. Base pan = ball-centered (move ball slightly right inside portrait)
        target_bias = crop_w * 0.25  # ball 25% from left
        pan = [(x - target_bias) for (_, x) in filtered]

        # 3. Soft clamp pan into valid region (no more rejection)
        clamped = [max(0.0, min(p, src_width - crop_w)) for p in pan]

        # 4. Smooth with adjustable parameters
        smoothed = _gaussian1d(clamped, sigma=6.0, mode="nearest")

        # 5. Predict forward 0.20s
        dt = 1.0 / float(fps) if fps > 0 else 0.0
        lead_frames = max(1, int(0.20 / dt)) if dt > 0 else 1
        lead = np.roll(smoothed, -lead_frames)

        # 6. Additional light smooth
        out = _gaussian1d(lead, sigma=3.0, mode="nearest")

        # Always return a plan — even if imperfect
        times = [t for (t, _) in filtered]
        return times, out.tolist()

    n_frames = int(round(duration_s * fps))
    frame_times = [i / float(fps) for i in range(n_frames)]

    times, pan_plan = build_plan_segment_predict(samples_tx, fps, portrait_w, src_w)

    if not pan_plan:
        # Fallback: steady center plan keeping crop fully inside frame
        default_center = float(src_w) / 2.0
        return [default_center] * n_frames

    # Interpolate plan to per-frame centers
    # Ensure strictly increasing times for interpolation
    uniq_times = []
    uniq_pans = []
    for t, p in zip(times, pan_plan):
        if uniq_times and t <= uniq_times[-1]:
            continue
        uniq_times.append(t)
        uniq_pans.append(p)

    if len(uniq_times) == 1:
        uniq_times = [0.0, max(frame_times[-1], 0.0)]
        uniq_pans = [uniq_pans[0], uniq_pans[0]]

    pan_interp = np.interp(frame_times, uniq_times, uniq_pans)
    center_x = (pan_interp + (portrait_w * 0.5)).tolist()
    return [max(portrait_w * 0.5, min(src_w - portrait_w * 0.5, cx)) for cx in center_x]


def _resample_path(values: Sequence[float], target_len: int) -> np.ndarray:
    if target_len <= 0:
        return np.zeros(0, dtype=float)
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.zeros(target_len, dtype=float)
    if arr.size == target_len:
        return arr.astype(float)
    x_src = np.linspace(0.0, 1.0, num=arr.size)
    x_dst = np.linspace(0.0, 1.0, num=target_len)
    return np.interp(x_dst, x_src, arr).astype(float)


def build_option_c_follow_centers(
    *,
    ball_samples: Sequence[Mapping[str, float]] | Sequence[BallSample] | None,
    fps: float,
    src_w: int,
    portrait_w: int,
    frame_count: int,
    anticipation_s: float = 0.25,
    max_speed_px_per_s: float = 900.0,
    debug_pan_overlay: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-frame horizontal centers for follow option C.

    Returns the pan centers along with per-frame crop width scale factors and
    recovery flags. The plan first attempts the segment-predict planner and
    then falls back to telemetry-driven smoothing with forward anticipation.
    """

    fps = float(fps if fps and fps > 0 else 24.0)
    portrait_w = int(max(1, portrait_w))
    half_crop = portrait_w * 0.5
    min_center = half_crop
    max_center = float(src_w) - half_crop
    mid_center = (min_center + max_center) * 0.5
    max_pan_x = max_center - mid_center

    def _clamp_center(x: float) -> float:
        return max(min_center, min(max_center, x))

    duration_s = frame_count / fps if fps > 0 else 0.0
    centers_segment = build_segment_predict_center_path(
        ball_samples=ball_samples or [],
        fps=fps,
        src_w=float(src_w),
        portrait_w=float(portrait_w),
        duration_s=float(duration_s),
    )

    centers_segment_arr = np.asarray(centers_segment, dtype=float)
    centers_segment_arr = _resample_path(centers_segment_arr, frame_count)

    samples_tx = _ball_tx_pairs(ball_samples or [])
    if samples_tx:
        telem_times = np.array([t for (t, _) in samples_tx], dtype=float)
        telem_x = np.array([x for (_, x) in samples_tx], dtype=float)
    else:
        telem_times = np.array([0.0, max(duration_s, 1.0 / max(fps, 1.0))], dtype=float)
        telem_x = np.array([float(src_w) / 2.0, float(src_w) / 2.0], dtype=float)

    frame_times = np.arange(frame_count, dtype=float) / fps if fps > 0 else np.zeros(frame_count)
    ball_x = np.interp(frame_times, telem_times, telem_x)
    vx = np.gradient(ball_x) * fps if fps > 0 else np.zeros_like(ball_x)
    max_anticipation_px = 0.08 * float(portrait_w)
    max_delta = max_speed_px_per_s / float(max(fps, 1.0))
    centers_option_c = np.zeros(frame_count, dtype=float)
    crop_scales = np.ones(frame_count, dtype=float)
    recovery_flags = np.zeros(frame_count, dtype=float)
    prev_center = _clamp_center(centers_segment_arr[0] if centers_segment_arr.size else ball_x[0])
    prev_delta = 0.0
    centers_option_c[0] = prev_center
    base_smoothing_alpha = 0.18
    reversal_threshold_px = 0.04 * float(portrait_w)
    saturation_margin = 0.85 * max_pan_x
    recovery_clear_margin = 0.65 * max_pan_x
    recovery_release_frames_required = 6
    recovery_release_counter = 0
    recovery_active = False
    current_scale = 1.0
    scale_alpha = 0.22  # ~4-6 frame lerp

    for idx in range(1, frame_count):
        ball_pos = ball_x[idx]
        vx_val = vx[idx]
        lookahead_s = float(anticipation_s)
        if recovery_active:
            lookahead_s += 0.15
        max_anticipation_eff = max_anticipation_px * min(1.6, lookahead_s / max(float(anticipation_s), 1e-6))
        anticipation_dx = np.clip(vx_val * lookahead_s, -max_anticipation_eff, max_anticipation_eff)
        predicted = _clamp_center(ball_pos + anticipation_dx)

        base = centers_segment_arr[idx] if idx < centers_segment_arr.size else prev_center
        desired = 0.6 * predicted + 0.4 * _clamp_center(base)

        predicted_pan = desired - mid_center
        if not recovery_active and abs(predicted_pan) > saturation_margin:
            recovery_active = True
            recovery_release_counter = 0
            if debug_pan_overlay:
                logger.info("[option_c] recovery start at frame %d", idx)
        elif recovery_active:
            if abs(predicted_pan) < recovery_clear_margin:
                recovery_release_counter += 1
                if recovery_release_counter >= recovery_release_frames_required:
                    recovery_active = False
                    recovery_release_counter = 0
                    if debug_pan_overlay:
                        logger.info("[option_c] recovery end at frame %d", idx)
            else:
                recovery_release_counter = 0

        if recovery_active:
            overshoot = max(0.0, abs(predicted_pan) - saturation_margin)
            overshoot_span = max(max_pan_x - saturation_margin, 1e-6)
            risk_frac = min(1.0, overshoot / overshoot_span)
            target_scale = 1.0 + 0.18 * risk_frac
        else:
            target_scale = 1.0
        current_scale += scale_alpha * (target_scale - current_scale)
        current_scale = min(1.18, max(1.0, current_scale))
        crop_scales[idx] = current_scale
        recovery_flags[idx] = 1.0 if recovery_active else 0.0

        follow_gain = 0.6 if recovery_active else 1.0
        new_delta = (desired - prev_center) * follow_gain
        if math.copysign(1.0, new_delta) != math.copysign(1.0, prev_delta) and abs(new_delta) < reversal_threshold_px:
            new_delta = prev_delta * 0.7
        if abs(new_delta) > max_delta:
            new_delta = math.copysign(max_delta, new_delta)
        target_center = _clamp_center(prev_center + new_delta)
        smoothing_alpha = base_smoothing_alpha * (0.75 if recovery_active else 1.0)
        smoothing_alpha = max(0.05, smoothing_alpha)
        smoothed_center = prev_center + smoothing_alpha * (target_center - prev_center)
        smoothed_center = _clamp_center(smoothed_center)
        prev_delta = smoothed_center - prev_center
        prev_center = smoothed_center
        centers_option_c[idx] = prev_center

    return centers_option_c, crop_scales, recovery_flags


def _ball_overlay_samples(
    ball_samples: Sequence[Mapping[str, float]] | Sequence[BallSample],
    fps: float,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    if not ball_samples:
        return [], []

    triples: list[tuple[float, float, float]] = []
    for sample in ball_samples:
        t_val = None
        x_val = None
        y_val = None
        if isinstance(sample, Mapping):
            t_val = sample.get("t") or sample.get("time") or sample.get("ts")
            x_val = (
                sample.get("x")
                or sample.get("cx")
                or sample.get("ball_x")
                or sample.get("bx")
                or sample.get("bx_stab")
            )
            y_val = (
                sample.get("y")
                or sample.get("cy")
                or sample.get("ball_y")
                or sample.get("by")
                or sample.get("by_stab")
            )
            if t_val is None:
                t_val = sample.get("frame")
                if t_val is not None and fps > 0:
                    t_val = float(t_val) / float(fps)
        else:
            t_val = getattr(sample, "t", None) or getattr(sample, "time", None) or getattr(sample, "ts", None)
            x_val = (
                getattr(sample, "x", None)
                or getattr(sample, "cx", None)
                or getattr(sample, "ball_x", None)
                or getattr(sample, "bx", None)
                or getattr(sample, "bx_stab", None)
            )
            y_val = (
                getattr(sample, "y", None)
                or getattr(sample, "cy", None)
                or getattr(sample, "ball_y", None)
                or getattr(sample, "by", None)
                or getattr(sample, "by_stab", None)
            )
            if t_val is None:
                frame_val = getattr(sample, "frame", None)
                if frame_val is not None and fps > 0:
                    t_val = float(frame_val) / float(fps)

        t_float = _safe_float(t_val)
        x_float = _safe_float(x_val)
        y_float = _safe_float(y_val)
        if t_float is None or x_float is None or y_float is None:
            continue
        triples.append((t_float, x_float, y_float))

    if not triples:
        return [], []

    triples.sort(key=lambda row: row[0])
    samples_x = [(t, x) for (t, x, _) in triples]
    samples_y = [(t, y) for (t, _, y) in triples]
    return samples_x, samples_y


def render_segment_smooth_follow(
    in_path: str,
    out_path: str,
    portrait: str,
    fps: float,
    duration: float,
    num_frames: int,
    ball_samples: list,
    draw_ball: bool,
    *,
    keep_scratch: bool = False,
    scratch_root: Path | None = None,
):
    """
    Segment-smooth follow:
      - produces smooth camera motion for the entire clip
      - uses ball_samples to drive the follow path
      - optionally overlays a red dot at ball position on every frame
    """
    input_path = Path(in_path).expanduser().resolve()
    output_path = Path(out_path).expanduser().resolve()
    portrait_dims = parse_portrait(portrait)
    if not portrait_dims:
        raise ValueError(f"Invalid portrait size: {portrait}")
    portrait_w, portrait_h = portrait_dims

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {input_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = float(fps if fps and fps > 0 else ffprobe_fps(input_path))
    frame_count = int(num_frames if num_frames and num_frames > 0 else cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = max(frame_count, 1)

    target_aspect = float(portrait_w) / float(portrait_h)
    _, _, crop_w, crop_h = compute_portrait_crop(
        cx=float(src_w) / 2.0,
        cy=float(src_h) / 2.0,
        zoom=1.0,
        src_w=src_w,
        src_h=src_h,
        target_aspect=target_aspect,
        pad=0.0,
    )
    crop_w = int(round(crop_w))
    crop_h = int(round(crop_h))

    centers, _, _ = build_option_c_follow_centers(
        ball_samples=ball_samples,
        fps=float(fps_in),
        src_w=src_w,
        portrait_w=crop_w,
        frame_count=frame_count,
    )
    centers = _resample_path(centers, frame_count)

    overlay_samples_x, overlay_samples_y = ([], [])
    overlay_times: list[float] = []
    if draw_ball and ball_samples:
        overlay_samples_x, overlay_samples_y = _ball_overlay_samples(ball_samples, fps_in)
        overlay_times = [row[0] for row in overlay_samples_x]

    temp_root = (scratch_root or _scratch_root()) / "autoframe_work"
    temp_dir = temp_root / "segment_smooth" / input_path.stem
    _prepare_temp_dir(temp_dir, clean=True)
    frames_dir = temp_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx in range(frame_count):
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        if draw_ball and ball_samples:
            t_val = frame_idx / float(fps_in) if fps_in > 0 else 0.0
            bx, by = _ball_xy_at_time(overlay_samples_x, overlay_samples_y, t_val, overlay_times)
            if bx is not None and by is not None:
                cv2.circle(frame, (int(round(bx)), int(round(by))), 8, (0, 0, 255), -1)

        center_x = float(centers[frame_idx]) if frame_idx < len(centers) else float(src_w) / 2.0
        x0, y0, crop_w, crop_h = compute_portrait_crop(
            cx=center_x,
            cy=float(src_h) / 2.0,
            zoom=1.0,
            src_w=src_w,
            src_h=src_h,
            target_aspect=target_aspect,
            pad=0.0,
        )
        x0 = int(round(x0))
        y0 = int(round(y0))
        crop_w = int(round(crop_w))
        crop_h = int(round(crop_h))
        x1 = max(0, min(src_w, x0 + crop_w))
        y1 = max(0, min(src_h, y0 + crop_h))
        cropped = frame[y0:y1, x0:x1]

        resized = cv2.resize(cropped, (portrait_w, portrait_h), interpolation=cv2.INTER_AREA)
        out_frame_path = frames_dir / f"frame_{frame_idx:06d}.png"
        cv2.imwrite(str(out_frame_path), resized)

    cap.release()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output_path = _temp_output_path(output_path)
    if temp_output_path.exists():
        temp_output_path.unlink(missing_ok=True)
    keyint = max(1, int(round(float(fps_in) * 4)))
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        str(fps_in),
        "-i",
        str(frames_dir / "frame_%06d.png"),
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "19",
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "high",
        "-level",
        "4.0",
        "-x264-params",
        f"keyint={keyint}:min-keyint={keyint}:scenecut=0",
        "-movflags",
        "+faststart",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(temp_output_path),
    ]
    subprocess.run(ffmpeg_cmd, check=True)
    temp_output_path.replace(output_path)
    print("[DONE] Video stitched successfully.")
    if not keep_scratch and temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
    return 0



def build_exact_follow_center_path(
    ball_samples,
    fps,
    src_w,
    portrait_w,
    duration_s,
):
    """Return raw, clamped camera centers directly from ball telemetry."""

    if fps <= 0 or src_w <= 0 or portrait_w <= 0 or duration_s <= 0:
        return []

    samples_tx = _ball_tx_pairs(ball_samples)
    if not samples_tx:
        return []

    n_frames = int(round(duration_s * fps))
    half_p = portrait_w * 0.5

    def clamp_center(xc):
        return max(half_p, min(src_w - half_p, xc))

    centers: list[float] = []
    for i in range(n_frames):
        t_frame = i / float(fps)
        cx_val = _interp_at_time(samples_tx, t_frame)
        if cx_val is None:
            cx_val = centers[-1] if centers else samples_tx[0][1]
        centers.append(clamp_center(float(cx_val)))

    return centers


def _load_ball_telemetry(path):
    from tools.render_follow_unified import load_any_telemetry

    telemetry_rows = load_any_telemetry(path)
    if not telemetry_rows:
        raise ValueError(f"No usable telemetry in {path}")

    out = []
    for row in telemetry_rows:
        cx = _safe_float(row.get("cx"))
        cy = _safe_float(row.get("cy"))
        if cx is None or cy is None:
            continue
        rec = dict(row)
        rec["cx"] = cx
        rec["cy"] = cy
        out.append(rec)

    return out

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Ball-follow tuning constants. These are intentionally easy to tweak so we can
# dial in a responsive-but-human feel without digging through the rendering
# code. Increase BALL_FOLLOW_ALPHA for snappier response (more jitter), or
# decrease it for smoother, lazier motion. Tighten BALL_MAX_DX/BALL_MAX_DY to
# further clamp per-frame motion when the ball makes sharp cuts.
BALL_FOLLOW_ALPHA = 0.25
BALL_MAX_DX = 60.0
BALL_MAX_DY = 40.0

# Ball-follow camera path tuning (portrait reels)
LEAD_WINDOW_FACTOR = 0.35  # ~0.35s lookahead
BALL_PATH_SMOOTHING_RADIUS = 5
CAM_SMOOTH_ALPHA_X = 0.22
CAM_SMOOTH_ALPHA_Y = 0.08
CAM_SMOOTH_ALPHA_ZOOM = 0.08
MAX_CAM_DX_PER_FRAME_FRAC = 0.03  # relative to frame width
MAX_CAM_DY_PER_FRAME_FRAC = 0.012
TELEPORT_THRESHOLD_FRAC = 0.14  # treat sudden jumps as long passes
EASE_FRAMES_FOR_TELEPORT = 10
BALL_CENTER_TOLERANCE_PCT = 0.20
ZOOM_MIN = 1.0
ZOOM_MAX = 1.4

BALL_CAM_CONFIG: dict[str, object] = {
    "min_coverage": 0.4,
    "min_confidence": 0.25,
    "lead_frames": 3,
    "base_alpha": 0.20,
    "fast_alpha": 0.60,
    "catchup_thresh_px": 80.0,
    "ball_margin_px": 80.0,
    "final_smooth_alpha": 0.1,
    "final_smooth_passes": 2,
    "max_pan_per_frame": 40.0,
    "max_accel_per_frame": 20.0,
    "zoom": {
        "base_crop_width": 1080,
        "min_crop_width": 800,
        "max_crop_width": 1080,
        "zoom_alpha": 0.2,
        "max_zoom_delta": 30.0,
    },
}

from tools.ball_telemetry import (
    BallSample,
    ExcludeZone,
    PersonBox,
    fuse_yolo_and_centroid,
    load_and_interpolate_telemetry,
    load_ball_telemetry,
    load_ball_telemetry_for_clip,
    load_exclude_zones,
    run_yolo_ball_detection,
    run_yolo_person_detection,
    set_telemetry_frame_bounds,
    telemetry_path_for_video,
)
from tools.offline_portrait_planner import (
    OfflinePortraitPlanner,
    PlannerConfig,
    analyze_ball_positions,
    keyframes_to_arrays,
    load_plan,
    plan_ball_portrait_crop,
)
from tools.upscale import upscale_video

# Confidence thresholds for telemetry-driven selection
BALL_CONF_THRESH = 0.5
PLAYER_CONF_THRESH = 0.5


def load_ball_path_from_jsonl(
    path: str,
    logger=None,
    *,
    use_red_fallback: bool = False,
    use_ball_telemetry: bool = True,
):
    """
    Return (ball_x, ball_y, stats) from a telemetry jsonl file.

    Prefers explicit ``ball_x/ball_y`` pairs when available, then falls back to
    legacy ``ball_src`` or ``ball`` tuples.
    """

    from tools.render_follow_unified import load_any_telemetry

    telemetry_rows = load_any_telemetry(path)
    if not telemetry_rows:
        raise ValueError(f"No usable telemetry in {path}")

    xs: list[float] = []
    ys: list[float] = []
    confs: list[float] = []
    total_rows = len(telemetry_rows)
    kept_rows = 0
    high_conf_valid_frames = 0

    for row in telemetry_rows:
        fallback_xy_raw = row.get("fallback_ball_xy")
        fallback_xy: tuple[float, float] | None = None
        if isinstance(fallback_xy_raw, Sequence) and len(fallback_xy_raw) >= 2:
            fx = _safe_float(fallback_xy_raw[0])
            fy = _safe_float(fallback_xy_raw[1])
            if fx is not None and fy is not None:
                fallback_xy = (float(fx), float(fy))

        motion_centroid_raw = row.get("motion_centroid")
        motion_centroid_xy: tuple[float, float] | None = None
        if isinstance(motion_centroid_raw, Sequence) and len(motion_centroid_raw) >= 2:
            mx = _safe_float(motion_centroid_raw[0])
            my = _safe_float(motion_centroid_raw[1])
            if mx is not None and my is not None:
                motion_centroid_xy = (float(mx), float(my))

        cx = _safe_float(row.get("cx"))
        cy = _safe_float(row.get("cy"))

        ball_xy = None
        if use_red_fallback and fallback_xy is not None:
            ball_xy = fallback_xy
        elif use_ball_telemetry and cx is not None and cy is not None:
            ball_xy = (float(cx), float(cy))
        elif motion_centroid_xy is not None:
            ball_xy = motion_centroid_xy

        if ball_xy is None:
            continue

        xs.append(float(ball_xy[0]))
        ys.append(float(ball_xy[1]))
        kept_rows += 1

        conf_val = _safe_float(row.get("conf"))
        if conf_val is not None:
            confs.append(conf_val)
        if bool(row.get("valid", True)) and (conf_val is None or conf_val >= BALL_CONF_THRESH):
            high_conf_valid_frames += 1

    xs_arr = np.asarray(xs, dtype=np.float32)
    ys_arr = np.asarray(ys, dtype=np.float32)
    total_frames = total_rows
    conf_threshold = BALL_CONF_THRESH

    if xs_arr.size == 0 or ys_arr.size == 0:
        telemetry_meta = {
            "kept": 0,
            "total": int(total_frames) if "total_frames" in locals() else None,
            "conf": float(conf_threshold) if "conf_threshold" in locals() else None,
            "reason": "no_valid_ball_samples_after_filter",
        }
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32), telemetry_meta

    meta = {
        "total_rows": total_rows,
        "kept_rows": kept_rows,
        "avg_conf": float(np.mean(confs)) if confs else 1.0,
        "telemetry_quality": float(high_conf_valid_frames) / float(total_rows) if total_rows else 0.0,
    }

    if logger:
        logger.info(
            "[BALL-TELEMETRY] ball_src_x=[%.1f, %.1f], ball_src_y=[%.1f, %.1f], kept=%d/%d, conf=%.2f",
            float(xs_arr.min()),
            float(xs_arr.max()),
            float(ys_arr.min()),
            float(ys_arr.max()),
            kept_rows,
            total_rows,
            meta["avg_conf"],
        )

    return xs_arr, ys_arr, meta


def emit_follow_telemetry(
    path: str | os.PathLike[str] | None,
    cx: Sequence[float],
    cy: Sequence[float],
    zoom: Sequence[float],
    *,
    workdir: str | os.PathLike[str] | None = None,
    basename: str | None = None,
) -> str:
    """Write a follow-telemetry JSONL file from camera centers.

    The output filename is derived from ``basename`` (or the telemetry stem) and
    is always placed under ``workdir`` (or the telemetry directory).
    """

    stem = basename or (Path(path).stem if path else "follow")
    root = Path(workdir) if workdir is not None else (Path(path).parent if path else Path.cwd())
    root.mkdir(parents=True, exist_ok=True)

    cx = [0.0 if v is None else float(v) for v in cx]
    cy = [0.0 if v is None else float(v) for v in cy]
    zoom = [1.0 if v is None else float(v) for v in zoom]

    out_path = root / f"{stem}.follow.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for i in range(len(cx)):
            cx_i = float(cx[i] or 0)
            cy_i = float(cy[i] or 0)
            zoom_i = float(zoom[i] or 1.0)

            f.write(
                json.dumps(
                    {
                        "f": i,
                        "cx": cx_i,
                        "cy": cy_i,
                        "zoom": zoom_i,
                    }
                )
                + "\n"
            )
    return os.fspath(out_path)


def smooth_follow_telemetry(path: str | os.PathLike[str]) -> str:
    """Apply smoothing to a follow telemetry file and return the smoothed path."""

    cx_vals: list[float] = []
    cy_vals: list[float] = []
    zoom_vals: list[float] = []

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

    smoothed_cx, smoothed_cy = smooth_and_limit_camera_path(cx_vals, cy_vals)
    zoom_arr = np.asarray(zoom_vals, dtype=float)
    if zoom_arr.size > 0:
        kernel = np.array([0.25, 0.5, 0.25], dtype=float)
        zoom_smoothed = np.convolve(zoom_arr, kernel, mode="same")
    else:
        zoom_smoothed = zoom_arr

    out_path = Path(path).with_name(f"{Path(path).stem}__smooth.jsonl")
    with out_path.open("w", encoding="utf-8") as handle:
        for idx, (x, y) in enumerate(zip(smoothed_cx, smoothed_cy)):
            handle.write(
                json.dumps(
                    {
                        "f": idx,
                        "cx": float(x),
                        "cy": float(y),
                        "zoom": float(zoom_smoothed[idx]) if idx < len(zoom_smoothed) else 1.0,
                    }
                )
                + "\n"
            )

    return os.fspath(out_path)


def edge_zoom_out(
    cx,
    cy,
    bx,
    by,
    crop_w,
    crop_h,
    W,
    H,
    margin_px,
    s_cap=1.30,
    *,
    edge_frac: float = 1.0,
):
    """Return a zoom-out multiplier that keeps the ball off the crop edge.

    ``s_out`` scales the crop dimensions (``eff_w = crop_w * s_out``) and is
    constrained by ``s_cap`` and the available border headroom.  ``edge_frac``
    softens the onset of the zoom so that we start nudging the crop outward
    before the ball fully breaches the requested ``margin_px``.
    """

    hx, hy = 0.5 * crop_w, 0.5 * crop_h
    dx, dy = abs(bx - cx), abs(by - cy)

    margin_px = max(0.0, float(margin_px))
    s_cap = max(1.0, float(s_cap))
    if not math.isfinite(edge_frac) or edge_frac <= 0.0:
        edge_frac = 1.0

    def _axis_scale(delta: float, half: float) -> float:
        if half <= 0.0:
            return 1.0

        need = max(1.0, (delta + margin_px) / max(half, 1e-6))
        if margin_px <= 0.0:
            return need

        actual_margin = max(0.0, half - delta)
        trigger_margin = margin_px / edge_frac if edge_frac > 0.0 else half
        trigger_margin = max(margin_px, min(trigger_margin, half))

        if actual_margin >= trigger_margin:
            return 1.0
        if actual_margin <= margin_px or trigger_margin <= margin_px:
            return need

        blend = (trigger_margin - actual_margin) / max(
            trigger_margin - margin_px, 1e-6
        )
        return 1.0 + blend * (need - 1.0)

    s_soft_x = _axis_scale(dx, hx)
    s_soft_y = _axis_scale(dy, hy)
    s_ball = max(1.0, s_soft_x, s_soft_y)

    # Border headroom: max zoom-out we can afford without leaving the image
    s_max_l = cx / max(hx, 1e-6)
    s_max_r = (W - 1 - cx) / max(hx, 1e-6)
    s_max_t = cy / max(hy, 1e-6)
    s_max_b = (H - 1 - cy) / max(hy, 1e-6)
    s_border = max(1.0, min(s_max_l, s_max_r, s_max_t, s_max_b))

    return max(1.0, min(s_ball, min(s_border, s_cap)))


def _inside_crop(bx, by, cx, cy, crop_w, crop_h, margin):
    x0 = cx - crop_w / 2 + margin
    x1 = cx + crop_w / 2 - margin
    y0 = cy - crop_h / 2 + margin
    y1 = cy + crop_h / 2 - margin
    return (bx >= x0) and (bx <= x1) and (by >= y0) and (by <= y1)


def _clamp_cam(cx, cy, W, H, crop_w, crop_h):
    cx = max(crop_w / 2, min(W - crop_w / 2, cx))
    cy = max(crop_h / 2, min(H - crop_h / 2, cy))
    return cx, cy


def _motion_centroid(
    prev_gray: Optional[np.ndarray],
    cur_gray: Optional[np.ndarray],
    field_mask: Optional[np.ndarray],
    flow_thresh_px: float = 1.6,
) -> Optional[Tuple[float, float]]:
    if prev_gray is None or cur_gray is None:
        return None
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, cur_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    mag = cv2.magnitude(flow[..., 0], flow[..., 1])
    mot = (mag >= float(flow_thresh_px)).astype(np.uint8) * 255
    if field_mask is not None and field_mask.size == mag.size:
        mot = cv2.bitwise_and(mot, field_mask)
    mot = cv2.morphologyEx(mot, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"] <= 1e-3:
        return None
    cx = float(M["m10"] / M["m00"])
    cy = float(M["m01"] / M["m00"])
    return cx, cy


def _get_ball_xy_src(rec, src_w, src_h):
    """
    Return ball center in *source pixel space* (x,y), regardless of which fields exist in the record.
    Accepts bx/by, bx_stab/by_stab, bx_raw/by_raw, or normalized u/v.
    """
    # priority: stabilized, then plain, then raw
    for kx, ky in (("bx_stab", "by_stab"), ("bx", "by"), ("bx_raw", "by_raw")):
        if kx in rec and ky in rec:
            return float(rec[kx]), float(rec[ky])

    # normalized fallback (0..1); tolerate slight overshoot
    if "u" in rec and "v" in rec:
        u = float(rec["u"])
        v = float(rec["v"])
        return max(0.0, min(1.0, u)) * (src_w - 1), max(0.0, min(1.0, v)) * (src_h - 1)

    # last resort: not found
    return None, None


def edge_aware_zoom(
    cx: float,
    cy: float,
    bx: Optional[float],
    by: Optional[float],
    cw: float,
    ch: float,
    width: float,
    height: float,
    margin_px: float,
    *,
    s_min: float = 0.75,
) -> float:
    """Return a zoom scale (<= 1.0) that avoids edge clamps while keeping the ball inside a margin."""

    if cw <= 0.0 or ch <= 0.0 or width <= 0.0 or height <= 0.0:
        return 1.0

    cx = float(cx)
    cy = float(cy)
    cw = float(cw)
    ch = float(ch)
    width = float(width)
    height = float(height)
    margin_px = max(0.0, float(margin_px))

    s_needed = 1.0

    half_w = cw / 2.0
    half_h = ch / 2.0

    if half_w <= 0.0 or half_h <= 0.0:
        return 1.0

    # compute minimum scale needed to avoid clamping against source edges
    s_clamp_x = 1.0
    s_clamp_y = 1.0
    if cx - half_w < 0.0:
        s_clamp_x = min(s_clamp_x, (max(cx, 0.0) * 2.0) / max(cw, 1e-6))
    if cx + half_w > width:
        s_clamp_x = min(s_clamp_x, (max(width - cx, 0.0) * 2.0) / max(cw, 1e-6))
    if cy - half_h < 0.0:
        s_clamp_y = min(s_clamp_y, (max(cy, 0.0) * 2.0) / max(ch, 1e-6))
    if cy + half_h > height:
        s_clamp_y = min(s_clamp_y, (max(height - cy, 0.0) * 2.0) / max(ch, 1e-6))

    s_needed = min(s_needed, s_clamp_x, s_clamp_y)

    if bx is not None and by is not None and math.isfinite(bx) and math.isfinite(by):
        bx = float(bx)
        by = float(by)
        dx = abs(bx - cx)
        dy = abs(by - cy)
        if margin_px > 0.0:
            hx_margin = max(half_w - margin_px, 0.0)
            hy_margin = max(half_h - margin_px, 0.0)
            if hx_margin > 0.0 and dx > hx_margin:
                s_needed = min(s_needed, hx_margin / max(dx, 1e-6))
            if hy_margin > 0.0 and dy > hy_margin:
                s_needed = min(s_needed, hy_margin / max(dy, 1e-6))

    s_min = max(0.0, min(1.0, float(s_min)))
    s_needed = max(s_min, min(1.0, float(s_needed)))
    return float(s_needed)


def smooth_and_limit_camera_path(
    cx_path: Sequence[float],
    cy_path: Sequence[float],
    *,
    max_dx: float = BALL_MAX_DX,
    max_dy: float = BALL_MAX_DY,
    alpha: float = BALL_FOLLOW_ALPHA,
) -> tuple[list[float], list[float]]:
    """
    Given per-frame crop centers (cx_path, cy_path), apply bidirectional
    exponential smoothing followed by a per-frame slew-rate clamp to keep
    the virtual camera feeling human-operated instead of snapping.

    Uses a forward+backward EMA averaged together so both the start *and*
    end of the clip receive equal smoothing (a forward-only EMA is smooth
    at the start but reactive/jittery at the end).

    - ``alpha`` controls responsiveness. Higher values react faster but can
      jitter; lower values are smoother.
    - ``max_dx`` / ``max_dy`` cap the per-frame delta so whip-pans are
      softened.
    """

    n = len(cx_path)
    if n == 0:
        return list(cx_path), list(cy_path)

    cx_f = list(cx_path)
    cy_f = list(cy_path)
    cx_b = list(cx_path)
    cy_b = list(cy_path)

    # Forward EMA pass
    for i in range(1, n):
        cx_f[i] = alpha * cx_f[i] + (1.0 - alpha) * cx_f[i - 1]
        cy_f[i] = alpha * cy_f[i] + (1.0 - alpha) * cy_f[i - 1]

    # Backward EMA pass
    for i in range(n - 2, -1, -1):
        cx_b[i] = alpha * cx_b[i] + (1.0 - alpha) * cx_b[i + 1]
        cy_b[i] = alpha * cy_b[i] + (1.0 - alpha) * cy_b[i + 1]

    # Average forward and backward for symmetric smoothing
    cx_s = [0.5 * (cx_f[i] + cx_b[i]) for i in range(n)]
    cy_s = [0.5 * (cy_f[i] + cy_b[i]) for i in range(n)]

    # Slew-rate clamp: forward pass
    for i in range(1, n):
        dx = cx_s[i] - cx_s[i - 1]
        dy = cy_s[i] - cy_s[i - 1]

        if abs(dx) > max_dx:
            cx_s[i] = cx_s[i - 1] + math.copysign(max_dx, dx)
        if abs(dy) > max_dy:
            cy_s[i] = cy_s[i - 1] + math.copysign(max_dy, dy)

    # Slew-rate clamp: backward pass to prevent end-of-clip jitter
    for i in range(n - 2, -1, -1):
        dx = cx_s[i] - cx_s[i + 1]
        dy = cy_s[i] - cy_s[i + 1]

        if abs(dx) > max_dx:
            cx_s[i] = cx_s[i + 1] + math.copysign(max_dx, dx)
        if abs(dy) > max_dy:
            cy_s[i] = cy_s[i + 1] + math.copysign(max_dy, dy)

    return cx_s, cy_s


def smooth_center_path(cx_path, cy_path, window=9, max_step_px=40.0):
    """
    Heavy smoothing so the camera feels human-operated.
    - window: moving average window
    - max_step_px: max allowed jump per frame for the center
    """

    n = len(cx_path)
    if n == 0:
        return list(cx_path), list(cy_path)

    def smooth_1d(values):
        half = window // 2
        out = [0.0] * n
        for i in range(n):
            j0 = max(0, i - half)
            j1 = min(n, i + half + 1)
            out[i] = sum(values[j0:j1]) / (j1 - j0)
        return out

    sx = smooth_1d(cx_path)
    sy = smooth_1d(cy_path)

    # Limit per-frame speed (jerk guard)
    for i in range(1, n):
        dx = sx[i] - sx[i - 1]
        dy = sy[i] - sy[i - 1]
        dist = (dx * dx + dy * dy) ** 0.5
        if dist > max_step_px and dist > 0:
            scale = max_step_px / dist
            sx[i] = sx[i - 1] + dx * scale
            sy[i] = sy[i - 1] + dy * scale

    return sx, sy


def build_raw_ball_center_path(
    telemetry: Sequence[Mapping[str, object]],
    frame_width: int,
    frame_height: int,
    crop_width: int,
    crop_height: int,
    *,
    default_y_frac: float = 0.45,
    vertical_bias_frac: float = 0.08,
    use_red_fallback: bool = False,
    use_ball_telemetry: bool = True,
) -> tuple[list[float], list[float]]:
    """
    Build raw crop centers from telemetry, clamping to keep the crop window in
    bounds and biasing the view slightly above the ball so there's forward
    field context.

    Missing or invalid telemetry frames are filled by carrying forward the last
    valid position, falling back to a neutral center when needed.
    """

    n_frames = len(telemetry)
    if n_frames <= 0:
        return [], []

    half_w = float(crop_width) / 2.0
    half_h = float(crop_height) / 2.0
    default_cx = float(frame_width) / 2.0
    default_cy = float(frame_height) * float(default_y_frac)
    bias_px = float(frame_height) * float(vertical_bias_frac)

    def clamp_center(cx: float, cy: float) -> tuple[float, float]:
        if crop_width >= frame_width:
            cx_clamped = float(frame_width) / 2.0
        else:
            max_x0 = max(0.0, float(frame_width) - float(crop_width))
            x0 = min(max(cx - half_w, 0.0), max_x0)
            cx_clamped = x0 + half_w

        if crop_height >= frame_height:
            cy_clamped = float(frame_height) / 2.0
        else:
            max_y0 = max(0.0, float(frame_height) - float(crop_height))
            y0 = min(max(cy - half_h, 0.0), max_y0)
            cy_clamped = y0 + half_h

        return cx_clamped, cy_clamped

    bx_vals: list[float] = []
    by_vals: list[float] = []

    for i, rec in enumerate(telemetry):
        row = rec if isinstance(rec, Mapping) else {}

        # visibility: accept explicit 'vis'/'visible', else infer from confidence if present
        vis = row.get("vis", row.get("visible", None))
        if vis is None:
            conf = row.get("conf", row.get("confidence", 0.0))
            try:
                vis = float(conf) >= 0.5
            except Exception:
                vis = False

        def _as_float(v):
            try:
                if v is None:
                    return float("nan")
                return float(v)
            except Exception:
                return float("nan")

        # telemetry keys can vary: bx/by, x/y, cx/cy (support all)
        bx_val = _as_float(row.get("bx", row.get("x", row.get("cx"))))
        by_val = _as_float(row.get("by", row.get("y", row.get("cy"))))

        fallback_xy_raw = row.get("fallback_ball_xy")
        fallback_xy: tuple[float, float] | None = None
        if isinstance(fallback_xy_raw, Sequence) and len(fallback_xy_raw) >= 2:
            fx = _as_float(fallback_xy_raw[0])
            fy = _as_float(fallback_xy_raw[1])
            if math.isfinite(fx) and math.isfinite(fy):
                fallback_xy = (fx, fy)

        motion_centroid_raw = row.get("motion_centroid")
        motion_centroid_xy: tuple[float, float] | None = None
        if isinstance(motion_centroid_raw, Sequence) and len(motion_centroid_raw) >= 2:
            mx = _as_float(motion_centroid_raw[0])
            my = _as_float(motion_centroid_raw[1])
            if math.isfinite(mx) and math.isfinite(my):
                motion_centroid_xy = (mx, my)

        telemetry_ok = vis and math.isfinite(bx_val) and math.isfinite(by_val)

        ball_xy = None
        if use_red_fallback and fallback_xy is not None:
            ball_xy = fallback_xy
        elif use_ball_telemetry and telemetry_ok:
            ball_xy = (bx_val, by_val)
        elif motion_centroid_xy is not None:
            ball_xy = motion_centroid_xy

        if ball_xy is not None:
            bx_vals.append(float(ball_xy[0]))
            by_vals.append(float(ball_xy[1]))
        else:
            bx_vals.append(float("nan"))
            by_vals.append(float("nan"))

    def _fill_nan_edges(xs: list[float], ys: list[float]) -> tuple[list[float], list[float]]:
        n = len(xs)
        if n == 0:
            return xs, ys

        first = None
        for idx in range(n):
            if math.isfinite(xs[idx]) and math.isfinite(ys[idx]):
                first = idx
                break

        last = None
        for idx in range(n - 1, -1, -1):
            if math.isfinite(xs[idx]) and math.isfinite(ys[idx]):
                last = idx
                break

        if first is None or last is None:
            return xs, ys

        for idx in range(0, first):
            xs[idx] = xs[first]
            ys[idx] = ys[first]

        for idx in range(last + 1, n):
            xs[idx] = xs[last]
            ys[idx] = ys[last]

        return xs, ys

    bx_vals, by_vals = _fill_nan_edges(bx_vals, by_vals)

    cx_vals: list[float] = []
    cy_vals: list[float] = []
    last_valid: tuple[float, float] | None = None
    for bx_val, by_val in zip(bx_vals, by_vals):
        if math.isfinite(bx_val) and math.isfinite(by_val):
            last_valid = (bx_val, by_val)

        if last_valid is not None:
            bx_use, by_use = last_valid
        else:
            bx_use, by_use = default_cx, default_cy

        cx_val = float(bx_use)
        cy_val = float(by_use - bias_px)
        cx_val, cy_val = clamp_center(cx_val, cy_val)

        cx_vals.append(cx_val)
        cy_vals.append(cy_val)

    return cx_vals, cy_vals


def clamp_center_path_to_bounds(
    cx_path: Sequence[float],
    cy_path: Sequence[float],
    frame_width: int,
    frame_height: int,
    crop_width: int,
    crop_height: int,
) -> tuple[list[float], list[float]]:
    """Clamp center paths so the implied crop stays inside the source frame."""

    if len(cx_path) != len(cy_path):
        return list(cx_path), list(cy_path)

    clamped_cx: list[float] = []
    clamped_cy: list[float] = []

    max_x0 = max(0.0, float(frame_width) - float(crop_width))
    max_y0 = max(0.0, float(frame_height) - float(crop_height))

    for cx, cy in zip(cx_path, cy_path):
        x0 = int(round(float(cx) - float(crop_width) / 2.0))
        y0 = int(round(float(cy) - float(crop_height) / 2.0))
        x0 = max(0, min(x0, int(max_x0)))
        y0 = max(0, min(y0, int(max_y0)))
        clamped_cx.append(float(x0 + float(crop_width) / 2.0))
        clamped_cy.append(float(y0 + float(crop_height) / 2.0))

    return clamped_cx, clamped_cy


def _clamp_ball_cam_center(cx: float, *, crop_width: float, frame_width: float) -> float:
    half = crop_width / 2.0
    return max(half, min(frame_width - half, cx))


def _jerk95(px: np.ndarray, *, fps: float) -> float:
    if len(px) < 4 or fps <= 0:
        return 0.0
    v = np.diff(px)
    a = np.diff(v)
    j = np.diff(a) * (fps**3)
    j_abs = np.abs(j)
    return float(np.percentile(j_abs, 95)) if j_abs.size else 0.0


def _interp_nan(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    out = values.copy()
    n = len(out)
    idx = np.arange(n)
    mask = np.isfinite(out)
    if not mask.any():
        return out
    out[~mask] = np.interp(idx[~mask], idx[mask], out[mask])
    return out


def smooth_path(values: Sequence[float], alpha: float = 0.2) -> list[float]:
    """Simple exponential smoother to tame frame-to-frame jitter."""

    smoothed: list[float] = []
    prev: float | None = None
    for v in values:
        if prev is None:
            prev = float(v)
        else:
            prev = alpha * float(v) + (1.0 - alpha) * prev
        smoothed.append(prev)
    return smoothed


def smooth_series(values, alpha: float = 0.1, passes: int = 3):
    """
    Smooth a 1D sequence using an exponential moving average, applied
    multiple passes to strongly low-pass the motion.

    Args:
        values: iterable of floats (e.g., ball cx samples).
        alpha: EMA smoothing factor in (0, 1]; lower = smoother.
        passes: how many forward/backward EMA passes to apply.

    Returns:
        List of smoothed values, same length as input.
    """
    vals = list(values)
    n = len(vals)
    if n <= 1:
        return vals

    for _ in range(passes):
        # forward pass
        prev = vals[0]
        for i in range(1, n):
            v = vals[i]
            prev = alpha * v + (1.0 - alpha) * prev
            vals[i] = prev

        # backward pass (for more “camera operator” feel)
        prev = vals[-1]
        for i in range(n - 2, -1, -1):
            v = vals[i]
            prev = alpha * v + (1.0 - alpha) * prev
            vals[i] = prev

    return vals


def _load_ball_cam_array(path: Path, num_frames: int) -> np.ndarray:
    samples = load_ball_telemetry(path)
    arr = np.full((num_frames, 3), np.nan, dtype=float)
    for frame_idx, sample in enumerate(samples):
        conf = _safe_float(getattr(sample, "conf", None))
        x = _safe_float(getattr(sample, "x", None))
        y = _safe_float(getattr(sample, "y", None))
        if x is None or y is None:
            continue
        arr[frame_idx, 0] = x
        arr[frame_idx, 1] = y
        arr[frame_idx, 2] = conf if conf is not None else 0.0
    return arr


def load_ball_telemetry_jsonl(path: str, src_w: int, src_h: int, logger=None):
    if logger:
        logger.info("[BALL-TELEMETRY] loading %s", path)

    ball_samples = _load_ball_telemetry(path)
    xs = [s["cx"] for s in ball_samples]
    ys = [s["cy"] for s in ball_samples]

    if logger:
        logger.info(
            "[BALL-TELEMETRY] tele_range_x=[%.1f, %.1f], tele_range_y=[%.1f, %.1f]",
            min(xs),
            max(xs),
            min(ys),
            max(ys),
        )

    return xs, ys


def telemetry_sanity(ball_x, ball_y, w, h, *, max_top_frac=0.33) -> float:
    ok = 0
    n = 0
    for x, y in zip(ball_x, ball_y):
        if x is None or y is None:
            continue
        n += 1
        if 0 <= x < w and (h * max_top_frac) <= y < h:
            ok += 1
    return (ok / max(1, n))


def build_raw_ball_path(telemetry: np.ndarray, fps: float) -> np.ndarray:
    """Return Nx2 array of raw ball positions with finite interpolation."""

    if telemetry.size == 0:
        return np.zeros((0, 2), dtype=float)

    valid_mask = np.isfinite(telemetry[:, 0]) & np.isfinite(telemetry[:, 1]) & (
        telemetry[:, 2] >= 0.5
    )
    raw = np.full((telemetry.shape[0], 2), np.nan, dtype=float)
    raw[valid_mask, 0] = telemetry[valid_mask, 0]
    raw[valid_mask, 1] = telemetry[valid_mask, 1]

    raw[:, 0] = _interp_nan(raw[:, 0])
    raw[:, 1] = _interp_nan(raw[:, 1])
    return raw


def build_target_ball_path(raw_path: np.ndarray, fps: float) -> np.ndarray:
    """Pre-smooth raw detections and add predictive lead."""

    if raw_path.size == 0:
        return raw_path

    radius = max(1, int(BALL_PATH_SMOOTHING_RADIUS))
    kernel_size = 2 * radius + 1
    kernel = np.ones(kernel_size, dtype=float) / kernel_size

    padded_x = np.pad(raw_path[:, 0], (radius, radius), mode="edge")
    padded_y = np.pad(raw_path[:, 1], (radius, radius), mode="edge")
    smooth_x = np.convolve(padded_x, kernel, mode="valid")
    smooth_y = np.convolve(padded_y, kernel, mode="valid")
    smooth = np.stack([smooth_x, smooth_y], axis=1)

    lead_frames = max(1, int(round(max(fps, 1.0) * LEAD_WINDOW_FACTOR)))
    target = smooth.copy()
    n = len(smooth)
    for i in range(n):
        j1 = min(n, i + lead_frames)
        window = smooth[i:j1]
        if window.size == 0:
            continue
        target[i, 0] = float(np.mean(window[:, 0]))
        target[i, 1] = float(np.mean(window[:, 1]))
    return target


def build_camera_path(
    target_ball_path: np.ndarray,
    fps: float,
    base_width: float,
    base_height: float,
    portrait_width: float,
    portrait_height: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return smooth camera center and zoom arrays."""

    if target_ball_path.size == 0:
        return (
            np.zeros(0, dtype=float),
            np.zeros(0, dtype=float),
            np.zeros(0, dtype=float),
        )

    n = len(target_ball_path)
    cam_x = np.zeros(n, dtype=float)
    cam_y = np.zeros(n, dtype=float)
    cam_zoom = np.zeros(n, dtype=float)

    half_w = portrait_width / 2.0
    half_h = portrait_height / 2.0
    max_dy = max(3.0, base_height * MAX_CAM_DY_PER_FRAME_FRAC)

    ball_x = target_ball_path[:, 0]
    ball_y = target_ball_path[:, 1]

    # Smooth + predictive target for X
    smoothed_x = smooth_path(ball_x, alpha=0.25)
    vx = np.diff(ball_x, prepend=ball_x[0])
    lookahead_frames = 4
    predicted_x = [sx + float(v) * lookahead_frames for sx, v in zip(smoothed_x, vx)]
    blend = 0.5
    target_cx = [((1.0 - blend) * sx) + (blend * px) for sx, px in zip(smoothed_x, predicted_x)]

    deadzone_px = 30.0
    max_pan_per_frame = 25.0
    min_dwell_frames = 3
    off_center_count = 0

    current_cx = float(np.clip(target_cx[0], half_w, base_width - half_w))
    cam_x[0] = current_cx
    cam_y[0] = float(np.clip(ball_y[0], half_h, base_height - half_h))
    cam_zoom[0] = ZOOM_MIN

    for i in range(1, n):
        desired_x = float(np.clip(target_cx[i], half_w, base_width - half_w))
        delta = desired_x - current_cx

        if abs(delta) < deadzone_px:
            off_center_count = 0
        else:
            off_center_count += 1
            if off_center_count >= min_dwell_frames:
                delta = max(-max_pan_per_frame, min(max_pan_per_frame, delta))
                current_cx = float(np.clip(current_cx + delta, half_w, base_width - half_w))
                off_center_count = 0

        cam_x[i] = current_cx

        desired_y = float(np.clip(ball_y[i], half_h, base_height - half_h))
        cam_y[i] = cam_y[i - 1] + CAM_SMOOTH_ALPHA_Y * (desired_y - cam_y[i - 1])
        dy = cam_y[i] - cam_y[i - 1]
        if abs(dy) > max_dy:
            cam_y[i] = cam_y[i - 1] + math.copysign(max_dy, dy)
        cam_y[i] = float(np.clip(cam_y[i], half_h, base_height - half_h))

    # Zoom driven by ball speed
    speed = np.abs(np.diff(target_ball_path[:, 0], prepend=target_ball_path[0, 0]))
    speed_norm = np.clip(speed / (base_width * 0.10), 0.0, 1.0)
    desired_zoom = ZOOM_MAX - speed_norm * (ZOOM_MAX - ZOOM_MIN)
    for i in range(n):
        if i == 0:
            cam_zoom[i] = desired_zoom[i]
        else:
            cam_zoom[i] = cam_zoom[i - 1] + CAM_SMOOTH_ALPHA_ZOOM * (desired_zoom[i] - cam_zoom[i - 1])
        cam_zoom[i] = float(np.clip(cam_zoom[i], ZOOM_MIN, ZOOM_MAX))

        crop_w = portrait_width / cam_zoom[i]
        crop_h = portrait_height / cam_zoom[i]
        cam_x[i] = float(np.clip(cam_x[i], crop_w / 2.0, base_width - crop_w / 2.0))
        cam_y[i] = float(np.clip(cam_y[i], crop_h / 2.0, base_height - crop_h / 2.0))

    return cam_x, cam_y, cam_zoom


def compute_locked_ball_cam_path(ball_cx_raw, src_w, crop_w, cfg, logger=None):
    """
    Compute a camera center path that stays locked to a smoothed ball trajectory.
    - ball_cx_raw: list/sequence of ball x positions per frame in source coords
    - src_w: source frame width in pixels
    - crop_w: width of the virtual crop in pixels
    """
    N = len(ball_cx_raw)
    if N == 0:
        return []

    lead_frames = int(cfg.get("ball_cam_lead_frames", 3))
    smooth_window = int(cfg.get("ball_cam_smooth_window", 5))
    max_speed = float(cfg.get("ball_cam_max_speed_px", 60.0))
    margin = float(cfg.get("ball_cam_margin_px", 100.0))

    # Safety clamps
    if smooth_window < 1:
        smooth_window = 1
    if max_speed <= 0:
        max_speed = 1.0

    half_crop_w = crop_w / 2.0

    cam_cx = [0.0] * N

    # Initialize at the ball position (with lead) on frame 0
    j0 = min(lead_frames, N - 1)
    cx0 = ball_cx_raw[j0]
    cx0 = max(half_crop_w - margin, min(src_w - half_crop_w + margin, cx0))
    cam_cx[0] = cx0

    for i in range(1, N):
        # Look ahead a little to avoid visual lag
        j = min(i + lead_frames, N - 1)

        # Local smoothing window around the (possibly leaded) index
        w = smooth_window
        start = max(0, j - w // 2)
        end = min(N, j + w // 2 + 1)
        count = end - start
        if count <= 0:
            desired = ball_cx_raw[j]
        else:
            desired = sum(ball_cx_raw[start:end]) / float(count)

        # Clamp movement per frame to avoid crazy jumps
        prev = cam_cx[i - 1]
        delta = desired - prev

        if delta > max_speed:
            delta = max_speed
        elif delta < -max_speed:
            delta = -max_speed

        cx = prev + delta

        # Keep crop inside frame, honoring margin
        cx = max(half_crop_w - margin, min(src_w - half_crop_w + margin, cx))

        cam_cx[i] = cx

    if logger:
        logger.info(
            "[BALL-CAM LOCK] N=%d, cx_range=[%.1f, %.1f], lead_frames=%d, "
            "smooth_window=%d, max_speed_px=%.1f, margin=%.1f",
            N,
            min(cam_cx),
            max(cam_cx),
            lead_frames,
            smooth_window,
            max_speed,
            margin,
        )

    return cam_cx


def compute_ball_lock_strict(
    ball_cx_raw,
    ball_cy_raw,
    src_w,
    src_h,
    crop_w,
    crop_h,
    cfg,
    logger=None,
):
    """
    Compute a strict ball-locked camera path:
    - For each frame, choose a crop center (cam_cx, cam_cy) that keeps the ball
      inside the 9:16 crop with a margin.
    - Uses a small temporal smoothing window + a lead in time to avoid visual lag.
    - Clamps camera so the resulting crop never goes outside the source frame.
    """
    N = len(ball_cx_raw)
    if N == 0:
        return [], []

    # If we don't have vertical telemetry, just fake a flat line so we at least
    # get perfect horizontal behavior.
    if not ball_cy_raw or len(ball_cy_raw) != N:
        ball_cy_raw = [src_h * 0.5] * N

    lead_frames = int(cfg.get("ball_cam_lead_frames", 3))
    smooth_window = int(cfg.get("ball_cam_smooth_window", 5))
    max_speed = float(cfg.get("ball_cam_max_speed_px", 120.0))
    margin = float(cfg.get("ball_cam_margin_px", 80.0))
    vpos = float(cfg.get("ball_cam_vertical_pos", 0.60))

    if smooth_window < 1:
        smooth_window = 1
    if max_speed <= 0:
        max_speed = 1.0
    if vpos < 0.0:
        vpos = 0.0
    if vpos > 1.0:
        vpos = 1.0

    half_w = crop_w / 2.0
    half_h = crop_h / 2.0

    cam_cx = [0.0] * N
    cam_cy = [0.0] * N

    # Build leaded index sequence (so we look a few frames ahead)
    idx_seq = [min(i + lead_frames, N - 1) for i in range(N)]

    # Smooth horizontal, use raw vertical (or you can smooth vertical similarly)
    smoothed_cx = [0.0] * N
    for i in range(N):
        j = idx_seq[i]
        # local average in a window around the leaded index
        start = max(0, j - smooth_window // 2)
        end = min(N, j + smooth_window // 2 + 1)
        count = max(1, end - start)
        smoothed_cx[i] = sum(ball_cx_raw[start:end]) / float(count)

    # Initialize camera at frame 0
    bx0 = smoothed_cx[0]
    by0 = ball_cy_raw[0]

    # Horizontal: center on ball, but clamp to valid crop range
    cx0 = bx0
    cx0 = max(half_w, min(src_w - half_w, cx0))

    # Vertical: place ball at 'vpos' fraction of the crop height (0=top,1=bottom)
    # We want: ball_y = cam_cy - half_h + vpos*crop_h
    cy0 = by0 - (vpos * crop_h - half_h)
    cy0 = max(half_h, min(src_h - half_h, cy0))

    cam_cx[0] = cx0
    cam_cy[0] = cy0

    for i in range(1, N):
        # Desired ball position for this frame
        j = idx_seq[i]
        bx = smoothed_cx[i]
        by = ball_cy_raw[j]

        # Ideal camera center before speed clamp
        desired_cx = bx
        desired_cy = by - (vpos * crop_h - half_h)

        # Clamp to legal center range so crop stays inside frame
        desired_cx = max(half_w, min(src_w - half_w, desired_cx))
        desired_cy = max(half_h, min(src_h - half_h, desired_cy))

        # Speed limit (per-frame) to avoid insane jumps, but keep it generous
        prev_cx = cam_cx[i - 1]
        prev_cy = cam_cy[i - 1]

        dx = desired_cx - prev_cx
        dy = desired_cy - prev_cy

        if dx > max_speed:
            dx = max_speed
        elif dx < -max_speed:
            dx = -max_speed

        if dy > max_speed:
            dy = max_speed
        elif dy < -max_speed:
            dy = -max_speed

        cx = prev_cx + dx
        cy = prev_cy + dy

        # Final clamp
        cx = max(half_w, min(src_w - half_w, cx))
        cy = max(half_h, min(src_h - half_h, cy))

        cam_cx[i] = cx
        cam_cy[i] = cy

    if logger:
        logger.info(
            "[BALL-CAM STRICT] N=%d, cx_range=[%.1f, %.1f], cy_range=[%.1f, %.1f], "
            "lead_frames=%d, smooth_window=%d, max_speed_px=%.1f, margin=%.1f, vpos=%.2f",
            N,
            min(cam_cx),
            max(cam_cx),
            min(cam_cy),
            max(cam_cy),
            lead_frames,
            smooth_window,
            max_speed,
            margin,
            vpos,
        )

    return cam_cx, cam_cy


def compute_ball_lock_raw(
    ball_cx_raw,
    ball_cy_raw,
    src_w,
    src_h,
    crop_w,
    crop_h,
    cfg,
    logger=None,
):
    """
    Hard lock: center the crop on the ball every frame (no smoothing, no speed limit).
    This is mainly for debugging and 'never lose the ball' behavior.
    """
    N = len(ball_cx_raw)
    if N == 0:
        return [], []

    if not ball_cy_raw or len(ball_cy_raw) != N:
        ball_cy_raw = [src_h * 0.5] * N

    vpos = float(cfg.get("ball_cam_vertical_pos", 0.60))
    if vpos < 0.0:
        vpos = 0.0
    if vpos > 1.0:
        vpos = 1.0

    half_w = crop_w / 2.0
    half_h = crop_h / 2.0

    cam_cx = [0.0] * N
    cam_cy = [0.0] * N

    for i in range(N):
        bx = ball_cx_raw[i]
        by = ball_cy_raw[i]

        # Ideal center: ball at vpos inside the crop
        cx = bx
        cy = by - (vpos * crop_h - half_h)

        # Clamp to valid center so crop stays within frame
        cx = max(half_w, min(src_w - half_w, cx))
        cy = max(half_h, min(src_h - half_h, cy))

        cam_cx[i] = cx
        cam_cy[i] = cy

    if logger:
        logger.info(
            "[BALL-CAM RAW] N=%d, cx_range=[%.1f, %.1f], cy_range=[%.1f, %.1f], vpos=%.2f",
            N,
            min(cam_cx),
            max(cam_cx),
            min(cam_cy),
            max(cam_cy),
            vpos,
        )

    return cam_cx, cam_cy


def compute_ema_ball_cam_path(ball_cx_raw, ball_cy_raw, src_w, src_h, crop_w, crop_h, cfg, logger=None):
    N = len(ball_cx_raw)
    if N == 0:
        return [], []

    lead_frames = int(cfg.get("ball_cam_lead_frames", 5))
    base_alpha = float(cfg.get("ball_cam_base_alpha", 0.30))
    fast_alpha = float(cfg.get("ball_cam_fast_alpha", 0.80))
    catchup_px = float(cfg.get("ball_cam_catchup_thresh_px", 40.0))
    margin = float(cfg.get("ball_cam_margin_px", 100.0))

    speed_fast_px = float(cfg.get("ball_cam_speed_fast_px", 40.0))
    max_pan_slow = float(cfg.get("ball_cam_max_pan_per_frame_slow", 24.0))
    max_pan_fast = float(cfg.get("ball_cam_max_pan_per_frame_fast", 44.0))

    half_crop_w = crop_w / 2.0
    cam_cx = [0.0] * N
    cam_cy = [float(src_h) / 2.0] * N
    cx_prev = float(ball_cx_raw[0]) if N > 0 else 0.0

    for i in range(N):
        j = min(i + lead_frames, N - 1)
        desired = float(ball_cx_raw[j])

        if i == 0:
            ball_vel = 0.0
        else:
            ball_vel = float(ball_cx_raw[i] - ball_cx_raw[i - 1])
        speed = abs(ball_vel)

        if speed_fast_px > 0:
            speed_scale = min(1.0, speed / speed_fast_px)
        else:
            speed_scale = 0.0

        alpha = base_alpha + speed_scale * (fast_alpha - base_alpha)

        delta = desired - cx_prev
        if abs(delta) > catchup_px:
            alpha = max(alpha, fast_alpha)

        step = alpha * delta

        max_pan = max_pan_slow + speed_scale * (max_pan_fast - max_pan_slow)
        if step > max_pan:
            step = max_pan
        elif step < -max_pan:
            step = -max_pan

        cx = cx_prev + step

        cx = max(half_crop_w - margin, min(src_w - half_crop_w + margin, cx))

        cam_cx[i] = cx
        cam_cy[i] = float(src_h) / 2.0 if ball_cy_raw is None else cam_cy[i]
        cx_prev = cx

    return cam_cx, cam_cy


def write_ball_crop_debug_clip(
    in_path: str,
    out_path: str,
    ball_cx: list[float],
    ball_cy: list[float],
    crop_x: list[float],
    crop_y: list[float],
    src_w: int,
    src_h: int,
    crop_w: int,
    crop_h: int,
    fps: float,
    logger=None,
):
    """
    Generate a 1920x1080 debug clip showing:
      - the ball telemetry position as a dot
      - the portrait crop window as a rectangle
    on top of the original wide frame.
    """
    if logger:
        logger.info(
            "[BALL-DEBUG] Writing ball/crop overlay debug clip: %s", out_path
        )

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        if logger:
            logger.error("[BALL-DEBUG] Failed to open %s", in_path)
        return

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (src_w, src_h))

    N = min(len(ball_cx), len(crop_x), len(crop_y))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx >= N:
            break

        bx = float(ball_cx[frame_idx])
        by = float(ball_cy[frame_idx])

        x0 = float(crop_x[frame_idx])
        y0 = float(crop_y[frame_idx])

        # Draw ball position (cyan-ish)
        cv2.circle(
            frame,
            (int(round(bx)), int(round(by))),
            8,
            (255, 255, 0),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

        # Draw crop rect (green)
        x0 = int(round(x0))
        y0 = int(round(y0))
        x1 = int(round(x0 + crop_w))
        y1 = int(round(y0 + crop_h))
        cv2.rectangle(
            frame,
            (x0, y0),
            (x1, y1),
            (0, 255, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    if logger:
        logger.info("[BALL-DEBUG] Finished debug clip: %s", out_path)


def build_ball_cam_plan(
    telemetry_path: Path,
    *,
    num_frames: int,
    fps: float,
    frame_width: int,
    frame_height: int,
    portrait_width: int,
    config: Mapping[str, object] | None = None,
    preset_name: str | None = None,
    in_path: Path | None = None,
    out_path: Path | None = None,
    min_sanity: float | None = None,
    use_red_fallback: bool = False,
    scratch_root: Path | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float]] | tuple[None, dict[str, float]]:
    cfg = dict(BALL_CAM_CONFIG)
    if config:
        cfg.update(config)

    if preset_name == "wide_follow":
        cfg["ball_cam_mode"] = "raw_lock"
        cfg["ball_cam_vertical_pos"] = 0.55
        cfg["ball_cam_margin_px"] = 0.0
        cfg["ball_debug_overlay"] = True

    min_coverage = float(cfg.get("min_coverage", 0.4))

    src_w = float(frame_width)
    src_h = float(frame_height)

    ball_cx_values, ball_cy_values, telemetry_meta = load_ball_path_from_jsonl(
        telemetry_path,
        logger,
        use_red_fallback=use_red_fallback,
        use_ball_telemetry=True,
    )
    if len(ball_cx_values) == 0 or len(ball_cy_values) == 0:
        return None, {
            "coverage": 0.0,
            "conf": 0.0,
            "reason": "empty_ball_telemetry",
            "telemetry_meta": telemetry_meta,
        }

    min_sanity = float(min_sanity if min_sanity is not None else 0.50)
    sanity = telemetry_sanity(ball_cx_values, ball_cy_values, src_w, src_h)
    if sanity < min_sanity:
        logger.warning(
            f"[BALL-TELEMETRY] sanity too low ({sanity:.2f}) -> disabling ball telemetry for this clip"
        )
        if use_red_fallback:
            logger.info("[BALL-FALLBACK-RED] telemetry sanity low -> enabling HSV red-ball fallback")
        return None, {
            "coverage": 0.0,
            "telemetry_rows": int(telemetry_meta.get("total_rows", 0)),
            "telemetry_kept": int(telemetry_meta.get("kept_rows", 0)),
            "telemetry_conf": float(telemetry_meta.get("avg_conf", 0.0)),
            "telemetry_quality": float(telemetry_meta.get("telemetry_quality", 0.0)),
            "sanity": float(sanity),
            "sanity_low": 1.0,
        }

    vpos = float(cfg.get("ball_cam_vertical_pos", 0.55))

    ball_cx_raw = np.full(num_frames, np.nan, dtype=float)
    ball_cy_raw = np.full(num_frames, np.nan, dtype=float)
    max_fill = min(num_frames, len(ball_cx_values), len(ball_cy_values))
    if max_fill > 0:
        ball_cx_raw[:max_fill] = ball_cx_values[:max_fill]
        ball_cy_raw[:max_fill] = ball_cy_values[:max_fill]

    N = max_fill
    logger.info(
        "[BALL-CAM RAW] N=%d, ball_cx_range=[%.1f, %.1f], ball_cy_range=[%.1f, %.1f], vpos=%.2f",
        N,
        float(np.nanmin(ball_cx_raw[:N])) if N else float("nan"),
        float(np.nanmax(ball_cx_raw[:N])) if N else float("nan"),
        float(np.nanmin(ball_cy_raw[:N])) if N else float("nan"),
        float(np.nanmax(ball_cy_raw[:N])) if N else float("nan"),
        vpos,
    )

    valid_mask = np.isfinite(ball_cx_raw) & np.isfinite(ball_cy_raw)
    coverage = float(np.mean(valid_mask)) if num_frames > 0 else 0.0

    stats = {
        "coverage": coverage,
        "telemetry_rows": int(telemetry_meta.get("total_rows", 0)),
        "telemetry_kept": int(telemetry_meta.get("kept_rows", 0)),
        "telemetry_conf": float(telemetry_meta.get("avg_conf", 1.0)),
        "telemetry_quality": float(telemetry_meta.get("telemetry_quality", coverage)),
        "sanity": float(sanity),
        "sanity_low": 0.0,
    }
    min_conf = float(cfg.get("min_confidence", 0.0))
    if coverage < min_coverage or stats["telemetry_conf"] < min_conf:
        logger.info(
            "[BALL-CAM] coverage/conf too weak (coverage=%.1f%%, conf=%.2f < %.2f); falling back to legacy follow",
            coverage * 100.0,
            stats["telemetry_conf"],
            min_conf,
        )
        return None, stats

    # Build a ball-centric camera path with a small predictive lead and adaptive smoothing.
    ball_cx_raw = _interp_nan(ball_cx_raw)
    ball_cy_raw = _interp_nan(ball_cy_raw)

    portrait_mode = portrait_width is not None and portrait_width > 0
    if portrait_mode:
        crop_h = float(src_h)
        crop_w = float(int(round(src_h * 9.0 / 16.0)))
        crop_w = max(1.0, min(crop_w, float(src_w)))
    else:
        crop_w = float(portrait_width)
        desired_crop_h = (
            float(portrait_width) * 16.0 / 9.0 if portrait_width > 0 else float(frame_height)
        )
        crop_h = min(desired_crop_h, float(frame_height))
    half_crop_w = crop_w / 2.0

    finite_ball_x = ball_cx_raw[np.isfinite(ball_cx_raw)]
    finite_ball_y = ball_cy_raw[np.isfinite(ball_cy_raw)]
    if finite_ball_x.size > 0 and float(np.nanmax(finite_ball_x)) <= 1.0:
        ball_cx_raw = ball_cx_raw * src_w
    if finite_ball_y.size > 0 and float(np.nanmax(finite_ball_y)) <= 1.0:
        ball_cy_raw = ball_cy_raw * src_h

    ball_cx_list = [float(v) for v in ball_cx_raw.tolist()]
    ball_cy_list = [float(v) for v in ball_cy_raw.tolist()]

    log_params: dict[str, float | int] | None = None
    cam_cx: List[float] = []
    cam_cy: List[float] = []
    mode = cfg.get("ball_cam_mode", "strict_lock")
    override_crop_x: np.ndarray | None = None
    override_crop_y: np.ndarray | None = None

    if mode == "perfect_follow":
        # Optional smoothing to remove tiny jitter
        window = int(cfg.get("ball_cam_smooth_window", 5))
        if window > 1:
            k = np.ones(window, dtype=np.float32) / float(window)
            ball_cx = np.convolve(ball_cx_raw, k, mode="same")
        else:
            ball_cx = ball_cx_raw.copy()

        ball_cy = ball_cy_raw.copy()

        # Vertical location: use a single fixed target line (e.g. 55% of frame height)
        vpos = float(cfg.get("ball_cam_vertical_pos", 0.55))
        target_cy = vpos * src_h

        # For this test: follow only horizontally; keep vertical fixed
        cam_cx_arr = ball_cx.copy()
        cam_cy_arr = np.full_like(cam_cx_arr, target_cy)

        # Compute crop_x, crop_y for each frame
        half_w = crop_w / 2.0
        half_h = crop_h / 2.0

        crop_x = cam_cx_arr - half_w
        crop_y = cam_cy_arr - half_h

        # Clamp to source bounds
        crop_x = np.clip(crop_x, 0, src_w - crop_w)
        crop_y = np.clip(crop_y, 0, src_h - crop_h)

        override_crop_x = crop_x.astype(float)
        override_crop_y = crop_y.astype(float)

        cam_cx = (crop_x + half_w).tolist()
        cam_cy = (crop_y + half_h).tolist()

    elif mode == "raw_lock":
        cam_cx, cam_cy = compute_ball_lock_raw(
            ball_cx_raw=ball_cx_list,
            ball_cy_raw=ball_cy_list,
            src_w=src_w,
            src_h=src_h,
            crop_w=crop_w,
            crop_h=crop_h,
            cfg=cfg,
            logger=logger,
        )
    elif mode == "strict_lock":
        cam_cx, cam_cy = compute_ball_lock_strict(
            ball_cx_raw=ball_cx_list,
            ball_cy_raw=ball_cy_list,
            src_w=src_w,
            src_h=src_h,
            crop_w=crop_w,
            crop_h=crop_h,
            cfg=cfg,
            logger=logger,
        )
    else:
        lead_frames = int(cfg.get("ball_cam_lead_frames", 5))
        base_alpha = float(cfg.get("ball_cam_base_alpha", 0.30))
        fast_alpha = float(cfg.get("ball_cam_fast_alpha", 0.80))
        catchup_px = float(cfg.get("ball_cam_catchup_thresh_px", 40.0))
        margin = float(cfg.get("ball_cam_margin_px", 100.0))

        speed_fast_px = float(cfg.get("ball_cam_speed_fast_px", 40.0))
        max_pan_slow = float(cfg.get("ball_cam_max_pan_per_frame_slow", 24.0))
        max_pan_fast = float(cfg.get("ball_cam_max_pan_per_frame_fast", 44.0))

        log_params = {
            "lead_frames": lead_frames,
            "base_alpha": base_alpha,
            "fast_alpha": fast_alpha,
            "catchup_px": catchup_px,
            "margin": margin,
            "speed_fast_px": speed_fast_px,
            "max_pan_slow": max_pan_slow,
            "max_pan_fast": max_pan_fast,
        }

        cam_cx, cam_cy = compute_ema_ball_cam_path(
            ball_cx_raw=ball_cx_list,
            ball_cy_raw=ball_cy_list,
            src_w=src_w,
            src_h=src_h,
            crop_w=crop_w,
            crop_h=crop_h,
            cfg=cfg,
            logger=logger,
        )

    cam_cx_arr = np.asarray(cam_cx, dtype=float)
    cam_cy_arr = np.asarray(cam_cy, dtype=float)
    cam_zoom = np.ones(num_frames, dtype=float)

    crop_w_arr = np.full(num_frames, float(crop_w), dtype=float)
    crop_h_arr = np.full(num_frames, float(crop_h), dtype=float)

    if override_crop_x is not None and override_crop_y is not None:
        x0 = override_crop_x
        y0 = override_crop_y
    else:
        x0 = np.clip(cam_cx_arr - (crop_w_arr / 2.0), 0.0, float(frame_width) - crop_w_arr)
        y0 = np.clip(cam_cy_arr - (crop_h_arr / 2.0), 0.0, float(frame_height) - crop_h_arr)

    cam_cx_arr = x0 + (crop_w_arr / 2.0)
    cam_cy_arr = y0 + (crop_h_arr / 2.0)

    inside_count = 0
    inside_strict = 0
    margin = float(cfg.get("ball_cam_margin_px", 80.0))

    for i in range(num_frames):
        bx = ball_cx_raw[i]
        by = ball_cy_raw[i]

        cx = cam_cx[i]
        cy = cam_cy[i]

        half_w = crop_w / 2.0
        half_h = crop_h / 2.0

        # EXACT same clamp as render path
        crop_x = max(0.0, min(src_w - crop_w, cx - half_w))
        crop_y = max(0.0, min(src_h - crop_h, cy - half_h))
        eff_cx = crop_x + half_w
        eff_cy = crop_y + half_h

        if abs(bx - eff_cx) <= half_w and abs(by - eff_cy) <= half_h:
            inside_count += 1

        if abs(bx - eff_cx) <= (half_w - margin) and abs(by - eff_cy) <= (half_h - margin):
            inside_strict += 1

    coverage_pct = 100.0 * inside_count / max(1, num_frames)

    jerk95_raw = _jerk95(np.asarray(ball_cx_raw, dtype=float), fps=fps)
    jerk95_cam = _jerk95(cam_cx_arr, fps=fps)
    stats["jerk95_raw"] = jerk95_raw
    stats["jerk95_cam"] = jerk95_cam
    stats["jerk95"] = jerk95_cam  # backwards compatibility
    stats["ball_in_crop_pct"] = coverage_pct
    stats["ball_in_crop_frames"] = inside_count

    plan_data = {
        "x0": x0.astype(float),
        "y0": y0.astype(float),
        "w": crop_w_arr.astype(float),
        "h": crop_h_arr.astype(float),
        "spd": np.full(
            num_frames,
            float(max(1.0, np.max(np.abs(np.diff(cam_cx_arr, prepend=cam_cx_arr[0]))))),
        ),
        "z": cam_zoom.astype(float),
        "cx": cam_cx_arr.astype(float),
        "cy": cam_cy_arr.astype(float),
    }

    if log_params is not None:
        logger.info(
            "[BALL-CAM] N=%d, cx_range=[%.1f, %.1f], lead_frames=%d, "
            "base_alpha=%.2f, fast_alpha=%.2f, catchup_thresh_px=%.1f, margin=%.1f, "
            "speed_fast_px=%.1f, max_pan_slow=%.1f, max_pan_fast=%.1f, hard_lock=%s",
            len(cam_cx_arr),
            float(np.nanmin(cam_cx_arr)) if cam_cx_arr.size else 0.0,
            float(np.nanmax(cam_cx_arr)) if cam_cx_arr.size else 0.0,
            int(log_params.get("lead_frames", 0)),
            float(log_params.get("base_alpha", 0.0)),
            float(log_params.get("fast_alpha", 0.0)),
            float(log_params.get("catchup_px", 0.0)),
            float(log_params.get("margin", 0.0)),
            float(log_params.get("speed_fast_px", 0.0)),
            float(log_params.get("max_pan_slow", 0.0)),
            float(log_params.get("max_pan_fast", 0.0)),
            False,
        )
    logger.info(
        "[BALL-CAM COVERAGE] ball_in_crop: %.1f%% (%d/%d), strict_with_margin: %.1f%% (%d/%d)",
        coverage_pct,
        inside_count,
        num_frames,
        100.0 * inside_strict / max(1, num_frames),
        inside_strict,
        num_frames,
    )

    debug_overlay = bool(cfg.get("ball_debug_overlay", False))
    if debug_overlay and in_path is not None and out_path is not None:
        scratch_root = scratch_root or _scratch_root()
        debug_dir = scratch_root / "ball_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_out = debug_dir / f"{Path(out_path).stem}__BALL_DEBUG_WIDE.mp4"
        stats["debug_overlay_path"] = str(debug_out)
        write_ball_crop_debug_clip(
            in_path=str(in_path),
            out_path=str(debug_out),
            ball_cx=ball_cx_list,
            ball_cy=ball_cy_list,
            crop_x=x0.tolist(),
            crop_y=y0.tolist(),
            src_w=int(src_w),
            src_h=int(src_h),
            crop_w=int(crop_w),
            crop_h=int(crop_h),
            fps=fps,
            logger=logger,
        )

    return plan_data, stats


class CamFollow2O:
    def __init__(
        self,
        zeta: float = 0.95,
        wn: float = 6.0,
        dt: float = 1 / 30,
        max_vel: Optional[float] = None,
        max_acc: Optional[float] = None,
        deadzone: float = 0.0,
    ) -> None:
        self.z = float(zeta)
        self.w = float(wn)
        self.dt = float(dt)
        self.cx = 0.0
        self.cy = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.max_vel = max_vel
        self.max_acc = max_acc
        self.dead = max(0.0, float(deadzone))

    def _clamp(self, vec: tuple[float, float], limit: Optional[float]) -> tuple[float, float]:
        if limit is None or limit <= 0:
            return vec
        vx, vy = vec
        mag = math.hypot(vx, vy)
        if mag <= limit:
            return vec
        scale = limit / max(mag, 1e-6)
        return vx * scale, vy * scale

    def step(self, target_x: float, target_y: float) -> tuple[float, float]:
        ex = target_x - self.cx
        ey = target_y - self.cy
        if math.hypot(ex, ey) < self.dead:
            ex = ey = 0.0

        dt = self.dt
        w = self.w
        z = self.z
        ax = w * w * ex - 2 * z * w * self.vx
        ay = w * w * ey - 2 * z * w * self.vy
        ax, ay = self._clamp((ax, ay), self.max_acc)
        self.vx += ax * dt
        self.vy += ay * dt
        self.vx, self.vy = self._clamp((self.vx, self.vy), self.max_vel)
        self.cx += self.vx * dt
        self.cy += self.vy * dt
        return self.cx, self.cy

    def damp_velocity(self, factor: float) -> None:
        factor = float(max(0.0, min(1.0, factor)))
        self.vx *= factor
        self.vy *= factor


class FollowHoldController:
    """Stateful helper that gates follow targets during dropouts."""

    def __init__(
        self,
        *,
        dt: float,
        release_frames: int = 3,
        decay_time: float = 0.4,
        initial_target: Optional[Tuple[float, float]] = None,
    ) -> None:
        self.release_frames = max(1, int(release_frames))
        self.valid_streak = self.release_frames
        self.decay_time = max(0.05, float(decay_time))
        self.decay_factor = float(
            math.exp(-dt / self.decay_time) if dt > 0 else 0.0
        )
        self._target = [0.0, 0.0]
        if initial_target is not None:
            self._target[0] = float(initial_target[0])
            self._target[1] = float(initial_target[1])
            self.valid_streak = self.release_frames
        self._initialised = initial_target is not None

    @property
    def target(self) -> Tuple[float, float]:
        return float(self._target[0]), float(self._target[1])

    def apply(
        self, target_x: float, target_y: float, valid: bool
    ) -> Tuple[float, float, bool]:
        if not math.isfinite(target_x) or not math.isfinite(target_y):
            valid = False
        if not self._initialised:
            self._target[0] = float(target_x)
            self._target[1] = float(target_y)
            self._initialised = True
        if valid:
            if self.valid_streak < self.release_frames:
                self.valid_streak += 1
            else:
                self.valid_streak = self.release_frames
            if self.valid_streak >= self.release_frames:
                self._target[0] = float(target_x)
                self._target[1] = float(target_y)
                return float(target_x), float(target_y), False
            return self.target[0], self.target[1], True
        self.valid_streak = 0
        return self.target[0], self.target[1], True

    def reset_target(self, cx: float, cy: float) -> None:
        self._target[0] = float(cx)
        self._target[1] = float(cy)
        self.valid_streak = self.release_frames
        self._initialised = True


def ema_path(xs: Sequence[float], ys: Sequence[float], alpha: float) -> tuple[List[float], List[float]]:
    if alpha <= 0:
        return list(xs), list(ys)
    sx: List[float] = []
    sy: List[float] = []
    for idx in range(len(xs)):
        x_val = float(xs[idx])
        y_val = float(ys[idx])
        if idx == 0:
            sx.append(x_val)
            sy.append(y_val)
            continue
        sx.append(alpha * x_val + (1 - alpha) * sx[-1])
        sy.append(alpha * y_val + (1 - alpha) * sy[-1])
    return sx, sy


# --- Jerk helpers --------------------------------------------------------


def compute_camera_jerk95(xs: Sequence[float], ys: Sequence[float], fps: float) -> float:
    """Return the 95th percentile jerk magnitude for a camera path."""

    if fps <= 0.0 or len(xs) < 4 or len(xs) != len(ys):
        return 0.0

    dt = 1.0 / float(fps)
    arr_x = np.asarray(xs, dtype=np.float64)
    arr_y = np.asarray(ys, dtype=np.float64)

    vx = np.gradient(arr_x, dt, edge_order=2)
    vy = np.gradient(arr_y, dt, edge_order=2)
    ax = np.gradient(vx, dt, edge_order=2)
    ay = np.gradient(vy, dt, edge_order=2)
    jx = np.gradient(ax, dt, edge_order=2)
    jy = np.gradient(ay, dt, edge_order=2)

    jerk_mag = np.hypot(jx, jy)
    jerk_mag = np.nan_to_num(jerk_mag, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.percentile(np.abs(jerk_mag), 95.0))


# --- Simple constant-velocity tracker (EMA-based Kalman-lite) ---

BallPathEntry = Union[Tuple[float, float, float], Tuple[float, float], Mapping[str, float], None]


class CV2DKalman:
    def __init__(self, bx, by):
        self.bx = float(bx)
        self.by = float(by)
        self.vx = 0.0
        self.vy = 0.0
        self.alpha_pos = 0.35
        self.alpha_vel = 0.25

    def predict(self):
        return self.bx + self.vx, self.by + self.vy

    def correct(self, mx, my):
        px, py = self.predict()
        rx, ry = (mx - px), (my - py)
        self.vx += self.alpha_vel * rx
        self.vy += self.alpha_vel * ry
        self.bx = (1 - self.alpha_pos) * px + self.alpha_pos * mx
        self.by = (1 - self.alpha_pos) * py + self.alpha_pos * my
        return self.bx, self.by


# --- Color/shape gating to remove grass and favor white-ish round blobs ---


def pick_ball(d, src_w, src_h):
    bx, by = _get_ball_xy_src(d, src_w, src_h)
    if bx is not None and by is not None:
        return bx, by
    if "ball" in d:
        ball_val = d["ball"]
    return None, None


def _get_ball_xy_src(
    rec: Optional[Union[Mapping[str, object], Sequence[object]]],
    src_w: float,
    src_h: float,
) -> tuple[Optional[float], Optional[float]]:
    """Return best-effort (x, y) ball coordinates in source pixels."""

    if rec is None:
        return None, None

    def _pair_from_mapping(
        mapping: Mapping[str, object],
        key_x: str,
        key_y: str,
    ) -> Optional[tuple[float, float]]:
        if key_x not in mapping or key_y not in mapping:
            return None
        val_x = mapping.get(key_x)
        val_y = mapping.get(key_y)
        return (float(val_x), float(val_y))

    def _pair_from_sequence(seq: Sequence[object]) -> Optional[tuple[float, float]]:
        if len(seq) < 2:
            return None
        return (float(seq[0]), float(seq[1]))

    def _to_src(pair: tuple[float, float]) -> tuple[float, float]:
        x, y = pair
        if src_w > 1 and src_h > 1 and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            return x * float(src_w), y * float(src_h)
        return x, y

    if isinstance(rec, Mapping):
        for key_x, key_y in ("bx_src", "by_src"), ("bx_raw", "by_raw"):
            pair = _pair_from_mapping(rec, key_x, key_y)
            if pair is not None:
                return _to_src(pair)

        ball_seq = rec.get("ball") if isinstance(rec, Mapping) else None
        if isinstance(ball_seq, Sequence):
            pair = _pair_from_sequence(ball_seq)
            if pair is not None:
                return _to_src(pair)

        for key_x, key_y in ("bx_stab", "by_stab"), ("bx", "by"):
            pair = _pair_from_mapping(rec, key_x, key_y)
            if pair is not None:
                return _to_src(pair)

    if isinstance(rec, Sequence) and not isinstance(rec, (str, bytes, bytearray)):
        pair = _pair_from_sequence(rec)
        if pair is not None:
            return _to_src(pair)

    return None, None


def estimate_global_shift(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    *,
    min_shift_px: float = 1.5,
) -> tuple[float, float]:
    """Estimate dominant global translation (camera pan) between two frames.

    Uses phase correlation (FFT-based, fast) to detect the shift.  Returns
    ``(dx, dy)`` in pixels — the translation that aligns *prev* with *curr*.
    Returns ``(0.0, 0.0)`` when the shift is below *min_shift_px* (noise /
    minor camera vibration).
    """
    prev_f = prev_gray.astype(np.float64)
    curr_f = curr_gray.astype(np.float64)
    # phaseCorrelate returns how much curr is shifted relative to prev.
    (dx, dy), _response = cv2.phaseCorrelate(prev_f, curr_f)
    if math.hypot(dx, dy) < min_shift_px:
        return 0.0, 0.0
    return dx, dy


# ------------------------------------------------------------------
# Vertical wobble stabilisation helpers
# ------------------------------------------------------------------

def load_camera_shifts(shifts_path: Path) -> Optional[np.ndarray]:
    """Load pre-computed per-frame camera shifts from a .cam_shifts.npy file.

    Returns an (N, 2) float32 array of (dx, dy) per frame, or *None* if the
    file does not exist or cannot be read.
    """
    if not shifts_path.exists():
        return None
    try:
        arr = np.load(str(shifts_path))
        if arr.ndim == 2 and arr.shape[1] == 2:
            return arr.astype(np.float64)
        return None
    except Exception:
        return None


def compute_camera_shifts(video_path: str, logger: logging.Logger) -> np.ndarray:
    """Fast single-pass computation of per-frame global camera shifts.

    Used as a fallback when the cached .cam_shifts.npy file does not exist
    (e.g. telemetry was cached before this feature was added).
    Returns an (N, 2) float64 array of (dx, dy) per frame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("[STAB] Cannot open video for shift computation: %s", video_path)
        return np.zeros((0, 2), dtype=np.float64)

    shifts = []
    prev_gray = None
    while True:
        ok, bgr = cap.read()
        if not ok or bgr is None:
            break
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            shifts.append((0.0, 0.0))
        else:
            shifts.append(estimate_global_shift(prev_gray, gray))
        prev_gray = gray
    cap.release()

    if shifts:
        logger.info("[STAB] Computed camera shifts for %d frames", len(shifts))
    return np.array(shifts, dtype=np.float64) if shifts else np.zeros((0, 2), dtype=np.float64)


def compute_wobble_corrections(
    shifts: np.ndarray,
    fps: float,
    *,
    smooth_window_s: float = 0.5,
    max_correction_px: float = 20.0,
) -> Optional[np.ndarray]:
    """Derive per-frame vertical wobble corrections from global camera shifts.

    1. Accumulate per-frame (dx, dy) into a cumulative camera trajectory.
    2. Smooth the trajectory with a Gaussian (``smooth_window_s`` seconds)
       to recover the intended camera motion.
    3. The residual (raw − smooth) is the high-frequency wobble.
    4. Clamp to ±``max_correction_px`` and return as an (N,) float64 array
       of vertical corrections (positive = shift frame down to cancel
       upward camera wobble).

    Returns *None* if the wobble is negligible (peak < 1.5 px).
    """
    if shifts is None or len(shifts) < 5:
        return None

    n = len(shifts)

    # Cumulative camera trajectory (vertical only)
    cum_dy = np.cumsum(shifts[:, 1])

    # Smooth with Gaussian to separate intended motion from wobble
    sigma_frames = max(2.0, smooth_window_s * max(fps, 1.0))
    radius = int(sigma_frames * 3.0 + 0.5)
    radius = min(radius, n // 2)
    if radius < 1:
        return None
    x_k = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x_k / sigma_frames) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(cum_dy, radius, mode="edge")
    smooth_dy = np.convolve(padded, kernel, mode="valid")

    # Residual = wobble
    wobble_dy = cum_dy - smooth_dy

    # Check if wobble is meaningful
    peak = float(np.max(np.abs(wobble_dy)))
    if peak < 1.5:
        return None

    # Clamp
    wobble_dy = np.clip(wobble_dy, -max_correction_px, max_correction_px)
    return wobble_dy


# ---------------------------------------------------------------------------
# Portrait framing validation — self-correcting feedback loop
# ---------------------------------------------------------------------------

def validate_portrait_framing(
    frames_dir: Path,
    states: list,
    positions: np.ndarray,
    fusion_source_labels: Optional[np.ndarray],
    source_width: float,
    *,
    sample_every: int = 5,
    min_conf: float = 0.15,
    edge_margin_frac: float = 0.10,
) -> Optional[np.ndarray]:
    """Run YOLO ball detection on rendered portrait frames to validate framing.

    Compares ball detections in the portrait crop against expected source
    positions.  Returns per-frame cx corrections (source pixels) if issues
    are found, or ``None`` if the framing passes validation.

    The function checks two things:
    1. Ball visibility: for frames where the source pipeline has a reliable
       ball position, is the ball actually visible in the portrait crop?
    2. Ball centering: is the ball pushed too close to the crop edge?

    Only frames with reliable source labels (YOLO=1, interp=3, hold=4,5)
    are checked — centroid-only (2) positions are too unreliable.
    """
    try:
        from ultralytics import YOLO as _YOLO
        _val_model = _YOLO("yolov8n.pt")
    except Exception as _e:
        print(f"[VALIDATE] Cannot load YOLO model ({_e}), skipping validation")
        return None

    from tools.ball_telemetry import detect_ball_in_frame

    n = len(states)
    corrections = np.zeros(n, dtype=np.float64)
    _reliable = {1, 3, 4, 5}
    _checked = 0
    _detected = 0
    _missing = 0
    _edge_warn = 0
    _corrected = 0

    for fi in range(0, n, sample_every):
        # Only validate frames with reliable source ball data
        if fi >= len(positions) or np.isnan(positions[fi]).any():
            continue
        if fusion_source_labels is not None and fi < len(fusion_source_labels):
            if int(fusion_source_labels[fi]) not in _reliable:
                continue
        elif fusion_source_labels is not None:
            continue

        frame_path = frames_dir / f"frame_{fi:06d}.png"
        if not frame_path.exists():
            continue

        portrait_frame = cv2.imread(str(frame_path))
        if portrait_frame is None:
            continue

        _checked += 1
        ph, pw = portrait_frame.shape[:2]

        # Run YOLO ball detection on the rendered portrait frame
        det_cx, det_cy, det_conf = detect_ball_in_frame(
            _val_model, portrait_frame, min_conf=min_conf,
        )

        # Expected ball position in portrait coordinates
        ball_sx = float(positions[fi][0])
        x0 = float(states[fi].x0)
        crop_w = float(states[fi].crop_w) if states[fi].crop_w > 1 else 607.0
        scale_x = pw / crop_w if crop_w > 0 else 1.0
        expected_px = (ball_sx - x0) * scale_x

        if det_cx is not None:
            _detected += 1
            # Ball found — check if it's dangerously close to the edge
            margin = pw * edge_margin_frac
            if det_cx < margin or det_cx > pw - margin:
                _edge_warn += 1
                # Mild correction: nudge toward center
                center_px = pw / 2.0
                error_px = det_cx - center_px
                error_src = error_px / scale_x  # convert to source pixels
                corrections[fi] = error_src * 0.5  # partial correction
                _corrected += 1
        else:
            # Ball NOT found in portrait crop
            ball_in_crop = (x0 <= ball_sx <= x0 + crop_w)
            if not ball_in_crop:
                # Ball is geometrically outside the crop — definite error
                _missing += 1
                corrections[fi] = ball_sx - float(states[fi].cx)
                _corrected += 1

    # Interpolate corrections for non-sampled frames
    if _corrected > 0:
        # Fill gaps between sampled corrections using linear interpolation
        _nonzero = np.nonzero(corrections)[0]
        if len(_nonzero) > 1:
            from scipy.interpolate import interp1d
            try:
                _interp = interp1d(
                    _nonzero, corrections[_nonzero],
                    kind="linear", fill_value=0.0, bounds_error=False,
                )
                _all_frames = np.arange(n)
                corrections = _interp(_all_frames)
            except Exception:
                pass  # keep sparse corrections

        # Gaussian smooth the corrections to prevent jitter
        _sigma = max(3.0, n * 0.005)
        _r = int(_sigma * 3.0 + 0.5)
        _r = min(_r, n // 2)
        if _r >= 1:
            _kx = np.arange(-_r, _r + 1, dtype=np.float64)
            _k = np.exp(-0.5 * (_kx / _sigma) ** 2)
            _k /= _k.sum()
            _padded = np.pad(corrections, _r, mode="edge")
            corrections = np.convolve(_padded, _k, mode="valid")

    print(
        f"[VALIDATE] Checked {_checked} portrait frames: "
        f"ball_detected={_detected}, ball_missing={_missing}, "
        f"edge_warnings={_edge_warn}, corrections={_corrected}"
    )

    if _missing == 0 and _edge_warn == 0:
        return None  # validation passed

    return corrections


def apply_framing_corrections(
    states: list,
    corrections: np.ndarray,
    source_width: float,
) -> int:
    """Apply cx corrections to camera states. Returns count of modified frames."""
    applied = 0
    for i in range(min(len(states), len(corrections))):
        if abs(corrections[i]) > 0.5:
            states[i].cx -= corrections[i]
            half_cw = states[i].crop_w / 2.0
            states[i].x0 = float(np.clip(
                states[i].cx - half_cw, 0.0,
                max(0.0, source_width - states[i].crop_w),
            ))
            applied += 1
    return applied


def detect_motion_centroid(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    *,
    prev_cx: float | None = None,
    prev_cy: float | None = None,
    ignore_top_frac: float = 0.30,
    motion_thresh: int = 25,
    blur_ksize: int = 41,
    min_area_frac: float = 0.0005,
    locality_radius: int = 200,
    global_shift: tuple[float, float] | None = None,
) -> tuple:
    """Find (cx, cy, confidence) of the motion centroid between two frames.

    Uses frame differencing + Gaussian heatmap.  When *prev_cx*/*prev_cy* are
    given, the centroid is biased toward the previous position so that the
    tracker follows one coherent action region rather than jumping between
    separate groups of players.

    Returns (None, None, 0.0) if insufficient motion is detected.

    When *global_shift* ``(dx, dy)`` is provided (from :func:`estimate_global_shift`),
    the previous frame is warped to cancel the source-camera pan before
    differencing.  This isolates player / ball motion from XBotGo camera
    rotation so the centroid tracks the action, not the pan direction.
    """
    h, w = curr_gray.shape[:2]

    # --- Source-camera pan compensation ---
    # Warp prev to align with curr so that the frame difference only
    # contains actual object motion (players, ball), not the global shift
    # caused by the physical camera panning / rotating.
    prev_for_diff = prev_gray
    if global_shift is not None:
        dx, dy = global_shift
        if abs(dx) > 0.5 or abs(dy) > 0.5:
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            prev_for_diff = cv2.warpAffine(
                prev_gray, M, (w, h), borderMode=cv2.BORDER_REPLICATE,
            )

    # Absolute frame difference (on pan-compensated pair when available)
    diff = cv2.absdiff(prev_for_diff, curr_gray)

    # Pre-blur to suppress compression artifacts
    diff = cv2.GaussianBlur(diff, (5, 5), 0)

    # Binary motion mask
    _, motion_mask = cv2.threshold(diff, motion_thresh, 255, cv2.THRESH_BINARY)

    # Ignore top region (stands, sky, banners)
    y_start = int(h * ignore_top_frac)
    motion_mask[:y_start, :] = 0

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Require minimum motion area
    motion_pixels = float(np.count_nonzero(motion_mask))
    total_pixels = float(h * w)
    if motion_pixels < total_pixels * min_area_frac:
        return None, None, 0.0

    # Gaussian heatmap to find densest motion cluster
    bk = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    heat = cv2.GaussianBlur(motion_mask.astype(np.float32), (bk, bk), 0)

    # Vertical/depth bias: weight near-field (lower rows) higher than
    # far-field (upper rows) so the centroid prefers foreground action
    # over background activity on adjacent pitches.  The gradient runs
    # from 0.10 at the top to 1.0 at the bottom — a 10x preference for
    # bottom-of-frame action, which aggressively suppresses background
    # ball/player motion on adjacent pitches.
    vert_weight = np.linspace(0.10, 1.0, h, dtype=np.float32).reshape(-1, 1)
    heat *= vert_weight

    peak_val = heat.max()
    if peak_val < 1.0:
        return None, None, 0.0

    # Find global peak location
    _, _, _, max_loc = cv2.minMaxLoc(heat)
    peak_x, peak_y = float(max_loc[0]), float(max_loc[1])

    # If we have a previous position, bias toward it: use a local
    # neighborhood around prev to find the best centroid near the action
    # we were already tracking.  Fall back to global peak if prev is far
    # from any motion.
    center_x, center_y = peak_x, peak_y
    if prev_cx is not None and prev_cy is not None:
        # Check if there's significant motion near previous position
        r = locality_radius
        px, py = int(round(prev_cx)), int(round(prev_cy))
        y0 = max(0, py - r)
        y1 = min(h, py + r)
        x0 = max(0, px - r)
        x1 = min(w, px + r)
        local_heat = heat[y0:y1, x0:x1]
        local_peak = float(local_heat.max()) if local_heat.size > 0 else 0.0

        # If local motion is at least 50% of global peak, track locally.
        # Higher threshold makes it easier to break free from background
        # lock-on: the tracker will jump to the global peak (foreground
        # action) when the local region only has weak residual motion.
        if local_peak >= peak_val * 0.50:
            local_thresh = local_peak * 0.4
            ly, lx = np.where(local_heat >= local_thresh)
            if len(lx) > 0:
                lw = local_heat[ly, lx].astype(np.float64)
                tw = lw.sum()
                if tw > 1e-6:
                    center_x = float(x0 + np.dot(lx.astype(np.float64), lw) / tw)
                    center_y = float(y0 + np.dot(ly.astype(np.float64), lw) / tw)
    else:
        # No previous position: use a small neighborhood around the global
        # peak instead of the full heatmap, to avoid averaging between
        # separate motion regions.
        r = locality_radius
        px, py = int(round(peak_x)), int(round(peak_y))
        y0 = max(0, py - r)
        y1 = min(h, py + r)
        x0 = max(0, px - r)
        x1 = min(w, px + r)
        local_heat = heat[y0:y1, x0:x1]
        local_peak = float(local_heat.max()) if local_heat.size > 0 else 0.0
        if local_peak > 1.0:
            local_thresh = local_peak * 0.4
            ly, lx = np.where(local_heat >= local_thresh)
            if len(lx) > 0:
                lw = local_heat[ly, lx].astype(np.float64)
                tw = lw.sum()
                if tw > 1e-6:
                    center_x = float(x0 + np.dot(lx.astype(np.float64), lw) / tw)
                    center_y = float(y0 + np.dot(ly.astype(np.float64), lw) / tw)

    # Confidence: fraction of frame in motion, capped at 1.0
    conf = min(1.0, motion_pixels / (total_pixels * 0.02))

    return center_x, center_y, conf


def build_ball_mask(bgr, grass_h=(35, 95), min_v=170, max_s=120):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    non_grass = (H < grass_h[0]) | (H > grass_h[1])
    bright = V >= min_v
    low_sat = S <= max_s
    mask = ((non_grass & bright) | (bright & low_sat)).astype(np.uint8) * 255
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask


def _red_mask_hsv(frame_bgr: np.ndarray) -> np.ndarray:
    """Binary mask for red/orange objects (soccer ball) in HSV space."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Red wraps around hue=0, so use two bands.
    lower1 = np.array([0, 120, 70], dtype=np.uint8)
    upper1 = np.array([12, 255, 255], dtype=np.uint8)

    lower2 = np.array([170, 120, 70], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)

    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(m1, m2)

    # Clean up speckle / fill small gaps
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask


def detect_red_ball_xy(
    frame_bgr: np.ndarray,
    prev_xy: tuple[float, float] | None,
    *,
    ignore_top_frac: float = 0.33,
    min_area: float = 20.0,
    max_area: float = 1200.0,
) -> tuple[float, float] | None:
    """
    Returns (x,y) of the most likely red/orange ball in the frame, else None.
    Uses color + region + size + circularity + temporal proximity scoring.
    """

    h, w = frame_bgr.shape[:2]

    mask = _red_mask_hsv(frame_bgr)

    # Ignore sky/tents/stands region (top part of the image)
    y0 = int(h * ignore_top_frac)
    mask[:y0, :] = 0

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    best = None
    best_score = -1e18

    for c in cnts:
        area = float(cv2.contourArea(c))
        if area < min_area or area > max_area:
            continue

        per = float(cv2.arcLength(c, True))
        if per <= 1e-6:
            continue

        # Circularity: 1.0 is perfect circle
        circ = 4.0 * np.pi * area / (per * per)

        x, y, ww, hh = cv2.boundingRect(c)
        aspect = ww / max(1.0, float(hh))
        if aspect < 0.5 or aspect > 1.8:
            continue

        M = cv2.moments(c)
        if abs(M["m00"]) < 1e-6:
            continue
        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])

        # Base score: prefer circular + “not huge”
        score = 0.0
        score += 3.0 * circ
        score += 0.5 * (1.0 - abs(aspect - 1.0))  # closer to 1 is better
        score += -0.001 * area  # slight bias toward smaller

        # Vertical/depth bias: prefer near-field (lower in frame) over
        # far-field (upper-middle).  cy/h ranges from ~0.30 (top cutoff)
        # to 1.0 (bottom); this adds up to +4.0 for bottom-of-frame
        # detections, aggressively discouraging background ball locks.
        score += 4.0 * (cy / h)

        # Temporal: prefer near previous location
        if prev_xy is not None:
            dx = cx - prev_xy[0]
            dy = cy - prev_xy[1]
            d2 = dx * dx + dy * dy
            score += -0.002 * d2

        if score > best_score:
            best_score = score
            best = (cx, cy)

    return best


def detect_ball_xy(
    frame_bgr: np.ndarray,
    prev_xy: tuple[float, float] | None,
    *,
    ignore_top_frac: float = 0.33,
    min_area: float = 20.0,
    max_area: float = 1200.0,
) -> tuple[float, float] | None:
    """General-purpose ball detector: tries red mask first, then build_ball_mask.

    Works for red/orange, white, yellow, and most ball colours against grass.
    """
    # Try red/orange first (higher specificity)
    result = detect_red_ball_xy(
        frame_bgr, prev_xy,
        ignore_top_frac=ignore_top_frac,
        min_area=min_area, max_area=max_area,
    )
    if result is not None:
        return result

    # Fallback: general bright-non-green mask (white/yellow balls)
    h, w = frame_bgr.shape[:2]
    mask = build_ball_mask(frame_bgr)
    y0 = int(h * ignore_top_frac)
    mask[:y0, :] = 0

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    best = None
    best_score = -1e18

    for c in cnts:
        area = float(cv2.contourArea(c))
        if area < min_area or area > max_area:
            continue
        per = float(cv2.arcLength(c, True))
        if per <= 1e-6:
            continue
        circ = 4.0 * np.pi * area / (per * per)
        if circ < 0.55:
            continue
        x_bb, y_bb, ww, hh = cv2.boundingRect(c)
        aspect = ww / max(1.0, float(hh))
        if aspect < 0.5 or aspect > 1.8:
            continue
        M = cv2.moments(c)
        if abs(M["m00"]) < 1e-6:
            continue
        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])

        score = 3.0 * circ + 0.5 * (1.0 - abs(aspect - 1.0)) - 0.001 * area
        # Vertical/depth bias: prefer near-field (lower in frame) — 4x
        # weight aggressively penalizes background ball detections.
        score += 4.0 * (cy / h)
        if prev_xy is not None:
            d2 = (cx - prev_xy[0]) ** 2 + (cy - prev_xy[1]) ** 2
            score += -0.002 * d2
        if score > best_score:
            best_score = score
            best = (cx, cy)

    return best


def write_red_ball_telemetry(
    in_path: Path,
    out_path: Path,
    fps_hint: float | None,
    logger: logging.Logger,
) -> None:
    """Generate motion-centroid telemetry with full-clip lookahead.

    Maps the entire action path before planning the camera, so the camera
    can anticipate where the action is going rather than reacting to it.

    Approach:
      1. Read all frames and compute raw motion per frame.
      2. Forward sweep with locality bias  → tracks action from start.
      3. Backward sweep with locality bias → tracks action from end.
      4. Stitch: use forward path where confident, backward where forward
         lost tracking, blend at overlap regions.
      5. Centered Gaussian smooth with forward bias for anticipation.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        logger.warning("[MOTION] Failed to open video for telemetry: %s", in_path)
        return

    fps_value = float(fps_hint or 0.0)
    if fps_value <= 0:
        fps_value = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"[MOTION] Generating full-clip motion telemetry for {in_path.name} "
          f"({total_frames} frames, {total_frames / fps_value:.1f}s)...")

    # --- Read all frames into grayscale ---
    grays: list[np.ndarray] = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            break
        grays.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))
    cap.release()

    n = len(grays)
    if n < 2:
        logger.warning("[MOTION] Not enough frames (%d) for motion telemetry", n)
        return

    h, w = grays[0].shape[:2]

    # --- Pre-compute source-camera pan compensation ---
    # Detect per-frame global translation caused by XBotGo rotating.
    # Phase correlation is FFT-based and very fast (~1ms per pair).
    # shifts[i] = (dx, dy) to align grays[i-1] with grays[i].
    shifts: list[tuple[float, float]] = [(0.0, 0.0)]  # frame 0 has no shift
    pan_frame_count = 0
    for i in range(1, n):
        shift = estimate_global_shift(grays[i - 1], grays[i])
        shifts.append(shift)
        if shift != (0.0, 0.0):
            pan_frame_count += 1
    if pan_frame_count > 0:
        print(f"[MOTION] Source-camera pan detected on {pan_frame_count}/{n} frames — compensating")

    # Persist per-frame camera shifts so the renderer can stabilise
    # vertical wobble without re-reading the entire video.
    _shifts_path = out_path.with_suffix(".cam_shifts.npy")
    try:
        np.save(str(_shifts_path), np.array(shifts, dtype=np.float32))
    except Exception as _shifts_err:  # non-critical — don't block pipeline
        logger.warning("[MOTION] Failed to save camera shifts: %s", _shifts_err)

    # --- Pass 1: global peak detection per frame (no locality bias) ---
    # This gives us the "ground truth" of where the most motion is.
    global_cx = np.full(n, np.nan, dtype=np.float64)
    global_cy = np.full(n, np.nan, dtype=np.float64)
    global_conf = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        cx, cy, conf = detect_motion_centroid(
            grays[i - 1], grays[i], global_shift=shifts[i],
        )
        if cx is not None:
            global_cx[i] = cx
            global_cy[i] = cy
            global_conf[i] = conf

    detected_global = int(np.sum(np.isfinite(global_cx)))
    print(f"[MOTION] Global detections: {detected_global}/{n} frames")

    # --- Pass 2: forward sweep with locality bias ---
    fwd_cx = np.full(n, np.nan, dtype=np.float64)
    fwd_cy = np.full(n, np.nan, dtype=np.float64)
    fwd_conf = np.zeros(n, dtype=np.float64)
    prev_cx_f: float | None = None
    prev_cy_f: float | None = None
    for i in range(1, n):
        cx, cy, conf = detect_motion_centroid(
            grays[i - 1], grays[i],
            prev_cx=prev_cx_f, prev_cy=prev_cy_f,
            global_shift=shifts[i],
        )
        if cx is not None:
            fwd_cx[i] = cx
            fwd_cy[i] = cy
            fwd_conf[i] = conf
            prev_cx_f = cx
            prev_cy_f = cy

    # --- Pass 3: backward sweep with locality bias ---
    # Reverse the shift sign since frame order is swapped.
    bwd_cx = np.full(n, np.nan, dtype=np.float64)
    bwd_cy = np.full(n, np.nan, dtype=np.float64)
    bwd_conf = np.zeros(n, dtype=np.float64)
    prev_cx_b: float | None = None
    prev_cy_b: float | None = None
    for i in range(n - 1, 0, -1):
        # Negate shift: shifts[i] aligns (i-1)->i, but here we diff i->i-1
        bwd_shift = (-shifts[i][0], -shifts[i][1])
        cx, cy, conf = detect_motion_centroid(
            grays[i], grays[i - 1],
            prev_cx=prev_cx_b, prev_cy=prev_cy_b,
            global_shift=bwd_shift,
        )
        if cx is not None:
            bwd_cx[i] = cx
            bwd_cy[i] = cy
            bwd_conf[i] = conf
            prev_cx_b = cx
            prev_cy_b = cy

    # --- Pass 4: stitch forward + backward paths ---
    # Blend forward and backward sweeps.  Forward is stronger early in
    # the clip (tracks from start), backward is stronger late (tracks
    # from end).  Distance to the global peak is used as a quality
    # signal — closer to the global peak means that sweep is more
    # accurate for that frame.
    stitched_cx = np.full(n, np.nan, dtype=np.float64)
    stitched_cy = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        has_fwd = np.isfinite(fwd_cx[i])
        has_bwd = np.isfinite(bwd_cx[i])
        has_glob = np.isfinite(global_cx[i])

        # Position-based preference: forward early, backward late
        t_frac = i / max(n - 1, 1)  # 0.0 at start, 1.0 at end
        pos_fwd = 1.0 - t_frac
        pos_bwd = t_frac

        if has_fwd and has_bwd and has_glob:
            gx, gy = global_cx[i], global_cy[i]
            dist_fwd = max(math.hypot(fwd_cx[i] - gx, fwd_cy[i] - gy), 1.0)
            dist_bwd = max(math.hypot(bwd_cx[i] - gx, bwd_cy[i] - gy), 1.0)
            # Quality-weighted blend: closer to global peak + positional preference
            w_fwd = (pos_fwd + 0.3) / dist_fwd
            w_bwd = (pos_bwd + 0.3) / dist_bwd
            total_w = w_fwd + w_bwd
            stitched_cx[i] = (fwd_cx[i] * w_fwd + bwd_cx[i] * w_bwd) / total_w
            stitched_cy[i] = (fwd_cy[i] * w_fwd + bwd_cy[i] * w_bwd) / total_w
        elif has_fwd and has_bwd:
            # No global peak — blend by position only
            w_fwd = pos_fwd + 0.3
            w_bwd = pos_bwd + 0.3
            total_w = w_fwd + w_bwd
            stitched_cx[i] = (fwd_cx[i] * w_fwd + bwd_cx[i] * w_bwd) / total_w
            stitched_cy[i] = (fwd_cy[i] * w_fwd + bwd_cy[i] * w_bwd) / total_w
        elif has_fwd:
            stitched_cx[i] = fwd_cx[i]
            stitched_cy[i] = fwd_cy[i]
        elif has_bwd:
            stitched_cx[i] = bwd_cx[i]
            stitched_cy[i] = bwd_cy[i]
        elif has_glob:
            stitched_cx[i] = global_cx[i]
            stitched_cy[i] = global_cy[i]

    # Fill frame 0 from first valid detection
    first_valid = -1
    for i in range(n):
        if np.isfinite(stitched_cx[i]):
            first_valid = i
            break
    if first_valid > 0:
        stitched_cx[:first_valid] = stitched_cx[first_valid]
        stitched_cy[:first_valid] = stitched_cy[first_valid]
    elif first_valid < 0:
        stitched_cx[:] = w / 2.0
        stitched_cy[:] = h * 0.55

    # Forward-fill remaining gaps
    for i in range(1, n):
        if not np.isfinite(stitched_cx[i]):
            stitched_cx[i] = stitched_cx[i - 1]
            stitched_cy[i] = stitched_cy[i - 1]

    # Jump dampening: limit frame-to-frame movement to prevent sudden
    # position changes during chaotic moments (shots, blocks, etc.).
    # Tightened from 12% to 7% to suppress background lock-on jumps.
    max_jump_px = 0.07 * w  # max 7% of frame width per frame
    for i in range(1, n):
        dx = stitched_cx[i] - stitched_cx[i - 1]
        dy = stitched_cy[i] - stitched_cy[i - 1]
        dist = math.hypot(dx, dy)
        if dist > max_jump_px:
            scale = max_jump_px / dist
            stitched_cx[i] = stitched_cx[i - 1] + dx * scale
            stitched_cy[i] = stitched_cy[i - 1] + dy * scale

    stitched_count = int(np.sum(np.isfinite(stitched_cx)))
    print(f"[MOTION] Stitched path: {stitched_count}/{n} frames")

    # --- Pass 5: centered Gaussian smooth with forward bias ---
    # Sigma backward = 0.4s (wide enough to dampen oscillation from
    #   background lock-on without losing responsiveness to real action)
    # Sigma forward  = 0.6s (anticipate where action is going)
    sigma_back = max(5, int(fps_value * 0.4))
    sigma_fwd = max(5, int(fps_value * 0.6))
    window_back = sigma_back * 3
    window_fwd = sigma_fwd * 3

    planned_cx = np.empty(n, dtype=np.float64)
    planned_cy = np.empty(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - window_back)
        hi = min(n, i + window_fwd + 1)
        t = np.arange(lo, hi) - i
        sigma = np.where(t <= 0, float(sigma_back), float(sigma_fwd))
        weights = np.exp(-0.5 * (t / sigma) ** 2)
        wsum = weights.sum()
        if wsum > 1e-12:
            planned_cx[i] = np.dot(stitched_cx[lo:hi], weights) / wsum
            planned_cy[i] = np.dot(stitched_cy[lo:hi], weights) / wsum
        else:
            planned_cx[i] = stitched_cx[i]
            planned_cy[i] = stitched_cy[i]

    # --- Write JSONL ---
    raw_conf_arr = np.maximum(fwd_conf, bwd_conf)
    with out_path.open("w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({
                "frame": i,
                "t": i / fps_value if fps_value else float(i),
                "cx": float(planned_cx[i]),
                "cy": float(planned_cy[i]),
                "conf": float(raw_conf_arr[i]) if raw_conf_arr[i] > 0 else 0.3,
            }) + "\n")

    cx_range = (float(np.nanmin(planned_cx)), float(np.nanmax(planned_cx)))
    print(f"[MOTION] Telemetry complete: {n} frames, "
          f"cx=[{cx_range[0]:.0f}, {cx_range[1]:.0f}], "
          f"lookahead={sigma_fwd / fps_value:.2f}s")


def _circularity(cnt):
    a = cv2.contourArea(cnt)
    p = cv2.arcLength(cnt, True)
    if p <= 0 or a <= 0:
        return 0.0
    return float(4 * math.pi * a / (p * p))


def ncc_score(gray_win, tpl_gray):
    if (
        gray_win.shape[0] < tpl_gray.shape[0]
        or gray_win.shape[1] < tpl_gray.shape[1]
    ):
        return -1.0
    g = cv2.equalizeHist(gray_win)
    t = cv2.equalizeHist(tpl_gray)
    r1 = cv2.matchTemplate(g, t, cv2.TM_CCOEFF_NORMED).max()
    h, w = t.shape[:2]
    tw, th = max(3, int(w * 0.75)), max(3, int(h * 0.75))
    t2 = cv2.resize(t, (tw, th), interpolation=cv2.INTER_AREA)
    r2 = cv2.matchTemplate(g, t2, cv2.TM_CCOEFF_NORMED).max()
    return float(max(r1, r2))


def find_ball_candidate(
    frame_bgr,
    pred_xy,
    tpl=None,
    search_r=260,
    min_r=6,
    max_r=22,
    min_circ=0.58,
):
    H, W = frame_bgr.shape[:2]
    px, py = pred_xy
    x0 = int(max(0, px - search_r))
    y0 = int(max(0, py - search_r))
    x1 = int(min(W, px + search_r))
    y1 = int(min(H, py + search_r))
    if x1 <= x0 + 2 or y1 <= y0 + 2:
        return None
    roi = frame_bgr[y0:y1, x0:x1]
    mask = build_ball_mask(roi)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    best = None
    best_score = -1e9

    for c in cnts:
        a = cv2.contourArea(c)
        if a < (min_r * min_r * 0.6) or a > (max_r * max_r * 3.5):
            continue
        circ = _circularity(c)
        if circ < min_circ:
            continue
        (cx, cy), rad = cv2.minEnclosingCircle(c)
        bx = x0 + cx
        by = y0 + cy
        dist = math.hypot(bx - px, by - py)
        ncc = 0.0
        if tpl is not None:
            side = int(max(16, min(96, rad * 6)))
            sx0 = int(max(0, bx - side // 2))
            sy0 = int(max(0, by - side // 2))
            sx1 = int(min(W, sx0 + side))
            sy1 = int(min(H, sy0 + side))
            win = gray_roi[(sy0 - y0) : (sy1 - y0), (sx0 - x0) : (sx1 - x0)]
            if win.size:
                ncc = ncc_score(win, tpl)
        score = (-0.02 * dist) + (3.0 * circ) + (1.8 * ncc)
        if score > best_score:
            best_score = score
            best = (bx, by, float(circ), float(ncc), float(dist))

    return best


def grab_frame_at_time(path, t_sec, fps_hint=30.0):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, None
    fps = cap.get(cv2.CAP_PROP_FPS) or fps_hint
    idx = max(0, int(round(t_sec * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    return (frame if ok else None), fps


def manual_select_ball(frame_bgr, window="Select ball"):
    # Fallback if selectROI is missing: simple click-to-center
    if not hasattr(cv2, "selectROI"):
        clicked = {"pt": None}

        def _cb(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                clicked["pt"] = (x, y)

        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.imshow(window, frame_bgr)
        cv2.setMouseCallback(window, _cb)
        print("Click the ball; press ENTER to confirm.")
        while True:
            k = cv2.waitKey(20) & 0xFF
            if k in (13, 32):
                break
            if k == 27:
                clicked["pt"] = None
                break
        cv2.destroyWindow(window)
        if clicked["pt"] is None:
            return None
        x, y = clicked["pt"]
        side = 56
        return (int(x - side // 2), int(y - side // 2), side, side)

    # Preferred: drag a rectangle
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    r = cv2.selectROI(window, frame_bgr, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window)
    if r is None or len(r) != 4 or r[2] <= 0 or r[3] <= 0:
        return None
    return tuple(int(v) for v in r)  # (x, y, w, h)


def _clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def _clamp_roi(x, y, w, h, W, H):
    x = _round_i(x)
    y = _round_i(y)
    w = _round_i(w)
    h = _round_i(h)
    x = _clamp(x, 0, max(0, W - 1))
    y = _clamp(y, 0, max(0, H - 1))
    w = _clamp(w, 2, max(2, W - x))
    h = _clamp(h, 2, max(2, H - y))
    return x, y, w, h


def _roi_around_point(bx, by, W, H, side):
    # side may be float â†' force int and keep odd size for better centering
    side_i = max(3, _round_i(side) | 1)
    x = _round_i(bx) - side_i // 2
    y = _round_i(by) - side_i // 2
    return _clamp_roi(x, y, side_i, side_i, W, H)


class ZoomPlanner:
    """
    Speed â†' zoom with hysteresis and slew-rate limiting.
    zoom=1.0 means the crop is exactly target height; >1 zooms in.
    """

    def __init__(
        self,
        z_min=1.0,
        z_max=1.8,
        s_lo=3.0,
        s_hi=18.0,
        hysteresis=0.20,
        max_step=0.03,
    ):
        self.z_min, self.z_max = z_min, z_max
        self.s_lo, self.s_hi = s_lo, s_hi
        self.hysteresis = hysteresis
        self.max_step = max_step

    def plan(self, spd):
        # map speed to raw zoom target
        t = np.clip((spd - self.s_lo) / (self.s_hi - self.s_lo), 0, 1)
        raw = self.z_min + t * (self.z_max - self.z_min)

        # apply hysteresis around last zoom to avoid flicker
        out = np.empty_like(raw)
        z = raw[0]
        out[0] = z
        for i in range(1, len(raw)):
            rz = raw[i]
            band = self.hysteresis * (self.z_max - self.z_min)
            if abs(rz - z) <= band:
                rz = z  # stick
            # limit zoom change per frame
            z += np.clip(rz - z, -self.max_step, self.max_step)
            out[i] = z
        # final mild smoothing
        return smooth_series(out, alpha=0.25)


FPS = 30.0


def plan_camera_from_ball(
    bx,
    by,
    frame_w,
    frame_h,
    target_aspect,
    pan_alpha=0.15,
    lead=0.10,
    bounds_pad=16,
    center_frac=0.5,
    *,
    margin_px_override: Optional[float] = None,
    headroom_frac_override: Optional[float] = None,
    lead_px_override: Optional[float] = None,
    adaptive: bool = False,
    adaptive_config: Optional[dict] = None,
    auto_tune: bool = False,
):
    """Plan a portrait crop using the offline cinematic planner."""

    if frame_w <= 0 or frame_h <= 0:
        raise ValueError("Frame dimensions must be positive")

    fps = FPS if FPS > 0 else 30.0
    crop_aspect = float(target_aspect) if target_aspect > 0 else (9.0 / 16.0)

    # ------------------------------------------------------------------
    # Auto-tune: analyse the ball trajectory and let PlannerConfig pick
    # optimal step/accel/margin/lead values for this specific clip.
    # ------------------------------------------------------------------
    if auto_tune:
        cfg = PlannerConfig.auto_from_telemetry(
            bx, by,
            frame_size=(float(frame_w), float(frame_h)),
            fps=float(fps),
            crop_aspect=crop_aspect,
        )
        # Honour explicit overrides even in auto mode.
        if margin_px_override is not None:
            cfg.margin_px = max(0.0, float(margin_px_override))
        if headroom_frac_override is not None:
            cfg.headroom_frac = float(np.clip(headroom_frac_override, -0.2, 0.4))
        if lead_px_override is not None:
            cfg.lead_px = max(0.0, float(lead_px_override))
    else:
        smooth_window = max(3, int(round((1.0 - float(np.clip(pan_alpha, 0.01, 0.95))) * 12.0)) | 1)
        headroom_frac = 0.5 - float(np.clip(center_frac, 0.0, 1.0))
        default_headroom = max(0.08, min(0.20, headroom_frac))
        lead_px = max(frame_w * 0.05, float(lead) * fps * 40.0)
        max_step_x = max(12.0, frame_w * 0.012)
        max_step_y = max(8.0, frame_h * 0.008)
        passes = 3 if pan_alpha < 0.3 else 2

        margin_value = max(bounds_pad, 90.0)

        adaptive_kwargs: dict = {}
        if adaptive:
            adaptive_kwargs["adaptive"] = True
            if adaptive_config and isinstance(adaptive_config, dict):
                for key in ("adaptive_v_lo", "adaptive_v_hi",
                            "adaptive_margin_scale", "adaptive_lead_scale",
                            "adaptive_step_scale"):
                    if key in adaptive_config:
                        adaptive_kwargs[key] = float(adaptive_config[key])

        cfg = PlannerConfig(
            frame_size=(float(frame_w), float(frame_h)),
            crop_aspect=crop_aspect,
            fps=float(fps),
            margin_px=float(margin_value),
            headroom_frac=float(default_headroom),
            lead_px=float(lead_px),
            smooth_window=int(smooth_window),
            max_step_x=float(max_step_x),
            max_step_y=float(max_step_y),
            accel_limit_x=float(max_step_x * 0.35),
            accel_limit_y=float(max_step_y * 0.35),
            smoothing_passes=passes,
            portrait_pad=float(bounds_pad),
            **adaptive_kwargs,
        )

    planner = OfflinePortraitPlanner(cfg)
    plan = planner.plan(bx, by)

    x0 = plan["x0"].round().astype(int)
    y0 = plan["y0"].round().astype(int)
    w = plan["w"].round().astype(int)
    h = plan["h"].round().astype(int)
    spd = plan["spd"].astype(float)
    z = plan["z"].astype(float)
    return x0, y0, w, h, spd, z


def guarantee_ball_in_crop(
    x0,
    y0,
    cw,
    ch,
    bx,
    by,
    src_w,
    src_h,
    zoom,
    zoom_min,
    zoom_max,
    margin=0.12,
    step_zoom=0.90,
):
    """Adjust the crop so the ball remains inside with a configurable margin."""

    inner_l = x0 + margin * cw
    inner_r = x0 + cw - margin * cw
    inner_t = y0 + margin * ch
    inner_b = y0 + ch - margin * ch

    dx = 0.0
    dy = 0.0
    if bx < inner_l:
        dx = bx - inner_l
    elif bx > inner_r:
        dx = bx - inner_r
    if by < inner_t:
        dy = by - inner_t
    elif by > inner_b:
        dy = by - inner_b

    if dx or dy:
        x0 += dx
        y0 += dy
        x0 = max(0.0, min(x0, src_w - cw))
        y0 = max(0.0, min(y0, src_h - ch))
        inner_l = x0 + margin * cw
        inner_r = x0 + cw - margin * cw
        inner_t = y0 + margin * ch
        inner_b = y0 + ch - margin * ch

    tries = 0
    while (
        (bx < inner_l or bx > inner_r or by < inner_t or by > inner_b)
        and tries < 12
    ):
        new_zoom = max(zoom_min, zoom * step_zoom)
        if abs(new_zoom - zoom) < 1e-6 and tries > 0:
            break
        zoom = new_zoom
        cx = x0 + cw / 2.0
        cy = y0 + ch / 2.0
        cx = 0.7 * bx + 0.3 * cx
        cy = 0.7 * by + 0.3 * cy

        aspect = cw / float(ch) if ch > 0 else 1080.0 / 1920.0
        ch = src_h / float(zoom) if zoom else src_h
        cw = ch * aspect
        if cw > src_w:
            cw = float(src_w)
            ch = cw / aspect if aspect else src_h

        x0 = max(0.0, min(cx - cw / 2.0, src_w - cw))
        y0 = max(0.0, min(cy - ch / 2.0, src_h - ch))

        inner_l = x0 + margin * cw
        inner_r = x0 + cw - margin * cw
        inner_t = y0 + margin * ch
        inner_b = y0 + ch - margin * ch
        tries += 1

    return x0, y0, cw, ch, zoom


import yaml


def to_jsonable(obj):
    """Recursively convert numpy/Path/datetime objects into JSON-serialisable types."""

    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(np, "generic") and isinstance(obj, np.generic):
        return obj.item()
    if hasattr(np, "ndarray") and isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (float, int, bool, str)) or obj is None:
        return obj


def plan_crop_from_ball(
    bx,
    by,
    src_w,
    src_h,
    out_w=1080,
    out_h=1920,
    zoom_min=0.55,
    zoom_max=0.95,
    pad=24,
    state=None,
    center_frac: float = 0.5,
):
    """Return integer (x0,y0,w,h) portrait crop centered on (bx,by) with damped smoothing."""

    if state is None:
        state = {}

    if src_w <= 0 or src_h <= 0:
        return 0, 0, src_w, src_h, state

    if out_w > 0 and out_h > 0:
        portrait_w = float(out_w)
        portrait_h = float(out_h)
    else:
        portrait_w = float(src_w)
        portrait_h = float(src_h)
    if portrait_w <= 0:
        portrait_w = float(src_w)
    if portrait_h <= 0:
        portrait_h = float(src_h)

    if portrait_h <= 0:
        portrait_h = 1.0
    aspect = portrait_w / float(portrait_h) if portrait_h else 1.0
    if aspect <= 0:
        aspect = float(src_w) / float(src_h) if src_h else 1.0


    zoom_value = state.get("zoom", 1.0)
    candidates = [zoom_value]
    zoom_value = next((c for c in reversed(candidates) if c and abs(c) > 1e-6), 1.0)

    if zoom_min > 0 and zoom_max > 0 and zoom_max >= zoom_min:
        zoom_value = max(zoom_min, min(zoom_max, zoom_value))
    elif zoom_min > 0 and (zoom_max <= 0 or zoom_max < zoom_min):
        zoom_value = max(zoom_min, zoom_value)
    elif zoom_max > 0:
        zoom_value = min(zoom_max, zoom_value)
    if abs(zoom_value) <= 1e-6:
        zoom_value = 1.0

    pad_px = 0.0
    pad_frac = 0.0

    shrink = 1.0
    if pad_frac > 0.0:
        shrink = max(0.0, 1.0 - 2.0 * pad_frac)

    max_crop_w = max(1.0, float(src_w) - 2.0 * pad_px)
    max_crop_h = max(1.0, float(src_h) - 2.0 * pad_px)

    max_zoom_candidates = [zoom_value]
    max_zoom = max(max_zoom_candidates) if max_zoom_candidates else 1.0
    if max_zoom <= 1e-6:
        max_zoom = 1.0

    min_crop_h = portrait_h / max_zoom if max_zoom else portrait_h
    if pad_frac > 0.0:
        min_crop_h *= shrink
    min_crop_h = max(1.0, min(min_crop_h, max_crop_h))
    min_crop_w = max(1.0, min(min_crop_h * aspect, max_crop_w))
    if min_crop_w < min_crop_h * aspect:
        min_crop_h = min_crop_w / max(aspect, 1e-6)

    target_crop_h = portrait_h / zoom_value
    if pad_frac > 0.0:
        target_crop_h *= shrink
    target_crop_h = max(min_crop_h, min(target_crop_h, max_crop_h))
    target_crop_w = target_crop_h * aspect
    if target_crop_w > max_crop_w:
        scale = max_crop_w / target_crop_w if target_crop_w > 0 else 1.0
        target_crop_w = max_crop_w
        target_crop_h *= scale
    if target_crop_h < min_crop_h:
        target_crop_h = min_crop_h
        target_crop_w = min(target_crop_h * aspect, max_crop_w)
        if target_crop_w <= 0:
            target_crop_w = min_crop_w
        if target_crop_w > max_crop_w:
            target_crop_w = max_crop_w
            target_crop_h = target_crop_w / max(aspect, 1e-6)

    prev_x = state.get("x", 0.0)
    prev_y = state.get("y", 0.0)
    prev_w = state.get("w", 0.0)
    prev_h = state.get("h", 0.0)

    if prev_w <= 0:
        prev_w = float(target_crop_w)
    if prev_h <= 0:
        prev_h = float(target_crop_h)

    prev_cx = prev_x + prev_w / 2.0
    prev_cy = prev_y + prev_h / 2.0

    bx = float(bx)
    by = float(by)

    alpha_center = 0.20
    alpha_zoom = 0.15

    cx = (1 - alpha_center) * prev_cx + alpha_center * bx
    cy = (1 - alpha_center) * prev_cy + alpha_center * by
    cy = cy + (0.5 - float(np.clip(center_frac, 0.0, 1.0))) * src_h

    cw = (1 - alpha_zoom) * prev_w + alpha_zoom * target_crop_w
    ch = (1 - alpha_zoom) * prev_h + alpha_zoom * target_crop_h

    ch = max(min_crop_h, min(ch, max_crop_h))
    cw = ch * aspect
    if cw > max_crop_w:
        scale = max_crop_w / cw if cw > 0 else 1.0
        cw = max_crop_w
        ch *= scale
    min_crop_w = min_crop_h * aspect
    if cw < min_crop_w:
        cw = min_crop_w
        ch = cw / max(aspect, 1e-6)

    half_w = cw / 2.0
    half_h = ch / 2.0
    min_x = float(pad_px)
    max_x = float(src_w) - float(pad_px) - cw
    min_y = float(pad_px)
    max_y = float(src_h) - float(pad_px) - ch

    x0 = cx - half_w
    y0 = cy - half_h

    if max_x < min_x:
        x0 = float(src_w) / 2.0 - half_w
    else:
        x0 = max(min_x, min(max_x, x0))
    if max_y < min_y:
        y0 = float(src_h) / 2.0 - half_h
    else:
        y0 = max(min_y, min(max_y, y0))

    cx = x0 + half_w
    cy = y0 + half_h

    zoom_actual = float(src_h) / ch if ch else zoom_value
    if zoom_min > 0 and zoom_max > 0 and zoom_max >= zoom_min:
        zoom_actual = max(zoom_min, min(zoom_max, zoom_actual))

    state["x"] = float(x0)
    state["y"] = float(y0)
    state["w"] = float(cw)
    state["h"] = float(ch)
    state["cx"] = float(cx)
    state["cy"] = float(cy)
    state["zoom"] = float(zoom_actual)

    x0_int = int(round(x0))
    y0_int = int(round(y0))
    w_int = max(1, int(round(cw)))
    h_int = max(1, int(round(ch)))

    return x0_int, y0_int, w_int, h_int, state


def compute_portrait_crop(cx, cy, zoom, src_w, src_h, target_aspect, pad):
    # target aspect (w/h)
    if target_aspect and target_aspect > 0:
        t_aspect = float(target_aspect)
    else:
        t_aspect = src_w / float(src_h)

    # derive crop size from zoom while honoring aspect
    crop_h = src_h / float(zoom)
    crop_w = crop_h * t_aspect
    if crop_w > src_w:  # bound if too wide
        crop_w = float(src_w)
        crop_h = crop_w / t_aspect if t_aspect else crop_h

    # pad shrinks the box a bit to keep safety margins around ball
    if pad and pad > 0:
        crop_w *= (1.0 - 2 * pad)
        crop_h *= (1.0 - 2 * pad)

    # clamp center so the crop stays inside the source
    x0 = max(0.0, min(cx - crop_w / 2.0, src_w - crop_w))
    y0 = max(0.0, min(cy - crop_h / 2.0, src_h - crop_h))

    return x0, y0, crop_w, crop_h


def dynamic_zoom(
    prev_zoom,
    bx,
    by,
    x0,
    y0,
    cw,
    ch,
    src_w,
    src_h,
    speed_px,
    target_zoom_min,
    target_zoom_max,
    k_speed_out=0.0006,
    edge_margin=0.14,
    edge_gain=0.08,
    z_rate=0.06,
):
    """Return a smoothed zoom that reacts to ball speed and proximity to edges."""

    z_target = max(
        target_zoom_min,
        min(target_zoom_max, target_zoom_max - k_speed_out * speed_px),
    )

    if cw > 1 and ch > 1:
        dl = (bx - x0) / cw
        dr = (x0 + cw - bx) / cw
        dt = (by - y0) / ch
        db = (y0 + ch - by) / ch
        proximity = max(0.0, edge_margin - min(dl, dr, dt, db)) / max(edge_margin, 1e-6)
        z_target = max(target_zoom_min, z_target - edge_gain * proximity)

    z_next = prev_zoom + (z_target - prev_zoom) * 0.20
    z_next = max(prev_zoom - z_rate, min(prev_zoom + z_rate, z_next))
    return z_next


PRESETS_PATH = Path(__file__).resolve().parent / "render_presets.yaml"
FOLLOW_DEFAULTS = {
    "zeta": 1.10,
    "wn": 3.5,
    "deadzone": 8.0,
    "max_vel": 250.0,
    "max_acc": 1200.0,
    "pre_smooth": 0.35,
    "lookahead": 2,
}

DEFAULT_PRESETS = {
    "cinematic": {
        "fps": 24,
        "portrait": "1080x1920",
        "lookahead": 0,
        "smoothing": 0.18,
        "pad": 0.04,
        "speed_limit": 500,
        "zoom_min": 1.0,
        "zoom_max": 1.5,
        "crf": 17,
        "keyint_factor": 4,
        "post_smooth_sigma": 8.0,
        "follow": {
            "speed_zoom": {
                "enabled": True,
                "v_lo": 2.0,
                "v_hi": 8.0,
                "zoom_lo": 1.30,
                "zoom_hi": 0.72,
            },
        },
    },
    "wide_follow": {
        "fps": 24,
        "portrait": "1080x1920",
        "lookahead": 10,
        "smoothing": 0.35,
        "pad": 0.02,
        "speed_limit": 1400,
        "zoom_min": 1.0,
        "zoom_max": 1.25,
        "crf": 19,
        "keyint_factor": 4,
        "follow": {
            "smoothing": 0.35,
            "lead_time": 0.10,
            "margin_px": 140,
            "zoom_out_max": 1.25,
            "zoom_edge_frac": 0.9,
            "speed_zoom": {
                "enabled": True,
                "v_lo": 2.0,
                "v_hi": 12.0,
                "zoom_lo": 1.20,
                "zoom_hi": 1.00,
            },
            "controller": {
                "zeta": 1.10,
                "wn": 2.20,
                "deadzone": 40,
                "max_vel": 220,
                "max_acc": 2200,
                "pre_smooth": 0.45,
                "lookahead": 4,
            },
        },
    },
    "segment_smooth": {
        "fps": 24,
        "portrait": "1080x1920",
        "lookahead": 10,
        "smoothing": 0.35,
        "pad": 0.02,
        "speed_limit": 1400,
        "zoom_min": 1.0,
        "zoom_max": 1.25,
        "crf": 19,
        "keyint_factor": 4,
        "follow": {
            "smoothing": 0.35,
            "lead_time": 0.10,
            "margin_px": 140,
            "zoom_out_max": 1.25,
            "zoom_edge_frac": 0.9,
            "speed_zoom": {
                "enabled": True,
                "v_lo": 2.0,
                "v_hi": 12.0,
                "zoom_lo": 1.20,
                "zoom_hi": 1.00,
            },
            "controller": {
                "zeta": 1.10,
                "wn": 2.20,
                "deadzone": 40,
                "max_vel": 220,
                "max_acc": 2200,
                "pre_smooth": 0.45,
                "lookahead": 4,
            },
        },
    },
    "gentle": {
        "fps": 30,
        "portrait": "1080x1920",
        "lookahead": 12,
        "smoothing": 0.55,
        "pad": 0.20,
        "speed_limit": 360,
        "zoom_min": 1.0,
        "zoom_max": 1.8,
        "crf": 20,
        "keyint_factor": 4,
    },
    "realzoom": {
        "fps": 30,
        "portrait": "1080x1920",
        "lookahead": 10,
        "smoothing": 0.50,
        "pad": 0.18,
        "speed_limit": 520,
        "zoom_min": 1.0,
        "zoom_max": 2.4,
        "crf": 19,
        "keyint_factor": 4,
    },
}


def ensure_presets_file() -> None:
    """Create the presets file with defaults when missing."""

    if PRESETS_PATH.exists():
        return
    PRESETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PRESETS_PATH.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(DEFAULT_PRESETS, handle, sort_keys=True)


def load_presets() -> dict:
    """Load the preset configuration, creating defaults if required."""

    ensure_presets_file()
    with PRESETS_PATH.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def ffprobe_fps(path):
    """
    Return the FPS of the input video using ffprobe.
    Always returns a float. Errors bubble upward for visibility.
    """
    import subprocess
    import json

    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "json", path
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # If ffprobe fails, raise an informative error
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    data = json.loads(result.stdout)

    if "streams" not in data or not data["streams"]:
        raise ValueError(f"No video stream found in {path}")

    rate = data["streams"][0].get("r_frame_rate", "0/0")

    # Convert "30000/1001" → float
    num, den = rate.split("/")
    num = float(num)
    den = float(den)
    if den == 0:
        raise ValueError(f"Invalid r_frame_rate in ffprobe: {rate}")

    return num / den


def ffprobe_duration(path: Path) -> float:
    """Return the media duration in seconds using ffprobe."""

    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]

    value = result.stdout.strip()
    if not value:
        raise RuntimeError("ffprobe did not return a duration value.")



def parse_portrait(value: Optional[str]) -> Optional[Tuple[int, int]]:
    """Convert a WxH string into integers."""

    if not value:
        return None
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Invalid portrait specification: {value}")
    width = int(parts[0])
    height = int(parts[1])
    if width <= 0 or height <= 0:
        raise ValueError("Portrait dimensions must be positive integers.")
    return width, height


def portrait_config_from_preset(
    value: Optional[Union[str, Mapping[str, object], Sequence[object]]]
) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[float, float]], float]:
    """Extract portrait size, minimum crop, and horizon lock from a preset entry."""

    portrait: Optional[Tuple[int, int]] = None
    min_box: Optional[Tuple[float, float]] = None
    horizon_lock = 0.0

    def _parse_size(size_value: object) -> Optional[Tuple[int, int]]:
        if size_value is None:
            return None
        if isinstance(size_value, str):
            return parse_portrait(size_value)
        if isinstance(size_value, Mapping):
            width = size_value.get("width")
            height = size_value.get("height")
            if width is None or height is None:
                return None
            if int(width) > 0 and int(height) > 0:
                return int(width), int(height)
            return None
        if isinstance(size_value, Sequence):
            seq = list(size_value)
            if len(seq) < 2:
                return None
            w, h = int(seq[0]), int(seq[1])
            if w > 0 and h > 0:
                return w, h
        return None

    if value is None:
        return None, None, 0.0

    if isinstance(value, Mapping):
        size_value = value.get("size")
        if size_value is None:
            size_value = value.get("canvas") or value.get("dimensions")
        portrait = _parse_size(size_value)
        if portrait is None and "width" in value and "height" in value:
            portrait = _parse_size({"width": value.get("width"), "height": value.get("height")})

        min_box_value = value.get("min_box_px") or value.get("min_box")
        if isinstance(min_box_value, Mapping):
            mbw = min_box_value.get("width")
            mbh = min_box_value.get("height")
            if mbw is not None and mbh is not None and float(mbw) > 0 and float(mbh) > 0:
                min_box = (float(mbw), float(mbh))
        elif isinstance(min_box_value, Sequence):
            seq = list(min_box_value)
    
        horizon_value = value.get("horizon_lock")

        if portrait is None:
            inline = value.get("size")
            if isinstance(inline, str):
                portrait = parse_portrait(inline)
    else:
        portrait = _parse_size(value)

    return portrait, min_box, horizon_lock


def find_label_files(stem: str, labels_root: str) -> List[Path]:
    root = Path(labels_root or "out/yolo").expanduser()
    # Match ANY depth .../labels/<stem>_*.txt
    return sorted(Path(p) for p in glob.glob(str(root / "**" / "labels" / f"{stem}_*.txt"), recursive=True))



def _detect_normalized(x: float, y: float, width: int, height: int) -> bool:
    """Return ``True`` when coordinates appear to be normalised."""

    return (0.0 <= x <= 1.0) and (0.0 <= y <= 1.0) and (width > 2 and height > 2)


def load_labels(
    paths: Sequence[Path],
    frame_width: int,
    frame_height: int,
    input_fps: float,
) -> List[Tuple[float, float, float]]:
    """Load label shards and return time-stamped positions in pixel space."""

    pts: List[Tuple[float, float, float]] = []
    fps = float(input_fps) if input_fps else 30.0

    for file_path in paths:
        with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = stripped.replace(",", " ").split()
                if len(parts) < 3:
                    continue

                if _detect_normalized(x_value, y_value, frame_width, frame_height):
                    x_value *= float(frame_width)
                    y_value *= float(frame_height)

                t_value = frame_idx / fps if fps else 0.0
                pts.append((t_value, x_value, y_value))

    pts.sort(key=lambda record: record[0])
    if not pts:
        return []

    import statistics

    dx = [pts[i + 1][1] - pts[i][1] for i in range(len(pts) - 1)]
    dy = [pts[i + 1][2] - pts[i][2] for i in range(len(pts) - 1)]

    def _trim(values: List[float]) -> set[int]:
        if len(values) < 8:
            return set()
        mean_value = statistics.mean(values)
        stdev_value = statistics.pstdev(values) or 1.0
        bad_indices: set[int] = set()
        for idx, value in enumerate(values):
            if abs((value - mean_value) / stdev_value) > 3.0:
                bad_indices.add(idx)
                bad_indices.add(idx + 1)
        return bad_indices

    bad_idx = _trim(dx) | _trim(dy)
    filtered = [record for idx, record in enumerate(pts) if idx not in bad_idx]
    return filtered


def resample_labels_by_time(
    label_pts: Sequence[Tuple[float, float, float]],
    render_fps: float,
    duration_s: float,
) -> List[Tuple[float, float, float]]:
    """Return per-frame (t, x, y) aligned to render frames by time."""

    if not label_pts:
        return []

    import bisect

    ts = [point[0] for point in label_pts]
    xs = [point[1] for point in label_pts]
    ys = [point[2] for point in label_pts]

    out: List[Tuple[float, float, float]] = []
    total_frames = int(round(max(duration_s, 0.0) * float(render_fps)))
    for frame_idx in range(total_frames):
        t_value = frame_idx / float(render_fps) if render_fps else 0.0
        pos = bisect.bisect_left(ts, t_value)
        if pos <= 0:
            x_value, y_value = xs[0], ys[0]
        elif pos >= len(ts):
            x_value, y_value = xs[-1], ys[-1]
        else:
            t0, t1 = ts[pos - 1], ts[pos]
            weight = 0.0 if t1 == t0 else (t_value - t0) / (t1 - t0)
            x_value = xs[pos - 1] * (1.0 - weight) + xs[pos] * weight
            y_value = ys[pos - 1] * (1.0 - weight) + ys[pos] * weight
        out.append((t_value, x_value, y_value))
    return out


def labels_to_positions(
    label_pts: Sequence[Tuple[float, float, float]],
    render_fps: float,
    duration_s: float,
    source_pts: Optional[Sequence[Tuple[float, float, float]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert per-frame label points into arrays for planning."""

    total_frames = int(round(max(duration_s, 0.0) * float(render_fps)))
    if total_frames <= 0:
        empty_positions = np.empty((0, 2), dtype=np.float32)
        empty_used = np.zeros(0, dtype=bool)
        return empty_positions, empty_used

    if not label_pts:
        positions = np.full((total_frames, 2), np.nan, dtype=np.float32)
        used = np.zeros(total_frames, dtype=bool)
        return positions, used

    resampled = list(label_pts)
    if len(resampled) > total_frames:
        resampled = resampled[:total_frames]
    while len(resampled) < total_frames and resampled:
        t_value = len(resampled) / float(render_fps) if render_fps else 0.0
        resampled.append((t_value, resampled[-1][1], resampled[-1][2]))

    positions = np.array([[x, y] for _, x, y in resampled], dtype=np.float32)

    reference = source_pts if source_pts is not None else resampled
    times = [point[0] for point in reference]
    import bisect

    used = np.zeros(len(resampled), dtype=bool)
    if times:
        threshold = 1.5 / float(render_fps) if render_fps else 0.0
        for idx, (t_value, _, _) in enumerate(resampled):
            insert_pos = bisect.bisect_left(times, t_value)
            best = float("inf")
            if insert_pos < len(times):
                best = min(best, abs(times[insert_pos] - t_value))
            if insert_pos > 0:
                best = min(best, abs(times[insert_pos - 1] - t_value))
            if best <= threshold:
                used[idx] = True

    return positions, used


def _positions_range(positions: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    """Return ``(min_x, max_x, min_y, max_y)`` for valid position samples."""

    if positions.size == 0:
        return None
    valid_mask = ~np.isnan(positions).any(axis=1)
    if not np.any(valid_mask):
        return None
    xs = positions[valid_mask, 0]
    ys = positions[valid_mask, 1]
    if xs.size == 0 or ys.size == 0:
        return None
    return (
        float(np.min(xs)),
        float(np.max(xs)),
        float(np.min(ys)),
        float(np.max(ys)),
    )


def rough_motion_path(
    video_path: str, fps: float, duration_s: float, sample_every: int = 2
) -> List[Tuple[float, float, float]]:
    """Estimate a coarse (t, x, y) path from optical flow as a labels fallback."""

    if fps <= 0.0 or duration_s <= 0.0:
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total = int(round(duration_s * fps))
    ok, prev = cap.read()
    if not ok:
        cap.release()
        return []

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    height, width = prev_gray.shape[:2]
    centers: List[Tuple[float, float, float]] = []
    cx, cy = width / 2.0, height / 2.0
    frame_idx = 1

    while len(centers) < total:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % max(1, sample_every) != 0:
            frame_idx += 1
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, 0.5, 2, 15, 3, 5, 1.2, 0
        )
        fx = float(np.median(flow[..., 0]))
        fy = float(np.median(flow[..., 1]))
        cx = float(np.clip(cx + fx, 0, width - 1))
        cy = float(np.clip(cy + fy, 0, height - 1))
        centers.append((len(centers) / float(fps), cx, cy))
        prev_gray = gray
        frame_idx += 1

    cap.release()

    if not centers:
        return []

    times = [t for t, _, _ in centers]
    xs = [x for _, x, _ in centers]
    ys = [y for _, _, y in centers]

    out: List[Tuple[float, float, float]] = []
    for frame in range(total):
        t_value = frame / float(fps)
        pos = np.searchsorted(times, t_value)
        if pos <= 0:
            x_value, y_value = xs[0], ys[0]
        elif pos >= len(times):
            x_value, y_value = xs[-1], ys[-1]
        else:
            t0, t1 = times[pos - 1], times[pos]
            weight = 0.0 if t1 == t0 else (t_value - t0) / (t1 - t0)
            x_value = xs[pos - 1] * (1.0 - weight) + xs[pos] * weight
            y_value = ys[pos - 1] * (1.0 - weight) + ys[pos] * weight
        out.append((t_value, float(x_value), float(y_value)))

    return out


def build_ball_keepinview_path(
    telemetry: list[dict],
    frame_width: int,
    frame_height: int,
    crop_width: int,
    crop_height: int,
    *,
    default_y_frac: float = 0.45,
    margin_frac: float = 0.15,
    smooth_radius: int = 5,
    max_speed_px: float = 80.0,
) -> tuple[list[float], list[float]]:
    """
    Given per-frame ball telemetry, return (center_x, center_y) for each frame
    such that the ball stays inside a crop window of size crop_width x crop_height.

    - `telemetry` is a list of dicts with keys: t, x, y, visible.
    - center_x/center_y must have length == len(telemetry).
    - All returned centers must be finite floats.

    Strategy:
    - For each frame i:
      - If visible and x/y are finite, use that ball point.
      - Else, carry forward the last valid ball center if we have one.
      - If we still don't have any valid ball yet (start of clip),
        use a neutral default center:
          cx = frame_width / 2
          cy = frame_height * default_y_frac
    - After we build raw center_x[], center_y[]:
      - Clamp centers so that the crop window stays fully inside the frame:
        left = cx - crop_width / 2, right = cx + crop_width / 2
        top  = cy - crop_height / 2, bottom = cy + crop_height / 2
        Adjust cx, cy if those bounds go outside [0, frame_width/height].
      - Apply a simple moving-average smoothing (radius = smooth_radius) to
        center_x and center_y to remove jitter.
      - Optionally clamp per-frame motion to max_speed_px in each direction.
    - Return the smoothed center_x, center_y.
    """

    import math

    n_frames = len(telemetry)
    if n_frames <= 0 or crop_width <= 0 or crop_height <= 0 or frame_width <= 0 or frame_height <= 0:
        return [], []

    half_w = float(crop_width) / 2.0
    half_h = float(crop_height) / 2.0
    margin_x = float(crop_width) * float(margin_frac)
    margin_y = float(crop_height) * float(margin_frac)
    default_cx = float(frame_width) / 2.0
    default_cy = float(frame_height) * float(default_y_frac)

    center_x: list[float] = []
    center_y: list[float] = []
    last_valid: tuple[float, float] | None = None

    for rec in telemetry:
        bx = rec.get("x") if isinstance(rec, Mapping) else None
        by = rec.get("y") if isinstance(rec, Mapping) else None
        vis = bool(rec.get("visible")) if isinstance(rec, Mapping) else False

        bx_val = float(bx) if bx is not None else float("nan")
        by_val = float(by) if by is not None else float("nan")

        if vis and math.isfinite(bx_val) and math.isfinite(by_val):
            last_valid = (float(bx_val), float(by_val))

        if last_valid is not None:
            bx_use, by_use = last_valid
        else:
            bx_use, by_use = default_cx, default_cy

        cx_val = float(bx_use)
        cy_val = float(by_use)

        if half_w > margin_x:
            cx_val = max(bx_use - (half_w - margin_x), min(cx_val, bx_use + (half_w - margin_x)))
        if half_h > margin_y:
            cy_val = max(by_use - (half_h - margin_y), min(cy_val, by_use + (half_h - margin_y)))

        cx_val = max(half_w, min(float(frame_width) - half_w, cx_val))
        cy_val = max(half_h, min(float(frame_height) - half_h, cy_val))

        center_x.append(cx_val)
        center_y.append(cy_val)

    def _smooth(values: list[float]) -> list[float]:
        radius = max(0, int(smooth_radius))
        if radius <= 0 or len(values) <= 1:
            return [float(v) for v in values]
        smoothed: list[float] = []
        for idx, _ in enumerate(values):
            lo = max(0, idx - radius)
            hi = min(len(values), idx + radius + 1)
            window = values[lo:hi]
            smoothed.append(float(sum(window) / max(len(window), 1)))
        return smoothed

    center_x = _smooth(center_x)
    center_y = _smooth(center_y)

    max_speed = float(max_speed_px)
    if math.isfinite(max_speed) and max_speed > 0.0:
        for idx in range(1, n_frames):
            dx = center_x[idx] - center_x[idx - 1]
            dy = center_y[idx] - center_y[idx - 1]
            if abs(dx) > max_speed:
                center_x[idx] = center_x[idx - 1] + math.copysign(max_speed, dx)
            if abs(dy) > max_speed:
                center_y[idx] = center_y[idx - 1] + math.copysign(max_speed, dy)
            center_x[idx] = max(half_w, min(float(frame_width) - half_w, center_x[idx]))
            center_y[idx] = max(half_h, min(float(frame_height) - half_h, center_y[idx]))

    return center_x, center_y


@dataclass
class CamState:
    frame: int
    cx: float
    cy: float
    zoom: float
    crop_w: float
    crop_h: float
    x0: float
    y0: float
    used_label: bool
    clamp_flags: List[str]
    ball: Optional[Tuple[float, float]] = None
    zoom_scale: float = 1.0
    keepinview_override: bool = False


class CameraPlanner:
    """Planner that tracks the ball and produces smoothed camera states."""

    def __init__(
        self,
        width: int,
        height: int,
        fps: float,
        lookahead: int,
        smoothing: float,
        pad: float,
        speed_limit: float,
        zoom_min: float,
        zoom_max: float,
        portrait: Optional[Tuple[int, int]] = None,
        *,
        margin_px: float = 0.0,
        lead_frames: int = 0,
        speed_zoom: Optional[Mapping[str, object]] = None,
        min_box: Optional[Tuple[float, float]] = None,
        horizon_lock: float = 0.0,
        emergency_gain: float = 0.6,
        emergency_zoom_max: float = 1.45,
        keepinview_margin_px: Optional[float] = None,
        keepinview_nudge_gain: float = 0.75,
        keepinview_zoom_gain: float = 0.4,
        keepinview_zoom_out_max: float = 1.6,
        center_frac: float = 0.5,
        post_smooth_sigma: float = 0.0,
        event_type: Optional[str] = None,
    ) -> None:
        self.width = float(width)
        self.height = float(height)
        self.fps = float(fps)
        self.lookahead = max(0, int(lookahead))
        self.smoothing = float(np.clip(smoothing, 0.0, 1.0))
        self.pad = max(0.0, min(0.45, float(pad)))
        self.speed_limit = max(0.0, float(speed_limit))
        self.zoom_min = max(0.1, float(zoom_min))
        self.zoom_max = max(self.zoom_min, float(zoom_max))
        self.portrait = portrait

        self.margin_px = max(0.0, float(margin_px))
        self.lead_frames = max(0, int(lead_frames))
        self.emergency_gain = float(np.clip(emergency_gain, 0.0, 1.0))
        self.emergency_zoom_max = max(1.0, float(emergency_zoom_max))

        if keepinview_margin_px is None:
            keepinview_margin_px = self.margin_px
        self.keepinview_margin_px = max(0.0, float(keepinview_margin_px))
        self.keepinview_nudge_gain = max(0.0, float(keepinview_nudge_gain))
        self.keepinview_zoom_gain = max(0.0, float(keepinview_zoom_gain))
        self.keepinview_zoom_out_max = max(1.0, float(keepinview_zoom_out_max))
        # Explicit keep-in-view band edges for any external planner code
        # that wants named attributes instead of the implicit band.
        # These are vertical fractions of the portrait height where we
        # consider the ball "comfortably framed".
        self.keepinview_min_band_frac = 0.15
        self.keepinview_max_band_frac = 0.85
        self._post_smooth_sigma = max(0.0, float(post_smooth_sigma))
        # Confidence-based speed damping floor: at zero confidence the
        # camera moves at this fraction of normal speed.  Event-type
        # handlers can lower this to make centroid-dominated clips calmer.
        self._conf_speed_floor = 0.70  # default: matches original formula

        # Event-type-aware parameter adjustments.
        # Different event types have fundamentally different ball dynamics
        # and require different camera behavior.
        self.event_type = (event_type or "").upper().strip()
        if self.event_type == "GOAL":
            # GOAL: the ball entering the net is the money shot.  Raise
            # speed limit 25% so the camera doesn't lag, tighten keepinview
            # margin 25% so the ball stays well-centered, and increase
            # nudge gain so corrections are snappier.
            self.speed_limit *= 1.25
            self.keepinview_margin_px *= 0.75
            self.keepinview_nudge_gain = min(1.0, self.keepinview_nudge_gain * 1.2)
            self.keepinview_min_band_frac = 0.18  # tighter vertical band
        elif self.event_type == "CROSS":
            # CROSS: fast horizontal ball movement (40-60+ px/frame).
            # Need much higher lateral speed limit and wider keepinview
            # margin during the flight phase.
            self.speed_limit *= 1.40
            self.keepinview_margin_px *= 0.85
            self.keepinview_nudge_gain = min(1.0, self.keepinview_nudge_gain * 1.15)
        elif self.event_type == "FREE_KICK":
            # FREE_KICK: ball is stationary during setup, then one fast
            # kick.  The camera must hold steady during setup (so the kick
            # taker stays in frame) and not chase centroid noise during the
            # aftermath.  Centroid frames (conf ~0.28) should barely move
            # the camera; only confident YOLO-backed positions trigger pans.
            self.keepinview_margin_px *= 1.80   # much wider margin = rare nudging
            self.keepinview_nudge_gain *= 0.45  # very gentle corrections
            self.speed_limit *= 0.45            # camera moves slowly overall
            self._post_smooth_sigma = max(self._post_smooth_sigma, 14.0)
            self._conf_speed_floor = 0.15       # centroid/hold frames -> near-locked camera

        if not math.isfinite(center_frac):
            center_frac = 0.5
        self.center_frac = float(np.clip(center_frac, 0.0, 1.0))
        self.center_bias_px = (0.5 - self.center_frac) * self.height

        render_fps = self.fps if self.fps > 0 else 30.0
        if render_fps <= 0:
            render_fps = 30.0
        self.render_fps = render_fps
        self.speed_norm_px = 24.0 * (render_fps / 24.0)
        self.zoom_slew = 0.04 * (render_fps / 24.0)

        self.min_box_w = 0.0
        self.min_box_h = 0.0
        if min_box is not None:
            if isinstance(min_box, (tuple, list)) and len(min_box) >= 2:
                self.min_box_w = max(0.0, float(min_box[0]))
                self.min_box_h = max(0.0, float(min_box[1]))
            elif isinstance(min_box, Mapping):
                self.min_box_w = max(0.0, float(min_box.get("width", 0)))
                self.min_box_h = max(0.0, float(min_box.get("height", 0)))

        self.horizon_lock = float(np.clip(horizon_lock, 0.0, 1.0))

        self.speed_zoom_config: Optional[dict[str, float]] = None

        base_side = min(self.width, self.height)
        base_side = max(1.0, base_side)
        target_final_side = max(base_side * (1.0 - 2.0 * self.pad), base_side * 0.35)
        shrink_factor = 1.0
        if self.pad > 0.0:
            shrink_factor = max(0.05, 1.0 - 2.0 * self.pad)
        pre_pad_target = target_final_side / shrink_factor
        desired_zoom = base_side / max(pre_pad_target, 1.0)
        self.base_zoom = float(np.clip(desired_zoom, self.zoom_min, self.zoom_max))
        self.edge_zoom_min_scale = 0.75

        # Populate speed_zoom_config from the preset's speed_zoom mapping.
        # The YAML preset provides v_lo, v_hi, zoom_lo, zoom_hi, enabled;
        # base_zoom is injected here from the computed self.base_zoom.
        if speed_zoom and isinstance(speed_zoom, Mapping):
            sz = dict(speed_zoom)
            if sz.get("enabled", True):
                sz.setdefault("base_zoom", self.base_zoom)
                sz.setdefault("v_lo", 2.0)
                sz.setdefault("v_hi", 10.0)
                sz.setdefault("zoom_lo", 1.0)
                sz.setdefault("zoom_hi", 1.0)
                for k in ("base_zoom", "v_lo", "v_hi", "zoom_lo", "zoom_hi"):
                    sz[k] = float(sz[k])
                self.speed_zoom_config = sz

    def plan(
        self,
        positions: np.ndarray,
        used_mask: np.ndarray,
        confidence: Optional[np.ndarray] = None,
        person_boxes: Optional[dict] = None,
    ) -> List[CamState]:
        frame_count = len(positions)
        states: List[CamState] = []

        # Per-frame confidence array (default 1.0 when not provided).
        if confidence is not None and len(confidence) == frame_count:
            frame_confidence = confidence.astype(np.float32)
        else:
            frame_confidence = np.ones(frame_count, dtype=np.float32)

        # Pre-smooth confidence with a gentle EMA to prevent zoom hunting.
        # Without this, rapid alternation between YOLO (conf ≥ 0.4) and
        # centroid-only (conf = 0.3) frames causes the confidence-based zoom
        # to oscillate, producing visible zoom jitter even with zoom_slew.
        #
        # FPS-correct: alpha_corrected = 1 - (1-alpha_base)^(30/fps) keeps
        # the effective smoothing window in seconds constant at any frame rate.
        render_fps = self.render_fps
        _conf_alpha_base = 0.15  # tuned for 30fps: ~6-frame effective window
        _fps_ratio = 30.0 / max(render_fps, 1.0)  # >1 at lower fps, 1.0 at 30fps
        _conf_alpha = 1.0 - (1.0 - _conf_alpha_base) ** _fps_ratio
        for _ci in range(1, frame_count):
            frame_confidence[_ci] = (
                _conf_alpha * frame_confidence[_ci]
                + (1.0 - _conf_alpha) * frame_confidence[_ci - 1]
            )

        # Initialize camera at first valid ball position (not frame center)
        # so the ball is in-frame from frame 0.
        init_cx = self.width / 2.0
        init_cy = self.height * self.center_frac
        for _init_i in range(frame_count):
            if bool(used_mask[_init_i]) and not np.isnan(positions[_init_i]).any():
                init_cx = float(positions[_init_i][0])
                init_cy = float(positions[_init_i][1])
                break
        prev_cx = init_cx
        prev_cy = init_cy
        prev_zoom = self.base_zoom
        px_per_sec_x = self.speed_limit * 1.35
        px_per_sec_y = self.speed_limit * 0.90
        pxpf_x = px_per_sec_x / render_fps if render_fps > 0 else 0.0
        pxpf_y = px_per_sec_y / render_fps if render_fps > 0 else 0.0
        center_alpha = float(np.clip(self.smoothing, 0.0, 1.0))
        if math.isclose(center_alpha, 0.0, abs_tol=1e-6):
            center_alpha = 0.28
        # FPS-correct tracking EMA so camera responds at the same real-time
        # rate regardless of frame rate.
        center_alpha = 1.0 - (1.0 - center_alpha) ** _fps_ratio
        zoom_slew = self.zoom_slew
        prev_target_x = prev_cx
        prev_target_y = prev_cy

        aspect_target = None
        aspect_ratio = self.width / max(self.height, 1e-6)
        if self.portrait:
            aspect_target = float(self.portrait[0]) / float(self.portrait[1])
            aspect_ratio = aspect_target

        def _clamp_axis(prev_value: float, current_value: float, limit: float) -> Tuple[float, bool]:
            if limit <= 0.0:
                if not math.isclose(current_value, prev_value, rel_tol=1e-9, abs_tol=1e-3):
                    return prev_value, True
                return current_value, False
            delta = current_value - prev_value
            if abs(delta) > limit:
                return prev_value + (limit if delta > 0 else -limit), True
            return current_value, False

        def _compute_crop_dimensions(
            zoom_value: float,
        ) -> Tuple[float, float, float]:
            zoom_clamped = float(np.clip(zoom_value, self.zoom_min, self.zoom_max))
            crop_h = self.height / max(zoom_clamped, 1e-6)
            crop_w = crop_h * aspect_ratio
            if crop_w > self.width:
                crop_w = self.width
                crop_h = crop_w / max(aspect_ratio, 1e-6)

            if self.pad > 0.0:
                pad_scale = max(0.0, 1.0 - 2.0 * self.pad)
                crop_w *= pad_scale
                crop_h *= pad_scale

            min_box_w = self.min_box_w
            min_box_h = self.min_box_h
            if min_box_w > 0.0 or min_box_h > 0.0:
                if min_box_w <= 0.0 and min_box_h > 0.0 and aspect_ratio > 0.0:
                    min_box_w = min_box_h * aspect_ratio
                elif min_box_h <= 0.0 and min_box_w > 0.0 and aspect_ratio > 0.0:
                    min_box_h = min_box_w / max(aspect_ratio, 1e-6)
                crop_w = max(crop_w, min_box_w)
                crop_h = max(crop_h, min_box_h)

            crop_w = float(np.clip(crop_w, 1.0, self.width))
            crop_h = float(np.clip(crop_h, 1.0, self.height))
            return zoom_clamped, crop_w, crop_h

        def _compute_crop(
            center_x: float,
            center_y: float,
            zoom_value: float,
        ) -> Tuple[float, float, float, float, float, float, float, bool]:
            zoom_clamped, crop_w, crop_h = _compute_crop_dimensions(zoom_value)
            adjusted_cy = center_y
            if aspect_target:
                adjusted_cy = adjusted_cy + 0.05 * crop_h
                if self.horizon_lock > 0.0:
                    anchor = self.height * self.horizon_lock
                    adjusted_cy = float(
                        (1.0 - self.horizon_lock) * adjusted_cy + self.horizon_lock * anchor
                    )

            desired_x0 = center_x - crop_w / 2.0
            desired_y0 = adjusted_cy - crop_h / 2.0
            max_x0 = max(0.0, self.width - crop_w)
            max_y0 = max(0.0, self.height - crop_h)
            x0 = float(np.clip(desired_x0, 0.0, max_x0))
            y0 = float(np.clip(desired_y0, 0.0, max_y0))
            bounds_clamped = not (
                math.isclose(x0, desired_x0, rel_tol=1e-6, abs_tol=1e-3)
                and math.isclose(y0, desired_y0, rel_tol=1e-6, abs_tol=1e-3)
            )

            actual_cx = x0 + crop_w / 2.0
            actual_cy = y0 + crop_h / 2.0

            return crop_w, crop_h, x0, y0, actual_cx, actual_cy, zoom_clamped, bounds_clamped

        anticipate_dt = float(self.lookahead) / render_fps if render_fps > 0 else 0.0
        keepinview_nudge = float(self.keepinview_nudge_gain)
        keepinview_zoom = float(self.keepinview_zoom_gain)
        keepinview_zoom_cap = float(self.keepinview_zoom_out_max)
        base_keepinview_margin = float(self.keepinview_margin_px)

        # Acceleration-aware zoom: track velocity history to detect
        # rapid direction changes (shots, blocks).  When acceleration
        # is high, zoom out to keep more action visible.
        prev_speed_pf = 0.0
        smooth_speed_pf = 0.0  # EMA-smoothed speed for zoom decisions
        speed_ema_alpha = 1.0 - (1.0 - 0.25) ** _fps_ratio  # fps-corrected ~4-frame window
        accel_zoom_out = 1.0  # multiplier: 1.0 = no change, <1.0 = zoom out
        accel_zoom_decay = 0.92 ** _fps_ratio  # fps-corrected per-frame decay
        accel_threshold_pf = 4.0 * _fps_ratio  # px/frame accel threshold (fps-corrected)
        accel_zoom_strength = 0.22  # max zoom reduction per acceleration event

        prev_bx = prev_cx
        prev_by = prev_cy
        _dz_hold_count = 0  # tracking deadzone: frames where camera held still

        # Flight-phase detector: when the ball sustains high speed over
        # multiple frames (a cross, a long pass, or a shot in flight),
        # boost the speed limit so the camera can actually keep up.
        # We track a short rolling window of per-frame ball deltas.
        _flight_window_size = max(3, int(0.25 * render_fps))  # ~0.25s window
        _flight_delta_history: list[float] = []
        _flight_speed_thresh = self.width * 0.012 * _fps_ratio  # 1.2% of frame/frame
        _flight_boost_active = False
        _avg_delta = 0.0

        # --- Scorer memory state (post-goal snap-back) ---
        # Track the last player closest to the ball during high-speed
        # moments (shots).  After the ball decelerates sharply (goal or
        # save), the camera biases toward the remembered scorer position
        # to catch the celebration.
        _scorer_x: Optional[float] = None
        _scorer_y: Optional[float] = None
        _scorer_frame: int = -999  # frame when scorer was last updated
        # For FREE_KICK: seed scorer from the kick taker so celebration
        # tracking follows the right player, not a defender/goalkeeper.
        _fk_scorer = getattr(self, '_free_kick_scorer_pos', None)
        if _fk_scorer is not None:
            _scorer_x = _fk_scorer[0]
            _scorer_y = _fk_scorer[1]
        _goal_event_frame: int = -999  # frame when goal event was detected
        _goal_snap_active = False
        _goal_snap_duration = 0  # frames since goal event
        _goal_snap_max_frames = 0  # max frames for snap-back (set on detection)
        _prev_smooth_speed = 0.0  # for deceleration detection
        _high_speed_sustained = 0  # count of consecutive high-speed frames
        # Smoothed scorer position (EMA) to prevent jitter from
        # frame-to-frame nearest-neighbor jumps between players.
        _scorer_smooth_x: Optional[float] = None
        _scorer_smooth_y: Optional[float] = None
        _SCORER_EMA_ALPHA = 1.0 - (1.0 - 0.35) ** _fps_ratio  # fps-corrected scorer smoothing
        # Goal origin: where the scorer was when the goal was detected.
        # Used to cap total drift — prevents the tracker from migrating
        # to the coach/sideline when the source camera pans.
        _goal_origin_x: Optional[float] = None
        _goal_origin_y: Optional[float] = None
        _SCORER_MAX_DRIFT = self.width * 0.12  # max 12% of frame from goal origin (tighter to prevent coach)

        # --- Shot pullback state ---
        # During shots, pull the camera back to show the full arc from
        # shooter to goal — like a broadcast operator pulling wide.
        _shot_pullback_active = False
        _shot_origin_x = 0.0
        _shot_origin_y = 0.0
        _shot_pullback_hold = 0
        _SHOT_PULLBACK_RAMP = 5  # frames to ramp to full blend
        _SHOT_PULLBACK_HOLD_MAX = int(render_fps * 0.7)
        _SHOT_PULLBACK_MIN_SPAN = 40.0  # min px span to activate

        for frame_idx in range(frame_count):
            pos = positions[frame_idx]
            has_position = bool(used_mask[frame_idx]) and not np.isnan(pos).any()

            # Ball position as pan target.
            # Positions are pre-smoothed by the bidirectional EMA filter
            # applied after fusion (before CameraPlanner construction).
            # The EMA below (center_alpha) provides additional tracking
            # smoothness; speed limit and keepinview guards constrain
            # large motions.
            if has_position:
                target = pos.copy()
            else:
                # Hold camera at previous position when tracking is lost.
                # Do NOT drift toward frame center — that creates visible
                # fighting between the ball-tracking path and the center
                # of the landscape frame.
                target = np.array([prev_cx, prev_cy], dtype=np.float32)

            bx_used = float(target[0])
            by_used = float(target[1])

            if has_position:
                target_center_y = float(
                    np.clip(by_used + self.center_bias_px, 0.0, self.height)
                )
            else:
                target_center_y = float(np.clip(by_used, 0.0, self.height))

            clamp_flags: List[str] = []

            raw_speed_pf = math.hypot(bx_used - prev_target_x, target_center_y - prev_target_y)
            # Smooth speed with EMA to prevent zoom/accel oscillation from
            # residual position noise (YOLO-centroid fusion artifacts).
            smooth_speed_pf = speed_ema_alpha * raw_speed_pf + (1.0 - speed_ema_alpha) * smooth_speed_pf
            speed_pf = smooth_speed_pf

            # --- Acceleration detection ---
            accel_pf = abs(speed_pf - prev_speed_pf)
            if accel_pf > accel_threshold_pf:
                # High acceleration: zoom out proportionally
                accel_frac = min(1.0, accel_pf / (accel_threshold_pf * 4.0))
                accel_zoom_out = min(accel_zoom_out, 1.0 - accel_zoom_strength * accel_frac)
                clamp_flags.append(f"accel_zoom={accel_zoom_out:.3f}")
            else:
                # Decay back toward 1.0 (no zoom-out)
                accel_zoom_out = accel_zoom_out + (1.0 - accel_zoom_out) * (1.0 - accel_zoom_decay)
            accel_zoom_out = float(np.clip(accel_zoom_out, 0.75, 1.0))
            prev_speed_pf = speed_pf

            # --- SCORER MEMORY: track shooter during high-speed ball ---
            # During shots/passes, remember the closest player to the ball.
            # When the ball rapidly decelerates (goal, save, out of play),
            # the camera can snap back to the scorer for the reaction.
            _shot_speed_thr = 6.0 * _fps_ratio  # px/frame: fps-corrected (same real-world speed at any fps)
            if smooth_speed_pf > _shot_speed_thr:
                _high_speed_sustained += 1
                # Update scorer position during shot.
                # FREE_KICK: track nearest person to the KICKER (not the ball)
                # so the celebration tracker follows the kick taker, not a
                # defender or goalkeeper near the ball's flight path.
                if _fk_scorer is not None and person_boxes:
                    _shot_persons = person_boxes.get(frame_idx)
                    if _shot_persons:
                        _ref_x = _scorer_x if _scorer_x is not None else _fk_scorer[0]
                        _ref_y = _scorer_y if _scorer_y is not None else _fk_scorer[1]
                        _best_p = None
                        _best_p_dist = float("inf")
                        for _sp in _shot_persons:
                            _sp_dist = math.hypot(_sp.cx - _ref_x, _sp.cy - _ref_y)
                            if _sp_dist < _best_p_dist and _sp_dist < self.width * 0.10:
                                _best_p_dist = _sp_dist
                                _best_p = _sp
                        if _best_p is not None:
                            _scorer_x = _best_p.cx
                            _scorer_y = _best_p.cy
                            _scorer_frame = frame_idx
                elif person_boxes and has_position:
                    # Default: nearest person to ball during shot
                    _shot_persons = person_boxes.get(frame_idx)
                    if _shot_persons:
                        _best_p = None
                        _best_p_dist = float("inf")
                        for _sp in _shot_persons:
                            _sp_dist = math.hypot(_sp.cx - bx_used, _sp.cy - by_used)
                            if _sp_dist < _best_p_dist and _sp_dist < self.width * 0.15:
                                _best_p_dist = _sp_dist
                                _best_p = _sp
                        if _best_p is not None:
                            _scorer_x = _best_p.cx
                            _scorer_y = _best_p.cy
                            _scorer_frame = frame_idx
            else:
                # Detect goal/deceleration event: ball was fast for >=5 frames,
                # now suddenly slow.  This triggers the snap-back.
                if _high_speed_sustained >= 5 and _scorer_x is not None:
                    _decel = _prev_smooth_speed - smooth_speed_pf
                    if _decel > 3.0 * _fps_ratio:  # rapid deceleration (fps-corrected)
                        _goal_event_frame = frame_idx
                        _goal_snap_active = True
                        _goal_snap_duration = 0
                        # Snap-back lasts 2.0s — captures initial celebration
                        # without following all the way to the kickoff restart.
                        _goal_snap_max_frames = int(render_fps * 2.0)
                        # Remember goal origin for drift capping — prevents
                        # the tracker from migrating to the coach/sideline.
                        _goal_origin_x = _scorer_x
                        _goal_origin_y = _scorer_y
                        _scorer_smooth_x = _scorer_x
                        _scorer_smooth_y = _scorer_y
                        clamp_flags.append("goal_event")
                _high_speed_sustained = 0
            _prev_smooth_speed = smooth_speed_pf

            # Advance goal-snap timer
            if _goal_snap_active:
                _goal_snap_duration += 1
                if _goal_snap_duration > _goal_snap_max_frames:
                    _goal_snap_active = False

            # --- Dynamic scorer tracking during celebration ---
            # The scorer runs, slides, and gets mobbed by teammates.
            # Follow them frame-by-frame using nearest-neighbor matching
            # to the last-known scorer position so the camera tracks
            # the celebration instead of staring at a fixed point.
            #
            # Stabilisation:
            #  1) Tighter matching radius (5% of width) prevents jumping
            #     to the coach when the source camera pans to the sideline.
            #  2) EMA smoothing on the scorer position eliminates jitter
            #     from frame-to-frame nearest-neighbor jumps.
            #  3) Drift cap: the smoothed scorer position cannot migrate
            #     more than 20% of frame width from the goal-event origin,
            #     keeping the camera on the celebration, not the bench.
            if _goal_snap_active and _scorer_x is not None and person_boxes:
                _celeb_persons = person_boxes.get(frame_idx)
                if _celeb_persons:
                    _celeb_best = None
                    _celeb_best_dist = float("inf")
                    # Tightened to 0.03 to prevent latching onto the coach
                    # or ball-boy when the source camera pans to sideline.
                    _celeb_max_dist = self.width * 0.03
                    for _cp in _celeb_persons:
                        _cp_dist = math.hypot(
                            _cp.cx - _scorer_x, _cp.cy - _scorer_y,
                        )
                        if _cp_dist < _celeb_best_dist and _cp_dist < _celeb_max_dist:
                            _celeb_best_dist = _cp_dist
                            _celeb_best = _cp
                    if _celeb_best is not None:
                        _scorer_x = _celeb_best.cx
                        _scorer_y = _celeb_best.cy

                # EMA-smooth the scorer position to prevent jitter from
                # frame-to-frame jumps between different detected persons.
                if _scorer_smooth_x is not None and _scorer_x is not None:
                    _scorer_smooth_x += _SCORER_EMA_ALPHA * (_scorer_x - _scorer_smooth_x)
                    _scorer_smooth_y += _SCORER_EMA_ALPHA * (_scorer_y - _scorer_smooth_y)
                elif _scorer_x is not None:
                    _scorer_smooth_x = _scorer_x
                    _scorer_smooth_y = _scorer_y

                # Drift cap: clamp smoothed position within max drift of
                # the goal-event origin.  Prevents migrating to the coach.
                if _scorer_smooth_x is not None and _goal_origin_x is not None:
                    _drift = math.hypot(
                        _scorer_smooth_x - _goal_origin_x,
                        _scorer_smooth_y - _goal_origin_y,
                    )
                    if _drift > _SCORER_MAX_DRIFT:
                        _drift_scale = _SCORER_MAX_DRIFT / _drift
                        _scorer_smooth_x = _goal_origin_x + (_scorer_smooth_x - _goal_origin_x) * _drift_scale
                        _scorer_smooth_y = _goal_origin_y + (_scorer_smooth_y - _goal_origin_y) * _drift_scale

            # Save pre-pullback ball position for zoom-to-fit later.
            _pre_pb_bx = bx_used
            _pre_pb_by = target_center_y

            # --- SHOT PULLBACK: widen frame during shots ---
            # During sustained high-speed ball flight (shot / long pass),
            # pull the camera target back toward the midpoint between
            # where the shot originated and where the ball is now.  This
            # captures the full action arc — run-up → strike → ball
            # hitting the net — instead of chasing the ball to the goal
            # and losing the shooter.
            if _high_speed_sustained == 1:
                # Ball just entered high speed — record origin
                _shot_origin_x = bx_used
                _shot_origin_y = target_center_y

            if _high_speed_sustained >= 3 and not _shot_pullback_active:
                # Confirmed shot (3+ frames of sustained high speed)
                _shot_pullback_active = True
                _shot_pullback_hold = 0

            if _shot_pullback_active:
                _shot_span = math.hypot(
                    bx_used - _shot_origin_x,
                    target_center_y - _shot_origin_y,
                )
                _still_fast = smooth_speed_pf > _shot_speed_thr * 0.5

                if _still_fast:
                    _shot_pullback_hold = 0
                else:
                    _shot_pullback_hold += 1
                    if _shot_pullback_hold > _SHOT_PULLBACK_HOLD_MAX:
                        _shot_pullback_active = False

                if _shot_pullback_active and _shot_span > _SHOT_PULLBACK_MIN_SPAN:
                    # Blend strength: ramp up during flight, fade during hold
                    if _still_fast:
                        _pb_blend = min(
                            1.0,
                            max(0, _high_speed_sustained - 3) / _SHOT_PULLBACK_RAMP,
                        )
                    else:
                        _pb_blend = max(
                            0.0,
                            1.0 - _shot_pullback_hold / max(_SHOT_PULLBACK_HOLD_MAX, 1),
                        )

                    # Midpoint between shot origin and current ball
                    _mid_x = (_shot_origin_x + bx_used) * 0.5
                    _mid_y = (_shot_origin_y + target_center_y) * 0.5

                    # Blend: 55% ball (keep tracking it) + 45% midpoint (show arc)
                    _pb_w = _pb_blend * 0.45
                    bx_used = bx_used * (1.0 - _pb_w) + _mid_x * _pb_w
                    target_center_y = target_center_y * (1.0 - _pb_w) + _mid_y * _pb_w
                    clamp_flags.append(f"shot_pb={_pb_blend:.2f}")

            if self.speed_zoom_config:
                config = self.speed_zoom_config
                v_lo = config["v_lo"]
                v_hi = config["v_hi"]
                if v_hi <= v_lo:
                    norm = 1.0 if speed_pf >= v_hi else 0.0
                else:
                    norm = float(np.clip((speed_pf - v_lo) / max(v_hi - v_lo, 1e-6), 0.0, 1.0))
                zoom_factor = config["zoom_lo"] + (config["zoom_hi"] - config["zoom_lo"]) * norm
                zoom_target = float(config["base_zoom"] * zoom_factor)
                zoom_target = float(np.clip(zoom_target, self.zoom_min, self.zoom_max))
            else:
                speed_norm_px = self.speed_norm_px
                norm = 0.0
                if speed_norm_px > 1e-6:
                    norm = min(1.0, speed_pf / speed_norm_px)
                zoom_target = self.zoom_min + (self.zoom_max - self.zoom_min) * (1.0 - norm)

            # Apply acceleration zoom-out on top of speed zoom
            zoom_target = float(np.clip(zoom_target * accel_zoom_out, self.zoom_min, self.zoom_max))

            # --- BALL-FLIGHT COMMITMENT ---
            # During shots and passes the ball moves fast but YOLO
            # confidence often dips (motion blur, small ball).  Without
            # this guard the confidence-based zoom widens the frame and
            # the camera loses commitment to the ball flight — the
            # rendered clip cuts away just as the ball reaches the goal.
            #
            # When ball speed is high, we boost effective confidence so
            # the camera stays committed to the trajectory.  The speed
            # threshold is tuned to trigger on shots/passes (~6 px/frame)
            # but not on normal dribbling (~2-3 px/frame).
            #
            # GUARD: Only apply flight commitment when the underlying
            # data has real YOLO confidence (>= 0.35).  Centroid and
            # interpolated frames have conf <= 0.30; their apparent
            # "speed" comes from position noise, not actual ball motion.
            # Boosting confidence on these frames prevents the
            # confidence-based zoom-out, keeping a tight crop on a
            # position that may be completely wrong.
            frame_conf = float(frame_confidence[frame_idx])
            _raw_conf = float(frame_confidence[frame_idx])  # before any boosting
            _flight_speed_thr = 5.0  # px/frame: above this = shot/pass in flight
            # Allow flight commitment on velocity-extrapolated frames (conf ~0.32)
            # and interpolated frames (conf ~0.28) during shots, not just YOLO
            # frames.  The previous threshold of 0.35 prevented commitment on
            # centroid/interpolated frames whose apparent speed came from the
            # extrapolation — which IS real ball motion, not position noise.
            _flight_min_conf = 0.27
            if smooth_speed_pf > _flight_speed_thr and _raw_conf >= _flight_min_conf:
                # Boost confidence floor proportional to speed
                _flight_frac = min(1.0, (smooth_speed_pf - _flight_speed_thr) / max(_flight_speed_thr, 1e-6))
                _flight_conf_floor = 0.50 + 0.30 * _flight_frac  # 0.50 at threshold, 0.80 at 2x threshold
                if frame_conf < _flight_conf_floor:
                    frame_conf = _flight_conf_floor
                    clamp_flags.append(f"flight_commit={_flight_frac:.2f}")

            if frame_conf < 0.60:
                # Map confidence 0..0.60 → zoom scale 0.45..1.0
                # Low-confidence frames (centroid/interp) zoom out more aggressively
                # to keep the ball visible when position is uncertain.  The lower
                # floor (0.45 vs 0.60) gives ~25% wider FOV during sparse-YOLO
                # stretches where centroid may track player clusters.
                conf_scale = 0.45 + 0.917 * frame_conf  # 0.45 at conf=0, 1.0 at conf=0.60
                zoom_target = float(np.clip(
                    zoom_target * conf_scale, self.zoom_min, self.zoom_max
                ))
                clamp_flags.append(f"conf_zoom={conf_scale:.2f}")

            # --- Player-aware zoom ceiling with convergence awareness ---
            # When persons are detected near the ball, cap the zoom so the
            # crop is wide enough to keep both the ball and nearby players
            # in frame.  During fast ball motion (passes, shots) the search
            # radius expands to capture players running onto the ball.
            # Players moving TOWARD the ball get extra weight so the camera
            # anticipates who will interact next.
            #
            # Context radius and padding are scaled to the *visible crop
            # width* (at zoom=1.0), not the full source width.  For portrait
            # 9:16 the crop is only ~33-56% as wide as the source, so
            # source-relative sizing would capture players well outside the
            # frame and force zoom to zoom_min on nearly every frame.
            if person_boxes and has_position:
                _frame_persons = person_boxes.get(frame_idx)
                if _frame_persons:
                    # Base crop width at zoom=1.0 (widest possible portrait crop)
                    _base_crop_w = self.height * aspect_ratio if aspect_ratio > 0 else self.width

                    # Dynamic context radius: expands during ball flight
                    # to capture goal area and nearby defenders during shots.
                    _ctx_frac = 0.50  # 50% of crop width at calm
                    _ctx_flight_boost = 0.0
                    if smooth_speed_pf > _flight_speed_thr:
                        _ctx_flight_boost = min(0.25, 0.25 * (smooth_speed_pf - _flight_speed_thr) / max(_flight_speed_thr, 1e-6))
                    _ctx_radius = _base_crop_w * (_ctx_frac + _ctx_flight_boost)
                    _ctx_pad = _base_crop_w * 0.10

                    # Find persons within the context radius
                    _nearby = []
                    for _p in _frame_persons:
                        _pdist = math.hypot(_p.cx - bx_used, _p.cy - by_used)
                        if _pdist >= _ctx_radius:
                            continue

                        # Convergence check: is this person moving toward the ball?
                        # Compare with the same person's previous-frame position
                        # (nearest-neighbor match in previous frame's detections).
                        _convergence_bonus = 0.0
                        if frame_idx > 0 and person_boxes:
                            _prev_persons = person_boxes.get(frame_idx - 1)
                            if _prev_persons:
                                # Find closest person in previous frame (simple nearest-neighbor)
                                _best_prev_dist = float("inf")
                                _best_prev = None
                                for _pp in _prev_persons:
                                    _pp_dist = math.hypot(_pp.cx - _p.cx, _pp.cy - _p.cy)
                                    if _pp_dist < _best_prev_dist and _pp_dist < self.width * 0.05:
                                        _best_prev_dist = _pp_dist
                                        _best_prev = _pp
                                if _best_prev is not None:
                                    # Person velocity
                                    _pvx = _p.cx - _best_prev.cx
                                    _pvy = _p.cy - _best_prev.cy
                                    # Direction from person to ball
                                    _to_ball_x = bx_used - _p.cx
                                    _to_ball_y = by_used - _p.cy
                                    _to_ball_mag = math.hypot(_to_ball_x, _to_ball_y)
                                    if _to_ball_mag > 1e-6:
                                        # Dot product: positive = moving toward ball
                                        _dot = (_pvx * _to_ball_x + _pvy * _to_ball_y) / _to_ball_mag
                                        if _dot > 1.0:  # moving toward ball at >1px/frame
                                            _convergence_bonus = min(1.0, _dot / 5.0)  # 0→1 over 1-5 px/frame

                        # Persons converging on the ball are included even if
                        # slightly outside the base radius (up to 1.3x).
                        _effective_radius = _ctx_radius * (1.0 + 0.3 * _convergence_bonus)
                        if _pdist < _effective_radius:
                            _nearby.append(_p)

                    if _nearby:
                        # Cap to 4 nearest players to prevent the bounding
                        # box from spanning the entire team and forcing zoom
                        # to zoom_min on every frame.
                        if len(_nearby) > 4:
                            _nearby.sort(key=lambda _pp: math.hypot(_pp.cx - bx_used, _pp.cy - by_used))
                            _nearby = _nearby[:4]
                        _all_x = [bx_used] + [p.cx for p in _nearby]
                        _all_y = [by_used] + [p.cy for p in _nearby]
                        _bbox_w = max(_all_x) - min(_all_x) + _ctx_pad * 2.0
                        _bbox_h = max(_all_y) - min(_all_y) + _ctx_pad * 2.0
                        # Convert action box to minimum crop dimensions
                        if aspect_ratio > 0:
                            _needed_crop_w = max(_bbox_w, _bbox_h * aspect_ratio)
                            _needed_crop_h = _needed_crop_w / max(aspect_ratio, 1e-6)
                        else:
                            _needed_crop_w = _bbox_w
                            _needed_crop_h = _bbox_h
                        # Zoom ceiling: can't zoom tighter than needed for action box
                        if _needed_crop_h > 1.0:
                            _zoom_ceiling = self.height / _needed_crop_h
                            if _zoom_ceiling < zoom_target:
                                zoom_target = max(self.zoom_min, _zoom_ceiling)
                                clamp_flags.append(f"person_ctx={len(_nearby)}")

            # --- Shot pullback zoom-to-fit ---
            # During pullback, cap zoom so both the shot origin and
            # the current ball position fit inside the crop.  Without
            # this the position blend helps but the zoom might still
            # be too tight to show both the shooter and the goal.
            if _shot_pullback_active:
                _pb_fit_crop_w = self.height * aspect_ratio if aspect_ratio > 0 else self.width
                _pb_span_x = abs(_pre_pb_bx - _shot_origin_x)
                _pb_span_y = abs(_pre_pb_by - _shot_origin_y)
                if _pb_span_x > _SHOT_PULLBACK_MIN_SPAN * 0.5 or _pb_span_y > _SHOT_PULLBACK_MIN_SPAN * 0.5:
                    _pb_pad = _pb_fit_crop_w * 0.15
                    _pb_box_w = _pb_span_x + _pb_pad * 2.0
                    _pb_box_h = _pb_span_y + _pb_pad * 2.0
                    if aspect_ratio > 0:
                        _pb_needed_w = max(_pb_box_w, _pb_box_h * aspect_ratio)
                        _pb_needed_h = _pb_needed_w / max(aspect_ratio, 1e-6)
                    else:
                        _pb_needed_h = _pb_box_h
                    if _pb_needed_h > 1.0:
                        _pb_zoom_ceil = self.height / _pb_needed_h
                        if _pb_zoom_ceil < zoom_target:
                            zoom_target = max(self.zoom_min, _pb_zoom_ceil)
                            clamp_flags.append("pb_zfit")

            # --- Celebration zoom-out: keep frame wide after goal ---
            # After a goal the ball is stationary → speed-zoom wants to
            # zoom in tight.  Override: push zoom toward zoom_min so the
            # camera stays wide enough to capture the celebration.
            if _goal_snap_active:
                _celeb_zoom_wide = self.zoom_min * 1.05
                _celeb_zramp = min(
                    1.0,
                    float(_goal_snap_duration) / max(int(render_fps * 0.4), 1),
                )
                # Blend zoom_target 60% toward wide framing
                zoom_target = (
                    zoom_target * (1.0 - _celeb_zramp * 0.60)
                    + _celeb_zoom_wide * (_celeb_zramp * 0.60)
                )
                zoom_target = float(np.clip(zoom_target, self.zoom_min, self.zoom_max))

            # Dynamic zoom slew: allow faster zoom transitions during
            # high-speed events (shots, goals) so the frame opens up
            # quickly enough to capture the full action.
            _zoom_slew_eff = zoom_slew
            if smooth_speed_pf > _flight_speed_thr:
                _slew_boost = min(2.5, 1.0 + 1.5 * (smooth_speed_pf - _flight_speed_thr) / max(_flight_speed_thr, 1e-6))
                _zoom_slew_eff = zoom_slew * _slew_boost
            elif _goal_snap_active:
                # Fast slew during celebration — zoom needs to widen
                # quickly even though ball speed is low.
                _zoom_slew_eff = zoom_slew * 2.0
            zoom_step = float(np.clip(zoom_target - prev_zoom, -_zoom_slew_eff, _zoom_slew_eff))
            zoom = float(np.clip(prev_zoom + zoom_step, self.zoom_min, self.zoom_max))

            # --- POST-GOAL: FOLLOW SCORER THROUGH CELEBRATION ---
            # After a goal event, the scorer is the subject — not the
            # ball (sitting in the net).  Blend camera target toward the
            # scorer with a fade-in / hold / fade-out envelope.  The
            # scorer position is updated dynamically each frame (see
            # "Dynamic scorer tracking" above) so the camera follows the
            # celebration run, slide, and team mob.
            if _goal_snap_active and _scorer_smooth_x is not None:
                _snap_ramp_frames = int(render_fps * 0.3)
                _snap_hold_end = _goal_snap_max_frames - int(render_fps * 1.0)
                if _goal_snap_duration < _snap_ramp_frames:
                    _snap_blend = float(_goal_snap_duration) / max(_snap_ramp_frames, 1)
                elif _goal_snap_duration < _snap_hold_end:
                    _snap_blend = 1.0
                else:
                    _snap_fade = float(_goal_snap_duration - _snap_hold_end) / max(
                        _goal_snap_max_frames - _snap_hold_end, 1
                    )
                    _snap_blend = max(0.0, 1.0 - _snap_fade)
                # Scorer gets up to 40% weight — enough to catch celebration
                # but not so much that mis-identified persons (coach, ball-boy)
                # pull the camera off the action.
                _snap_w = _snap_blend * 0.40
                # Use EMA-smoothed + drift-capped scorer position instead
                # of raw nearest-neighbor position.
                bx_used = bx_used * (1.0 - _snap_w) + _scorer_smooth_x * _snap_w
                target_center_y = target_center_y * (1.0 - _snap_w) + _scorer_smooth_y * _snap_w
                if _snap_w > 0.01:
                    clamp_flags.append(f"celeb={_snap_w:.2f}")

            vx = (bx_used - prev_bx) * render_fps if render_fps > 0 else 0.0
            vy = (target_center_y - prev_by) * render_fps if render_fps > 0 else 0.0

            # --- VELOCITY-BASED PAN LEAD (anticipation) ---
            # Broadcast cameras lead the action: they point slightly ahead of
            # the ball in its direction of travel so the viewer sees where the
            # play is going, not just where it's been.  The lead scales with
            # ball speed — faster motion gets more anticipation — and is
            # suppressed when speed is low (to avoid noise-driven drift).
            #
            # lead_time_sec controls how far ahead the camera looks.
            # The post-Gaussian smoothing will round this into an S-curve.
            _lead_time_sec = 0.25  # seconds of anticipation (raised from 0.20)
            _lead_speed_floor = 3.0  # px/frame: no lead below this speed (raised from 1.5 — YOLO/centroid noise on stationary balls produces ~1-2 px/frame jitter)
            _lead_max_px = self.width * 0.12  # cap lead at 12% of frame width (raised from 8%)
            if smooth_speed_pf > _lead_speed_floor:
                # Velocity direction unit vector
                _vel_mag = math.hypot(vx, vy)
                if _vel_mag > 1e-6:
                    _lead_px_x = (vx / _vel_mag) * min(smooth_speed_pf * render_fps * _lead_time_sec, _lead_max_px)
                    _lead_px_y = (vy / _vel_mag) * min(smooth_speed_pf * render_fps * _lead_time_sec, _lead_max_px) * 0.3
                else:
                    _lead_px_x = 0.0
                    _lead_px_y = 0.0
                # Scale lead by speed ramp (0 at floor, full at 2x floor)
                _lead_ramp = min(1.0, (smooth_speed_pf - _lead_speed_floor) / max(_lead_speed_floor, 1e-6))
                _lead_px_x *= _lead_ramp
                _lead_px_y *= _lead_ramp
            else:
                _lead_px_x = 0.0
                _lead_px_y = 0.0

            # Apply lead to the tracking target (before EMA smoothing).
            # The ball itself stays at bx_used for keepinview purposes,
            # but the camera aims at the led position.
            _led_bx = float(np.clip(bx_used + _lead_px_x, 0.0, self.width))
            _led_by = float(np.clip(target_center_y + _lead_px_y, 0.0, self.height))

            # --- FOLLOW: EMA smoothing with tracking deadzone ---
            # Broadcast cameras hold still when the action is centered,
            # only panning when the subject drifts significantly.  The
            # deadzone suppresses micro-drift from the EMA always
            # chasing tiny offsets — creating natural pauses between
            # pan movements instead of perpetual continuous motion.
            #
            # Two complementary deadzone checks:
            #  (a) Position: target is close to camera → hold
            #  (b) Velocity: target is barely moving → hold
            # The position check alone fails because EMA lag (alpha=0.15)
            # keeps the target ~26px from the camera at typical speeds,
            # exceeding the 18px deadzone radius.  The velocity check
            # catches cases where the ball is nearly stationary but the
            # camera hasn't caught up — preventing perpetual drift.
            #
            # Inside deadzone:  camera holds (alpha=0)
            # Ramp zone:        alpha linearly ramps to full
            # Beyond ramp zone: full tracking at center_alpha
            _, _dz_crop_w, _ = _compute_crop_dimensions(zoom)
            _dz_radius = _dz_crop_w * 0.065  # 6.5% of crop width — wider deadzone for broadcast-style pauses
            _dz_dist = math.hypot(_led_bx - prev_cx, _led_by - prev_cy)
            _target_delta = math.hypot(_led_bx - prev_target_x, _led_by - prev_target_y)
            # Position-based thresholds (original)
            _pos_hold = _dz_dist < _dz_radius
            _pos_ramp = _dz_dist < _dz_radius * 3.0
            # Velocity-based thresholds: hold when target moves < 15% of
            # deadzone radius per frame; ramp when < 50%.  Scale by fps
            # ratio so the same real-world speed triggers the same behavior.
            _vel_hold = _target_delta < _dz_radius * 0.15 * _fps_ratio
            _vel_ramp = _target_delta < _dz_radius * 0.50 * _fps_ratio
            if _pos_hold or _vel_hold:
                follow_alpha = 0.0
                _dz_hold_count += 1
            elif _pos_ramp or _vel_ramp:
                # Use the more restrictive (smaller) ramp value
                if _pos_ramp and _vel_ramp:
                    _ramp_pos = (_dz_dist - _dz_radius) / max(_dz_radius * 2.0, 1.0)
                    _ramp_vel = (_target_delta - _dz_radius * 0.15) / max(_dz_radius * 0.35, 1.0)
                    _dz_ramp = min(_ramp_pos, _ramp_vel)
                elif _pos_ramp:
                    _dz_ramp = (_dz_dist - _dz_radius) / max(_dz_radius * 2.0, 1.0)
                elif _vel_ramp:
                    _dz_ramp = (_target_delta - _dz_radius * 0.15) / max(_dz_radius * 0.35, 1.0)
                else:
                    _dz_ramp = 1.0
                _dz_ramp = float(np.clip(_dz_ramp, 0.0, 1.0))
                follow_alpha = center_alpha * _dz_ramp
            else:
                follow_alpha = center_alpha
            cx_smooth = follow_alpha * _led_bx + (1.0 - follow_alpha) * prev_cx
            cy_smooth = follow_alpha * _led_by + (1.0 - follow_alpha) * prev_cy

            # 4) KEEP-IN-VIEW GUARD (DOMINANT)
            # Uses the actual ball position (not the led position) so the
            # ball itself is guaranteed to stay in the crop.
            kv_zoom, kv_crop_w, kv_crop_h = _compute_crop_dimensions(zoom)
            kv_zoom = kv_zoom  # kept for clarity; zoom already clamped
            guard_frac = max(0.0, float(self.keepinview_min_band_frac))
            margin_x = max(base_keepinview_margin, kv_crop_w * guard_frac)
            margin_y = max(base_keepinview_margin, kv_crop_h * guard_frac)

            dx = abs(bx_used - cx_smooth)
            dy = abs(target_center_y - cy_smooth)

            # Proportional keep-in-view: instead of a hard binary switch,
            # ramp nudge strength based on how far outside the margin the
            # ball has drifted.  This avoids the visible jolt that occurs
            # when the override toggles on/off between consecutive frames.
            dx_excess = max(0.0, dx - margin_x) / max(margin_x, 1.0)
            dy_excess = max(0.0, dy - margin_y) / max(margin_y, 1.0)
            # Square the excess to create a gradual ease-in curve instead
            # of a linear ramp.  Small excesses produce negligible nudge
            # (0.1^2 = 0.01) while large excesses still trigger strong
            # recentering (0.8^2 = 0.64).  This eliminates the visible
            # snap when the ball crosses the margin boundary.
            excess_frac = min(1.0, max(dx_excess, dy_excess))
            excess_frac = excess_frac * excess_frac  # quadratic ramp

            keepinview_override = excess_frac > 0.0
            if keepinview_override:
                # Blend between smooth follow (center_alpha) and hard
                # recenter (keepinview_nudge) proportionally to how far
                # outside the safety band the ball is.
                adjusted_nudge = follow_alpha + (keepinview_nudge - follow_alpha) * excess_frac
                cx = prev_cx + adjusted_nudge * (bx_used - prev_cx)
                cy = prev_cy + adjusted_nudge * (target_center_y - prev_cy)

                clamp_flags.append(f"keepin_prop={excess_frac:.2f}")
            else:
                # Safe: allow anticipation + smoothing
                cx = cx_smooth
                cy = cy_smooth

            # If keep-in-view triggered strongly, suppress anticipation
            if keepinview_override and excess_frac > 0.3:
                vx = 0.0
                vy = 0.0

            ball_point: Optional[Tuple[float, float]] = None
            if has_position:
                ball_point = (float(pos[0]), float(pos[1]))
            edge_zoom_scale = 1.0


            keepinview_zoom_out = 1.0
            if keepinview_override and keepinview_zoom > 0.0:
                keepinview_zoom_out = min(1.0 + keepinview_zoom, keepinview_zoom_cap)

            if keepinview_zoom_out > 1.0:
                keepin_scale = 1.0 / keepinview_zoom_out
                zoom = max(self.zoom_min, zoom * keepin_scale)
                edge_zoom_scale *= keepin_scale
                if not any(flag.startswith("keepin_zoom=") for flag in clamp_flags):
                    clamp_flags.append(f"keepin_zoom={keepinview_zoom_out:.3f}")

            # Boost pan speed limit during high acceleration so camera
            # can follow fast action instead of falling behind.
            accel_speed_boost = 1.0 + 0.2 * (1.0 - accel_zoom_out) / 0.25  # up to 1.1x
            accel_speed_boost = min(1.1, accel_speed_boost)

            # Camera-pan-aware speed boost: when the ball's position jumps
            # significantly between frames (physical camera pan/tilt rather
            # than ball movement), temporarily allow faster virtual panning
            # so the crop keeps up with the physical camera.
            _ball_delta_pf = math.hypot(bx_used - prev_bx, target_center_y - prev_by)
            _pan_detect_thresh = self.width * 0.015 * _fps_ratio  # 1.5% of frame width/frame (fps-corrected)
            if _ball_delta_pf > _pan_detect_thresh:
                _pan_boost = min(2.5, 1.0 + (_ball_delta_pf / _pan_detect_thresh - 1.0) * 0.8)
                accel_speed_boost = max(accel_speed_boost, _pan_boost)

            # Flight-phase speed boost: when the ball sustains high speed
            # over a short window (~0.25s), boost the speed limit so the
            # camera can track fast crosses/shots instead of lagging behind.
            _flight_delta_history.append(_ball_delta_pf)
            if len(_flight_delta_history) > _flight_window_size:
                _flight_delta_history.pop(0)
            if len(_flight_delta_history) >= _flight_window_size:
                _avg_delta = sum(_flight_delta_history) / len(_flight_delta_history)
                _flight_boost_active = _avg_delta > _flight_speed_thresh
            if _flight_boost_active:
                _flight_boost = min(2.0, 1.0 + (_avg_delta / _flight_speed_thresh - 1.0) * 0.6)
                accel_speed_boost = max(accel_speed_boost, _flight_boost)
                if not any(f.startswith("flight_boost") for f in clamp_flags):
                    clamp_flags.append(f"flight_boost={_flight_boost:.2f}")

            # Keep-in-view speed boost: when the ball is drifting toward
            # the crop edge, raise the speed cap proportionally so the
            # safety correction isn't immediately clamped away.  The boost
            # is steeper than linear so that moderate excess (0.3-0.5)
            # already provides meaningful speed headroom, preventing the
            # destructive cycle of keepinview→clamp→keepinview jitter.
            if keepinview_override and excess_frac > 0.0:
                _keepin_boost = 1.0 + 3.5 * excess_frac  # up to 4.5x at full excess
                accel_speed_boost = max(accel_speed_boost, _keepin_boost)

            # Confidence-based speed damping: when the ball position is
            # uncertain (centroid-only, conf ~0.22-0.30), reduce the pan speed
            # limit so the camera doesn't chase noise across the field.
            # At full confidence (>=0.60) no damping; at zero confidence the
            # camera moves at _conf_speed_floor (default 0.70, lower for
            # event types like FREE_KICK where centroid data is unreliable).
            # Completely bypass confidence damping when keepinview is active
            # with meaningful excess — the camera must reach the ball even
            # if confidence is low, otherwise keepinview and speed-limit
            # fight each other every frame.
            _conf_speed_scale = 1.0
            _csf = self._conf_speed_floor
            if frame_conf < 0.60 and not (keepinview_override and excess_frac > 0.02):
                _conf_speed_scale = _csf + (1.0 - _csf) * (frame_conf / 0.60)
                clamp_flags.append(f"conf_speed={_conf_speed_scale:.2f}")

            _eff_speed_boost = accel_speed_boost * _conf_speed_scale
            cx, x_clamped = _clamp_axis(prev_cx, cx, pxpf_x * _eff_speed_boost)
            cy, y_clamped = _clamp_axis(prev_cy, cy, pxpf_y * _eff_speed_boost)
            speed_limited = x_clamped or y_clamped
            if speed_limited:
                clamp_flags.append("speed")

            _, est_crop_w, est_crop_h = _compute_crop_dimensions(zoom)
            bx_margin: Optional[float]
            by_margin: Optional[float]
            if ball_point:
                bx_margin, by_margin = ball_point
            else:
                bx_margin, by_margin = float(target[0]), float(target[1])
            zoom_scale = edge_aware_zoom(
                cx,
                cy,
                bx_margin,
                by_margin,
                est_crop_w,
                est_crop_h,
                self.width,
                self.height,
                self.margin_px,
                s_min=self.edge_zoom_min_scale,
            )
            if zoom_scale < 0.999:
                edge_zoom_scale *= zoom_scale
                zoom = max(self.zoom_min, zoom * zoom_scale)
                clamp_flags.append(f"edge_zoom={zoom_scale:.3f}")

            # --- emergency keep-in-view ---
            if math.isfinite(bx_margin) and math.isfinite(by_margin):
                margin = float(self.margin_px)
                _, crop_w_est, crop_h_est = _compute_crop_dimensions(zoom)
                if crop_w_est > 0.0 and crop_h_est > 0.0:
                    bx = float(bx_margin)
                    by = float(by_margin)
                    crop_w = float(crop_w_est)
                    crop_h = float(crop_h_est)

                    em_gain = self.emergency_gain if hasattr(self, "emergency_gain") else 0.6
                    em_zoom = self.emergency_zoom_max if hasattr(self, "emergency_zoom_max") else 1.45
                    em_gain = float(np.clip(em_gain, 0.0, 1.0))

                    halfW = crop_w * 0.5
                    halfH = crop_h * 0.5

                    dxL = bx - (cx - halfW + margin)
                    dxR = (cx + halfW - margin) - bx
                    dyT = by - (cy - halfH + margin)
                    dyB = (cy + halfH - margin) - by

                    need_dx = 0.0
                    if dxL < 0.0:
                        need_dx = -(margin - (bx - (cx - halfW)))
                    elif dxR < 0.0:
                        need_dx = margin - ((cx + halfW) - bx)

                    need_dy = 0.0
                    if dyT < 0.0:
                        need_dy = -(margin - (by - (cy - halfH)))
                    elif dyB < 0.0:
                        need_dy = margin - ((cy + halfH) - by)

                    if need_dx or need_dy:
                        cx += em_gain * need_dx
                        cy += em_gain * need_dy

                    left_d = bx - (cx - halfW + margin)
                    right_d = (cx + halfW - margin) - bx
                    top_d = by - (cy - halfH + margin)
                    bot_d = (cy + halfH - margin) - by
                    tight = min(left_d, right_d, top_d, bot_d)

                    if tight < 2.0:
                        req_halfW = max(bx - (cx - margin), (cx + margin) - bx)
                        req_halfH = max(by - (cy - margin), (cy + margin) - by)
                        needW = max(crop_w, 2.0 * req_halfW)
                        needH = max(crop_h, 2.0 * req_halfH)
                        zoom_out = max(1.0, min(em_zoom, max(needW / max(crop_w, 1e-6), needH / max(crop_h, 1e-6))))
                    else:
                        zoom_out = 1.0

                    if zoom_out > 1.0:
                        zoom = float(np.clip(zoom / zoom_out, self.zoom_min, self.zoom_max))
                        edge_zoom_scale *= 1.0 / zoom_out

            crop_w, crop_h, x0, y0, actual_cx, actual_cy, zoom, bounds_clamped = _compute_crop(
                cx, cy, zoom
            )

            if ball_point and crop_w > 1.0 and crop_h > 1.0:
                bx, by = ball_point
                dist_left = (bx - x0) / crop_w
                dist_right = (x0 + crop_w - bx) / crop_w
                dist_top = (by - y0) / crop_h
                dist_bot = (y0 + crop_h - by) / crop_h

                edge_thr = 0.12
                zoomout_gain = 0.10

                edge_risk = min(dist_left, dist_right, dist_top, dist_bot)
                if edge_risk < edge_thr:
                    zoom = max(self.zoom_min, zoom * (1.0 - zoomout_gain))
                    crop_w, crop_h, x0, y0, actual_cx, actual_cy, zoom, bounds_again = _compute_crop(
                        cx, cy, zoom
                    )
                    bounds_clamped = bounds_clamped or bounds_again

                max_x0 = max(0.0, self.width - crop_w)
                max_y0 = max(0.0, self.height - crop_h)

                margin_px = self.margin_px
                if margin_px > 0.0:
                    desired_x0 = x0
                    desired_y0 = y0
                    if bx < x0 + margin_px:
                        desired_x0 = max(0.0, min(max_x0, bx - margin_px))
                    elif bx > x0 + crop_w - margin_px:
                        desired_x0 = max(0.0, min(max_x0, bx + margin_px - crop_w))
                    if by < y0 + margin_px:
                        desired_y0 = max(0.0, min(max_y0, by - margin_px))
                    elif by > y0 + crop_h - margin_px:
                        desired_y0 = max(0.0, min(max_y0, by + margin_px - crop_h))
                    if not (
                        math.isclose(desired_x0, x0, rel_tol=1e-6, abs_tol=1e-3)
                        and math.isclose(desired_y0, y0, rel_tol=1e-6, abs_tol=1e-3)
                    ):
                        x0 = desired_x0
                        y0 = desired_y0
                        bounds_clamped = True

                def _rounded_bounds(x_start: float, y_start: float) -> Tuple[int, int, int, int]:
                    x1_i = int(round(x_start))
                    y1_i = int(round(y_start))
                    x2_i = int(round(min(x_start + crop_w, self.width)))
                    y2_i = int(round(min(y_start + crop_h, self.height)))
                    return x1_i, y1_i, x2_i, y2_i

                for _ in range(3):
                    x1_i, y1_i, x2_i, y2_i = _rounded_bounds(x0, y0)
                    moved = False
                    if bx < x1_i:
                        shift = x1_i - bx
                        new_x0 = max(0.0, min(max_x0, x0 - shift))
                        moved = moved or not math.isclose(new_x0, x0, rel_tol=1e-6, abs_tol=1e-3)
                        x0 = new_x0
                    elif bx > x2_i - 1:
                        shift = bx - (x2_i - 1)
                        new_x0 = max(0.0, min(max_x0, x0 + shift))
                        moved = moved or not math.isclose(new_x0, x0, rel_tol=1e-6, abs_tol=1e-3)
                        x0 = new_x0

                    if by < y1_i:
                        shift = y1_i - by
                        new_y0 = max(0.0, min(max_y0, y0 - shift))
                        moved = moved or not math.isclose(new_y0, y0, rel_tol=1e-6, abs_tol=1e-3)
                        y0 = new_y0
                    elif by > y2_i - 1:
                        shift = by - (y2_i - 1)
                        new_y0 = max(0.0, min(max_y0, y0 + shift))
                        moved = moved or not math.isclose(new_y0, y0, rel_tol=1e-6, abs_tol=1e-3)
                        y0 = new_y0

                    if not moved:
                        break

                actual_cx = x0 + crop_w / 2.0
                actual_cy = y0 + crop_h / 2.0
                if (
                    math.isclose(x0, 0.0, rel_tol=1e-6, abs_tol=1e-3)
                    or math.isclose(y0, 0.0, rel_tol=1e-6, abs_tol=1e-3)
                    or math.isclose(x0, max_x0, rel_tol=1e-6, abs_tol=1e-3)
                    or math.isclose(y0, max_y0, rel_tol=1e-6, abs_tol=1e-3)
                ):
                    bounds_clamped = True

            if bounds_clamped:
                clamp_flags.append("bounds")

            # --- FINAL SPEED CLAMP ---
            # Emergency keep-in-view and margin adjustments above can shift
            # actual_cx/actual_cy well beyond the speed limit, creating
            # visible frame-to-frame jitter ("rapid left-right glitching").
            # Re-apply the speed limit as a final gate so the rendered
            # camera motion never exceeds the configured maximum pan speed.
            actual_cx, _fsc_x = _clamp_axis(prev_cx, actual_cx, pxpf_x * accel_speed_boost)
            actual_cy, _fsc_y = _clamp_axis(prev_cy, actual_cy, pxpf_y * accel_speed_boost)
            if _fsc_x or _fsc_y:
                # Recompute x0/y0 from clamped center for state consistency.
                _max_x0 = max(0.0, self.width - crop_w)
                _max_y0 = max(0.0, self.height - crop_h)
                x0 = float(np.clip(actual_cx - crop_w / 2.0, 0.0, _max_x0))
                y0 = float(np.clip(actual_cy - crop_h / 2.0, 0.0, _max_y0))
                actual_cx = x0 + crop_w / 2.0
                actual_cy = y0 + crop_h / 2.0
                clamp_flags.append("final_speed")

            prev_cx = actual_cx
            prev_cy = actual_cy
            prev_zoom = zoom
            prev_target_x = bx_used
            prev_target_y = target_center_y
            prev_bx = bx_used
            prev_by = target_center_y

            states.append(
                CamState(
                    frame=frame_idx,
                    cx=actual_cx,
                    cy=actual_cy,
                    zoom=zoom,
                    crop_w=crop_w,
                    crop_h=crop_h,
                    x0=x0,
                    y0=y0,
                    used_label=bool(has_position),
                    clamp_flags=clamp_flags,
                    ball=ball_point,
                    zoom_scale=edge_zoom_scale,
                    keepinview_override=keepinview_override,
                )
            )

        self._deadzone_hold_frames = _dz_hold_count

        # --- POST-PLAN GAUSSIAN SMOOTHING ---
        # The forward EMA + speed clamping above produces discrete stepped
        # motion that looks jerky, especially on a cinematic preset.  A
        # Gaussian post-filter converts those hard steps into smooth curves
        # while preserving the overall planned trajectory.
        #
        # sigma_frames is proportional to the fps so the physical smoothing
        # window stays constant regardless of frame rate.
        sigma_frames = getattr(self, "_post_smooth_sigma", 0.0)
        if sigma_frames >= 0.5 and len(states) >= 5:
            cx_arr = np.array([s.cx for s in states], dtype=np.float64)
            cy_arr = np.array([s.cy for s in states], dtype=np.float64)
            zoom_arr = np.array([s.zoom for s in states], dtype=np.float64)

            # Build a 1-D Gaussian kernel.
            radius = int(sigma_frames * 3.0 + 0.5)
            radius = min(radius, len(states) // 2)
            if radius >= 1:
                x = np.arange(-radius, radius + 1, dtype=np.float64)
                kernel = np.exp(-0.5 * (x / sigma_frames) ** 2)
                kernel /= kernel.sum()

                # Pad with edge values (avoid zero-pull at boundaries).
                cx_pad = np.pad(cx_arr, radius, mode="edge")
                cy_pad = np.pad(cy_arr, radius, mode="edge")
                zm_pad = np.pad(zoom_arr, radius, mode="edge")

                cx_smooth = np.convolve(cx_pad, kernel, mode="valid")
                cy_smooth = np.convolve(cy_pad, kernel, mode="valid")
                zm_smooth = np.convolve(zm_pad, kernel, mode="valid")

                # Clamp zoom to valid range.
                zm_smooth = np.clip(zm_smooth, self.zoom_min, self.zoom_max)

                # --- ACCELERATION LIMITING ---
                # The Gaussian smooths positions but the camera still has
                # instantaneous velocity changes (trapezoidal profile) at
                # hold/track transitions.  An acceleration limiter converts
                # those into S-curve profiles: the camera eases in and out
                # of motion instead of snapping to full speed.
                #
                # max_accel = max_speed / (ramp_time * fps)
                # With ramp_time ~0.3s the camera takes ~9 frames to reach
                # full speed, producing visibly smooth acceleration.
                _max_spd_x = pxpf_x  # pre-computed speed limit px/frame
                _max_spd_y = pxpf_y
                _ramp_s = 0.35  # seconds to reach full speed (longer ramp = smoother starts/stops)
                _ramp_frames = max(1.0, _ramp_s * render_fps)
                _max_accel_x = _max_spd_x / _ramp_frames
                _max_accel_y = _max_spd_y / _ramp_frames

                n = len(cx_smooth)
                if n > 2:
                    # Compute velocities
                    vx_arr = np.diff(cx_smooth)
                    vy_arr = np.diff(cy_smooth)

                    # Forward pass: enforce acceleration limit
                    for _ai in range(1, len(vx_arr)):
                        dvx = vx_arr[_ai] - vx_arr[_ai - 1]
                        if abs(dvx) > _max_accel_x:
                            vx_arr[_ai] = vx_arr[_ai - 1] + np.clip(dvx, -_max_accel_x, _max_accel_x)
                        dvy = vy_arr[_ai] - vy_arr[_ai - 1]
                        if abs(dvy) > _max_accel_y:
                            vy_arr[_ai] = vy_arr[_ai - 1] + np.clip(dvy, -_max_accel_y, _max_accel_y)

                    # Re-apply speed limit on velocities
                    vx_arr = np.clip(vx_arr, -_max_spd_x, _max_spd_x)
                    vy_arr = np.clip(vy_arr, -_max_spd_y, _max_spd_y)

                    # Reconstruct positions from clamped velocities
                    cx_smooth[0] = cx_smooth[0]  # anchor first frame
                    cy_smooth[0] = cy_smooth[0]
                    for _ai in range(len(vx_arr)):
                        cx_smooth[_ai + 1] = cx_smooth[_ai] + vx_arr[_ai]
                        cy_smooth[_ai + 1] = cy_smooth[_ai] + vy_arr[_ai]

                    # Light final Gaussian to remove any remaining kinks
                    # from the acceleration clamping.
                    _final_sigma = min(4.0, sigma_frames * 0.5)
                    if _final_sigma >= 0.5:
                        _fr = int(_final_sigma * 3.0 + 0.5)
                        _fr = min(_fr, n // 2)
                        if _fr >= 1:
                            _fx = np.arange(-_fr, _fr + 1, dtype=np.float64)
                            _fk = np.exp(-0.5 * (_fx / _final_sigma) ** 2)
                            _fk /= _fk.sum()
                            _cx_p2 = np.pad(cx_smooth, _fr, mode="edge")
                            _cy_p2 = np.pad(cy_smooth, _fr, mode="edge")
                            cx_smooth = np.convolve(_cx_p2, _fk, mode="valid")
                            cy_smooth = np.convolve(_cy_p2, _fk, mode="valid")

                # --- EDGE PROTECTION ---
                # At clip boundaries the Gaussian averages with future (or
                # past) frames that may have very different ball positions
                # (e.g., free kick: ball stationary at kick taker then
                # flying across the field).  Edge padding (mode="edge")
                # replicates the boundary value but cannot prevent the
                # one-sided pull from rapidly-changing future values.
                #
                # Fix: blend smoothed positions back toward the pre-smooth
                # (speed-limited, EMA-tracked) values using a quadratic
                # ramp over 2*sigma frames at each clip end.  The
                # quadratic shape ensures zero derivative at the boundary
                # (no velocity cliff) and a gradual transition to the
                # fully smoothed trajectory.
                _edge_ramp = int(sigma_frames * 2.0 + 0.5)
                if _edge_ramp >= 1:
                    _er = min(_edge_ramp, n // 2)  # don't overlap
                    for _ei in range(_er):
                        _t = (_ei / _edge_ramp) ** 2
                        cx_smooth[_ei] = (1.0 - _t) * cx_arr[_ei] + _t * cx_smooth[_ei]
                        cy_smooth[_ei] = (1.0 - _t) * cy_arr[_ei] + _t * cy_smooth[_ei]
                    for _ei in range(max(0, n - _er), n):
                        _t = ((n - 1 - _ei) / _edge_ramp) ** 2
                        cx_smooth[_ei] = (1.0 - _t) * cx_arr[_ei] + _t * cx_smooth[_ei]
                        cy_smooth[_ei] = (1.0 - _t) * cy_arr[_ei] + _t * cy_smooth[_ei]

                # Re-derive crop dimensions from smoothed positions/zoom
                # and write back into the CamState objects.
                for i, st in enumerate(states):
                    st.cx = float(cx_smooth[i])
                    st.cy = float(cy_smooth[i])
                    st.zoom = float(zm_smooth[i])

                    z = float(np.clip(st.zoom, self.zoom_min, self.zoom_max))
                    c_h = self.height / max(z, 1e-6)
                    c_w = c_h * aspect_ratio
                    if c_w > self.width:
                        c_w = self.width
                        c_h = c_w / max(aspect_ratio, 1e-6)
                    if self.pad > 0.0:
                        ps = max(0.0, 1.0 - 2.0 * self.pad)
                        c_w *= ps
                        c_h *= ps
                    st.crop_w = c_w
                    st.crop_h = c_h
                    # Recompute x0/y0 from smoothed center.
                    st.x0 = float(np.clip(st.cx - c_w / 2.0, 0.0, max(0.0, self.width - c_w)))
                    st.y0 = float(np.clip(st.cy - c_h / 2.0, 0.0, max(0.0, self.height - c_h)))

        # --- POST-SMOOTH BALL-IN-CROP GUARANTEE ---
        # Gaussian + acceleration-limiter smoothing can cause the camera
        # to under-shoot direction changes (the smoothing averages the
        # pre-peak and post-peak positions, reducing peak amplitude).
        # This lets the ball drift toward — or past — the crop edge
        # for several frames during fast pans with direction reversals.
        #
        # Fix: a final forward pass that applies the minimum camera
        # shift needed to keep the ball inside a safety margin.
        # Uses CLAMPED crop bounds (x0/y0) rather than centered (cx-hw)
        # so the guarantee holds even when the crop is pushed against
        # the frame edge.
        _bic_margin = max(20.0, self.margin_px * 0.5)

        def _apply_bic(states_list, margin, fw, fh):
            """Apply ball-in-crop guarantee using actual clamped crop bounds.
            Returns number of corrected frames."""
            _count = 0
            for _s in states_list:
                if _s.ball is None:
                    continue
                _bx, _by = _s.ball
                _hw = _s.crop_w * 0.5
                _hh = _s.crop_h * 0.5
                # Use clamped bounds (x0/y0), not centered (cx - hw)
                _cl = _s.x0
                _cr = _s.x0 + _s.crop_w
                _ct = _s.y0
                _cb = _s.y0 + _s.crop_h

                _need_x = 0.0
                _ld = _bx - (_cl + margin)
                _rd = (_cr - margin) - _bx
                if _ld < 0.0:
                    _need_x = _ld
                elif _rd < 0.0:
                    _need_x = -_rd

                _need_y = 0.0
                _td = _by - (_ct + margin)
                _bd = (_cb - margin) - _by
                if _td < 0.0:
                    _need_y = _td
                elif _bd < 0.0:
                    _need_y = -_bd

                if abs(_need_x) > 0.5 or abs(_need_y) > 0.5:
                    _s.cx += _need_x
                    _s.cy += _need_y
                    _s.x0 = float(np.clip(
                        _s.cx - _hw, 0.0,
                        max(0.0, fw - _s.crop_w),
                    ))
                    _s.y0 = float(np.clip(
                        _s.cy - _hh, 0.0,
                        max(0.0, fh - _s.crop_h),
                    ))
                    _s.cx = _s.x0 + _hw
                    _s.cy = _s.y0 + _hh
                    _count += 1
            return _count

        _bic_count = _apply_bic(states, _bic_margin, self.width, self.height)

        if _bic_count > 0:
            logger.info(
                "[POST-SMOOTH] Ball-in-crop guarantee: adjusted %d/%d frames"
                " (margin=%.0fpx)",
                _bic_count, len(states), _bic_margin,
            )

            # Light Gaussian smooth after BIC corrections to prevent the
            # discrete per-frame shifts from creating visible jitter.
            _bic_sigma = 2.5
            _bic_r = int(_bic_sigma * 3.0 + 0.5)
            _bic_r = min(_bic_r, len(states) // 2)
            if _bic_r >= 1 and len(states) > 2 * _bic_r:
                _bic_cx = np.array([s.cx for s in states], dtype=np.float64)
                _bic_cy = np.array([s.cy for s in states], dtype=np.float64)
                _bic_x = np.arange(-_bic_r, _bic_r + 1, dtype=np.float64)
                _bic_k = np.exp(-0.5 * (_bic_x / _bic_sigma) ** 2)
                _bic_k /= _bic_k.sum()
                _bic_cx_p = np.pad(_bic_cx, _bic_r, mode="edge")
                _bic_cy_p = np.pad(_bic_cy, _bic_r, mode="edge")
                _bic_cx_s = np.convolve(_bic_cx_p, _bic_k, mode="valid")
                _bic_cy_s = np.convolve(_bic_cy_p, _bic_k, mode="valid")

                # Edge ramp: blend smoothed back toward pre-smooth at
                # clip boundaries so the BIC Gaussian doesn't undo the
                # main post-plan edge protection.
                _bic_n = len(_bic_cx)
                _bic_er = min(_bic_r * 2, _bic_n // 2)
                if _bic_er >= 1:
                    for _bei in range(_bic_er):
                        _bt = (_bei / _bic_er) ** 2
                        _bic_cx_s[_bei] = (1.0 - _bt) * _bic_cx[_bei] + _bt * _bic_cx_s[_bei]
                        _bic_cy_s[_bei] = (1.0 - _bt) * _bic_cy[_bei] + _bt * _bic_cy_s[_bei]
                    for _bei in range(max(0, _bic_n - _bic_er), _bic_n):
                        _bt = ((_bic_n - 1 - _bei) / _bic_er) ** 2
                        _bic_cx_s[_bei] = (1.0 - _bt) * _bic_cx[_bei] + _bt * _bic_cx_s[_bei]
                        _bic_cy_s[_bei] = (1.0 - _bt) * _bic_cy[_bei] + _bt * _bic_cy_s[_bei]

                for _bi, _bs in enumerate(states):
                    _bs.cx = float(_bic_cx_s[_bi])
                    _bs.cy = float(_bic_cy_s[_bi])
                    _hw = _bs.crop_w * 0.5
                    _hh = _bs.crop_h * 0.5
                    _bs.x0 = float(np.clip(
                        _bs.cx - _hw, 0.0,
                        max(0.0, self.width - _bs.crop_w),
                    ))
                    _bs.y0 = float(np.clip(
                        _bs.cy - _hh, 0.0,
                        max(0.0, self.height - _bs.crop_h),
                    ))
                    _bs.cx = _bs.x0 + _hw
                    _bs.cy = _bs.y0 + _hh

            # Re-apply BIC after smoothing — the Gaussian can undo
            # corrections by averaging toward uncorrected neighbors.
            _bic2 = _apply_bic(states, _bic_margin, self.width, self.height)
            if _bic2 > 0:
                logger.info(
                    "[POST-SMOOTH] BIC re-applied after smooth: %d frames",
                    _bic2,
                )

        return states


def _load_overlay(path: Optional[Path], output_size: Tuple[int, int]) -> Optional[np.ndarray]:
    if not path:
        return None
    if not path.exists():
        logging.warning("Brand overlay %s not found; skipping.", path)
        return None
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        logging.warning("Failed to read brand overlay at %s; skipping.", path)
        return None
    resized = cv2.resize(image, output_size, interpolation=cv2.INTER_LINEAR)
    return resized


def _apply_overlay(frame: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    if overlay.shape[2] < 4:
        return cv2.addWeighted(frame, 0.7, overlay[:, :, :3], 0.3, 0.0)
    alpha = overlay[:, :, 3:] / 255.0
    base = frame.astype(np.float32)
    overlay_rgb = overlay[:, :, :3].astype(np.float32)
    blended = overlay_rgb * alpha + base * (1.0 - alpha)
    return blended.astype(np.uint8)


class Renderer:
    def __init__(
        self,
        input_path: Path,
        output_path: Path,
        temp_dir: Path,
        fps_in: float,
        fps_out: float,
        flip180: bool,
        portrait: Optional[Tuple[int, int]],
        brand_overlay: Optional[Path],
        endcard: Optional[Path],
        pad: float,
        zoom_min: float,
        zoom_max: float,
        speed_limit: float,
        telemetry: Optional[TextIO],
        telemetry_simple: Optional[TextIO] = None,
        init_manual: bool = False,
        init_t: float = 0.8,
        ball_path: Optional[Sequence[BallPathEntry]] = None,
        follow_lead_time: float = 0.0,
        follow_margin_px: float = 0.0,
        follow_smoothing: float = 0.3,
        *,
        follow_zeta: float = 0.95,
        follow_wn: float = 6.0,
        follow_deadzone: float = 0.0,
        follow_max_vel: Optional[float] = None,
        follow_max_acc: Optional[float] = None,
        follow_lookahead: int = 0,
        follow_pre_smooth: float = 0.0,
        follow_zoom_out_max: float = 1.35,
        follow_zoom_edge_frac: float = 1.0,
        follow_center_frac: float = 0.5,
        lost_hold_ms: int = 500,
        lost_pan_ms: int = 1200,
        lost_lookahead_s: float = 6.0,
        lost_chase_motion_ms: int = 900,
        lost_motion_thresh: float = 1.6,
        lost_use_motion: bool = False,
        portrait_plan_margin_px: Optional[float] = None,
        portrait_plan_headroom: Optional[float] = None,
        portrait_plan_lead_px: Optional[float] = None,
        adaptive_tracking: bool = False,
        adaptive_tracking_config: Optional[dict] = None,
        auto_tune: bool = False,
        plan_override_data: Optional[dict[str, np.ndarray]] = None,
        plan_override_len: int = 0,
        ball_samples: Optional[List[BallSample]] = None,
        fallback_ball_samples: Optional[List[BallSample]] = None,
        keep_path_lookup_data: Optional[dict[int, Tuple[float, float]]] = None,
        debug_ball_overlay: bool = False,
        follow_override: Optional[Mapping[int, Mapping[str, float]]] = None,
        disable_controller: bool = False,
        follow_trajectory: Optional[List[Mapping[str, float]]] = None,
        debug_pan_overlay: bool = False,
    ) -> None:
        # Fallback initialization for variables that used to be set in try/except blocks
        motion_thresh_value = None
        # Ensure defaults for variables that may have been assigned inside removed try/except blocks
        zoom_edge_frac = None
        motion_thresh_value = None  # fallback for removed try/except

        self.input_path = input_path
        self.output_path = output_path
        self.temp_dir = temp_dir
        self.fps_in = fps_in
        self.fps_out = fps_out
        self.flip180 = flip180
        self.portrait = portrait
        self.brand_overlay_path = brand_overlay
        self.endcard_path = endcard
        self.pad = float(pad)
        self.zoom_min = float(zoom_min)
        self.zoom_max = float(zoom_max)
        self.speed_limit = float(speed_limit)
        self.telemetry = telemetry
        self.telemetry_simple = telemetry_simple
        self.last_ffmpeg_command: Optional[List[str]] = None
        self.init_manual = bool(init_manual)
        self.init_t = float(init_t)
        self.follow_lead_time = max(0.0, float(follow_lead_time))
        self.follow_margin_px = max(0.0, float(follow_margin_px))
        self.follow_smoothing = float(np.clip(follow_smoothing, 0.0, 1.0))
        self.follow_zeta = float(follow_zeta)
        self.follow_wn = float(follow_wn)
        self.follow_deadzone = max(0.0, float(follow_deadzone))
        self.follow_max_vel = None if follow_max_vel is None else float(follow_max_vel)
        self.follow_max_acc = None if follow_max_acc is None else float(follow_max_acc)
        self.follow_lookahead = int(follow_lookahead)
        self.follow_pre_smooth = float(np.clip(follow_pre_smooth, 0.0, 1.0))
        self.follow_zoom_out_max = max(1.0, float(follow_zoom_out_max))
        zoom_edge_frac = follow_zoom_edge_frac
        if zoom_edge_frac is None:
            zoom_edge_frac = 0.15  # safe default; avoids crash
        if not math.isfinite(zoom_edge_frac) or zoom_edge_frac <= 0.0:
            zoom_edge_frac = 1.0
        self.follow_zoom_edge_frac = zoom_edge_frac
        if not math.isfinite(follow_center_frac):
            follow_center_frac = 0.5
        self.follow_center_frac = float(np.clip(follow_center_frac, 0.0, 1.0))
        self.lost_hold_ms = max(0, int(lost_hold_ms))
        self.lost_pan_ms = max(0, int(lost_pan_ms))
        self.lost_chase_motion_ms = max(0, int(lost_chase_motion_ms))
        if not math.isfinite(lost_lookahead_s) or lost_lookahead_s < 0.0:
            lost_lookahead_s = 0.0
        self.lost_lookahead_s = lost_lookahead_s
        if motion_thresh_value is None:
            motion_thresh_value = 0.02  # safe small threshold
        # Ensure variable is initialized
        if motion_thresh_value is None:
            motion_thresh_value = 0.02  # safe fallback threshold
        if not math.isfinite(motion_thresh_value):
            motion_thresh_value = 1.6
        self.lost_motion_thresh = max(0.0, motion_thresh_value)
        self.lost_use_motion = bool(lost_use_motion)
        self.keepinview_min_band_frac = 0.12
        self.disable_controller = bool(disable_controller)
        self.follow_trajectory = list(follow_trajectory) if follow_trajectory else None
        self.debug_pan_overlay = bool(debug_pan_overlay)
        self.original_src_w: Optional[float] = None
        self.original_src_h: Optional[float] = None
        self.follow_plan_x: Optional[np.ndarray] = None
        self.pan_x_plan: Optional[np.ndarray] = None

        def _coerce_float(value: Optional[float]) -> Optional[float]:
            if value is None:
                return None

        self.portrait_plan_margin_px = _coerce_float(portrait_plan_margin_px)
        self.portrait_plan_headroom_frac = _coerce_float(portrait_plan_headroom)
        self.portrait_plan_lead_px = _coerce_float(portrait_plan_lead_px)
        self.adaptive_tracking = bool(adaptive_tracking)
        self.adaptive_tracking_config = adaptive_tracking_config or {}
        self.auto_tune = bool(auto_tune)
        self.plan_override_data = plan_override_data
        self.plan_override_len = int(plan_override_len or 0)
        self.ball_samples = ball_samples or []
        self.fallback_ball_samples = fallback_ball_samples or []
        self.keep_path_lookup_data = keep_path_lookup_data or {}
        self.debug_ball_overlay = bool(debug_ball_overlay)
        self.follow_override = follow_override

        normalized_ball_path: Optional[List[Optional[dict[str, float]]]] = None
        if ball_path:
            normalized_ball_path = []
            for entry in ball_path:
                if entry is None:
                    normalized_ball_path.append(None)
                    continue
                if isinstance(entry, Mapping):
                    sanitized: dict[str, float] = {}
                    for key, value in entry.items():
                        if isinstance(value, (int, float)):
                            sanitized[key] = float(value)
                    if "z" not in sanitized:
                        z_value = entry.get("z", 1.30) if hasattr(entry, "get") else 1.30
                    bx_norm = sanitized.get("bx")
                    by_norm = sanitized.get("by")
                    if bx_norm is None or by_norm is None:
                        bx_norm = sanitized.get("bx_stab", sanitized.get("bx_raw", bx_norm))
                        by_norm = sanitized.get("by_stab", sanitized.get("by_raw", by_norm))
                    if bx_norm is None or by_norm is None:
                        normalized_ball_path.append(None)
                        continue
                    sanitized["bx"] = float(bx_norm)
                    sanitized["by"] = float(by_norm)
                    normalized_ball_path.append(sanitized)
                else:
                    entry_seq = tuple(entry)
                    if len(entry_seq) < 2:
                        normalized_ball_path.append(None)
                        continue
                    bx_val = entry_seq[0]
                    by_val = entry_seq[1]
                    if bx_val is None or by_val is None:
                        normalized_ball_path.append(None)
                        continue
                    z_val = entry_seq[2] if len(entry_seq) >= 3 else 1.30
        self.offline_ball_path = normalized_ball_path

    def _simulate_follow_centers(
        self,
        states: Sequence[CamState],
        ball_samples: Sequence[BallSample],
    ) -> np.ndarray:
        """
        Compute horizontal camera centers for each frame using the
        predictive, windowed follow model.

        `states` is the list of CamState (or similar) objects built for
        each output frame. We only need timing and crop vs source widths.

        `ball_samples` is the list of BallSample objects loaded from
        telemetry, which have at least t, x, y (we only use t and x here).
        """
        import numpy as np

        if not states:
            return np.zeros(0, dtype=float)
        if not ball_samples:
            return np.zeros(len(states), dtype=float)

        fps = float(self.fps_out or self.fps_in or 24.0)
        dt = 1.0 / float(fps if fps > 0 else 24.0)
        frame_t = np.arange(len(states), dtype=float) * dt

        ball_t = np.array([getattr(b, "t", idx * dt) for idx, b in enumerate(ball_samples)], dtype=float)
        ball_x_samples = np.array([getattr(b, "x", float("nan")) for b in ball_samples], dtype=float)

        valid_mask = np.isfinite(ball_t) & np.isfinite(ball_x_samples)
        if not valid_mask.any():
            return np.full(len(states), float(np.nan), dtype=float)

        sort_idx = np.argsort(ball_t[valid_mask])
        sorted_t = ball_t[valid_mask][sort_idx]
        sorted_x = ball_x_samples[valid_mask][sort_idx]

        ball_x = np.interp(
            frame_t,
            sorted_t,
            sorted_x,
            left=sorted_x[0],
            right=sorted_x[-1],
        )

        if hasattr(self, "follow_crop_width") and self.follow_crop_width:
            crop_width = int(self.follow_crop_width)
        else:
            crop_width = int(round(np.median([s.crop_w for s in states]))) if states else 0
        if crop_width <= 0:
            crop_width = int(round(self.original_src_w or getattr(self, "src_width", 0) or 0))

        src_width = int(round(self.original_src_w or getattr(self, "src_width", crop_width)))
        if src_width <= 0:
            src_width = max(crop_width, int(np.nanmax(ball_x)) if np.isfinite(ball_x).any() else crop_width)

        centers = compute_predictive_follow_centers(
            ball_x=ball_x,
            t=frame_t,
            src_width=src_width,
            crop_width=crop_width,
            fps=fps,
            window_frac=0.25,
            horizon_s=0.35,
            alpha=0.90,
            max_cam_speed_px_per_s=850.0,
        )

        return centers

    @staticmethod
    def _center_bias_px_for_height(frame_h: float, center_frac: float) -> float:
        if not math.isfinite(center_frac):
            center_frac = 0.5
        return (0.5 - center_frac) * frame_h

    def _ball_overlay_samples(self, fps: float) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        samples = self.ball_samples or self.fallback_ball_samples
        if not samples:
            return [], []

        triples: list[tuple[float, float, float]] = []
        for sample in samples:
            t_val = None
            x_val = None
            y_val = None
            if isinstance(sample, Mapping):
                t_val = sample.get("t") or sample.get("time") or sample.get("ts")
                x_val = sample.get("x") or sample.get("cx") or sample.get("ball_x")
                y_val = sample.get("y") or sample.get("cy") or sample.get("ball_y")
                if t_val is None:
                    t_val = sample.get("frame")
                    if t_val is not None and fps > 0:
                        t_val = float(t_val) / float(fps)
            else:
                t_val = getattr(sample, "t", None) or getattr(sample, "time", None) or getattr(sample, "ts", None)
                x_val = getattr(sample, "x", None) or getattr(sample, "cx", None)
                y_val = getattr(sample, "y", None) or getattr(sample, "cy", None)
                if t_val is None:
                    frame_val = getattr(sample, "frame", None)
                    if frame_val is not None and fps > 0:
                        t_val = float(frame_val) / float(fps)

            t_float = _safe_float(t_val)
            x_float = _safe_float(x_val)
            y_float = _safe_float(y_val)
            if t_float is None or x_float is None or y_float is None:
                continue
            triples.append((t_float, x_float, y_float))

        if not triples:
            return [], []

        triples.sort(key=lambda row: row[0])
        samples_x = [(row[0], row[1]) for row in triples]
        samples_y = [(row[0], row[2]) for row in triples]
        return samples_x, samples_y

    def _compose_frame(
        self,
        frame: np.ndarray,
        t,
        pan_x: Optional[float] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        state: Optional[CamState] = kwargs.pop("state", None)
        if state is None and isinstance(t, CamState):
            state = t
        output_size: Optional[Tuple[int, int]] = kwargs.pop("output_size", None)
        if output_size is None:
            output_size = kwargs.pop("size", None)
        overlay_image: Optional[np.ndarray] = kwargs.pop("overlay_image", None)

        height, width = frame.shape[:2]
        self.src_w = float(getattr(self, "src_w", None) or self.original_src_w or width)
        self.src_h = float(getattr(self, "src_h", None) or self.original_src_h or height)
        portrait_w = float(
            getattr(self, "portrait_w", 0.0) or getattr(self, "follow_crop_width", 0.0) or width
        )
        self.portrait_w = portrait_w

        # --- horizontal pan center (source coords) ---
        if pan_x is not None and math.isfinite(pan_x):
            center_x = float(pan_x)
        elif state is not None and math.isfinite(getattr(state, "cx", float("nan"))):
            center_x = float(state.cx)
        else:
            center_x = self.src_w / 2.0
        center_y = float(getattr(state, "cy", 0.0)) if state is not None else 0.0

        if getattr(self, "debug_pan_overlay", False):
            cx = int(round(center_x))
            cv2.line(frame, (cx, 0), (cx, height - 1), (0, 255, 0), 2)

        half_w = portrait_w / 2.0
        crop_w_int = int(round(portrait_w))
        # Derive right from left + fixed width to prevent ±1px crop width
        # fluctuation caused by independent rounding of both edges.
        left = int(round(center_x - half_w))
        right = left + crop_w_int

        if left < 0:
            left = 0
            right = left + crop_w_int
        if right > int(self.src_w):
            right = int(self.src_w)
            left = max(0, right - crop_w_int)

        cropped = frame[:, left:right]
        width = cropped.shape[1]
        height = cropped.shape[0]
        frame = cropped
        center_x = float(center_x - left)
        if state is not None:
            state.cx = center_x

        if output_size is None:
            output_size = (width, height)

        src_w = float(width)
        if src_w <= 0:
            src_w = float(width)
        scale_x = float(width) / float(src_w) if src_w else 1.0
        scaled_center_x = center_x * scale_x
        target_ar = 0.0
        if output_size[0] > 0 and output_size[1] > 0:
            target_ar = float(output_size[0]) / float(output_size[1])

        crop_w = float(np.clip(state.crop_w, 1.0, float(width))) if state is not None else float(width)
        crop_h = float(np.clip(state.crop_h, 1.0, float(height))) if state is not None else float(height)

        if target_ar > 0.0 and crop_h > 0.0:
            desired_w = crop_h * target_ar
            desired_h = crop_w / target_ar if target_ar > 0.0 else crop_h
            if desired_w <= width and not math.isclose(desired_w, crop_w, rel_tol=1e-4, abs_tol=1e-3):
                crop_w = float(desired_w)
            elif desired_h <= height and not math.isclose(desired_h, crop_h, rel_tol=1e-4, abs_tol=1e-3):
                crop_h = float(desired_h)

        desired_x0 = scaled_center_x - crop_w / 2.0
        desired_y0 = center_y - crop_h / 2.0
        max_x0 = max(0.0, float(width) - crop_w)
        max_y0 = max(0.0, float(height) - crop_h)
        clamped_x0 = float(np.clip(desired_x0, 0.0, max_x0))
        clamped_y0 = float(np.clip(desired_y0, 0.0, max_y0))

        x2_f = clamped_x0 + crop_w
        y2_f = clamped_y0 + crop_h
        if x2_f > width:
            clamped_x0 = max(0.0, float(width) - crop_w)
            x2_f = clamped_x0 + crop_w
        if y2_f > height:
            clamped_y0 = max(0.0, float(height) - crop_h)
            y2_f = clamped_y0 + crop_h

        # Derive right/bottom from left/top + fixed integer size to prevent
        # ±1px crop dimension fluctuation caused by independent rounding of
        # both edges.  Same pattern used for the first (portrait_w) crop.
        crop_w_int2 = int(round(crop_w))
        crop_h_int2 = int(round(crop_h))
        crop_left = int(round(clamped_x0))
        y1 = int(round(clamped_y0))
        x1 = max(0, min(crop_left, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = min(x1 + crop_w_int2, width)
        y2 = min(y1 + crop_h_int2, height)
        # If right/bottom clamp pushed us, shift left/top back so size stays fixed.
        if x2 - x1 < crop_w_int2 and x1 > 0:
            x1 = max(0, x2 - crop_w_int2)
        if y2 - y1 < crop_h_int2 and y1 > 0:
            y1 = max(0, y2 - crop_h_int2)

        crop_right = x2
        logger.debug("[PAN-DEBUG] crop_left=%d crop_right=%d", crop_left, crop_right)

        if self.debug_pan_overlay:
            overlay_frame = frame.copy()
            if state is not None and state.ball is not None:
                bx, by = state.ball
                cv2.circle(overlay_frame, (int(round(bx)), int(round(by))), 6, (0, 0, 255), -1)
            if state is not None and getattr(state, "recovery_active", False):
                cv2.line(overlay_frame, (0, 8), (width - 1, 8), (0, 255, 255), 3)
            if state is not None and getattr(state, "keepinview_override", False):
                cv2.putText(
                    overlay_frame,
                    "KEEP-IN-VIEW OVERRIDE",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            actual_center_x = int(round(clamped_x0 + crop_w / 2.0))
            cv2.line(
                overlay_frame,
                (actual_center_x, 0),
                (actual_center_x, height - 1),
                (0, 255, 0),
                2,
            )
            if math.isfinite(center_x):
                cv2.line(
                    overlay_frame,
                    (int(center_x), 0),
                    (int(center_x), height - 1),
                    (0, 255, 0),
                    2,
                )
            frame = overlay_frame

        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            cropped = frame
            x1, y1 = 0, 0
            x2, y2 = width, height

        resized = cv2.resize(cropped, output_size, interpolation=cv2.INTER_LANCZOS4)

        if overlay_image is not None:
            resized = _apply_overlay(resized, overlay_image)

        actual_crop = (float(x1), float(y1), float(x2 - x1), float(y2 - y1))
        return resized, actual_crop


    def _append_endcard(self, output_size: Tuple[int, int]) -> List[np.ndarray]:
        if not self.endcard_path:
            return []
        if not self.endcard_path.exists():
            logging.warning("Endcard %s not found; skipping.", self.endcard_path)
            return []
        image = cv2.imread(str(self.endcard_path))
        if image is None:
            logging.warning("Failed to read endcard at %s; skipping.", self.endcard_path)
            return []
        resized = cv2.resize(image, output_size, interpolation=cv2.INTER_LINEAR)
        frame_count = int(round(self.fps_out * 2.0))
        return [resized for _ in range(frame_count)]

    def write_frames(
        self,
        states: Sequence[CamState],
        *,
        probe_only: bool = False,
        follow_centers: Optional[Sequence[float]] = None,
        pan_x_plan: Optional[Sequence[float]] = None,
    ) -> float:
        idx = -1
        frames_dir = self.temp_dir / "frames"
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)
        input_mp4 = str(self.input_path)
        cap = cv2.VideoCapture(input_mp4)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_mp4}")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or float(self.fps_in) or 30.0
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if src_w <= 0 or src_h <= 0:
            ok, _first_frame = cap.read()
            if not ok or _first_frame is None:
                cap.release()
                raise RuntimeError("No frames decoded from the input video.")
            src_h, src_w = _first_frame.shape[:2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        width = int(src_w)
        height = int(src_h)
        self.original_src_w = float(src_w)
        self.original_src_h = float(src_h)
        self.src_w = float(src_w)
        self.src_h = float(src_h)
        if self.portrait:
            output_size = self.portrait
        else:
            output_size = (width, height)

        target_w = int(output_size[0]) if output_size[0] else width
        target_h = int(output_size[1]) if output_size[1] else height
        if target_h <= 0:
            target_h = height
        if target_w <= 0:
            target_w = width
        target_aspect = float(target_w) / float(target_h) if target_h else (width / float(height))
        output_size = (target_w, target_h)

        overlay_image = _load_overlay(self.brand_overlay_path, output_size)
        tf = self.telemetry
        simple_tf = self.telemetry_simple

        is_portrait = target_h > target_w
        portrait_plan_state: dict[str, float] = {}
        out_w = target_w
        out_h = target_h
        portrait_w = out_w if is_portrait else None
        portrait_h = out_h if is_portrait else None
        portrait_crop_w = None
        portrait_crop_h = None
        if is_portrait:
            portrait_crop_h = float(src_h)
            portrait_crop_w = float(int(round(src_h * 9.0 / 16.0)))
            portrait_crop_w = max(1.0, min(portrait_crop_w, float(src_w)))

        self.portrait_w = float(portrait_crop_w or portrait_w or width)
        self.portrait_h = float(portrait_crop_h or portrait_h or height)

        scaling_factor = float(portrait_crop_w or width) / float(src_w) if src_w else 1.0
        offset_x = 0.0
        logger.info(
            "[PAN-DEBUG] src=(%d,%d) portrait_target=(%s,%s) scale_x=%.4f offset_x=%.2f",
            src_w,
            src_h,
            portrait_crop_w if portrait_crop_w is not None else "-",
            portrait_crop_h if portrait_crop_h is not None else "-",
            scaling_factor,
            offset_x,
        )

        offline_ball_path = self.offline_ball_path

        frame_count = len(states)
        keep_path_lookup: dict[int, tuple[float, float]] = dict(self.keep_path_lookup_data)
        keepinview_enabled = bool(
            is_portrait and portrait_w and portrait_h and portrait_w > 0 and portrait_h > 0
        )
        if keepinview_enabled:
            crop_w = float(portrait_crop_w if portrait_crop_w else (portrait_w or out_w))
            crop_h = float(portrait_crop_h if portrait_crop_h else (portrait_h or out_h))
            if not keep_path_lookup:
                samples = self.ball_samples or load_ball_telemetry_for_clip(str(self.input_path))
                if samples:
                    total_frames = frame_count
                    if total_frames <= 0:
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if total_frames <= 0:
                        total_frames = max((int(getattr(s, "frame", 0)) for s in samples), default=0) + 1
                    telemetry_frames = []
                    sample_by_frame: dict[int, BallSample] = {}
                    for idx, s in enumerate(samples):
                        sample_by_frame[idx] = s
                    fps_hint = float(self.fps_out or src_fps or 30.0)
                    for idx in range(total_frames):
                        sample = sample_by_frame.get(idx)
                        if sample is None:
                            telemetry_frames.append(
                                {
                                    "t": idx / fps_hint,
                                    "x": None,
                                    "y": None,
                                    "visible": False,
                                }
                            )
                            continue
                        bx = _safe_float(getattr(sample, "x", None))
                        by = _safe_float(getattr(sample, "y", None))
                        telemetry_frames.append(
                            {
                                "t": getattr(sample, "t", idx / fps_hint),
                                "x": bx,
                                "y": by,
                                "visible": bx is not None and by is not None,
                            }
                        )
                    raw_cx = raw_cy = None
                    if self.debug_ball_overlay:
                        raw_cx, raw_cy = build_raw_ball_center_path(
                            telemetry_frames,
                            frame_width=int(width),
                            frame_height=int(height),
                            crop_width=int(crop_w),
                            crop_height=int(crop_h),
                            use_red_fallback=bool(self.fallback_ball_samples),
                            use_ball_telemetry=bool(self.ball_samples),
                        )
                    if raw_cx and raw_cy and len(raw_cx) == len(raw_cy):
                        cx_vals, cy_vals = smooth_center_path(
                            raw_cx,
                            raw_cy,
                            window=9,
                            max_step_px=40.0,
                        )
                        cx_vals, cy_vals = clamp_center_path_to_bounds(
                            cx_vals,
                            cy_vals,
                            frame_width=int(width),
                            frame_height=int(height),
                            crop_width=int(crop_w),
                            crop_height=int(crop_h),
                        )
                        keep_path_lookup = {
                            idx: (float(cx), float(cy)) for idx, (cx, cy) in enumerate(zip(cx_vals, cy_vals))
                        }

        cam = [(state.cx, state.cy, state.zoom) for state in states]
        if cam:
            cx_values = [value[0] for value in cam]
            cy_values = [value[1] for value in cam]
        else:
            cx_values = []
            cy_values = []

        duration_s = frame_count / float(self.fps_out) if self.fps_out else 0.0
        if not cam or (
            (max(cx_values) - min(cx_values) if cx_values else 0.0) < 1.0
            and (max(cy_values) - min(cy_values) if cy_values else 0.0) < 1.0
        ):
            fallback_path = rough_motion_path(str(self.input_path), float(self.fps_out), duration_s)
            if fallback_path:
                default_zoom = cam[0][2] if cam else 1.2
                cam = [(x, y, default_zoom) for _, x, y in fallback_path]
            else:
                default_zoom = cam[0][2] if cam else 1.2
                cam = [(width / 2.0, height / 2.0, default_zoom) for _ in range(frame_count)]

        if frame_count and len(cam) < frame_count:
            last = cam[-1]
            cam.extend([last] * (frame_count - len(cam)))
        elif frame_count and len(cam) > frame_count:
            cam = cam[:frame_count]

        if not hasattr(self, "base_crop_w") or not self.base_crop_w:
            self.base_crop_w = float(width)
        if not hasattr(self, "base_crop_h") or not self.base_crop_h:
            self.base_crop_h = float(height)

        if self.follow_override:
            states = list(states)
            max_idx = max(self.follow_override.keys()) if self.follow_override else -1
            target_len = max(len(states), max_idx + 1)
            default_zoom = states[0].zoom if states else 1.0
            default_w = float(self.base_crop_w)
            default_h = float(self.base_crop_h)
            for _ in range(len(states), target_len):
                states.append(
                    CamState(
                        frame=len(states),
                        cx=width / 2.0,
                        cy=height / 2.0,
                        zoom=default_zoom,
                        crop_w=default_w,
                        crop_h=default_h,
                        x0=width / 2.0 - default_w / 2.0,
                        y0=height / 2.0 - default_h / 2.0,
                        used_label=False,
                        clamp_flags=[],
                    )
                )

            for frame_idx, override in self.follow_override.items():
                state = states[frame_idx]
                cx = safe_float(override.get("cx"), state.cx)
                cy = safe_float(override.get("cy"), state.cy)
                zoom = safe_float(override.get("zoom", state.zoom), state.zoom)
                crop_w = float(self.base_crop_w / zoom) if zoom else float(self.base_crop_w)
                crop_h = float(self.base_crop_h / zoom) if zoom else float(self.base_crop_h)
                state.cx = float(cx)
                state.cy = float(cy)
                state.zoom = float(zoom)
                state.crop_w = float(crop_w)
                state.crop_h = float(crop_h)
                state.x0 = float(state.cx) - float(crop_w) / 2.0
                state.y0 = float(state.cy) - float(crop_h) / 2.0

        # Use source fps since write_frames reads every source frame without
        # rate conversion.
        render_fps = float(self.fps_in) if self.fps_in and self.fps_in > 0 else float(self.fps_out)
        zoom_min = float(self.zoom_min)
        zoom_max = float(self.zoom_max)
        src_w_f = float(width)
        src_h_f = float(height)
        center_bias_px = self._center_bias_px_for_height(src_h_f, self.follow_center_frac)
        speed_px_sec = float(self.speed_limit or 3000.0)

        offline_plan_data: Optional[dict[str, np.ndarray]] = self.plan_override_data
        follow_targets: Optional[Tuple[List[float], List[float]]] = None
        follow_valid_mask: Optional[List[bool]] = None
        offline_plan_len = int(self.plan_override_len or 0)
        if offline_plan_data is None and offline_ball_path:
            bx_vals: List[float] = []
            by_vals: List[float] = []
            for entry in offline_ball_path:
                if entry is None:
                    bx_vals.append(float("nan"))
                    by_vals.append(float("nan"))
                    continue
                bx_val: Optional[float] = None
                by_val: Optional[float] = None
                if isinstance(entry, Mapping):
                    if "bx_stab" in entry and "by_stab" in entry:
                        bx_val = entry.get("bx_stab")
                        by_val = entry.get("by_stab")
                    elif "bx" in entry and "by" in entry:
                        bx_val = entry.get("bx")
                        by_val = entry.get("by")
                    elif "bx_raw" in entry and "by_raw" in entry:
                        bx_val = entry.get("bx_raw")
                        by_val = entry.get("by_raw")
                else:
                    entry_seq = tuple(entry)
                    if len(entry_seq) >= 2:
                        bx_candidate = entry_seq[0]
                        by_candidate = entry_seq[1]
                        if bx_candidate is not None and by_candidate is not None:
                            bx_val = bx_candidate
                            by_val = by_candidate

                if bx_val is None or by_val is None:
                    bx_vals.append(float("nan"))
                    by_vals.append(float("nan"))
                else:
                    bx_vals.append(float(bx_val))
                    by_vals.append(float(by_val))

            if bx_vals:
                bx_arr = np.asarray(bx_vals, dtype=float)
                by_arr = np.asarray(by_vals, dtype=float)
                valid_mask_arr = np.isfinite(bx_arr) & np.isfinite(by_arr)

                def _ffill_nan(arr: np.ndarray, default: float) -> np.ndarray:
                    out = arr.copy()
                    last = default
                    for idx in range(len(out)):
                        if np.isfinite(out[idx]):
                            last = out[idx]
                        else:
                            out[idx] = last
                    return out

                default_cx = float(width) / 2.0
                default_cy = float(height) * self.follow_center_frac
                if not np.isfinite(bx_arr[0]):
                    bx_arr[0] = default_cx
                if not np.isfinite(by_arr[0]):
                    by_arr[0] = default_cy

                bx_arr = _ffill_nan(bx_arr, default_cx)
                by_arr = _ffill_nan(by_arr, default_cy)

                bx_list = bx_arr.astype(float).tolist()
                by_list = by_arr.astype(float).tolist()
                if self.follow_pre_smooth > 0:
                    bx_list, by_list = ema_path(bx_list, by_list, self.follow_pre_smooth)
                bias_px = self._center_bias_px_for_height(src_h_f, self.follow_center_frac)
                by_list = [float(np.clip(y + bias_px, 0.0, src_h_f)) for y in by_list]
                follow_targets = (bx_list, by_list)
                follow_valid_mask = valid_mask_arr.astype(bool).tolist()

                fps_for_plan = render_fps if render_fps > 0 else (src_fps if src_fps > 0 else 30.0)
                if fps_for_plan <= 0:
                    fps_for_plan = 30.0
                global FPS
                FPS = float(fps_for_plan)

                if portrait_w and portrait_h and portrait_w > 0 and portrait_h > 0:
                    target_aspect = float(portrait_w) / float(portrait_h)
                else:
                    target_aspect = (float(width) / float(height)) if height > 0 else 1.0

                pan_alpha = float(np.clip(self.follow_smoothing, 0.05, 0.95))
                lead_seconds = max(0.0, float(self.follow_lead_time))
                bounds_pad = int(round(self.follow_margin_px)) if self.follow_margin_px > 0 else 16
                bounds_pad = max(8, bounds_pad)

                planner_enabled = bool(
                    is_portrait
                    and portrait_w
                    and portrait_h
                    and portrait_w > 0
                    and portrait_h > 0
                )
                if planner_enabled:
                    plan_x0, plan_y0, plan_w, plan_h, plan_spd, plan_zoom = plan_camera_from_ball(
                        bx_arr,
                        by_arr,
                        float(width),
                        float(height),
                        float(target_aspect),
                        pan_alpha=pan_alpha,
                        lead=lead_seconds,
                        bounds_pad=bounds_pad,
                        center_frac=self.follow_center_frac,
                        margin_px_override=self.portrait_plan_margin_px,
                        headroom_frac_override=self.portrait_plan_headroom_frac,
                        lead_px_override=self.portrait_plan_lead_px,
                        adaptive=self.adaptive_tracking,
                        adaptive_config=self.adaptive_tracking_config,
                        auto_tune=self.auto_tune,
                    )

                    offline_plan_len = len(plan_x0)
                    offline_plan_data = {
                        "x0": plan_x0.astype(float),
                        "y0": plan_y0.astype(float),
                        "w": plan_w.astype(float),
                        "h": plan_h.astype(float),
                        "spd": plan_spd.astype(float),
                        "z": plan_zoom.astype(float),
                    }

                else:
                    offline_plan_len = 0
                    offline_plan_data = None

        kal: Optional[CV2DKalman] = None
        template: Optional[np.ndarray] = None
        tpl_side = 64
        prev_cx = src_w_f / 2.0
        prev_cy = src_h_f / 2.0
        initial_zoom = cam[0][2] if cam else 1.2
        zoom = float(np.clip(float(initial_zoom), zoom_min, zoom_max))
        prev_zoom = float(zoom)
        prev_ball_x: Optional[float] = None
        prev_ball_y: Optional[float] = None
        prev_ball_source: Optional[str] = None
        prev_ball_src_x: Optional[float] = None
        prev_ball_src_y: Optional[float] = None
        prev_bx = float(prev_cx)
        prev_by = float(prev_cy)
        if self.disable_controller:
            follow_targets = None
            follow_valid_mask = None
        follow_targets_len = len(follow_targets[0]) if follow_targets else 0
        follow_lookahead_frames = max(0, int(self.follow_lookahead))

        crop_scale_plan = getattr(self, "follow_crop_scale_plan", None)
        recovery_flags = getattr(self, "pan_recovery_flags", None)
        original_crop_sizes: list[tuple[float, float]] = []
        if states:
            original_crop_sizes = [(float(s.crop_w), float(s.crop_h)) for s in states]
        if follow_centers is not None:
            states = list(states)
            for idx, cx in enumerate(follow_centers):
                if idx >= len(states):
                    break
                state = states[idx]
                base_crop_w, base_crop_h = original_crop_sizes[idx] if idx < len(original_crop_sizes) else (state.crop_w, state.crop_h)
                scale = 1.0
                if crop_scale_plan is not None and idx < len(crop_scale_plan):
                    scale = float(crop_scale_plan[idx])
                scaled_w = float(np.clip(base_crop_w * scale, 1.0, float(width)))
                scaled_h = float(np.clip(base_crop_h * scale, 1.0, float(height)))
                state.crop_w = scaled_w
                state.crop_h = scaled_h
                half_w = scaled_w / 2.0 if scaled_w > 0 else float(width) / 2.0
                clamped_cx = float(np.clip(float(cx), half_w, float(width) - half_w))
                state.cx = clamped_cx
                state.x0 = clamped_cx - scaled_w / 2.0
                if recovery_flags is not None and idx < len(recovery_flags):
                    state.recovery_active = bool(recovery_flags[idx])
        active_pan_plan = None
        if pan_x_plan is not None:
            active_pan_plan = pan_x_plan
        elif getattr(self, "pan_x_plan", None) is not None:
            active_pan_plan = self.pan_x_plan
        elif hasattr(self, "follow_plan_x"):
            active_pan_plan = self.follow_plan_x

        overlay_samples_x: list[tuple[float, float]] = []
        overlay_samples_y: list[tuple[float, float]] = []
        overlay_times: list[float] = []
        if self.debug_ball_overlay:
            overlay_samples_x, overlay_samples_y = self._ball_overlay_samples(render_fps)
            overlay_times = [row[0] for row in overlay_samples_x]

        # --- VERTICAL WOBBLE STABILISATION ---
        # Physical camera wobble (e.g. wind on a tripod) creates
        # high-frequency vertical jitter visible in the rendered
        # portrait crop (which uses the full frame height).
        # Load pre-computed camera shifts, extract the wobble
        # component, and apply a per-frame correction via warpAffine.
        _wobble_dy: Optional[np.ndarray] = None
        _stab_fps = render_fps if render_fps > 0 else 30.0
        _shifts_path = Path(
            telemetry_path_for_video(str(self.input_path))
        ).with_suffix(".cam_shifts.npy")
        _cam_shifts = load_camera_shifts(_shifts_path)
        if _cam_shifts is None:
            logger.info("[STAB] No cached camera shifts found — computing on the fly")
            _cam_shifts = compute_camera_shifts(str(self.input_path), logger)
        if _cam_shifts is not None and len(_cam_shifts) >= 5:
            _wobble_dy = compute_wobble_corrections(
                _cam_shifts, _stab_fps,
                smooth_window_s=1.0,
                max_correction_px=12.0,
            )
        # Adaptive overscan: zoom just enough to hide BORDER_REPLICATE
        # artifacts at frame edges.  The zoom is constant across all
        # frames so it doesn't introduce visible pulsing.
        _stab_overscan = 1.0
        if _wobble_dy is not None:
            _peak_wobble = float(np.max(np.abs(_wobble_dy)))
            _stab_src_h = float(src_h) if src_h > 0 else 1080.0
            # Enough zoom to cover the peak shift, plus 30% safety margin.
            _stab_overscan = 1.0 + (_peak_wobble / _stab_src_h) * 1.3
            _stab_overscan = min(_stab_overscan, 1.04)  # cap at 4%
            logger.info(
                "[STAB] Vertical stabilisation active: %d frames, "
                "peak=%.1fpx, rms=%.1fpx, overscan=%.3f",
                len(_wobble_dy), _peak_wobble,
                float(np.sqrt(np.mean(_wobble_dy ** 2))),
                _stab_overscan,
            )

        for frame_idx in range(frame_count):
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            if frame_idx >= len(states):
                break

            # Apply vertical wobble correction before cropping.
            # A slight overscan zoom hides the BORDER_REPLICATE edge
            # artifacts that would otherwise appear as a blurry bar.
            if _wobble_dy is not None and frame_idx < len(_wobble_dy):
                _dy_corr = float(_wobble_dy[frame_idx])
                if abs(_dy_corr) > 0.3 or _stab_overscan > 1.0:
                    _h, _w = frame.shape[:2]
                    _cx_f = _w / 2.0
                    _cy_f = _h / 2.0
                    _s = _stab_overscan
                    _M = np.float32([
                        [_s, 0.0, _cx_f * (1.0 - _s)],
                        [0.0, _s, _cy_f * (1.0 - _s) - _dy_corr * _s],
                    ])
                    frame = cv2.warpAffine(
                        frame, _M, (_w, _h),
                        borderMode=cv2.BORDER_REPLICATE,
                    )

            if self.debug_ball_overlay:
                overlay_frame = frame.copy()
                t_val = frame_idx / float(render_fps) if render_fps else 0.0
                bx, by = _ball_xy_at_time(overlay_samples_x, overlay_samples_y, t_val, overlay_times)
                if bx is not None and by is not None:
                    cv2.circle(overlay_frame, (int(round(bx)), int(round(by))), 8, (0, 0, 255), -1)
                frame = overlay_frame

            state = states[frame_idx]
            desired_center_x = float(state.cx)
            pan_vel_clamp = None
            pan_accel_clamp = None
            state.cx = desired_center_x
            pan_x = None
            if active_pan_plan is not None and frame_idx < len(active_pan_plan):
                pan_x = active_pan_plan[frame_idx]
            composed, _ = self._compose_frame(
                frame,
                state,
                pan_x=pan_x,
                output_size=output_size,
                overlay_image=overlay_image,
            )
            out_path = frames_dir / f"frame_{frame_idx:06d}.png"
            cv2.imwrite(str(out_path), composed)

        # --- PATCH: initialize tracking flags so they always exist ---
        have_ball = False
        bx = by = None

        jerk95 = 0.0
        centers_for_jerk: Optional[np.ndarray] = None
        if follow_centers is not None:
            centers_for_jerk = np.asarray(list(follow_centers), dtype=float)
        elif self.ball_samples and render_fps > 0:
            centers_for_jerk = self._simulate_follow_centers(states, self.ball_samples)

        if centers_for_jerk is not None and centers_for_jerk.size > 1:
            xs = centers_for_jerk.astype(float).tolist()
            ys = [float(state.cy) for state in states[: len(xs)]]
            jerk95 = compute_camera_jerk95(xs, ys, render_fps)
        self.last_jerk95 = float(jerk95)

        tf = self.telemetry

        endcard_frames = self._append_endcard(output_size)
        if endcard_frames:
            start_index = len(states)
            for offset, endcard_frame in enumerate(endcard_frames):
                out_path = frames_dir / f"frame_{start_index + offset:06d}.png"
                cv2.imwrite(str(out_path), endcard_frame)

        return float(jerk95)

def ffmpeg_stitch(
        self,
        crf: int,
        keyint: int,
        log_path: Optional[Path] = None,
    ) -> None:
        frames_dir = self.temp_dir / "frames"
        pattern = str(frames_dir / "frame_%06d.png")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_output_path = _temp_output_path(self.output_path)
        if temp_output_path.exists():
                temp_output_path.unlink(missing_ok=True)

        command = [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(self.fps_out),
                "-i",
                pattern,
                "-i",
                str(self.input_path),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0?",
        ]

        # Optional scale to portrait output
        if self.portrait and self.portrait[0] > 0 and self.portrait[1] > 0:
                out_w, out_h = self.portrait
                command.extend(["-vf", f"scale={int(out_w)}:{int(out_h)}"])

        command.extend([
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                "-profile:v",
                "high",
                "-level",
                "4.0",
                "-x264-params",
                f"keyint={keyint}:min-keyint={keyint}:scenecut=0",
                "-movflags",
                "+faststart",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
        ])

        command.append(str(temp_output_path))

        self.last_ffmpeg_command = list(command)
        subprocess.run(command, check=True)
        temp_output_path.replace(self.output_path)

def _temp_output_path(output_path: Path) -> Path:
    return output_path.with_name(f".tmp.{output_path.name}")

def _resolve_output_path(
    explicit_out: Optional[str],
    input_path: Path,
    preset: str,
    portrait: bool = False,
) -> Tuple[Path, bool]:
    """Return (output_path, is_explicit_file).

    If the user passed --out pointing to an existing directory (or a path
    without an extension), place the deterministically named file inside it.
    Otherwise treat --out as a full file path.  When --out is not given,
    fall back to ``_default_output_path``.
    """
    if explicit_out:
        p = Path(explicit_out)
        if p.is_dir():
            name = build_output_name(
                input_path=str(input_path),
                preset=preset,
                portrait=None,
                follow=None,
                is_final=False,
                extra_tags=[],
            )
            return p / name, False
        return p, True
    return _default_output_path(input_path, preset, portrait=portrait), False

def _default_output_path(
    input_path: Path,
    preset: str,
    portrait: bool = False,
) -> Path:
    name = build_output_name(
        input_path=str(input_path),
        preset=preset,
        portrait=None,
        follow=None,
        is_final=False,
        extra_tags=[],
    )
    if portrait:
        # Single output tree: out/portrait/{match_subdir}/
        # Keeps rendered clips separate from source atomic_clips and
        # ensures re-runs always overwrite in one location.
        out_dir = Path("out/portrait").resolve() / input_path.parent.name
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / name
    return input_path.with_name(name)


def _prepare_temp_dir(temp_dir: Path, clean: bool = False) -> None:
    """Ensure *temp_dir* exists, optionally wiping it first."""
    if clean and temp_dir.is_dir():
        shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

def _scratch_root(explicit: Optional[str] = None) -> Path:
    """Return the base scratch directory for temporary frame extraction."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    return Path("out/_scratch").resolve()

def _derive_run_key(input_path: Path) -> str:
    stem = input_path.stem
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", stem).strip("_")
    if not cleaned:
        return "run"
    if len(cleaned) <= 40:
        return cleaned
    digest = hashlib.sha1(stem.encode("utf-8")).hexdigest()[:8]
    return f"{cleaned[:32]}_{digest}"


def load_ball_path(
    path: Union[str, os.PathLike[str]],
    ball_key_x: str = "bx_stab",
    ball_key_y: str = "by_stab",
) -> List[Optional[dict[str, float]]]:
    """Load a planned ball path JSONL file with stabilized coordinates."""

    seq: List[Optional[dict[str, float]]] = []
    default_zoom = 1.30
    with open(path, "r", encoding="utf-8") as f:
        for _, line in enumerate(f):
            line = line.strip()
            if not line:
                seq.append(None)
                continue
            if not isinstance(data, Mapping):
                seq.append(None)
                continue

            bx_norm: Optional[float] = None
            by_norm: Optional[float] = None
            for key_x, key_y in (
                (ball_key_x, ball_key_y),
                ("bx", "by"),
                ("bx_raw", "by_raw"),
            ):
                val_x = data.get(key_x)
                val_y = data.get(key_y)
                if val_x is None or val_y is None:
                    continue
                else:
                    break

            if bx_norm is None or by_norm is None:
                seq.append(None)
                continue

            rec: dict[str, float] = {}

            seen_pairs: set[tuple[str, str]] = set()
            for key_x, key_y in (
                (ball_key_x, ball_key_y),
                ("bx_stab", "by_stab"),
                ("bx_raw", "by_raw"),
                ("bx", "by"),
            ):
                if (key_x, key_y) in seen_pairs:
                    continue
                seen_pairs.add((key_x, key_y))
                val_x = data.get(key_x)
                val_y = data.get(key_y)
                if val_x is None or val_y is None:
                    continue

            rec["bx"] = bx_norm
            rec["by"] = by_norm

            z_value = data.get("z", default_zoom)

            t_value = data.get("t")
            if isinstance(t_value, (int, float)):
                rec["t"] = float(t_value)

            frame_value = data.get("f")
            if isinstance(frame_value, (int, float)):
                rec["f"] = float(frame_value)

            seq.append(rec)

    return seq


def run(
    args: argparse.Namespace,
    telemetry_path: Optional[Path] = None,
    telemetry_simple_path: Optional[Path] = None,
) -> None:
    renderer = None  # initialize renderer safely
    render_telemetry_path = telemetry_path
    render_telemetry_simple_path = telemetry_simple_path
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    ### PATCH: sanitize controller parameters
    def _safe(val, default):
        try:
            if val is None:
                return default
            v = float(val)
            if not math.isfinite(v):
                return default
            return v
        except Exception:
            return default

    original_source_path = Path(args.in_path).expanduser().resolve()
    if not original_source_path.exists():
        raise FileNotFoundError(f"Input file not found: {original_source_path}")

    src_path = original_source_path
    upscale_factor = 1
    original_width_pre_upscale: Optional[int] = None
    original_height_pre_upscale: Optional[int] = None
    if getattr(args, "upscale", False):
        # Read original dimensions BEFORE upscale
        _pre_cap = cv2.VideoCapture(str(src_path))
        if _pre_cap.isOpened():
            original_width_pre_upscale = int(_pre_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height_pre_upscale = int(_pre_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            _pre_cap.release()

        scale_value = args.upscale_scale if args.upscale_scale and args.upscale_scale > 0 else 2
        force_reupscale = bool(getattr(args, "force_reupscale", False))
        upscaled_str = upscale_video(str(src_path), scale=scale_value, force=force_reupscale)
        src_path = Path(upscaled_str).expanduser().resolve()

        # Verify actual upscale ratio from the output file
        _post_cap = cv2.VideoCapture(str(src_path))
        if _post_cap.isOpened() and original_width_pre_upscale and original_width_pre_upscale > 0:
            actual_w = int(_post_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(_post_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            _post_cap.release()
            actual_ratio = actual_w / original_width_pre_upscale
            upscale_factor = round(actual_ratio)
            if upscale_factor < 1:
                upscale_factor = 1
            if abs(actual_ratio - scale_value) > 0.1:
                print(
                    f"[UPSCALE] WARNING: Requested {scale_value}x but actual ratio is "
                    f"{actual_ratio:.2f}x ({original_width_pre_upscale}x{original_height_pre_upscale}"
                    f" -> {actual_w}x{actual_h}). Using {upscale_factor}x."
                )
        else:
            if _post_cap.isOpened():
                _post_cap.release()
            upscale_factor = int(scale_value)

        logging.info(
            "Upscaled source with Real-ESRGAN (scale=%sx, actual=%sx): %s -> %s",
            scale_value,
            upscale_factor,
            original_source_path,
            src_path,
        )

    input_path = src_path

    presets = load_presets()
    preset_key = (args.preset or "cinematic").lower()
    if preset_key not in presets:
        raise ValueError(f"Preset '{preset_key}' not found in {PRESETS_PATH}")

    preset_config = presets[preset_key]

    # Auto-tune: when enabled the planner analyses the ball trajectory
    # per-clip and picks optimal step/accel/margin/lead parameters.
    auto_tune_enabled = bool(preset_config.get("auto_tune", False))

    # In auto mode, always enable ball telemetry and adaptive tracking
    # so the pipeline works end-to-end without extra flags.
    if auto_tune_enabled:
        setattr(args, "use_ball_telemetry", True)
        # Lower the sanity threshold so high-coverage motion-centroid
        # telemetry isn't rejected when YOLO detections are sparse.
        # The CLI default is 0.6 which causes good motion-centroid
        # data (239/239 frames, 0.98 conf) to be discarded when YOLO
        # only has a handful of detections.
        if getattr(args, "ball_min_sanity", None) is None:
            setattr(args, "ball_min_sanity", 0.10)
        # Ensure follow.adaptive.enabled is set
        follow_raw = preset_config.get("follow")
        if isinstance(follow_raw, dict):
            adaptive_block = follow_raw.get("adaptive")
            if isinstance(adaptive_block, dict):
                adaptive_block.setdefault("enabled", True)
            elif adaptive_block is None:
                follow_raw["adaptive"] = {"enabled": True}

    if getattr(args, "no_draw_ball", False):
        setattr(args, "draw_ball", False)
        setattr(args, "debug_ball_overlay", False)
    elif preset_key == "segment_smooth" and not getattr(args, "draw_ball", False):
        setattr(args, "draw_ball", True)

    if getattr(args, "draw_ball", False):
        setattr(args, "use_ball_telemetry", True)

    if getattr(args, "debug_ball_overlay", False):
        setattr(args, "use_ball_telemetry", True)

    fps_in = float(ffprobe_fps(input_path))
    fps_out = float(args.fps) if args.fps is not None else float(preset_config.get("fps", fps_in))
    if fps_out <= 0:
        fps_out = fps_in if fps_in > 0 else 30.0

    # --- Ensure duration_s always exists ---
    duration_s = None
    frame_count = None

    follow_config_raw = preset_config.get("follow")
    follow_config: Mapping[str, object] = {}
    if isinstance(follow_config_raw, Mapping):
        follow_config = follow_config_raw

    ball_cam_config_raw = preset_config.get("ball_cam")
    ball_cam_config: dict[str, object] = {}
    if isinstance(ball_cam_config_raw, Mapping):
        ball_cam_config = dict(ball_cam_config_raw)

    if preset_key in {"wide_follow", "segment_smooth"}:
        ball_cam_config.setdefault("ball_cam_enabled", True)
        ball_cam_config.setdefault("ball_cam_mode", "strict_lock")

        # Basic lock & smoothing parameters
        ball_cam_config.setdefault("ball_cam_lead_frames", 3)
        ball_cam_config.setdefault("ball_cam_smooth_window", 5)  # moving average window
        ball_cam_config.setdefault("ball_cam_max_speed_px", 120.0)  # generous; avoid visible lag

        # Margin: how much space between ball and crop edge (in pixels)
        ball_cam_config.setdefault("ball_cam_margin_px", 80.0)

        # Vertical bias: where to put the ball inside the portrait frame
        # 0.5 = dead center; 0.6 = slightly lower than center (more space above)
        ball_cam_config.setdefault("ball_cam_vertical_pos", 0.60)

    portrait_w = getattr(args, "portrait_w", None)
    portrait_h = getattr(args, "portrait_h", None)
    preset_portrait, portrait_min_box, portrait_horizon_lock = portrait_config_from_preset(
        preset_config.get("portrait")
    )
    portrait: Optional[Tuple[int, int]] = None
    if portrait_w is not None and portrait_h is not None:
        portrait = (portrait_w, portrait_h)
    elif args.portrait:
        portrait = parse_portrait(args.portrait)
        if portrait:
            portrait_w, portrait_h = portrait
    elif preset_portrait:
        portrait = preset_portrait
        portrait_w, portrait_h = portrait
    if portrait_w is not None:
        setattr(args, "portrait_w", portrait_w)
    if portrait_h is not None:
        setattr(args, "portrait_h", portrait_h)

    plan_lookahead_arg = getattr(args, "plan_lookahead", None)
    if plan_lookahead_arg is not None:
        lookahead = int(plan_lookahead_arg)
    else:
        lookahead = int(preset_config.get("lookahead", 18))
        if getattr(args, "_follow_lookahead_cli", False):
            lookahead = int(args.follow_lookahead)
    lookahead = max(0, lookahead)
    smoothing_default = preset_config.get("smoothing", 0.65)
    follow_smoothing = follow_config.get("smoothing") if follow_config else None
    smoothing = float(args.smoothing) if args.smoothing is not None else float(smoothing_default)
    pad = float(args.pad) if args.pad is not None else float(preset_config.get("pad", 0.22))
    speed_limit = float(args.speed_limit) if args.speed_limit is not None else float(preset_config.get("speed_limit", 480))
    zoom_min = float(args.zoom_min) if args.zoom_min is not None else float(preset_config.get("zoom_min", 1.0))
    zoom_max = float(args.zoom_max) if args.zoom_max is not None else float(preset_config.get("zoom_max", 2.2))

    ### PATCH START: Safe cy_frac initialization

    # Initialize cy_frac safely
    if hasattr(args, "cy_frac") and args.cy_frac is not None:
        cy_frac = float(args.cy_frac)
    else:
        # Default: keep subject ~55% down the frame (good for portrait soccer tracking)
        cy_frac = 0.55

    # Validate
    try:
        if not math.isfinite(cy_frac) or cy_frac <= 0 or cy_frac >= 1:
            cy_frac = 0.55
    except Exception:
        cy_frac = 0.55

    ### PATCH END

    if not math.isfinite(cy_frac):
        cy_frac = 0.46
    cy_frac = float(np.clip(cy_frac, 0.0, 1.0))
    crf = int(args.crf) if args.crf is not None else int(preset_config.get("crf", 19))
    keyint_factor = int(args.keyint_factor) if args.keyint_factor is not None else int(preset_config.get("keyint_factor", 4))

    controller_config_raw = follow_config.get("controller") if follow_config else None
    controller_config: Mapping[str, object] = {}
    if isinstance(controller_config_raw, Mapping):
        controller_config = controller_config_raw

    def _controller_value(key: str) -> Optional[object]:
        if key in controller_config:
            return controller_config[key]
        preset_key_name = f"follow_{key}"
        if preset_key_name in preset_config:
            return preset_config[preset_key_name]
        return None

    def _controller_float(name, default):
        """
        Safe float reader for controller parameters.
        Returns sanitized defaults when args or values are missing or invalid.
        """
        val = getattr(args, name, None)
        if val is None:
            return default

        try:
            x = float(val)
            if not math.isfinite(x):
                return default
            return x
        except Exception:
            return default

    def _controller_optional_float(key: str, fallback: Optional[float]) -> Optional[float]:
        value = _controller_value(key)
        if value is None:
            return fallback
        if isinstance(value, str) and value.strip().lower() in {"none", "", "null"}:
            return None

    def _controller_int(key: str, fallback: int) -> int:
        value = _controller_value(key)
        if value is None:
            return fallback

    follow_zeta = (
        float(args.follow_zeta)
        if args.follow_zeta is not None
        else _controller_float("zeta", FOLLOW_DEFAULTS["zeta"])
    )
    follow_wn = (
        float(args.follow_wn)
        if args.follow_wn is not None
        else _controller_float("wn", FOLLOW_DEFAULTS["wn"])
    )
    follow_deadzone = max(
        0.0,
        _safe(
            _controller_float("deadzone", FOLLOW_DEFAULTS["deadzone"]),
            FOLLOW_DEFAULTS["deadzone"],
        ),
    )
    follow_max_vel = (
        float(args.max_vel)
        if args.max_vel is not None
        else _controller_optional_float("max_vel", FOLLOW_DEFAULTS["max_vel"])
    )
    follow_max_acc = (
        float(args.max_acc)
        if args.max_acc is not None
        else _controller_optional_float("max_acc", FOLLOW_DEFAULTS["max_acc"])
    )
    # ---- SAFE LOOKAHEAD ----
    raw_lookahead = getattr(args, "follow_lookahead", None)

    try:
        follow_lookahead_value = float(raw_lookahead)
        if not math.isfinite(follow_lookahead_value):
            raise ValueError()
    except Exception:
        follow_lookahead_value = FOLLOW_DEFAULTS.get("lookahead", 3.0)

    follow_lookahead_frames = max(0, int(follow_lookahead_value))
    follow_pre_smooth = (
        float(np.clip(float(args.pre_smooth), 0.0, 1.0))
        if args.pre_smooth is not None
        else float(
            np.clip(
                _controller_float("pre_smooth", FOLLOW_DEFAULTS["pre_smooth"]),
                0.0,
                1.0,
            )
        )
    )

    margin_px = 0.0
    margin_val = follow_config.get("margin_px") if follow_config else None
    if margin_val is not None:
        margin_px = float(margin_val)

    keepinview_margin = max(96.0, margin_px)
    keepinview_nudge = 0.6
    keepinview_zoom_gain = 0.55
    keepinview_zoom_cap = 1.8
    keepinview_cfg = follow_config.get("keepinview") if follow_config else None
    # Apply preset-level keepinview overrides (e.g. cinematic uses gentle values)
    if keepinview_cfg and isinstance(keepinview_cfg, dict):
        if "nudge_gain" in keepinview_cfg:
            keepinview_nudge = float(keepinview_cfg["nudge_gain"])
        if "zoom_gain" in keepinview_cfg:
            keepinview_zoom_gain = float(keepinview_cfg["zoom_gain"])
        if "zoom_cap" in keepinview_cfg:
            keepinview_zoom_cap = float(keepinview_cfg["zoom_cap"])
        if "margin_px" in keepinview_cfg:
            keepinview_margin = max(96.0, float(keepinview_cfg["margin_px"]))

    keepinview_margin_arg = getattr(args, "keepinview_margin", None)

    plan_override_data: Optional[dict[str, np.ndarray]] = None
    plan_override_len = 0
    plan_arg = getattr(args, "plan", None)
    if plan_arg:
        plan_path = Path(plan_arg).expanduser()
        if not plan_path.exists():
            raise FileNotFoundError(f"Plan file not found: {plan_path}")
        keyframes, _ = load_plan(plan_path)
        plan_override_data = keyframes_to_arrays(keyframes)
        plan_override_len = len(keyframes)
        logging.info(
            "Loaded camera plan %s (%s keyframes)",
            plan_path,
            plan_override_len,
        )
    keepinview_nudge_arg = getattr(args, "keepinview_nudge", None)
    keepinview_zoom_arg = getattr(args, "keepinview_zoom", None)
    keepinview_zoom_cap_arg = getattr(args, "keepinview_zoom_cap", None)
    keepinview_zoom_cap_override = False
    # CLI args override preset keepinview values
    if keepinview_nudge_arg is not None:
        keepinview_nudge = float(keepinview_nudge_arg)
    if keepinview_zoom_arg is not None:
        keepinview_zoom_gain = float(keepinview_zoom_arg)
    if keepinview_zoom_cap_arg is not None:
        keepinview_zoom_cap = float(keepinview_zoom_cap_arg)
        keepinview_zoom_cap_override = True

    zoom_out_max_default = follow_config.get("zoom_out_max") if follow_config else None
    follow_zoom_out_max = 1.35
    if getattr(args, "zoom_out_max", None) is not None:
        follow_zoom_out_max = max(1.0, float(args.zoom_out_max))

    if not keepinview_zoom_cap_override:
        keepinview_zoom_cap = min(keepinview_zoom_cap, follow_zoom_out_max)
    keepinview_zoom_cap = max(1.0, float(keepinview_zoom_cap))

    zoom_edge_frac_default = follow_config.get("zoom_edge_frac") if follow_config else None
    follow_zoom_edge_frac = 0.80
    if getattr(args, "zoom_edge_frac", None) is not None:
        follow_zoom_edge_frac = float(args.zoom_edge_frac)
    lost_use_motion = bool(getattr(args, "lost_use_motion", False))
    lost_hold_ms = getattr(args, "lost_hold_ms", 500)
    lost_pan_ms = getattr(args, "lost_pan_ms", 1200)
    lost_lookahead_s = getattr(args, "lost_lookahead_s", 6.0)
    lost_chase_motion_ms = getattr(args, "lost_chase_motion_ms", 900)
    lost_motion_thresh = getattr(args, "lost_motion_thresh", 1.6)

    lead_time_s = 0.0
    lead_val = follow_config.get("lead_time") if follow_config else None
    if lead_val is not None:
        lead_time_s = float(lead_val)
    lead_frames = int(round(lead_time_s * fps_in)) if fps_in > 0 else 0

    speed_zoom_value = follow_config.get("speed_zoom") if follow_config else None
    speed_zoom_config = speed_zoom_value if isinstance(speed_zoom_value, Mapping) else None

    # Adaptive tracking: per-frame modulation of margin/lead/speed based on
    # ball velocity.  Enabled via follow.adaptive.enabled in preset YAML.
    adaptive_raw = follow_config.get("adaptive") if follow_config else None
    adaptive_tracking_enabled = False
    adaptive_tracking_cfg: dict = {}
    if isinstance(adaptive_raw, Mapping):
        adaptive_tracking_enabled = bool(adaptive_raw.get("enabled", False))
        adaptive_tracking_cfg = dict(adaptive_raw)
    elif isinstance(adaptive_raw, bool):
        adaptive_tracking_enabled = adaptive_raw

    default_ball_key_x = "bx_stab"
    default_ball_key_y = "by_stab"
    keys_value = follow_config.get("keys") if follow_config else None

    if not getattr(args, "ball_key_x", None):
        setattr(args, "ball_key_x", default_ball_key_x)
    if not getattr(args, "ball_key_y", None):
        setattr(args, "ball_key_y", default_ball_key_y)

    # Deterministic naming: reruns overwrite.
    if args.out:
        requested_output = Path(args.out)
        normalized_name = normalize_tags_in_stem(requested_output.stem)
        output_path = requested_output.with_name(f"{normalized_name}{requested_output.suffix}")
    else:
        output_path = _default_output_path(original_source_path, preset_key, portrait=bool(portrait))
    output_path = output_path.expanduser().resolve()
    if getattr(args, "no_clobber", False):
        if output_path.exists() and output_path.stat().st_size > 0:
            logging.info("[SKIP] Output exists: %s", output_path)
            return

    base_scratch_root = _scratch_root(getattr(args, "scratch_root", None))
    base_scratch_root.mkdir(parents=True, exist_ok=True)
    run_key = _derive_run_key(original_source_path)
    scratch_root = base_scratch_root / run_key
    scratch_root.mkdir(parents=True, exist_ok=True)
    scratch_cleanup_paths: list[Path] = [scratch_root]

    labels_root = args.labels_root or "out/yolo"
    label_files = find_label_files(original_source_path.stem, labels_root)

    log_dict: dict[str, object] = {}

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError("Unable to open input video for metadata extraction.")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()

    if duration_s is None:
        duration_s = (frame_count / fps_in) if (frame_count and fps_in and fps_in > 0) else 0.0

    if duration_s <= 0 and frame_count and frame_count > 0 and fps_in and fps_in > 0:
        duration_s = frame_count / float(fps_in)
    if duration_s <= 0 and frame_count > 0:
        fallback_fps = fps_in if fps_in > 0 else 30.0
        duration_s = frame_count / float(fallback_fps)

    follow_crop_width = width
    if portrait_w and portrait_h and portrait_w < portrait_h:
        follow_crop_width = int(round(height * 9.0 / 16.0))
        follow_crop_width = max(1, min(follow_crop_width, width))

    override_samples = getattr(args, "follow_override_samples", None)
    total_frames = frame_count
    if total_frames <= 0:
        fps_hint = fps_in if fps_in > 0 else fps_out
        total_frames = int(round((duration_s or 0.0) * fps_hint)) if fps_hint > 0 else 0
    total_frames = max(total_frames, 1)

    renderer: Optional[Renderer]
    ball_samples: List[BallSample] = []
    fallback_ball_samples: List[BallSample] = []
    keep_path_lookup_data: dict[int, Tuple[float, float]] = {}
    keepinview_path: list[tuple[float, float]] | None = None
    follow_telemetry_path: str | None = None
    use_ball_telemetry = bool(getattr(args, "use_ball_telemetry", False))
    # In portrait mode the whole point is to follow the ball/action,
    # so enable ball telemetry automatically unless explicitly disabled.
    if portrait is not None and not use_ball_telemetry:
        use_ball_telemetry = True
    use_red_fallback = bool(getattr(args, "ball_fallback_red", False))
    telemetry_path: Path | None = None
    offline_ball_path: Optional[List[Optional[dict[str, float]]]] = None

    disable_controller = bool(getattr(args, "disable_controller", False))
    def _add_follow_override_row(target: dict[int, dict[str, float]], row: Mapping[str, object]):
        frame_val = row.get("frame")
        if frame_val is None:
            return
        try:
            frame_idx = int(frame_val)
        except Exception:
            return

        cx_val = safe_float(row.get("cx"), None)
        cy_val = safe_float(row.get("cy"), None)
        zoom_val = safe_float(row.get("zoom", 1.0), 1.0)
        t_val = safe_float(row.get("t"), None)

        target[frame_idx] = {
            "t": t_val,
            "cx": cx_val,
            "cy": cy_val,
            "zoom": zoom_val,
        }

    follow_override_map: Optional[Mapping[int, Mapping[str, float]]] = None
    if args.follow_override:
        follow_override_map = {}
        with open(args.follow_override, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                _add_follow_override_row(follow_override_map, row)
    elif override_samples:
        follow_override_map = {}
        for row in override_samples:
            _add_follow_override_row(follow_override_map, row)
    elif "follow_frames" in locals() and len(follow_frames) > 0:
        follow_override_map = {}
        for f in follow_frames:
            if f.get("frame") is None:
                continue
            _add_follow_override_row(follow_override_map, f)

    if follow_override_map is not None:
        follow_override_map = _normalize_follow_override_map(follow_override_map, total_frames, fps_in)

    if follow_override_map is not None:
        keep_path_lookup_data = {}
        use_ball_telemetry = False

    if override_samples:
        use_ball_telemetry = False

    follow_mode = "option_c"
    log_dict["follow_mode"] = follow_mode

    if getattr(args, "telemetry", None):
        telemetry_rows = load_any_telemetry(args.telemetry)

        # === FOLLOW TELEMETRY NORMALIZATION (NEW) ===
        follow_frames = []
        for row in telemetry_rows:
            if not row.get("valid", True):
                continue
            cx = row.get("cx")
            cy = row.get("cy")
            zoom = row.get("zoom", 1.0)
            t = row.get("t")
            frame = row.get("frame")

            # Reject unusable rows
            if cx is None or cy is None or t is None:
                continue

            follow_frames.append({
                "t": float(t),
                "frame": int(frame) if frame is not None else None,
                "cx": float(cx),
                "cy": float(cy),
                "zoom": float(zoom),
            })

        if not follow_frames:
            raise ValueError(f"No valid follow telemetry in {args.telemetry}")

    telemetry_coverage = 0
    telemetry_coverage_ratio = 0.0
    ball_cam_stats: dict[str, float] = {}
    fallback_active = False
    if use_ball_telemetry:
        telemetry_in = getattr(args, "ball_telemetry", None)
        telemetry_path = Path(telemetry_in).expanduser() if telemetry_in else Path(telemetry_path_for_video(input_path))
        # Always regenerate auto-telemetry (motion-centroid) unless the user
        # explicitly supplied a telemetry file via --ball-telemetry.
        if telemetry_in and telemetry_path.is_file():
            logger.info("[MOTION] Using user-supplied telemetry: %s", telemetry_path)
        else:
            logger.info("[MOTION] Generating motion-centroid telemetry at %s", telemetry_path)
            write_red_ball_telemetry(
                input_path,
                telemetry_path,
                fps_in if fps_in > 0 else fps_out,
                logger,
            )

        if telemetry_path.is_file():
            set_telemetry_frame_bounds(width, height)
            total_frames = frame_count
            if total_frames <= 0:
                fps_hint = fps_in if fps_in > 0 else fps_out
                total_frames = int(round((duration_s or 0.0) * fps_hint)) if fps_hint > 0 else 0
            total_frames = max(total_frames, 1)

            plan_data, ball_cam_stats = build_ball_cam_plan(
                telemetry_path,
                num_frames=total_frames,
                fps=fps_in if fps_in > 0 else fps_out,
                frame_width=width,
                frame_height=height,
                portrait_width=int(portrait[0]) if portrait else 1080,
                config=ball_cam_config,
                preset_name=preset_key,
                in_path=input_path,
                out_path=output_path,
                min_sanity=float(getattr(args, "ball_min_sanity", None) or 0.60),
                use_red_fallback=use_red_fallback,
                scratch_root=scratch_root,
            )
            debug_overlay_path = ball_cam_stats.get("debug_overlay_path")
            if debug_overlay_path:
                scratch_cleanup_paths.append(Path(debug_overlay_path))
            if plan_data is None and use_red_fallback and ball_cam_stats.get("sanity_low"):
                fallback_path = telemetry_path.with_suffix(".ball.red.jsonl")
                if not fallback_path.is_file():
                    logger.info("[BALL-FALLBACK-RED] Building HSV red-ball telemetry at %s", fallback_path)
                    write_red_ball_telemetry(
                        input_path,
                        fallback_path,
                        fps_in if fps_in > 0 else fps_out,
                        logger,
                    )
                if fallback_path.is_file():
                    fallback_plan, fallback_stats = build_ball_cam_plan(
                        fallback_path,
                        num_frames=total_frames,
                        fps=fps_in if fps_in > 0 else fps_out,
                        frame_width=width,
                        frame_height=height,
                        portrait_width=int(portrait[0]) if portrait else 1080,
                        config=ball_cam_config,
                        preset_name=preset_key,
                        in_path=input_path,
                        out_path=output_path,
                        min_sanity=0.0,
                        use_red_fallback=False,
                        scratch_root=scratch_root,
                    )
                    debug_overlay_path = fallback_stats.get("debug_overlay_path")
                    if debug_overlay_path:
                        scratch_cleanup_paths.append(Path(debug_overlay_path))
                    if fallback_plan is not None:
                        logger.info("[BALL-FALLBACK-RED] Using HSV red-ball fallback telemetry for planning")
                        plan_data = fallback_plan
                        ball_cam_stats = fallback_stats
                        telemetry_path = fallback_path
                        fallback_active = True
            telemetry_coverage_ratio = float(ball_cam_stats.get("coverage", 0.0))
            telemetry_coverage = int(round(telemetry_coverage_ratio * total_frames))
            if plan_data is None:
                use_ball_telemetry = False
                keep_path_lookup_data = {}
            else:
                cx_vals = plan_data.get("cx")
                cy_vals = plan_data.get("cy")
                if cx_vals is not None and cy_vals is not None:
                    keep_path_lookup_data = {
                        idx: (float(cx_vals[idx]), float(cy_vals[idx])) for idx in range(len(cx_vals))
                    }
                if plan_override_data is None:
                    plan_override_data = {k: v for k, v in plan_data.items() if k in {"x0", "y0", "w", "h", "spd", "z"}}
                    plan_override_len = len(next(iter(plan_data.values()))) if plan_data else 0
                if telemetry_path is not None:
                    cx_vals = plan_data.get("cx")
                    cy_vals = plan_data.get("cy")
                    zoom_vals = plan_data.get("z")
                    if cx_vals is not None and cy_vals is not None and zoom_vals is not None:
                        follow_telemetry_path = emit_follow_telemetry(
                            telemetry_path,
                            cx_vals,
                            cy_vals,
                            zoom_vals,
                            basename=telemetry_path.stem,
                        )
                        follow_telemetry_path = smooth_follow_telemetry(follow_telemetry_path)
                        log_dict["follow_telemetry"] = follow_telemetry_path
                ball_samples = load_ball_telemetry(telemetry_path)
                # Telemetry from this path was generated from the upscaled
                # video (write_red_ball_telemetry uses input_path), so
                # coordinates are already in upscaled space.  Do NOT scale.
                if fallback_active:
                    fallback_ball_samples = list(ball_samples)
                jerk_stat = ball_cam_stats.get("jerk95_cam", ball_cam_stats.get("jerk95", 0.0))
                jerk_raw = ball_cam_stats.get("jerk95_raw", 0.0)
                ball_in_crop_pct = ball_cam_stats.get("ball_in_crop_pct")
                ball_in_crop_frames = int(ball_cam_stats.get("ball_in_crop_frames", 0))
                telemetry_conf = float(ball_cam_stats.get("telemetry_conf", 1.0))
                telemetry_quality = float(ball_cam_stats.get("telemetry_quality", 1.0))
                cx_range = (float(np.nanmin(plan_data.get("cx", np.array([0.0])))), float(np.nanmax(plan_data.get("cx", np.array([0.0])))))
                logger.info(
                    "[BALL-CAM] coverage: %.1f%% (conf=%.2f, quality=%.2f), path: %d frames, cx range=[%.1f, %.1f], jerk95_raw=%.1f, jerk95_cam=%.1f px/s^3",
                    telemetry_coverage_ratio * 100.0,
                    telemetry_conf,
                    telemetry_quality,
                    len(plan_data.get("cx", [])),
                    cx_range[0],
                    cx_range[1],
                    jerk_raw,
                    jerk_stat,
                )
                if telemetry_quality < 0.6:
                    logger.info("[FORCE-BALL-FOLLOW] Using ball telemetry even if weak")
                if ball_in_crop_pct is not None and plan_data is not None:
                    logger.info(
                        "[BALL-CAM] ball_in_crop_horiz: %.1f%% (%d/%d)",
                        ball_in_crop_pct,
                        ball_in_crop_frames,
                        len(plan_data.get("cx", [])),
                    )
        else:
            logger.info("[BALL-CAM] No ball telemetry found for %s; reactive follow", input_path)
            use_ball_telemetry = False

    raw_points = load_labels(label_files, width, height, fps_in)
    log_dict["labels_raw_count"] = len(raw_points)
    if raw_points:
        max_label_time = max(point[0] for point in raw_points)
        if duration_s <= max_label_time:
            frame_step = 1.0 / float(fps_in) if fps_in > 0 else 0.0
            duration_s = max_label_time + frame_step

    # Resample at source fps so we get one position per source frame.
    # write_frames reads every source frame, so states must match fps_in.
    render_fps_for_plan = fps_in if fps_in > 0 else fps_out
    label_pts = resample_labels_by_time(raw_points, render_fps_for_plan, duration_s)

    def _rng(arr):
        xs = [a[1] for a in arr]
        ys = [a[2] for a in arr]
        return (min(xs), max(xs), min(ys), max(ys)) if arr else None

    log_dict["labels_resampled_count"] = len(label_pts)
    log_dict["labels_resampled_range"] = _rng(label_pts)

    positions, used_mask = labels_to_positions(label_pts, render_fps_for_plan, duration_s, raw_points)

    if len(positions) == 0 and frame_count > 0:
        positions = np.full((frame_count, 2), np.nan, dtype=np.float32)
        used_mask = np.zeros(frame_count, dtype=bool)

    # --- YOLO ball detection + fusion with motion centroid ---
    # Run real YOLO ball detection on the source video to get actual
    # ball positions.  Fuse with the motion-centroid telemetry using
    # confidence weighting: YOLO is preferred when confident, centroid
    # fills gaps.  The per-frame confidence drives dynamic zoom (zoom
    # out when uncertain to keep more field visible).
    valid_label_count = int(used_mask.sum()) if len(used_mask) > 0 else 0
    fusion_confidence: Optional[np.ndarray] = None
    fusion_source_labels: Optional[np.ndarray] = None

    if valid_label_count < max(1, len(used_mask)) * 0.1:
        # YOLO label files are sparse/absent — run real-time YOLO detection
        yolo_samples = run_yolo_ball_detection(
            original_source_path,
            min_conf=0.20,
            cache=True,
        )

        # Scale YOLO detections if the source was upscaled
        if yolo_samples and upscale_factor > 1:
            for _ys in yolo_samples:
                _ys.x *= upscale_factor
                _ys.y *= upscale_factor

        if yolo_samples or ball_samples:
            # Load exclusion zones if provided via CLI
            _exclude_zones: list[ExcludeZone] | None = None
            _yolo_exclude_path = getattr(args, "yolo_exclude", None)
            if _yolo_exclude_path:
                _exclude_zones = load_exclude_zones(_yolo_exclude_path)
                if _exclude_zones:
                    print(f"[FUSION] Loaded {len(_exclude_zones)} exclusion zone(s) from {_yolo_exclude_path}")

            # Fuse YOLO + centroid → merged positions with confidence
            fused_positions, fused_mask, fusion_confidence, fusion_source_labels = fuse_yolo_and_centroid(
                yolo_samples=yolo_samples,
                centroid_samples=ball_samples,
                frame_count=len(positions),
                width=float(width),
                height=float(height),
                fps=float(render_fps_for_plan),
                exclude_zones=_exclude_zones,
            )
            # Replace positions/mask with fused result
            positions = fused_positions
            used_mask = fused_mask
            logger.info(
                "[PLANNER] Using YOLO+centroid fusion: %d/%d frames covered, "
                "avg confidence=%.2f",
                int(used_mask.sum()),
                len(used_mask),
                float(fusion_confidence[used_mask].mean()) if used_mask.any() else 0.0,
            )

            # --- Wrong-ball sanity check ---
            # When two balls are visible (e.g. adjacent field in the
            # background), the multi-ball filter may keep the wrong
            # cluster.  Detect this by checking whether the fused ball
            # position at frame 0 falls far outside the centroid
            # (player-cluster) x-range.  If so, some YOLO detections
            # are tracking a background ball — filter them out and
            # re-run fusion with only the good YOLO detections.
            if ball_samples and len(positions) > 0 and not np.isnan(positions[0]).any():
                _cs_xs = [float(s.x) for s in ball_samples
                          if hasattr(s, "x") and math.isfinite(s.x)]
                if _cs_xs:
                    _cs_min = min(_cs_xs)
                    _cs_max = max(_cs_xs)
                    _fused_x0 = float(positions[0][0])
                    _cs_margin = max(150.0, (_cs_max - _cs_min) * 0.10)
                    if _fused_x0 < _cs_min - _cs_margin or _fused_x0 > _cs_max + _cs_margin:
                        # Filter out only the YOLO detections outside
                        # the centroid x-range (the wrong-ball ones),
                        # keep the rest so we still have precise ball
                        # positions for most of the clip.
                        _lo = _cs_min - _cs_margin
                        _hi = _cs_max + _cs_margin
                        _filtered_yolo = [
                            s for s in yolo_samples
                            if _lo <= float(s.x) <= _hi
                        ]
                        _n_removed = len(yolo_samples) - len(_filtered_yolo)
                        print(
                            f"[FUSION] Wrong-ball detected: fused x0={_fused_x0:.0f} "
                            f"outside centroid range [{_cs_min:.0f}, {_cs_max:.0f}] "
                            f"(margin={_cs_margin:.0f}px) — removed {_n_removed} "
                            f"bad YOLO, re-running fusion with {len(_filtered_yolo)} good YOLO"
                        )
                        fused_positions, fused_mask, fusion_confidence, fusion_source_labels = fuse_yolo_and_centroid(
                            yolo_samples=_filtered_yolo,
                            centroid_samples=ball_samples,
                            frame_count=len(positions),
                            width=float(width),
                            height=float(height),
                            fps=float(render_fps_for_plan),
                            exclude_zones=_exclude_zones,
                        )
                        positions = fused_positions
                        used_mask = fused_mask

                        # --- Second-pass: tighter check ---
                        # If the re-run fused x0 is still outside the
                        # raw centroid range, the 150px margin let
                        # through some wrong-ball YOLO detections that
                        # now anchor the fused path.  Tighten the margin
                        # and re-filter so the FREE_KICK anchor is
                        # based on the real ball, not the background one.
                        if not np.isnan(positions[0]).any():
                            _fused_x0_2 = float(positions[0][0])
                            _tight_margin = 50.0
                            if (_fused_x0_2 < _cs_min - _tight_margin
                                    or _fused_x0_2 > _cs_max + _tight_margin):
                                _lo2 = _cs_min - _tight_margin
                                _hi2 = _cs_max + _tight_margin
                                _filtered_yolo_2 = [
                                    s for s in _filtered_yolo
                                    if _lo2 <= float(s.x) <= _hi2
                                ]
                                _n_removed_2 = len(_filtered_yolo) - len(_filtered_yolo_2)
                                if _n_removed_2 > 0 and len(_filtered_yolo_2) > 0:
                                    print(
                                        f"[FUSION] Wrong-ball pass 2: fused x0={_fused_x0_2:.0f} "
                                        f"still outside [{_cs_min:.0f}, {_cs_max:.0f}] "
                                        f"(tight margin={_tight_margin:.0f}px) — removed "
                                        f"{_n_removed_2} more bad YOLO, re-running with "
                                        f"{len(_filtered_yolo_2)} good YOLO"
                                    )
                                    fused_positions, fused_mask, fusion_confidence, fusion_source_labels = fuse_yolo_and_centroid(
                                        yolo_samples=_filtered_yolo_2,
                                        centroid_samples=ball_samples,
                                        frame_count=len(positions),
                                        width=float(width),
                                        height=float(height),
                                        fps=float(render_fps_for_plan),
                                        exclude_zones=_exclude_zones,
                                    )
                                    positions = fused_positions
                                    used_mask = fused_mask
        elif ball_samples:
            # No YOLO available, fall back to centroid-only merge
            merged = 0
            for sample in ball_samples:
                fidx = getattr(sample, "frame", None)
                sx = _safe_float(getattr(sample, "x", None))
                sy = _safe_float(getattr(sample, "y", None))
                if fidx is None or sx is None or sy is None:
                    continue
                fidx = int(fidx)
                if 0 <= fidx < len(positions) and not used_mask[fidx]:
                    positions[fidx, 0] = sx
                    positions[fidx, 1] = sy
                    used_mask[fidx] = True
                    merged += 1
            if merged > 0:
                logger.info(
                    "[PLANNER] Merged %d motion-centroid samples (no YOLO available; "
                    "YOLO labels: %d/%d frames)",
                    merged, valid_label_count, len(used_mask),
                )

    # --- YOLO person detection for player-aware zoom ---
    person_boxes_by_frame: Optional[dict[int, list[PersonBox]]] = None
    try:
        person_boxes_by_frame = run_yolo_person_detection(
            original_source_path,
            min_conf=0.30,
            cache=True,
        )
        if person_boxes_by_frame and upscale_factor > 1:
            for _frame_persons in person_boxes_by_frame.values():
                for _pb in _frame_persons:
                    _pb.cx *= upscale_factor
                    _pb.cy *= upscale_factor
                    _pb.w *= upscale_factor
                    _pb.h *= upscale_factor
    except Exception as _pe:
        logger.warning("[YOLO] Person detection failed (non-fatal): %s", _pe)
        person_boxes_by_frame = None

    if args.flip180 and len(positions) > 0:
        flipped_positions = positions.copy()
        valid_mask = ~np.isnan(flipped_positions).any(axis=1)
        if valid_mask.any():
            flipped_positions[valid_mask, 0] = float(width) - flipped_positions[valid_mask, 0]
            flipped_positions[valid_mask, 1] = float(height) - flipped_positions[valid_mask, 1]
        positions = flipped_positions

    # Forward-fill NaN gaps in positions so CameraPlanner always has
    # a usable target.  Without this, gaps in YOLO/centroid coverage
    # cause the planner to hold at a stale position, and large gaps
    # at the start of the clip leave the planner stuck at frame center.
    if len(positions) > 0:
        _ffill_count = 0
        _last_valid: Optional[np.ndarray] = None
        for _fi in range(len(positions)):
            if used_mask[_fi] and not np.isnan(positions[_fi]).any():
                _last_valid = positions[_fi].copy()
            elif _last_valid is not None:
                positions[_fi] = _last_valid.copy()
                used_mask[_fi] = True
                if fusion_confidence is not None and _fi < len(fusion_confidence):
                    fusion_confidence[_fi] = max(fusion_confidence[_fi], 0.15)
                _ffill_count += 1
        if _ffill_count > 0:
            logger.info(
                "[PLANNER] Forward-filled %d/%d NaN position gaps",
                _ffill_count, len(positions),
            )

    # ------------------------------------------------------------------
    # Post-fusion position smoothing
    # ------------------------------------------------------------------
    # The fused YOLO+centroid positions can have frame-to-frame jitter
    # when the source switches between YOLO (actual ball) and centroid
    # (player activity cluster).  These two signals can differ by
    # 50-200px, and alternating between them creates visible camera
    # oscillation that a single-pass EMA (alpha=0.55) cannot dampen.
    #
    # Apply a bidirectional EMA: forward pass + backward pass, then
    # average.  This eliminates high-frequency jitter while preserving
    # the real trajectory and adding zero directional lag.
    _n_pos = len(positions) if len(positions) > 0 else 0
    if _n_pos > 5:
        _pos_alpha_base = min(0.12, smoothing)  # suppress centroid-YOLO jitter aggressively — lower alpha = heavier smoothing

        # Build per-frame alpha: high-confidence sources (YOLO, interpolated
        # between YOLO anchors) resist being pulled by neighbours, while
        # low-confidence sources (centroid, hold) are smoothed heavily.
        # This prevents the EMA from contaminating clean YOLO-interpolated
        # positions with centroid noise (the root cause of early-clip wobble
        # on sparse-YOLO clips like free kicks).
        _per_alpha = np.full(_n_pos, _pos_alpha_base, dtype=np.float64)
        if fusion_source_labels is not None and len(fusion_source_labels) >= _n_pos:
            _FUSE_YOLO = np.uint8(1)
            _FUSE_INTERP = np.uint8(4)
            _FUSE_HOLD = np.uint8(5)
            _high_conf_alpha = 0.85  # trust own value — minimal neighbour pull
            for _si in range(_n_pos):
                _sl = fusion_source_labels[_si]
                if _sl == _FUSE_YOLO or _sl == _FUSE_INTERP:
                    _per_alpha[_si] = _high_conf_alpha
                elif _sl == _FUSE_HOLD:
                    # FUSE_HOLD frames are backward-filled copies of a
                    # YOLO anchor — they should NOT be pulled by the
                    # EMA's backward pass.  With alpha=0.85 the 15%
                    # leakage compounds across 10-50 hold frames and
                    # lets distant centroid positions drag the ball far
                    # from its true location, pushing the kick taker to
                    # the frame edge on free-kick clips.  Alpha=1.0
                    # locks them to the YOLO anchor position.
                    _per_alpha[_si] = 1.0

        # Compute max delta before smoothing (for diagnostics)
        _pre_deltas = []
        for _si in range(1, _n_pos):
            if used_mask[_si] and used_mask[_si - 1]:
                _d = float(np.linalg.norm(positions[_si] - positions[_si - 1]))
                _pre_deltas.append(_d)
        _pre_max = max(_pre_deltas) if _pre_deltas else 0.0

        # Forward EMA pass
        _fwd = positions.copy()
        for _si in range(1, _n_pos):
            if used_mask[_si] and used_mask[_si - 1]:
                _a = _per_alpha[_si]
                _fwd[_si] = _a * positions[_si] + (1.0 - _a) * _fwd[_si - 1]

        # Backward EMA pass
        _bwd = positions.copy()
        for _si in range(_n_pos - 2, -1, -1):
            if used_mask[_si] and used_mask[_si + 1]:
                _a = _per_alpha[_si]
                _bwd[_si] = _a * positions[_si] + (1.0 - _a) * _bwd[_si + 1]

        # Average forward and backward passes (zero-lag result)
        for _si in range(_n_pos):
            if used_mask[_si]:
                positions[_si] = (_fwd[_si] + _bwd[_si]) * 0.5

        # Compute max delta after smoothing
        _post_deltas = []
        for _si in range(1, _n_pos):
            if used_mask[_si] and used_mask[_si - 1]:
                _d = float(np.linalg.norm(positions[_si] - positions[_si - 1]))
                _post_deltas.append(_d)
        _post_max = max(_post_deltas) if _post_deltas else 0.0

        _high_count = int((_per_alpha > _pos_alpha_base + 0.01).sum()) if fusion_source_labels is not None else 0
        logger.info(
            "[PLANNER] Position smoothing: %d frames (%d high-conf protected), "
            "max delta %.1f->%.1f px/frame",
            _n_pos, _high_count, _pre_max, _post_max,
        )

    # Override min_box to match the actual portrait crop dimensions.
    # The preset min_box_px may be larger than the real visible area
    # (e.g., [486, 864] vs actual 405x720 for a 720p source).  If
    # CameraPlanner uses a too-large min_box, its safety margins are
    # computed against a crop wider than what's rendered, allowing the
    # ball to appear outside the visible frame.
    effective_min_box = portrait_min_box
    if follow_crop_width > 0:
        actual_crop_w = float(follow_crop_width)
        actual_crop_h = float(height)
        if portrait_min_box is not None:
            if isinstance(portrait_min_box, (list, tuple)) and len(portrait_min_box) >= 2:
                preset_w = float(portrait_min_box[0])
                if preset_w > actual_crop_w:
                    effective_min_box = (actual_crop_w, actual_crop_h)
                    logger.info(
                        "[PLANNER] min_box capped to actual portrait crop: %.0f x %.0f (was %.0f x %.0f)",
                        actual_crop_w, actual_crop_h, preset_w, float(portrait_min_box[1]),
                    )
        else:
            effective_min_box = (actual_crop_w, actual_crop_h)

    planner = CameraPlanner(
        width=width,
        height=height,
        fps=fps_in if fps_in > 0 else fps_out,
        lookahead=lookahead,
        smoothing=smoothing,
        pad=pad,
        speed_limit=speed_limit,
        zoom_min=zoom_min,
        zoom_max=zoom_max,
        portrait=portrait,
        margin_px=margin_px,
        lead_frames=lead_frames,
        speed_zoom=speed_zoom_config,
        min_box=effective_min_box,
        horizon_lock=portrait_horizon_lock,
        emergency_gain=getattr(args, "emergency_gain", 0.6),
        emergency_zoom_max=getattr(args, "emergency_zoom_max", 1.45),
        keepinview_margin_px=keepinview_margin,
        keepinview_nudge_gain=keepinview_nudge,
        keepinview_zoom_gain=keepinview_zoom_gain,
        keepinview_zoom_out_max=keepinview_zoom_cap,
        center_frac=cy_frac,
        post_smooth_sigma=float(preset_config.get("post_smooth_sigma", 0.0)),
        event_type=getattr(args, "event_type", None),
    )

    # --- FREE_KICK: identify the kick taker BEFORE the planner runs ---
    # The planner's scorer-memory logic normally tracks the nearest player
    # to the ball during flight.  For a free kick, that's a defender or
    # goalkeeper — not the kicker.  Pre-calculate the kicker position and
    # pass it to the planner so it tracks the right person for celebration.
    _snap_event = getattr(args, "event_type", None) or ""
    planner._free_kick_scorer_pos = None
    if _snap_event == "FREE_KICK" and len(positions) > 0 and not np.isnan(positions[0]).any():
        _pre_ball_x = float(positions[0][0])
        _pre_ball_y = float(positions[0][1])
        if person_boxes_by_frame:
            _pre_kicker_dists = []
            for _pf in range(min(12, len(positions))):
                if _pf in person_boxes_by_frame:
                    for _pb in person_boxes_by_frame[_pf]:
                        _d = math.hypot(_pb.cx - _pre_ball_x, _pb.cy - _pre_ball_y)
                        if _d < 200.0:
                            _pre_kicker_dists.append((_d, float(_pb.cx), float(_pb.cy)))
            if _pre_kicker_dists:
                _pre_kicker_dists.sort()
                _top_n = min(5, len(_pre_kicker_dists))
                planner._free_kick_scorer_pos = (
                    float(np.median([kd[1] for kd in _pre_kicker_dists[:_top_n]])),
                    float(np.median([kd[2] for kd in _pre_kicker_dists[:_top_n]])),
                )
    if not ball_samples:
        ball_samples = load_ball_telemetry_for_clip(str(original_source_path))
        if ball_samples and upscale_factor > 1:
            for _s in ball_samples:
                _s.x *= upscale_factor
                _s.y *= upscale_factor
        if not ball_samples:
            expected_path = Path(telemetry_path_for_video(original_source_path))
            print(f"[BALL] No ball telemetry found (expected {expected_path})")
    num_frames = frame_count
    fps = render_fps_for_plan
    print(f"[DEBUG] num_frames={num_frames} fps={fps} positions={len(positions)} duration={duration_s if 'duration_s' in locals() else 'n/a'}")
    print(f"[DEBUG] ball_samples={len(ball_samples) if 'ball_samples' in locals() else 'n/a'}")
    states = planner.plan(positions, used_mask, confidence=fusion_confidence, person_boxes=person_boxes_by_frame)

    # --- STARTUP SNAP + KICK-HOLD: keep camera on ball and kick taker ---
    # For FREE_KICK clips, the viewer needs to see the kick taker
    # through the moment of the kick.  We detect when the ball first
    # moves significantly (the kick), then HOLD the camera on the
    # kicker for a beat so the viewer registers the kick, then
    # smoothly ease the camera toward the planner's trajectory.
    #
    # For other events, a short snap (20 frames) corrects the initial
    # Gaussian pull when the gap exceeds 50 px.
    _snap_event = getattr(args, "event_type", None) or ""
    _snap_fps = float(fps_in if fps_in and fps_in > 0 else fps_out or 30.0)
    _kick_hold_end = 0       # frames 0.._kick_hold_end are locked on kicker (hold phase only)
    _kick_hold_trans_end = 0  # frames 0.._kick_hold_trans_end are protected from gravity clamp
    if states and len(positions) > 0:
        _snap_ball_x0 = float(positions[0][0]) if not np.isnan(positions[0]).any() else None
        _snap_cam_x0 = float(states[0].cx) if states else None

        _is_free_kick = (_snap_event == "FREE_KICK")
        _snap_gap = abs(_snap_ball_x0 - _snap_cam_x0) if (_snap_ball_x0 is not None and _snap_cam_x0 is not None) else 0.0

        if _is_free_kick or _snap_gap > 50.0:
            _snap_fw = float(width) if width > 0 else 1920.0

            if _is_free_kick and _snap_ball_x0 is not None:
                # --- FREE_KICK hold-then-follow ---
                # Phase 1 (pre-kick + hold): lock camera on kicker
                # Phase 2 (transition): cubic ease from kicker to planner
                # Phase 3: planner takes over

                # Find the actual kicker using person detections.
                # The kicker is the nearest person to the ball in
                # pre-kick frames (averaged over first 10 frames).
                _ball_x0 = _snap_ball_x0
                _ball_y0 = float(positions[0][1]) if not np.isnan(positions[0]).any() else 540.0
                _kicker_cx = None
                if person_boxes_by_frame:
                    _kicker_dists = []  # (dist, person_cx) pairs
                    for _pf in range(min(12, len(positions))):
                        if _pf in person_boxes_by_frame:
                            for _pb in person_boxes_by_frame[_pf]:
                                _d = ((float(_pb.cx) - _ball_x0) ** 2 + (float(_pb.cy) - _ball_y0) ** 2) ** 0.5
                                if _d < 200.0:  # within 200px of ball
                                    _kicker_dists.append((_d, float(_pb.cx)))
                    if _kicker_dists:
                        # Take the median cx of the closest-person candidates
                        _kicker_dists.sort()
                        _top_n = min(5, len(_kicker_dists))
                        _kicker_cx = float(np.median([kd[1] for kd in _kicker_dists[:_top_n]]))

                # ---- self-tuning FREE_KICK camera strategy ----
                # Try the default kicker-anchor strategy, evaluate ball-
                # in-crop at the plan level, and retry with a ball-
                # centric anchor if the ball escapes too severely.  This
                # removes the need to hand-triage edge-constrained clips.

                _pcw = float(follow_crop_width) if follow_crop_width > 0 else _snap_fw

                # Detect kick frame once (used by all strategies)
                _kick_frame = None
                _kick_threshold = 80.0
                for _kf in range(1, min(len(positions), int(_snap_fps * 6))):
                    if not np.isnan(positions[_kf]).any():
                        if abs(float(positions[_kf][0]) - _ball_x0) > _kick_threshold:
                            _kick_frame = _kf
                            break
                if _kick_frame is None:
                    _kick_frame = int(_snap_fps * 2)

                def _apply_fk_strategy(
                    anchor_x: float,
                    hold_s: float,
                    trans_s: float,
                    label: str,
                ) -> tuple:
                    """Apply a hold-then-follow strategy and return plan-
                    level ball-in-crop metrics without rendering.

                    Returns (hold_end, trans_end, outside_count,
                             total_checked, max_escape_px, label).
                    """
                    _hf = int(_snap_fps * hold_s)
                    _tf = int(_snap_fps * trans_s)

                    # --- ball-escape cap: shorten hold if ball leaves
                    # the crop before the hold expires ---
                    _h_crop_left = float(np.clip(
                        anchor_x - _pcw / 2.0, 0.0,
                        max(0.0, _snap_fw - _pcw),
                    ))
                    _h_crop_right = _h_crop_left + _pcw
                    _h_margin = _pcw * 0.06  # 6 % inset so we react before full escape
                    _nominal_hold_end = _kick_frame + _hf
                    for _bf in range(_kick_frame, min(_nominal_hold_end, len(positions))):
                        if not np.isnan(positions[_bf]).any():
                            _bfx = float(positions[_bf][0])
                            if _bfx < _h_crop_left + _h_margin or _bfx > _h_crop_right - _h_margin:
                                # Ball is about to leave — cap hold here
                                _hf = max(3, _bf - _kick_frame)
                                label = label + "+capped"
                                break

                    _he = _kick_frame + _hf
                    _te = min(_he + _tf, len(states) - 1)

                    # Write hold phase
                    for _si in range(min(_he, len(states))):
                        states[_si].cx = anchor_x
                        _hcw = states[_si].crop_w / 2.0
                        states[_si].x0 = float(np.clip(
                            anchor_x - _hcw, 0.0,
                            max(0.0, _snap_fw - states[_si].crop_w),
                        ))

                    # Write transition phase
                    if _te > _he and _te < len(states):
                        for _si in range(_he, _te):
                            _t = (_si - _he) / float(_te - _he)
                            _ease = 1.0 - (1.0 - _t) ** 2
                            if _si < len(positions) and not np.isnan(positions[_si]).any():
                                _tgt = float(positions[_si][0])
                            else:
                                _tgt = float(states[_si].cx)
                            _ncx = (1.0 - _ease) * anchor_x + _ease * _tgt
                            states[_si].cx = _ncx
                            _hcw = states[_si].crop_w / 2.0
                            states[_si].x0 = float(np.clip(
                                _ncx - _hcw, 0.0,
                                max(0.0, _snap_fw - states[_si].crop_w),
                            ))

                    # --- plan-level ball-in-crop evaluation ---
                    # Only count reliable frames (YOLO, blended, interp,
                    # hold) to avoid inflated scores from centroid-only
                    # frames that self-validate against planner positions.
                    _fk_rel = {1, 3, 4, 5}
                    _out = 0
                    _tot = 0
                    _max_esc = 0.0
                    _eval_end = min(_te + int(_snap_fps * 2), len(states))
                    for _ei in range(_eval_end):
                        if _ei >= len(positions) or np.isnan(positions[_ei]).any():
                            continue
                        if fusion_source_labels is not None and _ei < len(fusion_source_labels):
                            if int(fusion_source_labels[_ei]) not in _fk_rel:
                                continue
                        elif fusion_source_labels is not None:
                            continue
                        _ebx = float(positions[_ei][0])
                        _ecx = float(states[_ei].cx)
                        _ecw = float(states[_ei].crop_w) if states[_ei].crop_w > 0 else _pcw
                        _el = float(np.clip(
                            _ecx - _ecw / 2.0, 0.0,
                            max(0.0, _snap_fw - _ecw),
                        ))
                        _er = _el + _ecw
                        _tot += 1
                        if _ebx < _el or _ebx > _er:
                            _out += 1
                            _max_esc = max(_max_esc, max(_el - _ebx, _ebx - _er))

                    return (_he, _te, _out, _tot, _max_esc, label)

                # --- strategy candidates ---
                _strategies: list[tuple] = []  # (anchor, hold_s, trans_s, label)

                # Strategy A: kicker anchor, standard timing
                _anchor_kicker = _kicker_cx if _kicker_cx is not None else _ball_x0
                _strategies.append((_anchor_kicker, 0.8, 1.5, "kicker"))

                # Strategy B: ball anchor, standard timing
                _strategies.append((_ball_x0, 0.8, 1.5, "ball"))

                # Strategy C: midpoint anchor, standard timing
                if _kicker_cx is not None:
                    _mid = (_kicker_cx + _ball_x0) / 2.0
                    _strategies.append((_mid, 0.8, 1.5, "midpoint"))

                # Strategy D: kicker anchor, short hold
                _strategies.append((_anchor_kicker, 0.4, 1.2, "kicker_short"))

                # Evaluate each strategy (lightweight — no rendering)
                _best = None
                _best_score = -1.0
                _results_log: list[str] = []
                _saved_states_cx = [s.cx for s in states]
                _saved_states_x0 = [s.x0 for s in states]

                for _anc, _hs, _ts, _lbl in _strategies:
                    # Restore planner states before each strategy attempt
                    for _ri in range(len(states)):
                        states[_ri].cx = _saved_states_cx[_ri]
                        states[_ri].x0 = _saved_states_x0[_ri]

                    _he, _te, _out, _tot, _mesc, _lbl2 = _apply_fk_strategy(
                        _anc, _hs, _ts, _lbl,
                    )
                    _in_pct = 100.0 * (1.0 - _out / max(1, _tot))
                    # Score: maximise ball-in-crop %, penalise large escapes
                    _score = _in_pct - min(50.0, _mesc / 5.0)
                    _results_log.append(
                        f"  {_lbl2}: anchor={_anc:.0f}, hold={_hs:.1f}s, "
                        f"trans={_ts:.1f}s, ball_in={_in_pct:.1f}%, "
                        f"max_esc={_mesc:.0f}px, score={_score:.1f}"
                    )
                    if _score > _best_score:
                        _best_score = _score
                        _best = (_anc, _hs, _ts, _lbl2, _he, _te)

                # Strategy E: "follow" — skip the hold/transition override
                # entirely and let the planner's natural ball tracking
                # drive the camera.  This is best when the ball moves
                # fast or the kicker/ball are at the source frame edge.
                #
                # IMPORTANT: Only evaluate ball-in-crop on reliable
                # frames (YOLO, blended, interpolated, hold — NOT
                # centroid-only).  Centroid frames are self-validating:
                # the planner tracks them, so they're trivially in-crop.
                # Including them inflates the follow score to ~100%
                # even when the actual ball (per YOLO) is off-screen.
                _fk_reliable = {1, 3, 4, 5}  # yolo, blended, interp, hold
                for _ri in range(len(states)):
                    states[_ri].cx = _saved_states_cx[_ri]
                    states[_ri].x0 = _saved_states_x0[_ri]
                _follow_out = 0
                _follow_tot = 0
                _follow_max_esc = 0.0
                _follow_eval_end = min(len(states), len(positions))
                for _ei in range(_follow_eval_end):
                    if np.isnan(positions[_ei]).any():
                        continue
                    # Skip centroid-only frames — they self-validate
                    if fusion_source_labels is not None and _ei < len(fusion_source_labels):
                        if int(fusion_source_labels[_ei]) not in _fk_reliable:
                            continue
                    elif fusion_source_labels is not None:
                        continue
                    _ebx = float(positions[_ei][0])
                    _ecx = float(states[_ei].cx)
                    _ecw = float(states[_ei].crop_w) if states[_ei].crop_w > 0 else _pcw
                    _el = float(np.clip(
                        _ecx - _ecw / 2.0, 0.0,
                        max(0.0, _snap_fw - _ecw),
                    ))
                    _er = _el + _ecw
                    _follow_tot += 1
                    if _ebx < _el or _ebx > _er:
                        _follow_out += 1
                        _follow_max_esc = max(
                            _follow_max_esc,
                            max(_el - _ebx, _ebx - _er),
                        )
                _follow_in_pct = 100.0 * (1.0 - _follow_out / max(1, _follow_tot))
                _follow_score = _follow_in_pct - min(50.0, _follow_max_esc / 5.0)
                _results_log.append(
                    f"  follow: anchor=planner, hold=0.0s, "
                    f"trans=0.0s, ball_in={_follow_in_pct:.1f}%, "
                    f"max_esc={_follow_max_esc:.0f}px, score={_follow_score:.1f}"
                )
                if _follow_score > _best_score:
                    _best_score = _follow_score
                    _best = (None, 0.0, 0.0, "follow", _kick_frame, _kick_frame)

                assert _best is not None
                _anchor_x, _hold_s, _trans_s, _fk_label, _, _ = _best

                # Restore planner states before applying the winner
                for _ri in range(len(states)):
                    states[_ri].cx = _saved_states_cx[_ri]
                    states[_ri].x0 = _saved_states_x0[_ri]

                if _anchor_x is None:
                    # "follow" strategy won — but we still enforce a
                    # minimum hold so the viewer sees the kicker at
                    # the moment of the kick.  Without this, the
                    # planner may track centroid noise and pan away
                    # from the kicker/ball before the kick happens.
                    _min_hold_s = 0.3
                    _min_trans_s = 2.5  # long transition to track ball flight
                    _follow_anchor = _kicker_cx if _kicker_cx is not None else _ball_x0
                    _fk_label = "follow+hold"
                    _hold_end_final, _trans_end_final, _out_f, _tot_f, _mesc_f, _fk_label = (
                        _apply_fk_strategy(_follow_anchor, _min_hold_s, _min_trans_s, _fk_label)
                    )
                    _hold_end = _hold_end_final
                    _trans_end = _trans_end_final
                    _hold_frames = _hold_end - _kick_frame
                    _trans_frames = _trans_end - _hold_end
                    _kick_hold_end = _hold_end
                    _kick_hold_trans_end = _trans_end
                    _in_pct_f = 100.0 * (1.0 - _out_f / max(1, _tot_f))
                    _anchor_x = _follow_anchor
                else:
                    _hold_end_final, _trans_end_final, _out_f, _tot_f, _mesc_f, _fk_label = (
                        _apply_fk_strategy(_anchor_x, _hold_s, _trans_s, _fk_label)
                    )
                    _hold_end = _hold_end_final
                    _trans_end = _trans_end_final
                    _hold_frames = _hold_end - _kick_frame
                    _trans_frames = _trans_end - _hold_end
                    _kick_hold_end = _hold_end
                    _kick_hold_trans_end = _trans_end
                    _in_pct_f = 100.0 * (1.0 - _out_f / max(1, _tot_f))

                # --- diagnostics ---
                print(
                    f"[CAMERA] FREE_KICK anchor: ball_x={_ball_x0:.0f}, "
                    f"kicker_cx={f'{_kicker_cx:.0f}' if _kicker_cx is not None else 'N/A'}, "
                    f"anchor_x={_anchor_x:.0f}"
                )
                print(
                    f"[CAMERA] FREE_KICK strategy: {_fk_label} (score={_best_score:.1f}), "
                    f"plan_ball_in={_in_pct_f:.1f}%, max_esc={_mesc_f:.0f}px"
                )
                for _rl in _results_log:
                    print(f"[CAMERA] FREE_KICK candidate:{_rl}")
                print(
                    f"[CAMERA] FREE_KICK hold-then-follow: kick_frame={_kick_frame}, "
                    f"hold_end={_hold_end}, trans_end={_trans_end} "
                    f"(hold={_hold_frames / _snap_fps:.1f}s, "
                    f"transition={_trans_frames / _snap_fps:.1f}s)"
                )
            else:
                # Non-FREE_KICK: short startup snap to ball position
                _snap_n = 20
                _snap_n = min(_snap_n, len(states) // 3)
                _snap_n = max(_snap_n, 1)
                for _si in range(_snap_n):
                    _blend = (_si / _snap_n) ** 2
                    if _si < len(positions) and not np.isnan(positions[_si]).any():
                        _ball_x_i = float(positions[_si][0])
                    elif _snap_ball_x0 is not None:
                        _ball_x_i = _snap_ball_x0
                    else:
                        continue
                    _new_cx = (1.0 - _blend) * _ball_x_i + _blend * states[_si].cx
                    states[_si].cx = _new_cx
                    _half_cw = states[_si].crop_w / 2.0
                    states[_si].x0 = float(np.clip(
                        _new_cx - _half_cw, 0.0,
                        max(0.0, _snap_fw - states[_si].crop_w),
                    ))
                print(
                    f"[CAMERA] startup snap: gap={_snap_gap:.0f}px, ramp={_snap_n}f"
                )

    # --- BALL-GRAVITY CLAMP: keep camera near known ball positions ---
    # The planner's centroid-tracking can drift the camera far from
    # the ball during gaps in YOLO coverage (69% centroid-only on
    # this clip).  Apply a soft clamp: if the camera is more than
    # ``max_drift_frac`` of crop width from the ball, pull it back.
    # Corrections are Gaussian-smoothed to prevent jitter.
    if states and len(positions) > 0 and len(states) > 5:
        _grav_max_frac = 0.32  # ball must be within 32% of crop half-width from center
        _grav_corrections = np.zeros(len(states), dtype=np.float64)
        # Only trust YOLO-confirmed or interpolated/held ball positions,
        # NOT centroid-only (label 2) which tracks player mass, not ball.
        _grav_reliable = {1, 3, 4, 5}  # YOLO, interpolated, hold-fwd, hold-bwd
        for _gi in range(len(states)):
            # Skip frames protected by kick-hold — we placed them deliberately
            if _gi < _kick_hold_trans_end:
                continue
            # Skip centroid-only frames — ball position is unreliable
            if fusion_source_labels is not None and _gi < len(fusion_source_labels):
                if int(fusion_source_labels[_gi]) not in _grav_reliable:
                    continue
            elif fusion_source_labels is not None:
                continue  # out of range → no label → skip
            if _gi < len(positions) and not np.isnan(positions[_gi]).any():
                _grav_bx = float(positions[_gi][0])
                _grav_cx = float(states[_gi].cx)
                _grav_cw = float(states[_gi].crop_w) if states[_gi].crop_w > 1 else 607.0
                _grav_max_drift = _grav_cw * _grav_max_frac
                _grav_drift = _grav_cx - _grav_bx
                if abs(_grav_drift) > _grav_max_drift:
                    _grav_corrections[_gi] = _grav_drift - np.sign(_grav_drift) * _grav_max_drift

        # Smooth corrections with a Gaussian to prevent jitter
        _grav_sigma = max(2.0, _snap_fps * 0.15)  # ~4-5 frames
        _grav_r = int(_grav_sigma * 3.0 + 0.5)
        _grav_r = min(_grav_r, len(states) // 2)
        if _grav_r >= 1 and np.any(np.abs(_grav_corrections) > 0.5):
            _grav_kx = np.arange(-_grav_r, _grav_r + 1, dtype=np.float64)
            _grav_k = np.exp(-0.5 * (_grav_kx / _grav_sigma) ** 2)
            _grav_k /= _grav_k.sum()
            _grav_padded = np.pad(_grav_corrections, _grav_r, mode="edge")
            _grav_smooth = np.convolve(_grav_padded, _grav_k, mode="valid")

            _grav_applied = 0
            for _gi in range(len(states)):
                if abs(_grav_smooth[_gi]) > 0.5:
                    states[_gi].cx -= _grav_smooth[_gi]
                    _half_cw_g = states[_gi].crop_w / 2.0
                    _fw_g = float(width) if width > 0 else 1920.0
                    states[_gi].x0 = float(np.clip(
                        states[_gi].cx - _half_cw_g, 0.0,
                        max(0.0, _fw_g - states[_gi].crop_w),
                    ))
                    _grav_applied += 1
            if _grav_applied > 0:
                print(
                    f"[CAMERA] Ball-gravity clamp: corrected {_grav_applied}/{len(states)} frames "
                    f"(max_drift={_grav_max_frac:.0%} of crop, "
                    f"peak_correction={float(np.max(np.abs(_grav_smooth))):.1f}px)"
                )

    # --- CAMERA SPEED LIMITER: prevent choppy single-frame jumps ---
    # After all corrections (snap, gravity), cap the maximum per-frame
    # camera movement.  This is a forward pass that propagates limits.
    # Derive from the preset speed_limit (px/s) so the post-processing
    # limiter is consistent with the planner's configured speed.
    # Apply the same event-type scaling the CameraPlanner uses internally
    # so that FREE_KICK's 0.45x slowdown and GOAL/CROSS boosts are
    # respected by the post-processing pass.
    if states and len(states) > 2:
        _sl_fps = float(fps_in if fps_in and fps_in > 0 else fps_out or 30.0)
        _sl_event_scale = 1.0
        _sl_event = (getattr(args, "event_type", None) or "").upper().strip()
        if _sl_event == "FREE_KICK":
            _sl_event_scale = 0.45
        elif _sl_event == "GOAL":
            _sl_event_scale = 1.25
        elif _sl_event == "CROSS":
            _sl_event_scale = 1.40
        _max_speed = max(15.0, speed_limit * _sl_event_scale / _sl_fps)  # px/frame from preset speed_limit (px/s)
        _speed_clipped = 0
        _speed_fw = float(width) if width > 0 else 1920.0
        _sl_start = max(1, _kick_hold_end)  # don't speed-limit the kick-hold
        _trans_max_speed = max(40.0, _max_speed * 1.5)  # fast during FREE_KICK transition to track ball flight
        # Post-transition catch-up: the planner output is slow (conf-damped)
        # so give the camera 3s at higher speed to reach the ball before
        # falling back to the normal limit.
        _catchup_end = _kick_hold_trans_end + int(_sl_fps * 3.0) if _kick_hold_trans_end > 0 else 0
        _catchup_speed = max(25.0, _max_speed * 1.2)
        for _si in range(_sl_start, len(states)):
            if _kick_hold_end > 0 and _si <= _kick_hold_trans_end:
                _eff_max = _trans_max_speed
            elif _kick_hold_end > 0 and _si <= _catchup_end:
                # Linear taper from catch-up speed to normal speed
                _cu_t = (_si - _kick_hold_trans_end) / max(1, _catchup_end - _kick_hold_trans_end)
                _eff_max = _catchup_speed + (_max_speed - _catchup_speed) * _cu_t
            else:
                _eff_max = _max_speed
            _delta = states[_si].cx - states[_si - 1].cx
            if abs(_delta) > _eff_max:
                _clamped_cx = states[_si - 1].cx + np.sign(_delta) * _eff_max
                states[_si].cx = _clamped_cx
                _half_cw_s = states[_si].crop_w / 2.0
                states[_si].x0 = float(np.clip(
                    _clamped_cx - _half_cw_s, 0.0,
                    max(0.0, _speed_fw - states[_si].crop_w),
                ))
                _speed_clipped += 1
        if _speed_clipped > 0:
            print(
                f"[CAMERA] Speed limiter: clamped {_speed_clipped}/{len(states)} frames "
                f"(max {_max_speed:.0f}px/f = {_max_speed * _sl_fps:.0f}px/s)"
            )

    # --- Camera plan diagnostics (printed to stdout for batch visibility) ---
    if states and len(states) > 1:
        _cx_arr = np.array([s.cx for s in states], dtype=np.float64)
        _cx_deltas = np.abs(np.diff(_cx_arr))
        _cx_max_delta = float(_cx_deltas.max())
        _cx_mean_delta = float(_cx_deltas.mean())
        _cx_p95_delta = float(np.percentile(_cx_deltas, 95))
        # Count direction reversals (sign changes in velocity)
        _cx_vel = np.diff(_cx_arr)
        _reversals = int(np.sum(np.diff(np.sign(_cx_vel)) != 0))
        _cx_range = float(_cx_arr.max() - _cx_arr.min())
        _dz_hold = getattr(planner, '_deadzone_hold_frames', 0)
        _dz_pct = 100.0 * _dz_hold / max(len(states), 1)
        # Count frames where camera is effectively still (delta < 0.3px)
        _still_count = int(np.sum(_cx_deltas < 0.3))
        _still_pct = 100.0 * _still_count / max(len(_cx_deltas), 1)
        # Zoom diagnostics
        _zm_arr = np.array([s.zoom for s in states], dtype=np.float64)
        _zm_deltas = np.abs(np.diff(_zm_arr))
        _zm_max_delta = float(_zm_deltas.max())
        _zm_mean_delta = float(_zm_deltas.mean())
        _zm_range = float(_zm_arr.max() - _zm_arr.min())
        _zm_reversals = int(np.sum(np.diff(np.sign(np.diff(_zm_arr))) != 0))
        print(
            f"[CAMERA] cx: range={_cx_range:.0f}px, "
            f"max_delta={_cx_max_delta:.1f}px/f, "
            f"mean_delta={_cx_mean_delta:.1f}px/f, "
            f"p95_delta={_cx_p95_delta:.1f}px/f, "
            f"reversals={_reversals}, "
            f"deadzone_hold={_dz_pct:.0f}%, still={_still_pct:.0f}%"
        )
        _person_ctx_frames = sum(
            1 for s in states
            if any(f.startswith("person_ctx=") for f in (s.clamp_flags or []))
        )
        _flight_commit_frames = sum(
            1 for s in states
            if any(f.startswith("flight_commit=") for f in (s.clamp_flags or []))
        )
        _goal_snap_frames = sum(
            1 for s in states
            if any(f.startswith("goal_snap=") for f in (s.clamp_flags or []))
        )
        _goal_events = sum(
            1 for s in states
            if any(f == "goal_event" for f in (s.clamp_flags or []))
        )
        _extras = ""
        if _person_ctx_frames > 0:
            _extras += f", person_ctx={_person_ctx_frames}f"
        if _flight_commit_frames > 0:
            _extras += f", flight={_flight_commit_frames}f"
        if _goal_events > 0:
            _extras += f", goals={_goal_events}, snap={_goal_snap_frames}f"
        print(
            f"[CAMERA] zoom: range={_zm_range:.2f}, "
            f"max_delta={_zm_max_delta:.4f}/f, "
            f"mean_delta={_zm_mean_delta:.4f}/f, "
            f"reversals={_zm_reversals}{_extras}"
        )

    # --- POST-GOAL TAIL TRIM ---
    # For clips tagged as GOAL (BUILD_UP_AND_GOAL, etc.), detect the goal
    # event in the camera plan and trim the output to avoid following the
    # celebration all the way back to the kickoff restart.
    _POST_GOAL_TAIL_S = 3.0  # seconds of footage to keep after goal event
    _clip_stem_upper = original_source_path.stem.upper()
    if states and "GOAL" in _clip_stem_upper:
        _last_goal_frame = -1
        for _gi, _gs in enumerate(states):
            if _gs.clamp_flags and "goal_event" in _gs.clamp_flags:
                _last_goal_frame = _gi
        if _last_goal_frame >= 0:
            _render_fps_trim = float(fps_in if fps_in > 0 else fps_out or 30.0)
            _trim_end = _last_goal_frame + int(_POST_GOAL_TAIL_S * _render_fps_trim)
            if _trim_end < len(states) - 1:
                _trimmed_count = len(states) - _trim_end
                states = states[:_trim_end]
                # Also trim fusion arrays to match
                if positions is not None and len(positions) > _trim_end:
                    positions = positions[:_trim_end]
                if used_mask is not None and len(used_mask) > _trim_end:
                    used_mask = used_mask[:_trim_end]
                if fusion_confidence is not None and len(fusion_confidence) > _trim_end:
                    fusion_confidence = fusion_confidence[:_trim_end]
                if fusion_source_labels is not None and len(fusion_source_labels) > _trim_end:
                    fusion_source_labels = fusion_source_labels[:_trim_end]
                frame_count = len(states)
                print(
                    f"[TRIM] Goal clip: trimmed {_trimmed_count} frames after goal "
                    f"(goal@f{_last_goal_frame}, kept {_POST_GOAL_TAIL_S:.1f}s tail, "
                    f"new length={len(states)}f/{len(states)/_render_fps_trim:.1f}s)"
                )

    # --- HARD MAX-DURATION TRIM ---
    # Per-clip override: discard frames beyond --max-duration seconds.
    _max_dur_s = getattr(args, "max_duration_s", None)
    if _max_dur_s is not None and states:
        _render_fps_md = float(fps_in if fps_in > 0 else fps_out or 30.0)
        _max_frames = int(round(_max_dur_s * _render_fps_md))
        if _max_frames < len(states):
            _trimmed_md = len(states) - _max_frames
            states = states[:_max_frames]
            if positions is not None and len(positions) > _max_frames:
                positions = positions[:_max_frames]
            if used_mask is not None and len(used_mask) > _max_frames:
                used_mask = used_mask[:_max_frames]
            if fusion_confidence is not None and len(fusion_confidence) > _max_frames:
                fusion_confidence = fusion_confidence[:_max_frames]
            if fusion_source_labels is not None and len(fusion_source_labels) > _max_frames:
                fusion_source_labels = fusion_source_labels[:_max_frames]
            frame_count = len(states)
            print(
                f"[TRIM] Max-duration override: trimmed {_trimmed_md} frames "
                f"(cap={_max_dur_s:.1f}s, new length={len(states)}f/"
                f"{len(states)/_render_fps_md:.1f}s)"
            )

    # --- Ball-in-crop diagnostic (always printed to stdout for batch visibility) ---
    _FUSE_LABELS = {0: "none", 1: "yolo", 2: "centroid", 3: "blended", 4: "interp", 5: "hold"}
    if states and len(states) > 0 and follow_crop_width > 0:
        _n_states = len(states)
        _half_pw = float(follow_crop_width) / 2.0
        _crop_h_est = float(height)  # at zoom=1.0; refined per-frame below
        _inside_total = 0
        _outside_total = 0
        _outside_by_src = {k: 0 for k in _FUSE_LABELS}
        _inside_by_src = {k: 0 for k in _FUSE_LABELS}
        _max_outside_dist = 0.0
        _worst_outside_frames: list[tuple[int, float, int]] = []  # (frame, dist, src)
        _speed_limited_frames = 0
        _keepinview_frames = 0

        for _fi in range(_n_states):
            _st = states[_fi]

            # Count engine flags
            _flags = _st.clamp_flags or []
            if any(f == "speed" or f == "final_speed" for f in _flags):
                _speed_limited_frames += 1
            if any(f.startswith("keepin_prop=") for f in _flags):
                _keepinview_frames += 1

            # Get ball position for this frame
            _bx: Optional[float] = None
            _by: Optional[float] = None
            if _st.ball is not None:
                _bx, _by = float(_st.ball[0]), float(_st.ball[1])
            elif _fi < len(positions) and used_mask[_fi]:
                _bx = float(positions[_fi, 0])
                _by = float(positions[_fi, 1])
            if _bx is None:
                continue

            # Source label
            _src = int(fusion_source_labels[_fi]) if (fusion_source_labels is not None and _fi < len(fusion_source_labels)) else 0

            # Crop rectangle from state — use per-frame x0/crop_w (clamped
            # bounds that match rendering) rather than follow_crop_width.
            _st_cw = _st.crop_w if hasattr(_st, "crop_w") and _st.crop_w > 0 else follow_crop_width
            _st_ch = _st.crop_h if hasattr(_st, "crop_h") and _st.crop_h > 0 else _crop_h_est
            _crop_left = _st.x0 if hasattr(_st, "x0") else max(0.0, _st.cx - _st_cw / 2.0)
            _crop_right = _crop_left + _st_cw
            _crop_top = _st.y0 if hasattr(_st, "y0") else max(0.0, _st.cy - _st_ch / 2.0)
            _crop_bottom = _crop_top + _st_ch

            # Distance to nearest crop edge (negative = outside)
            _dx_left = _bx - _crop_left
            _dx_right = _crop_right - _bx
            _dy_top = _by - _crop_top
            _dy_bottom = _crop_bottom - _by
            _min_dist = min(_dx_left, _dx_right, _dy_top, _dy_bottom)

            if _min_dist >= 0:
                _inside_total += 1
                _inside_by_src[_src] = _inside_by_src.get(_src, 0) + 1
            else:
                _outside_total += 1
                _outside_by_src[_src] = _outside_by_src.get(_src, 0) + 1
                _out_dist = abs(_min_dist)
                if _out_dist > _max_outside_dist:
                    _max_outside_dist = _out_dist
                _worst_outside_frames.append((_fi, _out_dist, _src))

        _checked = _inside_total + _outside_total
        _in_pct = 100.0 * _inside_total / max(1, _checked)
        print(
            f"[DIAG] Ball in crop: {_inside_total}/{_checked} ({_in_pct:.1f}%) | "
            f"Outside: {_outside_total} frames | Max escape: {_max_outside_dist:.0f}px"
        )
        # Source breakdown
        _src_parts = []
        _yolo_confirmed_total = 0
        _centroid_only_total = 0
        for _sk in sorted(_FUSE_LABELS.keys()):
            _s_in = _inside_by_src.get(_sk, 0)
            _s_out = _outside_by_src.get(_sk, 0)
            _s_tot = _s_in + _s_out
            if _s_tot > 0:
                _s_pct_out = 100.0 * _s_out / max(1, _s_tot)
                _src_parts.append(f"{_FUSE_LABELS[_sk]}={_s_tot}({_s_out} out/{_s_pct_out:.0f}%)")
            if _sk in (1, 3):  # yolo, blended
                _yolo_confirmed_total += _s_tot
            elif _sk == 2:  # centroid
                _centroid_only_total += _s_tot
        if _src_parts:
            print(f"[DIAG] Source breakdown: {', '.join(_src_parts)}")
        # Warn when centroid dominates — the "ball in crop" metric is self-
        # referential for centroid frames (camera follows the centroid
        # estimate, so the estimate is always "in crop").  The actual ball
        # may be elsewhere.
        if _checked > 0 and _centroid_only_total > _checked * 0.50:
            _yolo_pct = 100.0 * _yolo_confirmed_total / max(1, _checked)
            print(
                f"[DIAG] WARNING: {_centroid_only_total}/{_checked} frames "
                f"({100.0 * _centroid_only_total / _checked:.0f}%) use centroid-only "
                f"tracking (YOLO-confirmed: {_yolo_pct:.0f}%). "
                f"Ball-in-crop metric is unreliable for centroid frames — "
                f"actual ball may be outside the crop."
            )
        # YOLO-only ball-in-crop: the ground-truth metric.  Only counts
        # frames where the ball position comes from actual YOLO detection
        # (source 1=yolo or 3=blended).  Interpolated/centroid/hold frames
        # are excluded because the camera follows those positions by
        # construction — they are always "in crop" but may not reflect
        # where the real ball is.
        if _yolo_confirmed_total > 0:
            _yolo_in = _inside_by_src.get(1, 0) + _inside_by_src.get(3, 0)
            _yolo_out = _outside_by_src.get(1, 0) + _outside_by_src.get(3, 0)
            _yolo_crop_pct = 100.0 * _yolo_in / max(1, _yolo_confirmed_total)
            _interp_total = sum(
                _inside_by_src.get(k, 0) + _outside_by_src.get(k, 0)
                for k in (4, 5)  # interp + hold
            )
            print(
                f"[DIAG] YOLO-only ball in crop: {_yolo_in}/{_yolo_confirmed_total} "
                f"({_yolo_crop_pct:.1f}%) | "
                f"Interpolated/held frames (not checked): {_interp_total}"
            )
        print(
            f"[DIAG] Speed-limited: {_speed_limited_frames}f ({100.0 * _speed_limited_frames / max(1, _n_states):.0f}%) | "
            f"Keepinview: {_keepinview_frames}f ({100.0 * _keepinview_frames / max(1, _n_states):.0f}%)"
        )

        # Confidence-weighted ball-in-crop: a single number that reflects
        # how trustworthy the "ball in crop" metric is by weighting each
        # frame by source reliability:
        #   YOLO (source 1, 3): weight 1.0
        #   Centroid within 100px of recent YOLO: weight 0.5
        #   All other centroid/interp/hold: weight 0.0
        _cw_weighted_in = 0.0
        _cw_total_weight = 0.0
        for _fi in range(_n_states):
            if _fi >= len(positions) or not used_mask[_fi]:
                continue
            _src = int(fusion_source_labels[_fi]) if (fusion_source_labels is not None and _fi < len(fusion_source_labels)) else 0
            if _src in (1, 3):  # yolo, blended
                _w = 1.0
            elif _src == 2:  # centroid
                # Weight centroid frames by proximity to nearest YOLO frame
                _nearest_yolo_dist = float('inf')
                if fusion_source_labels is not None:
                    for _yd in range(max(0, _fi - 30), min(len(fusion_source_labels), _fi + 31)):
                        if int(fusion_source_labels[_yd]) in (1, 3):
                            _d = abs(_yd - _fi)
                            if _d < _nearest_yolo_dist:
                                _nearest_yolo_dist = _d
                _w = 0.5 if _nearest_yolo_dist <= 15 else 0.0
            else:
                _w = 0.0
            if _w > 0:
                _cw_total_weight += _w
                _st_cw2 = states[_fi].crop_w if hasattr(states[_fi], "crop_w") and states[_fi].crop_w > 0 else follow_crop_width
                _st_ch2 = states[_fi].crop_h if hasattr(states[_fi], "crop_h") and states[_fi].crop_h > 0 else _crop_h_est
                _cl2 = states[_fi].x0 if hasattr(states[_fi], "x0") else max(0.0, states[_fi].cx - _st_cw2 / 2.0)
                _cr2 = _cl2 + _st_cw2
                _ct2 = states[_fi].y0 if hasattr(states[_fi], "y0") else max(0.0, states[_fi].cy - _st_ch2 / 2.0)
                _cb2 = _ct2 + _st_ch2
                _bx2 = float(positions[_fi, 0])
                _by2 = float(positions[_fi, 1])
                _in2 = (_bx2 >= _cl2 and _bx2 <= _cr2 and _by2 >= _ct2 and _by2 <= _cb2)
                if _in2:
                    _cw_weighted_in += _w
        _cw_pct = 100.0 * _cw_weighted_in / max(1e-6, _cw_total_weight) if _cw_total_weight > 0 else 0.0
        print(
            f"[DIAG] Confidence-weighted ball-in-crop: {_cw_pct:.1f}% "
            f"(weighted frames: {_cw_total_weight:.0f}/{_checked})"
        )
        if _cw_total_weight < _checked * 0.15:
            print(
                f"[DIAG] WARNING: Only {_cw_total_weight:.0f} frames have "
                f"trustworthy ball position data — framing quality cannot "
                f"be reliably assessed for this clip."
            )

        # Camera cx timeline at 1-second intervals
        if _n_states >= 2:
            _render_fps = fps if fps > 0 else 24
            _cx_step = max(1, int(_render_fps))
            _cx_samples = list(range(0, _n_states, _cx_step))
            if _cx_samples and _cx_samples[-1] != _n_states - 1:
                _cx_samples.append(_n_states - 1)
            _cx_parts = [f"f{_cf}={states[_cf].cx:.0f}" for _cf in _cx_samples]
            print(f"[DIAG] Camera cx timeline (1s): {', '.join(_cx_parts)}")

        # Show worst escape frames
        if _worst_outside_frames:
            _worst_outside_frames.sort(key=lambda x: x[1], reverse=True)
            _top_worst = _worst_outside_frames[:5]
            _worst_str = ", ".join(
                f"f{wf[0]}:{wf[1]:.0f}px({_FUSE_LABELS.get(wf[2], '?')})"
                for wf in _top_worst
            )
            print(f"[DIAG] Worst escapes: {_worst_str}")

        # --- Per-frame diagnostics CSV (when --diagnostics is set) ---
        if getattr(args, "diagnostics", False):
            _diag_path = output_path.with_suffix(".diag.csv")
            try:
                import csv as _csv_mod
                with open(_diag_path, "w", newline="") as _df:
                    _dw = _csv_mod.writer(_df)
                    _dw.writerow([
                        "frame", "ball_x", "ball_y", "source", "confidence",
                        "cam_cx", "cam_cy", "crop_x0", "crop_y0", "crop_w", "crop_h",
                        "ball_in_crop", "dist_to_edge", "speed_limited", "keepinview",
                        "clamp_flags",
                    ])
                    for _fi in range(_n_states):
                        _st = states[_fi]
                        _bx_d = ""
                        _by_d = ""
                        if _st.ball is not None:
                            _bx_d, _by_d = f"{_st.ball[0]:.1f}", f"{_st.ball[1]:.1f}"
                        elif _fi < len(positions) and used_mask[_fi]:
                            _bx_d = f"{positions[_fi, 0]:.1f}"
                            _by_d = f"{positions[_fi, 1]:.1f}"
                        _src_d = _FUSE_LABELS.get(
                            int(fusion_source_labels[_fi]) if (fusion_source_labels is not None and _fi < len(fusion_source_labels)) else 0,
                            "none"
                        )
                        _conf_d = f"{fusion_confidence[_fi]:.2f}" if (fusion_confidence is not None and _fi < len(fusion_confidence)) else ""
                        _flags_d = _st.clamp_flags or []
                        _sl_d = "1" if any(f == "speed" or f == "final_speed" for f in _flags_d) else "0"
                        _kv_d = "1" if any(f.startswith("keepin_prop=") for f in _flags_d) else "0"
                        # Ball-in-crop distance
                        _dist_d = ""
                        _bic_d = ""
                        if _bx_d and _by_d:
                            _bxf = float(_bx_d)
                            _byf = float(_by_d)
                            _cl = max(0.0, _st.cx - _half_pw)
                            if _cl + follow_crop_width > width:
                                _cl = max(0.0, width - follow_crop_width)
                            _cr = _cl + follow_crop_width
                            _ct = _st.y0 if hasattr(_st, "y0") else max(0.0, _st.cy - _crop_h_est / 2.0)
                            _cb = _ct + (_st.crop_h if hasattr(_st, "crop_h") else _crop_h_est)
                            _md = min(_bxf - _cl, _cr - _bxf, _byf - _ct, _cb - _byf)
                            _dist_d = f"{_md:.1f}"
                            _bic_d = "1" if _md >= 0 else "0"
                        _dw.writerow([
                            _fi, _bx_d, _by_d, _src_d, _conf_d,
                            f"{_st.cx:.1f}", f"{_st.cy:.1f}",
                            f"{_st.x0:.1f}", f"{_st.y0:.1f}",
                            f"{_st.crop_w:.1f}", f"{_st.crop_h:.1f}",
                            _bic_d, _dist_d, _sl_d, _kv_d,
                            "|".join(_flags_d),
                        ])
                print(f"[DIAG] Per-frame CSV: {_diag_path}")
            except Exception as _diag_exc:
                print(f"[DIAG] CSV write failed: {_diag_exc}")

    temp_root = scratch_root / "autoframe_work"
    temp_dir = temp_root / preset_key / original_source_path.stem
    _prepare_temp_dir(temp_dir, args.clean_temp)
    frames_dir = temp_dir / "frames"
    if frames_dir.is_dir() and not getattr(args, "resume", False):
        shutil.rmtree(frames_dir, ignore_errors=True)
    scratch_cleanup_paths.append(temp_dir)

    brand_overlay_path = Path(args.brand_overlay).expanduser() if args.brand_overlay else None
    endcard_path = Path(args.endcard).expanduser() if args.endcard else None
    offline_ball_path: Optional[List[Optional[dict[str, float]]]] = None
    if getattr(args, "ball_path", None):
        ball_path_file = Path(args.ball_path).expanduser()
        if not ball_path_file.exists():
            raise FileNotFoundError(f"Ball path file not found: {ball_path_file}")
        offline_ball_path = load_ball_path(
            ball_path_file,
            ball_key_x=str(getattr(args, "ball_key_x", "bx_stab")),
            ball_key_y=str(getattr(args, "ball_key_y", "by_stab")),
        )
    elif use_ball_telemetry and ball_samples:
        max_frame_idx = max(
            (
                int(frame_idx)
                for frame_idx in (getattr(s, "frame", None) for s in ball_samples)
                if frame_idx is not None and isinstance(frame_idx, (int, float))
            ),
            default=0,
        )
        total_frames = max(len(states), frame_count, max_frame_idx + 1)
        path: list[Optional[dict[str, float]]] = [None] * total_frames
        for idx, sample in enumerate(ball_samples):
            bx_val = _safe_float(getattr(sample, "x", None))
            by_val = _safe_float(getattr(sample, "y", None))
            if bx_val is None or by_val is None:
                continue
            if idx >= len(path):
                path.extend([None] * (idx - len(path) + 1))
            path[idx] = {"bx": float(bx_val), "by": float(by_val)}
        offline_ball_path = path

        if portrait:
            portrait_w, portrait_h = portrait
            plan_config = PlannerConfig(
                frame_size=(float(width), float(height)),
                crop_aspect=float(portrait_w) / float(portrait_h),
                fps=float(fps_in) if fps_in > 0 else float(fps_out) if fps_out else 30.0,
                keep_in_frame_frac_x=(0.4, 0.6),
                keep_in_frame_frac_y=(0.4, 0.6),
                min_zoom=float(zoom_min),
                max_zoom=float(zoom_max),
            )
            planned = plan_ball_portrait_crop(
                ball_samples,
                src_w=width,
                src_h=height,
                portrait_w=int(portrait_w),
                portrait_h=int(portrait_h),
                config=plan_config,
            )
            if planned and not keep_path_lookup_data:
                keep_path_lookup_data = {frame: (cx, cy) for frame, (cx, cy, _z) in planned.items()}
    if not use_ball_telemetry:
        keep_path_lookup_data = {}
        args.debug_ball_overlay = False

    debug_ball_overlay = bool(getattr(args, "debug_ball_overlay", False) and use_ball_telemetry)

    telemetry_file: Optional[TextIO] = None
    telemetry_simple_file: Optional[TextIO] = None
    if render_telemetry_path:
        telemetry_file = render_telemetry_path.open("w", encoding="utf-8")
    if render_telemetry_simple_path:
        telemetry_simple_file = render_telemetry_simple_path.open("w", encoding="utf-8")

    jerk95 = 0.0
    renderer = Renderer(
        input_path=input_path,
        output_path=output_path,
        temp_dir=temp_dir,
        fps_in=fps_in,
        fps_out=fps_out,
        flip180=args.flip180,
        portrait=portrait,
        brand_overlay=brand_overlay_path,
        endcard=endcard_path,
        pad=pad,
        zoom_min=zoom_min,
        zoom_max=zoom_max,
        speed_limit=speed_limit,
        telemetry=telemetry_file,
        telemetry_simple=telemetry_simple_file,
        init_manual=getattr(args, "init_manual", False),
        init_t=getattr(args, "init_t", 0.8),
        ball_path=offline_ball_path,
        follow_lead_time=lead_time_s,
        follow_margin_px=keepinview_margin,
        follow_smoothing=smoothing,
        follow_zeta=follow_zeta,
        follow_wn=follow_wn,
        follow_deadzone=follow_deadzone,
        follow_max_vel=follow_max_vel,
        follow_max_acc=follow_max_acc,
        follow_lookahead=follow_lookahead_frames,
        follow_pre_smooth=follow_pre_smooth,
        follow_zoom_out_max=follow_zoom_out_max,
        follow_zoom_edge_frac=follow_zoom_edge_frac,
        follow_center_frac=cy_frac,
        lost_hold_ms=lost_hold_ms,
        lost_pan_ms=lost_pan_ms,
        lost_lookahead_s=lost_lookahead_s,
        lost_chase_motion_ms=lost_chase_motion_ms,
        lost_motion_thresh=lost_motion_thresh,
        lost_use_motion=lost_use_motion,
        portrait_plan_margin_px=getattr(args, "portrait_plan_margin", None),
        portrait_plan_headroom=getattr(args, "portrait_plan_headroom", None),
        portrait_plan_lead_px=getattr(args, "portrait_plan_lead", None),
        adaptive_tracking=adaptive_tracking_enabled,
        adaptive_tracking_config=adaptive_tracking_cfg,
        auto_tune=auto_tune_enabled,
        plan_override_data=plan_override_data,
        plan_override_len=plan_override_len,
        ball_samples=ball_samples,
        fallback_ball_samples=fallback_ball_samples,
        keep_path_lookup_data=keep_path_lookup_data,
        debug_ball_overlay=debug_ball_overlay,
        follow_override=follow_override_map,
        disable_controller=disable_controller,
        follow_trajectory=None,
        debug_pan_overlay=bool(getattr(args, "debug_pan_overlay", False)),
    )
    renderer.src_width = float(width)
    renderer.src_height = float(height)
    renderer.follow_crop_width = float(follow_crop_width)
    renderer.original_src_w = float(width)
    renderer.original_src_h = float(height)

    follow_centers: Optional[Sequence[float]] = None
    follow_centers_int: Optional[list[int]] = None
    pan_plan: Optional[np.ndarray] = None
    fps_for_path = float(fps_in if fps_in and fps_in > 0 else fps_out or 30.0)
    portrait_crop_width = float(follow_crop_width or width)

    # When CameraPlanner has produced valid states from motion-centroid
    # telemetry, let it drive panning directly.  Skip the Option C
    # override which uses a separate, less-accurate tracking algorithm.
    _use_camera_planner_direct = bool(states and len(states) > 0)
    if _use_camera_planner_direct:
        logger.info("[PAN] CameraPlanner driving panning directly (Option C disabled)")
        # Extract CameraPlanner centers so renderer has a plan, but
        # do NOT set follow_centers (which would override state.cx).
        cam_plan_cx = np.array([s.cx for s in states], dtype=float)
        renderer.follow_plan_x = cam_plan_cx
        # Do NOT set renderer.pan_x_plan — this would override state.cx
        # in _compose_frame via active_pan_plan.
        renderer.pan_x_plan = None

    elif follow_override_map is None:

        if portrait_crop_width > 0 and width > 0:
            total_frames_for_pan = len(states) if states else int(round(duration_s * fps_for_path))
            (
                centers_option_c,
                crop_scales_option_c,
                recovery_flags_option_c,
            ) = build_option_c_follow_centers(
                ball_samples=ball_samples,
                fps=fps_for_path,
                src_w=int(width),
                portrait_w=int(round(portrait_crop_width)),
                frame_count=total_frames_for_pan if total_frames_for_pan > 0 else int(frame_count or 0),
                anticipation_s=0.25,
                max_speed_px_per_s=900.0,
                debug_pan_overlay=bool(getattr(args, "debug_pan_overlay", False)),
            )
            centers_option_c = _resample_path(centers_option_c, int(frame_count))
            centers_option_c = np.nan_to_num(centers_option_c, nan=float(width) / 2.0)
            crop_scales_option_c = _resample_path(crop_scales_option_c, int(frame_count))
            recovery_flags_option_c = _resample_path(recovery_flags_option_c, int(frame_count))
            recovery_flags_option_c = recovery_flags_option_c > 0.5
            follow_centers = centers_option_c.tolist()
            follow_centers_int = [int(round(x)) for x in centers_option_c]
            renderer.follow_plan_x = centers_option_c
            renderer.pan_x_plan = centers_option_c
            renderer.follow_crop_scale_plan = np.asarray(crop_scales_option_c, dtype=float)
            renderer.pan_recovery_flags = np.asarray(recovery_flags_option_c, dtype=bool)

    if renderer.pan_x_plan is None and not _use_camera_planner_direct:
        frames_for_plan = len(states)
        (
            centers_option_c,
            crop_scales_option_c,
            recovery_flags_option_c,
        ) = build_option_c_follow_centers(
            ball_samples=ball_samples,
            fps=fps_for_path,
            src_w=int(width),
            portrait_w=int(round(portrait_crop_width)),
            frame_count=frames_for_plan,
            anticipation_s=0.25,
            max_speed_px_per_s=900.0,
            debug_pan_overlay=bool(getattr(args, "debug_pan_overlay", False)),
        )
        centers_option_c = _resample_path(centers_option_c, frames_for_plan)
        centers_option_c = np.nan_to_num(centers_option_c, nan=float(width) / 2.0)
        crop_scales_option_c = _resample_path(crop_scales_option_c, frames_for_plan)
        recovery_flags_option_c = _resample_path(recovery_flags_option_c, frames_for_plan)
        recovery_flags_option_c = recovery_flags_option_c > 0.5
        renderer.pan_x_plan = centers_option_c
        renderer.follow_crop_scale_plan = np.asarray(crop_scales_option_c, dtype=float)
        renderer.pan_recovery_flags = np.asarray(recovery_flags_option_c, dtype=bool)
        if follow_centers is None:
            follow_centers = centers_option_c.tolist()
            follow_centers_int = [int(round(x)) for x in centers_option_c]

    if renderer.follow_plan_x is None and follow_centers is not None:
        renderer.follow_plan_x = np.asarray(follow_centers, dtype=float)
    if renderer.pan_x_plan is None and pan_plan is not None:
        renderer.pan_x_plan = pan_plan
    elif renderer.pan_x_plan is None and renderer.follow_plan_x is not None:
        renderer.pan_x_plan = renderer.follow_plan_x

    pan_plan_for_render = getattr(renderer, "pan_x_plan", None)
    num_frames = len(states)
    assert pan_plan_for_render is not None, "Follow plan missing centers_x"
    def _resample_1d(seq, n):
        """Resample seq to length n with linear interpolation (or pad/trim)."""
        seq = list(seq) if seq is not None else []
        if n <= 0:
            return []
        if len(seq) == n:
            return seq
        if len(seq) == 0:
            return [0.0] * n
        if len(seq) == 1:
            return [seq[0]] * n
        # linear resample
        out = []
        m = len(seq)
        for i in range(n):
            t = (i * (m - 1)) / (n - 1)  # 0..m-1
            j = int(t)
            a = t - j
            if j >= m - 1:
                out.append(seq[-1])
            else:
                out.append((1 - a) * seq[j] + a * seq[j + 1])
        return out

    def _align_pan_plan(pan_plan, n):
        """pan_plan can be list[float] or list[tuple] or list[dict]; coerce to length n."""
        if pan_plan is None:
            return None
        if len(pan_plan) == n:
            return pan_plan

        # floats (center_x only)
        if isinstance(pan_plan[0], (int, float)):
            return _resample_1d(pan_plan, n)

        # tuples/lists like (cx, cy) or (cx,)
        if isinstance(pan_plan[0], (tuple, list)):
            k = len(pan_plan[0])
            cols = [[] for _ in range(k)]
            for row in pan_plan:
                for c in range(k):
                    cols[c].append(row[c])
            cols_rs = [_resample_1d(col, n) for col in cols]
            return [tuple(cols_rs[c][i] for c in range(k)) for i in range(n)]

        # dicts like {"cx":..., "cy":...}
        if isinstance(pan_plan[0], dict):
            keys = list(pan_plan[0].keys())
            cols = {k: [] for k in keys}
            for row in pan_plan:
                for k in keys:
                    cols[k].append(row.get(k))
            # only interpolate numeric keys; for non-numeric, pad/trim using nearest
            out_cols = {}
            for k in keys:
                if all(isinstance(v, (int, float)) for v in cols[k] if v is not None):
                    # replace None with last known numeric to avoid interpolation crashes
                    cleaned = []
                    last = 0.0
                    for v in cols[k]:
                        if isinstance(v, (int, float)):
                            last = float(v)
                        cleaned.append(last)
                    out_cols[k] = _resample_1d(cleaned, n)
                else:
                    # nearest pad/trim
                    base = cols[k]
                    if len(base) == 0:
                        out_cols[k] = [None] * n
                    elif len(base) >= n:
                        out_cols[k] = base[:n]
                    else:
                        out_cols[k] = base + [base[-1]] * (n - len(base))
            return [{k: out_cols[k][i] for k in keys} for i in range(n)]

        # fallback: just pad/trim
        if len(pan_plan) >= n:
            return pan_plan[:n]
        return pan_plan + [pan_plan[-1]] * (n - len(pan_plan))

    # --- where the assert used to be ---
    if len(pan_plan_for_render) != num_frames:
        print(
            "[WARN] Follow plan length mismatch: "
            f"plan={len(pan_plan_for_render)} num_frames={num_frames}. Aligning by resample/pad."
        )
        pan_plan_for_render = _align_pan_plan(pan_plan_for_render, num_frames)

    # If it STILL doesn't match, then something is deeply broken.
    assert len(pan_plan_for_render) == num_frames, (
        "Follow plan centers_x length mismatch (after align)"
    )
    # ----------------------------------------------------------------
    # SELF-CORRECTING RENDER LOOP
    # Render frames, validate ball visibility via YOLO on the portrait
    # output, correct camera plan if needed, and re-render.  Converges
    # in 1-3 passes.
    # ----------------------------------------------------------------
    _MAX_CORRECTION_PASSES = 3
    _src_width = float(width) if width > 0 else 1920.0

    for _pass_i in range(_MAX_CORRECTION_PASSES):
        jerk95 = renderer.write_frames(
            states,
            follow_centers=follow_centers_int or follow_centers,
            pan_x_plan=pan_plan_for_render,
        )

        if _pass_i >= _MAX_CORRECTION_PASSES - 1:
            break  # last pass — accept whatever we have

        _val_frames_dir = Path(renderer.temp_dir) / "frames"
        _val_corrections = validate_portrait_framing(
            _val_frames_dir,
            states,
            positions,
            fusion_source_labels,
            _src_width,
            sample_every=5,
        )

        if _val_corrections is None:
            print(f"[VALIDATE] Pass {_pass_i + 1}: PASSED — no corrections needed")
            break

        # Protect kick-hold frames from validation corrections — these
        # were deliberately placed on the kicker and must not be shifted.
        if _kick_hold_end > 0 and _val_corrections is not None:
            _val_corrections[:_kick_hold_end] = 0.0

        # Apply corrections
        _val_applied = apply_framing_corrections(states, _val_corrections, _src_width)

        # Re-apply speed limiter after corrections (skip kick-hold frames)
        # Use the same preset-derived speed limit and event-type scaling
        # as the main speed limiter.
        _sl_fps_val = float(fps_in if fps_in and fps_in > 0 else fps_out or 30.0)
        _val_speed_clipped = 0
        _val_max_speed = max(15.0, speed_limit * _sl_event_scale / _sl_fps_val)
        _val_sl_start = max(1, _kick_hold_end)  # don't speed-limit the hold
        _val_trans_max_speed = max(25.0, _val_max_speed * 1.2)  # faster during transition
        for _vsi in range(_val_sl_start, len(states)):
            _eff_max_v = _val_trans_max_speed if (_kick_hold_end > 0 and _vsi <= _kick_hold_trans_end) else _val_max_speed
            _vd = states[_vsi].cx - states[_vsi - 1].cx
            if abs(_vd) > _eff_max_v:
                _vc = states[_vsi - 1].cx + np.sign(_vd) * _eff_max_v
                states[_vsi].cx = _vc
                _hcw = states[_vsi].crop_w / 2.0
                states[_vsi].x0 = float(np.clip(
                    _vc - _hcw, 0.0, max(0.0, _src_width - states[_vsi].crop_w),
                ))
                _val_speed_clipped += 1

        # Rebuild the pan plan from the corrected states so the next
        # render pass actually uses the updated cx values.  Without
        # this, pan_x in _compose_frame would override state.cx with
        # stale pre-correction values, silently discarding fixes.
        pan_plan_for_render = np.array([s.cx for s in states], dtype=float)

        print(
            f"[VALIDATE] Pass {_pass_i + 1}: corrected {_val_applied} frames, "
            f"speed-limited {_val_speed_clipped} — re-rendering..."
        )

    assert renderer is not None

    keyint = max(1, int(round(float(keyint_factor) * float(fps_out))))
    log_path = Path(args.log).expanduser() if args.log else None

    # ------------------------------------------------------------
    # FINAL STITCHING STEP
    # Assemble rendered PNG frames into the final output MP4.
    # ------------------------------------------------------------
    import subprocess

    frames_dir = Path(renderer.temp_dir) / "frames"

    # Frame pattern used by Renderer.write_frames()
    frame_pattern = str(frames_dir / "frame_%06d.png")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output_path = _temp_output_path(output_path)
    if temp_output_path.exists():
        temp_output_path.unlink(missing_ok=True)
    # Use source fps for frame stitching since write_frames() passes through
    # every source frame without rate conversion.  The output can be re-encoded
    # to a different fps later if needed.
    stitch_fps = fps_in if fps_in > 0 else fps_out
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(stitch_fps),
        "-i", frame_pattern,
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-g", str(keyint),
        os.fspath(temp_output_path),
    ]

    renderer.last_ffmpeg_command = list(ffmpeg_cmd)

    print(f"[FFMPEG] Stitching frames from {frames_dir} -> {output_path}")
    subprocess.run(ffmpeg_cmd, check=True)
    temp_output_path.replace(output_path)

    print("[DONE] Video stitched successfully.")

    log_dict["jerk95"] = float(jerk95)

    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "input": os.fspath(input_path),
            "output": os.fspath(output_path),
            "fps_in": float(fps_in),
            "fps_out": float(fps_out),
            "labels_found": int(len(raw_points)),
            "preset": preset_key,
            "ffmpeg_command": renderer.last_ffmpeg_command,
        }
        summary["jerk95"] = float(jerk95)
        summary.update(log_dict)
        with log_path.open("w", encoding="utf-8") as handle:
            json.dump(to_jsonable(log_dict), handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    if not getattr(args, "keep_scratch", False):
        for scratch_path in scratch_cleanup_paths:
            if scratch_path.is_dir():
                shutil.rmtree(scratch_path, ignore_errors=True)
            elif scratch_path.is_file():
                scratch_path.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified cinematic ball-follow renderer")
    def _add_argument_once(p, *option_strings, **kwargs):
        # Avoid argparse crash if flags are accidentally defined twice
        for a in p._actions:
            if any(os in a.option_strings for os in option_strings):
                return
        p.add_argument(*option_strings, **kwargs)

    parser.add_argument("--in", dest="in_path", required=True, help="Input MP4 path")
    parser.add_argument("--src", dest="src", help="Legacy compatibility input path (ignored)")
    parser.add_argument("--out", dest="out", help="Output MP4 path")
    parser.add_argument(
        "--no-clobber",
        action="store_true",
        help="Skip rendering if output exists and is non-empty.",
    )
    parser.add_argument(
        "--scratch-root",
        help="Root directory for scratch artifacts (default: out/_scratch).",
    )
    parser.add_argument(
        "--keep-scratch",
        action="store_true",
        help="Keep scratch artifacts under out/_scratch after rendering.",
    )
    parser.add_argument("--preset", dest="preset", default="cinematic", help="Preset name to load from render_presets.yaml")
    parser.add_argument("--portrait", dest="portrait", help="Portrait canvas WxH")
    parser.add_argument(
        "--portrait-plan-margin",
        dest="portrait_plan_margin",
        type=float,
        help="Margin in px for offline portrait planner keep-in-frame band",
    )
    parser.add_argument(
        "--portrait-plan-headroom",
        dest="portrait_plan_headroom",
        type=float,
        help="Headroom fraction override for offline portrait planner",
    )
    parser.add_argument(
        "--portrait-plan-lead",
        dest="portrait_plan_lead",
        type=float,
        help="Lead distance (px) for offline portrait planner",
    )
    parser.add_argument(
        "--upscale",
        action="store_true",
        help="Upscale source with Real-ESRGAN (or ffmpeg fallback) before processing.",
    )
    parser.add_argument(
        "--upscale-scale",
        type=int,
        default=2,
        help="Scale factor for upscaling (default: 2).",
    )
    parser.add_argument(
        "--force-reupscale",
        dest="force_reupscale",
        action="store_true",
        help="Force regeneration of upscaled video (ignore cache).",
    )
    parser.add_argument("--fps", dest="fps", type=float, help="Output FPS")
    parser.add_argument("--flip180", dest="flip180", action="store_true", help="Rotate frames by 180 degrees before processing")
    parser.add_argument("--labels-root", dest="labels_root", help="Root directory containing YOLO label shards")
    parser.add_argument("--clean-temp", dest="clean_temp", action="store_true", help="Remove temporary frame folder before rendering")
    parser.add_argument("--resume", dest="resume", action="store_true", help="Reuse existing temp frames when resuming a previous run")
    parser.add_argument(
        "--lookahead",
        "--follow-lookahead",
        dest="follow_lookahead",
        type=int,
        default=None,
        help="frames to look ahead when following (defaults to preset)",
    )
    parser.add_argument(
        "--follow-zeta",
        dest="follow_zeta",
        type=float,
        default=None,
        help="2nd-order damping ratio (>=1 is overdamped; defaults to preset)",
    )
    parser.add_argument(
        "--follow-wn",
        dest="follow_wn",
        type=float,
        default=None,
        help="2nd-order natural freq (rad/s; defaults to preset)",
    )
    parser.add_argument(
        "--deadzone",
        dest="deadzone",
        type=float,
        default=None,
        help="pixels; ignore target error inside this radius (defaults to preset)",
    )
    parser.add_argument(
        "--max-vel",
        dest="max_vel",
        type=float,
        default=None,
        help="px/s clamp on camera velocity (defaults to preset)",
    )
    parser.add_argument(
        "--max-acc",
        dest="max_acc",
        type=float,
        default=None,
        help="px/s^2 clamp on camera acceleration (defaults to preset)",
    )
    parser.add_argument(
        "--pre-smooth",
        dest="pre_smooth",
        type=float,
        default=None,
        help="EMA alpha to pre-smooth bx/by (0..1; defaults to preset)",
    )
    parser.add_argument(
        "--jerk-threshold",
        dest="jerk_threshold",
        type=float,
        default=0.0,
        help="Max allowed camera jerk95 in px/s^3 (0 disables the gate)",
    )
    parser.add_argument(
        "--jerk-wn-scale",
        dest="jerk_wn_scale",
        type=float,
        default=0.9,
        help="Multiplier applied to --follow-wn when jerk exceeds the threshold",
    )
    parser.add_argument(
        "--jerk-deadzone-step",
        dest="jerk_deadzone_step",
        type=float,
        default=2.0,
        help="Deadzone increment (px) when jerk exceeds the threshold",
    )
    parser.add_argument(
        "--jerk-max-attempts",
        dest="jerk_max_attempts",
        type=int,
        default=3,
        help="Maximum number of retune attempts when enforcing jerk threshold",
    )
    parser.add_argument(
        "--plan-lookahead",
        dest="plan_lookahead",
        type=int,
        help="Frames of lookahead for planning",
    )
    parser.add_argument("--smoothing", dest="smoothing", type=float, help="EMA smoothing factor")
    parser.add_argument("--pad", dest="pad", type=float, help="Edge padding ratio used to derive zoom")
    parser.add_argument("--speed-limit", dest="speed_limit", type=float, help="Maximum pan speed in px/sec")
    parser.add_argument("--zoom-min", dest="zoom_min", type=float, help="Minimum zoom multiplier")
    parser.add_argument("--zoom-max", dest="zoom_max", type=float, help="Maximum zoom multiplier")
    parser.add_argument(
        "--zoom-out-max",
        dest="zoom_out_max",
        type=float,
        default=None,
        help="Maximum automatic zoom-out multiplier",
    )
    parser.add_argument(
        "--zoom-edge-frac",
        dest="zoom_edge_frac",
        type=float,
        default=0.80,
        help="Fraction of safe margin where zoom-out begins to ease out",
    )
    parser.add_argument(
        "--cy-frac",
        dest="cy_frac",
        type=float,
        default=0.46,
        help=(
            "Desired ball vertical position as a fraction of frame height (0=top, 1=bottom). "
            "Default 0.46 puts ball slightly above center to avoid bottom-edge saturation."
        ),
    )
    parser.add_argument(
        "--emergency-gain",
        dest="emergency_gain",
        type=float,
        default=0.6,
        help="Emergency recenter gain when the ball breaches the safety margin",
    )
    parser.add_argument(
        "--emergency-zoom-max",
        dest="emergency_zoom_max",
        type=float,
        default=1.45,
        help="Maximum emergency zoom-out multiplier to keep the ball in view",
    )
    parser.add_argument(
        "--lost-hold-ms",
        type=int,
        default=500,
        help="when ball leaves frame, hold the last camera center this long (ms)",
    )
    parser.add_argument(
        "--lost-pan-ms",
        type=int,
        default=1200,
        help="duration of the slow pan toward re-entry (ms)",
    )
    parser.add_argument(
        "--lost-chase-motion-ms",
        type=int,
        default=900,
        help="during LOST, time to pan toward motion centroid before re-entry (ms)",
    )
    parser.add_argument(
        "--lost-motion-thresh",
        type=float,
        default=1.6,
        help="optical flow magnitude threshold (px/frame) to pick 'action' blobs",
    )
    parser.add_argument(
        "--lost-use-motion",
        action="store_true",
        help="enable motion centroid chase while ball is out",
    )
    parser.add_argument(
        "--lost-lookahead-s",
        type=float,
        default=6.0,
        help="search window ahead to find when ball returns inside",
    )
    parser.add_argument(
        "--keepinview-margin",
        dest="keepinview_margin",
        type=float,
        help="Safety band in pixels for the keep-in-view guard (defaults to follow margin)",
    )
    parser.add_argument(
        "--keepinview-nudge",
        dest="keepinview_nudge",
        type=float,
        help="Proportional gain for keep-in-view recenter nudges",
    )
    parser.add_argument(
        "--keepinview-zoom",
        dest="keepinview_zoom",
        type=float,
        help="Gain controlling keep-in-view adaptive zoom-out strength",
    )
    parser.add_argument(
        "--keepinview-zoom-cap",
        dest="keepinview_zoom_cap",
        type=float,
        help="Maximum keep-in-view zoom-out multiplier (defaults to follow zoom limit)",
    )
    parser.add_argument(
        "--telemetry",
        "--ball-telemetry",
        dest="telemetry",
        help="Input ball telemetry JSONL (default: out/telemetry/<clip>.ball.jsonl)",
    )
    parser.add_argument(
        "--use-ball-telemetry",
        action="store_true",
        help="Enable ball-aware portrait planning when telemetry is available",
    )
    parser.add_argument(
        "--ball-min-sanity",
        type=float,
        default=None,
        help="Minimum sanity score required to trust ball telemetry (default: 0.6, auto preset: 0.10).",
    )
    _add_argument_once(
        parser,
        "--ball-fallback-red",
        action="store_true",
        help="If ball telemetry sanity is low, try HSV red-ball",
    )
    parser.add_argument(
        "--debug-ball-overlay",
        action="store_true",
        help="Draw detected ball markers before cropping (requires --use-ball-telemetry)",
    )
    parser.add_argument(
        "--draw-ball",
        action="store_true",
        help="Overlay red dot ball position for debugging",
    )
    parser.add_argument(
        "--no-draw-ball",
        action="store_true",
        help="Disable ball overlay even if preset enables it",
    )
    parser.add_argument(
        "--yolo-exclude",
        dest="yolo_exclude",
        type=str,
        default=None,
        help=(
            "Path to a JSON file defining YOLO exclusion zones. "
            "Each zone is a rectangle (x_min/x_max/y_min/y_max in pixels) "
            "with optional frame_start/frame_end bounds. Detections inside "
            "any zone are dropped before fusion. Useful for suppressing "
            "stray balls on sidelines or adjacent fields."
        ),
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Write per-frame diagnostic CSV alongside output (ball position, crop, source, flags)",
    )
    parser.add_argument(
        "--debug-pan-overlay",
        action="store_true",
        help="Draw pan/crop diagnostics before cropping",
    )
    parser.add_argument("--render-telemetry", dest="render_telemetry", help="Output JSONL telemetry file")
    parser.add_argument(
        "--plan",
        dest="plan",
        help="Optional camera plan JSON (skips the internal planner)",
    )
    parser.add_argument(
        "--telemetry-out",
        dest="telemetry_out",
        default=None,
        help="Write per-frame JSONL with t,cx,cy,bx,by",
    )
    parser.add_argument("--brand-overlay", dest="brand_overlay", help="PNG overlay composited on every frame")
    parser.add_argument("--endcard", dest="endcard", help="Optional endcard image displayed for ~2 seconds")
    parser.add_argument("--log", dest="log", help="Optional render log path")
    parser.add_argument("--crf", dest="crf", type=int, help="Override CRF value")
    parser.add_argument("--keyint-factor", dest="keyint_factor", type=int, help="Override keyint factor")
    parser.add_argument(
        "--ball-key-x",
        dest="ball_key_x",
        default=None,
        help="Preferred X key when reading planned ball path JSONL",
    )
    parser.add_argument(
        "--ball-key-y",
        dest="ball_key_y",
        default=None,
        help="Preferred Y key when reading planned ball path JSONL",
    )
    parser.add_argument(
        "--init-manual",
        dest="init_manual",
        action="store_true",
        help="Manually select the ball ROI at the start of the clip.",
    )
    parser.add_argument(
        "--init-t",
        dest="init_t",
        type=float,
        default=0.8,
        help="Time in seconds to show for manual init (default 0.8).",
    )
    parser.add_argument(
        "--ball-path",
        dest="ball_path",
        help="JSONL from ball_path_planner_v2.py",
    )
    parser.add_argument(
        "--max-duration",
        dest="max_duration_s",
        type=float,
        default=None,
        help="Hard cap on output duration in seconds. Frames beyond this are discarded.",
    )
    parser.add_argument(
        "--follow-override",
        dest="follow_override",
        type=str,
        default=None,
        help="Path to a JSONL file with explicit per-frame cx/cy/zoom overrides.",
    )
    parser.add_argument(
        "--follow",
        dest="follow_mode",
        choices=["option_c"],
        default="option_c",
        help="Portrait follow mode (option_c: segment + smoothing + anticipation).",
    )
    parser.add_argument(
        "--event-type",
        dest="event_type",
        default=None,
        help="Event type for this clip (GOAL, SHOT, CROSS). Adjusts camera "
             "speed limits and keepinview parameters for the event dynamics.",
    )
    return parser


def _load_override_samples(path: Optional[str]) -> Optional[list[dict]]:
    if not path:
        return None

    samples: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if "frame" in row and "cx" in row and "cy" in row:
                samples.append(row)
    return samples


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(argv)
    override_samples = _load_override_samples(args.follow_override)
    follow_lookahead_cli = any(
        arg == "--lookahead" or arg.startswith("--lookahead=") for arg in raw_argv
    )
    setattr(args, "_follow_lookahead_cli", follow_lookahead_cli)
    # --- portrait helpers ---
    portrait_w, portrait_h = (None, None)
    setattr(args, "portrait_w", portrait_w)
    setattr(args, "portrait_h", portrait_h)
    render_telemetry_path: Optional[Path] = None
    telemetry_simple_path: Optional[Path] = None
    if getattr(args, "telemetry", None):
        setattr(args, "ball_telemetry", args.telemetry)
    if getattr(args, "render_telemetry", None):
        render_telemetry_path = Path(args.render_telemetry).expanduser()
        render_telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        args.render_telemetry = os.fspath(render_telemetry_path)
    if getattr(args, "telemetry_out", None):
        telemetry_simple_path = Path(args.telemetry_out).expanduser()
        telemetry_simple_path.parent.mkdir(parents=True, exist_ok=True)
        args.telemetry_out = os.fspath(telemetry_simple_path)

    if getattr(args, "use_ball_telemetry", False) and not getattr(args, "ball_telemetry", None):
        # Resolve input video path (argparse uses in_path for --in)
        video_path = getattr(args, "in_path", None)

        # Backward compatibility (older variants)
        if video_path is None:
            video_path = getattr(args, "input", None)

        if not video_path:
            raise ValueError("No input video provided (expected --in <video>)")

        args.ball_telemetry = telemetry_path_for_video(Path(video_path))

    setattr(args, "follow_override_samples", override_samples)
    run(args, telemetry_path=render_telemetry_path, telemetry_simple_path=telemetry_simple_path)


if __name__ == "__main__":
    import traceback

    try:
        main()
    except SystemExit:
        # allow argparse / sys.exit() to behave normally
        raise
    except Exception as exc:
        print("\n[FATAL] Unhandled exception in render_follow_unified.py:")
        print(f"  {exc!r}")
        traceback.print_exc()
        # re-raise so the process has a non-zero exit code
        raise



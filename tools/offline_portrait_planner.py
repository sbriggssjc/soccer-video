"""Offline portrait camera planner that consumes telemetry samples.

The helper is split into three layers:

``BallSample``
    Defined in :mod:`tools.ball_telemetry`; represents per-frame ball
    detections.
``OfflinePortraitPlanner``
    Smooths/filters the telemetry into window coordinates that satisfy the
    aspect-ratio and keep-in-view constraints.
``CameraKeyframe`` / plan helpers
    Convert planner output into serialisable keyframes that can be reloaded by
    :mod:`tools.render_follow_unified` without re-running the planner.

The new workflow is:

1. Load telemetry via :func:`tools.ball_telemetry.load_ball_telemetry`.
2. Call :func:`plan_camera_from_ball` with source dimensions + portrait aspect.
3. Save the resulting keyframes using :func:`save_plan`.
4. Render portrait reels by pointing :mod:`tools.render_follow_unified` at the
   saved ``.plan.json`` file.
"""

from __future__ import annotations

import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from tools.ball_telemetry import BallSample, load_ball_telemetry


DEBUG_KEEPINVIEW = False
def compute_follow_trajectory(ball_samples, src_w, src_h, fps,
                              zoom_min=1.0, zoom_max=2.8,
                              smoothing=0.85, center_weight=0.65):
    """
    Local fallback trajectory builder for portrait follow.
    """
    out = []
    last_cx = src_w * 0.5
    last_cy = src_h * 0.55
    last_zoom = 1.4

    for b in ball_samples:
        cx = b.cx
        cy = b.cy

        # temporal smoothing
        cx = last_cx * (1 - smoothing) + cx * smoothing
        cy = last_cy * (1 - smoothing) + cy * smoothing

        # zoom based on ball distance from center
        dx = abs(cx - src_w * 0.5)
        dy = abs(cy - src_h * 0.5)
        dist = max(dx, dy)
        norm = dist / (src_w * 0.5)

        zoom = zoom_min + (zoom_max - zoom_min) * (1 - center_weight * (1 - norm))
        zoom = max(zoom_min, min(zoom, zoom_max))

        out.append({
            "f": b.f,
            "t": b.t,
            "cx": float(cx),
            "cy": float(cy),
            "zoom": float(zoom),
        })

        last_cx, last_cy, last_zoom = cx, cy, zoom

    return out


def _infer_fps_from_samples(samples: Sequence[BallSample]) -> float:
    prev_frame: Optional[int] = None
    prev_time: Optional[float] = None
    fps_values: List[float] = []
    for sample in samples:
        if not math.isfinite(sample.t):
            continue
        frame = int(sample.frame)
        if prev_frame is not None and prev_time is not None:
            frame_delta = frame - prev_frame
            time_delta = sample.t - prev_time
            if frame_delta > 0 and time_delta > 1e-6:
                fps_values.append(frame_delta / time_delta)
        prev_frame = frame
        prev_time = sample.t
    if fps_values:
        return max(1.0, float(sum(fps_values) / len(fps_values)))
    return 30.0


def _critically_damped(values: List[float], fps: float, follow_strength: float) -> List[float]:
    if not values:
        return []
    dt = 1.0 / max(fps, 1.0)
    omega = max(0.5, float(follow_strength) * 12.0)
    pos = float(values[0])
    vel = 0.0
    smoothed: List[float] = []
    for target in values:
        delta = float(target) - pos
        accel = (omega * omega) * delta - 2.0 * omega * vel
        vel += accel * dt
        pos += vel * dt
        smoothed.append(pos)
    return smoothed


def plan_keepinview_path(
    samples: List[BallSample],
    src_w: int,
    src_h: int,
    portrait_w: int,
    portrait_h: int,
    *,
    band_margin_px: int = 80,
    reacquire_max_gap_s: float = 0.5,
    follow_strength: float = 0.8,
) -> Dict[int, Tuple[float, float]]:
    """Compute a per-frame portrait centre that keeps the ball inside the crop."""

    if not samples or src_w <= 0 or src_h <= 0 or portrait_w <= 0 or portrait_h <= 0:
        return {}

    sorted_samples = sorted(samples, key=lambda s: int(s.frame))
    frames_all = [int(s.frame) for s in sorted_samples]
    if not frames_all:
        return {}

    frame_start = min(frames_all)
    frame_end = max(frames_all)
    fps_guess = _infer_fps_from_samples(sorted_samples)
    max_gap_frames = max(1, int(round(reacquire_max_gap_s * fps_guess)))

    best_sample_per_frame: Dict[int, BallSample] = {}
    for sample in sorted_samples:
        frame = int(sample.frame)
        prev = best_sample_per_frame.get(frame)
        if prev is None or sample.conf >= prev.conf:
            best_sample_per_frame[frame] = sample

    raw_positions: Dict[int, Optional[Tuple[float, float]]] = {}
    last_pos: Optional[Tuple[float, float]] = None
    last_frame_with_pos: Optional[int] = None
    last_velocity = (0.0, 0.0)
    for frame in range(frame_start, frame_end + 1):
        sample = best_sample_per_frame.get(frame)
        if sample and math.isfinite(sample.x) and math.isfinite(sample.y):
            pos = (float(sample.x), float(sample.y))
            if last_pos is not None and last_frame_with_pos is not None and frame > last_frame_with_pos:
                dt = frame - last_frame_with_pos
                if dt > 0:
                    last_velocity = (
                        (pos[0] - last_pos[0]) / dt,
                        (pos[1] - last_pos[1]) / dt,
                    )
            else:
                last_velocity = (0.0, 0.0)
            last_pos = pos
            last_frame_with_pos = frame
            raw_positions[frame] = pos
        else:
            if last_pos is not None and last_frame_with_pos is not None:
                gap = frame - last_frame_with_pos
                if gap <= max_gap_frames:
                    pos = (
                        last_pos[0] + last_velocity[0] * gap,
                        last_pos[1] + last_velocity[1] * gap,
                    )
                    raw_positions[frame] = pos
                    continue
            last_pos = None
            last_frame_with_pos = None
            raw_positions[frame] = None

    half_w = 0.5 * float(portrait_w)
    half_h = 0.5 * float(portrait_h)
    margin_x = min(float(band_margin_px), max(0.0, half_w - 1.0))
    margin_y = min(float(band_margin_px), max(0.0, half_h - 1.0))
    scene_cx = float(src_w) * 0.5
    scene_cy = float(src_h) * 0.5
    decay_frames = max(max_gap_frames * 2, 1)

    def clamp_center(cx: float, cy: float) -> Tuple[float, float]:
        min_cx = half_w
        max_cx = max(half_w, float(src_w) - half_w)
        min_cy = half_h
        max_cy = max(half_h, float(src_h) - half_h)
        cx = min(max(cx, min_cx), max_cx)
        cy = min(max(cy, min_cy), max_cy)
        return cx, cy

    def enforce_band(bx: float, by: float, cx: float, cy: float) -> Tuple[float, float]:
        if half_w > margin_x:
            min_center = bx - (half_w - margin_x)
            max_center = bx + (half_w - margin_x)
            cx = min(max(cx, min_center), max_center)
        if half_h > margin_y:
            min_center_y = by - (half_h - margin_y)
            max_center_y = by + (half_h - margin_y)
            cy = min(max(cy, min_center_y), max_center_y)
        return cx, cy

    centers: Dict[int, Tuple[float, float]] = {}
    frames_since_ball = decay_frames
    last_target = (scene_cx, scene_cy)
    for frame in range(frame_start, frame_end + 1):
        ball_pos = raw_positions.get(frame)
        if ball_pos is not None and math.isfinite(ball_pos[0]) and math.isfinite(ball_pos[1]):
            frames_since_ball = 0
            cx, cy = ball_pos
            cx, cy = enforce_band(ball_pos[0], ball_pos[1], cx, cy)
        else:
            frames_since_ball += 1
            if frames_since_ball <= max_gap_frames:
                cx, cy = last_target
            else:
                decay = min(1.0, (frames_since_ball - max_gap_frames) / float(decay_frames))
                cx = last_target[0] + (scene_cx - last_target[0]) * decay
                cy = last_target[1] + (scene_cy - last_target[1]) * decay
        cx, cy = clamp_center(cx, cy)
        centers[frame] = (cx, cy)
        last_target = (cx, cy)

    frames_sorted = sorted(centers.keys())
    target_x = [centers[idx][0] for idx in frames_sorted]
    target_y = [centers[idx][1] for idx in frames_sorted]
    smooth_x = _critically_damped(target_x, fps_guess, follow_strength)
    smooth_y = _critically_damped(target_y, fps_guess, follow_strength)

    keep_path: Dict[int, Tuple[float, float]] = {}
    for idx, frame in enumerate(frames_sorted):
        cx = smooth_x[idx]
        cy = smooth_y[idx]
        bx_by = raw_positions.get(frame)
        if bx_by is not None:
            cx, cy = enforce_band(bx_by[0], bx_by[1], cx, cy)
        cx, cy = clamp_center(cx, cy)
        keep_path[frame] = (cx, cy)
        if DEBUG_KEEPINVIEW and idx % 120 == 0:
            bx_val = bx_by[0] if bx_by else float("nan")
            by_val = bx_by[1] if bx_by else float("nan")
            print(
                f"[KEEPINVIEW] frame={frame} cx={cx:.1f} cy={cy:.1f} bx={bx_val:.1f} by={by_val:.1f}"
            )

    return keep_path



@dataclass
class CameraKeyframe:
    """Planned crop state at a given presentation timestamp."""

    t: float
    frame: int
    cx: float
    cy: float
    zoom: float
    width: float
    height: float

    def to_json(self) -> Mapping[str, float]:
        return {
            "t": float(self.t),
            "frame": int(self.frame),
            "cx": float(self.cx),
            "cy": float(self.cy),
            "zoom": float(self.zoom),
            "width": float(self.width),
            "height": float(self.height),
        }


@dataclass
class PlannerConfig:
    """Configuration for :class:`OfflinePortraitPlanner`."""

    frame_size: Tuple[float, float]
    crop_aspect: float = 9.0 / 16.0
    fps: float = 30.0
    margin_px: float = 90.0
    headroom_frac: float = 0.08
    lead_px: float = 120.0
    gap_interp_s: float = 0.75
    smooth_window: int = 7
    max_step_x: float = 32.0
    max_step_y: float = 18.0
    accel_limit_x: float = 3.5
    accel_limit_y: float = 2.0
    smoothing_passes: int = 2
    portrait_pad: float = 24.0
    keep_in_frame_frac_x: Tuple[float, float] = (0.4, 0.6)
    keep_in_frame_frac_y: Tuple[float, float] = (0.4, 0.6)
    min_zoom: float = 1.0
    max_zoom: float = 2.0
    # Adaptive tracking: when True, margin_px / lead_px / max_step are
    # modulated per-frame based on ball speed.  The cfg values become the
    # *base* (slow-play) values; during fast play the planner tightens
    # margin, increases lead, and raises the speed limit automatically.
    adaptive: bool = False
    # Speed thresholds (px/frame) for the adaptive ramp.
    adaptive_v_lo: float = 1.5
    adaptive_v_hi: float = 10.0
    # Per-parameter scale at max speed (multiplied against the base value).
    adaptive_margin_scale: float = 2.0   # 2x base margin at full speed (tighter)
    adaptive_lead_scale: float = 2.0     # 2x base lead at full speed
    adaptive_step_scale: float = 2.5     # 2.5x base max_step at full speed

    def __post_init__(self) -> None:
        w, h = self.frame_size
        if w <= 0 or h <= 0:
            raise ValueError("frame_size must be positive")
        if not math.isfinite(self.crop_aspect) or self.crop_aspect <= 0:
            self.crop_aspect = 9.0 / 16.0
        if self.fps <= 0:
            self.fps = 30.0
        self.margin_px = max(0.0, float(self.margin_px))
        self.headroom_frac = float(np.clip(self.headroom_frac, -0.2, 0.4))
        self.lead_px = max(0.0, float(self.lead_px))
        self.gap_interp_s = max(0.0, float(self.gap_interp_s))
        self.smooth_window = max(1, int(self.smooth_window) | 1)
        self.max_step_x = max(1.0, float(self.max_step_x))
        self.max_step_y = max(1.0, float(self.max_step_y))
        self.accel_limit_x = max(0.0, float(self.accel_limit_x))
        self.accel_limit_y = max(0.0, float(self.accel_limit_y))
        self.smoothing_passes = max(1, int(self.smoothing_passes))
        self.portrait_pad = max(0.0, float(self.portrait_pad))
        self.keep_in_frame_frac_x = (
            float(min(max(self.keep_in_frame_frac_x[0], 0.0), 1.0)),
            float(min(max(self.keep_in_frame_frac_x[1], 0.0), 1.0)),
        )
        self.keep_in_frame_frac_y = (
            float(min(max(self.keep_in_frame_frac_y[0], 0.0), 1.0)),
            float(min(max(self.keep_in_frame_frac_y[1], 0.0), 1.0)),
        )
        if self.keep_in_frame_frac_x[0] > self.keep_in_frame_frac_x[1]:
            self.keep_in_frame_frac_x = tuple(reversed(self.keep_in_frame_frac_x))
        if self.keep_in_frame_frac_y[0] > self.keep_in_frame_frac_y[1]:
            self.keep_in_frame_frac_y = tuple(reversed(self.keep_in_frame_frac_y))
        self.min_zoom = max(1.0, float(self.min_zoom))
        self.max_zoom = max(self.min_zoom, float(self.max_zoom))

    # ------------------------------------------------------------------
    # Auto-tuning from telemetry
    # ------------------------------------------------------------------

    @classmethod
    def auto_from_telemetry(
        cls,
        bx: Sequence[float],
        by: Sequence[float],
        frame_size: Tuple[float, float],
        fps: float = 24.0,
        crop_aspect: float = 9.0 / 16.0,
    ) -> "PlannerConfig":
        """Create an auto-tuned config by analyzing a ball trajectory.

        Computes per-clip velocity/acceleration statistics and sets planner
        parameters so the virtual camera can keep up with the fastest action
        in the clip while staying smooth during slow play.
        """
        stats = analyze_ball_positions(bx, by, fps=fps)

        peak_x = stats["peak_speed_x"]
        peak_y = stats["peak_speed_y"]
        p95 = stats["p95_speed"]
        p95_accel = stats["p95_accel"]

        # Base step: comfortably above the 95th-percentile speed so only
        # the most extreme 1-frame spikes need adaptive scaling.
        base_step_x = max(32.0, p95 * 1.4, peak_x * 1.15)
        base_step_y = max(18.0, p95 * 0.8, peak_y * 1.15)

        # Acceleration limits: handle direction reversals.
        base_accel_x = max(3.5, p95_accel * 0.50)
        base_accel_y = max(2.0, p95_accel * 0.30)

        # Adaptive step scale: ensure even the absolute peak is reachable
        # when the adaptive ramp is fully engaged.
        peak_total = stats["peak_speed"]
        step_scale = max(2.0, min(4.0, (peak_total * 1.3) / max(base_step_x, 1.0)))

        # Lead: more anticipation for faster clips.
        speed_factor = min(1.0, p95 / 10.0)
        lead_px = 120.0 + 80.0 * speed_factor          # 120→200 px

        # Margin: tighter base for faster clips (adaptive will widen on
        # slow play to give a looser, cinematic feel).
        margin_px = 90.0 - 30.0 * speed_factor          # 90→60 px

        # Adaptive v_hi: calibrate to the clip's own speed distribution
        # so the ramp reaches 1.0 near the clip's fast-play threshold.
        adaptive_v_hi = max(8.0, p95 * 0.9)

        return cls(
            frame_size=frame_size,
            crop_aspect=crop_aspect,
            fps=fps,
            margin_px=margin_px,
            lead_px=lead_px,
            smooth_window=9,
            max_step_x=base_step_x,
            max_step_y=base_step_y,
            accel_limit_x=base_accel_x,
            accel_limit_y=base_accel_y,
            smoothing_passes=3,
            adaptive=True,
            adaptive_v_lo=1.5,
            adaptive_v_hi=adaptive_v_hi,
            adaptive_margin_scale=2.0,
            adaptive_lead_scale=2.0,
            adaptive_step_scale=step_scale,
        )


def analyze_ball_positions(
    bx: Sequence[float],
    by: Sequence[float],
    fps: float = 24.0,
) -> dict:
    """Compute velocity/acceleration statistics from a ball trajectory.

    Returns a dict with keys: peak_speed, p95_speed, median_speed,
    peak_speed_x, peak_speed_y, peak_accel, p95_accel,
    horizontal_range, vertical_range, fps.
    """
    bx_arr = np.asarray(bx, dtype=float)
    by_arr = np.asarray(by, dtype=float)
    if bx_arr.size < 2:
        return {
            "peak_speed": 0.0, "p95_speed": 0.0, "median_speed": 0.0,
            "peak_speed_x": 0.0, "peak_speed_y": 0.0,
            "peak_accel": 0.0, "p95_accel": 0.0,
            "horizontal_range": 0.0, "vertical_range": 0.0, "fps": fps,
        }

    dx = np.diff(bx_arr)
    dy = np.diff(by_arr)
    speed = np.hypot(dx, dy)

    ax = np.diff(dx)
    ay = np.diff(dy)
    accel = np.hypot(ax, ay) if ax.size > 0 else np.array([0.0])

    return {
        "peak_speed": float(np.max(speed)),
        "p95_speed": float(np.percentile(speed, 95)),
        "median_speed": float(np.median(speed)),
        "peak_speed_x": float(np.max(np.abs(dx))),
        "peak_speed_y": float(np.max(np.abs(dy))),
        "peak_accel": float(np.max(accel)),
        "p95_accel": float(np.percentile(accel, 95)),
        "horizontal_range": float(np.ptp(bx_arr)),
        "vertical_range": float(np.ptp(by_arr)),
        "fps": fps,
    }


def _nan_to_default(arr: np.ndarray, default: float) -> np.ndarray:
    out = arr.copy()
    mask = ~np.isfinite(out)
    if not mask.any():
        return out
    out[mask] = float(default)
    return out


def _fill_short_gaps(arr: np.ndarray, max_gap: int) -> np.ndarray:
    out = arr.copy()
    n = len(out)
    i = 0
    while i < n:
        if math.isfinite(out[i]):
            i += 1
            continue
        start = i
        while i < n and not math.isfinite(out[i]):
            i += 1
        gap_len = i - start
        start_val = out[start - 1] if start > 0 else None
        end_val = out[i] if i < n else None
        if gap_len <= max_gap and start_val is not None and math.isfinite(start_val) and end_val is not None and math.isfinite(end_val):
            interp = np.linspace(start_val, end_val, gap_len + 2)[1:-1]
            out[start:i] = interp
        elif start_val is not None and math.isfinite(start_val):
            out[start:i] = start_val
        elif end_val is not None and math.isfinite(end_val):
            out[start:i] = end_val
    return out


def _moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return arr.copy()
    pad = window // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed.astype(float)


def _speed_limit(series: np.ndarray, mins: np.ndarray, maxs: np.ndarray,
                  max_step, accel_limit, passes: int) -> np.ndarray:
    """Clamp per-frame deltas.

    ``max_step`` and ``accel_limit`` may be scalars **or** per-frame arrays
    (same length as *series*).  When arrays are provided the limit varies
    per-frame, enabling adaptive tracking responsiveness.
    """
    out = series.copy()
    n = len(out)
    step_arr = np.broadcast_to(np.asarray(max_step, dtype=float), (n,))
    accel_arr = np.broadcast_to(np.asarray(accel_limit, dtype=float), (n,))
    for _ in range(passes):
        for idx in range(1, n):
            delta = out[idx] - out[idx - 1]
            ms = step_arr[idx]
            if abs(delta) > ms:
                out[idx] = out[idx - 1] + math.copysign(ms, delta)
            out[idx] = np.clip(out[idx], mins[idx], maxs[idx])
        for idx in range(n - 2, -1, -1):
            delta = out[idx] - out[idx + 1]
            ms = step_arr[idx]
            if abs(delta) > ms:
                out[idx] = out[idx + 1] + math.copysign(ms, delta)
            out[idx] = np.clip(out[idx], mins[idx], maxs[idx])
        if n >= 3:
            for idx in range(1, n - 1):
                al = accel_arr[idx]
                if al <= 0:
                    continue
                jerk = out[idx + 1] - 2 * out[idx] + out[idx - 1]
                if abs(jerk) > al:
                    adjust = 0.5 * (abs(jerk) - al)
                    out[idx] += math.copysign(adjust, jerk)
                    out[idx] = np.clip(out[idx], mins[idx], maxs[idx])
    return out


def _samples_to_series(samples: Sequence[BallSample]) -> Tuple[np.ndarray, np.ndarray, int]:
    frames: List[int] = []
    fallback = 0
    for sample in samples:
        idx = sample.frame_idx if isinstance(sample.frame_idx, int) else fallback
        if idx < 0:
            idx = fallback
        frames.append(idx)
        fallback = idx + 1
    if not frames:
        raise ValueError("Telemetry must contain at least one sample")
    start = min(frames)
    end = max(frames)
    total = max(1, end - start + 1)
    bx = np.full(total, float("nan"), dtype=float)
    by = np.full(total, float("nan"), dtype=float)
    for sample, frame_idx in zip(samples, frames):
        pos = frame_idx - start
        if pos < 0 or pos >= total:
            continue
        if math.isfinite(sample.x):
            bx[pos] = float(sample.x)
        if math.isfinite(sample.y):
            by[pos] = float(sample.y)
    if not np.isfinite(bx).any() or not np.isfinite(by).any():
        raise ValueError("Telemetry does not contain finite ball coordinates")
    return bx, by, start


def plan_ball_portrait_crop(
    samples: Sequence[BallSample],
    *,
    src_w: int,
    src_h: int,
    portrait_w: int,
    portrait_h: int,
    config: PlannerConfig,
) -> Dict[int, Tuple[float, float, float]]:
    """Compute a per-frame (cx, cy, zoom) trajectory that keeps the ball framed.

    The returned mapping is keyed by frame index (source frame numbers) and the
    values are ``(cx, cy, zoom)`` in source pixel space.  The zoom factor uses
    the same convention as :mod:`render_follow_unified` where ``zoom=1.0`` means
    full-frame and larger numbers zoom in.
    """

    if not samples or src_w <= 0 or src_h <= 0 or portrait_w <= 0 or portrait_h <= 0:
        return {}

    bx, by, start_frame = _samples_to_series(list(samples))
    fps_guess = _infer_fps_from_samples(list(samples))
    total = len(bx)

    bx = _fill_short_gaps(bx, max_gap=int(round(config.gap_interp_s * fps_guess)))
    by = _fill_short_gaps(by, max_gap=int(round(config.gap_interp_s * fps_guess)))
    bx = _moving_average(_nan_to_default(bx, np.nanmean(bx)), config.smooth_window)
    by = _moving_average(_nan_to_default(by, np.nanmean(by)), config.smooth_window)

    aspect = float(portrait_w) / float(portrait_h)
    half_w_base = 0.5 * float(portrait_w)
    half_h_base = 0.5 * float(portrait_h)
    band_low_x, band_high_x = config.keep_in_frame_frac_x
    band_low_y, band_high_y = config.keep_in_frame_frac_y

    centres: Dict[int, Tuple[float, float, float]] = {}

    cx_prev = float(src_w) * 0.5
    cy_prev = float(src_h) * 0.5
    zoom_prev = float(config.min_zoom)

    for idx in range(total):
        bx_val = float(bx[idx]) if math.isfinite(bx[idx]) else None
        by_val = float(by[idx]) if math.isfinite(by[idx]) else None
        if bx_val is None or by_val is None:
            centres[start_frame + idx] = (cx_prev, cy_prev, zoom_prev)
            continue

        cx = bx_val
        cy = by_val

        half_w = half_w_base / zoom_prev
        half_h = half_h_base / zoom_prev

        def enforce_band(val: float, half: float, low: float, high: float) -> Tuple[float, float]:
            if low >= high:
                return val, zoom_prev
            min_c = val - (high - 0.5) * (2.0 * half)
            max_c = val - (low - 0.5) * (2.0 * half)
            return min_c, max_c

        min_cx, max_cx = enforce_band(bx_val, half_w, band_low_x, band_high_x)
        min_cy, max_cy = enforce_band(by_val, half_h, band_low_y, band_high_y)

        min_cx = max(min_cx, half_w)
        max_cx = min(max_cx, float(src_w) - half_w)
        min_cy = max(min_cy, half_h)
        max_cy = min(max_cy, float(src_h) - half_h)

        cx = min(max(cx, min_cx), max_cx)
        cy = min(max(cy, min_cy), max_cy)

        pad_x = max(0.0, abs(cx - bx_val) - (half_w - config.portrait_pad))
        pad_y = max(0.0, abs(cy - by_val) - (half_h - config.portrait_pad))
        if pad_x > 0 or pad_y > 0:
            need_zoom = max(
                (abs(cx - bx_val) + config.portrait_pad) / max(half_w, 1e-6),
                (abs(cy - by_val) + config.portrait_pad) / max(half_h, 1e-6),
            )
            zoom_prev = min(max(config.min_zoom, zoom_prev / need_zoom), config.max_zoom)
            half_w = half_w_base / zoom_prev
            half_h = half_h_base / zoom_prev
            cx = min(max(cx, half_w), float(src_w) - half_w)
            cy = min(max(cy, half_h), float(src_h) - half_h)

        cx_prev = cx
        cy_prev = cy
        centres[start_frame + idx] = (cx, cy, zoom_prev)

    return centres


def _plan_to_keyframes(
    plan: Mapping[str, np.ndarray],
    *,
    start_frame: int,
    fps: float,
) -> List[CameraKeyframe]:
    x0 = plan.get("x0")
    y0 = plan.get("y0")
    w = plan.get("w")
    h = plan.get("h")
    cx = plan.get("cx")
    cy = plan.get("cy")
    z = plan.get("z")
    if not all(isinstance(arr, np.ndarray) for arr in (x0, y0, w, h, cx, cy, z)):
        raise ValueError("Plan dictionary is missing required arrays")
    size = len(x0)  # type: ignore[arg-type]
    keyframes: List[CameraKeyframe] = []
    for idx in range(size):
        frame = start_frame + idx
        t = frame / fps if fps > 0 else idx / 30.0
        width = float(w[idx])
        height = float(h[idx])
        keyframes.append(
            CameraKeyframe(
                t=float(t),
                frame=int(frame),
                cx=float(cx[idx]),
                cy=float(cy[idx]),
                zoom=float(z[idx]),
                width=width,
                height=height,
            )
        )
    return keyframes


class OfflinePortraitPlanner:
    """Offline cinematic camera planner for portrait crops."""

    def __init__(self, config: PlannerConfig) -> None:
        self.cfg = config
        self.frame_w, self.frame_h = config.frame_size
        pad = float(config.portrait_pad)
        max_crop_w = max(32.0, self.frame_w - 2 * pad)
        max_crop_h = max(32.0, self.frame_h - 2 * pad)
        crop_w = min(max_crop_w, max_crop_h * config.crop_aspect)
        crop_h = crop_w / config.crop_aspect
        if crop_h > max_crop_h:
            crop_h = max_crop_h
            crop_w = crop_h * config.crop_aspect
        self.crop_w = crop_w
        self.crop_h = crop_h
        self.cx_bounds = (crop_w / 2.0, self.frame_w - crop_w / 2.0)
        self.cy_bounds = (crop_h / 2.0, self.frame_h - crop_h / 2.0)

    def _clean_track(self, coords: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        default = float(coords[np.isfinite(coords)][0]) if np.isfinite(coords).any() else 0.5 * self.frame_w
        arr = _nan_to_default(coords, default)
        max_gap = int(round(cfg.gap_interp_s * cfg.fps))
        if max_gap > 0:
            arr = _fill_short_gaps(arr, max_gap)
        arr = _moving_average(arr, cfg.smooth_window)
        return arr

    def _adaptive_ramp(self, speed: np.ndarray) -> np.ndarray:
        """Return a 0-1 ramp based on ball speed (px/frame).

        0 = slow (at or below ``adaptive_v_lo``),
        1 = fast (at or above ``adaptive_v_hi``).
        The ramp is smoothed with a short moving average so parameter
        transitions don't cause visible jitter.
        """
        cfg = self.cfg
        denom = max(cfg.adaptive_v_hi - cfg.adaptive_v_lo, 1e-6)
        raw = np.clip((speed - cfg.adaptive_v_lo) / denom, 0.0, 1.0)
        # Smooth the ramp with a 7-frame window to prevent flicker
        return _moving_average(raw, max(3, cfg.smooth_window))

    def plan(self, bx: Sequence[float], by: Sequence[float]) -> dict[str, np.ndarray]:
        bx_arr = np.asarray(bx, dtype=float)
        by_arr = np.asarray(by, dtype=float)
        if bx_arr.size == 0 or by_arr.size == 0:
            raise ValueError("ball arrays must be non-empty")
        bx_clean = self._clean_track(bx_arr)
        by_clean = self._clean_track(by_arr)

        vx = np.gradient(bx_clean)
        vy = np.gradient(by_clean)
        speed = np.hypot(vx, vy)
        norm = np.maximum(speed, 1e-6)

        cfg = self.cfg

        if cfg.adaptive:
            # Per-frame ramp: 0 at slow play, 1 at fast play
            t = self._adaptive_ramp(speed)
            # Lead increases with speed (want more anticipation on fast balls)
            lead_scale = cfg.lead_px * (1.0 + (cfg.adaptive_lead_scale - 1.0) * t)
            # Margin shrinks with speed (tighter tracking on fast balls)
            margin_base = cfg.margin_px * (1.0 + (cfg.adaptive_margin_scale - 1.0) * t)
            # Speed limit increases with speed (camera allowed to pan faster)
            max_step_x = cfg.max_step_x * (1.0 + (cfg.adaptive_step_scale - 1.0) * t)
            max_step_y = cfg.max_step_y * (1.0 + (cfg.adaptive_step_scale - 1.0) * t)
            accel_x = cfg.accel_limit_x * (1.0 + (cfg.adaptive_step_scale - 1.0) * t)
            accel_y = cfg.accel_limit_y * (1.0 + (cfg.adaptive_step_scale - 1.0) * t)
        else:
            lead_scale = cfg.lead_px
            margin_base = cfg.margin_px
            max_step_x = cfg.max_step_x
            max_step_y = cfg.max_step_y
            accel_x = cfg.accel_limit_x
            accel_y = cfg.accel_limit_y

        lead_x = lead_scale * (vx / norm)
        lead_y = lead_scale * (vy / norm)
        headroom = cfg.headroom_frac * self.crop_h

        ideal_cx = bx_clean + lead_x
        ideal_cy = by_clean + headroom + lead_y * 0.1

        half_w = self.crop_w / 2.0
        half_h = self.crop_h / 2.0
        margin_x = np.minimum(half_w - 4.0, margin_base)
        margin_y = np.minimum(half_h - 4.0, margin_base * 1.35)
        margin_x = np.maximum(0.0, margin_x)
        margin_y = np.maximum(0.0, margin_y)

        cx_min = np.maximum(self.cx_bounds[0], bx_clean - (half_w - margin_x))
        cx_max = np.minimum(self.cx_bounds[1], bx_clean + (half_w - margin_x))
        cy_min = np.maximum(self.cy_bounds[0], by_clean - (half_h - margin_y))
        cy_max = np.minimum(self.cy_bounds[1], by_clean + (half_h - margin_y))

        cx_min = np.minimum(cx_min, cx_max)
        cy_min = np.minimum(cy_min, cy_max)

        target_cx = np.clip(ideal_cx, cx_min, cx_max)
        target_cy = np.clip(ideal_cy, cy_min, cy_max)

        cx = _speed_limit(
            target_cx,
            cx_min,
            cx_max,
            max_step_x,
            accel_x,
            cfg.smoothing_passes,
        )
        cy = _speed_limit(
            target_cy,
            cy_min,
            cy_max,
            max_step_y,
            accel_y,
            cfg.smoothing_passes,
        )

        x0 = np.clip(cx - half_w, 0.0, max(0.0, self.frame_w - self.crop_w))
        y0 = np.clip(cy - half_h, 0.0, max(0.0, self.frame_h - self.crop_h))
        w = np.full_like(cx, self.crop_w)
        h = np.full_like(cy, self.crop_h)
        zoom = np.full_like(cx, self.frame_h / self.crop_h)

        center_speed = np.concatenate([[0.0], np.hypot(np.diff(cx), np.diff(cy))])

        return {
            "x0": x0,
            "y0": y0,
            "w": w,
            "h": h,
            "cx": cx,
            "cy": cy,
            "z": zoom,
            "spd": center_speed,
        }


def plan_camera_from_ball(
    telemetry: Sequence[BallSample],
    src_width: float,
    src_height: float,
    *,
    out_aspect: float = 9.0 / 16.0,
    pad_frac: float = 0.2,
    zoom_min: float = 1.0,
    zoom_max: float = 2.5,
    smooth_strength: float = 0.15,
    inner_band_frac: float = 0.6,
    fps: float = 30.0,
) -> List[CameraKeyframe]:
    """Build camera keyframes from telemetry samples."""

    if not telemetry:
        raise ValueError("Telemetry list is empty")
    bx, by, start_frame = _samples_to_series(telemetry)
    fps = float(fps) if fps > 0 else 30.0
    pad_frac = max(0.0, float(pad_frac))
    inner_band_frac = float(np.clip(inner_band_frac, 0.2, 0.95))
    smooth_strength = float(np.clip(smooth_strength, 0.0, 0.99))
    zoom_min = max(0.5, float(zoom_min))
    zoom_max = max(zoom_min, float(zoom_max))

    base_size = min(float(src_width), float(src_height))
    margin_px = pad_frac * base_size
    headroom_frac = max(0.02, min(0.35, 0.5 - 0.5 * inner_band_frac))
    smooth_window = max(3, int(round((1.0 - smooth_strength) * 14.0)) | 1)
    max_step_x = max(8.0, float(src_width) * (0.010 + 0.006 * (1.0 - smooth_strength)))
    max_step_y = max(6.0, float(src_height) * (0.008 + 0.004 * (1.0 - smooth_strength)))

    cfg = PlannerConfig(
        frame_size=(float(src_width), float(src_height)),
        crop_aspect=float(out_aspect) if out_aspect > 0 else (9.0 / 16.0),
        fps=fps,
        margin_px=float(margin_px),
        headroom_frac=float(headroom_frac),
        lead_px=max(24.0, float(src_width) * 0.06),
        smooth_window=smooth_window,
        max_step_x=float(max_step_x),
        max_step_y=float(max_step_y),
        accel_limit_x=float(max_step_x * 0.35),
        accel_limit_y=float(max_step_y * 0.35),
        smoothing_passes=2,
        portrait_pad=float(margin_px * 0.6),
    )

    planner = OfflinePortraitPlanner(cfg)
    plan_dict = planner.plan(bx, by)
    keyframes = _plan_to_keyframes(plan_dict, start_frame=start_frame, fps=fps)
    for kf in keyframes:
        kf.zoom = max(zoom_min, min(zoom_max, kf.zoom))
    return keyframes


def save_plan(
    path: Path,
    keyframes: Sequence[CameraKeyframe],
    *,
    meta: Optional[Mapping[str, object]] = None,
) -> None:
    data = {
        "version": 1,
        "meta": dict(meta or {}),
        "keyframes": [kf.to_json() for kf in keyframes],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def load_plan(path: Path) -> Tuple[List[CameraKeyframe], Mapping[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    meta = data.get("meta") if isinstance(data, Mapping) else {}
    rows = data.get("keyframes") if isinstance(data, Mapping) else None
    keyframes: List[CameraKeyframe] = []
    if isinstance(rows, Sequence):
        for idx, rec in enumerate(rows):
            if not isinstance(rec, Mapping):
                continue
            width = float(rec.get("width", rec.get("w", 0.0)))
            height = float(rec.get("height", rec.get("h", 0.0)))
            keyframes.append(
                CameraKeyframe(
                    t=float(rec.get("t", idx / 30.0)),
                    frame=int(rec.get("frame", idx)),
                    cx=float(rec.get("cx", 0.0)),
                    cy=float(rec.get("cy", 0.0)),
                    zoom=float(rec.get("zoom", 1.0)),
                    width=width,
                    height=height,
                )
            )
    if not keyframes:
        raise ValueError(f"Plan {path} does not contain any keyframes")
    return keyframes, meta if isinstance(meta, Mapping) else {}


def keyframes_to_arrays(keyframes: Sequence[CameraKeyframe]) -> dict[str, np.ndarray]:
    n = len(keyframes)
    arr = {
        "x0": np.zeros(n, dtype=float),
        "y0": np.zeros(n, dtype=float),
        "w": np.zeros(n, dtype=float),
        "h": np.zeros(n, dtype=float),
        "cx": np.zeros(n, dtype=float),
        "cy": np.zeros(n, dtype=float),
        "z": np.zeros(n, dtype=float),
        "spd": np.zeros(n, dtype=float),
    }
    for idx, kf in enumerate(keyframes):
        width = max(1.0, float(kf.width))
        height = max(1.0, float(kf.height))
        arr["cx"][idx] = float(kf.cx)
        arr["cy"][idx] = float(kf.cy)
        arr["w"][idx] = width
        arr["h"][idx] = height
        arr["x0"][idx] = float(kf.cx) - 0.5 * width
        arr["y0"][idx] = float(kf.cy) - 0.5 * height
        arr["z"][idx] = float(kf.zoom)
    if n >= 2:
        diffs = np.hypot(np.diff(arr["cx"]), np.diff(arr["cy"]))
        arr["spd"][1:] = diffs
    return arr


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline portrait planner from telemetry")
    parser.add_argument("ball_path", help="Telemetry JSONL/CSV")
    parser.add_argument("--width", type=float, required=True, help="Source width in pixels")
    parser.add_argument("--height", type=float, required=True, help="Source height in pixels")
    parser.add_argument("--fps", type=float, default=30.0, help="Source FPS")
    parser.add_argument("--aspect", type=float, default=9.0 / 16.0, help="Portrait crop aspect (W/H)")
    parser.add_argument("--pad-frac", type=float, default=0.2, help="Padding fraction around the ball")
    parser.add_argument("--inner-band", type=float, default=0.6, help="Inner keep-in-view band (0-1)")
    parser.add_argument("--zoom-max", type=float, default=2.5, help="Maximum zoom factor")
    parser.add_argument("--smooth", type=float, default=0.15, help="0..1 smoothing strength")
    parser.add_argument("--out", type=Path, help="Optional .plan.json output path")
    return parser


def main(cli_args: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(cli_args)
    samples = load_ball_telemetry(Path(args.ball_path))
    keyframes = plan_camera_from_ball(
        samples,
        args.width,
        args.height,
        out_aspect=args.aspect,
        pad_frac=args.pad_frac,
        zoom_max=args.zoom_max,
        smooth_strength=args.smooth,
        inner_band_frac=args.inner_band,
        fps=args.fps,
    )
    meta = {
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
        "aspect": args.aspect,
    }
    if args.out:
        save_plan(Path(args.out), keyframes, meta=meta)
    else:
        print(json.dumps({"meta": meta, "keyframes": [kf.to_json() for kf in keyframes]}, indent=2))


if __name__ == "__main__":
    main()


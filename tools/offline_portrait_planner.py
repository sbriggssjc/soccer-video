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

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from tools.ball_telemetry import BallSample, load_ball_telemetry


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


def _speed_limit(series: np.ndarray, mins: np.ndarray, maxs: np.ndarray, max_step: float, accel_limit: float, passes: int) -> np.ndarray:
    out = series.copy()
    n = len(out)
    for _ in range(passes):
        for idx in range(1, n):
            delta = out[idx] - out[idx - 1]
            if abs(delta) > max_step:
                out[idx] = out[idx - 1] + math.copysign(max_step, delta)
            out[idx] = np.clip(out[idx], mins[idx], maxs[idx])
        for idx in range(n - 2, -1, -1):
            delta = out[idx] - out[idx + 1]
            if abs(delta) > max_step:
                out[idx] = out[idx + 1] + math.copysign(max_step, delta)
            out[idx] = np.clip(out[idx], mins[idx], maxs[idx])
        if accel_limit > 0 and n >= 3:
            for idx in range(1, n - 1):
                jerk = out[idx + 1] - 2 * out[idx] + out[idx - 1]
                if abs(jerk) > accel_limit:
                    adjust = 0.5 * (abs(jerk) - accel_limit)
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
        lead_scale = self.cfg.lead_px
        lead_x = lead_scale * (vx / norm)
        lead_y = lead_scale * (vy / norm)
        headroom = self.cfg.headroom_frac * self.crop_h

        ideal_cx = bx_clean + lead_x
        ideal_cy = by_clean + headroom + lead_y * 0.1

        half_w = self.crop_w / 2.0
        half_h = self.crop_h / 2.0
        margin_x = min(half_w - 4.0, self.cfg.margin_px)
        margin_y = min(half_h - 4.0, self.cfg.margin_px * 1.35)
        margin_x = max(0.0, margin_x)
        margin_y = max(0.0, margin_y)

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
            self.cfg.max_step_x,
            self.cfg.accel_limit_x,
            self.cfg.smoothing_passes,
        )
        cy = _speed_limit(
            target_cy,
            cy_min,
            cy_max,
            self.cfg.max_step_y,
            self.cfg.accel_limit_y,
            self.cfg.smoothing_passes,
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

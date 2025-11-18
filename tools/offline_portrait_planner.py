"""Offline portrait camera planner.

This module implements a deterministic offline planner for portrait crops that
keeps the ball inside the frame while favouring smooth, cinematic motion.  It
operates on an already-tracked ball path (telemetry) and produces per-frame
camera windows that can be consumed by :mod:`tools.render_follow_unified` or by
other automation scripts.

The implementation intentionally mirrors the design brief from the repo owner:

* Clean up the ball telemetry (gap filling, smoothing).
* Compute an ideal framing target based on the ball velocity and configurable
  headroom/lead offsets.
* Intersect the target with the feasible camera bands that keep both the crop
  and the ball inside the source frame.
* Run a multi-pass forward/backward speed limiter to keep the motion
  cinematic (bounded velocity and jerk).

The module exposes a small ``OfflinePortraitPlanner`` class so that both the
CLI helper and the rendering pipeline can share the behaviour without re-
implementing the heuristics in multiple locations.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


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


def _load_ball_series(path: Path) -> Tuple[List[float], List[float]]:
    bx: List[float] = []
    by: List[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                bx.append(float("nan"))
                by.append(float("nan"))
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bx.append(float("nan"))
                by.append(float("nan"))
                continue
            if isinstance(rec, dict):
                for key_x, key_y in (("bx", "by"), ("bx_stab", "by_stab"), ("ball", "ball")):
                    val_x = rec.get(key_x)
                    val_y = rec.get(key_y)
                    if isinstance(val_x, Sequence) and not isinstance(val_x, (bytes, str)):
                        val_x = val_x[0]
                    if isinstance(val_y, Sequence) and not isinstance(val_y, (bytes, str)):
                        val_y = val_y[1 if len(val_y) > 1 else 0]
                    try:
                        bx_val = float(val_x)
                        by_val = float(val_y)
                    except (TypeError, ValueError):
                        continue
                    bx.append(bx_val)
                    by.append(by_val)
                    break
                else:
                    bx.append(float("nan"))
                    by.append(float("nan"))
            elif isinstance(rec, Sequence) and len(rec) >= 2:
                try:
                    bx.append(float(rec[0]))
                    by.append(float(rec[1]))
                except (TypeError, ValueError):
                    bx.append(float("nan"))
                    by.append(float("nan"))
            else:
                bx.append(float("nan"))
                by.append(float("nan"))
    if not bx:
        raise ValueError(f"No ball telemetry found in {path}")
    return bx, by


def plan_from_file(ball_path: Path, config: PlannerConfig) -> dict[str, np.ndarray]:
    bx, by = _load_ball_series(ball_path)
    planner = OfflinePortraitPlanner(config)
    return planner.plan(bx, by)


def _write_plan(out_path: Path, plan: dict[str, np.ndarray]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for idx in range(len(plan["x0"])):
            row = {key: float(val[idx]) for key, val in plan.items()}
            handle.write(json.dumps(row) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline portrait planner from ball telemetry")
    parser.add_argument("ball_path", help="JSONL with bx/by entries per frame")
    parser.add_argument("--width", type=float, required=True)
    parser.add_argument("--height", type=float, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--aspect", type=float, default=9.0 / 16.0, help="Portrait crop aspect (W/H)")
    parser.add_argument("--margin", type=float, default=90.0, help="Ball margin inside crop (px)")
    parser.add_argument("--headroom", type=float, default=0.08, help="Headroom fraction of crop height")
    parser.add_argument("--lead", type=float, default=120.0, help="Look-ahead distance in px")
    parser.add_argument("--out", type=Path, help="Optional JSONL output path")
    return parser


def main(cli_args: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(cli_args)
    config = PlannerConfig(
        frame_size=(args.width, args.height),
        crop_aspect=args.aspect,
        fps=args.fps,
        margin_px=args.margin,
        headroom_frac=args.headroom,
        lead_px=args.lead,
    )
    plan = plan_from_file(Path(args.ball_path), config)
    if args.out:
        _write_plan(Path(args.out), plan)
    else:
        print(json.dumps({key: val.tolist() for key, val in plan.items()}))


if __name__ == "__main__":
    main()

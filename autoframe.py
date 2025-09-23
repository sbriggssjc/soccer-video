"""Motion-aware crop center and zoom estimator for soccer reels."""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import yaml


@dataclass
class FrameState:
    """Per-frame state tracked while iterating through the clip."""

    center: np.ndarray
    zoom: float


def deep_update(base: Dict, override: Dict) -> Dict:
    """Recursively merge ``override`` into ``base`` and return a copy."""

    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path, profile: str, roi: str) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    defaults = data.get("defaults", {})
    profiles = data.get("profiles", {})
    rois = data.get("roi", {})

    config = dict(defaults)
    if profile in profiles:
        config = deep_update(config, profiles[profile])
    if roi in rois:
        config = deep_update(config, rois[roi])

    # fallbacks when config omits values
    config.setdefault("aspect_ratio", 0.5625)
    config.setdefault("ema_alpha", 0.25)
    config.setdefault("deadband_pct", 0.012)
    config.setdefault("max_center_step_pct", 0.045)
    config.setdefault("fallback_bias_y", 0.56)

    flow_cfg = config.setdefault("flow", {})
    flow_cfg.setdefault("pyr_scale", 0.5)
    flow_cfg.setdefault("levels", 3)
    flow_cfg.setdefault("winsize", 21)
    flow_cfg.setdefault("iterations", 3)
    flow_cfg.setdefault("poly_n", 5)
    flow_cfg.setdefault("poly_sigma", 1.2)
    flow_cfg.setdefault("flags", 0)
    flow_cfg.setdefault("blur_sigma", 3.5)
    flow_cfg.setdefault("saliency_percentile", 92.0)
    flow_cfg.setdefault("min_salient_pixels_pct", 0.004)
    flow_cfg.setdefault("topk_mean_floor", 0.35)
    flow_cfg.setdefault("fallback_mean_thresh", 0.12)

    zoom_cfg = config.setdefault("zoom", {})
    zoom_cfg.setdefault("padding_pct", 0.08)
    zoom_cfg.setdefault("dz_max", 0.03)
    zoom_cfg.setdefault("min", 1.05)
    zoom_cfg.setdefault("max", 2.4)
    zoom_cfg.setdefault("base_roi_pct", 0.34)
    zoom_cfg.setdefault("min_roi_pct", 0.28)
    zoom_cfg.setdefault("max_roi_pct", 0.55)
    zoom_cfg.setdefault("std_scale", 2.8)
    zoom_cfg.setdefault("mean_scale", 2.2)

    return config


@dataclass
class FlowMeasurement:
    center: Optional[np.ndarray]
    spread_y: Optional[float]
    spread_x: Optional[float]
    avg_mag: float
    top_mean: float
    mask_pixels: int


class AutoFramer:
    """Compute motion-driven crop center/zoom tracks."""

    def __init__(self, cfg: Dict) -> None:
        self.cfg = cfg
        self.flow_cfg = cfg["flow"]
        self.zoom_cfg = cfg["zoom"]
        self.aspect_ratio = float(cfg.get("aspect_ratio", 0.5625))
        self.ema_alpha = float(cfg.get("ema_alpha", 0.25))
        self.deadband_pct = float(cfg.get("deadband_pct", 0.012))
        self.max_center_step_pct = float(cfg.get("max_center_step_pct", 0.045))
        self.fallback_bias_y = float(cfg.get("fallback_bias_y", 0.56))

    def run(self, video_path: Path) -> Tuple[List[FrameState], float, Tuple[int, int]]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise SystemExit(f"Failed to open video: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            fps = 30.0

        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise SystemExit("Failed to read first frame")

        height, width = frame.shape[:2]
        grid_x, grid_y = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
        )
        prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        fallback_center = np.array(
            [width / 2.0, height * self.fallback_bias_y], dtype=np.float64
        )
        deadband_px = max(1.0, self.deadband_pct * max(width, height))
        max_step_px = max(1.0, self.max_center_step_pct * max(width, height))

        base_roi_height = height * float(self.zoom_cfg.get("base_roi_pct", 0.34))
        min_roi_height = height * float(self.zoom_cfg.get("min_roi_pct", 0.28))
        max_roi_height = height * float(self.zoom_cfg.get("max_roi_pct", 0.55))
        padding_pct = float(self.zoom_cfg.get("padding_pct", 0.08))
        min_zoom = float(self.zoom_cfg.get("min", 1.05))
        max_zoom = float(self.zoom_cfg.get("max", 2.4))
        dz_max = float(self.zoom_cfg.get("dz_max", 0.03))

        min_zoom_width = (height * self.aspect_ratio) / max(width, 1)
        if min_zoom_width > min_zoom:
            min_zoom = min_zoom_width

        current_center = fallback_center.copy()
        current_zoom = self._height_to_zoom(base_roi_height * (1 + padding_pct), height)
        current_zoom = float(np.clip(current_zoom, min_zoom, max_zoom))

        states: List[FrameState] = [FrameState(center=current_center.copy(), zoom=current_zoom)]

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                gray,
                None,
                self.flow_cfg["pyr_scale"],
                int(self.flow_cfg["levels"]),
                int(self.flow_cfg["winsize"]),
                int(self.flow_cfg["iterations"]),
                int(self.flow_cfg["poly_n"]),
                float(self.flow_cfg["poly_sigma"]),
                int(self.flow_cfg.get("flags", 0)),
            )
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            blur_sigma = float(self.flow_cfg.get("blur_sigma", 3.5))
            if blur_sigma > 0:
                magnitude = cv2.GaussianBlur(magnitude, (0, 0), blur_sigma)

            measurement = self._measure_flow(magnitude, grid_x, grid_y)

            current_center = self._update_center(
                current_center,
                fallback_center,
                measurement.center,
                deadband_px,
                max_step_px,
                width,
                height,
            )

            current_zoom = self._update_zoom(
                current_zoom,
                measurement,
                base_roi_height,
                min_roi_height,
                max_roi_height,
                padding_pct,
                min_zoom,
                max_zoom,
                dz_max,
                height,
            )

            states.append(FrameState(center=current_center.copy(), zoom=current_zoom))
            prev_gray = gray

        cap.release()
        return states, fps, (width, height)

    def _measure_flow(
        self, magnitude: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray
    ) -> FlowMeasurement:
        saliency = float(self.flow_cfg.get("saliency_percentile", 92.0))
        threshold = float(np.percentile(magnitude, saliency)) if magnitude.size else 0.0
        if not np.isfinite(threshold) or threshold <= 0.0:
            mask = magnitude > 0.0
        else:
            mask = magnitude >= threshold

        weights = np.where(mask, magnitude, 0.0)
        total_weight = float(weights.sum())
        mask_pixels = int(mask.sum())
        top_mean = float(weights[mask].mean()) if mask_pixels > 0 else 0.0
        avg_mag = float(magnitude.mean()) if magnitude.size else 0.0

        min_pixels_pct = float(self.flow_cfg.get("min_salient_pixels_pct", 0.004))
        min_pixels = int(mask.size * min_pixels_pct)
        mean_floor = float(self.flow_cfg.get("topk_mean_floor", 0.35))
        fallback_mean = float(self.flow_cfg.get("fallback_mean_thresh", 0.12))

        confident = (
            total_weight > 1e-5
            and mask_pixels >= max(10, min_pixels)
            and top_mean >= mean_floor
            and avg_mag >= fallback_mean
        )

        if not confident:
            return FlowMeasurement(
                center=None,
                spread_y=None,
                spread_x=None,
                avg_mag=avg_mag,
                top_mean=top_mean,
                mask_pixels=mask_pixels,
            )

        cx = float((weights * grid_x).sum() / total_weight)
        cy = float((weights * grid_y).sum() / total_weight)
        center = np.array([cx, cy], dtype=np.float64)

        spread_y = math.sqrt(
            max(float((weights * ((grid_y - cy) ** 2)).sum() / total_weight), 0.0)
        )
        spread_x = math.sqrt(
            max(float((weights * ((grid_x - cx) ** 2)).sum() / total_weight), 0.0)
        )

        return FlowMeasurement(
            center=center,
            spread_y=spread_y,
            spread_x=spread_x,
            avg_mag=avg_mag,
            top_mean=top_mean,
            mask_pixels=mask_pixels,
        )

    def _update_center(
        self,
        prev_center: np.ndarray,
        fallback_center: np.ndarray,
        measurement: Optional[np.ndarray],
        deadband_px: float,
        max_step_px: float,
        width: int,
        height: int,
    ) -> np.ndarray:
        target = measurement if measurement is not None else fallback_center

        delta = target - prev_center
        if float(np.hypot(delta[0], delta[1])) <= deadband_px:
            target = prev_center

        blended = prev_center * (1.0 - self.ema_alpha) + target * self.ema_alpha
        step = blended - prev_center
        dist = float(np.hypot(step[0], step[1]))
        if dist > max_step_px > 0.0:
            scale = max_step_px / dist
            blended = prev_center + step * scale

        blended[0] = float(np.clip(blended[0], 0.0, max(0.0, width - 1.0)))
        blended[1] = float(np.clip(blended[1], 0.0, max(0.0, height - 1.0)))
        return blended

    def _height_to_zoom(self, roi_height: float, frame_height: int) -> float:
        roi_height = max(1e-3, min(float(roi_height), float(frame_height)))
        return float(frame_height) / roi_height

    def _update_zoom(
        self,
        prev_zoom: float,
        measurement: FlowMeasurement,
        base_roi_height: float,
        min_roi_height: float,
        max_roi_height: float,
        padding_pct: float,
        min_zoom: float,
        max_zoom: float,
        dz_max: float,
        frame_height: int,
    ) -> float:
        roi_height = base_roi_height
        if measurement.spread_y is not None and measurement.spread_x is not None:
            spread_component = measurement.spread_y * float(self.zoom_cfg.get("std_scale", 2.8))
            mean_component = measurement.spread_y * float(self.zoom_cfg.get("mean_scale", 2.2))
            roi_height = max(spread_component * 2.0, mean_component * 2.0, base_roi_height)

        roi_height = float(np.clip(roi_height, min_roi_height, max_roi_height))
        roi_height *= 1.0 + padding_pct

        target_zoom = self._height_to_zoom(roi_height, frame_height)
        target_zoom = float(np.clip(target_zoom, min_zoom, max_zoom))

        delta = target_zoom - prev_zoom
        if abs(delta) > dz_max > 0:
            delta = math.copysign(dz_max, delta)
        return prev_zoom + delta


def write_csv(out_path: Path, states: Iterable[FrameState]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "cx", "cy", "z"])
        for idx, state in enumerate(states):
            writer.writerow([idx, f"{state.center[0]:.4f}", f"{state.center[1]:.4f}", f"{state.zoom:.5f}"])


def compute_box(
    center: np.ndarray,
    zoom: float,
    aspect_ratio: float,
    frame_size: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    width, height = frame_size
    crop_h = height / max(zoom, 1e-6)
    crop_w = (height * aspect_ratio) / max(zoom, 1e-6)

    crop_w = min(crop_w, width)
    crop_h = min(crop_h, height)

    x0 = float(center[0] - crop_w / 2.0)
    y0 = float(center[1] - crop_h / 2.0)
    x0 = np.clip(x0, 0.0, max(0.0, width - crop_w))
    y0 = np.clip(y0, 0.0, max(0.0, height - crop_h))

    x1 = x0 + crop_w
    y1 = y0 + crop_h
    return int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))


def render_preview(
    video_path: Path,
    preview_path: Path,
    states: List[FrameState],
    fps: float,
    aspect_ratio: float,
    frame_size: Tuple[int, int],
) -> None:
    if preview_path is None:
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video for preview: {video_path}")

    width, height = frame_size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(preview_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"Failed to open preview writer: {preview_path}")

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or idx >= len(states):
            break

        state = states[idx]
        x0, y0, x1, y1 = compute_box(state.center, state.zoom, aspect_ratio, (width, height))
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 3)
        cv2.circle(frame, (int(round(state.center[0])), int(round(state.center[1]))), 8, (0, 255, 0), -1)
        cv2.putText(
            frame,
            f"z={state.zoom:.2f}",
            (int(x0) + 10, int(y0) + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
        idx += 1

    cap.release()
    writer.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate crop center/zoom tracks from motion")
    parser.add_argument("--in", dest="input_path", type=Path, required=True, help="Input video path")
    parser.add_argument("--csv", dest="csv_path", type=Path, required=True, help="Output CSV path")
    parser.add_argument("--preview", dest="preview_path", type=Path, help="Optional debug preview MP4")
    parser.add_argument(
        "--roi",
        choices=["generic", "goal"],
        default="generic",
        help="ROI tuning preset",
    )
    parser.add_argument(
        "--profile",
        choices=["portrait", "landscape"],
        default="portrait",
        help="Output profile controls aspect ratio",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/zoom.yaml"),
        help="YAML tuning file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.profile, args.roi)
    framer = AutoFramer(cfg)
    states, fps, frame_size = framer.run(args.input_path)
    write_csv(args.csv_path, states)
    if args.preview_path:
        render_preview(args.input_path, args.preview_path, states, fps, framer.aspect_ratio, frame_size)
    print(f"Wrote {len(states)} motion samples to {args.csv_path}")
    if args.preview_path:
        print(f"Preview: {args.preview_path}")


if __name__ == "__main__":
    main()

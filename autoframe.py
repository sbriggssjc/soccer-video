"""Motion-aware crop center and zoom estimator for soccer reels."""
from __future__ import annotations

import argparse
import csv
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class FrameResult:
    """Final crop command for a single frame."""

    frame: int
    center: Tuple[float, float]
    zoom: float
    width: float
    height: float
    x: float
    y: float
    conf: float
    crowding: float

    def to_row(self) -> List[str]:
        return [
            str(self.frame),
            f"{self.center[0]:.4f}",
            f"{self.center[1]:.4f}",
            f"{self.zoom:.5f}",
            f"{self.width:.4f}",
            f"{self.height:.4f}",
            f"{self.x:.4f}",
            f"{self.y:.4f}",
            f"{self.conf:.4f}",
            f"{self.crowding:.4f}",
        ]


class MotionEstimator:
    """Estimate a motion saliency mask and centroid from frame-to-frame changes."""

    def __init__(self, width: int, height: int, flow_thresh: float) -> None:
        self.width = int(width)
        self.height = int(height)
        self.flow_thresh = float(flow_thresh)
        self.running_p90: float = 0.0
        self.initialized = False
        self.prev_conf: float = 0.0
        self.prev_centroid: Optional[np.ndarray] = None
        self.grid_x, self.grid_y = np.meshgrid(
            np.arange(self.width, dtype=np.float32),
            np.arange(self.height, dtype=np.float32),
        )
        self.diagonal = math.hypot(self.width, self.height)

    def _update_running_p90(self, magnitude: np.ndarray) -> float:
        if magnitude.size == 0:
            current = 0.0
        else:
            current = float(np.percentile(magnitude, 90))
            if not np.isfinite(current):
                current = 0.0
        if not self.initialized:
            self.running_p90 = current
            self.initialized = True
        else:
            decay = 0.85
            self.running_p90 = decay * self.running_p90 + (1.0 - decay) * current
        return max(self.running_p90, 1e-6)

    def _fallback_diff(self, prev_gray: np.ndarray, gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        diff = cv2.absdiff(gray, prev_gray)
        diff = cv2.GaussianBlur(diff, (0, 0), 1.2)
        scale = float(np.percentile(diff, 90)) if diff.size else 0.0
        scale = max(scale, 1e-6)
        norm = np.clip(diff.astype(np.float32) / scale, 0.0, 1.0)
        mask = norm >= self.flow_thresh
        return norm, mask

    def _select_component(
        self,
        labels: np.ndarray,
        stats: np.ndarray,
        centroids: np.ndarray,
        prev_center: Optional[np.ndarray],
    ) -> int:
        label = 0
        if prev_center is not None:
            px = int(round(prev_center[0]))
            py = int(round(prev_center[1]))
            if 0 <= px < self.width and 0 <= py < self.height:
                label_at_prev = int(labels[py, px])
                if label_at_prev > 0:
                    return label_at_prev
        best_score = -1.0
        for idx in range(1, centroids.shape[0]):
            area = stats[idx, cv2.CC_STAT_AREA]
            if area <= 0:
                continue
            cx, cy = centroids[idx]
            if not np.isfinite(cx) or not np.isfinite(cy):
                continue
            if prev_center is not None:
                dist = math.hypot(cx - prev_center[0], cy - prev_center[1])
            else:
                dist = math.hypot(cx - self.width / 2.0, cy - self.height / 2.0)
            score = float(area) / (1.0 + dist * 0.02)
            if score > best_score:
                best_score = score
                label = idx
        return label

    def _measure_component(
        self,
        weights: np.ndarray,
        labels: np.ndarray,
        stats: np.ndarray,
        centroids: np.ndarray,
        selected: int,
        prev_center: Optional[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], float, float]:
        component_mask = labels == selected
        area = int(stats[selected, cv2.CC_STAT_AREA])
        if area <= 0:
            return None, 0.0, 0.0
        comp_weights = weights * component_mask
        weight_sum = float(comp_weights.sum())
        if weight_sum <= 1e-6:
            comp_weights = component_mask.astype(np.float32)
            weight_sum = float(comp_weights.sum())
        if weight_sum <= 1e-6:
            return None, 0.0, 0.0
        cx = float((comp_weights * self.grid_x).sum() / weight_sum)
        cy = float((comp_weights * self.grid_y).sum() / weight_sum)
        centroid = np.array([cx, cy], dtype=np.float64)

        area_norm = min(1.0, area / float(self.width * self.height))
        bbox_w = stats[selected, cv2.CC_STAT_WIDTH]
        bbox_h = stats[selected, cv2.CC_STAT_HEIGHT]
        spread_area = math.sqrt(area) / math.sqrt(self.width * self.height)
        spread_bbox = math.hypot(bbox_w, bbox_h) / max(self.diagonal, 1e-6)
        spread = float(np.clip(0.5 * (spread_area + spread_bbox), 0.0, 1.0))

        mean_strength = float(np.clip(weight_sum / max(area, 1), 0.0, 1.0))
        if prev_center is not None:
            dist = math.hypot(cx - prev_center[0], cy - prev_center[1])
        elif self.prev_centroid is not None:
            dist = math.hypot(cx - self.prev_centroid[0], cy - self.prev_centroid[1])
        else:
            dist = 0.0
        temporal = math.exp(-((dist / (0.15 * self.diagonal + 1e-6)) ** 2))
        conf_raw = np.clip(area_norm * 3.5, 0.0, 1.0) * (0.5 + 0.5 * mean_strength)
        conf = float(np.clip(0.6 * conf_raw * temporal + 0.4 * self.prev_conf, 0.0, 1.0))
        self.prev_conf = conf
        self.prev_centroid = centroid
        return centroid, spread, conf

    def compute(
        self,
        prev_gray: np.ndarray,
        gray: np.ndarray,
        prev_center: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray], float, float]:
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            gray,
            None,
            0.5,
            3,
            21,
            3,
            5,
            1.2,
            0,
        )
        mag = cv2.magnitude(flow[..., 0], flow[..., 1])
        mag = cv2.GaussianBlur(mag, (0, 0), 1.5)
        scale = self._update_running_p90(mag)
        norm = np.clip(mag.astype(np.float32) / scale, 0.0, 1.0)
        mask = norm >= self.flow_thresh

        mask_pixels = int(mask.sum())
        if mask_pixels < 25:
            diff_norm, diff_mask = self._fallback_diff(prev_gray, gray)
            if diff_mask.sum() > mask_pixels:
                norm = diff_norm
                mask = diff_mask
                mask_pixels = int(mask.sum())

        centroid: Optional[np.ndarray] = None
        spread = 0.0
        conf = 0.0
        if mask_pixels > 0:
            _, labels, stats, centroids = cv2.connectedComponentsWithStats(
                mask.astype(np.uint8), 8, cv2.CV_32S
            )
            selected = self._select_component(labels, stats, centroids, prev_center)
            centroid, spread, conf = self._measure_component(
                norm, labels, stats, centroids, selected, prev_center
            )
        else:
            self.prev_conf *= 0.7

        return mask.astype(np.float32), centroid, spread, conf


def apply_deadband(
    predicted: np.ndarray, prev_cmd: np.ndarray, deadband: float
) -> np.ndarray:
    result = predicted.copy()
    for axis in range(2):
        delta = predicted[axis] - prev_cmd[axis]
        if abs(delta) < deadband:
            result[axis] = prev_cmd[axis]
    return result


def apply_slew(
    prev_cmd: np.ndarray, target: np.ndarray, slew_xy: Sequence[float]
) -> np.ndarray:
    result = prev_cmd.copy()
    for axis in range(2):
        delta = float(target[axis] - prev_cmd[axis])
        max_delta = float(slew_xy[axis])
        delta = max(-max_delta, min(max_delta, delta))
        result[axis] = prev_cmd[axis] + delta
    return result


def compute_crop_dimensions(
    center: np.ndarray,
    zoom: float,
    profile: str,
    padx: float,
    pady: float,
    frame_size: Tuple[int, int],
) -> Tuple[float, float, float, float, np.ndarray]:
    width, height = frame_size
    if profile == "portrait":
        base_h = height / max(zoom, 1e-6)
        base_w = base_h * (9.0 / 16.0)
    else:
        base_w = width / max(zoom, 1e-6)
        base_h = base_w * (9.0 / 16.0)

    w = base_w * (1.0 + padx)
    h = base_h * (1.0 + pady)

    scale = min(1.0, width / max(w, 1e-6), height / max(h, 1e-6))
    w *= scale
    h *= scale

    half_w = w / 2.0
    half_h = h / 2.0
    x = float(np.clip(center[0] - half_w, 0.0, max(0.0, width - w)))
    y = float(np.clip(center[1] - half_h, 0.0, max(0.0, height - h)))

    adjusted_center = np.array([x + half_w, y + half_h], dtype=np.float64)
    return w, h, x, y, adjusted_center


def draw_preview(
    frame: np.ndarray,
    crop: Tuple[float, float, float, float],
    center: Tuple[float, float],
    zoom: float,
    conf: float,
) -> None:
    x, y, w, h = crop
    overlay = frame.copy()
    top_left = (int(round(x)), int(round(y)))
    bottom_right = (int(round(x + w)), int(round(y + h)))
    cv2.rectangle(overlay, top_left, bottom_right, (0, 0, 255), 2)
    cv2.circle(overlay, (int(round(center[0])), int(round(center[1]))), 6, (0, 255, 0), -1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
    cv2.putText(
        frame,
        f"z={zoom:.2f} conf={conf:.2f}",
        (top_left[0] + 8, max(30, top_left[1] + 24)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def write_csv(
    csv_path: Path,
    results: Iterable[FrameResult],
    fps: float,
    frame_size: Tuple[int, int],
    zoom_min: float,
    zoom_max: float,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = frame_size
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(f"# fps={fps:.4f}\n")
        handle.write(f"# width={width},height={height}\n")
        handle.write(f"# zoom_min={zoom_min:.5f},zoom_max={zoom_max:.5f}\n")
        writer = csv.writer(handle)
        writer.writerow(["frame", "cx", "cy", "z", "w", "h", "x", "y", "conf", "crowding"])
        for result in results:
            writer.writerow(result.to_row())


def render_preview(
    video_path: Path,
    preview_path: Path,
    results: Sequence[FrameResult],
    fps: float,
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
    for idx, result in enumerate(results):
        ok, frame = cap.read()
        if not ok:
            break
        crop = (result.x, result.y, result.width, result.height)
        draw_preview(frame, crop, result.center, result.zoom, result.conf)
        writer.write(frame)
    cap.release()
    writer.release()


def parse_vector(arg: str, count: int) -> Tuple[float, ...]:
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    if len(parts) != count:
        raise argparse.ArgumentTypeError(f"expected {count} comma-separated values")
    return tuple(float(p) for p in parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate crop center/zoom tracks from motion"
    )
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
        help="Reserved for compatibility; current pipeline is driven by CLI flags",
    )
    parser.add_argument("--lead", type=int, default=6, help="Frames to lead/predict center")
    parser.add_argument("--deadband", type=float, default=10.0, help="Ignore small center deltas (px)")
    parser.add_argument(
        "--slew_xy",
        type=lambda s: parse_vector(s, 2),
        default=(40.0, 40.0),
        help="Max per-frame change for cx,cy (px,px)",
    )
    parser.add_argument("--slew_z", type=float, default=0.06, help="Max zoom change per frame")
    parser.add_argument("--padx", type=float, default=0.20, help="Horizontal padding around action")
    parser.add_argument("--pady", type=float, default=0.16, help="Vertical padding around action")
    parser.add_argument("--zoom_min", type=float, default=1.08, help="Minimum zoom (crop scale denominator)")
    parser.add_argument("--zoom_max", type=float, default=2.40, help="Maximum zoom (tighter crop)")
    parser.add_argument("--zoom_k", type=float, default=0.85, help="Crowding→zoom responsiveness gain")
    parser.add_argument(
        "--zoom_asym",
        type=lambda s: parse_vector(s, 2),
        default=(0.75, 0.35),
        help="Asymmetric damping for zoom in/out",
    )
    parser.add_argument("--smooth_ema", type=float, default=0.35, help="EMA smoothing alpha for center")
    parser.add_argument(
        "--smooth_win",
        type=int,
        default=0,
        help="Optional boxcar window (odd) applied before EMA; 0 disables",
    )
    parser.add_argument(
        "--hold_frames",
        type=int,
        default=8,
        help="Hold last command this many frames when confidence drops",
    )
    parser.add_argument("--conf_floor", type=float, default=0.15, help="Confidence floor")
    parser.add_argument("--flow_thresh", type=float, default=0.18, help="Motion mask threshold after normalization")
    return parser.parse_args()


def run_autoframe(
    args: argparse.Namespace,
) -> Tuple[List[FrameResult], float, Tuple[int, int], float, float]:
    cap = cv2.VideoCapture(str(args.input_path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {args.input_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 30.0

    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise SystemExit("Failed to read first frame")

    height, width = frame.shape[:2]
    estimator = MotionEstimator(width, height, args.flow_thresh)
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    zoom_min = float(args.zoom_min)
    zoom_max = float(args.zoom_max)
    if zoom_min > zoom_max:
        zoom_min, zoom_max = zoom_max, zoom_min

    lead_frames = float(args.lead)
    deadband = float(args.deadband)
    slew_xy = tuple(float(v) for v in args.slew_xy)
    slew_z = float(args.slew_z)
    zoom_gain = float(args.zoom_k)
    zoom_asym_in = float(args.zoom_asym[0])
    zoom_asym_out = float(args.zoom_asym[1])
    alpha = float(args.smooth_ema)
    hold_frames = max(0, int(args.hold_frames))
    conf_floor = float(args.conf_floor)
    padx = max(0.0, float(args.padx))
    pady = max(0.0, float(args.pady))
    smooth_win = int(args.smooth_win)
    if smooth_win < 0:
        smooth_win = 0
    if smooth_win and smooth_win % 2 == 0:
        smooth_win += 1

    results: List[FrameResult] = []
    commanded_center = np.array([width / 2.0, height / 2.0], dtype=np.float64)
    ema_center = commanded_center.copy()
    raw_history: Deque[np.ndarray] = deque(maxlen=smooth_win if smooth_win > 0 else 1)
    raw_history.append(commanded_center.copy())
    z_cmd = zoom_min
    hold_counter = 0

    w0, h0, x0, y0, adjusted_center = compute_crop_dimensions(
        commanded_center, z_cmd, args.profile, padx, pady, (width, height)
    )
    commanded_center = adjusted_center
    ema_center = adjusted_center.copy()
    results.append(
        FrameResult(
            frame=0,
            center=(commanded_center[0], commanded_center[1]),
            zoom=z_cmd,
            width=w0,
            height=h0,
            x=x0,
            y=y0,
            conf=0.0,
            crowding=0.0,
        )
    )

    frame_idx = 1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        mask, centroid, spread, conf = estimator.compute(prev_gray, gray, commanded_center)
        prev_gray = gray

        raw_center = commanded_center.copy()
        if centroid is not None:
            raw_center = centroid
        raw_history.append(raw_center)
        if smooth_win > 1:
            averaged = np.mean(raw_history, axis=0)
        else:
            averaged = raw_history[-1]

        ema_prev = ema_center.copy()
        ema_center = ema_center * (1.0 - alpha) + averaged * alpha
        velocity = ema_center - ema_prev
        predicted = ema_center + velocity * lead_frames

        predicted = np.clip(predicted, [0.0, 0.0], [width - 1.0, height - 1.0])
        predicted = apply_deadband(predicted, commanded_center, deadband)
        target_center = apply_slew(commanded_center, predicted, slew_xy)

        crowding = float(np.clip(spread, 0.0, 1.0))
        z_target = zoom_min + (zoom_max - zoom_min) * (crowding ** zoom_gain)
        dz = z_target - z_cmd
        if dz >= 0:
            dz = min(dz, slew_z * zoom_asym_in)
        else:
            dz = max(dz, -slew_z * zoom_asym_out)
        z_next = float(np.clip(z_cmd + dz, zoom_min, zoom_max))

        if conf < conf_floor:
            hold_counter = hold_frames
        if hold_counter > 0:
            target_center = commanded_center
            z_next = z_cmd
            hold_counter = max(hold_counter - 1, 0)

        w, h, x, y, adjusted_center = compute_crop_dimensions(
            target_center, z_next, args.profile, padx, pady, (width, height)
        )
        commanded_center = adjusted_center
        ema_center = ema_center * 0.5 + commanded_center * 0.5
        z_cmd = z_next

        results.append(
            FrameResult(
                frame=frame_idx,
                center=(commanded_center[0], commanded_center[1]),
                zoom=z_cmd,
                width=w,
                height=h,
                x=x,
                y=y,
                conf=float(conf),
                crowding=crowding,
            )
        )
        frame_idx += 1

    cap.release()
    return results, fps, (width, height), zoom_min, zoom_max


def main() -> None:
    args = parse_args()
    results, fps, frame_size, zoom_min, zoom_max = run_autoframe(args)
    write_csv(args.csv_path, results, fps, frame_size, zoom_min, zoom_max)
    if args.preview_path:
        render_preview(args.input_path, args.preview_path, results, fps, frame_size)
    print(f"Wrote {len(results)} motion samples to {args.csv_path}")
    if args.preview_path:
        print(f"Preview: {args.preview_path}")


if __name__ == "__main__":
    main()

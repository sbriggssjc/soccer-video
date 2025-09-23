"""Motion-aware crop center and zoom estimator for soccer reels."""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


try:  # pragma: no cover - optional dependency
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    YOLO = None


@dataclass
class GoalBox:
    """Simple rectangle helper used for goal anchoring."""

    x: float
    y: float
    w: float
    h: float

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    def clamp(self, width: int, height: int) -> "GoalBox":
        w = float(np.clip(self.w, 1.0, width))
        h = float(np.clip(self.h, 1.0, height))
        x = float(np.clip(self.x, 0.0, max(0.0, width - w)))
        y = float(np.clip(self.y, 0.0, max(0.0, height - h)))
        return GoalBox(x, y, w, h)

    def expanded(self, factor: float, width: int, height: int) -> "GoalBox":
        cx, cy = self.center
        new_w = self.w * (1.0 + factor)
        new_h = self.h * (1.0 + factor)
        box = GoalBox(cx - new_w / 2.0, cy - new_h / 2.0, new_w, new_h)
        return box.clamp(width, height)

    def shift(self, dx: float, dy: float, width: int, height: int) -> "GoalBox":
        box = GoalBox(self.x + dx, self.y + dy, self.w, self.h)
        return box.clamp(width, height)

    def blend(self, other: "GoalBox", weight: float, width: int, height: int) -> "GoalBox":
        weight = float(np.clip(weight, 0.0, 1.0))
        keep = 1.0 - weight
        box = GoalBox(
            keep * self.x + weight * other.x,
            keep * self.y + weight * other.y,
            keep * self.w + weight * other.w,
            keep * self.h + weight * other.h,
        )
        return box.clamp(width, height)


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
    flow_mag: float
    goal_box: Optional[GoalBox]
    anchor_iou: float

    def to_row(self) -> List[str]:
        goal_values = ("", "", "", "")
        if self.goal_box is not None:
            goal_values = (
                f"{self.goal_box.x:.4f}",
                f"{self.goal_box.y:.4f}",
                f"{self.goal_box.w:.4f}",
                f"{self.goal_box.h:.4f}",
            )
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
            f"{self.flow_mag:.4f}",
            *goal_values,
            f"{self.anchor_iou:.4f}",
        ]


@dataclass
class Target:
    """Weighted point to pull the frame towards."""

    position: np.ndarray
    weight: float
    radius: float
    velocity: np.ndarray
    source: str


@dataclass
class FlowSample:
    """Sampled optical-flow vector for preview overlay."""

    origin: Tuple[int, int]
    vector: Tuple[float, float]


@dataclass
class DebugState:
    """Per-frame debug values used for overlay rendering."""

    raw_center: Tuple[float, float]
    lead_center: Tuple[float, float]
    target_center: Tuple[float, float]
    velocity: Tuple[float, float]
    zoom_target: float
    spread: float
    has_targets: bool
    confidence: float
    speed: float
    state: str
    flow_samples: Sequence[FlowSample]


def parse_vector(arg: str, count: int) -> Tuple[float, ...]:
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    if len(parts) != count:
        raise argparse.ArgumentTypeError(f"expected {count} comma-separated values")
    return tuple(float(p) for p in parts)


def lerp(a: float, b: float, t: float) -> float:
    return float(a + (b - a) * np.clip(t, 0.0, 1.0))


def smoothstep(x: float, edge0: float, edge1: float) -> float:
    if edge0 == edge1:
        return float(x >= edge1)
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return float(t * t * (3.0 - 2.0 * t))


def savgol_smooth_series(series: Sequence[float], window: int, order: int = 3) -> np.ndarray:
    """Apply a Savitzky–Golay style smoothing using local polyfits."""

    arr = np.asarray(series, dtype=np.float64)
    n = len(arr)
    if n == 0 or window <= 2:
        return arr.copy()
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    window = min(window, n if n % 2 == 1 else n - 1 if n > 1 else n)
    if window <= 2:
        return arr.copy()
    order = int(max(1, min(order, window - 1)))
    half = window // 2
    smoothed = np.empty_like(arr)
    for idx in range(n):
        start = max(0, idx - half)
        end = min(n, start + window)
        if end - start < window:
            start = max(0, end - window)
        segment = arr[start:end]
        if segment.size <= order:
            smoothed[idx] = arr[idx]
            continue
        x = np.arange(segment.size, dtype=np.float64)
        try:
            coeffs = np.polyfit(x, segment, order)
        except np.linalg.LinAlgError:
            smoothed[idx] = arr[idx]
            continue
        rel = idx - start
        smoothed[idx] = float(np.polyval(coeffs, rel))
    return smoothed


def apply_deadband(target: np.ndarray, current: np.ndarray, deadband: Sequence[float]) -> np.ndarray:
    result = target.copy()
    for axis in range(2):
        if abs(result[axis] - current[axis]) < deadband[axis]:
            result[axis] = current[axis]
    return result


def clamp_vector(vec: np.ndarray, limit: float) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= limit or norm <= 1e-6:
        return vec
    return vec * (limit / norm)


def clamp_delta(delta: float, max_step: float) -> float:
    return float(np.clip(delta, -max_step, max_step))


def round_even(value: float, limit: float) -> float:
    if value <= 0:
        return 2.0
    rounded = int(round(value))
    if rounded % 2 == 1:
        rounded += 1
    rounded = int(np.clip(rounded, 2, int(limit) - (int(limit) % 2)))
    return float(rounded)


def compute_crop_geometry(
    center: np.ndarray,
    zoom: float,
    profile: str,
    frame_size: Tuple[int, int],
    padx: float,
    pady: float,
) -> Tuple[float, float, float, float, np.ndarray]:
    width, height = frame_size
    zoom = max(zoom, 1e-6)
    if profile == "portrait":
        base_h = height / zoom
        base_w = base_h * (9.0 / 16.0)
    else:
        base_w = width / zoom
        base_h = base_w * (9.0 / 16.0)
    w_des = base_w * (1.0 + padx)
    h_des = base_h * (1.0 + pady)
    w_des = min(w_des, width)
    h_des = min(h_des, height)
    w = round_even(w_des, width)
    h = round_even(h_des, height)
    half_w = w / 2.0
    half_h = h / 2.0
    cx = float(np.clip(center[0], half_w, width - half_w))
    cy = float(np.clip(center[1], half_h, height - half_h))
    x = cx - half_w
    y = cy - half_h
    return w, h, x, y, np.array([cx, cy], dtype=np.float64)


def remap(value: float, low: float, high: float, target: Tuple[float, float]) -> float:
    if math.isclose(high, low):
        return target[0]
    t = np.clip((value - low) / (high - low), 0.0, 1.0)
    return float(target[0] + (target[1] - target[0]) * t)


def ray_intersects_box(origin: np.ndarray, direction: np.ndarray, box: GoalBox) -> bool:
    if direction.shape != (2,):
        return False
    bounds = box.bounds
    dir_x, dir_y = float(direction[0]), float(direction[1])
    t_min = -float("inf")
    t_max = float("inf")
    for axis in range(2):
        o = origin[axis]
        d = dir_x if axis == 0 else dir_y
        min_b = bounds[axis]
        max_b = bounds[axis + 2]
        if abs(d) < 1e-6:
            if o < min_b or o > max_b:
                return False
            continue
        t1 = (min_b - o) / d
        t2 = (max_b - o) / d
        t1, t2 = (t1, t2) if t1 <= t2 else (t2, t1)
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)
        if t_min > t_max:
            return False
    return t_max >= 0.0 and t_max >= t_min


def compute_iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 1e-6:
        return 0.0
    return float(np.clip(inter / union, 0.0, 1.0))


def default_goal_box(width: int, height: int, side: str) -> GoalBox:
    goal_w = width * 0.18
    goal_h = height * 0.42
    y = max(0.0, height * 0.29 - goal_h * 0.25)
    if side == "left":
        x = 0.0
    else:
        x = width - goal_w
    return GoalBox(x, y, goal_w, goal_h).clamp(width, height)


def sample_flow_vectors(flow: Optional[np.ndarray], step: int = 32, scale: float = 6.0) -> List[FlowSample]:
    if flow is None or flow.size == 0:
        return []
    height, width = flow.shape[:2]
    samples: List[FlowSample] = []
    for y in range(step // 2, height, step):
        for x in range(step // 2, width, step):
            vec = flow[y, x]
            mag = float(np.linalg.norm(vec))
            if mag < 0.6:
                continue
            samples.append(
                FlowSample(
                    origin=(x, y),
                    vector=(float(vec[0] * scale), float(vec[1] * scale)),
                )
            )
    return samples


class YOLODetector:
    """Optional Ultralytics YOLO-based detector for players and the ball."""

    def __init__(self, conf_floor: float, frame_size: Tuple[int, int]) -> None:
        self.conf_floor = float(conf_floor)
        self.frame_size = frame_size
        self.diagonal = math.hypot(frame_size[0], frame_size[1])
        self.prev_tracks: List[Tuple[int, np.ndarray]] = []
        self.model = None
        self.failed = False
        self.valid_ids: List[int] = []

    def _ensure_model(self) -> None:
        if self.model is not None or self.failed:
            return
        if YOLO is None:
            self.failed = True
            return
        try:
            self.model = YOLO("yolov8n.pt")
            names = getattr(self.model, "names", {}) or {}
            if isinstance(names, dict):
                for idx, name in names.items():
                    if isinstance(name, str) and name.lower() in {"person", "player", "sports ball", "ball"}:
                        self.valid_ids.append(int(idx))
            if not self.valid_ids:
                self.valid_ids = [0, 32]
        except Exception:
            self.model = None
            self.failed = True

    def _match_track(self, cls_id: int, center: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        best_idx = -1
        best_dist = float("inf")
        for idx, (prev_cls, prev_center) in enumerate(self.prev_tracks):
            if prev_cls != cls_id:
                continue
            dist = float(np.linalg.norm(prev_center - center))
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        if best_idx >= 0:
            return self.prev_tracks[best_idx][1], best_dist
        return None, 0.0

    def detect(self, frame: np.ndarray) -> List[Target]:
        self._ensure_model()
        if self.model is None:
            self.prev_tracks = []
            return []
        try:
            results = self.model(frame, verbose=False)
        except Exception:
            self.failed = True
            self.model = None
            self.prev_tracks = []
            return []
        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        detections: List[Target] = []
        next_tracks: List[Tuple[int, np.ndarray]] = []
        for box in boxes:
            cls_tensor = getattr(box, "cls", None)
            conf_tensor = getattr(box, "conf", None)
            xyxy = getattr(box, "xyxy", None)
            if cls_tensor is None or conf_tensor is None or xyxy is None:
                continue
            cls_id = int(cls_tensor[0])
            if self.valid_ids and cls_id not in self.valid_ids:
                continue
            conf = float(conf_tensor[0])
            if conf < self.conf_floor:
                continue
            x1, y1, x2, y2 = [float(v) for v in xyxy[0]]
            center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float64)
            prev_center, _ = self._match_track(cls_id, center)
            velocity = (
                center - prev_center
                if prev_center is not None
                else np.zeros(2, dtype=np.float64)
            )
            speed = float(np.linalg.norm(velocity))
            speed_norm = max(speed / (self.diagonal + 1e-6), 1e-3)
            weight = conf * (speed_norm ** 1.5) * self.diagonal
            radius = 0.5 * math.hypot(x2 - x1, y2 - y1)
            detections.append(
                Target(
                    position=center,
                    weight=max(weight, 1e-3),
                    radius=max(radius, 4.0),
                    velocity=velocity,
                    source="yolo",
                )
            )
            next_tracks.append((cls_id, center))
        self.prev_tracks = next_tracks
        return detections


@dataclass
class FlowResult:
    flow: np.ndarray
    targets: List[Target]
    median_mag: float
    mean_mag: float
    max_mag: float
    strength: float
    cut: bool
    points: np.ndarray
    vectors: np.ndarray
    magnitudes: np.ndarray


class FlowClusterer:
    """Cluster dense optical flow into salient motion targets."""

    def __init__(self, frame_size: Tuple[int, int], flow_thresh: float) -> None:
        self.frame_size = frame_size
        self.flow_thresh = float(flow_thresh)
        self.diagonal = math.hypot(frame_size[0], frame_size[1])
        self.max_points = 1600
        self.cluster_radius = max(32.0, 0.05 * self.diagonal)
        self.max_clusters = 4

    def _cluster(self, points: np.ndarray, vectors: np.ndarray, mags: np.ndarray) -> List[Target]:
        if points.size == 0:
            return []
        order = np.argsort(mags)[::-1]
        clusters: List[Dict[str, np.ndarray]] = []
        totals: List[float] = []
        samples: List[List[Tuple[np.ndarray, float]]] = []
        flows: List[np.ndarray] = []
        for idx in order:
            pos = points[idx]
            vec = vectors[idx]
            mag = float(mags[idx])
            if mag <= 0.0:
                continue
            assigned = False
            for c_idx, cluster in enumerate(clusters):
                center = cluster["sum_pos"] / max(totals[c_idx], 1e-6)
                if float(np.linalg.norm(center - pos)) <= self.cluster_radius:
                    totals[c_idx] += mag
                    cluster["sum_pos"] += pos * mag
                    flows[c_idx] += vec * mag
                    samples[c_idx].append((pos.copy(), mag))
                    assigned = True
                    break
            if not assigned and len(clusters) < self.max_clusters:
                clusters.append({"sum_pos": pos * mag})
                totals.append(mag)
                flows.append(vec * mag)
                samples.append([(pos.copy(), mag)])
        targets: List[Target] = []
        for cluster, total, flow_sum, pts in zip(clusters, totals, flows, samples):
            if total <= 1e-6:
                continue
            center = cluster["sum_pos"] / total
            pts_arr = np.asarray([p for p, _ in pts], dtype=np.float64)
            weights = np.asarray([w for _, w in pts], dtype=np.float64)
            if pts_arr.size == 0:
                radius = max(6.0, 0.02 * self.diagonal)
            else:
                diffs = pts_arr - center
                radius = float(
                    math.sqrt(
                        np.sum((np.linalg.norm(diffs, axis=1) ** 2) * weights)
                        / max(weights.sum(), 1e-6)
                    )
                )
            velocity = flow_sum / total
            targets.append(
                Target(
                    position=center,
                    weight=float(total),
                    radius=max(radius, 6.0),
                    velocity=velocity,
                    source="flow",
                )
            )
        return targets

    def compute(self, prev_gray: np.ndarray, gray: np.ndarray) -> FlowResult:
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
        median_mag = float(np.median(mag))
        mean_mag = float(np.mean(mag))
        max_mag = float(np.percentile(mag, 95))
        thresh = max(median_mag * self.flow_thresh, 1e-6)
        mask = mag >= thresh
        ys, xs = np.where(mask)
        if xs.size > self.max_points:
            idx = np.linspace(0, xs.size - 1, self.max_points).astype(np.int32)
            xs = xs[idx]
            ys = ys[idx]
        points = np.column_stack((xs.astype(np.float64), ys.astype(np.float64)))
        vectors = flow[ys, xs]
        mags = mag[ys, xs]
        targets = self._cluster(points, vectors, mags)
        diff_mean = float(np.mean(cv2.absdiff(prev_gray, gray)))
        cut = diff_mean > 18.0 and max_mag > 6.5
        strength = float(np.clip(max_mag / (self.diagonal * 0.12 + 1e-6), 0.0, 1.0))
        return FlowResult(
            flow=flow,
            targets=targets,
            median_mag=median_mag,
            mean_mag=mean_mag,
            max_mag=max_mag,
            strength=strength,
            cut=cut,
            points=points,
            vectors=vectors,
            magnitudes=mags,
        )


def detect_goal_candidates(frame: np.ndarray, width: int, height: int, expansion: float) -> Dict[str, Tuple[GoalBox, float]]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 200], dtype=np.uint8)
    upper = np.array([180, 70, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    kernel_h = max(9, int(round(height * 0.12)))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, kernel_h))
    vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    vertical = cv2.dilate(vertical, vertical_kernel, iterations=1)

    detections: Dict[str, Tuple[GoalBox, float]] = {}
    if np.count_nonzero(vertical) < 50:
        return detections

    _, _, stats, _ = cv2.connectedComponentsWithStats(vertical, 8, cv2.CV_32S)
    components: List[Tuple[GoalBox, float, str]] = []
    for idx in range(1, stats.shape[0]):
        x = float(stats[idx, cv2.CC_STAT_LEFT])
        y = float(stats[idx, cv2.CC_STAT_TOP])
        w = float(stats[idx, cv2.CC_STAT_WIDTH])
        h = float(stats[idx, cv2.CC_STAT_HEIGHT])
        area = float(stats[idx, cv2.CC_STAT_AREA])
        if area < 0.0025 * width * height:
            continue
        if h <= 0:
            continue
        aspect = h / max(w, 1.0)
        if aspect < 2.2:
            continue
        cx = x + w / 2.0
        side = "left" if cx < width * 0.55 else "right"
        edge_bias = 1.0 + 0.8 * (1.0 - min(cx, width - cx) / max(width * 0.5, 1.0))
        score = area * aspect * edge_bias
        components.append((GoalBox(x, y, w, h), score, side))

    by_side: Dict[str, List[Tuple[GoalBox, float]]] = {"left": [], "right": []}
    for box, score, side in components:
        by_side.setdefault(side, []).append((box, score))

    for side, boxes in by_side.items():
        if not boxes:
            continue
        boxes.sort(key=lambda item: item[1], reverse=True)
        selected = boxes[:2]
        xs = [b.x for b, _ in selected]
        ys = [b.y for b, _ in selected]
        x2s = [b.x + b.w for b, _ in selected]
        y2s = [b.y + b.h for b, _ in selected]
        x1 = min(xs)
        y1 = min(ys)
        x2 = max(x2s)
        y2 = max(y2s)
        if len(selected) == 1:
            single = selected[0][0]
            pad = single.w * 1.6
            if side == "left":
                x2 = min(width, single.x + pad)
                x1 = max(0.0, x2 - pad * 1.2)
            else:
                x1 = max(0.0, single.x + single.w - pad)
                x2 = min(width, x1 + pad * 1.2)
        agg = GoalBox(x1, y1, x2 - x1, y2 - y1).clamp(width, height)
        detections[side] = (agg.expanded(expansion, width, height), selected[0][1])
    return detections


class GoalTracker:
    """Detect and track goal boxes for anchoring."""

    def __init__(self, width: int, height: int, fps: float, side_mode: str, expansion: float) -> None:
        self.width = width
        self.height = height
        self.side_mode = side_mode
        self.expansion = expansion
        self.detect_window = max(1, int(round(fps * 2.0)))
        self.candidates: Dict[str, List[Tuple[GoalBox, float]]] = {"left": [], "right": []}
        self.motion_samples: List[Tuple[float, float]] = []
        self.goal_box: Optional[GoalBox] = None
        self.side: Optional[str] = None
        self.latest_detection: Dict[str, Tuple[GoalBox, float]] = {}

    def observe(self, frame_idx: int, frame: np.ndarray, motion_center: Optional[np.ndarray]) -> None:
        if motion_center is not None:
            self.motion_samples.append((float(motion_center[0]), float(motion_center[1])))
        detections = detect_goal_candidates(frame, self.width, self.height, self.expansion)
        self.latest_detection = detections
        if frame_idx < self.detect_window or self.side is None:
            for side, item in detections.items():
                self.candidates.setdefault(side, []).append(item)
        self._maybe_finalize(frame_idx)
        if self.side and self.goal_box is None and self.side in detections:
            self.goal_box = detections[self.side][0]

    def _maybe_finalize(self, frame_idx: int) -> None:
        if self.side is not None:
            return
        ready = False
        if self.side_mode in {"left", "right"}:
            ready = bool(self.candidates.get(self.side_mode))
        elif frame_idx + 1 >= self.detect_window:
            ready = True
        elif all(self.candidates.get(s) for s in ("left", "right")):
            ready = True
        if not ready:
            return
        aggregates: Dict[str, GoalBox] = {}
        for side, entries in self.candidates.items():
            if not entries:
                continue
            weights = np.array([max(score, 1e-3) for _, score in entries], dtype=np.float64)
            xs = np.array([box.x for box, _ in entries], dtype=np.float64)
            ys = np.array([box.y for box, _ in entries], dtype=np.float64)
            ws = np.array([box.w for box, _ in entries], dtype=np.float64)
            hs = np.array([box.h for box, _ in entries], dtype=np.float64)
            total = float(weights.sum())
            if total <= 0:
                continue
            agg = GoalBox(
                float(np.dot(xs, weights) / total),
                float(np.dot(ys, weights) / total),
                float(np.dot(ws, weights) / total),
                float(np.dot(hs, weights) / total),
            ).clamp(self.width, self.height)
            aggregates[side] = agg
        if not aggregates:
            return
        if self.side_mode in {"left", "right"} and aggregates.get(self.side_mode):
            self.side = self.side_mode
        elif self.side_mode in {"left", "right"}:
            self.side = next(iter(aggregates))
        else:
            if self.motion_samples:
                mean_motion = np.mean(np.asarray(self.motion_samples, dtype=np.float64), axis=0)
            else:
                mean_motion = np.array([self.width / 2.0, self.height / 2.0], dtype=np.float64)
            best_side = None
            best_dist = float("inf")
            for side, box in aggregates.items():
                gx, gy = box.center
                dist = math.hypot(gx - mean_motion[0], gy - mean_motion[1])
                if dist < best_dist:
                    best_side = side
                    best_dist = dist
            self.side = best_side if best_side is not None else next(iter(aggregates))
        if self.side and self.side in aggregates:
            self.goal_box = aggregates[self.side]

    def track(self, flow: np.ndarray) -> None:
        if self.goal_box is None or flow is None:
            return
        x1 = int(np.clip(self.goal_box.x, 0.0, self.width - 1))
        y1 = int(np.clip(self.goal_box.y, 0.0, self.height - 1))
        x2 = int(np.clip(self.goal_box.x + self.goal_box.w, 0.0, self.width - 1))
        y2 = int(np.clip(self.goal_box.y + self.goal_box.h, 0.0, self.height - 1))
        if x2 <= x1 or y2 <= y1:
            return
        patch = flow[y1:y2, x1:x2]
        if patch.size == 0:
            return
        mean_flow = patch.reshape(-1, 2).mean(axis=0)
        dx = float(np.clip(mean_flow[0], -30.0, 30.0))
        dy = float(np.clip(mean_flow[1], -30.0, 30.0))
        predicted = self.goal_box.shift(dx, dy, self.width, self.height)
        if self.side and self.latest_detection.get(self.side):
            detected_box, _ = self.latest_detection[self.side]
            predicted = predicted.blend(detected_box, 0.25, self.width, self.height)
        self.goal_box = self.goal_box.blend(predicted, 0.35, self.width, self.height)

    def get_box(self) -> Optional[GoalBox]:
        return self.goal_box


def draw_preview(
    frame: np.ndarray,
    crop: Tuple[float, float, float, float],
    result: FrameResult,
    debug: DebugState,
    zoom_history: Sequence[float],
    zoom_bounds: Tuple[float, float],
) -> None:
    x, y, w, h = crop
    overlay = frame.copy()
    top_left = (int(round(x)), int(round(y)))
    bottom_right = (int(round(x + w)), int(round(y + h)))
    cv2.rectangle(overlay, top_left, bottom_right, (0, 0, 255), 2)
    if result.goal_box is not None:
        gx1 = int(round(result.goal_box.x))
        gy1 = int(round(result.goal_box.y))
        gx2 = int(round(result.goal_box.x + result.goal_box.w))
        gy2 = int(round(result.goal_box.y + result.goal_box.h))
        cv2.rectangle(overlay, (gx1, gy1), (gx2, gy2), (0, 180, 0), 2)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    smooth_pt = (int(round(result.center[0])), int(round(result.center[1])))
    raw_pt = (int(round(debug.raw_center[0])), int(round(debug.raw_center[1])))
    lead_pt = (int(round(debug.lead_center[0])), int(round(debug.lead_center[1])))
    cv2.drawMarker(frame, smooth_pt, (0, 255, 255), cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
    cv2.circle(frame, raw_pt, 6, (255, 128, 0), 2)
    cv2.circle(frame, lead_pt, 4, (255, 255, 0), 1)
    arrow_scale = 0.2
    arrow_end = (
        int(round(debug.raw_center[0] + debug.velocity[0] * arrow_scale)),
        int(round(debug.raw_center[1] + debug.velocity[1] * arrow_scale)),
    )
    cv2.arrowedLine(frame, raw_pt, arrow_end, (0, 200, 255), 2, cv2.LINE_AA, tipLength=0.3)

    for sample in debug.flow_samples:
        start = (int(round(sample.origin[0])), int(round(sample.origin[1])))
        end = (
            int(round(sample.origin[0] + sample.vector[0])),
            int(round(sample.origin[1] + sample.vector[1])),
        )
        cv2.arrowedLine(frame, start, end, (60, 200, 255), 1, cv2.LINE_AA, tipLength=0.25)

    hud_w, hud_h = 220, 72
    hud_margin = 12
    hud_origin = (hud_margin, frame.shape[0] - hud_h - hud_margin)
    cv2.rectangle(
        frame,
        hud_origin,
        (hud_origin[0] + hud_w, hud_origin[1] + hud_h),
        (25, 25, 25),
        thickness=-1,
    )
    zoom_min, zoom_max = zoom_bounds
    if zoom_max <= zoom_min:
        zoom_max = zoom_min + 1.0
    history = list(zoom_history[-hud_w:])
    if history:
        norm = [
            (float(z) - zoom_min) / max(zoom_max - zoom_min, 1e-6)
            for z in history
        ]
        points = []
        for idx, value in enumerate(norm):
            px = hud_origin[0] + idx
            py = int(round(hud_origin[1] + hud_h - value * hud_h))
            points.append((px, py))
        for idx in range(1, len(points)):
            cv2.line(frame, points[idx - 1], points[idx], (120, 255, 120), 2, cv2.LINE_AA)
    text_lines = [
        f"state: {debug.state}",
        f"conf={debug.confidence:.2f} |v|={debug.speed:.1f}",
        f"z={result.zoom:.2f} tgt={debug.zoom_target:.2f}",
        f"spread={debug.spread:.1f} targets={'Y' if debug.has_targets else 'N'}",
    ]
    for idx, text in enumerate(text_lines):
        cv2.putText(
            frame,
            text,
            (hud_origin[0] + 10, hud_origin[1] + 22 + idx * 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )


def format_cli_args(args: argparse.Namespace) -> str:
    parts: List[str] = []
    for key in sorted(vars(args)):
        value = getattr(args, key)
        if isinstance(value, Path):
            value = str(value)
        elif isinstance(value, tuple):
            value = ",".join(str(v) for v in value)
        parts.append(f"{key}={value}")
    return " ".join(parts)


def write_csv(
    csv_path: Path,
    results: Iterable[FrameResult],
    fps: float,
    frame_size: Tuple[int, int],
    zoom_min: float,
    zoom_max: float,
    cli_args: Optional[argparse.Namespace] = None,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = frame_size
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(f"# fps={fps:.4f}\n")
        handle.write(f"# width={width},height={height}\n")
        handle.write(f"# zoom_min={zoom_min:.5f},zoom_max={zoom_max:.5f}\n")
        if cli_args is not None:
            handle.write(f"# cli={format_cli_args(cli_args)}\n")
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame",
                "cx",
                "cy",
                "z",
                "w",
                "h",
                "x",
                "y",
                "conf",
                "crowding",
                "flow_mag",
                "goal_x",
                "goal_y",
                "goal_w",
                "goal_h",
                "anchor_iou",
            ]
        )
        for result in results:
            writer.writerow(result.to_row())


def render_preview(
    video_path: Path,
    preview_path: Optional[Path],
    compare_path: Optional[Path],
    results: Sequence[FrameResult],
    debug_states: Sequence[DebugState],
    fps: float,
    frame_size: Tuple[int, int],
    zoom_bounds: Tuple[float, float],
) -> None:
    if preview_path is None and compare_path is None:
        return
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video for preview: {video_path}")
    width, height = frame_size
    writer = None
    compare_writer = None
    if preview_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(preview_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise SystemExit(f"Failed to open preview writer: {preview_path}")
    if compare_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        compare_writer = cv2.VideoWriter(str(compare_path), fourcc, fps, (width * 2, height))
        if not compare_writer.isOpened():
            if writer is not None:
                writer.release()
            cap.release()
            raise SystemExit(f"Failed to open compare writer: {compare_path}")
    zoom_history: List[float] = []
    for result, debug in zip(results, debug_states):
        ok, frame = cap.read()
        if not ok:
            break
        crop = (result.x, result.y, result.width, result.height)
        zoom_history.append(result.zoom)
        overlay = frame.copy()
        draw_preview(overlay, crop, result, debug, zoom_history, zoom_bounds)
        if writer is not None:
            writer.write(overlay)
        if compare_writer is not None:
            side_by_side = np.hstack((frame, overlay))
            compare_writer.write(side_by_side)
    cap.release()
    if writer is not None:
        writer.release()
    if compare_writer is not None:
        compare_writer.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate crop center/zoom tracks from motion"
    )
    parser.add_argument("--in", dest="input_path", type=Path, required=True, help="Input video path")
    parser.add_argument("--csv", dest="csv_path", type=Path, required=True, help="Output CSV path")
    parser.add_argument("--preview", dest="preview_path", type=Path, help="Optional debug preview MP4")
    parser.add_argument(
        "--compare",
        dest="compare_path",
        type=Path,
        help="Optional side-by-side debug MP4",
    )
    parser.add_argument(
        "--roi",
        choices=["generic", "goal"],
        default="generic",
        help="ROI tuning preset",
    )
    parser.add_argument(
        "--goal_side",
        choices=["auto", "left", "right"],
        default="auto",
        help="Side of field to anchor goal framing",
    )
    parser.add_argument(
        "--profile",
        choices=["portrait", "landscape"],
        default="portrait",
        help="Output profile controls aspect ratio",
    )
    parser.add_argument("--config", type=Path, help="Reserved for compatibility; unused")
    parser.add_argument("--lead", type=int, help="Legacy frames to lead/predict center")
    parser.add_argument(
        "--lead_frames",
        type=int,
        default=10,
        help="Frames of velocity lead when predicting center",
    )
    parser.add_argument("--deadband", type=float, default=10.0, help="Ignore small center deltas (px)")
    parser.add_argument(
        "--deadband_xy",
        type=lambda s: parse_vector(s, 2),
        default=(12.0, 12.0),
        help="Axis-specific deadband for center updates (px,px)",
    )
    parser.add_argument(
        "--slew_xy",
        type=lambda s: parse_vector(s, 2),
        default=(40.0, 40.0),
        help="Max per-frame change for cx,cy (px,px)",
    )
    parser.add_argument("--slew_z", type=float, default=0.05, help="Max zoom change per frame")
    parser.add_argument("--padx", type=float, default=0.22, help="Horizontal padding around action")
    parser.add_argument("--pady", type=float, default=0.18, help="Vertical padding around action")
    parser.add_argument(
        "--zoom_min", type=float, default=1.08, help="Minimum zoom (crop scale denominator)"
    )
    parser.add_argument(
        "--zoom_max", type=float, default=2.40, help="Maximum zoom (tighter crop)"
    )
    parser.add_argument("--zoom_k", type=float, default=9.0, help="Zoom spring stiffness")
    parser.add_argument(
        "--zoom_d",
        type=float,
        default=6.0,
        help="Zoom spring damping coefficient",
    )
    parser.add_argument(
        "--anchor_weight",
        type=float,
        default=0.35,
        help="Weight applied when blending toward goal anchor",
    )
    parser.add_argument(
        "--anchor_iou_min",
        type=float,
        default=0.15,
        help="Minimum IoU before anchor pull is increased",
    )
    parser.add_argument(
        "--smooth_ema", type=float, default=0.35, help="EMA smoothing alpha for center"
    )
    parser.add_argument(
        "--smooth_win",
        type=int,
        default=0,
        help="Optional Savitzky–Golay window (odd); 0 disables",
    )
    parser.add_argument(
        "--hold_frames",
        type=int,
        default=8,
        help="Hold last command this many frames after cut/lock loss",
    )
    parser.add_argument("--conf_floor", type=float, default=0.15, help="Confidence floor")
    parser.add_argument(
        "--flow_thresh",
        type=float,
        default=0.18,
        help="Motion mask threshold after normalization",
    )
    parser.add_argument(
        "--follow_k",
        type=float,
        default=140.0,
        help="Critically damped spring stiffness for center follow",
    )
    parser.add_argument(
        "--follow_d",
        type=float,
        default=24.0,
        help="Critically damped spring damping term for center follow",
    )
    parser.add_argument(
        "--max_xy_speed",
        type=float,
        default=48.0,
        help="Maximum allowed tracking speed (px/s)",
    )
    parser.add_argument(
        "--max_xy_accel",
        type=float,
        default=240.0,
        help="Maximum allowed tracking acceleration (px/s^2)",
    )
    parser.add_argument(
        "--zoom_vel_k",
        type=float,
        default=0.006,
        help="Velocity contribution when computing zoom bias",
    )
    parser.add_argument(
        "--zoom_ema",
        type=float,
        default=0.32,
        help="EMA factor applied to zoom target",
    )
    parser.add_argument(
        "--zoom_emergency",
        type=float,
        default=0.92,
        help="Multiplier applied to zoom_max during emergency zoom out",
    )
    parser.add_argument(
        "--edge_guard_x",
        type=int,
        default=120,
        help="Horizontal guard band preventing over-cropping",
    )
    parser.add_argument(
        "--edge_guard_y",
        type=int,
        default=90,
        help="Vertical guard band preventing over-cropping",
    )
    parser.add_argument(
        "--goal_bias_k",
        type=float,
        default=0.25,
        help="Strength of goal alignment bias",
    )
    parser.add_argument(
        "--goal_roi",
        choices=["left", "right", "auto"],
        default="auto",
        help="Goal mouth selection for biasing",
    )
    parser.add_argument(
        "--deadband_min",
        type=float,
        default=6.0,
        help="Minimum follow deadband",
    )
    parser.add_argument(
        "--deadband_max",
        type=float,
        default=18.0,
        help="Maximum follow deadband",
    )
    parser.add_argument(
        "--v_lo",
        type=float,
        default=40.0,
        help="Velocity threshold where deadband begins shrinking",
    )
    parser.add_argument(
        "--v_hi",
        type=float,
        default=220.0,
        help="Velocity threshold where deadband is minimal",
    )
    parser.add_argument(
        "--bary_k",
        type=int,
        default=400,
        help="Top-K flow vectors to use for barycentric fallback",
    )
    return parser.parse_args()


def run_autoframe(
    args: argparse.Namespace,
) -> Tuple[List[FrameResult], List[DebugState], float, Tuple[int, int], float, float]:
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
    frame_size = (width, height)
    diagonal = math.hypot(width, height)
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    zoom_min = float(args.zoom_min)
    zoom_max = float(args.zoom_max)
    if zoom_min > zoom_max:
        zoom_min, zoom_max = zoom_max, zoom_min

    lead_frames = args.lead_frames
    if args.lead is not None:
        lead_frames = args.lead
    lead_frames = max(float(lead_frames), 0.0)
    dt = 1.0 / max(fps, 1e-6)
    lead_time = lead_frames / max(fps, 1e-6)

    deadband_scalar = max(0.0, float(args.deadband))
    deadband_min = max(0.0, float(args.deadband_min))
    deadband_max = max(deadband_min, float(args.deadband_max))
    if deadband_scalar > 0.0:
        deadband_min = max(deadband_min, deadband_scalar * 0.5)
        deadband_max = max(deadband_max, deadband_scalar)
    base_deadband_xy = np.array(
        [
            max(deadband_scalar, float(args.deadband_xy[0])),
            max(deadband_scalar, float(args.deadband_xy[1])),
        ],
        dtype=np.float64,
    )
    slew_xy = tuple(float(v) for v in args.slew_xy)
    slew_z = float(args.slew_z)
    alpha_pos = float(args.smooth_ema)
    smooth_win = int(args.smooth_win)
    if smooth_win < 0:
        smooth_win = 0
    if smooth_win and smooth_win % 2 == 0:
        smooth_win += 1
    padx = max(0.0, float(args.padx))
    pady = max(0.0, float(args.pady))
    hold_frames = max(0, int(args.hold_frames))
    conf_floor = float(args.conf_floor)
    flow_thresh = float(args.flow_thresh)
    follow_k = float(args.follow_k)
    follow_d = float(args.follow_d)
    max_xy_speed = max(1e-6, float(args.max_xy_speed))
    max_xy_accel = max(1e-6, float(args.max_xy_accel))
    if slew_xy:
        max_xy_speed = min(max_xy_speed, min(slew_xy) * fps)
    max_zoom_speed = abs(slew_z) * fps
    zoom_k = float(args.zoom_k)
    zoom_d = float(args.zoom_d)
    zoom_vel_k = float(args.zoom_vel_k)
    zoom_ema_alpha = float(args.zoom_ema)
    zoom_emergency = float(args.zoom_emergency)
    edge_guard_x = max(0.0, float(args.edge_guard_x))
    edge_guard_y = max(0.0, float(args.edge_guard_y))
    goal_bias_k = float(args.goal_bias_k)
    v_lo = float(args.v_lo)
    v_hi = max(v_lo + 1e-6, float(args.v_hi))
    bary_k = max(1, int(args.bary_k))

    detector = YOLODetector(conf_floor, frame_size)
    flow_clusterer = FlowClusterer(frame_size, flow_thresh)

    goal_preference = args.goal_roi
    if goal_preference == "auto" and args.goal_side in {"left", "right"}:
        goal_preference = args.goal_side
    goal_tracker: Optional[GoalTracker]
    if args.roi == "goal":
        goal_tracker = GoalTracker(width, height, fps, goal_preference, expansion=0.07)
    else:
        goal_tracker = None

    center_state = np.array([width / 2.0, height / 2.0], dtype=np.float64)
    center_velocity = np.zeros(2, dtype=np.float64)
    ema_measure = center_state.copy()
    prev_filtered_center = center_state.copy()
    prev_spread = diagonal * 0.25
    zoom_state = zoom_min
    zoom_velocity = 0.0
    zoom_target_ema = zoom_state
    conf_smooth = 1.0
    low_conf_counter = 0
    emergency_timer = 0
    spike_timer = 0

    raw_x: List[float] = [center_state[0]]
    raw_y: List[float] = [center_state[1]]

    results: List[FrameResult] = []
    debug_states: List[DebugState] = []

    frame_idx = 0
    current_frame = frame
    while True:
        if frame_idx > 0:
            ok, current_frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            flow_result = flow_clusterer.compute(prev_gray, gray)
            prev_gray = gray
        else:
            flow_result = FlowResult(
                flow=np.zeros((height, width, 2), dtype=np.float32),
                targets=[],
                median_mag=0.0,
                mean_mag=0.0,
                max_mag=0.0,
                strength=0.0,
                cut=False,
                points=np.zeros((0, 2), dtype=np.float64),
                vectors=np.zeros((0, 2), dtype=np.float64),
                magnitudes=np.zeros((0,), dtype=np.float64),
            )

        detections = detector.detect(current_frame)
        flow_targets = flow_result.targets
        targets = detections + flow_targets
        has_targets = bool(targets)

        if has_targets:
            positions = np.array([t.position for t in targets], dtype=np.float64)
            weights = np.array([max(t.weight, 1e-3) for t in targets], dtype=np.float64)
            weight_sum = float(weights.sum())
            if weight_sum <= 1e-6:
                weights.fill(1.0)
                weight_sum = float(weights.sum())
            raw_center = (weights[:, None] * positions).sum(axis=0) / weight_sum
            radii = np.array([max(t.radius, 1.0) for t in targets], dtype=np.float64)
            distances = np.linalg.norm(positions - raw_center, axis=1)
            spread_samples = distances + radii
            spread = float(np.percentile(spread_samples, 75)) if spread_samples.size else prev_spread
            velocities = np.array([t.velocity for t in targets], dtype=np.float64)
            flow_velocity = (weights[:, None] * velocities).sum(axis=0) / weight_sum
        else:
            raw_center = prev_filtered_center.copy()
            spread = prev_spread * 1.03
            weight_sum = 0.0
            flow_velocity = np.zeros(2, dtype=np.float64)

        spread = float(np.clip(spread, diagonal * 0.025, diagonal))
        raw_x.append(raw_center[0])
        raw_y.append(raw_center[1])
        if smooth_win > 1 and len(raw_x) >= smooth_win:
            smooth_center = np.array(
                [
                    savgol_smooth_series(raw_x, smooth_win)[-1],
                    savgol_smooth_series(raw_y, smooth_win)[-1],
                ],
                dtype=np.float64,
            )
        else:
            smooth_center = raw_center.copy()

        ema_prev = ema_measure.copy()
        ema_measure = ema_measure * (1.0 - alpha_pos) + smooth_center * alpha_pos
        measured_center = ema_measure.copy()

        bary_center: Optional[np.ndarray] = None
        if flow_result.magnitudes.size:
            order = np.argsort(flow_result.magnitudes)[::-1][:bary_k]
            weights_bary = flow_result.magnitudes[order]
            total_bary = float(weights_bary.sum())
            if total_bary > 1e-6:
                bary_center = (
                    weights_bary[:, None] * flow_result.points[order]
                ).sum(axis=0) / total_bary

        confidence = float(
            np.clip(
                0.6 * (weight_sum / (diagonal + 1e-6)) + 0.4 * flow_result.strength,
                0.0,
                1.0,
            )
        )
        crowding = float(np.clip(spread / (diagonal * 0.6 + 1e-6), 0.0, 1.0))
        flow_mag = float(np.clip(flow_result.strength, 0.0, 1.0))

        if confidence < conf_floor:
            emergency_timer = hold_frames
            low_conf_counter += 1
        else:
            emergency_timer = max(0, emergency_timer - 1)
            low_conf_counter = 0

        bary_active = low_conf_counter > hold_frames
        state = "FOLLOW"
        if bary_active and bary_center is not None:
            measured_center = bary_center.copy()
            flow_velocity = np.zeros(2, dtype=np.float64)
            state = "BARYCENTER"
        elif emergency_timer > 0:
            state = "EMERGENCY"

        if goal_tracker is not None:
            motion_hint = measured_center if has_targets else None
            goal_tracker.observe(frame_idx, current_frame, motion_hint)
            goal_tracker.track(flow_result.flow)
        goal_box = goal_tracker.get_box() if goal_tracker else None
        if goal_box is None and goal_preference in {"left", "right"}:
            goal_box = default_goal_box(width, height, goal_preference)

        velocity_px_sec = flow_velocity * fps
        if not has_targets and not bary_active:
            velocity_px_sec = (ema_measure - ema_prev) / max(dt, 1e-6)
        flow_speed = float(np.linalg.norm(velocity_px_sec))

        if flow_result.cut or flow_speed > v_hi:
            spike_timer = hold_frames
        else:
            spike_timer = max(0, spike_timer - 1)

        deadband_value = lerp(
            deadband_min,
            deadband_max,
            1.0 - smoothstep(flow_speed, v_lo, v_hi),
        )
        deadband_value = float(np.clip(deadband_value, deadband_min, deadband_max))
        deadband_vec = np.maximum(
            np.full(2, deadband_value, dtype=np.float64),
            base_deadband_xy * (deadband_value / max(deadband_max, 1e-6)),
        )

        lead_center = measured_center + velocity_px_sec * lead_time
        lead_center = np.clip(lead_center, [0.0, 0.0], [width - 1.0, height - 1.0])

        target_point = apply_deadband(lead_center, center_state, deadband_vec)
        pre_anchor_center = target_point.copy()

        accel = follow_k * (target_point - center_state) - follow_d * center_velocity
        accel = clamp_vector(accel, max_xy_accel)
        center_velocity = clamp_vector(center_velocity + accel * dt, max_xy_speed)
        new_center = center_state + center_velocity * dt

        if goal_box is not None and flow_speed > 1e-3:
            goal_hint = np.array(goal_box.center, dtype=np.float64)
            goal_roi_box = goal_box.expanded(0.65, width, height)
            if ray_intersects_box(measured_center, velocity_px_sec, goal_roi_box):
                direction_norm = velocity_px_sec / flow_speed
                to_goal = goal_hint - new_center
                dist_goal = float(np.linalg.norm(to_goal))
                if dist_goal > 1e-6:
                    to_goal_norm = to_goal / dist_goal
                    bias = float(np.clip(np.dot(direction_norm, to_goal_norm), 0.0, 1.0))
                    if bias > 0.0:
                        new_center = (
                            new_center * (1.0 - bias * goal_bias_k)
                            + goal_hint * (bias * goal_bias_k)
                        )

        guard_x = min(edge_guard_x, width / 2.0)
        guard_y = min(edge_guard_y, height / 2.0)
        min_x = guard_x
        max_x = width - guard_x
        min_y = guard_y
        max_y = height - guard_y
        if min_x > max_x:
            min_x = max_x = width / 2.0
        if min_y > max_y:
            min_y = max_y = height / 2.0
        new_center[0] = float(np.clip(new_center[0], min_x, max_x))
        new_center[1] = float(np.clip(new_center[1], min_y, max_y))

        conf_smooth = conf_smooth * (1.0 - zoom_ema_alpha) + confidence * zoom_ema_alpha
        z_spread = remap(spread, 250.0, 1200.0, (2.2, 1.1))
        z_vel = 1.0 / max(1.0 + zoom_vel_k * flow_speed, 1e-6)
        z_conf = lerp(zoom_max, z_spread * z_vel, conf_smooth)
        if emergency_timer > 0 or bary_active:
            z_conf = max(z_conf, zoom_max * zoom_emergency)
        if spike_timer > 0:
            z_conf = max(z_conf, zoom_max * 0.96)
        z_target = float(np.clip(z_conf, zoom_min, zoom_max))
        zoom_target_ema = zoom_target_ema * (1.0 - zoom_ema_alpha) + z_target * zoom_ema_alpha
        zoom_accel = zoom_k * (zoom_target_ema - zoom_state) - zoom_d * zoom_velocity
        zoom_velocity += zoom_accel * dt
        if max_zoom_speed > 0.0:
            zoom_velocity = float(np.clip(zoom_velocity, -max_zoom_speed, max_zoom_speed))
        zoom_state += zoom_velocity * dt
        if zoom_state < zoom_min or zoom_state > zoom_max:
            zoom_state = float(np.clip(zoom_state, zoom_min, zoom_max))
            zoom_velocity = 0.0

        w, h, x, y, adjusted_center = compute_crop_geometry(
            new_center, zoom_state, args.profile, frame_size, padx, pady
        )
        center_prev = center_state.copy()
        center_state = adjusted_center
        center_velocity = clamp_vector((center_state - center_prev) / max(dt, 1e-6), max_xy_speed)

        prev_filtered_center = measured_center
        prev_spread = spread

        anchor_iou = 0.0
        if goal_box is not None:
            crop_bounds = (x, y, x + w, y + h)
            anchor_iou = compute_iou(crop_bounds, goal_box.bounds)

        flow_samples = sample_flow_vectors(flow_result.flow)

        results.append(
            FrameResult(
                frame=frame_idx,
                center=(center_state[0], center_state[1]),
                zoom=zoom_state,
                width=w,
                height=h,
                x=x,
                y=y,
                conf=confidence,
                crowding=crowding,
                flow_mag=flow_mag,
                goal_box=goal_box,
                anchor_iou=float(anchor_iou),
            )
        )
        debug_states.append(
            DebugState(
                raw_center=(measured_center[0], measured_center[1]),
                lead_center=(lead_center[0], lead_center[1]),
                target_center=(pre_anchor_center[0], pre_anchor_center[1]),
                velocity=(velocity_px_sec[0], velocity_px_sec[1]),
                zoom_target=z_target,
                spread=spread,
                has_targets=has_targets,
                confidence=confidence,
                speed=flow_speed,
                state=state,
                flow_samples=flow_samples,
            )
        )

        frame_idx += 1

    cap.release()
    return results, debug_states, fps, frame_size, zoom_min, zoom_max

def main() -> None:
    args = parse_args()
    results, debug_states, fps, frame_size, zoom_min, zoom_max = run_autoframe(args)
    write_csv(args.csv_path, results, fps, frame_size, zoom_min, zoom_max, args)
    if args.preview_path or args.compare_path:
        render_preview(
            args.input_path,
            args.preview_path,
            args.compare_path,
            results,
            debug_states,
            fps,
            frame_size,
            (zoom_min, zoom_max),
        )
    print(f"Wrote {len(results)} motion samples to {args.csv_path}")
    if args.preview_path:
        print(f"Preview: {args.preview_path}")
    if args.compare_path:
        print(f"Compare preview: {args.compare_path}")


if __name__ == "__main__":
    main()

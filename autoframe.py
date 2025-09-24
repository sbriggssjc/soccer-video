"""Motion-aware crop center and zoom estimator for soccer reels."""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp *x* to the inclusive range ``[lo, hi]``."""

    return lo if x < lo else hi if x > hi else x


def compute_motion_spread(
    flow: Optional[np.ndarray],
    cx: float,
    cy: float,
    flow_thresh: float,
    frame_w: int,
    frame_h: int,
) -> float:
    """Estimate how widely motion is distributed around the target point."""

    if flow is None or flow.size == 0:
        return 0.5 * max(frame_w, frame_h)

    fx = flow[..., 0]
    fy = flow[..., 1]
    mag = np.hypot(fx, fy)
    if mag.size == 0:
        return 0.5 * max(frame_w, frame_h)

    m90 = float(np.percentile(mag, 90.0)) if mag.size else 0.0
    thresh = max(flow_thresh, 0.25 * m90)
    ys, xs = np.nonzero(mag > thresh)
    if len(xs) >= 12:
        d = np.hypot(xs - cx, ys - cy)
        if d.size:
            return float(np.percentile(d, 85.0))
    return 0.5 * max(frame_w, frame_h)


def update_camera(
    n: int,
    target_cx: float,
    target_cy: float,
    flow: Optional[np.ndarray],
    flow_strength: Optional[float],
    conf: Optional[float],
    params: SimpleNamespace,
    state: Dict[str, float],
    frame_w: int,
    frame_h: int,
) -> Tuple[float, float, float, float, bool]:
    """Compute the camera center/zoom using velocity lead and emergency zoom-outs."""

    lead_frames = float(params.lead_frames)
    follow_k = float(params.follow_k)
    follow_d = float(np.clip(params.follow_d, 0.0, 1.0))
    vmax = float(params.max_xy_speed)
    amax = float(params.max_xy_accel)
    zoom_min = float(params.zoom_min)
    zoom_max = float(params.zoom_max)
    zoom_base = float(params.zoom_base)
    zoom_vel_k = float(params.zoom_vel_k)
    zoom_edge_boost = float(params.zoom_edge_boost)
    zoom_ema = float(np.clip(params.zoom_ema, 0.0, 1.0))
    emergency_gain = float(params.emergency_gain)
    flow_thresh = float(params.flow_thresh)
    flow_spike = float(params.flow_spike)
    conf_floor = float(params.conf_floor)
    edge_guard_x = float(params.edge_guard_x)
    edge_guard_y = float(params.edge_guard_y)

    cx = float(target_cx)
    cy = float(target_cy)

    prev_cx = state.get("prev_target_cx", cx)
    prev_cy = state.get("prev_target_cy", cy)
    vx = cx - prev_cx
    vy = cy - prev_cy
    cx_pred = cx + vx * lead_frames
    cy_pred = cy + vy * lead_frames

    cam_cx = state.get("cam_x", cx)
    cam_cy = state.get("cam_y", cy)
    cam_vx = state.get("cam_vx", 0.0)
    cam_vy = state.get("cam_vy", 0.0)
    zoom = state.get("zoom", zoom_base)

    ax = follow_k * (cx_pred - cam_cx) - (1.0 - follow_d) * cam_vx
    ay = follow_k * (cy_pred - cam_cy) - (1.0 - follow_d) * cam_vy

    a_mag = math.hypot(ax, ay)
    if a_mag > amax > 0.0:
        scale = amax / (a_mag + 1e-9)
        ax *= scale
        ay *= scale

    cam_vx += ax
    cam_vy += ay

    v_mag = math.hypot(cam_vx, cam_vy)
    if v_mag > vmax > 0.0:
        scale = vmax / (v_mag + 1e-9)
        cam_vx *= scale
        cam_vy *= scale

    cam_cx += cam_vx
    cam_cy += cam_vy
    cam_cx = float(np.clip(cam_cx, 0.0, frame_w))
    cam_cy = float(np.clip(cam_cy, 0.0, frame_h))

    target_speed = math.hypot(vx, vy)
    z_target = zoom_base + zoom_vel_k * target_speed

    if edge_guard_x > 1e-6:
        edge_x = max(0.0, edge_guard_x - min(cam_cx, frame_w - cam_cx)) / max(
            edge_guard_x, 1e-6
        )
    else:
        edge_x = 0.0
    if edge_guard_y > 1e-6:
        edge_y = max(0.0, edge_guard_y - min(cam_cy, frame_h - cam_cy)) / max(
            edge_guard_y, 1e-6
        )
    else:
        edge_y = 0.0
    z_target -= zoom_edge_boost * max(edge_x, edge_y)
    z_target = clamp(z_target, zoom_min, zoom_max)

    zoom = zoom_ema * z_target + (1.0 - zoom_ema) * zoom

    spike = False
    if flow_strength is not None and flow_strength > flow_spike:
        spike = True
    if conf is not None and conf < conf_floor:
        spike = True
    if spike:
        zoom = max(zoom_min, min(zoom_max, zoom * emergency_gain))

    spread = compute_motion_spread(flow, cx, cy, flow_thresh, frame_w, frame_h)

    state.update(
        dict(
            cam_x=float(cam_cx),
            cam_y=float(cam_cy),
            cam_vx=float(cam_vx),
            cam_vy=float(cam_vy),
            zoom=float(zoom),
            z_ema=float(zoom),
            prev_target_cx=float(cx),
            prev_target_cy=float(cy),
            cx_prev=float(cx),
            cy_prev=float(cy),
            cx_raw=float(cx),
            cy_raw=float(cy),
            cx_pred=float(cx_pred),
            cy_pred=float(cy_pred),
            spread=float(spread),
            spd_tgt=float(target_speed),
            spike=bool(spike),
            zoom_desire=float(z_target),
        )
    )

    return float(cam_cx), float(cam_cy), float(zoom), float(spread), bool(spike)


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
    confidence: float = 1.0


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
                    confidence=float(np.clip(conf, 0.0, 1.0)),
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
                    confidence=float(
                        np.clip(total / (self.diagonal + 1e-6), 0.0, 1.0)
                    ),
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
        default=12,
        help="Frames of velocity lead when predicting center",
    )
    parser.add_argument(
        "--ball_window_x",
        type=int,
        default=None,
        help="Half-width of the ball-biased window for point weighting",
    )
    parser.add_argument(
        "--ball_window_y",
        type=int,
        default=None,
        help="Half-height of the ball-biased window for point weighting",
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
        "--zoom_max", type=float, default=2.80, help="Maximum zoom (tighter crop)"
    )
    parser.add_argument(
        "--zoom_base",
        type=float,
        default=1.20,
        help="Baseline zoom level before motion/edge adjustments",
    )
    parser.add_argument(
        "--zoom_edge_boost",
        type=float,
        default=0.20,
        help="How strongly to zoom out when the camera nears field edges",
    )
    parser.add_argument(
        "--spread_lo",
        type=float,
        default=None,
        help="Motion spread mapped to minimum zoom",
    )
    parser.add_argument(
        "--spread_hi",
        type=float,
        default=None,
        help="Motion spread mapped to maximum zoom",
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
    parser.add_argument("--conf_floor", type=float, default=0.25, help="Confidence floor")
    parser.add_argument(
        "--flow_thresh",
        type=float,
        default=0.18,
        help="Motion mask threshold after normalization",
    )
    parser.add_argument(
        "--flow_spike",
        type=float,
        default=0.18,
        help="Flow magnitude threshold triggering emergency zoom-out",
    )
    parser.add_argument(
        "--flow_thresh_high",
        type=float,
        default=None,
        help="High watermark for detecting motion spikes",
    )
    parser.add_argument(
        "--follow_k",
        type=float,
        default=0.20,
        help="How strongly the camera chases the target center",
    )
    parser.add_argument(
        "--follow_d",
        type=float,
        default=0.85,
        help="Damping factor for the follow spring (0..1)",
    )
    parser.add_argument(
        "--max_xy_speed",
        type=float,
        default=85.0,
        help="Maximum allowed tracking speed (px/frame)",
    )
    parser.add_argument(
        "--max_xy_accel",
        type=float,
        default=380.0,
        help="Maximum allowed tracking acceleration (px/frame^2)",
    )
    parser.add_argument(
        "--zoom_vel_k",
        type=float,
        default=0.0025,
        help="Velocity contribution when computing zoom bias",
    )
    parser.add_argument(
        "--zoom_ema",
        type=float,
        default=0.22,
        help="EMA factor applied to zoom target",
    )
    parser.add_argument(
        "--zoom_emergency",
        type=float,
        default=0.992,
        help="Per-frame multiplier applied to zoom during emergencies",
    )
    parser.add_argument(
        "--edge_guard_x",
        type=int,
        default=150,
        help="Horizontal guard band preventing over-cropping",
    )
    parser.add_argument(
        "--edge_guard_y",
        type=int,
        default=120,
        help="Vertical guard band preventing over-cropping",
    )
    parser.add_argument(
        "--keep_box_x",
        type=float,
        default=None,
        help="Relative half-width of the keep-inside box",
    )
    parser.add_argument(
        "--keep_box_y",
        type=float,
        default=None,
        help="Relative half-height of the keep-inside box",
    )
    parser.add_argument(
        "--goal_bias_k",
        type=float,
        default=0.25,
        help="Strength of goal alignment bias",
    )
    parser.add_argument(
        "--goal_blend",
        type=float,
        default=0.35,
        help="Blend factor nudging the camera toward the goal mouth",
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
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    zoom_min = float(args.zoom_min)
    zoom_max = float(args.zoom_max)
    if zoom_min > zoom_max:
        zoom_min, zoom_max = zoom_max, zoom_min

    lead_frames = args.lead_frames
    if args.lead is not None:
        lead_frames = args.lead
    lead_frames = max(float(lead_frames), 0.0)
    args.lead_frames = lead_frames

    padx = max(0.0, float(args.padx))
    pady = max(0.0, float(args.pady))
    conf_floor = float(args.conf_floor)
    flow_thresh = float(args.flow_thresh)
    follow_k = float(args.follow_k)
    follow_d = float(args.follow_d)
    max_xy_speed = max(1e-6, float(args.max_xy_speed))
    max_xy_accel = max(1e-6, float(args.max_xy_accel))
    zoom_vel_k = float(args.zoom_vel_k)
    zoom_ema_alpha = float(args.zoom_ema)
    zoom_emergency = float(args.zoom_emergency)
    zoom_base = clamp(float(args.zoom_base), zoom_min, zoom_max)
    zoom_edge_boost = float(args.zoom_edge_boost)
    edge_guard_x = float(args.edge_guard_x)
    edge_guard_y = float(args.edge_guard_y)
    goal_bias_k = float(args.goal_bias_k)
    goal_blend = float(args.goal_blend)
    flow_spike = float(args.flow_spike)
    if args.flow_thresh_high is not None:
        flow_spike = float(args.flow_thresh_high)
    ball_window_x = int(args.ball_window_x) if args.ball_window_x is not None else 240
    ball_window_y = int(args.ball_window_y) if args.ball_window_y is not None else 180
    spread_hi = float(args.spread_hi) if args.spread_hi is not None else 420.0

    camera_params = SimpleNamespace(
        lead_frames=lead_frames,
        follow_k=follow_k,
        follow_d=follow_d,
        max_xy_speed=max_xy_speed,
        max_xy_accel=max_xy_accel,
        edge_guard_x=edge_guard_x,
        edge_guard_y=edge_guard_y,
        zoom_min=zoom_min,
        zoom_max=zoom_max,
        zoom_base=zoom_base,
        zoom_vel_k=zoom_vel_k,
        zoom_edge_boost=zoom_edge_boost,
        zoom_ema=zoom_ema_alpha,
        emergency_gain=zoom_emergency,
        flow_thresh=flow_thresh,
        flow_spike=flow_spike,
        conf_floor=conf_floor,
    )

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

    def clamp_value(value: float, low: float, high: float) -> float:
        return float(np.clip(value, low, high))

    initial_zoom = clamp(zoom_base, zoom_min, zoom_max)
    state: Dict[str, float] = {
        "cam_x": width / 2.0,
        "cam_y": height / 2.0,
        "cam_vx": 0.0,
        "cam_vy": 0.0,
        "zoom": initial_zoom,
        "z_ema": initial_zoom,
        "cx_prev": width / 2.0,
        "cy_prev": height / 2.0,
        "prev_target_cx": width / 2.0,
        "prev_target_cy": height / 2.0,
    }
    prev_cam_cx = width / 2.0
    prev_cam_cy = height / 2.0

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

        points: List[Tuple[float, float, float]] = []
        for target in targets:
            conf_val = float(np.clip(target.confidence, 0.0, 1.0))
            points.append(
                (float(target.position[0]), float(target.position[1]), conf_val)
            )
        if flow_result.points.size and flow_result.magnitudes.size:
            max_mag = float(np.max(flow_result.magnitudes))
            inv_mag = 1.0 / max_mag if max_mag > 1e-6 else 0.0
            for pos, mag in zip(flow_result.points, flow_result.magnitudes):
                weight = float(np.clip(mag * inv_mag, 0.0, 1.0))
                if weight <= 0.0:
                    continue
                points.append((float(pos[0]), float(pos[1]), weight))

        if targets:
            weights = np.array([max(t.weight, 1e-3) for t in targets], dtype=np.float64)
            velocities = np.array([t.velocity for t in targets], dtype=np.float64)
            total_weight = float(weights.sum())
            if total_weight > 1e-6:
                avg_velocity = (weights[:, None] * velocities).sum(axis=0) / total_weight
            else:
                avg_velocity = np.zeros(2, dtype=np.float64)
        else:
            avg_velocity = np.zeros(2, dtype=np.float64)
        vx = float(avg_velocity[0])
        vy = float(avg_velocity[1])
        motion_mag = float(flow_result.strength)

        W, H = frame_size
        conf_raw = 0.0
        use_points: List[Tuple[float, float, float]] = []
        if not points:
            cx_hint, cy_hint = prev_cam_cx, prev_cam_cy
        else:
            bx = sum(p[0] for p in points) / len(points)
            by = sum(p[1] for p in points) / len(points)
            loc = [
                (x, y, c)
                for (x, y, c) in points
                if abs(x - prev_cam_cx) <= ball_window_x
                and abs(y - prev_cam_cy) <= ball_window_y
            ]
            threshold = max(10, len(points) // 8)
            use_points = loc if len(loc) >= threshold else points
            wsum = sum(max(c, conf_floor) for (*_, c) in use_points)
            if wsum == 0:
                cx_hint, cy_hint = bx, by
            else:
                cx_hint = sum(x * max(c, conf_floor) for (x, _, c) in use_points) / wsum
                cy_hint = sum(y * max(c, conf_floor) for (_, y, c) in use_points) / wsum
                conf_raw = min(1.0, wsum / max(len(use_points), 1))
        flow_mag = motion_mag

        if goal_tracker is not None:
            motion_hint = (
                np.array([cx_hint, cy_hint], dtype=np.float64) if points else None
            )
            goal_tracker.observe(frame_idx, current_frame, motion_hint)
            goal_tracker.track(flow_result.flow)
        goal_box = goal_tracker.get_box() if goal_tracker else None
        if goal_box is None and goal_preference in {"left", "right"}:
            goal_box = default_goal_box(width, height, goal_preference)
        goal_x = goal_box.center[0] if goal_box is not None else None

        conf_value = conf_raw if points else 1.0

        target_cx = cx_hint
        target_cy = cy_hint
        if goal_box is not None and goal_blend > 1e-6:
            goal_cx, goal_cy = goal_box.center
            dist = math.hypot(target_cx - goal_cx, target_cy - goal_cy)
            field_norm = max(math.hypot(width, height) * 0.35, 1e-6)
            proximity = 1.0 - float(np.clip(dist / field_norm, 0.0, 1.0))
            confidence_boost = float(np.clip(conf_raw, 0.0, 1.0))
            alpha = goal_blend * proximity * max(confidence_boost, 0.25)
            alpha = clamp(alpha, 0.0, 0.75)
            target_cx = (1.0 - alpha) * target_cx + alpha * goal_cx
            target_cy = (1.0 - alpha) * target_cy + alpha * goal_cy

        cam_cx, cam_cy, z, spread_est, spike = update_camera(
            frame_idx,
            target_cx,
            target_cy,
            flow_result.flow,
            flow_mag,
            conf_value,
            camera_params,
            state,
            width,
            height,
        )

        if goal_bias_k and goal_x is not None:
            dist_to_goal = abs(cam_cx - goal_x)
            bias = 1.0 - clamp_value(dist_to_goal / (W * 0.5), 0.0, 1.0)
            z_goal = max(
                z,
                zoom_min + bias * (zoom_max - zoom_min) * goal_bias_k,
            )
            if z_goal != z:
                z = clamp_value(z_goal, zoom_min, zoom_max)
                state["z_ema"] = z
                state["zoom"] = z

        crop_center = np.array([cam_cx, cam_cy], dtype=np.float64)
        w, h, x, y, adjusted_center = compute_crop_geometry(
            crop_center, z, args.profile, frame_size, padx, pady
        )
        cam_cx, cam_cy = float(adjusted_center[0]), float(adjusted_center[1])
        state["cam_x"] = cam_cx
        state["cam_y"] = cam_cy

        prev_cam_cx = cam_cx
        prev_cam_cy = cam_cy

        confidence = float(np.clip(conf_raw, 0.0, 1.0))
        spread_val = float(spread_est)
        crowding = float(np.clip(spread_val / max(spread_hi, 1e-6), 0.0, 1.0))
        flow_mag = float(np.clip(flow_mag, 0.0, 1.0))
        anchor_iou = 0.0
        if goal_box is not None:
            crop_bounds = (x, y, x + w, y + h)
            anchor_iou = compute_iou(crop_bounds, goal_box.bounds)

        flow_samples = sample_flow_vectors(flow_result.flow)

        results.append(
            FrameResult(
                frame=frame_idx,
                center=(cam_cx, cam_cy),
                zoom=z,
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
                raw_center=(
                    float(state.get("cx_raw", cam_cx)),
                    float(state.get("cy_raw", cam_cy)),
                ),
                lead_center=(
                    float(state.get("cx_pred", cam_cx)),
                    float(state.get("cy_pred", cam_cy)),
                ),
                target_center=(cam_cx, cam_cy),
                velocity=(
                    float(state.get("cam_vx", 0.0)),
                    float(state.get("cam_vy", 0.0)),
                ),
                zoom_target=float(state.get("zoom_desire", z)),
                spread=spread_val,
                has_targets=bool(points),
                confidence=confidence,
                speed=math.hypot(vx, vy),
                state="SPIKE" if spike else "TRACK",
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

"""Motion-aware crop center and zoom estimator for soccer reels."""
from __future__ import annotations

import argparse
import csv
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


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
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], float, float, Optional[np.ndarray], float]:
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
        motion_conf = 0.0
        if mask_pixels > 0:
            _, labels, stats, centroids = cv2.connectedComponentsWithStats(
                mask.astype(np.uint8), 8, cv2.CV_32S
            )
            selected = self._select_component(labels, stats, centroids, prev_center)
            centroid, spread, motion_conf = self._measure_component(
                norm, labels, stats, centroids, selected, prev_center
            )
        else:
            self.prev_conf *= 0.7

        weight_sum = float(norm.sum())
        motion_center: Optional[np.ndarray]
        if weight_sum > 1e-6:
            mx = float((norm * self.grid_x).sum() / weight_sum)
            my = float((norm * self.grid_y).sum() / weight_sum)
            motion_center = np.array([mx, my], dtype=np.float64)
        else:
            motion_center = None
        motion_strength = float(np.clip(norm.mean() if norm.size else 0.0, 0.0, 1.0))

        return flow, norm, centroid, spread, motion_conf, motion_center, motion_strength


def parse_vector(arg: str, count: int) -> Tuple[float, ...]:
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    if len(parts) != count:
        raise argparse.ArgumentTypeError(f"expected {count} comma-separated values")
    return tuple(float(p) for p in parts)


def apply_deadband_xy(
    predicted: np.ndarray, prev_cmd: np.ndarray, deadband: Sequence[float]
) -> np.ndarray:
    result = predicted.copy()
    for axis in range(2):
        delta = predicted[axis] - prev_cmd[axis]
        if abs(delta) < deadband[axis]:
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


def compute_crop_size(
    zoom: float, profile: str, padx: float, pady: float, frame_size: Tuple[int, int]
) -> Tuple[float, float]:
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
    return w * scale, h * scale


def compute_crop_dimensions(
    center: np.ndarray,
    zoom: float,
    profile: str,
    padx: float,
    pady: float,
    frame_size: Tuple[int, int],
) -> Tuple[float, float, float, float, np.ndarray]:
    width, height = frame_size
    w, h = compute_crop_size(zoom, profile, padx, pady, frame_size)
    half_w = w / 2.0
    half_h = h / 2.0
    x = float(np.clip(center[0] - half_w, 0.0, max(0.0, width - w)))
    y = float(np.clip(center[1] - half_h, 0.0, max(0.0, height - h)))
    adjusted_center = np.array([x + half_w, y + half_h], dtype=np.float64)
    return w, h, x, y, adjusted_center


def clamp_center(
    center: np.ndarray, crop_size: Tuple[float, float], frame_size: Tuple[int, int]
) -> np.ndarray:
    width, height = frame_size
    w, h = crop_size
    half_w = w / 2.0
    half_h = h / 2.0
    clamped = np.array(center, dtype=np.float64)
    clamped[0] = float(np.clip(clamped[0], half_w, width - half_w))
    clamped[1] = float(np.clip(clamped[1], half_h, height - half_h))
    return clamped


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


def estimate_extent(
    norm: np.ndarray, center: np.ndarray, frame_size: Tuple[int, int]
) -> float:
    height, width = norm.shape[:2]
    cx = int(round(float(np.clip(center[0], 0.0, width - 1.0))))
    cy = int(round(float(np.clip(center[1], 0.0, height - 1.0))))
    radius = int(max(8, 0.25 * min(width, height)))
    x1 = max(0, cx - radius)
    y1 = max(0, cy - radius)
    x2 = min(width, cx + radius)
    y2 = min(height, cy + radius)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    patch = norm[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0
    local_max = float(patch.max())
    if local_max <= 1e-6:
        return 0.0
    thresh = local_max * 0.3
    mask = patch >= thresh
    area = float(mask.sum())
    if area <= 1.0:
        return 0.0
    eq_radius = math.sqrt(area / math.pi)
    return float(np.clip(eq_radius / max(float(radius), 1.0), 0.0, 1.0))


def detect_goal_candidates(
    frame: np.ndarray, width: int, height: int, expansion: float
) -> Dict[str, Tuple[GoalBox, float]]:
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
    center: Tuple[float, float],
    zoom: float,
    conf: float,
    goal_box: Optional[GoalBox],
    anchor_iou: float,
) -> None:
    x, y, w, h = crop
    overlay = frame.copy()
    top_left = (int(round(x)), int(round(y)))
    bottom_right = (int(round(x + w)), int(round(y + h)))
    cv2.rectangle(overlay, top_left, bottom_right, (0, 0, 255), 2)
    if goal_box is not None:
        gx1 = int(round(goal_box.x))
        gy1 = int(round(goal_box.y))
        gx2 = int(round(goal_box.x + goal_box.w))
        gy2 = int(round(goal_box.y + goal_box.h))
        cv2.rectangle(overlay, (gx1, gy1), (gx2, gy2), (0, 180, 0), 2)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
    cx = int(round(center[0]))
    cy = int(round(center[1]))
    cv2.drawMarker(frame, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
    text = f"z={zoom:.2f} conf={conf:.2f} iou={anchor_iou:.2f}"
    origin = (top_left[0] + 8, max(30, top_left[1] + 24))
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
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
    fps: float,
    frame_size: Tuple[int, int],
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
        compare_writer = cv2.VideoWriter(
            str(compare_path), fourcc, fps, (width * 2, height)
        )
        if not compare_writer.isOpened():
            if writer is not None:
                writer.release()
            cap.release()
            raise SystemExit(f"Failed to open compare writer: {compare_path}")
    for idx, result in enumerate(results):
        ok, frame = cap.read()
        if not ok:
            break
        crop = (result.x, result.y, result.width, result.height)
        overlay = frame.copy()
        draw_preview(
            overlay,
            crop,
            result.center,
            result.zoom,
            result.conf,
            result.goal_box,
            result.anchor_iou,
        )
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
    parser.add_argument(
        "--config",
        type=Path,
        help="Reserved for compatibility; current pipeline is driven by CLI flags",
    )
    parser.add_argument("--lead", type=int, default=6, help="Frames to lead/predict center")
    parser.add_argument(
        "--lead_ms",
        type=int,
        default=180,
        help="Predictive lead from optical flow (ms)",
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
    parser.add_argument("--slew_z", type=float, default=0.06, help="Max zoom change per frame")
    parser.add_argument("--padx", type=float, default=0.22, help="Horizontal padding around action")
    parser.add_argument("--pady", type=float, default=0.18, help="Vertical padding around action")
    parser.add_argument(
        "--zoom_min", type=float, default=1.08, help="Minimum zoom (crop scale denominator)"
    )
    parser.add_argument(
        "--zoom_max", type=float, default=2.40, help="Maximum zoom (tighter crop)"
    )
    parser.add_argument(
        "--zoom_k",
        type=float,
        default=0.85,
        help="Crowding→zoom responsiveness gain",
    )
    parser.add_argument(
        "--zoom_asym",
        type=lambda s: parse_vector(s, 2),
        default=(0.75, 0.35),
        help="Asymmetric damping for zoom out/in",
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
        help="Optional boxcar window (odd) applied before EMA; 0 disables",
    )
    parser.add_argument(
        "--chaos_thresh",
        type=float,
        default=0.18,
        help="Scene motion magnitude threshold triggering defensive zoom",
    )
    parser.add_argument(
        "--hold_frames",
        type=int,
        default=8,
        help="Hold last command this many frames after regaining lock",
    )
    parser.add_argument("--conf_floor", type=float, default=0.15, help="Confidence floor")
    parser.add_argument(
        "--flow_thresh",
        type=float,
        default=0.18,
        help="Motion mask threshold after normalization",
    )
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
    lead_ms = max(0, int(args.lead_ms))
    lead_factor = fps * (lead_ms / 1000.0)
    deadband_scalar = float(args.deadband)
    deadband_xy = tuple(float(v) for v in args.deadband_xy)
    deadband_vec = (
        max(deadband_scalar, deadband_xy[0]),
        max(deadband_scalar, deadband_xy[1]),
    )
    slew_xy = tuple(float(v) for v in args.slew_xy)
    slew_z = float(args.slew_z)
    zoom_gain = float(args.zoom_k)
    zoom_asym_out = float(args.zoom_asym[0])
    zoom_asym_in = float(args.zoom_asym[1])
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
    anchor_weight = float(args.anchor_weight)
    anchor_iou_min = float(args.anchor_iou_min)
    chaos_thresh = float(args.chaos_thresh)

    goal_tracker: Optional[GoalTracker] = None
    if args.roi == "goal":
        goal_tracker = GoalTracker(
            width,
            height,
            fps,
            args.goal_side,
            expansion=0.07,
        )
        goal_tracker.observe(0, frame, None)

    results: List[FrameResult] = []
    commanded_center = np.array([width / 2.0, height / 2.0], dtype=np.float64)
    ema_center = commanded_center.copy()
    raw_history: Deque[np.ndarray] = deque(maxlen=smooth_win if smooth_win > 0 else 1)
    raw_history.append(commanded_center.copy())
    z_cmd = zoom_min
    hold_counter = 0
    was_locked = False

    w0, h0, x0, y0, adjusted_center = compute_crop_dimensions(
        commanded_center, z_cmd, args.profile, padx, pady, (width, height)
    )
    commanded_center = adjusted_center
    ema_center = adjusted_center.copy()
    if goal_tracker and goal_tracker.get_box() is not None:
        gb = goal_tracker.get_box()
        initial_goal_box = GoalBox(gb.x, gb.y, gb.w, gb.h)
    else:
        initial_goal_box = None
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
            flow_mag=0.0,
            goal_box=initial_goal_box,
            anchor_iou=0.0,
        )
    )

    frame_idx = 1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow, norm, centroid, spread, motion_conf, motion_center, motion_strength = estimator.compute(
            prev_gray, gray, commanded_center
        )
        prev_gray = gray

        if goal_tracker is not None:
            goal_tracker.observe(frame_idx, frame, motion_center)
            goal_tracker.track(flow)
            gb = goal_tracker.get_box()
            goal_box = GoalBox(gb.x, gb.y, gb.w, gb.h) if gb is not None else None
        else:
            goal_box = None

        if motion_center is None and centroid is not None:
            motion_center = centroid
        if motion_center is None:
            motion_center = commanded_center.copy()
        lead_region_radius = max(6, int(0.05 * min(width, height)))
        cx = int(np.clip(round(motion_center[0]), 0, width - 1))
        cy = int(np.clip(round(motion_center[1]), 0, height - 1))
        x1 = max(0, cx - lead_region_radius)
        y1 = max(0, cy - lead_region_radius)
        x2 = min(width, cx + lead_region_radius)
        y2 = min(height, cy + lead_region_radius)
        if x2 <= x1 or y2 <= y1:
            mean_flow = np.zeros(2, dtype=np.float64)
        else:
            patch_flow = flow[y1:y2, x1:x2]
            patch_norm = norm[y1:y2, x1:x2]
            weights = patch_norm + 1e-3
            weighted = np.dstack((patch_flow[..., 0] * weights, patch_flow[..., 1] * weights))
            denom = float(weights.sum())
            if denom <= 1e-6:
                mean_flow = np.zeros(2, dtype=np.float64)
            else:
                mean_flow = np.array(
                    [float(weighted[..., 0].sum() / denom), float(weighted[..., 1].sum() / denom)],
                    dtype=np.float64,
                )
        lead_vec = mean_flow * lead_factor
        raw_center = np.array(motion_center, dtype=np.float64) + lead_vec
        raw_center = np.clip(raw_center, [0.0, 0.0], [width - 1.0, height - 1.0])
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
        predicted = apply_deadband_xy(predicted, commanded_center, deadband_vec)

        crop_size = compute_crop_size(z_cmd, args.profile, padx, pady, (width, height))
        half_w = crop_size[0] / 2.0
        half_h = crop_size[1] / 2.0
        provisional = (
            predicted[0] - half_w,
            predicted[1] - half_h,
            predicted[0] + half_w,
            predicted[1] + half_h,
        )
        anchor_iou = 0.0
        anchored_center = predicted.copy()
        if goal_box is not None:
            gx1, gy1, gx2, gy2 = goal_box.bounds
            anchor_iou = compute_iou(
                (
                    max(0.0, provisional[0]),
                    max(0.0, provisional[1]),
                    min(width, provisional[2]),
                    min(height, provisional[3]),
                ),
                (gx1, gy1, gx2, gy2),
            )
            weight = anchor_weight if anchor_iou >= anchor_iou_min else min(1.0, anchor_weight * 1.8)
            goal_center = np.array(goal_box.center, dtype=np.float64)
            anchored_center = (1.0 - weight) * predicted + weight * goal_center
        anchored_center = clamp_center(anchored_center, crop_size, (width, height))

        tightness = 0.0
        zoom_target = z_cmd
        chaos = motion_strength > chaos_thresh
        if goal_box is not None or not chaos:
            tightness = float(np.clip(1.0 - estimate_extent(norm, anchored_center, (width, height)), 0.0, 1.0))
            tightness = float(np.clip(tightness ** zoom_gain, 0.0, 1.0))
            zoom_target = zoom_min + (zoom_max - zoom_min) * tightness
        if chaos:
            zoom_target = zoom_min

        current_locked = (
            goal_box is not None
            and anchor_iou >= anchor_iou_min
            and motion_conf >= conf_floor
        )
        if current_locked and not was_locked:
            hold_counter = hold_frames
        was_locked = current_locked

        if hold_counter > 0:
            hold_counter -= 1
            target_center = commanded_center.copy()
            zoom_target = z_cmd
        else:
            target_center = anchored_center
            target_center = apply_slew(commanded_center, target_center, slew_xy)
        dz = zoom_target - z_cmd
        if dz < 0:
            dz = max(dz, -slew_z * zoom_asym_out)
        else:
            dz = min(dz, slew_z * zoom_asym_in)
        z_next = float(np.clip(z_cmd + dz, zoom_min, zoom_max))

        w, h, x, y, adjusted_center = compute_crop_dimensions(
            target_center, z_next, args.profile, padx, pady, (width, height)
        )
        commanded_center = adjusted_center
        ema_center = ema_center * 0.5 + commanded_center * 0.5
        z_cmd = z_next

        anchor_health = 0.0
        if anchor_iou > 0.0 and anchor_iou_min > 0.0:
            anchor_health = float(np.clip(anchor_iou / anchor_iou_min, 0.0, 1.2))
        stability = float(np.clip(1.0 - motion_strength, 0.0, 1.0))
        conf = float(np.clip(0.5 * motion_conf + 0.3 * anchor_health + 0.2 * stability, 0.0, 1.0))

        results.append(
            FrameResult(
                frame=frame_idx,
                center=(commanded_center[0], commanded_center[1]),
                zoom=z_cmd,
                width=w,
                height=h,
                x=x,
                y=y,
                conf=conf,
                crowding=tightness,
                flow_mag=float(np.clip(motion_strength, 0.0, 1.0)),
                goal_box=goal_box,
                anchor_iou=float(anchor_iou),
            )
        )
        frame_idx += 1

    cap.release()
    return results, fps, (width, height), zoom_min, zoom_max


def main() -> None:
    args = parse_args()
    results, fps, frame_size, zoom_min, zoom_max = run_autoframe(args)
    write_csv(args.csv_path, results, fps, frame_size, zoom_min, zoom_max, args)
    if args.preview_path or args.compare_path:
        render_preview(
            args.input_path,
            args.preview_path,
            args.compare_path,
            results,
            fps,
            frame_size,
        )
    print(f"Wrote {len(results)} motion samples to {args.csv_path}")
    if args.preview_path:
        print(f"Preview: {args.preview_path}")
    if args.compare_path:
        print(f"Compare preview: {args.compare_path}")


if __name__ == "__main__":
    main()

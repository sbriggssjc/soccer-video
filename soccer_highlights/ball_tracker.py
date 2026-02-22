"""Utilities for optional YOLO-based ball tracking."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import importlib
import importlib.util

try:  # pragma: no cover - optional dependency
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

import numpy as np


def _resolve_yolo_class():
    spec = importlib.util.find_spec("ultralytics")
    if spec is None:
        return None
    module = importlib.import_module("ultralytics")
    return getattr(module, "YOLO", None)


YOLO_CLASS = _resolve_yolo_class()


@dataclass
class BallTrack:
    """Single frame of ball tracking data."""

    frame: int
    cx: float
    cy: float
    width: float
    height: float
    conf: float
    raw_cx: float
    raw_cy: float
    raw_width: float
    raw_height: float
    raw_conf: float

    @property
    def center(self) -> Tuple[float, float]:
        return (self.cx, self.cy)


class ConstantVelocityFilter:
    """Very small helper that keeps a constant-velocity estimate."""

    def __init__(self, alpha: float = 0.6) -> None:
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self._position: Optional[np.ndarray] = None
        self._velocity = np.zeros(2, dtype=np.float32)
        self._frame_index: Optional[int] = None
        self._last_measurement: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._position = None
        self._velocity.fill(0.0)
        self._frame_index = None
        self._last_measurement = None

    def update(self, measurement: np.ndarray, frame_index: int) -> np.ndarray:
        measurement = measurement.astype(np.float32)
        if self._position is None or self._frame_index is None:
            self._position = measurement.copy()
            self._velocity = np.zeros_like(measurement)
            self._frame_index = int(frame_index)
            self._last_measurement = measurement.copy()
            return self._position.copy()

        dt = max(int(frame_index) - int(self._frame_index), 1)
        predicted = self._position + self._velocity * dt
        residual = measurement - predicted
        gain = float(self.alpha)
        self._position = predicted + gain * residual
        if self._last_measurement is None:
            measured_velocity = np.zeros_like(measurement)
        else:
            measured_velocity = (measurement - self._last_measurement) / float(dt)
        self._velocity = (1.0 - gain) * self._velocity + gain * measured_velocity
        self._frame_index = int(frame_index)
        self._last_measurement = measurement.copy()
        return self._position.copy()

    def predict(self, frame_index: int) -> Optional[np.ndarray]:
        if self._position is None or self._frame_index is None:
            return None
        dt = int(frame_index) - int(self._frame_index)
        if dt <= 0:
            return self._position.copy()
        return self._position + self._velocity * dt


class BallTracker:
    """Tiny wrapper around YOLO inference with light temporal smoothing."""

    def __init__(
        self,
        weights_path: Optional[Path],
        min_conf: float = 0.35,
        iou: float = 0.45,
        device: Optional[str] = None,
        input_size: int = 640,
        smooth_alpha: float = 0.25,
        max_gap: int = 12,
    ) -> None:
        self.weights_path = Path(weights_path) if weights_path else None
        self.min_conf = float(np.clip(min_conf, 0.0, 1.0))
        self.iou = float(np.clip(iou, 0.0, 1.0))
        self.device = device
        self.input_size = int(max(96, input_size))
        self.smooth_alpha = float(np.clip(smooth_alpha, 0.0, 1.0))
        self.max_gap = int(max(0, max_gap))
        self._model = None
        self._failed = False
        self.failure_reason: Optional[str] = None
        self._ball_ids: List[int] = []
        self._velocity_filter = ConstantVelocityFilter(alpha=0.65)
        self._ema_center: Optional[np.ndarray] = None
        self._ema_size: Optional[np.ndarray] = None
        self._ema_conf: Optional[float] = None
        self._last_frame: Optional[int] = None

        if YOLO_CLASS is None:
            self._failed = True
            self.failure_reason = "Ultralytics YOLO not available"
        elif self.weights_path and not self.weights_path.exists():
            self._failed = True
            self.failure_reason = f"Ball weights not found: {self.weights_path}"

    @property
    def is_ready(self) -> bool:
        return not self._failed

    def _ensure_model(self) -> None:
        if self._model is not None or self._failed:
            return
        if YOLO_CLASS is None:
            self._failed = True
            if self.failure_reason is None:
                self.failure_reason = "Ultralytics YOLO not available"
            return
        weights = self.weights_path.as_posix() if self.weights_path else "yolov8n.pt"
        try:
            self._model = YOLO_CLASS(weights)
        except Exception as exc:  # pragma: no cover - defensive guard for runtime errors
            self._failed = True
            self.failure_reason = f"Failed to initialise ball YOLO: {exc}"
            self._model = None
            return

        names = getattr(self._model, "names", {}) or {}
        if isinstance(names, dict):
            for idx, name in names.items():
                if isinstance(name, str) and "ball" in name.lower():
                    self._ball_ids.append(int(idx))
        if not self._ball_ids:
            # Sports ball class id in COCO
            self._ball_ids = [32]

    def _reset_if_stale(self, frame_index: int) -> None:
        if self._last_frame is None:
            return
        if frame_index - self._last_frame > self.max_gap:
            self._velocity_filter.reset()
            self._ema_center = None
            self._ema_size = None
            self._ema_conf = None

    def _update_ema(self, value: np.ndarray, store: Optional[np.ndarray]) -> np.ndarray:
        if store is None:
            return value.copy()
        alpha = float(self.smooth_alpha)
        return alpha * value + (1.0 - alpha) * store

    def update(self, frame_index: int, frame: np.ndarray) -> Optional[BallTrack]:
        self._reset_if_stale(frame_index)
        self._ensure_model()
        if self._failed or self._model is None:
            return None

        height, width = frame.shape[:2]
        params = {
            "conf": max(0.05, min(self.min_conf, 0.99)),
            "iou": self.iou,
            "imgsz": self.input_size,
            "verbose": False,
        }
        if self.device:
            params["device"] = self.device

        result = self._model.predict(frame, **params)
        if not result:
            self._register_miss(frame_index)
            return None

        boxes = getattr(result[0], "boxes", None)
        if boxes is None:
            self._register_miss(frame_index)
            return None

        cls = getattr(boxes, "cls", None)
        conf = getattr(boxes, "conf", None)
        xyxy = getattr(boxes, "xyxy", None)
        if cls is None or conf is None or xyxy is None:
            self._register_miss(frame_index)
            return None

        cls = cls.detach().cpu().numpy() if hasattr(cls, "detach") else np.asarray(cls)
        conf = conf.detach().cpu().numpy() if hasattr(conf, "detach") else np.asarray(conf)
        xyxy = xyxy.detach().cpu().numpy() if hasattr(xyxy, "detach") else np.asarray(xyxy)

        # Collect all valid ball candidates before selecting
        _candidates: list[tuple[int, float, float, float, np.ndarray]] = []
        for idx, (cls_id, c_score, box) in enumerate(zip(cls, conf, xyxy)):
            if self._ball_ids and int(cls_id) not in self._ball_ids:
                continue
            c_val = float(c_score)
            if c_val < self.min_conf:
                continue
            x1, y1, x2, y2 = box.astype(float)
            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            if bw <= 1.0 or bh <= 1.0:
                continue
            if bw > width * 0.6 or bh > height * 0.6:
                continue
            cx = x1 + bw * 0.5
            cy = y1 + bh * 0.5
            if not (0.0 <= cx <= width and 0.0 <= cy <= height):
                continue
            _candidates.append((idx, c_val, cx, cy,
                                np.array([x1, y1, x2, y2], dtype=np.float32)))

        if not _candidates:
            self._register_miss(frame_index)
            return None

        # Spatial continuity: when we have a predicted position, prefer
        # detections near it.  A far-away detection (background ball, spare
        # ball on sideline) must have substantially higher confidence to
        # override spatial proximity.  Penalty: 0.001 conf per pixel of
        # distance from predicted position (~0.2 penalty at 200px,
        # ~0.5 at 500px).
        #
        # Depth bias: prefer lower-in-frame detections (closer to camera
        # in a typical soccer broadcast).  A ball at the bottom of the
        # frame gets up to +0.15 confidence bonus vs one at the very top.
        # This prevents the tracker from locking onto a spare ball or
        # background-pitch ball that happens to have similar YOLO confidence.
        _DIST_PENALTY_SCALE = 0.001
        _DEPTH_BIAS = 0.15  # max bonus for bottom-of-frame detection
        predicted = self._velocity_filter.predict(frame_index)
        if predicted is not None and len(_candidates) > 1:
            pred_x, pred_y = float(predicted[0]), float(predicted[1])
            best_score = float("-inf")
            best_cand = _candidates[0]
            for cand in _candidates:
                _, c_val, cx_c, cy_c, _ = cand
                dist = math.hypot(cx_c - pred_x, cy_c - pred_y)
                depth_bonus = (cy_c / max(height, 1.0)) * _DEPTH_BIAS
                score = c_val - dist * _DIST_PENALTY_SCALE + depth_bonus
                if score > best_score:
                    best_score = score
                    best_cand = cand
            best_idx, best_conf, _, _, best_box = best_cand
        else:
            # Single candidate or no prediction: pick highest confidence,
            # with depth bias to prefer closer (lower-in-frame) detections.
            best_cand = max(_candidates, key=lambda c: c[1] + (c[3] / max(height, 1.0)) * _DEPTH_BIAS)
            best_idx, best_conf, _, _, best_box = best_cand

        x1, y1, x2, y2 = best_box
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        cx = x1 + bw * 0.5
        cy = y1 + bh * 0.5

        center = np.array([cx, cy], dtype=np.float32)
        filtered = self._velocity_filter.update(center, frame_index)
        if self._ema_center is None:
            self._ema_center = filtered.copy()
        else:
            self._ema_center = self._update_ema(filtered, self._ema_center)

        size = np.array([bw, bh], dtype=np.float32)
        if self._ema_size is None:
            self._ema_size = size.copy()
        else:
            self._ema_size = self._update_ema(size, self._ema_size)

        conf_val = float(best_conf)
        if self._ema_conf is None:
            self._ema_conf = conf_val
        else:
            alpha = float(self.smooth_alpha)
            self._ema_conf = alpha * conf_val + (1.0 - alpha) * self._ema_conf

        self._last_frame = int(frame_index)

        return BallTrack(
            frame=int(frame_index),
            cx=float(self._ema_center[0]),
            cy=float(self._ema_center[1]),
            width=float(self._ema_size[0]),
            height=float(self._ema_size[1]),
            conf=float(self._ema_conf),
            raw_cx=float(center[0]),
            raw_cy=float(center[1]),
            raw_width=float(size[0]),
            raw_height=float(size[1]),
            raw_conf=float(conf_val),
        )

    def _register_miss(self, frame_index: int) -> None:
        if self._last_frame is None:
            return
        if frame_index - self._last_frame > self.max_gap:
            self._velocity_filter.reset()
            self._ema_center = None
            self._ema_size = None
            self._ema_conf = None

    def track(
        self, video_path: Path | str, out_csv: Optional[Path] = None
    ) -> Iterator[Optional[BallTrack]]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video for ball tracking: {video_path}")

        csv_file = None
        writer = None
        frame_idx = 0
        try:
            if out_csv is not None:
                out_csv = Path(out_csv)
                out_csv.parent.mkdir(parents=True, exist_ok=True)
                csv_file = out_csv.open("w", newline="", encoding="utf-8")
                writer = csv.writer(csv_file)
                writer.writerow(["frame", "cx", "cy", "w", "h", "conf", "raw_cx", "raw_cy", "raw_w", "raw_h", "raw_conf"])
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                track = self.update(frame_idx, frame)
                if writer is not None:
                    if track is None:
                        writer.writerow([frame_idx, "", "", "", "", "", "", "", "", "", ""])
                    else:
                        writer.writerow(
                            [
                                track.frame,
                                track.cx,
                                track.cy,
                                track.width,
                                track.height,
                                track.conf,
                                track.raw_cx,
                                track.raw_cy,
                                track.raw_width,
                                track.raw_height,
                                track.raw_conf,
                            ]
                        )
                yield track
                frame_idx += 1
        finally:
            cap.release()
            if csv_file is not None:
                csv_file.close()


__all__ = ["BallTracker", "BallTrack"]


"""Color utilities for pitch/kit masking and auto calibration."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    import cv2  # type: ignore
    _cv2_error = cv2.error  # type: ignore[attr-defined]
except Exception as exc:  # pragma: no cover
    cv2 = None  # type: ignore
    _cv2_error = RuntimeError
    _cv2_import_error = exc
else:
    _cv2_import_error = None
from ._loguru import logger

from .config import ColorsConfig, HSVRange


@dataclass
class HSVTolerance:
    h: float = 12.0
    s: float = 60.0
    v: float = 60.0


DEFAULT_TOLERANCE = HSVTolerance()


def hsv_mask(frame_bgr: np.ndarray, center: HSVRange, tol: HSVTolerance = DEFAULT_TOLERANCE) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    lower = np.array([
        max(0.0, center.h - tol.h),
        max(0.0, center.s - tol.s),
        max(0.0, center.v - tol.v),
    ])
    upper = np.array([
        min(180.0, center.h + tol.h),
        min(255.0, center.s + tol.s),
        min(255.0, center.v + tol.v),
    ])
    mask = cv2.inRange(hsv, lower, upper)
    return mask


def calibrate_colors(video_path: str, config: ColorsConfig, sample_frames: int = 20) -> ColorsConfig:
    logger.info("Calibrating HSV ranges from %s", video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("Failed to open %s for calibration", video_path)
        return config

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        logger.warning('Video has no frames; skipping calibration')
        return config
    indices = sorted(random.sample(range(total_frames), min(sample_frames, total_frames)))

    samples_pitch = []
    samples_team = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        center_region = hsv[h // 3 : 2 * h // 3, w // 4 : 3 * w // 4]
        flat = center_region.reshape(-1, 3).astype(np.float32)
        if flat.size == 0:
            continue
        k = min(3, len(flat))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        compactness, labels, centers = cv2.kmeans(flat, k, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
        centers = centers.astype(np.float32)
        samples_pitch.append(np.median(centers, axis=0))

        # assume jerseys in top half of frame
        jersey_region = hsv[: h // 2, :]
        jr = jersey_region.reshape(-1, 3).astype(np.float32)
        if jr.size:
            k_j = min(3, len(jr))
            _, _, centers_j = cv2.kmeans(jr, k_j, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
            samples_team.append(np.median(centers_j, axis=0))

    cap.release()

    if samples_pitch:
        pitch_center = np.mean(samples_pitch, axis=0)
        config.pitch_hsv = HSVRange(h=float(pitch_center[0]), s=float(pitch_center[1]), v=float(pitch_center[2]))
    if samples_team:
        team_center = np.mean(samples_team, axis=0)
        config.team_primary = HSVRange(h=float(team_center[0]), s=float(team_center[1]), v=float(team_center[2]))

    logger.info(
        "Calibrated pitch HSV=(%.1f, %.1f, %.1f) team HSV=(%.1f, %.1f, %.1f)",
        config.pitch_hsv.h,
        config.pitch_hsv.s,
        config.pitch_hsv.v,
        config.team_primary.h,
        config.team_primary.s,
        config.team_primary.v,
    )
    return config


def mask_ratio(mask: np.ndarray) -> float:
    return float(mask.mean() / 255.0)


def combine_masks(masks: Iterable[np.ndarray]) -> np.ndarray:
    result = None
    for mask in masks:
        if result is None:
            result = mask.astype(np.uint8)
        else:
            result = cv2.bitwise_or(result, mask.astype(np.uint8))
    return result if result is not None else np.zeros((1, 1), dtype=np.uint8)


def pitch_mask(frame_bgr: np.ndarray, config: ColorsConfig) -> np.ndarray:
    return hsv_mask(frame_bgr, config.pitch_hsv)


def team_mask(frame_bgr: np.ndarray, config: ColorsConfig) -> np.ndarray:
    return hsv_mask(frame_bgr, config.team_primary)

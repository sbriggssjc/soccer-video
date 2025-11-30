"""Stage 1: Build rich telemetry from a raw clip.

Input: 1920×1080 source clip (or any 16:9) and outputs JSONL telemetry with
ball + carrier hints. Each line matches the schema::

    {
      "t": 3.125,
      "f": 75,
      "ball_x": 1570.3,
      "ball_y": 360.2,
      "ball_conf": 0.91,
      "carrier_x": 1505.2,
      "carrier_y": 520.7,
      "carrier_conf": 0.88,
      "source": "ball",         # or "carrier" / "players"
      "is_valid": true
    }

The builder favors direct ball detections, then short-term carrier holds, and
finally motion-based fallbacks when the ball disappears. Downstream consumers
can safely fall back to reactive follow when ``is_valid`` is ``false`` or
confidence drops.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

from tools.ball_telemetry import telemetry_path_for_video


@dataclass
class Detection:
    x: float
    y: float
    conf: float
    source: str


def _get_video_fps(cap: cv2.VideoCapture) -> float:
    fps_val = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    return fps_val if math.isfinite(fps_val) and fps_val > 0 else 30.0


def _white_ball_detector(frame: np.ndarray, width: int, height: int) -> Detection | None:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 180], dtype=np.uint8)
    upper_white = np.array([180, 80, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_white, upper_white)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    frame_area = max(1, width * height)
    best: tuple[float, float, float] | None = None
    for c in contours:
        area = cv2.contourArea(c)
        if area <= 2 or area > frame_area * 0.01:
            continue
        peri = cv2.arcLength(c, True)
        circularity = 0.0 if peri <= 0 else 4 * math.pi * area / (peri * peri)
        M = cv2.moments(c)
        if M["m00"] <= 0:
            continue
        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])
        score = circularity * area
        if best is None or score > best[2]:
            best = (cx, cy, circularity)

    if best is None:
        return None

    cx, cy, quality = best
    cx = max(0.0, min(width, cx))
    cy = max(0.0, min(height, cy))
    conf = max(0.0, min(1.0, quality))
    return Detection(x=cx, y=cy, conf=conf, source="ball")


def _motion_centroid(
    prev_gray: np.ndarray,
    gray: np.ndarray,
    frame_shape: tuple[int, int],
    *,
    hint: tuple[float, float] | None = None,
) -> Detection | None:
    diff = cv2.absdiff(gray, prev_gray)
    blur = cv2.GaussianBlur(diff, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, np.ones((3, 3), np.uint8), iterations=2)

    if hint is not None:
        mask = np.zeros_like(thresh)
        cx, cy = hint
        radius = 120
        cv2.circle(mask, (int(round(cx)), int(round(cy))), radius, 255, thickness=-1)
        thresh = cv2.bitwise_and(thresh, thresh, mask=mask)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h, w = frame_shape
    frame_area = max(1, w * h)
    best_area = 0.0
    best_pt: Optional[tuple[float, float]] = None
    for c in contours:
        area = float(cv2.contourArea(c))
        if area <= 8:
            continue
        M = cv2.moments(c)
        if M["m00"] <= 0:
            continue
        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])
        if area > best_area:
            best_area = area
            best_pt = (cx, cy)

    if best_pt is None:
        return None

    cx, cy = best_pt
    cx = max(0.0, min(w, cx))
    cy = max(0.0, min(h, cy))
    conf = max(0.05, min(0.9, best_area / float(frame_area)))
    return Detection(x=cx, y=cy, conf=conf, source="carrier")


def _iter_frames(cap: cv2.VideoCapture) -> Iterable[tuple[int, np.ndarray]]:
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        yield idx, frame
        idx += 1


def build_telemetry(video_path: Path, out_path: Path) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = _get_video_fps(cap)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[TELEMETRY] Building ball/carrier telemetry for {video_path} ({total_frames} frames @ {fps:.2f}fps)")

    prev_gray: np.ndarray | None = None
    last_ball: tuple[float, float, int, float] | None = None
    carried_frames = 0
    valid_rows = 0

    processed_frames = 0

    with out_path.open("w", encoding="utf-8") as handle:
        for frame_idx, frame in _iter_frames(cap):
            t_val = frame_idx / fps if fps > 0 else 0.0
            detection = _white_ball_detector(frame, width, height)

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion_hint = (last_ball[0], last_ball[1]) if last_ball else None
            motion_det = None
            if prev_gray is not None:
                motion_det = _motion_centroid(prev_gray, frame_gray, (height, width), hint=motion_hint)

            source = "players"
            ball_x = ball_y = float("nan")
            carrier_x = carrier_y = float("nan")
            ball_conf = carrier_conf = 0.0
            is_valid = False

            if detection:
                ball_x, ball_y, ball_conf = detection.x, detection.y, detection.conf
                carrier_x, carrier_y, carrier_conf = ball_x, ball_y, max(ball_conf, 0.6)
                source = detection.source
                is_valid = True
                last_ball = (ball_x, ball_y, frame_idx, ball_conf)
                carried_frames = 0
            elif last_ball and frame_idx - last_ball[2] <= 12:
                decay = 0.85 ** float(frame_idx - last_ball[2])
                ball_x, ball_y = last_ball[0], last_ball[1]
                carrier_x, carrier_y = ball_x, ball_y
                ball_conf = max(0.0, min(1.0, last_ball[3] * decay))
                carrier_conf = max(0.2, ball_conf * 0.8)
                source = "carrier"
                is_valid = True
                carried_frames += 1
            elif motion_det:
                carrier_x, carrier_y, carrier_conf = motion_det.x, motion_det.y, motion_det.conf
                ball_x, ball_y = carrier_x, carrier_y
                ball_conf = 0.0
                source = motion_det.source
                is_valid = True
            else:
                carrier_x = width / 2.0
                carrier_y = height * 0.55
                carrier_conf = 0.1
                ball_x, ball_y = carrier_x, carrier_y
                source = "players"
                is_valid = False

            row = {
                "t": round(t_val, 3),
                "f": int(frame_idx),
                "ball_x": None if not math.isfinite(ball_x) else float(ball_x),
                "ball_y": None if not math.isfinite(ball_y) else float(ball_y),
                "ball_conf": float(ball_conf),
                "carrier_x": None if not math.isfinite(carrier_x) else float(carrier_x),
                "carrier_y": None if not math.isfinite(carrier_y) else float(carrier_y),
                "carrier_conf": float(carrier_conf),
                "source": source,
                "is_valid": bool(is_valid),
            }
            handle.write(json.dumps(row) + "\n")

            if is_valid:
                valid_rows += 1

            prev_gray = frame_gray
            processed_frames += 1

    cap.release()
    frame_den = total_frames if total_frames > 0 else processed_frames
    frame_den = max(1, frame_den)
    coverage = 100.0 * valid_rows / frame_den
    print(
        f"[TELEMETRY] Wrote {valid_rows}/{frame_den} usable rows "
        f"({coverage:.1f}% coverage) -> {out_path}"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build ball/carrier telemetry JSONL from a raw clip")
    ap.add_argument("--video", required=True, help="Input MP4 path")
    ap.add_argument(
        "--out",
        help="Output telemetry path (default: out/telemetry/<stem>.ball.jsonl)",
    )
    args = ap.parse_args(argv)

    video_path = Path(args.video).expanduser()
    out_path = Path(args.out or telemetry_path_for_video(video_path)).expanduser()
    build_telemetry(video_path, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

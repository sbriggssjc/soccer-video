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
      "is_valid": true,
      "cam_motion": 3.2
    }

The builder favors direct ball detections, then short-term carrier holds, and
finally motion-based fallbacks when the ball disappears.  Camera motion is
estimated each frame via sparse optical-flow + RANSAC affine so that:

  * carrier-hold positions are warped to compensate for camera pans,
  * ball detection runs on a motion-compensated frame (less blur),
  * the motion-centroid fallback uses a stabilized diff (global motion
    subtracted), so it finds the *ball* motion, not the *camera* motion.

Downstream consumers can safely fall back to reactive follow when
``is_valid`` is ``false`` or confidence drops.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np

from tools.ball_telemetry import telemetry_path_for_video

# --- Velocity-predicted carrier hold ------------------------------------------

# When detection fails, instead of holding a static position we advance the
# ball along its recent velocity vector.  This keeps the virtual camera
# tracking the ball's likely trajectory during detection gaps (especially
# camera pans where motion blur kills the detector for several frames).

_VEL_EMA_ALPHA = 0.35          # EMA weight for velocity update (higher = more reactive)
_VEL_MAX_PX_PER_FRAME = 30.0   # cap per-frame prediction to avoid runaway drift
_VEL_DECAY_PER_FRAME = 0.92    # velocity decays each predicted frame so we coast to a stop
_VEL_MIN_DETECTIONS = 2        # need at least 2 detections to estimate velocity


@dataclass
class _VelocityState:
    """Running estimate of ball velocity in raw-pixel space."""

    vx: float = 0.0
    vy: float = 0.0
    prev_x: float = 0.0
    prev_y: float = 0.0
    prev_frame: int = -1
    n_detections: int = 0

    def update(self, x: float, y: float, frame_idx: int) -> None:
        """Feed a confirmed detection to update the velocity estimate."""
        if self.prev_frame >= 0 and frame_idx > self.prev_frame:
            dt = frame_idx - self.prev_frame
            raw_vx = (x - self.prev_x) / dt
            raw_vy = (y - self.prev_y) / dt
            self.vx += _VEL_EMA_ALPHA * (raw_vx - self.vx)
            self.vy += _VEL_EMA_ALPHA * (raw_vy - self.vy)
        self.prev_x = x
        self.prev_y = y
        self.prev_frame = frame_idx
        self.n_detections += 1

    def predict(
        self, x: float, y: float, frames_ahead: int, width: int, height: int
    ) -> Tuple[float, float]:
        """Extrapolate (x, y) forward using the current velocity estimate."""
        if self.n_detections < _VEL_MIN_DETECTIONS:
            return x, y
        vx = self.vx
        vy = self.vy
        speed = math.hypot(vx, vy)
        if speed > _VEL_MAX_PX_PER_FRAME:
            scale = _VEL_MAX_PX_PER_FRAME / speed
            vx *= scale
            vy *= scale
        px, py = x, y
        for _ in range(frames_ahead):
            px += vx
            py += vy
            vx *= _VEL_DECAY_PER_FRAME
            vy *= _VEL_DECAY_PER_FRAME
        px = max(0.0, min(float(width), px))
        py = max(0.0, min(float(height), py))
        return px, py


# --- Camera-motion estimation ------------------------------------------------

# Carrier hold is normally 12 frames; during fast camera motion we extend to 24.
_CARRIER_HOLD_BASE = 12
_CARRIER_HOLD_EXTENDED = 24
# Camera motion (in pixels) above which we consider the camera "panning fast".
_CAM_MOTION_FAST_THRESH = 6.0


def _estimate_camera_motion(
    prev_gray: np.ndarray, cur_gray: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Return (2×3 affine warp matrix, motion magnitude in px).

    The affine maps *current* frame pixels back to *previous* frame space,
    effectively undoing the camera movement.  ``motion_mag`` is the
    translational component magnitude (useful as a scalar "how much did the
    camera move" signal).
    """
    p0 = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=800,
        qualityLevel=0.01,
        minDistance=10,
        blockSize=7,
    )
    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    if p0 is None or len(p0) < 12:
        return identity, 0.0

    p1, st, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, cur_gray, p0, None, winSize=(21, 21), maxLevel=3
    )
    if p1 is None or st is None:
        return identity, 0.0

    good_prev = p0[st == 1].reshape(-1, 2)
    good_cur = p1[st == 1].reshape(-1, 2)
    if len(good_prev) < 12:
        return identity, 0.0

    M, _inliers = cv2.estimateAffinePartial2D(
        good_cur, good_prev, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )
    if M is None:
        return identity, 0.0

    # Translation component = the displacement of the frame centre.
    tx, ty = float(M[0, 2]), float(M[1, 2])
    motion_mag = math.hypot(tx, ty)
    return M, motion_mag


def _warp_point(
    x: float, y: float, M: np.ndarray, width: int, height: int
) -> Tuple[float, float]:
    """Apply 2×3 affine *inverse* to a point (prev-frame coords → cur-frame coords)."""
    # M maps cur→prev.  We need prev→cur, so invert.
    try:
        M_inv = cv2.invertAffineTransform(M)
    except cv2.error:
        return x, y
    nx = float(M_inv[0, 0] * x + M_inv[0, 1] * y + M_inv[0, 2])
    ny = float(M_inv[1, 0] * x + M_inv[1, 1] * y + M_inv[1, 2])
    return max(0.0, min(float(width), nx)), max(0.0, min(float(height), ny))


def _stabilize_frame(
    frame: np.ndarray, M: np.ndarray
) -> np.ndarray:
    """Warp *frame* so that it aligns with the previous frame (undo camera motion)."""
    h, w = frame.shape[:2]
    return cv2.warpAffine(
        frame, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


# --- Detection helpers --------------------------------------------------------

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
    """Find the largest motion blob.

    When *prev_gray* and *gray* are already stabilized (camera motion removed),
    the diff isolates object motion (the ball, players) rather than the global
    camera pan.
    """
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
    prev_stab_gray: np.ndarray | None = None
    last_ball: tuple[float, float, int, float] | None = None
    carried_frames = 0
    valid_rows = 0
    vel = _VelocityState()

    processed_frames = 0

    with out_path.open("w", encoding="utf-8") as handle:
        for frame_idx, frame in _iter_frames(cap):
            t_val = frame_idx / fps if fps > 0 else 0.0
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # --- Camera motion estimation ---
            cam_motion_mag = 0.0
            cam_M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
            if prev_gray is not None:
                cam_M, cam_motion_mag = _estimate_camera_motion(prev_gray, frame_gray)

            camera_is_moving = cam_motion_mag >= _CAM_MOTION_FAST_THRESH

            # --- Stabilized frame for detection ---
            # Warp current frame to align with previous frame's coordinate
            # system.  This reduces motion blur artefacts during camera pans
            # and makes the white-ball detector's circularity scoring more
            # reliable.
            if camera_is_moving and prev_gray is not None:
                stab_frame = _stabilize_frame(frame, cam_M)
                stab_gray = cv2.cvtColor(stab_frame, cv2.COLOR_BGR2GRAY)
            else:
                stab_frame = frame
                stab_gray = frame_gray

            # --- Ball detection (try stabilized frame first, raw as fallback) ---
            detection = _white_ball_detector(stab_frame, width, height)
            det_in_stab_space = True
            if detection is None and camera_is_moving:
                # Stabilized frame didn't help; try raw frame too.
                detection = _white_ball_detector(frame, width, height)
                det_in_stab_space = False

            # If detection was in stabilized space, map coords back to raw.
            if detection is not None and det_in_stab_space and camera_is_moving:
                rx, ry = _warp_point(detection.x, detection.y, cam_M, width, height)
                detection = Detection(x=rx, y=ry, conf=detection.conf, source=detection.source)

            # --- Motion centroid (use stabilized diff to subtract camera motion) ---
            motion_hint = (last_ball[0], last_ball[1]) if last_ball else None
            # Warp hint to current frame if camera moved.
            if motion_hint is not None and camera_is_moving:
                motion_hint = _warp_point(
                    motion_hint[0], motion_hint[1], cam_M, width, height
                )

            motion_det = None
            if prev_stab_gray is not None:
                motion_det = _motion_centroid(
                    prev_stab_gray, stab_gray, (height, width), hint=motion_hint
                )
                # Map back to raw pixel space.
                if motion_det is not None and camera_is_moving:
                    rx, ry = _warp_point(
                        motion_det.x, motion_det.y, cam_M, width, height
                    )
                    motion_det = Detection(
                        x=rx, y=ry, conf=motion_det.conf, source=motion_det.source
                    )

            # --- Carrier hold: warp last-known position by camera motion ---
            if last_ball is not None and camera_is_moving:
                wx, wy = _warp_point(
                    last_ball[0], last_ball[1], cam_M, width, height
                )
                last_ball = (wx, wy, last_ball[2], last_ball[3])

            # Dynamic hold limit: extend during camera pans so we bridge the
            # detection gap caused by motion blur.
            max_hold = _CARRIER_HOLD_EXTENDED if camera_is_moving else _CARRIER_HOLD_BASE

            # --- Decision cascade ---
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
                vel.update(ball_x, ball_y, frame_idx)
                last_ball = (ball_x, ball_y, frame_idx, ball_conf)
                carried_frames = 0
            elif last_ball and frame_idx - last_ball[2] <= max_hold:
                gap = frame_idx - last_ball[2]
                decay = 0.85 ** float(gap)
                # Predict position using velocity instead of static hold.
                ball_x, ball_y = vel.predict(
                    last_ball[0], last_ball[1], gap, width, height
                )
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
                "cam_motion": round(cam_motion_mag, 2),
            }
            handle.write(json.dumps(row) + "\n")

            if is_valid:
                valid_rows += 1

            prev_gray = frame_gray
            prev_stab_gray = stab_gray
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

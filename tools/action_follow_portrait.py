#!/usr/bin/env python
import argparse
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def load_telemetry(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    if not rows:
        raise SystemExit(f"[ERROR] No telemetry rows loaded from {path}")
    print(f"[INFO] Loaded {len(rows)} telemetry rows from {path}")
    print(f"[INFO] Sample telemetry keys: {sorted(rows[0].keys())}")
    return rows


def pick_center_from_row(
    row: Dict,
    fallback_center: Optional[Tuple[float, float]] = None,
) -> Optional[Tuple[float, float]]:
    """
    Try multiple key patterns to find an (x, y) center for the action.
    Preference order:
      - action_x/action_y
      - center_x/center_y
      - ball_out_x/ball_out_y
      - ball_x/ball_y
      - x/y
    """
    key_pairs = [
        ("action_x", "action_y"),
        ("center_x", "center_y"),
        ("ball_out_x", "ball_out_y"),
        ("ball_x", "ball_y"),
        ("x", "y"),
    ]

    for kx, ky in key_pairs:
        if kx in row and ky in row:
            try:
                x = float(row[kx])
                y = float(row[ky])
                return (x, y)
            except (TypeError, ValueError):
                continue

    # As a last resort, keep previous center if we have one
    return fallback_center


def smooth_value(prev: float, target: float, alpha: float) -> float:
    """Exponential smoothing: alpha ~ 0.8 means heavy smoothing."""
    return alpha * prev + (1.0 - alpha) * target


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def compute_zoom_option_a(
    center: Tuple[float, float],
    prev_center: Optional[Tuple[float, float]],
    prev_zoom: float,
    src_w: int,
    src_h: int,
) -> float:
    """
    Option A – natural broadcast behavior:
      - Zoom in moderately when action is further from the vertical center
      - Zoom out when motion is fast
      - Smooth zoom changes
    """
    cx, cy = center

    # Distance from vertical middle of frame, normalized (0..~0.5+)
    dist_norm = abs(cy - (src_h / 2.0)) / float(src_h)

    # Base zoom from distance (1.0 = no zoom, up to ~2.4)
    base_zoom = 1.0 + dist_norm * 1.4

    # Motion-based zoom out if the action is moving quickly
    motion_zoom_factor = 1.0
    if prev_center is not None:
        px, py = prev_center
        dx = cx - px
        dy = cy - py
        speed = math.hypot(dx, dy)

        # Tuned for 1920x1080 with ~24fps; adjust if needed
        SPEED_SOFT = 8.0   # start reacting
        SPEED_HARD = 20.0  # strong zoom-out

        if speed > SPEED_SOFT:
            # between 0.8 and 1.0; more speed -> more zoom out
            t = clamp((speed - SPEED_SOFT) / (SPEED_HARD - SPEED_SOFT + 1e-6), 0.0, 1.0)
            motion_zoom_factor = 1.0 - 0.2 * t  # up to 20% zoom out

    zoom_target = base_zoom * motion_zoom_factor

    # Clamp zoom levels
    MIN_ZOOM = 1.0   # don't zoom out beyond full frame
    MAX_ZOOM = 2.8   # moderate zoom-in
    zoom_target = clamp(zoom_target, MIN_ZOOM, MAX_ZOOM)

    # Smooth zoom
    ZOOM_ALPHA = 0.85  # 0.85 = fairly smooth, not too laggy
    zoom = smooth_value(prev_zoom, zoom_target, ZOOM_ALPHA)
    zoom = clamp(zoom, MIN_ZOOM, MAX_ZOOM)

    return zoom


def compute_crop_window(
    center: Tuple[float, float],
    zoom: float,
    src_w: int,
    src_h: int,
    out_w: int,
    out_h: int,
) -> Tuple[int, int, int, int]:
    """
    Compute a portrait-shaped crop (same aspect ratio as out_w/out_h),
    zoomed in by `zoom`, centered on `center`, and clamped to frame bounds.
    """
    out_ar = out_w / float(out_h)

    # Start from full frame and zoom in
    crop_h = src_h / zoom
    crop_w = crop_h * out_ar

    # If we exceed source width, adjust
    if crop_w > src_w:
        crop_w = float(src_w)
        crop_h = crop_w / out_ar

    crop_w = min(crop_w, src_w)
    crop_h = min(crop_h, src_h)

    cx, cy = center
    x0 = cx - crop_w / 2.0
    y0 = cy - crop_h / 2.0

    # Clamp so crop stays within the source frame
    if x0 < 0:
        x0 = 0.0
    if y0 < 0:
        y0 = 0.0
    if x0 + crop_w > src_w:
        x0 = float(src_w) - crop_w
    if y0 + crop_h > src_h:
        y0 = float(src_h) - crop_h

    return int(round(x0)), int(round(y0)), int(round(crop_w)), int(round(crop_h))


def draw_debug_overlay(
    frame: np.ndarray,
    center: Tuple[float, float],
    crop_rect: Tuple[int, int, int, int],
) -> None:
    """Draw debug overlays on the *source* frame: center dot + crop box."""
    h, w = frame.shape[:2]

    # Center dot (yellow)
    cx, cy = center
    cv2.circle(
        frame,
        (int(round(cx)), int(round(cy))),
        6,
        (0, 255, 255),
        thickness=-1,
        lineType=cv2.LINE_AA,
    )

    # Crop rectangle (magenta)
    x0, y0, cw, ch = crop_rect
    cv2.rectangle(
        frame,
        (x0, y0),
        (x0 + cw, y0 + ch),
        (255, 0, 255),
        thickness=2,
        lineType=cv2.LINE_AA,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", required=True, help="Source video path (landscape)")
    parser.add_argument("--telemetry", required=True, help="Telemetry JSONL for action/ball")
    parser.add_argument("--out", required=True, help="Output portrait video path")
    parser.add_argument("--width", type=int, default=1080, help="Output width (default: 1080)")
    parser.add_argument("--height", type=int, default=1920, help="Output height (default: 1920)")
    parser.add_argument("--min-conf", type=float, default=0.0, help="(reserved) minimum confidence")
    parser.add_argument(
        "--debug-overlay",
        action="store_true",
        help="Draw debug overlays (center + crop) onto frames",
    )
    parser.add_argument(
        "--use-planner",
        action="store_true",
        help="(accepted for compatibility; currently unused)",
    )
    args = parser.parse_args()

    clip_path = args.clip
    tele_path = args.telemetry
    out_path = args.out
    out_w = args.width
    out_h = args.height
    debug_overlay = args.debug_overlay

    print(f"[DEBUG] RUNNING FILE: {os.path.abspath(clip_path)}")

    # Load telemetry
    rows = load_telemetry(tele_path)

    # Open video
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        raise SystemExit(f"[ERROR] Could not open video: {clip_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Source video: {src_w}x{src_h} @ {fps:.3f} fps, frames={total_frames}")
    print(f"[INFO] Output portrait: {out_w}x{out_h}")

    # Video writer (mp4v)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))

    if not writer.isOpened():
        raise SystemExit(f"[ERROR] Could not open writer for: {out_path}")

    # State for smoothing
    prev_center: Optional[Tuple[float, float]] = None
    smoothed_center: Optional[Tuple[float, float]] = None
    zoom = 1.0

    CENTER_ALPHA = 0.80  # smoothing for center

    frame_idx = 0
    num_rows = len(rows)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx >= num_rows:
            row = rows[-1]
        else:
            row = rows[frame_idx]

        # Pick current center
        raw_center = pick_center_from_row(row, fallback_center=prev_center)
        if raw_center is None:
            # If we have nothing at all, just keep previous or default to center frame
            if smoothed_center is not None:
                raw_center = smoothed_center
            else:
                raw_center = (src_w / 2.0, src_h / 2.0)

        # Smooth center
        if smoothed_center is None:
            smoothed_center = raw_center
        else:
            cx, cy = smoothed_center
            tx, ty = raw_center
            cx = smooth_value(cx, tx, CENTER_ALPHA)
            cy = smooth_value(cy, ty, CENTER_ALPHA)
            smoothed_center = (cx, cy)

        # Compute zoom (Option A behavior)
        zoom = compute_zoom_option_a(
            center=smoothed_center,
            prev_center=prev_center,
            prev_zoom=zoom,
            src_w=src_w,
            src_h=src_h,
        )

        # Compute crop window, then crop
        x0, y0, cw, ch = compute_crop_window(
            center=smoothed_center,
            zoom=zoom,
            src_w=src_w,
            src_h=src_h,
            out_w=out_w,
            out_h=out_h,
        )

        # Optional debug overlay on the source frame
        if debug_overlay and smoothed_center is not None:
            draw_debug_overlay(frame, smoothed_center, (x0, y0, cw, ch))

        # Crop and resize to portrait
        crop = frame[y0 : y0 + ch, x0 : x0 + cw]
        if crop.size == 0:
            # Fallback: use full frame
            crop = frame

        portrait = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)

        # Write frame
        writer.write(portrait)

        prev_center = raw_center
        frame_idx += 1

        if frame_idx % 50 == 0:
            print(f"[INFO] Processed {frame_idx}/{total_frames} frames...")

    cap.release()
    writer.release()
    print(f"[DONE] Wrote portrait follow clip to: {out_path}")


if __name__ == "__main__":
    main()

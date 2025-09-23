"""Generate smooth polynomial zoom coefficients from motion analysis."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from fit_utils import MotionBounds, ffmpeg_polyfit, normalized_speed, velocity_clamped_ema


def flow_motion_centroid(prev: np.ndarray, curr: np.ndarray) -> tuple[float, float]:
    """Estimate the dominant motion centroid between two grayscale frames."""
    flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mag = cv2.GaussianBlur(mag, (0, 0), 3.0)

    thr = float(np.percentile(mag, 92)) if mag.size else 0.0
    if np.isfinite(thr) and thr > 0.0:
        mag = np.where(mag >= thr, mag, 0.0)

    total = float(mag.sum())
    if total <= 1e-6:
        h, w = mag.shape
        return w / 2.0, h / 2.0

    idx = int(np.argmax(mag))
    y, x = np.unravel_index(idx, mag.shape)
    return float(x), float(y)


def read_motion_paths(video_path: Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Read the video and compute per-frame motion centroids."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open input: {video_path}")

    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise SystemExit("Failed to read first frame")

    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = prev_gray.shape

    centers_x = [width / 2.0]
    centers_y = [height / 2.0]

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y = flow_motion_centroid(prev_gray, gray)
        centers_x.append(x)
        centers_y.append(y)
        prev_gray = gray

    cap.release()

    cx = np.asarray(centers_x, dtype=np.float64)
    cy = np.asarray(centers_y, dtype=np.float64)
    return cx, cy, width, height


def compute_zoom_path(cx: np.ndarray, cy: np.ndarray) -> np.ndarray:
    """Derive a zoom factor path that widens with motion speed."""
    speed = normalized_speed(cx, cy)
    zoom = 2.2 - 0.7 * speed
    return np.clip(zoom, 1.1, 2.2)


def save_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["clip", "profile", "cx_poly", "cy_poly", "z_poly"]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def generate_coefficients(input_path: Path, coeffs_path: Path, profile: str = "tiktok") -> dict:
    cx_raw, cy_raw, width, height = read_motion_paths(input_path)
    if len(cx_raw) < 6:
        raise SystemExit("Too few frames for fitting")

    bounds = MotionBounds(width=float(width), height=float(height))
    cx_smooth = velocity_clamped_ema(cx_raw, vmax=width * 0.012, alpha=0.2)
    cy_smooth = velocity_clamped_ema(cy_raw, vmax=height * 0.012, alpha=0.2)
    cx_smooth, cy_smooth = bounds.clamp(cx_smooth, cy_smooth)

    zoom = compute_zoom_path(cx_smooth, cy_smooth)

    row = {
        "clip": input_path.name,
        "profile": profile,
        "cx_poly": ffmpeg_polyfit(cx_smooth, 2),
        "cy_poly": ffmpeg_polyfit(cy_smooth, 2),
        "z_poly": ffmpeg_polyfit(zoom, 2),
    }
    save_row(coeffs_path, row)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit polynomial zoom/crop coefficients")
    parser.add_argument("input_mp4", type=Path, help="Input MP4 clip")
    parser.add_argument("coeffs_csv", type=Path, help="Destination CSV for coefficients")
    parser.add_argument("profile", nargs="?", default="tiktok", help="Profile label")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    row = generate_coefficients(args.input_mp4, args.coeffs_csv, args.profile)
    print(
        f"Wrote coefficients for {row['clip']} ({row['profile']}):",
        row["cx_poly"],
        row["cy_poly"],
        row["z_poly"],
    )


if __name__ == "__main__":
    main()

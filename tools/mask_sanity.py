"""Utility script to sanity check the field mask for a specific frame.

This mirrors the quick one-off snippet that engineers often run directly in
the interpreter, but wraps it with a slightly more ergonomic CLI and better
error handling.  When the requested frame is available it will dump both the
raw frame and the computed mask to disk and print the white pixel coverage so
you can confirm the mask is roughly covering the playing surface.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def field_mask_bgr(frame: np.ndarray) -> np.ndarray:
    """Generate a binary field mask from a BGR frame."""

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    mask = ((h >= 35) & (h <= 95) & (s >= 40) & (v >= 40)).astype("uint8") * 255
    mask = cv2.medianBlur(mask, 5)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    return mask


def dump_mask(
    video_path: Path, frame_index: int, output_dir: Path
) -> Tuple[Path, Path, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_index >= total_frames:
        raise ValueError(
            f"Requested frame {frame_index} out of bounds. Video only has "
            f"{total_frames} frames."
        )

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(
            f"Failed to read frame {frame_index} from {video_path}."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    stem = f"frame{frame_index:04d}"
    raw_path = output_dir / f"{stem}_raw.png"
    mask_path = output_dir / f"{stem}_mask.png"

    cv2.imwrite(str(raw_path), frame)
    mask = field_mask_bgr(frame)
    cv2.imwrite(str(mask_path), mask)

    coverage = float(mask.mean() / 255.0)

    return raw_path, mask_path, coverage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Path to the stabilized video clip")
    parser.add_argument(
        "--frame",
        type=int,
        default=180,
        help="Frame index to inspect (default: 180)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("out/diag_masks"),
        help="Directory where diagnostic images are written",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_path, mask_path, coverage = dump_mask(args.video, args.frame, args.out_dir)
    print(f"Raw frame written to {raw_path}")
    print(f"Mask written to {mask_path}")
    print(f"Mask coverage: {coverage:.1%}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
action_follow_portrait.py

Use Atomic/telemetry planner crop to generate a smooth portrait follow clip.

Expected telemetry (.jsonl) keys from Atomic planner:
- Either:
    crop_src: [x0, y0, w, h]   # portrait-shaped crop inside 1920x1080
  Or:
    cx, cy, w, h               # center + size in source pixels

We assume source is 1920x1080 landscape and planner crop is already 9:16-ish.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--clip", required=True, help="Input atomic clip (e.g. 1920x1080 mp4)")
    p.add_argument("--telemetry", required=True, help="Atomic telemetry .jsonl with planner crop")
    p.add_argument("--out", required=True, help="Output portrait mp4 path")
    p.add_argument("--width", type=int, default=1080, help="Output width (portrait)")
    p.add_argument("--height", type=int, default=1920, help="Output height (portrait)")
    p.add_argument("--min-conf", type=float, default=0.0, help="(reserved) min confidence for detections")
    p.add_argument(
        "--debug-overlay",
        action="store_true",
        help="Draw simple debug overlay on portrait frames",
    )
    p.add_argument(
        "--use-planner",
        action="store_true",
        help="Use planner crop from telemetry (crop_src / cx,cy,w,h). "
             "If not set, falls back to center crop.",
    )
    return p.parse_args()


def load_telemetry(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        raise SystemExit(f"[ERROR] No valid telemetry rows parsed from {path}")
    print(f"[INFO] Loaded {len(rows)} telemetry rows from {path}")
    return rows


def get_crop_from_row(row: dict, src_w: int, src_h: int):
    """
    Return (x0, y0, w, h) in source pixels.

    Priority:
    1) row['crop_src'] = [x0,y0,w,h]
    2) row['cx'], row['cy'], row['w'], row['h']
    3) fallback = full frame
    """
    if "crop_src" in row and isinstance(row["crop_src"], (list, tuple)) and len(row["crop_src"]) == 4:
        x0, y0, w, h = row["crop_src"]
    elif all(k in row for k in ("cx", "cy", "w", "h")):
        cx = float(row["cx"])
        cy = float(row["cy"])
        w = float(row["w"])
        h = float(row["h"])
        x0 = cx - w / 2.0
        y0 = cy - h / 2.0
    else:
        # Fallback: use full frame
        x0, y0, w, h = 0.0, 0.0, float(src_w), float(src_h)

    # Clamp to source
    x0 = max(0.0, min(x0, src_w - 1.0))
    y0 = max(0.0, min(y0, src_h - 1.0))
    w = max(1.0, min(w, src_w - x0))
    h = max(1.0, min(h, src_h - y0))

    return int(round(x0)), int(round(y0)), int(round(w)), int(round(h))


def center_crop_fallback(src_w: int, src_h: int, out_aspect: float):
    """
    Simple centered portrait crop if planner not requested:
    out_aspect = H_out / W_out (e.g. 1920/1080 ~ 1.777)
    """
    # We want a portrait crop inside landscape 1920x1080
    # Use full height, reduce width to match aspect
    crop_h = src_h
    crop_w = int(round(crop_h / out_aspect))
    crop_w = min(crop_w, src_w)
    x0 = (src_w - crop_w) // 2
    y0 = 0
    return x0, y0, crop_w, crop_h


def main():
    args = parse_args()

    clip_path = Path(args.clip)
    tele_path = Path(args.telemetry)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[DEBUG] RUNNING FILE: {clip_path}")

    telemetry_rows = load_telemetry(tele_path)

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise SystemExit(f"[ERROR] Could not open video {clip_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Source video: {src_w}x{src_h} @ {fps:.3f} fps, frames={frame_count}")
    print(f"[INFO] Output portrait: {args.width}x{args.height}")
    print(f"[INFO] use_planner={args.use_planner}, debug_overlay={args.debug_overlay}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (args.width, args.height))
    if not writer.isOpened():
        raise SystemExit(f"[ERROR] Could not open writer for {out_path}")

    out_aspect = args.height / args.width
    n_rows = len(telemetry_rows)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if args.use_planner:
            # Map frame -> telemetry row (clamp to last)
            row_idx = min(frame_idx, n_rows - 1)
            row = telemetry_rows[row_idx]
            x0, y0, cw, ch = get_crop_from_row(row, src_w, src_h)
        else:
            # Simple center portrait crop
            x0, y0, cw, ch = center_crop_fallback(src_w, src_h, out_aspect)

        # Crop + clamp again just to be safe
        x1 = min(src_w, x0 + cw)
        y1 = min(src_h, y0 + ch)
        if x1 <= x0 or y1 <= y0:
            # Degenerate, fall back to center
            x0, y0, cw, ch = center_crop_fallback(src_w, src_h, out_aspect)
            x1 = min(src_w, x0 + cw)
            y1 = min(src_h, y0 + ch)

        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            # Fallback to full frame if something goes weird
            crop = frame

        # Resize to desired portrait resolution
        portrait = cv2.resize(crop, (args.width, args.height), interpolation=cv2.INTER_AREA)

        if args.debug_overlay:
            # Simple overlay: draw a small dot at center and a border
            h, w = portrait.shape[:2]
            cx = w // 2
            cy = h // 2
            cv2.circle(portrait, (cx, cy), 8, (0, 255, 255), thickness=-1)  # center dot
            cv2.rectangle(portrait, (4, 4), (w - 5, h - 5), (255, 255, 255), thickness=2)

        writer.write(portrait)
        frame_idx += 1
        if frame_idx % 300 == 0:
            print(f"[INFO] Processed {frame_idx}/{frame_count} frames...")

    cap.release()
    writer.release()
    print(f"[DONE] Wrote portrait follow clip to: {out_path}")


if __name__ == "__main__":
    main()

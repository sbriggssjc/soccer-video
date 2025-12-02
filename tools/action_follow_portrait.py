#!/usr/bin/env python
import argparse
import json
import os
import cv2
import numpy as np


def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"[ERROR] No telemetry rows in {path}")
    print(f"[INFO] Loaded {len(rows)} rows from {path}")
    return rows


def get_planner_center(row):
    """Atomic planner center fields: cx, cy."""
    if "cx" in row and "cy" in row:
        try:
            return float(row["cx"]), float(row["cy"])
        except:
            pass
    return None


def get_planner_zoom(row):
    """
    Planner zoom is usually stored as:
      - zoom  (atomic final applied zoom)
      - plan_zoom (planner-proposed zoom)
    We prefer 'zoom' if present.
    """
    if "zoom" in row:
        return float(row["zoom"])
    if "plan_zoom" in row:
        return float(row["plan_zoom"])
    return 1.0


def compute_portrait_crop(center, zoom, src_w, src_h, out_w, out_h):
    """
    Given planner center + planner zoom, compute a portrait crop window.
    """
    cx, cy = center

    # Aspect ratio match the output 9:16 (1080x1920)
    aspect = out_w / float(out_h)

    # Effective crop height from zoom (1.0 = full height)
    crop_h = src_h / zoom
    crop_w = crop_h * aspect

    # If too wide, clamp and re-adjust height
    if crop_w > src_w:
        crop_w = src_w
        crop_h = crop_w / aspect

    # Clamp
    crop_w = min(crop_w, src_w)
    crop_h = min(crop_h, src_h)

    x0 = cx - crop_w / 2
    y0 = cy - crop_h / 2

    # Boundaries
    if x0 < 0:
        x0 = 0
    if y0 < 0:
        y0 = 0
    if x0 + crop_w > src_w:
        x0 = src_w - crop_w
    if y0 + crop_h > src_h:
        y0 = src_h - crop_h

    return int(x0), int(y0), int(crop_w), int(crop_h)


def draw_debug_overlay(frame, center, crop_rect):
    """Draw center point + crop rectangle on the SOURCE frame."""
    cx, cy = center
    x0, y0, w, h = crop_rect

    # Yellow center
    cv2.circle(frame, (int(cx), int(cy)), 8, (0, 255, 255), -1, cv2.LINE_AA)

    # Magenta crop
    cv2.rectangle(
        frame,
        (x0, y0),
        (x0 + w, y0 + h),
        (255, 0, 255),
        2,
        cv2.LINE_AA,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--telemetry", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=1080)
    ap.add_argument("--height", type=int, default=1920)
    ap.add_argument("--debug-overlay", action="store_true")
    ap.add_argument("--use-planner", action="store_true")
    args = ap.parse_args()

    print(f"[DEBUG] RUNNING FILE: {args.clip}")

    rows = load_rows(args.telemetry)

    cap = cv2.VideoCapture(args.clip)
    if not cap.isOpened():
        raise SystemExit(f"[ERROR] Could not open: {args.clip}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Source: {src_w}x{src_h} @ {fps:.2f} fps")
    print(f"[INFO] Output: {args.width}x{args.height}")
    print(f"[INFO] use_planner={args.use_planner}")

    planner = args.use_planner

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    writer = cv2.VideoWriter(args.out, fourcc, fps, (args.width, args.height))

    if not writer.isOpened():
        raise SystemExit(f"[ERROR] Cannot open writer for {args.out}")

    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        row = rows[min(frame_idx, len(rows) - 1)]

        use_follow = False

        if "cx" in row and "cy" in row:
            # This is follow telemetry
            try:
                cx = float(row.get("cx") or 0)
                cy = float(row.get("cy") or 0)
                zoom = float(row.get("zoom") or 1.0)
                use_follow = True
            except (TypeError, ValueError):
                use_follow = False

        if use_follow:
            # Follow telemetry overrides planner
            planner = None
            center = (cx, cy)
        else:
            # ---------------------------------------------------------
            # 1. EXTRACT TRUE PLANNER CAMERA CENTER
            # ---------------------------------------------------------
            center = get_planner_center(row)
            if center is None:
                # Fallback to screen center if exceptions
                center = (src_w / 2, src_h / 2)

            # ---------------------------------------------------------
            # 2. EXTRACT TRUE PLANNER ZOOM
            # ---------------------------------------------------------
            zoom = get_planner_zoom(row)

        if zoom <= 0:
            zoom = 1.0

        cx, cy = center
        cx = max(0, min(cx, src_w))
        cy = max(0, min(cy, src_h))
        zoom = max(0.2, min(zoom, 4.0))
        center = (cx, cy)

        crop_w = int(args.width / zoom)
        crop_h = int(args.height / zoom)

        x1 = int(center[0] - crop_w / 2)
        y1 = int(center[1] - crop_h / 2)
        x1 = max(0, min(x1, src_w - crop_w))
        y1 = max(0, min(y1, src_h - crop_h))

        crop = (x1, y1, crop_w, crop_h)

        # Debug overlay BEFORE cropping (so you can see planner behavior)
        if args.debug_overlay:
            draw_debug_overlay(frame, center, crop)

        crop_frame = frame[y1:y1 + crop_h, x1:x1 + crop_w]
        outframe = cv2.resize(crop_frame, (args.width, args.height), interpolation=cv2.INTER_AREA)

        writer.write(outframe)

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"[INFO] Processed {frame_idx}/{total_frames} frames...")

    cap.release()
    writer.release()
    print(f"[DONE] Wrote portrait clip to: {args.out}")


if __name__ == "__main__":
    main()

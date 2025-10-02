#!/usr/bin/env python3
"""Interactive crop tracker for stabilized camera work."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Callable, Tuple

import cv2


def _find_tracker_factory(name: str) -> Callable[[], "cv2.Tracker"]:
    name = name.lower()

    def lookup(mod, attr):
        return getattr(mod, attr, None)

    if name == "csrt":
        factory = lookup(cv2, "TrackerCSRT_create")
        if factory is None and hasattr(cv2, "legacy"):
            factory = lookup(cv2.legacy, "TrackerCSRT_create")
    elif name == "kcf":
        factory = lookup(cv2, "TrackerKCF_create")
        if factory is None and hasattr(cv2, "legacy"):
            factory = lookup(cv2.legacy, "TrackerKCF_create")
    else:
        factory = None

    if factory is None:
        raise SystemExit(
            "Unsupported tracker '{name}'. Choose from: csrt, kcf".format(name=name)
        )
    return factory


def build_tracker(name: str):
    factory = _find_tracker_factory(name)
    return factory()


def parse_roi(value: str) -> Tuple[int, int, int, int]:
    parts = value.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROI must be x,y,w,h")
    try:
        x, y, w, h = (int(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI values must be integers") from exc
    return x, y, w, h


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="Path to the input video")
    parser.add_argument("--out", default="out/crops.jsonl", help="Output JSONL with tracked crops")
    parser.add_argument("--tracker", default="csrt", help="OpenCV tracker to use (csrt, kcf)")
    parser.add_argument("--display", action="store_true", help="Show the tracking preview")
    parser.add_argument("--roi", help="Initial ROI as x,y,w,h (pixels). If set, skip selection UI.")
    parser.add_argument("--save-roi", help="Path to write the ROI json after first run (x,y,w,h).")
    parser.add_argument("--load-roi", help="Path to read ROI json (overrides --roi).")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Unable to read first frame from video")

    if args.load_roi and os.path.exists(args.load_roi):
        with open(args.load_roi, "r", encoding="utf-8") as f:
            rect = json.load(f)
        init_rect = (
            int(rect["x"]),
            int(rect["y"]),
            int(rect["w"]),
            int(rect["h"]),
        )
    elif args.roi:
        init_rect = parse_roi(args.roi)
    else:
        init_rect = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select ROI")

    if args.save_roi:
        with open(args.save_roi, "w", encoding="utf-8") as f:
            x, y, w, h = init_rect
            json.dump({"x": int(x), "y": int(y), "w": int(w), "h": int(h)}, f)

    tracker = build_tracker(args.tracker)
    tracker.init(frame, init_rect)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(idx: int, bbox: Tuple[float, float, float, float]) -> None:
        x, y, w, h = bbox
        record = {
            "frame": idx,
            "t": idx / fps,
            "x": int(round(x)),
            "y": int(round(y)),
            "w": int(round(w)),
            "h": int(round(h)),
        }
        json.dump(record, out_file)
        out_file.write("\n")

    with out_path.open("w", encoding="utf-8") as out_file:
        emit(frame_idx, init_rect)

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            ok, bbox = tracker.update(frame)
            if not ok:
                break
            emit(frame_idx, bbox)

            if args.display:
                x, y, w, h = map(int, bbox)
                preview = frame.copy()
                cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imshow("Tracking", preview)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                    break

    cap.release()
    if args.display:
        cv2.destroyWindow("Tracking")


if __name__ == "__main__":
    main()

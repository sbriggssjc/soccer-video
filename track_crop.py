#!/usr/bin/env python3
"""Interactive crop tracker for stabilized camera work."""

from __future__ import annotations

import argparse
import json
import os
import sys
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
    parser.add_argument("--video-out", help="Optional path to save cropped preview video")
    parser.add_argument(
        "--frames-dir",
        help="Optional directory to dump cropped frames and fps.txt",
    )
    parser.add_argument("--roi", help="Initial ROI as x,y,w,h (pixels). If set, skip selection UI.")
    parser.add_argument("--save-roi", help="Path to write the ROI json after first run (x,y,w,h).")
    parser.add_argument("--load-roi", help="Path to read ROI json (overrides --roi).")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open input: {args.video}")
        sys.exit(1)

    ok, frame = cap.read()
    if not ok:
        print("[ERROR] Unable to read first frame from video")
        sys.exit(1)

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

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    out_w, out_h = int(init_rect[2]), int(init_rect[3])
    writer = None
    video_writer_path: Path | None = None
    frames_dir_path: Path | None = None
    if args.frames_dir:
        frames_dir_path = Path(args.frames_dir)
        frames_dir_path.mkdir(parents=True, exist_ok=True)
    if args.video_out:
        video_path = (
            args.video_out
            if args.video_out.lower().endswith(".mp4")
            else args.video_out + ".mp4"
        )
        video_writer_path = Path(video_path)
        video_writer_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(video_writer_path), fourcc, fps, (out_w, out_h), isColor=True
        )
        if not writer.isOpened():
            print(f"[ERROR] Failed to open VideoWriter for: {video_writer_path}")
            sys.exit(1)

    frame_idx = 0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames_written = 0
    frames_dumped = 0

    def write_crop_frame(src_frame, bbox: Tuple[float, float, float, float]) -> None:
        nonlocal frames_written, frames_dumped
        x, y, w, h = bbox
        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(src_frame.shape[1], int(x + w))
        y1 = min(src_frame.shape[0], int(y + h))
        if x1 <= x0 or y1 <= y0:
            return
        crop = src_frame[y0:y1, x0:x1]
        if crop.size == 0:
            return
        crop_resized = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)
        if crop_resized.dtype != "uint8":
            crop_resized = crop_resized.astype("uint8")
        if writer is not None:
            writer.write(crop_resized)
            frames_written += 1
        if frames_dir_path is not None:
            frame_name = f"f{frames_dumped:06d}.jpg"
            frame_path = frames_dir_path / frame_name
            cv2.imwrite(
                str(frame_path),
                crop_resized,
                [int(cv2.IMWRITE_JPEG_QUALITY), 92],
            )
            frames_dumped += 1

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

    try:
        with out_path.open("w", encoding="utf-8") as out_file:
            emit(frame_idx, init_rect)
            write_crop_frame(frame, init_rect)

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_idx += 1
                ok, bbox = tracker.update(frame)
                if not ok:
                    break
                emit(frame_idx, bbox)
                write_crop_frame(frame, bbox)

                if args.display:
                    x, y, w, h = map(int, bbox)
                    preview = frame.copy()
                    cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.imshow("Tracking", preview)
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                        break
    finally:
        if writer is not None:
            writer.release()
            if video_writer_path is not None:
                try:
                    os.utime(str(video_writer_path), None)
                except Exception:
                    pass
        cap.release()
        cv2.destroyAllWindows()

    if video_writer_path is not None:
        output_str = str(video_writer_path)
        size = os.path.getsize(output_str) if os.path.exists(output_str) else 0
        if size < 1024 or frames_written == 0:
            print(
                f"[ERROR] Output looks invalid (size={size}, frames={frames_written})"
            )
            sys.exit(1)
        else:
            print(f"[OK] Wrote {frames_written} frames to {output_str}")

    if frames_dir_path is not None:
        fps_file = frames_dir_path / "fps.txt"
        with fps_file.open("w", encoding="utf-8") as f:
            f.write(str(fps))
        print(
            f"[track_crop] Wrote {frames_dumped} frames to {frames_dir_path} @ {fps} fps"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Motion guided auto-crop/zoom.

Reads a video, estimates per-frame motion using background subtraction
and outputs smoothed crop windows with constant aspect ratio. Crop
coordinates are written to ``out/crops.jsonl`` and a temporary video is
produced at ``out/zoomed_temp.mp4`` using a named pipe feeding FFmpeg.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess as sp
from pathlib import Path

import cv2
import numpy as np


def ensure_bounds(cx, cy, cw, ch, W, H):
    cx = np.clip(cx, cw / 2, W - cw / 2)
    cy = np.clip(cy, ch / 2, H - ch / 2)
    return cx, cy


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="inp", default="full_game_stabilized.mp4")
    p.add_argument("--out", dest="out_dir", default="out")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--smooth", type=float, default=0.85)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    jsonl = (out_dir / "crops.jsonl").open("w")

    cap = cv2.VideoCapture(args.inp)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cw, ch = min(args.width, W), min(args.height, H)
    back = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=64, detectShadows=False)

    pipe = out_dir / "pipe.raw"
    if pipe.exists():
        pipe.unlink()
    os.mkfifo(pipe)
    ffmpeg = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{cw}x{ch}",
        "-r",
        f"{fps}",
        "-i",
        str(pipe),
        "-an",
        str(out_dir / "zoomed_temp.mp4"),
    ]
    proc = sp.Popen(ffmpeg)
    pipe_f = open(pipe, "wb")

    cx, cy = W / 2, H / 2
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fg = back.apply(frame)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, None)
        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            x, y, w, h = cv2.boundingRect(np.vstack(cnts))
            tcx, tcy = x + w / 2, y + h / 2
        else:
            tcx, tcy = cx, cy
        # EMA smoothing
        cx = args.smooth * cx + (1 - args.smooth) * tcx
        cy = args.smooth * cy + (1 - args.smooth) * tcy
        # keep inside frame with gentle easing
        ncx, ncy = ensure_bounds(cx, cy, cw, ch, W, H)
        cx += (ncx - cx) / 12.0
        cy += (ncy - cy) / 12.0
        x0 = int(round(cx - cw / 2))
        y0 = int(round(cy - ch / 2))
        jsonl.write(
            json.dumps({
                "frame": frame_idx,
                "t": frame_idx / fps,
                "x": x0,
                "y": y0,
                "w": cw,
                "h": ch,
            })
            + "\n"
        )
        crop = frame[y0 : y0 + ch, x0 : x0 + cw]
        pipe_f.write(crop.tobytes())
        frame_idx += 1

    pipe_f.close()
    proc.wait()
    os.unlink(pipe)
    jsonl.close()
    cap.release()


if __name__ == "__main__":
    main()

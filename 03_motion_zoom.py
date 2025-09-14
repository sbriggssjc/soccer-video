#!/usr/bin/env python3
"""Motion guided auto-crop/zoom.

Reads a video, downsamples it for motion estimation and outputs smoothed
crop windows with constant aspect ratio. Crop coordinates are written to
``out/crops.jsonl`` and a temporary video is produced at
``out/zoomed_temp.mp4``. The script feeds raw BGR frames to FFmpeg via
stdin and falls back to OpenCV's ``VideoWriter`` when FFmpeg is missing.
"""
from __future__ import annotations

import argparse
import json
import subprocess as sp
from pathlib import Path

import cv2
import numpy as np


def ensure_bounds(cx: float, cy: float, cw: int, ch: int, W: int, H: int) -> tuple[float, float]:
    """Clamp center so the crop stays inside the full frame."""

    cx = float(np.clip(cx, cw / 2, W - cw / 2))
    cy = float(np.clip(cy, ch / 2, H - ch / 2))
    return cx, cy


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="inp", default="full_game_stabilized.mp4")
    p.add_argument("--out", dest="out_dir", default="out")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--smooth", type=float, default=0.85)
    p.add_argument("--downscale", type=int, default=640, help="width used for motion estimation")
    p.add_argument("--max-zoom", type=float, default=1.8, help="maximum zoom factor")
    p.add_argument("--border", type=float, default=0.06, help="safe border fraction")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    jsonl = (out_dir / "crops.jsonl").open("w")

    cap = cv2.VideoCapture(args.inp)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Base crop that fits inside the input frame.
    cw = min(args.width, W)
    ch = min(args.height, H)
    base_scale = cw / args.width  # same as ch / args.height
    min_scale = base_scale / args.max_zoom
    max_scale = base_scale

    # Dimensions for motion estimation.
    ds_w = min(args.downscale, W)
    ds_h = int(H * ds_w / W)
    scale_x, scale_y = W / ds_w, H / ds_h
    back = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=64, detectShadows=False)

    # Try to launch FFmpeg, otherwise fall back to VideoWriter.
    ffmpeg = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{args.width}x{args.height}",
        "-r",
        f"{fps}",
        "-i",
        "-",
        "-an",
        str(out_dir / "zoomed_temp.mp4"),
    ]
    writer = None
    try:
        proc = sp.Popen(ffmpeg, stdin=sp.PIPE)
        out_f = proc.stdin
    except FileNotFoundError:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_dir / "zoomed_temp.mp4"), fourcc, fps, (args.width, args.height))
        proc = None
        out_f = None

    cx, cy = W / 2, H / 2  # current center
    scale = max_scale  # current scale factor
    still = 0  # frames with little motion
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Downscaled frame for motion estimation.
        small = cv2.resize(frame, (ds_w, ds_h), interpolation=cv2.INTER_LINEAR)
        fg = back.apply(small)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, None)
        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if cnts:
            # motion area to decide if the scene is static
            motion = sum(cv2.contourArea(c) for c in cnts)
            if motion < 0.002 * ds_w * ds_h:
                still += 1
            else:
                still = 0
            x, y, w, h = cv2.boundingRect(np.vstack(cnts))
            tcx = (x + w / 2) * scale_x
            tcy = (y + h / 2) * scale_y
            w_full, h_full = w * scale_x, h * scale_y
        else:
            still += 1
            tcx, tcy = cx, cy
            w_full = h_full = 0

        # Smooth target center.
        cx = args.smooth * cx + (1 - args.smooth) * tcx
        cy = args.smooth * cy + (1 - args.smooth) * tcy

        # Desired scale so the motion stays inside the crop with margin.
        req_scale = max(
            w_full / (1 - args.border) / args.width,
            h_full / (1 - args.border) / args.height,
            min_scale,
        )
        if still > fps * 0.75:
            req_scale = max_scale  # ease back to wide
        scale = args.smooth * scale + (1 - args.smooth) * req_scale
        scale = float(np.clip(scale, min_scale, max_scale))

        cw = int(round(args.width * scale))
        ch = int(round(args.height * scale))
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
        crop = cv2.resize(crop, (args.width, args.height), interpolation=cv2.INTER_LINEAR)

        if writer is not None:
            writer.write(crop)
        else:
            assert out_f is not None
            out_f.write(crop.tobytes())

        frame_idx += 1

    if writer is not None:
        writer.release()
    else:
        assert proc is not None and out_f is not None
        out_f.close()
        proc.wait()

    jsonl.close()
    cap.release()


if __name__ == "__main__":
    main()


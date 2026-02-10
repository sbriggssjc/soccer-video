"""Ball tracking helper that emits FFmpeg expressions for smart crops."""

import argparse
import json
import math
import os
import sys

import cv2
import numpy as np


# Usage:
#   python track_ball_and_emit_expr.py --video "C:\\path\\to\\in.mp4" --start 1705.2 --end 1717.9 --mode WIDE
#   python track_ball_and_emit_expr.py --video "..." --start ... --end ... --mode REEL
#
# Emits three lines to stdout:
#   X_EXPR=
#   Y_EXPR=
#   Z_EXPR=
# You pass these directly into your PowerShell to build -vf / -filter_complex.
#
# ---- minimal YOLOv8-like detection via OpenCV ball color & size heuristic ----
# If you keep your YOLO step elsewhere, you can swap this with your detector and
# just fill cx, cy per frame.


def detect_ball_bgr(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=40,
        param1=110,
        param2=18,
        minRadius=3,
        maxRadius=25,
    )
    if circles is not None:
        c = circles[0, 0]
        return float(c[0]), float(c[1]), 0.65
    return None, None, 0.0


def lerp(a, b, t):
    return a + (b - a) * t


def smooth_centers(centers, lead=5):
    out = []
    n = len(centers)
    for i, (x, y) in enumerate(centers):
        j = min(i + lead, n - 1)
        fx, fy = centers[j]
        out.append((0.7 * x + 0.3 * fx, 0.7 * y + 0.3 * fy))
    return out


def moving_limit(prev, curr, max_delta):
    dx = np.clip(curr[0] - prev[0], -max_delta, max_delta)
    dy = np.clip(curr[1] - prev[1], -max_delta, max_delta)
    return (prev[0] + dx, prev[1] + dy)


def build_if_chain(values):
    parts = []
    for start, end, val in values:
        parts.append(f"(between(n\\,{start}\\,{end})*({val:.6f}))")
    return " + ".join(parts)


def compress_to_runs(arr):
    if not arr:
        return []
    runs = []
    rstart = 0
    prev = arr[0]
    for i in range(1, len(arr)):
        if abs(arr[i] - prev) > 1e-3:
            runs.append((rstart, i - 1, prev))
            rstart = i
            prev = arr[i]
    runs.append((rstart, len(arr) - 1, prev))
    return runs


def clamp(v, lo, hi):
    return float(max(lo, min(hi, v)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--end", type=float, required=True)
    ap.add_argument("--fps", type=float, default=59.94)
    ap.add_argument("--mode", choices=["WIDE", "REEL"], required=True)
    ap.add_argument("--manual", default="", help="optional manual_keyframes.json")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("X_EXPR=")
        print("Y_EXPR=")
        print("Z_EXPR=")
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    f1 = int(round(args.start * src_fps))
    f2 = int(round(args.end * src_fps))
    total = max(0, f2 - f1)
    if total <= 0:
        print("X_EXPR=")
        print("Y_EXPR=")
        print("Z_EXPR=")
        return

    if args.mode == "REEL":
        scale_w = int(round(W * (1920.0 / H)))
        scale_h = 1920
        crop_w, crop_h = 1080, 1920
    else:
        scale_w = 1920
        scale_h = int(round(H * (1920.0 / W)))
        crop_w, crop_h = 1920, 1080

    centers = []
    confs = []
    people_centers = []

    manual = {}
    if args.manual and os.path.exists(args.manual):
        try:
            data = json.load(open(args.manual, "r", encoding="utf-8"))
            for kf in data:
                manual[int(kf["frame"])] = (float(kf["x"]), float(kf["y"]))
        except Exception:
            pass

    cap.set(cv2.CAP_PROP_POS_FRAMES, f1)
    for i in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        frame_s = cv2.resize(frame, (scale_w, scale_h), interpolation=cv2.INTER_CUBIC)

        if i in manual:
            cx, cy = manual[i]
            conf = 1.0
        else:
            cx, cy, conf = detect_ball_bgr(frame_s)

        if cx is None:
            gx = cv2.Sobel(cv2.cvtColor(frame_s, cv2.COLOR_BGR2GRAY), cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(cv2.cvtColor(frame_s, cv2.COLOR_BGR2GRAY), cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.magnitude(gx, gy)
            thr = np.percentile(mag, 90)
            ys, xs = np.where(mag >= thr)
            if xs.size > 100:
                cx, cy = float(xs.mean()), float(ys.mean())
                conf = 0.35

        if (cx is None or conf < 0.3) and centers:
            cx, cy = centers[-1]

        if cx is None:
            cx, cy = scale_w / 2.0, scale_h / 2.0

        centers.append((float(cx), float(cy)))
        confs.append(conf)

    cap.release()

    if not centers:
        print("X_EXPR=")
        print("Y_EXPR=")
        print("Z_EXPR=")
        return

    centers = smooth_centers(centers, lead=5)

    MAX_STEP = 40.0
    path = [centers[0]]
    for i in range(1, len(centers)):
        path.append(moving_limit(path[-1], centers[i], MAX_STEP))

    zoom = []
    for cx, cy in path:
        dx = abs(cx - scale_w / 2.0)
        dy = abs(cy - scale_h / 2.0)
        dist = math.hypot(dx, dy)
        z = 1.0 + min(0.30, dist / (max(scale_w, scale_h) / 3.0))
        zoom.append(z)

    xs = []
    ys = []
    for i, (cx, cy) in enumerate(path):
        z = zoom[i]
        vw = crop_w / z
        vh = crop_h / z
        x = clamp(cx - vw / 2.0, 0, scale_w - vw)
        y = clamp(cy - vh / 2.0, 0, scale_h - vh)
        xs.append(x)
        ys.append(y)
        zoom[i] = z

    xs = [round(v, 1) for v in xs]
    ys = [round(v, 1) for v in ys]
    zoom = [round(z, 3) for z in zoom]

    x_runs = compress_to_runs(xs)
    y_runs = compress_to_runs(ys)
    z_runs = compress_to_runs(zoom)

    x_expr = build_if_chain(x_runs)
    y_expr = build_if_chain(y_runs)
    z_expr = build_if_chain(z_runs)

    print(f"X_EXPR={x_expr}")
    print(f"Y_EXPR={y_expr}")
    print(f"Z_EXPR={z_expr}")


if __name__ == "__main__":
    main()

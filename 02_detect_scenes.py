#!/usr/bin/env python3
"""Pick highlight segments based on audio and motion.

Scores each second by combining RMS audio energy and total motion
magnitude. Segments whose combined score exceed a dynamic threshold are
expanded with pre/post roll and merged. Results are written to
``out/highlights.csv`` with columns ``start,end,score`` in seconds.

CLI options:
  --video PATH     Input video (default full_game_stabilized.mp4)
  --min-gap SEC    Minimum gap between merged segments (default 2.0)
  --pre SEC        Seconds to prepend before a hit (default 5)
  --post SEC       Seconds to append after a hit (default 6)
  --max-count N    Limit number of segments (default 40)
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import librosa
import numpy as np


def compute_audio_scores(path: str) -> np.ndarray:
    """Return per-second RMS energy."""
    y, sr = librosa.load(path, sr=None, mono=True)
    hop = sr  # 1 second steps
    rms = librosa.feature.rms(y=y, frame_length=sr, hop_length=hop)[0]
    return rms


def compute_motion_scores(path: str) -> np.ndarray:
    """Return per-second sum of optical-flow magnitudes."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames_per_sec = int(round(fps))
    ret, prev = cap.read()
    if not ret:
        return np.zeros(0)
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    acc = []
    total = 0.0
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag = np.linalg.norm(flow, axis=2).sum()
        total += mag
        count += 1
        if count == frames_per_sec:
            acc.append(total)
            total = 0.0
            count = 0
        prev = gray
    if count:
        acc.append(total)
    cap.release()
    return np.array(acc, dtype=float)


def normalize(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(float)
    if arr.size == 0:
        return arr
    arr -= arr.min()
    maxv = arr.max()
    if maxv > 0:
        arr /= maxv
    return arr


def pick_segments(scores: np.ndarray, rate: float, args) -> list[tuple[float, float, float]]:
    thr = scores.mean() + scores.std() * 0.5
    hits = np.where(scores > thr)[0]
    segments = []
    for idx in hits:
        start = max(idx - args.pre, 0)
        end = idx + 1 + args.post
        segments.append((start, end, float(scores[idx])))
    # merge overlapping
    segments.sort()
    merged = []
    for seg in segments:
        if not merged or seg[0] - merged[-1][1] >= args.min_gap:
            merged.append(list(seg))
        else:
            merged[-1][1] = max(merged[-1][1], seg[1])
            merged[-1][2] = max(merged[-1][2], seg[2])
    # limit count
    merged.sort(key=lambda s: s[2], reverse=True)
    merged = merged[: args.max_count]
    merged.sort(key=lambda s: s[0])
    return [(s[0], s[1], s[2]) for s in merged]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", default="full_game_stabilized.mp4")
    p.add_argument("--min-gap", type=float, default=2.0)
    p.add_argument("--pre", type=float, default=5.0)
    p.add_argument("--post", type=float, default=6.0)
    p.add_argument("--max-count", type=int, default=40)
    args = p.parse_args()

    audio = compute_audio_scores(args.video)
    motion = compute_motion_scores(args.video)
    n = max(len(audio), len(motion))
    audio = np.pad(audio, (0, max(0, n - len(audio))))
    motion = np.pad(motion, (0, max(0, n - len(motion))))
    audio = normalize(audio)
    motion = normalize(motion)
    scores = 0.5 * (audio + motion)

    segs = pick_segments(scores, 1.0, args)
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / "highlights.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["start", "end", "score"])
        for s, e, sc in segs:
            writer.writerow([f"{s:.2f}", f"{e:.2f}", f"{sc:.3f}"])


if __name__ == "__main__":
    main()

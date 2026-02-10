"""Scene and event detection combining motion and audio."""
from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np
from ._loguru import logger
from ._tqdm import tqdm
from dataclasses import dataclass

from .config import AppConfig
from .goals import detect_goal_windows
from .io import video_stream_info
from .utils import HighlightWindow, merge_overlaps, summary_stats, write_highlights


@dataclass
class DetectionOutput:
    windows: List[HighlightWindow]
    adaptive_threshold: float
    low_threshold: float
    mean_score: float
    std_score: float


def _compute_audio_scores(video_path: Path, target_rate: int = 1) -> np.ndarray:
    try:
        import librosa
    except Exception as exc:
        logger.warning("librosa unavailable (%s); audio scores disabled", exc)
        return np.zeros(0, dtype=np.float32)

    try:
        y, sr = librosa.load(video_path, sr=None, mono=True)
    except Exception as exc:
        logger.warning("Failed to load audio: %s", exc)
        return np.zeros(0, dtype=np.float32)
    hop = sr // target_rate
    hop = max(1, hop)
    rms = librosa.feature.rms(y=y, frame_length=sr, hop_length=hop)[0]
    if rms.size == 0:
        return np.zeros(0, dtype=np.float32)
    return rms.astype(np.float32)


def _compute_motion_scores(video_path: Path, fps: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Could not open video %s", video_path)
        return np.zeros(0, dtype=np.float32)
    frames_per_bin = max(1, int(round(fps)))
    ok, prev = cap.read()
    if not ok:
        cap.release()
        return np.zeros(0, dtype=np.float32)
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    flow_acc: List[float] = []
    accum = 0.0
    count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    try:
        with tqdm(total=max(total_frames - 1, 0), desc="motion", unit="frame", leave=False) as bar:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag = np.linalg.norm(flow, axis=2).sum()
                accum += float(mag)
                count += 1
                bar.update(1)
                if count == frames_per_bin:
                    flow_acc.append(accum)
                    accum = 0.0
                    count = 0
                prev = gray
        if count:
            flow_acc.append(accum)
    finally:
        cap.release()
    if not flow_acc:
        return np.zeros(0, dtype=np.float32)
    arr = np.array(flow_acc, dtype=np.float32)
    return arr


def _normalize(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    arr = arr.astype(np.float32)
    arr = arr - arr.min()
    maxv = arr.max()
    if maxv > 1e-6:
        arr = arr / maxv
    return arr


def _expand_window(idx_start: int, idx_end: int, pre: float, post: float, total_duration: float) -> tuple[float, float]:
    # Each bin index corresponds to ~1 second of video (bin size = round(fps) frames).
    # Convert indices to seconds explicitly for clarity.
    start_time = max(0.0, float(idx_start) - pre)
    end_time = float(idx_end + 1) + post
    end_time = min(total_duration, end_time)
    return start_time, end_time


def detect_highlights(config: AppConfig, video_path: Path, output_csv: Path) -> DetectionOutput:
    """Detect highlight windows by combining motion and audio analysis."""
    info = video_stream_info(video_path)
    logger.info("Detecting highlights from %s (fps=%.2f, duration=%.2fs)", video_path, info.fps, info.duration)

    motion = _compute_motion_scores(video_path, info.fps)
    audio = _compute_audio_scores(video_path)

    n = max(len(motion), len(audio))
    if n == 0:
        logger.warning("No frames or audio samples found; nothing to detect")
        return DetectionOutput([], 0.0, 0.0, 0.0, 0.0)
    motion = np.pad(motion, (0, max(0, n - len(motion))), constant_values=0.0)
    audio = np.pad(audio, (0, max(0, n - len(audio))), constant_values=0.0)

    motion = _normalize(motion)
    audio = _normalize(audio)
    w = config.detect.audio_weight
    scores = (1.0 - w) * motion + w * audio

    mean_score = float(scores.mean())
    std_score = float(scores.std())
    adaptive_thr = mean_score + std_score * config.detect.threshold_std
    low_thr = adaptive_thr * (1.0 - config.detect.hysteresis)
    sustain_frames = max(1, int(round(config.detect.sustain)))

    logger.info(
        "Adaptive threshold %.3f (low %.3f) mean %.3f std %.3f", adaptive_thr, low_thr, mean_score, std_score
    )

    windows: List[HighlightWindow] = []
    idx = 0
    while idx < n:
        if scores[idx] >= adaptive_thr:
            # verify sustain
            sustain_ok = True
            for j in range(idx, min(idx + sustain_frames, n)):
                if scores[j] < adaptive_thr:
                    sustain_ok = False
                    break
            if not sustain_ok:
                idx += 1
                continue
            peak_score = float(scores[idx])
            start_idx = idx
            idx += 1
            while idx < n and scores[idx] >= low_thr:
                peak_score = max(peak_score, float(scores[idx]))
                idx += 1
            end_idx = max(idx - 1, start_idx)
            start_sec, end_sec = _expand_window(start_idx, end_idx, config.detect.pre, config.detect.post, info.duration)
            windows.append(HighlightWindow(start=start_sec, end=end_sec, score=peak_score, event="scene"))
        else:
            idx += 1

    banned = {event.lower() for event in (config.detect.exclude_events or [])}
    if banned:
        windows = [w for w in windows if str(w.event).lower() not in banned]
    goal_windows = detect_goal_windows(config, video_path, info, windows)
    if goal_windows:
        windows.extend(goal_windows)

    merged = merge_overlaps(windows, config.detect.min_gap)
    merged.sort(key=lambda w: w.score, reverse=True)
    if config.detect.max_count:
        merged = merged[: config.detect.max_count]
    merged.sort(key=lambda w: w.start)

    stats = summary_stats(merged)
    logger.info("Detected %d windows", stats["count"])

    return DetectionOutput(
        windows=merged,
        adaptive_threshold=adaptive_thr,
        low_threshold=low_thr,
        mean_score=mean_score,
        std_score=std_score,
    )


def run_detect(config: AppConfig, video_path: Path, output_csv: Path) -> DetectionOutput:
    """Run highlight detection and write results to *output_csv*."""
    result = detect_highlights(config, video_path, output_csv)
    write_highlights(output_csv, result.windows)
    return result

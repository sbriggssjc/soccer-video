"""Refine highlight windows around motion/audio peaks."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

try:  # pragma: no cover - optional dependency
    import cv2  # type: ignore
    _cv2_error = cv2.error  # type: ignore[attr-defined]
except Exception as exc:  # pragma: no cover
    cv2 = None  # type: ignore
    _cv2_error = RuntimeError
    _cv2_import_error = exc
else:
    _cv2_import_error = None
from ._loguru import logger
from ._tqdm import tqdm

from .config import AppConfig
from .colors import calibrate_colors, hsv_mask
from .io import video_stream_info
from .utils import HighlightWindow, clamp, read_highlights, trim_to_duration, write_highlights


def _audio_envelope(video_path: Path) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        import librosa
    except Exception as exc:
        logger.warning("librosa unavailable (%s); audio guidance disabled", exc)
        return None, None
    try:
        y, sr = librosa.load(video_path, sr=None, mono=True)
    except Exception as exc:
        logger.warning("Failed to load audio for envelope: %s", exc)
        return None, None
    hop = 512
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop).astype(np.float32)
    t = librosa.frames_to_time(np.arange(len(env)), sr=sr, hop_length=hop)
    if env.size:
        env = (env - env.min()) / (env.max() - env.min() + 1e-8)
    return t, env


def _motion_timeseries(cap: cv2.VideoCapture, f0: int, f1: int, use_mask: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    ok, prev = cap.read()
    if not ok:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    prev_small = cv2.resize(prev, (0, 0), fx=0.5, fy=0.5)
    prev_gray = cv2.GaussianBlur(cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    energies: List[float] = []
    times: List[float] = []

    for frame_idx in range(f0 + 1, f1):
        ok, frame = cap.read()
        if not ok:
            break
        sm = cv2.resize(frame, (prev_small.shape[1], prev_small.shape[0]))
        gray = cv2.GaussianBlur(cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        diff = cv2.absdiff(gray, prev_gray).astype(np.float32)
        base = float(diff.sum())
        if use_mask is not None:
            mask_resized = cv2.resize(use_mask, (sm.shape[1], sm.shape[0]))
            weighted = float((diff * (mask_resized / 255.0)).sum())
            energy = 0.65 * base + 0.35 * weighted
        else:
            energy = base
        energies.append(energy)
        times.append(frame_idx / fps)
        prev_gray = gray
    if not energies:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    arr = np.array(energies, dtype=np.float32)
    ker = np.ones(9, np.float32) / 9.0
    arr = np.convolve(arr, ker, mode="same")
    if arr.max() > 0:
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr, np.array(times, dtype=np.float32)


def _find_peak(
    cap: cv2.VideoCapture,
    win: HighlightWindow,
    audio_t: Optional[np.ndarray],
    audio_env: Optional[np.ndarray],
    use_mask: Optional[np.ndarray],
    total_frames: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame = clamp(int(win.start * fps), 0, max(total_frames - 2, 0))
    end_frame = clamp(int(win.end * fps), start_frame + 1, max(total_frames - 1, start_frame + 1))
    motion, times = _motion_timeseries(cap, start_frame, end_frame, use_mask)
    if motion.size == 0 or times.size == 0:
        center = (win.start + win.end) / 2.0
        return center, motion, times
    score = motion.copy()
    if audio_t is not None and audio_env is not None and audio_env.size:
        interp = np.interp(times, audio_t, audio_env)
        interp = (interp - interp.min()) / (interp.max() - interp.min() + 1e-8)
        score = 0.75 * score + 0.25 * interp
    idx = int(np.argmax(score))
    return float(times[idx]), motion, times


def _ball_in_play_gate(
    times: np.ndarray,
    motion: np.ndarray,
    peak_time: float,
    default_start: float,
    default_end: float,
    total_dur: float,
) -> tuple[float, float]:
    if motion.size == 0 or times.size == 0:
        return default_start, default_end
    idx = int(np.argmin(np.abs(times - peak_time)))
    if idx < 0 or idx >= motion.size:
        return default_start, default_end
    peak_val = float(motion[idx])
    if peak_val <= 1e-6:
        return default_start, default_end
    baseline = float(np.percentile(motion, 40))
    level = max(0.18, min(peak_val * 0.7, baseline + 0.15))
    if level >= peak_val:
        level = peak_val * 0.6
    if level <= 1e-6:
        return default_start, default_end
    start_idx = idx
    while start_idx > 0 and motion[start_idx - 1] >= level:
        start_idx -= 1
    end_idx = idx
    n = motion.size
    while end_idx + 1 < n and motion[end_idx + 1] >= level:
        end_idx += 1
    start_time = float(times[start_idx])
    end_time = float(times[end_idx])
    start_time = max(0.0, start_time - 0.4)
    end_time = min(total_dur, end_time + 0.9)
    start = max(default_start, start_time)
    end = min(default_end, end_time)
    if end <= start:
        end = min(default_end, max(start + 1.2, end_time + 0.5))
    if end < peak_time + 0.4:
        end = min(default_end, max(end, peak_time + 0.4))
    return start, min(end, total_dur)


def _tracked_frames(cap: cv2.VideoCapture, f0: int, f1: int) -> Iterable[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    for _ in range(f0, f1):
        ok, frame = cap.read()
        if not ok:
            break
        yield frame


def _tracked_writer(out_path: Path, frames: Iterable[np.ndarray], width: int, height: int, aspect: str, zoom: float, fps: float) -> None:
    if aspect == "vertical":
        out_w, out_h = 1080, 1920
        crop_h = height
        crop_w = min(width, int(round(crop_h * 9 / 16)))
    else:
        out_w, out_h = 1920, 1080
        crop_w = max(640, min(width, int(round(width / zoom))))
        crop_h = max(360, min(height, int(round(height / zoom))))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open {out_path} for writing")
    cx, cy = width // 2, height // 2
    alpha = 0.12
    sw, sh = width // 2, height // 2
    prev_small = None
    for frame in frames:
        small = cv2.resize(frame, (sw, sh))
        gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        if prev_small is None:
            prev_small = gray
        diff = cv2.absdiff(gray, prev_small)
        prev_small = gray
        thr = max(10, int(diff.mean() + 2 * diff.std()))
        _, mask = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=1)
        ys, xs = np.where(mask > 0)
        if xs.size:
            cx_s = np.mean(xs)
            cy_s = np.mean(ys)
            cx = int((1 - alpha) * cx + alpha * (cx_s * (width / float(sw))))
            cy = int((1 - alpha) * cy + alpha * (cy_s * (height / float(sh))))
        else:
            cx = int((1 - alpha) * cx + alpha * (width // 2))
            cy = int((1 - alpha) * cy + alpha * (height // 2))
        x0 = int(round(cx - crop_w / 2))
        y0 = int(round(cy - crop_h / 2))
        x0 = int(clamp(x0, 0, width - crop_w))
        y0 = int(clamp(y0, 0, height - crop_h))
        crop = frame[y0 : y0 + crop_h, x0 : x0 + crop_w]
        resized = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
        writer.write(resized)
    writer.release()


def smart_shrink(config: AppConfig, video_path: Path, windows: List[HighlightWindow]) -> List[HighlightWindow]:
    """Refine highlight windows using motion/audio peak analysis."""
    if cv2 is None:
        raise RuntimeError("OpenCV is required for smart shrink mode but is not installed")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps else 0.0
    audio_t, audio_env = _audio_envelope(video_path)
    colors = config.colors
    if colors.calibrate:
        colors = calibrate_colors(str(video_path), colors)
        config.colors = colors
    mask = None
    refined: List[HighlightWindow] = []
    try:
        if config.shrink.bias_blue:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
            if ok:
                mask = hsv_mask(frame, colors.team_primary)
        tracker_out = config.shrink.write_clips
        if tracker_out:
            Path(tracker_out).mkdir(parents=True, exist_ok=True)
        for idx, win in enumerate(tqdm(windows, desc="shrink", unit="clip", leave=False), start=1):
            peak, motion_series, motion_times = _find_peak(cap, win, audio_t, audio_env, mask, total_frames)
            rs = clamp(peak - config.shrink.pre, 0.0, total_dur)
            re = clamp(peak + config.shrink.post, 0.0, total_dur)
            if win.event == "goal":
                rs, re = _ball_in_play_gate(motion_times, motion_series, peak, rs, re, total_dur)
            if re <= rs + 0.1:
                re = clamp(rs + 0.1, 0.0, total_dur)
            refined.append(HighlightWindow(start=rs, end=re, score=win.score, event=win.event))
            if tracker_out:
                start_frame = int(rs * fps)
                end_frame = min(total_frames - 1, int(re * fps))
                if end_frame > start_frame:
                    out_path = Path(tracker_out) / f"clip_{idx:04d}.mp4"
                    frame_gen = _tracked_frames(cap, start_frame, end_frame)
                    _tracked_writer(out_path, frame_gen, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), config.shrink.aspect, config.shrink.zoom, fps)
                    logger.info("Tracked clip %s", out_path)
    finally:
        cap.release()
    return refined


def simple_shrink(config: AppConfig, video_path: Path, windows: List[HighlightWindow]) -> List[HighlightWindow]:
    """Trim highlight windows using fixed pre/post offsets without video analysis."""
    info = video_stream_info(Path(video_path))
    refined: List[HighlightWindow] = []
    for win in windows:
        start, end = trim_to_duration(win.start, win.end, config.shrink.pre, config.shrink.post, info.duration)
        refined.append(HighlightWindow(start=start, end=end, score=win.score, event=win.event))
    return refined


def run_shrink(config: AppConfig, video_path: Path, csv_in: Path, csv_out: Path) -> List[HighlightWindow]:
    """Run the configured shrink mode and write refined windows to *csv_out*."""
    windows = read_highlights(csv_in)
    if not windows:
        logger.warning("No input windows found in %s", csv_in)
        return []
    if config.shrink.mode == "simple":
        refined = simple_shrink(config, video_path, windows)
    else:
        refined = smart_shrink(config, video_path, windows)
    write_highlights(csv_out, refined)
    return refined

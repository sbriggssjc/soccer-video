"""Rank candidate clips and produce Top-K lists."""
from __future__ import annotations

import csv
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np
from ._loguru import logger

from .clip_gating import first_live_frame
from .config import AppConfig


@dataclass
class RankedClip:
    path: Path
    inpoint: float
    duration: float
    motion: float
    audio: float
    score: float


def _activity_profile(path: Path, sample_fps: int = 6) -> tuple[List[float], np.ndarray, np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    stride = max(1, int(round(fps / sample_fps)))
    ok, prev = cap.read()
    if not ok:
        cap.release()
        return [], np.zeros(0), np.zeros(0)
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    times: List[float] = []
    cover: List[float] = []
    mag: List[float] = []
    while True:
        for _ in range(stride - 1):
            if not cap.grab():
                break
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev)
        prev = gray
        m = float(diff.mean()) / 255.0
        mask = (diff > 10).astype(np.uint8)
        c = float(mask.mean())
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        times.append(t)
        cover.append(c)
        mag.append(m)
    cap.release()
    return times, np.array(cover), np.array(mag)


def _find_first_active(times: List[float], cover: np.ndarray, mag: np.ndarray, sustain_sec: float, sample_fps: int = 6) -> float:
    if not times:
        return 0.0
    cov_base = np.percentile(cover, 20) if cover.size else 0.0
    cov_thr = max(0.01, cov_base + 0.01)
    mag_thr = max(0.01, np.percentile(mag, 20) + 0.005) if mag.size else 0.01
    active = (cover > cov_thr) & (mag > mag_thr)
    need = max(1, int(round(sustain_sec * sample_fps)))
    run = 0
    for idx, flag in enumerate(active):
        run = run + 1 if flag else 0
        if run >= need:
            return max(0.0, times[idx] - 1.0)
    return 0.0


def _audio_rms(path: Path) -> float:
    try:
        import librosa
    except Exception:
        return 0.0
    try:
        y, sr = librosa.load(path, sr=None, mono=True)
    except Exception:
        return 0.0
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512).ravel()
    if rms.size == 0:
        return 0.0
    return float(np.percentile(rms, 90))


def _clip_duration(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return float(frames / fps) if frames else 0.0


def score_clip(path: Path, sustain_sec: float) -> RankedClip:
    times, cover, mag = _activity_profile(path)
    inpoint = _find_first_active(times, cover, mag, sustain_sec)
    if times and cover.size and mag.size:
        frame_metrics = []
        ball_metrics = []
        for cov, motion in zip(cover.tolist(), mag.tolist()):
            cov_f = float(max(0.0, cov))
            mot_f = float(max(0.0, motion))
            frame_metrics.append(
                {
                    "pitch_ratio": cov_f,
                    "moving_players": cov_f * 30.0,
                    "touch_prob": mot_f,
                    "motion": mot_f,
                }
            )
            ball_metrics.append({"speed": mot_f * 40.0, "touch_prob": mot_f})
        idx = first_live_frame(frame_metrics, ball_metrics, None)
        if idx is not None and idx < len(times):
            if len(times) >= 2:
                step = max(0.1, float(times[idx] - times[idx - 1]) if idx > 0 else float(times[1] - times[0]))
            else:
                step = 0.3
            gate_start = max(0.0, float(times[idx]) - step)
            inpoint = max(inpoint, gate_start)
    motion_score = float(np.percentile(mag[cover > 0] if cover.size and (cover > 0).any() else mag, 80)) if mag.size else 0.0
    audio_score = _audio_rms(path)
    duration = _clip_duration(path)
    score = 0.65 * motion_score + 0.35 * audio_score
    return RankedClip(path=path, inpoint=round(inpoint, 3), duration=duration, motion=motion_score, audio=audio_score, score=score)


def _candidate_files(paths: Iterable[Path]) -> List[Path]:
    files: List[Path] = []
    for directory in paths:
        pattern = str(directory / "clip_*.mp4")
        files.extend(Path(f) for f in sorted(glob.glob(pattern)))
    return files


def write_rankings(csv_path: Path, clips: List[RankedClip]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "inpoint", "duration", "motion", "audio", "score"])
        for clip in clips:
            writer.writerow([clip.path.as_posix(), f"{clip.inpoint:.3f}", f"{clip.duration:.3f}", f"{clip.motion:.4f}", f"{clip.audio:.4f}", f"{clip.score:.4f}"])


def write_concat(list_path: Path, clips: List[RankedClip], max_len: float) -> None:
    list_path.parent.mkdir(parents=True, exist_ok=True)
    with list_path.open("w", newline="\n", encoding="utf-8") as f:
        for clip in clips:
            outpoint = min(clip.duration, clip.inpoint + max_len)
            f.write(f"file '{clip.path.as_posix()}'\n")
            f.write(f"inpoint {clip.inpoint:.3f}\n")
            f.write(f"outpoint {outpoint:.3f}\n")


def run_topk(config: AppConfig, candidate_dirs: List[Path], csv_out: Path, concat_out: Path, k: int | None = None, max_len: float | None = None) -> List[RankedClip]:
    k = k or config.rank.k
    max_len = max_len or config.rank.max_len
    files = _candidate_files(candidate_dirs)
    if not files:
        logger.warning("No candidate clips found in %s", candidate_dirs)
        return []
    scored = [score_clip(path, config.rank.sustain) for path in files]
    filtered = [clip for clip in scored if clip.duration - clip.inpoint >= config.rank.min_tail]
    if len(filtered) < k:
        filtered = scored
    ranked = sorted(filtered, key=lambda c: c.score, reverse=True)[:k]
    write_rankings(csv_out, ranked)
    write_concat(concat_out, ranked, max_len)
    logger.info("Wrote %s and %s", csv_out, concat_out)
    return ranked

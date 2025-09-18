"""Rank candidate clips and produce Top-K lists."""
from __future__ import annotations

import csv
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np
from ._loguru import logger

from .clip_gating import first_live_frame
from .config import AppConfig
from .utils import clamp


@dataclass
class RankedClip:
    path: Path
    inpoint: float
    outpoint: float
    duration: float
    motion: float
    audio: float
    score: float
    event: Optional[str] = None


@dataclass
class ClipMetadata:
    event: Optional[str] = None
    passes: Optional[int] = None
    start: Optional[float] = None
    end: Optional[float] = None


_DEFAULT_PRE = 0.7
_DEFAULT_POST = 1.6
_SHOT_PRE = 1.2
_SHOT_POST = 2.2
_BUILDUP_POST = 0.9
_SAMPLE_FPS = 6
_SHOT_EVENT_KEYWORDS = {"goal", "shot"}
_BUILDUP_KEYWORDS = {"pass", "build", "chain"}


def _parse_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _load_clip_metadata(directory: Path) -> Dict[Path, ClipMetadata]:
    meta_path = directory / "clips_metadata.csv"
    if not meta_path.exists():
        return {}
    mapping: Dict[Path, ClipMetadata] = {}
    with meta_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            filename = (row.get("filename") or row.get("path") or "").strip()
            if not filename:
                continue
            clip_path = (directory / filename).resolve()
            event = (row.get("event") or "").strip() or None
            start = _parse_float(row.get("start"))
            end = _parse_float(row.get("end"))
            passes = _parse_int(row.get("passes"))
            mapping[clip_path] = ClipMetadata(event=event, passes=passes, start=start, end=end)
    return mapping


def _collect_metadata(directories: Iterable[Path]) -> Dict[Path, ClipMetadata]:
    combined: Dict[Path, ClipMetadata] = {}
    for directory in directories:
        combined.update(_load_clip_metadata(directory))
    return combined


def _classify_event(meta: Optional[ClipMetadata]) -> str:
    if meta is None or not meta.event:
        return "default"
    event = meta.event.lower()
    if any(keyword in event for keyword in _SHOT_EVENT_KEYWORDS):
        return "shot"
    if any(keyword in event for keyword in _BUILDUP_KEYWORDS):
        if meta.passes is None or meta.passes >= 4:
            return "buildup"
    return "default"


def _activity_thresholds(cover: np.ndarray, mag: np.ndarray) -> tuple[float, float]:
    cov_base = float(np.percentile(cover, 20)) if cover.size else 0.0
    cov_thr = max(0.01, cov_base + 0.01)
    mag_thr = max(0.01, float(np.percentile(mag, 20) + 0.005)) if mag.size else 0.01
    return cov_thr, mag_thr


def _find_active_bounds(times: np.ndarray, cover: np.ndarray, mag: np.ndarray, sustain_sec: float, duration: float) -> tuple[float, float]:
    if times.size == 0:
        return 0.0, duration
    cov_thr, mag_thr = _activity_thresholds(cover, mag)
    active = (cover > cov_thr) & (mag > mag_thr)
    need = max(1, int(round(sustain_sec * _SAMPLE_FPS)))
    first_idx = 0
    run = 0
    candidate = None
    for idx, flag in enumerate(active.tolist()):
        if flag:
            run += 1
            if candidate is None:
                candidate = idx
            if run >= need and candidate is not None:
                first_idx = candidate
                break
        else:
            run = 0
            candidate = None
    last_idx = times.size - 1
    run = 0
    candidate = None
    for offset, flag in enumerate(reversed(active.tolist())):
        idx = times.size - 1 - offset
        if flag:
            run += 1
            if candidate is None:
                candidate = idx
            if run >= need and candidate is not None:
                last_idx = candidate
                break
        else:
            run = 0
            candidate = None
    first_time = float(times[min(max(first_idx, 0), times.size - 1)])
    last_time = float(times[min(max(last_idx, 0), times.size - 1)])
    return first_time, last_time


def _peak_time(times: np.ndarray, cover: np.ndarray, mag: np.ndarray, duration: float) -> float:
    if times.size == 0:
        return duration / 2.0 if duration > 0 else 0.0
    if mag.size == 0:
        return float(times[min(0, times.size - 1)])
    combined = mag
    if cover.size:
        combined = 0.7 * mag + 0.3 * cover
    idx = int(np.argmax(combined))
    idx = min(max(idx, 0), times.size - 1)
    return float(times[idx])


def _compute_trim_bounds(category: str, first_time: float, peak_time: float, last_time: float, duration: float) -> tuple[float, float]:
    if duration <= 0:
        return 0.0, 0.0
    if category == "buildup":
        start = clamp(first_time, 0.0, duration)
        end = clamp(max(last_time, peak_time) + _BUILDUP_POST, 0.0, duration)
    else:
        if category == "shot":
            pre, post = _SHOT_PRE, _SHOT_POST
        else:
            pre, post = _DEFAULT_PRE, _DEFAULT_POST
        start = clamp(peak_time - pre, 0.0, duration)
        end = clamp(peak_time + post, 0.0, duration)
        if category == "shot":
            end = max(end, clamp(last_time + 0.5, 0.0, duration))
    if end - start < 0.1:
        end = clamp(start + min(0.5, duration), 0.0, duration)
        if end - start < 0.1:
            end = clamp(start + 0.1, 0.0, duration)
    return start, end


def _activity_profile(path: Path, sample_fps: int = _SAMPLE_FPS) -> tuple[List[float], np.ndarray, np.ndarray]:
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


def _estimate_step(times: np.ndarray, index: int) -> float:
    if times.size <= 1:
        if times.size == 1:
            return max(0.1, float(times[0]))
        return 0.3
    if index <= 0:
        return max(0.1, float(times[1] - times[0]))
    prev = float(times[index - 1])
    curr = float(times[min(index, times.size - 1)])
    step = curr - prev
    if step <= 1e-6 and index + 1 < times.size:
        step = float(times[index + 1] - curr)
    return max(0.1, step)



def score_clip(path: Path, sustain_sec: float, meta: Optional[ClipMetadata]) -> RankedClip:
    times_list, cover_arr, mag_arr = _activity_profile(path)
    times = np.asarray(times_list, dtype=np.float32)
    cover = np.asarray(cover_arr, dtype=np.float32)
    mag = np.asarray(mag_arr, dtype=np.float32)

    duration = _clip_duration(path)
    first_time, last_time = _find_active_bounds(times, cover, mag, sustain_sec, duration)
    peak_time = _peak_time(times, cover, mag, duration)
    category = _classify_event(meta)
    trim_start, trim_end = _compute_trim_bounds(category, first_time, peak_time, last_time, duration)


    inpoint = max(0.0, first_time)

    gate_start = first_time

    if times.size and cover.size and mag.size:
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
        if idx is not None and 0 <= idx < len(times):

            step = _estimate_step(times, idx)
            gate_start = max(0.0, float(times[idx]) - step)
            inpoint = max(inpoint, gate_start)

    trim_start = max(trim_start, inpoint)
    if trim_end < trim_start:
        trim_end = trim_start

    motion_ref = mag[cover > 0] if cover.size and (cover > 0).any() else mag

            if len(times) >= 2:
                if idx > 0:
                    prev_time = float(times[idx - 1])
                else:
                    prev_time = float(times[1])
                step = max(0.1, float(times[idx]) - prev_time)
            else:
                step = 0.3
            gate_start = max(gate_start, max(0.0, float(times[idx]) - step))

    trim_start = max(trim_start, gate_start)
    if trim_end - trim_start < 0.1:
        trim_end = min(duration, max(trim_start + 0.1, trim_end))
        if trim_end <= trim_start:
            trim_end = min(duration, trim_start + 0.1)

    motion_ref = mag[cover > 0] if cover.size and np.any(cover > 0) else mag

    motion_score = float(np.percentile(motion_ref, 80)) if motion_ref.size else 0.0
    audio_score = _audio_rms(path)
    score = 0.65 * motion_score + 0.35 * audio_score

    return RankedClip(
        path=path,
        inpoint=round(trim_start, 3),
        outpoint=round(trim_end, 3),
        duration=duration,
        motion=motion_score,
        audio=audio_score,
        score=score,
        event=meta.event if meta else None,
    )


def _candidate_files(paths: Iterable[Path]) -> List[Path]:
    files: List[Path] = []
    for directory in paths:
        pattern = str(directory / "clip_*.mp4")
        files.extend(Path(f).resolve() for f in sorted(glob.glob(pattern)))
    return files


def write_rankings(csv_path: Path, clips: List[RankedClip]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "inpoint", "outpoint", "duration", "motion", "audio", "score", "event"])
        for clip in clips:
            writer.writerow(
                [
                    clip.path.as_posix(),
                    f"{clip.inpoint:.3f}",
                    f"{clip.outpoint:.3f}",
                    f"{clip.duration:.3f}",
                    f"{clip.motion:.4f}",
                    f"{clip.audio:.4f}",
                    f"{clip.score:.4f}",
                    clip.event or "",
                ]
            )


def write_concat(list_path: Path, clips: List[RankedClip], max_len: float) -> None:
    list_path.parent.mkdir(parents=True, exist_ok=True)
    top_lines: List[str] = []
    with list_path.open("w", newline="\n", encoding="utf-8") as f:
        for clip in clips:
            base_out = clip.outpoint if clip.outpoint > clip.inpoint else clip.duration
            outpoint = min(base_out, clip.inpoint + max_len, clip.duration)
            file_line = f"file '{clip.path.as_posix()}'"
            in_line = f"inpoint {clip.inpoint:.3f}"
            out_line = f"outpoint {outpoint:.3f}"
            for line in (file_line, in_line, out_line):
                f.write(f"{line}\n")
            top_lines.extend([file_line, in_line, out_line])

    goals_path = list_path.parent / "concat_goals.txt"
    combined_path = list_path.parent / "concat_goals_plus_top.txt"
    if goals_path.exists():
        base_text = goals_path.read_text(encoding="utf-8")
        base_lines = base_text.splitlines()
        combined_lines = base_lines + top_lines if top_lines else base_lines
        combined_text = "\n".join(combined_lines)
        if combined_text:
            combined_text += "\n"
        combined_path.write_text(combined_text, encoding="utf-8")


def run_topk(config: AppConfig, candidate_dirs: List[Path], csv_out: Path, concat_out: Path, k: int | None = None, max_len: float | None = None) -> List[RankedClip]:
    k = k or config.rank.k
    max_len = max_len or config.rank.max_len
    files = _candidate_files(candidate_dirs)
    if not files:
        logger.warning("No candidate clips found in %s", candidate_dirs)
        return []
    metadata = _collect_metadata(candidate_dirs)
    scored = [score_clip(path, config.rank.sustain, metadata.get(path)) for path in files]
    filtered = [clip for clip in scored if clip.outpoint - clip.inpoint >= config.rank.min_tail]
    if len(filtered) < k:
        filtered = scored
    ranked = sorted(filtered, key=lambda c: c.score, reverse=True)[:k]
    write_rankings(csv_out, ranked)
    write_concat(concat_out, ranked, max_len)
    logger.info("Wrote %s and %s", csv_out, concat_out)
    return ranked

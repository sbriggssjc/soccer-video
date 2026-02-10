from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    import cv2  # type: ignore
    _cv2_error = cv2.error  # type: ignore[attr-defined]
except Exception as exc:  # pragma: no cover - runtime guard
    cv2 = None  # type: ignore
    _cv2_error = RuntimeError
    _cv2_import_error = exc
else:
    _cv2_import_error = None

from ._loguru import logger
from .config import AppConfig
from .io import VideoStreamInfo
from .utils import HighlightWindow, clamp

try:  # pragma: no cover - optional dependency
    import pytesseract  # type: ignore
except Exception:  # pragma: no cover - best effort fallback
    pytesseract = None  # type: ignore


@dataclass
class GoalSignal:
    records: List[Tuple[float, str]]

    def add(self, time: float, source: str) -> None:
        self.records.append((time, source))

    @property
    def sources(self) -> set[str]:
        return {src for _, src in self.records}

    def anchor_time(self) -> float:
        non_score = [t for t, src in self.records if src != "scoreboard"]
        score = [t for t, src in self.records if src == "scoreboard"]
        if non_score:
            anchor = float(min(non_score))
            if score:
                anchor = min(anchor, max(0.0, float(min(score)) - 1.5))
            return anchor
        earliest = float(min(t for t, _ in self.records))
        return max(0.0, earliest - 1.5)


def _merge_signals(signals: Sequence[Tuple[float, str]], tolerance: float = 4.0) -> List[GoalSignal]:
    if not signals:
        return []
    ordered = sorted(signals, key=lambda item: item[0])
    groups: List[GoalSignal] = []
    for time, source in ordered:
        placed = False
        for group in groups:
            for existing_time, existing_source in group.records:
                effective_tol = tolerance
                if "scoreboard" in {source, existing_source}:
                    effective_tol = tolerance + 2.5
                if abs(time - existing_time) <= effective_tol:
                    group.add(time, source)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            groups.append(GoalSignal(records=[(time, source)]))
    groups.sort(key=lambda g: g.anchor_time())
    return groups


def _scoreboard_rois(frame: np.ndarray) -> List[np.ndarray]:
    h, w = frame.shape[:2]
    top = max(1, int(round(h * 0.22)))
    rois = [
        frame[0:top, 0 : max(1, int(round(w * 0.35)))],
        frame[0:top, int(round(w * 0.65)) : w],
        frame[0 : max(1, int(round(h * 0.18))), int(round(w * 0.3)) : int(round(w * 0.7))],
    ]
    return [roi for roi in rois if roi.size]


def _parse_score_text(text: str) -> Optional[Tuple[int, int]]:
    cleaned = text.upper().replace("O", "0").replace("S", "5")
    cleaned = cleaned.replace("I", "1")
    pair = re.findall(r"(\d{1,2})\D+(\d{1,2})", cleaned)
    for a, b in pair:
        try:
            sa, sb = int(a), int(b)
        except ValueError:
            continue
        if sa <= 15 and sb <= 15:
            return sa, sb
    digits = [int(ch) for ch in re.findall(r"\d", cleaned)]
    digits = [d for d in digits if d <= 15]
    if len(digits) >= 2:
        return digits[0], digits[1]
    return None


def _ocr_score(roi: np.ndarray) -> Optional[Tuple[int, int]]:
    if pytesseract is None or cv2 is None:
        return None
    try:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    except _cv2_error:
        return None
    if gray.size == 0:
        return None
    gray = cv2.equalizeHist(gray)
    target = 320
    scale = max(1.0, target / max(gray.shape))
    gray = cv2.resize(gray, (int(gray.shape[1] * scale), int(gray.shape[0] * scale)), interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    config = "--psm 6 -c tessedit_char_whitelist=0123456789:-|"
    for candidate in (binary, 255 - binary):
        try:
            text = pytesseract.image_to_string(candidate, config=config)
        except Exception as exc:  # pragma: no cover - OCR best effort
            logger.debug("pytesseract failed: %s", exc)
            continue
        score = _parse_score_text(text)
        if score is not None:
            return score
    return None


def detect_scoreboard_deltas(video_path: Path, sample_rate: float = 1.5) -> List[float]:
    if pytesseract is None or cv2 is None:
        logger.debug("OCR prerequisites unavailable; skipping scoreboard delta detection")
        return []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Could not open %s for scoreboard OCR", video_path)
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps / sample_rate)))
    committed: Optional[Tuple[int, int]] = None
    pending: Optional[Tuple[int, int]] = None
    pending_count = 0
    last_event = -10.0
    events: List[float] = []
    try:
        for idx in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                break
            rois = _scoreboard_rois(frame)
            readings = [val for roi in rois if (val := _ocr_score(roi)) is not None]
            if not readings:
                continue
            if committed is not None:
                readings.sort(key=lambda s: abs(s[0] - committed[0]) + abs(s[1] - committed[1]))
            current = readings[0]
            if committed is None:
                committed = current
                continue
            if current == committed:
                pending = None
                pending_count = 0
                continue
            if pending == current:
                pending_count += 1
            else:
                pending = current
                pending_count = 1
            if pending_count < 2:
                continue
            diff0 = pending[0] - committed[0]
            diff1 = pending[1] - committed[1]
            if diff0 < 0 or diff1 < 0 or diff0 + diff1 == 0 or diff0 + diff1 > 2:
                pending = None
                pending_count = 0
                continue
            t = idx / fps
            if t - last_event >= 6.0:
                events.append(t)
                last_event = t
            committed = pending
            pending = None
            pending_count = 0
    finally:
        cap.release()
    return events


def _normalize_series(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    if arr.size == 0:
        return arr
    arr = arr - arr.min()
    maxv = float(arr.max())
    if maxv > 1e-6:
        arr /= maxv
    return arr


def _find_motion_spikes(times: Sequence[float], energy: Sequence[float], min_distance: float = 4.0) -> List[float]:
    if not times or len(energy) < 3:
        return []
    arr = _normalize_series(np.array(energy, dtype=np.float32))
    if arr.size == 0 or float(arr.max()) <= 1e-6:
        return []
    mean = float(arr.mean())
    std = float(arr.std())
    thr = min(0.95, max(0.4, mean + std * 1.5))
    peaks: List[float] = []
    for idx in range(1, len(arr) - 1):
        if arr[idx] >= thr and arr[idx] >= arr[idx - 1] and arr[idx] >= arr[idx + 1]:
            t = float(times[idx])
            if not peaks or all(abs(t - prev) >= min_distance for prev in peaks):
                peaks.append(t)
    return peaks


def detect_net_events(video_path: Path, fps: float, sample_fps: float = 15.0) -> List[float]:
    if cv2 is None:
        logger.debug("OpenCV unavailable; skipping net-region analysis")
        return []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Could not open %s for net-region analysis", video_path)
        return []
    step = max(1, int(round(fps / max(sample_fps, 1.0))))
    ok, frame = cap.read()
    if not ok:
        cap.release()
        return []
    small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    prev_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    frame_idx = step
    times: List[float] = []
    left_energy: List[float] = []
    right_energy: List[float] = []
    try:
        while True:
            grabbed = True
            for _ in range(step - 1):
                grabbed = cap.grab()
                if not grabbed:
                    break
            if not grabbed:
                break
            ok, frame = cap.read()
            if not ok:
                break
            small = cv2.resize(frame, (prev_gray.shape[1], prev_gray.shape[0]))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray, prev_gray).astype(np.float32)
            prev_gray = gray
            h, w = diff.shape
            border = max(8, int(round(w * 0.18)))
            left = diff[:, :border]
            right = diff[:, w - border :]
            left_energy.append(float(left.mean()) / 255.0)
            right_energy.append(float(right.mean()) / 255.0)
            times.append(frame_idx / fps)
            frame_idx += step
    finally:
        cap.release()
    spikes = _find_motion_spikes(times, left_energy) + _find_motion_spikes(times, right_energy)
    spikes.sort()
    merged: List[float] = []
    for t in spikes:
        if not merged or t - merged[-1] >= 2.0:
            merged.append(t)
    return merged


def detect_crowd_spikes(video_path: Path, hop_s: float = 0.12, win_s: float = 0.5) -> List[float]:
    try:  # pragma: no cover - optional dependency
        import librosa
    except Exception:
        logger.debug("librosa unavailable; skipping crowd spike detection")
        return []
    try:
        y, sr = librosa.load(str(video_path), sr=None, mono=True)
    except Exception as exc:  # pragma: no cover - audio best effort
        logger.debug("Failed to load audio for crowd spikes: %s", exc)
        return []
    if y.size == 0:
        return []
    hop = max(1, int(sr * hop_s))
    win = max(hop, int(sr * win_s))
    rms = librosa.feature.rms(y=y, frame_length=win, hop_length=hop)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop)
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    if times.size == 0:
        return []
    bw = bandwidth[0] if bandwidth.ndim > 1 else bandwidth
    score = 0.65 * _normalize_series(rms) + 0.35 * _normalize_series(bw)
    mean = float(score.mean())
    std = float(score.std())
    thr = min(0.95, max(0.45, mean + std * 1.2))
    peaks: List[float] = []
    for idx in range(1, len(score) - 1):
        if score[idx] >= thr and score[idx] >= score[idx - 1] and score[idx] >= score[idx + 1]:
            t = float(times[idx])
            if not peaks or all(abs(t - prev) >= 6.0 for prev in peaks):
                peaks.append(t)
    return peaks


def detect_goal_windows(
    config: AppConfig,
    video_path: Path,
    info: VideoStreamInfo,
    windows: Sequence[HighlightWindow],
) -> List[HighlightWindow]:
    scoreboard = detect_scoreboard_deltas(video_path)
    net = detect_net_events(video_path, info.fps)
    crowd = detect_crowd_spikes(video_path)
    signals: List[Tuple[float, str]] = []
    signals.extend((t, "scoreboard") for t in scoreboard)
    signals.extend((t, "net") for t in net)
    signals.extend((t, "crowd") for t in crowd)
    groups = _merge_signals(signals)
    forced: List[HighlightWindow] = []
    used: List[float] = []
    for group in groups:
        anchor = group.anchor_time()
        if any(abs(anchor - u) <= 1.0 for u in used):
            continue
        used.append(anchor)
        matched = None
        for win in windows:
            if win.start - 4.0 <= anchor <= win.end + 4.0:
                matched = win
                break
        if matched is not None:
            # Create a replacement window instead of mutating the caller's data.
            idx = list(windows).index(matched)
            replacement = HighlightWindow(
                start=matched.start,
                end=matched.end,
                score=max(matched.score, 1.0),
                event="goal",
            )
            forced.append(replacement)
        else:
            start = clamp(anchor - config.detect.pre, 0.0, info.duration)
            end = clamp(anchor + config.detect.post, 0.0, info.duration)
            forced.append(HighlightWindow(start=start, end=end, score=1.0, event="goal"))
    if groups:
        logger.info(
            "Goal signals merged: scoreboard=%d net=%d crowd=%d -> %d events",
            len(scoreboard),
            len(net),
            len(crowd),
            len(groups),
        )
    return forced

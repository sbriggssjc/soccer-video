"""Utilities for loading pre-computed ball telemetry.

The follow pipeline already produces several flavours of telemetry JSON/CSV
files depending on which detector or tracker ran beforehand.  This module
normalises the most common formats into a single :class:`BallSample` dataclass
so downstream tools (offline planners, renderers, debuggers) can consume the
data without duplicating parsing logic.

Supported inputs
-----------------

* ``*.telemetry.jsonl`` written by :mod:`tools.render_follow_unified`.
* ``*.ball_path.jsonl`` produced by the various ball trackers.
* CSV exports containing ``t``/``frame``/``x``/``y`` columns.

The loader performs the following steps:

1. Discover a candidate telemetry file for a given atomic clip.
2. Parse each record, extracting time/frame indexes, pixel coordinates and a
   soft confidence score if present.
3. Emit :class:`BallSample` instances with sane defaults (frame-relative time
   and ``NaN`` coordinates when a sample is missing).

The helpers intentionally avoid any detector-specific imports so they can run
in lightweight orchestration scripts such as :mod:`tools.follow_pipeline`.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class BallSample:
    """Single telemetry sample in source pixel space."""

    t: float
    frame_idx: int
    x: float
    y: float
    conf: float = 1.0

    def is_finite(self) -> bool:
        return math.isfinite(self.x) and math.isfinite(self.y)


def _maybe_path(text: Optional[str | Path]) -> Optional[Path]:
    if text is None:
        return None
    p = Path(text)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    return p


def _coerce_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _coerce_int(value: object, default: int) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _extract_xy(rec: object) -> Tuple[float, float]:
    if not isinstance(rec, dict):
        return float("nan"), float("nan")
    # Priority order roughly mirrors render telemetry structure.
    key_pairs: Sequence[Tuple[str, str]] = (
        ("bx_stab", "by_stab"),
        ("bx", "by"),
        ("ball", "ball"),
        ("ball_src", "ball_src"),
        ("ball_path", "ball_path"),
        ("ball_out", "ball_out"),
        ("u", "v"),
    )
    for key_x, key_y in key_pairs:
        if key_x not in rec or key_y not in rec:
            continue
        vx = rec.get(key_x)
        vy = rec.get(key_y)
        if key_x == key_y and isinstance(vx, Sequence):
            vals = list(vx)
            if len(vals) >= 2:
                vx, vy = vals[0], vals[1]
        elif isinstance(vx, Sequence) and not isinstance(vx, (bytes, str)):
            vx = vx[0]
        if isinstance(vy, Sequence) and not isinstance(vy, (bytes, str)):
            vy = vy[0]
        x = _coerce_float(vx)
        y = _coerce_float(vy)
        if math.isfinite(x) and math.isfinite(y):
            return x, y
    return float("nan"), float("nan")


def _extract_time(rec: dict, default_t: float) -> float:
    for key in ("t", "time", "timestamp", "ts"):
        if key in rec:
            return _coerce_float(rec.get(key))
    return default_t


def _extract_conf(rec: dict) -> float:
    for key in ("ball_conf", "conf", "confidence", "score", "p"):
        if key in rec:
            val = _coerce_float(rec.get(key))
            if math.isfinite(val):
                return max(0.0, min(1.0, val))
    return 1.0


def _extract_frame(rec: dict, fallback: int) -> int:
    for key in ("frame", "frame_idx", "f", "idx"):
        if key in rec:
            return _coerce_int(rec.get(key), fallback)
    return fallback


def _iter_jsonl(path: Path) -> Iterator[BallSample]:
    frame_counter = 0
    t_counter = 0.0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                frame_counter += 1
                t_counter += 1.0 / 30.0
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                frame_counter += 1
                t_counter += 1.0 / 30.0
                continue
            if not isinstance(rec, dict):
                frame_counter += 1
                t_counter += 1.0 / 30.0
                continue
            frame_idx = _extract_frame(rec, frame_counter)
            t_val = _extract_time(rec, t_counter)
            conf = _extract_conf(rec)
            x, y = _extract_xy(rec)
            yield BallSample(t=t_val, frame_idx=frame_idx, x=x, y=y, conf=conf)
            frame_counter = frame_idx + 1
            t_counter = t_val + 1.0 / 30.0


def _iter_csv(path: Path) -> Iterator[BallSample]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        frame_counter = 0
        for row in reader:
            if not row:
                frame_counter += 1
                continue
            frame_idx = _extract_frame(row, frame_counter)
            t_val = _extract_time(row, frame_idx / 30.0)
            conf = _extract_conf(row)
            x = _coerce_float(row.get("x") or row.get("bx") or row.get("u"))
            y = _coerce_float(row.get("y") or row.get("by") or row.get("v"))
            yield BallSample(t=t_val, frame_idx=frame_idx, x=x, y=y, conf=conf)
            frame_counter = frame_idx + 1


def load_ball_telemetry(path: Path) -> List[BallSample]:
    """Load samples from ``path`` regardless of the supported container."""

    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    samples = list(_iter_csv(path)) if suffix == ".csv" else list(_iter_jsonl(path))
    logging.debug("Loaded %s telemetry samples from %s", len(samples), path)
    return samples


def discover_telemetry_paths(clip_path: Path, match_key: str = "") -> List[Path]:
    """Return candidate telemetry paths sorted by priority."""

    stem = clip_path.stem
    work_root = REPO_ROOT / "out" / "autoframe_work"
    candidates: List[Path] = []
    parent = clip_path.with_suffix("")
    suffixes = [
        ".telemetry.jsonl",
        ".ball_path.jsonl",
        ".ball.jsonl",
        ".jsonl",
    ]
    for suf in suffixes:
        candidates.append(clip_path.with_suffix(suf))
    candidates.append(parent.with_suffix(".telemetry.jsonl"))
    candidates.append(parent.with_suffix(".ball_path.jsonl"))
    if match_key:
        maybe_match = REPO_ROOT / "out" / "telemetry" / match_key
        candidates.append(maybe_match / f"{stem}.telemetry.jsonl")
    candidates.append(work_root / stem / f"{stem}.telemetry.jsonl")
    candidates.append(work_root / stem / f"{stem}.ball_path.jsonl")
    candidates.append(work_root / stem / "ball_path.jsonl")
    unique: List[Path] = []
    seen = set()
    for cand in candidates:
        if cand is None:
            continue
        if cand in seen:
            continue
        seen.add(cand)
        unique.append(cand)
    return unique


def load_ball_telemetry_for_clip(
    clip_path: str | Path,
    *,
    match_key: str = "",
    telemetry_path: Optional[str | Path] = None,
) -> Tuple[Optional[List[BallSample]], Optional[Path]]:
    """Try to load telemetry for ``clip_path``.

    Returns ``(samples, path)`` when successful, or ``(None, None)`` when no
    telemetry is available.
    """

    clip = Path(clip_path)
    if telemetry_path:
        cand = _maybe_path(telemetry_path)
        if not cand or not cand.exists():
            logger.warning("Telemetry override %s not found", telemetry_path)
            return None, None
        return load_ball_telemetry(cand), cand

    for candidate in discover_telemetry_paths(clip, match_key=match_key):
        if candidate.exists():
            try:
                samples = load_ball_telemetry(candidate)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to load telemetry %s: %s", candidate, exc)
                continue
            if samples:
                return samples, candidate
    return None, None


def telemetry_is_usable(
    samples: Optional[Sequence[BallSample]],
    *,
    min_count: int = 24,
    min_avg_conf: float = 0.35,
) -> bool:
    if not samples:
        return False
    finite = [s for s in samples if s.is_finite()]
    if len(finite) < min_count:
        return False
    avg_conf = sum(max(0.0, min(1.0, s.conf)) for s in finite) / max(len(finite), 1)
    return avg_conf >= min_avg_conf


def summarise(samples: Sequence[BallSample]) -> str:
    finite = sum(1 for s in samples if s.is_finite())
    avg_conf = 0.0
    if samples:
        avg_conf = sum(max(0.0, min(1.0, s.conf)) for s in samples) / len(samples)
    return f"{finite} samples, avg conf {avg_conf:.2f}"


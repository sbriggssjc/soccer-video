"""Ball telemetry loader used by follow/portrait planners."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping


# ---------------------------------------------------------------------------
# Data model


@dataclass
class BallSample:
    """Single telemetry sample in source pixel coordinates."""

    t: float
    frame: int
    x: float
    y: float
    conf: float = 1.0

    @property
    def frame_idx(self) -> int:
        """Backwards-compatible alias for older callers."""

        return self.frame


# ---------------------------------------------------------------------------
# Parsing helpers


_X_KEYS = (
    "ball_x",
    "ballx",
    "ball_x_px",
    "bx",
    "x",
    "u",
    "ball",
)
_Y_KEYS = (
    "ball_y",
    "bally",
    "ball_y_px",
    "by",
    "y",
    "v",
    "ball",
)
_CONF_KEYS = ("ball_conf", "conf", "confidence", "score", "p")
_TIME_KEYS = ("t", "time", "timestamp", "ts")
_FRAME_KEYS = ("frame", "frame_idx", "idx", "f")
_FPS_KEYS = ("fps", "frame_rate", "frameRate", "video_fps")


def _as_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _as_int(value: object, fallback: int) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return fallback


def _extract_xy(rec: Mapping[str, object]) -> tuple[float, float]:
    for key_x, key_y in zip(_X_KEYS, _Y_KEYS):
        if key_x not in rec or key_y not in rec:
            continue
        val_x = rec.get(key_x)
        val_y = rec.get(key_y)
        if key_x == key_y and isinstance(val_x, (list, tuple)):
            if len(val_x) >= 2:
                val_x, val_y = val_x[0], val_x[1]
            else:
                continue
        if isinstance(val_x, (list, tuple)):
            val_x = val_x[0] if val_x else None
        if isinstance(val_y, (list, tuple)):
            val_y = val_y[0] if val_y else None
        x = _as_float(val_x)
        y = _as_float(val_y)
        if math.isfinite(x) and math.isfinite(y):
            return x, y
    return float("nan"), float("nan")


def _extract_conf(rec: Mapping[str, object]) -> float:
    for key in _CONF_KEYS:
        if key in rec:
            val = _as_float(rec.get(key))
            if math.isfinite(val):
                return max(0.0, min(1.0, val))
    return 1.0


def _extract_time(rec: Mapping[str, object]) -> float:
    for key in _TIME_KEYS:
        if key in rec:
            val = _as_float(rec.get(key))
            if math.isfinite(val):
                return val
    return float("nan")


def _extract_frame(rec: Mapping[str, object], fallback: int) -> int:
    for key in _FRAME_KEYS:
        if key in rec:
            return _as_int(rec.get(key), fallback)
    return fallback


def _extract_fps(rec: Mapping[str, object], prev: float | None) -> float | None:
    for key in _FPS_KEYS:
        if key in rec:
            val = _as_float(rec.get(key))
            if math.isfinite(val) and val > 0:
                return val
    return prev


def _finalise_sample(
    *,
    t_val: float,
    frame: int,
    x: float,
    y: float,
    conf: float,
    fps_hint: float | None,
) -> BallSample:
    if not math.isfinite(t_val):
        if fps_hint and fps_hint > 0:
            t_val = frame / fps_hint
        else:
            t_val = 0.0
    if not (math.isfinite(x) and math.isfinite(y)):
        conf = 0.0
        x = float("nan")
        y = float("nan")
    return BallSample(t=float(t_val), frame=int(frame), x=float(x), y=float(y), conf=float(conf))


def _iter_jsonl(path: Path) -> Iterator[BallSample]:
    fps_hint: float | None = None
    frame_counter = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                frame_counter += 1
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                frame_counter += 1
                continue
            if not isinstance(rec, Mapping):
                frame_counter += 1
                continue
            fps_hint = _extract_fps(rec, fps_hint)
            frame_idx = _extract_frame(rec, frame_counter)
            t_val = _extract_time(rec)
            conf = _extract_conf(rec)
            x, y = _extract_xy(rec)
            yield _finalise_sample(
                t_val=t_val,
                frame=frame_idx,
                x=x,
                y=y,
                conf=conf,
                fps_hint=fps_hint,
            )
            frame_counter = frame_idx + 1


def _iter_csv(path: Path) -> Iterator[BallSample]:
    fps_hint: float | None = None
    frame_counter = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row:
                frame_counter += 1
                continue
            fps_hint = _extract_fps(row, fps_hint)
            frame_idx = _extract_frame(row, frame_counter)
            t_val = _extract_time(row)
            conf = _extract_conf(row)
            x, y = _extract_xy(row)
            yield _finalise_sample(
                t_val=t_val,
                frame=frame_idx,
                x=x,
                y=y,
                conf=conf,
                fps_hint=fps_hint,
            )
            frame_counter = frame_idx + 1


def load_ball_telemetry(path: str | Path) -> list[BallSample]:
    """Load telemetry samples from ``path`` regardless of container type."""

    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        samples = list(_iter_csv(file_path))
    else:
        samples = list(_iter_jsonl(file_path))
    return samples


# ---------------------------------------------------------------------------
# Clip discovery


def _candidate_paths(clip_path: Path) -> Iterable[Path]:
    base = clip_path
    yield base.with_suffix(".ball_path.jsonl")
    yield base.with_suffix(".telemetry.jsonl")

    parent = base.parent
    if parent.is_dir():
        suffixes = (".jsonl", ".csv")
        pattern = f"{base.stem}*"
        seen: set[Path] = set()
        for suffix in suffixes:
            for candidate in sorted(parent.glob(pattern + suffix)):
                if candidate in seen:
                    continue
                seen.add(candidate)
                yield candidate


def load_ball_telemetry_for_clip(atomic_path: str) -> list[BallSample]:
    """Discover and load telemetry for ``atomic_path`` if present."""

    clip_path = Path(atomic_path)
    samples: list[BallSample] = []
    for candidate in _candidate_paths(clip_path):
        if not candidate.is_file():
            continue
        try:
            samples = load_ball_telemetry(candidate)
        except Exception:  # noqa: BLE001 - best-effort loader
            continue
        if samples:
            print(f"[BALL] Loaded {len(samples)} ball samples for {clip_path}")
            return samples
    print(f"[BALL] No ball telemetry found for {clip_path}")
    return []


__all__ = ["BallSample", "load_ball_telemetry", "load_ball_telemetry_for_clip"]


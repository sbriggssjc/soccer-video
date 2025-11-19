"""Ball telemetry loader and CLI utilities.

Telemetry JSONL schema (per line)::

    {"t": <seconds>, "x": <pixels>, "y": <pixels>, "confidence": <0-1>}

Canonical path mapping: ``out/telemetry/<basename>.ball.jsonl`` derived via
``telemetry_path_for_video``.  Example PowerShell workflow::

    python tools/ball_telemetry.py detect --video C:\\path\\to\\clip.mp4
    python tools/ball_telemetry.py annotate --video C:\\path\\to\\clip.mp4
    python tools/render_follow_unified.py --preset wide_follow --in C:\\path\\to\\clip.mp4 --use-ball-telemetry
"""

from __future__ import annotations

import csv
import json
import math
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence


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


def telemetry_path_for_video(video_path: str | Path) -> str:
    """Return the canonical telemetry path for ``video_path``.

    The telemetry lives under ``out/telemetry`` with a ``.ball.jsonl`` suffix
    and mirrors the source basename.
    """

    video_path = Path(video_path)
    stem = video_path.stem
    out_dir = Path("out") / "telemetry"
    return str(out_dir / f"{stem}.ball.jsonl")


def load_ball_telemetry_for_clip(atomic_path: str) -> list[BallSample]:
    """Discover and load telemetry for ``atomic_path`` if present."""

    clip_path = Path(atomic_path)
    samples: list[BallSample] = []

    default_path = Path(telemetry_path_for_video(clip_path))
    candidates = list(_candidate_paths(clip_path))
    if default_path not in candidates:
        candidates.insert(0, default_path)

    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            samples = load_ball_telemetry(candidate)
        except Exception:  # noqa: BLE001 - best-effort loader
            continue
        if samples:
            t_vals = [s.t for s in samples if math.isfinite(s.t)]
            t_min = min(t_vals) if t_vals else 0.0
            t_max = max(t_vals) if t_vals else 0.0
            print(
                f"[BALL] Loaded {len(samples)} ball samples from {candidate} (t={t_min:.2f}–{t_max:.2f}s)"
            )
            return samples
    print(f"[BALL] No ball telemetry found for {clip_path}")
    return []


def save_ball_telemetry_jsonl(out_path: Path, samples: Sequence[Mapping[str, object]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for rec in samples:
            handle.write(json.dumps(rec) + "\n")


def _detect_ball(args: argparse.Namespace) -> int:
    video_path = Path(args.video).expanduser()
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[BALL] No built-in detector available; please annotate manually ({video_path})")
    return 1


def _annotate_ball(args: argparse.Namespace) -> int:
    import cv2

    video_path = Path(args.video).expanduser()
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}", file=sys.stderr)
        return 2

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    positions: list[tuple[int, float, float]] = []
    clicks: dict[int, tuple[float, float]] = {}

    window_name = "Annotate ball (ESC to quit, SPACE/ENTER next frame, click to mark)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    state = {"current_frame": 0}

    def on_mouse(event, x, y, *_args):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks[state["current_frame"]] = (float(x), float(y))
            print(f"[BALL] Frame {state['current_frame']} → ({x}, {y})")

    cv2.setMouseCallback(window_name, on_mouse)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        state["current_frame"] = frame_idx
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(0) & 0xFF
        if key in (27,):  # ESC
            break
        frame_idx += args.step
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    cap.release()
    cv2.destroyAllWindows()

    if not clicks:
        print("[WARN] No annotations captured; nothing to write")
        return 0

    frames_sorted = sorted(clicks.keys())
    first_frame, last_frame = frames_sorted[0], frames_sorted[-1]
    samples: list[dict[str, object]] = []

    def interp(v0: float, v1: float, alpha: float) -> float:
        return v0 + (v1 - v0) * alpha

    for idx, frame in enumerate(range(first_frame, last_frame + 1)):
        if frame in clicks:
            x, y = clicks[frame]
        else:
            prev_frames = [f for f in frames_sorted if f < frame]
            next_frames = [f for f in frames_sorted if f > frame]
            if not prev_frames or not next_frames:
                continue
            f0 = max(prev_frames)
            f1 = min(next_frames)
            if f1 == f0:
                continue
            alpha = (frame - f0) / float(f1 - f0)
            x0, y0 = clicks[f0]
            x1, y1 = clicks[f1]
            x = interp(x0, x1, alpha)
            y = interp(y0, y1, alpha)
        t = frame / fps if fps > 0 else 0.0
        samples.append({"t": float(t), "x": float(x), "y": float(y), "source": "manual"})

    save_ball_telemetry_jsonl(out_path, samples)

    t_vals = [rec["t"] for rec in samples if isinstance(rec.get("t"), (int, float))]
    t_min = min(t_vals) if t_vals else 0.0
    t_max = max(t_vals) if t_vals else 0.0
    print(
        f"[BALL] Wrote {len(samples)} samples covering t={t_min:.2f}–{t_max:.2f}s to {out_path}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ball telemetry utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    detect_p = sub.add_parser("detect", help="Auto-detect ball positions from video")
    detect_p.add_argument("--video", required=True, help="Input video path")
    detect_p.add_argument("--out", help="Output telemetry JSONL (default derived)")
    detect_p.add_argument("--sport", default="soccer", help="Sport context (default: soccer)")
    detect_p.set_defaults(func=_detect_ball)

    ann_p = sub.add_parser("annotate", help="Manually annotate ball positions")
    ann_p.add_argument("--video", required=True, help="Input video path")
    ann_p.add_argument("--out", help="Output telemetry JSONL (default derived)")
    ann_p.add_argument("--step", type=int, default=2, help="Advance this many frames per step (default: 2)")
    ann_p.set_defaults(func=_annotate_ball)

    args = parser.parse_args(argv)
    if not getattr(args, "out", None):
        args.out = telemetry_path_for_video(Path(args.video))

    return args.func(args)


__all__ = [
    "BallSample",
    "load_ball_telemetry",
    "load_ball_telemetry_for_clip",
    "telemetry_path_for_video",
    "save_ball_telemetry_jsonl",
]


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


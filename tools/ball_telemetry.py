"""Ball telemetry loader and CLI utilities.

Telemetry JSONL schema (per line)::

    {"frame": <int>, "t": <seconds>, "cx": <pixels>, "cy": <pixels>}

Canonical path mapping: ``out/telemetry/<basename>.ball.jsonl`` derived via
``telemetry_path_for_video``.  Example PowerShell workflow::

    python tools/ball_telemetry.py detect --video C:\\path\\to\\clip.mp4
    python tools/ball_telemetry.py annotate --video C:\\path\\to\\clip.mp4
    python tools/render_follow_unified.py --preset wide_follow --in C:\\path\\to\\clip.mp4 --use-ball-telemetry
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass

# Ensure stdout can handle Unicode on Windows (cp1252 default chokes on
# em-dashes, arrows, etc. in diagnostic messages).
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(errors="replace")
    except Exception:
        pass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Optional, Sequence, Tuple

import numpy as np


DEFAULT_MODEL_NAME = "yolov8n.pt"
DEFAULT_MIN_CONF = 0.35

# Source label constants (must match local definitions in fuse_yolo_and_centroid)
FUSE_TRACKER = 7  # CSRT visual tracker (real pixel data, not estimated)


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

    @property
    def cx(self) -> float:
        return self.x

    @property
    def cy(self) -> float:
        return self.y


@dataclass
class PersonBox:
    """Single person detection bounding box in source pixel coordinates."""

    frame: int
    cx: float
    cy: float
    w: float
    h: float
    conf: float = 0.5


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
    "cx",
)
_Y_KEYS = (
    "ball_y",
    "bally",
    "ball_y_px",
    "by",
    "y",
    "v",
    "ball",
    "cy",
)
_CONF_KEYS = ("ball_conf", "conf", "confidence", "score", "p")
_TIME_KEYS = ("t", "time", "timestamp", "ts")
_FRAME_KEYS = ("frame", "frame_idx", "idx", "f")
_FPS_KEYS = ("fps", "frame_rate", "frameRate", "video_fps")


_FRAME_BOUNDS_HINT: tuple[float, float] | None = None


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


def _extract_point(rec: Mapping[str, object], prefix: str) -> tuple[float, float] | None:
    x_key = f"{prefix}_x"
    y_key = f"{prefix}_y"
    if x_key not in rec or y_key not in rec:
        return None
    x = _as_float(rec.get(x_key))
    y = _as_float(rec.get(y_key))
    if math.isfinite(x) and math.isfinite(y):
        return x, y
    return None


def _extract_conf_key(rec: Mapping[str, object], key: str) -> float | None:
    if key not in rec:
        return None
    val = _as_float(rec.get(key))
    if math.isfinite(val):
        return max(0.0, min(1.0, val))
    return None


def _extract_xy_with_source(rec: Mapping[str, object]) -> tuple[float, float, float]:
    """Prefer ball/carrier coordinates using the declared source when present."""

    is_valid = rec.get("is_valid")
    if isinstance(is_valid, bool) and not is_valid:
        return float("nan"), float("nan"), 0.0

    source = str(rec.get("source") or "").lower()

    ball_pt = _extract_point(rec, "ball")
    carrier_pt = _extract_point(rec, "carrier")
    ball_conf = _extract_conf_key(rec, "ball_conf")
    carrier_conf = _extract_conf_key(rec, "carrier_conf")

    if source == "ball":
        candidates = ((ball_pt, ball_conf), (carrier_pt, carrier_conf))
    elif source == "carrier":
        candidates = ((carrier_pt, carrier_conf), (ball_pt, ball_conf))
    elif source == "players":
        candidates = ((carrier_pt, carrier_conf), (ball_pt, ball_conf))
    else:
        candidates = ((ball_pt, ball_conf), (carrier_pt, carrier_conf))

    for pt, conf_val in candidates:
        if pt is None:
            continue
        x, y = pt
        if math.isfinite(x) and math.isfinite(y):
            conf = conf_val if conf_val is not None else _extract_conf(rec)
            return x, y, conf

    # Fall back to legacy key search
    x, y = _extract_xy(rec)
    conf = _extract_conf(rec)
    return x, y, conf


def set_telemetry_frame_bounds(width: float | int, height: float | int) -> None:
    """Provide a frame-size hint for interpolation clamping."""

    global _FRAME_BOUNDS_HINT
    try:
        w = float(width)
        h = float(height)
    except (TypeError, ValueError):
        return
    if not (math.isfinite(w) and math.isfinite(h)):
        return
    _FRAME_BOUNDS_HINT = (max(1.0, w), max(1.0, h))


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
    if math.isfinite(x) and math.isfinite(y):
        x, y = _clamp_xy(float(x), float(y))
    else:
        conf = 0.0
        x = float("nan")
        y = float("nan")
    return BallSample(t=float(t_val), frame=int(frame), x=float(x), y=float(y), conf=float(conf))


def smooth_telemetry(samples: list[dict], window: int = 5) -> list[dict]:
    """
    samples: list of dicts with keys: frame, t, cx, cy
    returns: new list with cx, cy smoothed by a centered moving average
    """

    if not samples:
        return samples

    n = len(samples)
    half = window // 2
    out: list[dict] = []
    for i in range(n):
        j0 = max(0, i - half)
        j1 = min(n, i + half + 1)
        sx = sum(s["cx"] for s in samples[j0:j1]) / (j1 - j0)
        sy = sum(s["cy"] for s in samples[j0:j1]) / (j1 - j0)
        s = dict(samples[i])
        s["cx"] = sx
        s["cy"] = sy
        out.append(s)
    return out


def _coerce_sample_dict(sample: Mapping[str, object] | BallSample) -> dict[str, float]:
    frame = _as_int(sample.get("frame") if isinstance(sample, Mapping) else getattr(sample, "frame", 0), 0)
    cx = _as_float(sample.get("cx") if isinstance(sample, Mapping) else getattr(sample, "x", float("nan")))
    cy = _as_float(sample.get("cy") if isinstance(sample, Mapping) else getattr(sample, "y", float("nan")))
    t_val = _as_float(sample.get("t") if isinstance(sample, Mapping) else getattr(sample, "t", 0.0))
    return {"frame": frame, "t": t_val if math.isfinite(t_val) else 0.0, "cx": cx, "cy": cy}


def clean_ball_telemetry(
    samples: Sequence[Mapping[str, object] | BallSample], max_jump_px: float = 200.0, min_run_length: int = 2
) -> list[dict[str, float]]:
    """
    Remove single-frame spikes from dense telemetry by interpolating isolated outliers.

    The cleaner only activates when the telemetry frames form a contiguous run so sparse
    detections remain untouched.
    """

    if not samples:
        return []

    coerced = [_coerce_sample_dict(s) for s in samples]

    frames = sorted(s["frame"] for s in coerced)
    if len(frames) < 3:
        return coerced

    # Require dense coverage so we do not reshape sparse telemetry.
    if any(b - a != 1 for a, b in zip(frames, frames[1:])):
        return coerced

    cleaned = [dict(s) for s in sorted(coerced, key=lambda s: s["frame"])]
    for i in range(1, len(cleaned) - 1):
        prev_rec = cleaned[i - 1]
        curr_rec = cleaned[i]
        next_rec = cleaned[i + 1]

        dx_prev = curr_rec["cx"] - prev_rec["cx"]
        dy_prev = curr_rec["cy"] - prev_rec["cy"]
        dx_next = next_rec["cx"] - curr_rec["cx"]
        dy_next = next_rec["cy"] - curr_rec["cy"]

        dist_prev = math.hypot(dx_prev, dy_prev)
        dist_next = math.hypot(dx_next, dy_next)
        bridge_dist = math.hypot(next_rec["cx"] - prev_rec["cx"], next_rec["cy"] - prev_rec["cy"])

        run_len = next_rec["frame"] - prev_rec["frame"]

        if (
            run_len <= min_run_length
            and dist_prev > max_jump_px
            and dist_next > max_jump_px
            and bridge_dist <= max_jump_px
        ):
            span = next_rec["frame"] - prev_rec["frame"]
            alpha = 0.5 if span <= 0 else (curr_rec["frame"] - prev_rec["frame"]) / float(span)
            curr_rec["cx"] = prev_rec["cx"] + (next_rec["cx"] - prev_rec["cx"]) * alpha
            curr_rec["cy"] = prev_rec["cy"] + (next_rec["cy"] - prev_rec["cy"]) * alpha

    return cleaned


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
            x, y, conf = _extract_xy_with_source(rec)
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
    cleaned_samples = clean_ball_telemetry(samples)
    cleaned: list[dict[str, float]] = []
    for rec in cleaned_samples:
        frame_idx = _as_int(rec.get("frame") if isinstance(rec, Mapping) else None, 0)
        t_val = _as_float(rec.get("t") if isinstance(rec, Mapping) else None)
        if not math.isfinite(t_val):
            t_val = 0.0
        cx_val = _as_float(rec.get("cx") if isinstance(rec, Mapping) else None)
        cy_val = _as_float(rec.get("cy") if isinstance(rec, Mapping) else None)
        if not (math.isfinite(cx_val) and math.isfinite(cy_val)):
            continue
        cx_val, cy_val = _clamp_xy(float(cx_val), float(cy_val))
        cleaned.append({"frame": frame_idx, "t": float(t_val), "cx": float(cx_val), "cy": float(cy_val)})

    with out_path.open("w", encoding="utf-8") as handle:
        for rec in cleaned:
            handle.write(json.dumps(rec) + "\n")


def _get_video_fps(video_path: Path) -> float:
    """Return a reliable FPS using OpenCV first, then ffprobe as fallback."""

    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            fps_val = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            cap.release()
            if math.isfinite(fps_val) and fps_val > 0:
                return fps_val
    except Exception:  # noqa: BLE001 - best-effort only
        pass

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=r_frame_rate",
                "-of",
                "default=nokey=1:noprint_wrappers=1",
                str(video_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        rate = result.stdout.strip()
        if "/" in rate:
            num, den = rate.split("/", 1)
            fps_val = float(num) / float(den)
        else:
            fps_val = float(rate)
        if math.isfinite(fps_val) and fps_val > 0:
            return fps_val
    except Exception:  # noqa: BLE001 - ffprobe optional
        pass

    return 30.0


def _clamp_xy(cx: float, cy: float) -> tuple[float, float]:
    if _FRAME_BOUNDS_HINT is None:
        return cx, cy
    width, height = _FRAME_BOUNDS_HINT
    cx = max(0.0, min(width, cx))
    cy = max(0.0, min(height, cy))
    return cx, cy


def load_and_interpolate_telemetry(path: str, total_frames: int, fps: float) -> list[dict]:
    """
    Load JSONL telemetry and return a list of length ``total_frames``.

    For each frame i, output dict with keys:
        - frame, t, cx, cy, has_ball (bool)
    Interpolate linearly between observed points where needed.
    For leading/trailing gaps, hold the nearest known value.
    Ensure all cx, cy are finite and clamped to frame bounds.
    """

    fps = float(fps) if math.isfinite(fps) and fps > 0 else 30.0
    total_frames = max(0, int(total_frames))
    observations: dict[int, tuple[float, float, float]] = {}

    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(rec, Mapping):
                    continue
                frame_idx = _as_int(rec.get("frame"), 0)
                cx = _as_float(rec.get("cx"))
                cy = _as_float(rec.get("cy"))
                t_val = _as_float(rec.get("t"))
                if not math.isfinite(t_val):
                    t_val = frame_idx / fps
                if not (math.isfinite(cx) and math.isfinite(cy)):
                    continue
                cx, cy = _clamp_xy(float(cx), float(cy))
                observations[frame_idx] = (float(cx), float(cy), float(t_val))
    except FileNotFoundError:
        return [
            {"frame": i, "t": i / fps, "cx": 0.0, "cy": 0.0, "has_ball": False}
            for i in range(total_frames)
        ]

    if not observations:
        print(f"[BALL] Telemetry empty or invalid at {path}; interpolation skipped")
        return [
            {"frame": i, "t": i / fps, "cx": 0.0, "cy": 0.0, "has_ball": False}
            for i in range(total_frames)
        ]

    frames_sorted = sorted(observations.keys())
    seen_frames = set(frames_sorted)
    print(
        f"[BALL] Loaded telemetry for {len(seen_frames)} frames; interpolating to {total_frames} frames"
    )

    interpolated: list[dict] = []
    used_interp = False

    for i in range(total_frames):
        if i in observations:
            cx, cy, t_val = observations[i]
            has_ball = True
        else:
            pos = 0
            while pos < len(frames_sorted) and frames_sorted[pos] < i:
                pos += 1
            if pos <= 0:
                ref = frames_sorted[0]
                cx, cy, t_val = observations[ref]
            elif pos >= len(frames_sorted):
                ref = frames_sorted[-1]
                cx, cy, t_val = observations[ref]
            else:
                f0 = frames_sorted[pos - 1]
                f1 = frames_sorted[pos]
                cx0, cy0, _t0 = observations[f0]
                cx1, cy1, _t1 = observations[f1]
                alpha = 0.0 if f1 == f0 else (i - f0) / float(f1 - f0)
                cx = cx0 + (cx1 - cx0) * alpha
                cy = cy0 + (cy1 - cy0) * alpha
                t_val = i / fps
            has_ball = False
            used_interp = True

        cx, cy = _clamp_xy(float(cx), float(cy))
        t_val = float(i / fps if not math.isfinite(t_val) else t_val)
        interpolated.append({"frame": i, "t": t_val, "cx": cx, "cy": cy, "has_ball": has_ball})

    if used_interp:
        print("[BALL] Interpolated telemetry gaps for smoother follow path")

    return interpolated


def _render_debug_overlay(video_path: Path, telemetry_path: Path, overlay_out: Path, *, fps: float | None = None) -> None:
    try:
        import cv2
    except Exception:  # noqa: BLE001 - optional debug helper
        print("[BALL] OpenCV is required for debug overlay output")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[BALL] Could not open video for debug overlay: {video_path}")
        return

    fps_val = fps if fps and math.isfinite(fps) and fps > 0 else _get_video_fps(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    set_telemetry_frame_bounds(width, height)

    telemetry = load_and_interpolate_telemetry(str(telemetry_path), total_frames=total_frames, fps=fps_val)

    overlay_out.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(overlay_out), fourcc, fps_val if fps_val > 0 else 30.0, (width, height))

    for idx in range(total_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if idx < len(telemetry):
            rec = telemetry[idx]
            cx = _as_float(rec.get("cx"))
            cy = _as_float(rec.get("cy"))
            has_ball = bool(rec.get("has_ball"))
            if math.isfinite(cx) and math.isfinite(cy):
                pt = (int(round(cx)), int(round(cy)))
                color = (0, 255, 0) if has_ball else (0, 165, 255)
                cv2.circle(frame, pt, 10, color, thickness=-1, lineType=cv2.LINE_AA)
        writer.write(frame)

    writer.release()
    cap.release()
    print(f"[BALL] Wrote debug overlay to {overlay_out}")


def load_ball_model(args: argparse.Namespace):
    """Load a YOLO model for ball detection, handling missing deps gracefully."""

    model_name = getattr(args, "model", None) or DEFAULT_MODEL_NAME
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[BALL] YOLO/ultralytics is not installed; please install it or use 'annotate' mode.")
        return None

    try:
        model = YOLO(model_name)
    except Exception as exc:  # noqa: BLE001
        print(f"[BALL] Failed to load YOLO model '{model_name}': {exc}")
        return None

    sport = getattr(args, "sport", "soccer") or "soccer"
    print(f"[BALL] Loaded YOLO model '{model_name}' for sport='{sport}'")
    return model


def _ball_class_ids(model, sport: str) -> set[int]:
    sport = (sport or "").lower()
    preferred_labels = {"sports ball", "sportsball", "ball"}
    target_ids: set[int] = set()
    names = getattr(model, "names", {}) or {}
    for cls_id, cls_name in names.items():
        if str(cls_name).lower() in preferred_labels:
            target_ids.add(int(cls_id))
    # COCO "sports ball" class id
    target_ids.add(32)

    if sport and sport != "soccer":
        return target_ids
    return target_ids


def detect_ball_in_frame(model, frame, sport: str = "soccer", min_conf: float = DEFAULT_MIN_CONF) -> Tuple[float | None, float | None, float | None]:
    """Run detector on a single frame and return centre coords + confidence."""

    try:
        results = model.predict(frame, verbose=False, device="cpu")
    except Exception as exc:  # noqa: BLE001
        print(f"[BALL] Detection failed on frame: {exc}")
        return None, None, None

    target_ids = _ball_class_ids(model, sport)
    best: Tuple[float, float, float] | None = None

    for res in results:
        boxes = getattr(res, "boxes", None)
        if boxes is None or not hasattr(boxes, "xyxy"):
            continue
        try:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy()
        except Exception:  # noqa: BLE001
            continue

        for box, conf, cls_id in zip(xyxy, confs, classes):
            if conf < float(min_conf):
                continue
            cls_int = int(round(float(cls_id)))
            if target_ids and cls_int not in target_ids:
                continue
            x0, y0, x1, y1 = map(float, box[:4])
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            if best is None or conf > best[2]:
                best = (cx, cy, float(conf))

    if best is None:
        return None, None, None
    return best


def _detect_ball(args: argparse.Namespace) -> int:
    try:
        import cv2
    except ImportError:
        print("[ERROR] OpenCV is required for detection", file=sys.stderr)
        return 2

    video_path = Path(args.video).expanduser()
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}", file=sys.stderr)
        return 2

    fps = _get_video_fps(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    set_telemetry_frame_bounds(width, height)
    print(
        f"[BALL] Detecting ball positions in {video_path} ({total_frames} frames, {width}x{height} @ {fps:.2f}fps)"
    )

    lower_white = np.array([0, 0, 180], dtype=np.uint8)
    upper_white = np.array([180, 80, 255], dtype=np.uint8)

    def simple_ball_detector(frame: np.ndarray) -> tuple[float | None, float | None]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_white, upper_white)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None

        frame_area = max(1, width * height)
        best_c: tuple[float, float] | None = None
        best_score = -1.0
        for c in contours:
            area = cv2.contourArea(c)
            if area <= 2 or area > frame_area * 0.01:
                continue
            peri = cv2.arcLength(c, True)
            circularity = 0.0 if peri <= 0 else 4 * math.pi * area / (peri * peri)
            M = cv2.moments(c)
            if M["m00"] <= 0:
                continue
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])
            score = circularity * area
            if score > best_score:
                best_score = score
                best_c = (cx, cy)

        return best_c if best_c is not None else (None, None)

    samples: list[dict[str, object]] = []
    detections = 0
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        t_val = frame_idx / fps if fps > 0 else 0.0
        cx, cy = simple_ball_detector(frame)
        if cx is not None and cy is not None and math.isfinite(cx) and math.isfinite(cy):
            cx = max(0.0, min(width, float(cx)))
            cy = max(0.0, min(height, float(cy)))
            samples.append(
                {
                    "frame": int(frame_idx),
                    "t": float(t_val),
                    "cx": float(cx),
                    "cy": float(cy),
                }
            )
            detections += 1

        if frame_idx % 50 == 0:
            print(f"[BALL] Processed frame {frame_idx}/{total_frames or '?'} (t={t_val:.2f}s)")

        frame_idx += 1

    cap.release()

    samples = smooth_telemetry(samples)
    save_ball_telemetry_jsonl(out_path, samples)

    if samples:
        t_vals = [float(rec.get("t", 0.0)) for rec in samples if isinstance(rec.get("t"), (int, float))]
        t_min = min(t_vals) if t_vals else 0.0
        t_max = max(t_vals) if t_vals else 0.0
        coverage = 100.0 * detections / max(1, total_frames)
        print(
            f"[BALL] Wrote {len(samples)} samples covering t={t_min:.2f}–{t_max:.2f}s to {out_path}"
        )
        print(
            f"[BALL] Detection summary: {detections}/{total_frames or '?'} frames ({coverage:.1f}%), t_min={t_min:.2f}, t_max={t_max:.2f}"
        )
    else:
        print(f"[BALL] No ball detections were found; wrote empty telemetry to {out_path}")

    if getattr(args, "debug_overlay_out", None) and out_path.is_file():
        _render_debug_overlay(video_path, out_path, Path(args.debug_overlay_out).expanduser(), fps=fps)

    return 0


def _annotate_ball(args: argparse.Namespace) -> int:
    import cv2

    video_path = Path(args.video).expanduser()
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}", file=sys.stderr)
        return 2

    fps = _get_video_fps(video_path)
    width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0)
    height = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0)
    set_telemetry_frame_bounds(width, height)
    positions: list[tuple[int, float, float]] = []
    clicks: dict[int, tuple[float, float]] = {}

    window_name = "Annotate ball (ESC to quit, SPACE/ENTER next frame, click to mark)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    state = {"current_frame": 0}

    def on_mouse(event, x, y, *_args):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks[state["current_frame"]] = (float(x), float(y))
            print(f"[BALL] Frame {state['current_frame']} -> ({x}, {y})")

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
        x, y = _clamp_xy(float(x), float(y))
        samples.append(
            {
                "frame": int(frame),
                "t": float(t),
                "cx": float(x),
                "cy": float(y),
            }
        )

    samples = smooth_telemetry(samples)
    save_ball_telemetry_jsonl(out_path, samples)

    t_vals = [rec["t"] for rec in samples if isinstance(rec.get("t"), (int, float))]
    t_min = min(t_vals) if t_vals else 0.0
    t_max = max(t_vals) if t_vals else 0.0
    print(
        f"[BALL] Wrote {len(samples)} samples covering t={t_min:.2f}–{t_max:.2f}s to {out_path}"
    )
    if getattr(args, "debug_overlay_out", None) and out_path.is_file():
        _render_debug_overlay(video_path, out_path, Path(args.debug_overlay_out).expanduser(), fps=fps)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ball telemetry utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    detect_p = sub.add_parser("detect", help="Auto-detect ball positions from video")
    detect_p.add_argument("--video", required=True, help="Input video path")
    detect_p.add_argument("--out", help="Output telemetry JSONL (default derived)")
    detect_p.add_argument("--sport", default="soccer", help="Sport context (default: soccer)")
    detect_p.add_argument("--model", default=DEFAULT_MODEL_NAME, help="YOLO model to load (default: yolov8n.pt)")
    detect_p.add_argument("--min-conf", dest="min_conf", type=float, default=DEFAULT_MIN_CONF, help="Minimum confidence for detections")
    detect_p.add_argument("--debug-overlay-out", help="Optional debug MP4 showing smoothed telemetry overlays")
    detect_p.set_defaults(func=_detect_ball)

    ann_p = sub.add_parser("annotate", help="Manually annotate ball positions")
    ann_p.add_argument("--video", required=True, help="Input video path")
    ann_p.add_argument("--out", help="Output telemetry JSONL (default derived)")
    ann_p.add_argument("--step", type=int, default=2, help="Advance this many frames per step (default: 2)")
    ann_p.add_argument("--debug-overlay-out", help="Optional debug MP4 showing smoothed telemetry overlays")
    ann_p.set_defaults(func=_annotate_ball)

    args = parser.parse_args(argv)
    if not getattr(args, "out", None):
        args.out = telemetry_path_for_video(Path(args.video))

    return args.func(args)


# ---------------------------------------------------------------------------
# YOLO ball detection with caching
# ---------------------------------------------------------------------------


def yolo_telemetry_path_for_video(video_path: str | Path) -> str:
    """Return the canonical YOLO ball telemetry cache path."""
    video_path = Path(video_path)
    stem = video_path.stem
    out_dir = Path("out") / "telemetry"
    return str(out_dir / f"{stem}.yolo_ball.jsonl")


def run_yolo_ball_detection(
    video_path: str | Path,
    *,
    min_conf: float = 0.20,
    cache: bool = True,
    model_name: str | None = None,
) -> list[BallSample]:
    """Run YOLO ball detection on every frame, returning BallSamples with confidence.

    Results are cached to ``out/telemetry/<stem>.yolo_ball.jsonl``
    (or ``<stem>.yolo_ball.<model>.jsonl`` when a non-default model is
    used).  If a cache file already exists and *cache* is True, the
    cached result is returned without re-running detection.

    Uses the BallTracker from soccer_highlights.ball_tracker which wraps
    ultralytics YOLO with constant-velocity smoothing.
    """
    video_path = Path(video_path)
    _effective_model = model_name or DEFAULT_MODEL_NAME

    # Model-specific cache so different weights don't collide.
    # Default model uses the original path for backward compat.
    _model_stem = Path(_effective_model).stem        # e.g. "yolov8m"
    _default_stem = Path(DEFAULT_MODEL_NAME).stem    # e.g. "yolov8n"
    if _model_stem == _default_stem:
        cache_path = Path(yolo_telemetry_path_for_video(video_path))
    else:
        base = Path(yolo_telemetry_path_for_video(video_path))
        cache_path = base.with_suffix(f".{_model_stem}.jsonl")

    # Return cached results if available
    if cache and cache_path.is_file():
        samples = _load_yolo_cache(cache_path)
        if samples:
            print(f"[YOLO] Loaded {len(samples)} cached YOLO detections from {cache_path}")
            return samples

    try:
        import cv2 as _cv2
    except ImportError:
        print("[YOLO] OpenCV is required for YOLO ball detection")
        return []

    try:
        from soccer_highlights.ball_tracker import BallTracker
    except ImportError:
        print("[YOLO] soccer_highlights.ball_tracker not available; skipping YOLO detection")
        return []

    tracker = BallTracker(
        weights_path=None,
        min_conf=min_conf,
        device="cpu",
        input_size=1920,    # native resolution — maximize detection of small/distant balls
        smooth_alpha=0.25,
        max_gap=12,
    )

    # Override model if a non-default was requested
    if _model_stem != _default_stem:
        try:
            from ultralytics import YOLO as _YOLO_CLS
            _model_path = Path(_effective_model)
            if not _model_path.is_absolute():
                # Search repo root, then CWD
                _repo = Path(__file__).resolve().parent.parent
                if (_repo / _effective_model).is_file():
                    _model_path = _repo / _effective_model
            tracker._model = _YOLO_CLS(str(_model_path))
            # Re-detect ball class IDs for the new model
            _names = getattr(tracker._model, "names", {}) or {}
            tracker._ball_ids = []
            if isinstance(_names, dict):
                for _idx, _nm in _names.items():
                    if isinstance(_nm, str) and "ball" in _nm.lower():
                        tracker._ball_ids.append(int(_idx))
            if not tracker._ball_ids:
                tracker._ball_ids = [32]
            print(f"[YOLO] Using model: {_model_path.name}")
        except Exception as _exc:
            print(f"[YOLO] Failed to load model '{_effective_model}': {_exc}")
            print(f"[YOLO] Falling back to default {DEFAULT_MODEL_NAME}")
    if not tracker.is_ready:
        reason = tracker.failure_reason or "unknown"
        print(f"[YOLO] BallTracker not ready: {reason}")
        return []

    cap = _cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[YOLO] Could not open video: {video_path}")
        return []

    fps = _get_video_fps(video_path)
    total_frames = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(_cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(_cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    set_telemetry_frame_bounds(width, height)

    print(
        f"[YOLO] Running ball detection on {video_path.name} "
        f"({total_frames} frames, {width}x{height} @ {fps:.1f}fps)"
    )

    samples: list[BallSample] = []
    detections = 0
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            t_val = frame_idx / fps if fps > 0 else 0.0
            track = tracker.update(frame_idx, frame)
            if track is not None:
                cx, cy = float(track.cx), float(track.cy)
                conf = float(track.conf)
                cx, cy = _clamp_xy(cx, cy)
                samples.append(BallSample(
                    t=t_val,
                    frame=frame_idx,
                    x=cx,
                    y=cy,
                    conf=conf,
                ))
                detections += 1
            if frame_idx % 100 == 0 and total_frames > 0:
                print(f"[YOLO] Frame {frame_idx}/{total_frames} ({100*frame_idx/total_frames:.0f}%)")
            frame_idx += 1
    finally:
        cap.release()

    coverage = 100.0 * detections / max(1, frame_idx)
    print(f"[YOLO] Detection complete: {detections}/{frame_idx} frames ({coverage:.1f}%)")

    # Save cache
    if cache:
        _save_yolo_cache(cache_path, samples)

    return samples


def _save_yolo_cache(path: Path, samples: list[BallSample]) -> None:
    """Write YOLO detections to JSONL with confidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            rec = {
                "frame": s.frame,
                "t": round(s.t, 6),
                "cx": round(s.x, 2),
                "cy": round(s.y, 2),
                "conf": round(s.conf, 4),
            }
            f.write(json.dumps(rec) + "\n")
    print(f"[YOLO] Cached {len(samples)} detections to {path}")


def _load_yolo_cache(path: Path) -> list[BallSample]:
    """Load YOLO detections from JSONL cache."""
    samples: list[BallSample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            frame_idx = _as_int(rec.get("frame"), 0)
            t_val = _as_float(rec.get("t"))
            cx = _as_float(rec.get("cx"))
            cy = _as_float(rec.get("cy"))
            conf = _as_float(rec.get("conf"))
            if not (math.isfinite(cx) and math.isfinite(cy)):
                continue
            if not math.isfinite(conf):
                conf = 0.0
            if not math.isfinite(t_val):
                t_val = 0.0
            samples.append(BallSample(t=t_val, frame=frame_idx, x=cx, y=cy, conf=conf))
    return samples


# ---------------------------------------------------------------------------
# YOLO person detection — per-frame bounding boxes for player-aware zoom
# ---------------------------------------------------------------------------


def yolo_person_telemetry_path_for_video(video_path) -> str:
    """Return the canonical YOLO person telemetry cache path."""
    video_path = Path(video_path)
    stem = video_path.stem
    out_dir = Path("out") / "telemetry"
    return str(out_dir / f"{stem}.yolo_person.jsonl")


def _save_person_cache(path: Path, data: dict[int, list[PersonBox]]) -> None:
    """Write per-frame person detections to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with path.open("w", encoding="utf-8") as f:
        for frame_idx in sorted(data.keys()):
            boxes = data[frame_idx]
            rec = {
                "frame": frame_idx,
                "persons": [
                    {
                        "cx": round(p.cx, 1),
                        "cy": round(p.cy, 1),
                        "w": round(p.w, 1),
                        "h": round(p.h, 1),
                        "conf": round(p.conf, 3),
                    }
                    for p in boxes
                ],
            }
            f.write(json.dumps(rec) + "\n")
            total += len(boxes)
    print(f"[YOLO] Cached {total} person detections ({len(data)} frames) to {path}")


def _load_person_cache(path: Path) -> dict[int, list[PersonBox]]:
    """Load per-frame person detections from JSONL cache."""
    result: dict[int, list[PersonBox]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            frame_idx = _as_int(rec.get("frame"), 0)
            persons_raw = rec.get("persons", [])
            boxes: list[PersonBox] = []
            for p in persons_raw:
                cx = _as_float(p.get("cx"))
                cy = _as_float(p.get("cy"))
                w = _as_float(p.get("w"))
                h = _as_float(p.get("h"))
                conf = _as_float(p.get("conf"))
                if not (math.isfinite(cx) and math.isfinite(cy)):
                    continue
                if not math.isfinite(w):
                    w = 0.0
                if not math.isfinite(h):
                    h = 0.0
                if not math.isfinite(conf):
                    conf = 0.5
                boxes.append(PersonBox(frame=frame_idx, cx=cx, cy=cy, w=w, h=h, conf=conf))
            if boxes:
                result[frame_idx] = boxes
    return result


def run_yolo_person_detection(
    video_path: str | Path,
    *,
    min_conf: float = 0.30,
    cache: bool = True,
) -> dict[int, list[PersonBox]]:
    """Run YOLO person detection on every frame, returning per-frame person boxes.

    Results are cached to ``out/telemetry/<stem>.yolo_person.jsonl``.
    Uses the same yolov8n.pt model as ball detection (COCO class 0 = person).
    """
    video_path = Path(video_path)
    cache_path = Path(yolo_person_telemetry_path_for_video(video_path))

    if cache and cache_path.is_file():
        result = _load_person_cache(cache_path)
        if result:
            total = sum(len(v) for v in result.values())
            print(f"[YOLO] Loaded {total} cached person detections ({len(result)} frames) from {cache_path}")
            return result

    try:
        import cv2 as _cv2
    except ImportError:
        print("[YOLO] OpenCV required for person detection")
        return {}

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[YOLO] ultralytics not available for person detection")
        return {}

    model = YOLO("yolov8n.pt")
    person_class_id = 0  # COCO class 0 = person

    cap = _cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[YOLO] Could not open video for person detection: {video_path}")
        return {}

    total_frames = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT) or 0)
    src_w = int(cap.get(_cv2.CAP_PROP_FRAME_WIDTH) or 0)
    src_h = int(cap.get(_cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    print(
        f"[YOLO] Running person detection on {video_path.name} "
        f"({total_frames} frames, {src_w}x{src_h})"
    )

    result: dict[int, list[PersonBox]] = {}
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            preds = model.predict(
                frame, conf=min_conf, iou=0.45, imgsz=640, verbose=False, device="cpu",
            )
            if preds:
                boxes = getattr(preds[0], "boxes", None)
                if boxes is not None:
                    cls = boxes.cls.detach().cpu().numpy() if hasattr(boxes.cls, "detach") else np.asarray(boxes.cls)
                    conf = boxes.conf.detach().cpu().numpy() if hasattr(boxes.conf, "detach") else np.asarray(boxes.conf)
                    xyxy = boxes.xyxy.detach().cpu().numpy() if hasattr(boxes.xyxy, "detach") else np.asarray(boxes.xyxy)

                    frame_persons: list[PersonBox] = []
                    for cls_id, c_score, box in zip(cls, conf, xyxy):
                        if int(cls_id) != person_class_id:
                            continue
                        if float(c_score) < min_conf:
                            continue
                        x0, y0, x1, y1 = map(float, box[:4])
                        bw = max(0.0, x1 - x0)
                        bh = max(0.0, y1 - y0)
                        # Filter out implausibly small or large boxes
                        if bw < 10 or bh < 20:
                            continue
                        if bw > src_w * 0.4 or bh > src_h * 0.8:
                            continue
                        frame_persons.append(PersonBox(
                            frame=frame_idx,
                            cx=(x0 + x1) / 2.0,
                            cy=(y0 + y1) / 2.0,
                            w=bw,
                            h=bh,
                            conf=float(c_score),
                        ))
                    if frame_persons:
                        result[frame_idx] = frame_persons

            if frame_idx % 100 == 0 and total_frames > 0:
                print(f"[YOLO] Person detection: frame {frame_idx}/{total_frames} ({100*frame_idx/total_frames:.0f}%)")
            frame_idx += 1
    finally:
        cap.release()

    total_persons = sum(len(v) for v in result.values())
    print(f"[YOLO] Person detection complete: {total_persons} detections across {len(result)}/{frame_idx} frames")

    if cache:
        _save_person_cache(cache_path, result)

    return result


# ---------------------------------------------------------------------------
# Fusion: YOLO + motion-centroid → merged positions + confidence
# ---------------------------------------------------------------------------


@dataclass
class ExcludeZone:
    """Rectangular region + optional frame range where YOLO detections are suppressed.

    Coordinates are in source pixels.  Any YOLO detection whose (x, y) falls
    inside the rectangle AND whose frame index is within [frame_start, frame_end)
    is dropped before fusion.  Omit frame bounds to apply the zone to the entire
    clip.

    Typical use-cases:
      - Spare ball on the sideline (fixed region, entire clip)
      - Ball on an adjacent field visible only late in the clip
    """

    x_min: float = 0.0
    x_max: float = float("inf")
    y_min: float = 0.0
    y_max: float = float("inf")
    frame_start: int = 0
    frame_end: int = int(2**31)

    def contains(self, x: float, y: float, frame: int) -> bool:
        return (
            self.x_min <= x <= self.x_max
            and self.y_min <= y <= self.y_max
            and self.frame_start <= frame < self.frame_end
        )


def load_exclude_zones(path: str | Path) -> list[ExcludeZone]:
    """Load exclusion zones from a JSON file.

    Expected format — a JSON array of zone objects::

        [
          {"x_min": 0, "x_max": 400, "y_min": 0, "y_max": 1080,
           "frame_start": 1100, "frame_end": 1500,
           "note": "adjacent field ball visible late in clip"},
          {"x_min": 1500, "x_max": 1920, "y_min": 800, "y_max": 1080,
           "note": "spare ball on sideline"}
        ]

    Unrecognised keys (like ``note``) are silently ignored.
    """
    path = Path(path)
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raw = [raw]
    zones: list[ExcludeZone] = []
    for entry in raw:
        zones.append(ExcludeZone(
            x_min=float(entry.get("x_min", 0)),
            x_max=float(entry.get("x_max", float("inf"))),
            y_min=float(entry.get("y_min", 0)),
            y_max=float(entry.get("y_max", float("inf"))),
            frame_start=int(entry.get("frame_start", 0)),
            frame_end=int(entry.get("frame_end", 2**31)),
        ))
    return zones


def fuse_yolo_and_centroid(
    yolo_samples: list[BallSample],
    centroid_samples: list[BallSample],
    frame_count: int,
    width: float,
    height: float,
    *,
    fps: float = 30.0,
    exclude_zones: list[ExcludeZone] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Merge YOLO ball detections with motion-centroid positions.

    Returns:
        positions: (frame_count, 2) float32 array of (x, y) per frame
        used_mask: (frame_count,) bool array — True where a position is available
        confidence: (frame_count,) float32 array in [0, 1] — per-frame confidence
        source_labels: (frame_count,) uint8 array — per-frame source type:
            0=none, 1=yolo, 2=centroid, 3=blended, 4=interpolated, 5=hold

    Fusion strategy:
        - YOLO detection present + confident (>= 0.4): use YOLO position, high confidence
        - YOLO present but low confidence: blend YOLO + centroid weighted by YOLO conf
        - Only centroid present: use centroid, moderate confidence (0.3)
        - Neither present: NaN position, zero confidence

    *exclude_zones*: optional list of :class:`ExcludeZone` regions.  Any YOLO
    detection falling inside an exclusion zone is dropped before fusion, which
    prevents the camera from tracking stray balls (sideline spares, adjacent
    fields, etc.).

    Handles degenerate cases: zero frames, empty inputs, single-sample clips.
    """
    FUSE_NONE = np.uint8(0)
    FUSE_YOLO = np.uint8(1)
    FUSE_CENTROID = np.uint8(2)
    FUSE_BLENDED = np.uint8(3)
    FUSE_INTERP = np.uint8(4)
    FUSE_HOLD = np.uint8(5)
    FUSE_SHOT_HOLD = np.uint8(6)  # shot-hold / pan-hold (frozen at YOLO anchor)
    FUSE_TRACKER = np.uint8(7)    # CSRT visual tracker (real pixel data)

    # Guard: zero-frame or negative frame_count
    if frame_count <= 0:
        print("[FUSION] frame_count <= 0; returning empty arrays")
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty(0, dtype=bool),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.uint8),
        )

    positions = np.full((frame_count, 2), np.nan, dtype=np.float32)
    used_mask = np.zeros(frame_count, dtype=bool)
    confidence = np.zeros(frame_count, dtype=np.float32)
    source_labels = np.zeros(frame_count, dtype=np.uint8)

    # FPS-aware temporal scaling: all frame-count constants below were
    # originally tuned for 30fps.  Scale them so time-based durations
    # (stale gating, interpolation gaps, extrapolation windows) stay
    # correct at any frame rate.
    _fps = max(fps, 1.0)
    _fps_scale = _fps / 30.0

    # Guard: both inputs empty — return centred default position so camera
    # holds at frame centre rather than producing all-NaN.
    if not yolo_samples and not centroid_samples:
        print("[FUSION] No YOLO or centroid samples; holding at frame centre")
        positions[:, 0] = width * 0.5
        positions[:, 1] = height * 0.5
        confidence[:] = 0.10
        used_mask[:] = True
        source_labels[:] = FUSE_HOLD
        return positions, used_mask, confidence, source_labels

    # Edge margin: YOLO detections within this fraction of the frame edge
    # are almost always false positives (goalposts, shadows, partial views).
    # Filter them at ingestion time so they don't contaminate the centroid
    # gating via _last_yolo_x for many subsequent frames.
    #
    # Adaptive: when YOLO detections are sparse, every detection is critical
    # for anchoring the ball path.  Reduce the margin so near-edge detections
    # (which are often the ball at the edge of the field, not goalposts) are
    # kept.  At 5% density the margin shrinks to ~2.5%; at 15%+ it stays 6%.
    #
    # Two-pass: if the first pass discards more than 25% of detections the
    # margin is likely too aggressive (ball genuinely near the goalmouth or
    # sideline).  Halve the margin and re-filter so we keep those critical
    # near-edge detections while still removing obvious false positives.
    EDGE_MARGIN_FRAC = 0.045  # base: 4.5% of frame width (reduced to keep more near-goal YOLO)
    _raw_yolo_count = sum(
        1 for s in yolo_samples
        if 0 <= int(s.frame) < frame_count
        and math.isfinite(s.x) and math.isfinite(s.y)
    )
    _raw_density = _raw_yolo_count / max(frame_count, 1)
    if _raw_density < 0.15 and _raw_yolo_count > 0:
        # Scale margin down: at 5% density → 0.02, at 15% → 0.045
        EDGE_MARGIN_FRAC = max(0.02, 0.045 * (_raw_density / 0.15))

    _EDGE_REFILTER_THRESH = 0.20  # re-filter if >20% of detections removed (more aggressive halving)

    def _edge_filter_pass(samples, fcount, w, margin_frac):
        """Filter near-edge YOLO detections. Returns (by_frame dict, n_filtered)."""
        edge_px = w * margin_frac if w > 0 else 0
        by_frame: dict[int, BallSample] = {}
        n_filt = 0
        for s in samples:
            fidx = int(s.frame)
            if 0 <= fidx < fcount and math.isfinite(s.x) and math.isfinite(s.y):
                if edge_px > 0 and (s.x < edge_px or s.x > w - edge_px):
                    n_filt += 1
                    continue
                by_frame[fidx] = s
        return by_frame, n_filt, edge_px

    # First pass with the base margin.
    yolo_by_frame, _edge_filtered, _edge_px_ingest = _edge_filter_pass(
        yolo_samples, frame_count, width, EDGE_MARGIN_FRAC)

    # Second pass: if we discarded too many, halve the margin and retry.
    if (_edge_filtered > 0
            and _raw_yolo_count > 0
            and _edge_filtered / _raw_yolo_count > _EDGE_REFILTER_THRESH):
        _reduced_frac = EDGE_MARGIN_FRAC * 0.5
        yolo_by_frame, _edge_filtered_2, _edge_px_ingest = _edge_filter_pass(
            yolo_samples, frame_count, width, _reduced_frac)
        print(
            f"[FUSION] Edge re-filter: {_edge_filtered}/{_raw_yolo_count} "
            f"({_edge_filtered/_raw_yolo_count:.0%}) exceeded {_EDGE_REFILTER_THRESH:.0%} threshold; "
            f"reduced margin {width * EDGE_MARGIN_FRAC:.0f}px -> {_edge_px_ingest:.0f}px, "
            f"now filtered {_edge_filtered_2}"
        )
        # Update EDGE_MARGIN_FRAC so downstream checks (near-edge flight
        # skip) use the same margin that survived the re-filter pass.
        EDGE_MARGIN_FRAC = _reduced_frac
        _edge_filtered = _edge_filtered_2

        # Third pass: if we're STILL discarding >20% after halving, the
        # ball genuinely lives near the edge (source-camera panning,
        # corner kicks, goalmouth action).  Drop to a minimal 1% margin
        # that only catches obvious false positives at the very frame
        # boundary — not real ball detections during pans.
        if (_edge_filtered_2 > 0
                and _edge_filtered_2 / _raw_yolo_count > _EDGE_REFILTER_THRESH):
            _minimal_frac = 0.01  # ~19px on 1920-wide frame
            yolo_by_frame, _edge_filtered_3, _edge_px_ingest = _edge_filter_pass(
                yolo_samples, frame_count, width, _minimal_frac)
            print(
                f"[FUSION] Edge triple-filter: still {_edge_filtered_2}/{_raw_yolo_count} "
                f"({_edge_filtered_2/_raw_yolo_count:.0%}); "
                f"reduced to minimal {_edge_px_ingest:.0f}px, now filtered {_edge_filtered_3}"
            )
            EDGE_MARGIN_FRAC = _minimal_frac
            _edge_filtered = _edge_filtered_3

    if _edge_filtered > 0:
        print(
            f"[FUSION] Filtered {_edge_filtered} near-edge YOLO detections "
            f"(margin={_edge_px_ingest:.0f}px)"
        )

    # --- Exclusion-zone filtering ---
    # Drop YOLO detections that fall inside any caller-supplied exclusion
    # rectangle (e.g. spare ball on the sideline, ball on an adjacent field).
    _zone_filtered = 0
    if exclude_zones:
        surviving: dict[int, BallSample] = {}
        for fidx, s in yolo_by_frame.items():
            if any(z.contains(s.x, s.y, fidx) for z in exclude_zones):
                _zone_filtered += 1
            else:
                surviving[fidx] = s
        yolo_by_frame = surviving
        if _zone_filtered > 0:
            print(
                f"[FUSION] Excluded {_zone_filtered} YOLO detections in "
                f"{len(exclude_zones)} exclusion zone(s)"
            )

    # --- Multi-ball spatial consistency filter ---
    # When two balls are visible (spare ball on sideline, ball on adjacent
    # pitch), YOLO detections alternate between them.  Detect a bimodal
    # spatial distribution and keep only the dominant cluster.
    #
    # Selection priority:
    # 1. Centroid proximity — the cluster whose mean position is closer to
    #    the centroid (player-cluster) positions is on the main field.  This
    #    overrides cluster size when the proximity difference is clear
    #    (>150px), because a background ball can have MORE detections than
    #    the game ball (it's stationary and easy for YOLO to detect).
    # 2. Cluster SIZE — when centroid proximity is ambiguous, the cluster
    #    with many more detections is likely the game ball.
    # 3. Vertical position (tiebreaker) — when clusters are similar size
    #    and distance, prefer lower-in-frame (closer to camera).
    #
    # Build a centroid lookup for proximity checks before the filter runs.
    _centroid_lookup: dict[int, tuple[float, float]] = {}
    for _cs in centroid_samples:
        _cf = int(_cs.frame)
        if 0 <= _cf < frame_count and math.isfinite(_cs.x) and math.isfinite(_cs.y):
            _centroid_lookup[_cf] = (float(_cs.x), float(_cs.y))

    if len(yolo_by_frame) >= 6:
        _mb_frames = sorted(yolo_by_frame.keys())
        _mb_xs = [float(yolo_by_frame[f].x) for f in _mb_frames]
        _mb_ys = [float(yolo_by_frame[f].y) for f in _mb_frames]

        # Find largest gap in x-distribution
        _xs_sorted = sorted(_mb_xs)
        _max_gap_x = 0.0
        _split_x = 0.0
        for _gi in range(1, len(_xs_sorted)):
            _gx = _xs_sorted[_gi] - _xs_sorted[_gi - 1]
            if _gx > _max_gap_x:
                _max_gap_x = _gx
                _split_x = (_xs_sorted[_gi - 1] + _xs_sorted[_gi]) / 2.0

        # Find largest gap in y-distribution
        _ys_sorted = sorted(_mb_ys)
        _max_gap_y = 0.0
        _split_y = 0.0
        for _gi in range(1, len(_ys_sorted)):
            _gy = _ys_sorted[_gi] - _ys_sorted[_gi - 1]
            if _gy > _max_gap_y:
                _max_gap_y = _gy
                _split_y = (_ys_sorted[_gi - 1] + _ys_sorted[_gi]) / 2.0

        # Use the dimension with the proportionally larger gap
        _x_frac = _max_gap_x / max(width, 1.0)
        _y_frac = _max_gap_y / max(height, 1.0)
        _SPLIT_THRESH = 0.25  # 25% of frame dimension

        if _x_frac >= _SPLIT_THRESH or _y_frac >= _SPLIT_THRESH:
            if _x_frac >= _y_frac:
                _grp_a = [f for f in _mb_frames if yolo_by_frame[f].x < _split_x]
                _grp_b = [f for f in _mb_frames if yolo_by_frame[f].x >= _split_x]
            else:
                _grp_a = [f for f in _mb_frames if yolo_by_frame[f].y < _split_y]
                _grp_b = [f for f in _mb_frames if yolo_by_frame[f].y >= _split_y]

            # Both groups need at least 2 detections to confirm two targets
            if len(_grp_a) >= 2 and len(_grp_b) >= 2:
                _size_ratio = max(len(_grp_a), len(_grp_b)) / min(len(_grp_a), len(_grp_b))
                _mean_y_a = sum(yolo_by_frame[f].y for f in _grp_a) / len(_grp_a)
                _mean_y_b = sum(yolo_by_frame[f].y for f in _grp_b) / len(_grp_b)

                # --- Centroid-proximity check ---
                # Compute mean distance of each cluster to the nearest
                # centroid (player-cluster) position.  The cluster on the
                # main field will be close to centroids; a background ball
                # on another field will be far away.
                def _cluster_centroid_dist(grp: list[int]) -> float:
                    dists: list[float] = []
                    for _gf in grp:
                        _gx = float(yolo_by_frame[_gf].x)
                        _gy = float(yolo_by_frame[_gf].y)
                        if _gf in _centroid_lookup:
                            _ccx, _ccy = _centroid_lookup[_gf]
                            dists.append(math.hypot(_gx - _ccx, _gy - _ccy))
                        else:
                            # Try nearest centroid frame within ±15 frames
                            _best_d = float("inf")
                            for _off in range(-15, 16):
                                _nf = _gf + _off
                                if _nf in _centroid_lookup:
                                    _ccx, _ccy = _centroid_lookup[_nf]
                                    _best_d = min(_best_d, math.hypot(_gx - _ccx, _gy - _ccy))
                            if _best_d < float("inf"):
                                dists.append(_best_d)
                    return sum(dists) / len(dists) if dists else float("inf")

                _dist_a = _cluster_centroid_dist(_grp_a)
                _dist_b = _cluster_centroid_dist(_grp_b)
                _PROX_OVERRIDE_PX = 150.0  # override size-based when >150px closer

                # Decide which cluster to keep:
                # 1. Centroid proximity wins when the difference is clear
                # 2. Size ratio wins when proximity is ambiguous
                # 3. Vertical position (closer to camera) is final tiebreaker
                if abs(_dist_a - _dist_b) > _PROX_OVERRIDE_PX:
                    # Strong proximity signal — the closer cluster is
                    # on the main field, even if the other has more
                    # detections (background balls are easy for YOLO).
                    if _dist_a <= _dist_b:
                        _keep = set(_grp_a)
                        _kept_y = _mean_y_a
                        _reason = f"centroid proximity ({_dist_a:.0f}px vs {_dist_b:.0f}px)"
                    else:
                        _keep = set(_grp_b)
                        _kept_y = _mean_y_b
                        _reason = f"centroid proximity ({_dist_b:.0f}px vs {_dist_a:.0f}px)"
                elif _size_ratio >= 3.0:
                    # Proximity is ambiguous — fall back to cluster size.
                    if len(_grp_a) >= len(_grp_b):
                        _keep = set(_grp_a)
                        _kept_y = _mean_y_a
                        _reason = "dominant cluster"
                    else:
                        _keep = set(_grp_b)
                        _kept_y = _mean_y_b
                        _reason = "dominant cluster"
                else:
                    # Clusters are comparable size — use vertical
                    # position as tiebreaker (lower in frame = closer).
                    if _mean_y_a >= _mean_y_b:
                        _keep = set(_grp_a)
                        _kept_y = _mean_y_a
                        _reason = "closer to camera"
                    else:
                        _keep = set(_grp_b)
                        _kept_y = _mean_y_b
                        _reason = "closer to camera"

                _n_removed = len(yolo_by_frame) - len(_keep)
                if _n_removed > 0:
                    # Safety: don't remove detections with high confidence —
                    # a conf>0.45 detection is almost certainly the real ball,
                    # even if it's far from centroid.  The centroid tracks
                    # player-mass which can be totally wrong.
                    _removed_frames = set(yolo_by_frame.keys()) - _keep
                    _max_removed_conf = max(
                        (float(yolo_by_frame[f].conf) for f in _removed_frames),
                        default=0.0,
                    )
                    if _max_removed_conf >= 0.45:
                        print(
                            f"[FUSION] Multi-ball filter: SKIPPED removal of "
                            f"{_n_removed} detections — removed cluster contains "
                            f"high-confidence detection (conf={_max_removed_conf:.3f} >= 0.45)"
                        )
                    else:
                        yolo_by_frame = {f: yolo_by_frame[f] for f in _keep}
                        print(
                            f"[FUSION] Multi-ball filter: removed {_n_removed}/{_n_removed + len(_keep)} "
                            f"YOLO detections ({_reason}, "
                            f"size={len(_keep)}v{_n_removed}, ratio={_size_ratio:.1f}x, "
                            f"gap={max(_max_gap_x, _max_gap_y):.0f}px, "
                            f"kept mean_y={_kept_y:.0f})"
                        )

    # Index centroid samples by frame
    centroid_by_frame: dict[int, BallSample] = {}
    for s in centroid_samples:
        fidx = int(s.frame)
        if 0 <= fidx < frame_count and math.isfinite(s.x) and math.isfinite(s.y):
            centroid_by_frame[fidx] = s

    yolo_used = 0
    centroid_used = 0
    blended = 0
    conf_threshold = 0.20  # YOLO conf above which we trust it fully (lowered from 0.28)
    # Adaptive conf threshold: when YOLO is sparse, lower the threshold
    # further so we accept more marginal detections.
    _yolo_density_pre = len(yolo_by_frame) / max(frame_count, 1)
    if _yolo_density_pre < 0.15:
        _density_ratio = _yolo_density_pre / 0.15
        conf_threshold = max(0.12, conf_threshold * (0.65 + 0.35 * _density_ratio))
        print(
            f"[FUSION] Sparse YOLO ({_yolo_density_pre:.1%}): "
            f"lowered conf_threshold to {conf_threshold:.2f}"
        )

    # Track last trusted YOLO position for centroid gating.
    # When centroid-only frames appear far from the last YOLO position,
    # the centroid is probably tracking a player cluster, not the ball.
    # In that case we blend toward the YOLO hold to prevent the camera
    # from snapping to the wrong part of the field.
    #
    # The gating decays over time: as frames pass without a new YOLO,
    # the reference becomes stale and the centroid (which follows the
    # source camera's lookahead-smoothed pan) is trusted more.  This
    # lets the camera lead the play during long ball-flight gaps.
    _last_yolo_x: Optional[float] = None
    _last_yolo_y: Optional[float] = None
    _last_yolo_frame: int = -999
    YOLO_HOLD_DIST = 140.0  # px: centroid beyond this → suspect (tightened from 200 to catch background lock-on sooner)
    YOLO_HOLD_BLEND = 0.55  # weight for last YOLO when centroid diverges (raised from 0.40 for stronger gating)
    YOLO_STALE_FRAMES = max(1, int(30 * _fps_scale))  # ~1s at any fps

    # Velocity-based prediction: maintain a short history of the last
    # 2-3 YOLO detections so we can predict where the ball *should* be
    # now, rather than gating centroid against a single stale position.
    # This prevents the "gravity well" effect where a stale YOLO anchor
    # holds the camera in place while the ball has moved on.
    _yolo_history: list[tuple[int, float, float]] = []  # (frame, x, y)
    _YOLO_HISTORY_MAX = 3

    # Adaptive gating: when YOLO density is very low (sparse detections),
    # increase gating strength so centroid can't freely wander to track
    # player clusters instead of the ball.
    _yolo_density = len(yolo_by_frame) / max(frame_count, 1)
    if _yolo_density < 0.15:
        # Scale stale frames inversely with density: fewer YOLO → trust centroid
        # less for longer after each YOLO detection.
        # At 5% density → ~3s; at 15% density → ~1s (unchanged)
        YOLO_STALE_FRAMES = max(1, int((30 + 60 * (1.0 - _yolo_density / 0.15)) * _fps_scale))
        YOLO_HOLD_BLEND = min(0.70, 0.40 + 0.30 * (1.0 - _yolo_density / 0.15))
        print(
            f"[FUSION] Sparse YOLO ({_yolo_density:.1%}): "
            f"gating stale_frames={YOLO_STALE_FRAMES}, hold_blend={YOLO_HOLD_BLEND:.2f}"
        )

    def _update_yolo_history(frame_idx: int, x: float, y: float) -> None:
        """Append to YOLO history, keeping only the last N entries."""
        nonlocal _last_yolo_x, _last_yolo_y, _last_yolo_frame
        _last_yolo_x, _last_yolo_y = x, y
        _last_yolo_frame = frame_idx
        _yolo_history.append((frame_idx, x, y))
        if len(_yolo_history) > _YOLO_HISTORY_MAX:
            _yolo_history.pop(0)

    def _predict_ball_position(at_frame: int) -> tuple[float, float]:
        """Predict ball position at *at_frame* using velocity from YOLO history.

        With 2+ history points we compute a velocity vector and extrapolate.
        With only 1 point we fall back to the static last-YOLO position.
        The prediction is clamped to the frame so it doesn't fly off-screen.
        """
        if len(_yolo_history) < 2:
            return float(_last_yolo_x), float(_last_yolo_y)  # type: ignore[arg-type]
        # Use the two most recent YOLO detections for velocity
        f0, x0, y0 = _yolo_history[-2]
        f1, x1, y1 = _yolo_history[-1]
        dt = max(1, f1 - f0)
        vx = (x1 - x0) / dt
        vy = (y1 - y0) / dt
        elapsed = at_frame - f1
        # Decelerate prediction: ball slows down over time (drag model)
        # Use sqrt decay so prediction doesn't overshoot on long gaps
        eff_elapsed = math.sqrt(max(0, elapsed)) * math.sqrt(max(1, dt))
        pred_x = x1 + vx * eff_elapsed
        pred_y = y1 + vy * eff_elapsed
        # Clamp to frame bounds
        pred_x = max(0.0, min(float(width), pred_x))
        pred_y = max(0.0, min(float(height), pred_y))
        return pred_x, pred_y

    for i in range(frame_count):
        yolo = yolo_by_frame.get(i)
        centroid = centroid_by_frame.get(i)

        if yolo is not None and centroid is not None:
            yolo_conf = max(0.0, min(1.0, yolo.conf))
            if yolo_conf >= conf_threshold:
                # High-confidence YOLO: use directly
                positions[i, 0] = yolo.x
                positions[i, 1] = yolo.y
                confidence[i] = yolo_conf
                source_labels[i] = FUSE_YOLO
                yolo_used += 1
                _update_yolo_history(i, float(yolo.x), float(yolo.y))
            else:
                # Low-confidence YOLO: blend with centroid
                # Weight YOLO by its confidence, centroid gets the remainder
                w_yolo = yolo_conf / conf_threshold  # 0..1
                w_cent = 1.0 - w_yolo
                positions[i, 0] = w_yolo * yolo.x + w_cent * centroid.x
                positions[i, 1] = w_yolo * yolo.y + w_cent * centroid.y
                confidence[i] = 0.3 + 0.5 * w_yolo  # 0.3 to 0.8
                source_labels[i] = FUSE_BLENDED
                blended += 1
                _update_yolo_history(i, float(yolo.x), float(yolo.y))
            used_mask[i] = True
        elif yolo is not None:
            # Only YOLO
            positions[i, 0] = yolo.x
            positions[i, 1] = yolo.y
            confidence[i] = max(0.0, min(1.0, yolo.conf))
            source_labels[i] = FUSE_YOLO
            used_mask[i] = True
            yolo_used += 1
            _update_yolo_history(i, float(yolo.x), float(yolo.y))
        elif centroid is not None:
            cx, cy = float(centroid.x), float(centroid.y)
            # Gate centroid against predicted YOLO position, with
            # time-based decay so stale YOLO references don't anchor
            # the camera in the wrong part of the field.  Uses velocity-
            # based prediction from last 2-3 YOLO detections instead of
            # a static last-seen position, preventing the "gravity well"
            # effect on sparse clips.
            if _last_yolo_x is not None:
                pred_x, pred_y = _predict_ball_position(i)
                dist = math.hypot(cx - pred_x, cy - pred_y)
                if dist > YOLO_HOLD_DIST:
                    frames_since = i - _last_yolo_frame
                    decay = max(0.0, 1.0 - frames_since / YOLO_STALE_FRAMES)
                    w = YOLO_HOLD_BLEND * decay
                    if w > 0.01:
                        cx = w * pred_x + (1.0 - w) * cx
                        cy = w * pred_y + (1.0 - w) * cy
                        confidence[i] = 0.22 + 0.08 * (1.0 - decay)
                    else:
                        confidence[i] = 0.30  # YOLO stale — trust centroid
                else:
                    confidence[i] = 0.30  # centroid agrees with predicted area
            else:
                confidence[i] = 0.30
            positions[i, 0] = cx
            positions[i, 1] = cy
            source_labels[i] = FUSE_CENTROID
            used_mask[i] = True
            centroid_used += 1

    total_covered = int(used_mask.sum())
    coverage = 100.0 * total_covered / max(1, frame_count)
    print(
        f"[FUSION] {total_covered}/{frame_count} frames covered ({coverage:.1f}%): "
        f"yolo={yolo_used}, centroid={centroid_used}, blended={blended}, "
        f"avg_conf={float(confidence[used_mask].mean()) if total_covered > 0 else 0:.2f}"
    )

    # --- YOLO interpolation pass ---
    # When YOLO loses the ball for short periods, interpolate between
    # surrounding YOLO detections instead of using centroid.  This
    # eliminates the camera oscillation caused by alternating between
    # YOLO (actual ball) and centroid (player activity cluster) which
    # can differ by 50-200px and cause visible camera hunting.
    SHORT_INTERP_GAP = max(1, int(15 * _fps_scale))   # ~0.5s at any fps
    LONG_INTERP_GAP = max(1, int(90 * _fps_scale))    # ~3s at any fps
    MIN_FLIGHT_DIST = 100.0  # px: long gaps only interpolated if ball clearly traveled
    INTERP_CONF = 0.38       # confidence for YOLO-interpolated frames (raised from
                             # 0.28 — these are anchored between two real YOLO
                             # detections and deserve higher trust than raw centroid)
    EASE_OUT_CONF = 0.50     # higher confidence for ease-out interpolated frames:
                             # the deceleration curve closely matches kicked-ball
                             # physics, so the camera should trust these positions
                             # and respond at near-full speed (reduces EMA lag)
    # Step-function threshold: only use step (jump to receiver) for gaps
    # >= this many frames.  Shorter flights use linear interpolation so the
    # Gaussian can absorb rapid back-and-forth movements without whipsawing
    # the camera.  ~1.5s — enough for the camera to arrive and settle at
    # the receiver.
    STEP_THRESHOLD = max(1, int(45 * _fps_scale))
    # Adaptive: when YOLO is sparse, allow longer interpolation gaps so the
    # camera interpolates between known ball positions instead of relying on
    # centroid tracking (which follows player clusters, not the ball).
    # At 5% density → ~6s; at 15% density → ~3s (unchanged).
    _BASE_LONG_INTERP_GAP = LONG_INTERP_GAP
    if _yolo_density < 0.15:
        LONG_INTERP_GAP = max(1, int((90 + 90 * (1.0 - _yolo_density / 0.15)) * _fps_scale))
        print(
            f"[FUSION] Sparse YOLO: extended LONG_INTERP_GAP "
            f"{_BASE_LONG_INTERP_GAP}->{LONG_INTERP_GAP} frames"
        )

    # Collect frames that have YOLO data (used directly or blended)
    yolo_frames = sorted(yolo_by_frame.keys())

    if len(yolo_frames) >= 1:
        interpolated = 0
        long_interp = 0
        long_flight_info = []  # collect info about long-flight gaps
        # For each pair of consecutive YOLO frames, fill the gap
        for seg_idx in range(len(yolo_frames) - 1):
            fi = yolo_frames[seg_idx]
            fj = yolo_frames[seg_idx + 1]
            gap = fj - fi
            if gap <= 1 or gap > LONG_INTERP_GAP:
                continue  # no gap or too large to interpolate

            # YOLO positions at the endpoints
            yi = yolo_by_frame[fi]
            yj = yolo_by_frame[fj]
            x0, y0 = float(yi.x), float(yi.y)
            x1, y1 = float(yj.x), float(yj.y)

            # For long gaps (> SHORT_INTERP_GAP), only interpolate if the
            # ball clearly traveled across the field.  This prevents bad
            # interpolation between false-positive YOLO detections that are
            # spatially close but temporally far apart.
            is_long = gap > SHORT_INTERP_GAP
            if is_long:
                dist = math.hypot(x1 - x0, y1 - y0)
                if dist < MIN_FLIGHT_DIST:
                    continue  # small move: centroid tracking is fine
                # Skip flights where either endpoint is near the frame
                # edge — likely a false-positive YOLO detection.
                _edge_px = width * EDGE_MARGIN_FRAC
                if x0 < _edge_px or x0 > width - _edge_px or \
                   x1 < _edge_px or x1 > width - _edge_px:
                    print(
                        f"[FUSION] Skipping near-edge flight: "
                        f"frames {fi}->{fj}, ball x={x0:.0f}->{x1:.0f} "
                        f"(edge margin={_edge_px:.0f}px)"
                    )
                    continue
                long_interp += gap - 1

                # --- SHOT-HOLD MODE ---
                # When the ball was moving fast at the gap start and then
                # YOLO drops out for a long time, the ball likely entered
                # the net (goal) or went out of play.  Interpolating toward
                # the next YOLO creates a phantom trajectory because the
                # broadcast camera pans between detections, shifting pixel
                # coordinates.  Instead, hold near the last YOLO position
                # with heavy deceleration — the ball stopped, the camera
                # should stay put.
                _LARGE_DIST_PX = 500.0  # ~quarter-field on 1920px
                _SHOT_HOLD_GAP = max(1, int(30 * _fps_scale))  # ~1s
                _SHOT_HOLD_SPEED = 4.0 * (30.0 / _fps)  # px/frame fps-corrected
                _shot_hold = False
                if gap > _SHOT_HOLD_GAP:
                    # Compute speed at gap start from preceding YOLO frames
                    _sh_vel_frames = []
                    for _shi in range(seg_idx, max(seg_idx - 4, -1), -1):
                        _shf = yolo_frames[_shi]
                        if _shf < fi - 10:
                            break
                        _sh_vel_frames.append(_shf)
                    _sh_vel_frames.reverse()
                    if len(_sh_vel_frames) >= 3:
                        _sh_speeds = []
                        for _shk in range(1, len(_sh_vel_frames)):
                            _sha = yolo_by_frame[_sh_vel_frames[_shk - 1]]
                            _shb = yolo_by_frame[_sh_vel_frames[_shk]]
                            _shdt = _sh_vel_frames[_shk] - _sh_vel_frames[_shk - 1]
                            if _shdt > 0:
                                _sh_speeds.append(
                                    math.hypot(
                                        float(_shb.x) - float(_sha.x),
                                        float(_shb.y) - float(_sha.y),
                                    ) / _shdt
                                )
                        if len(_sh_speeds) >= 2 and min(_sh_speeds) > _SHOT_HOLD_SPEED:
                            _shot_hold = True

                    # Pan-hold: large-distance gap where ball was NOT fast.
                    # Use centroid tracking during the gap to distinguish:
                    #   - Real ball movement (cross/pass): centroid tracks
                    #     the ball toward the endpoint → allow interpolation
                    #   - Pan artifact: centroid stays near the start or
                    #     moves differently → hold at start position
                    # This generalises across clip types: crosses keep the
                    # camera following, post-goal pans keep it still.
                    if not _shot_hold and dist > _LARGE_DIST_PX:
                        _centroid_xs = []
                        for _ci in range(fi + 1, fj):
                            if int(source_labels[_ci]) in (
                                int(FUSE_CENTROID), int(FUSE_BLENDED),
                            ):
                                _centroid_xs.append(float(positions[_ci, 0]))
                        _pan_hold_activate = False
                        if len(_centroid_xs) >= 5:
                            _centroid_med = float(np.median(_centroid_xs))
                            _interp_dx = x1 - x0
                            _centroid_drift = _centroid_med - x0
                            # If centroid median stayed within 30% of the
                            # interpolation distance from the start, the
                            # ball didn't really travel → pan artifact.
                            if abs(_interp_dx) > 1.0:
                                _drift_ratio = _centroid_drift / _interp_dx
                                if _drift_ratio < 0.30:
                                    _pan_hold_activate = True
                        else:
                            # Very few centroid detections — can't tell.
                            # Default to interpolation (avoid false freezes).
                            pass
                        if _pan_hold_activate:
                            _shot_hold = True
                            _sh_vel_frames = []  # zero velocity — hold at start

                if _shot_hold:
                    # Hold at last YOLO with decelerating extrapolation,
                    # then freeze.  This matches "ball enters net" physics.
                    _SH_DECEL = 0.90 ** (30.0 / _fps)  # per-frame decay
                    _SH_EXTRAP = min(15, gap // 2)       # decel frames
                    _SH_CONF = 0.25                       # low confidence
                    _sh_yi = yolo_by_frame[fi]
                    # Average velocity from the pre-gap YOLO segment
                    _sh_vx_sum = 0.0
                    _sh_vy_sum = 0.0
                    _sh_vpairs = 0
                    for _shk in range(1, len(_sh_vel_frames)):
                        _sha = yolo_by_frame[_sh_vel_frames[_shk - 1]]
                        _shb = yolo_by_frame[_sh_vel_frames[_shk]]
                        _shdt = _sh_vel_frames[_shk] - _sh_vel_frames[_shk - 1]
                        if _shdt > 0:
                            _sh_vx_sum += (float(_shb.x) - float(_sha.x)) / _shdt
                            _sh_vy_sum += (float(_shb.y) - float(_sha.y)) / _shdt
                            _sh_vpairs += 1
                    _sh_vx = _sh_vx_sum / max(1, _sh_vpairs)
                    _sh_vy = _sh_vy_sum / max(1, _sh_vpairs)
                    # Compute the deceleration endpoint (geometric sum)
                    _sh_hold_x = float(_sh_yi.x)
                    _sh_hold_y = float(_sh_yi.y)
                    _sh_decay_vx = _sh_vx
                    _sh_decay_vy = _sh_vy
                    for _shd in range(_SH_EXTRAP):
                        _sh_hold_x += _sh_decay_vx
                        _sh_hold_y += _sh_decay_vy
                        _sh_decay_vx *= _SH_DECEL
                        _sh_decay_vy *= _SH_DECEL
                    # Clamp hold position to frame bounds
                    _sh_hold_x = max(0.0, min(width, _sh_hold_x))
                    _sh_hold_y = max(0.0, min(height, _sh_hold_y))
                    # Fill the gap
                    _sh_cur_x = float(_sh_yi.x)
                    _sh_cur_y = float(_sh_yi.y)
                    _sh_cur_vx = _sh_vx
                    _sh_cur_vy = _sh_vy
                    for k in range(fi + 1, fj):
                        _sh_t = k - fi
                        if _sh_t <= _SH_EXTRAP:
                            _sh_cur_x += _sh_cur_vx
                            _sh_cur_y += _sh_cur_vy
                            _sh_cur_vx *= _SH_DECEL
                            _sh_cur_vy *= _SH_DECEL
                        else:
                            _sh_cur_x = _sh_hold_x
                            _sh_cur_y = _sh_hold_y
                        positions[k, 0] = max(0.0, min(width, _sh_cur_x))
                        positions[k, 1] = max(0.0, min(height, _sh_cur_y))
                        confidence[k] = _SH_CONF
                        source_labels[k] = FUSE_SHOT_HOLD
                        used_mask[k] = True
                        interpolated += 1
                    _is_pan_hold = len(_sh_vel_frames) == 0
                    _interp_mode = "pan_hold" if _is_pan_hold else "shot_hold"
                    long_flight_info.append((fi, fj, x0, y0, x1, y1, dist, _interp_mode))
                    _hold_label = "Pan-hold" if _is_pan_hold else "Shot-hold"
                    print(
                        f"[FUSION] {_hold_label}: frames {fi}->{fj} "
                        f"({gap} frames), holding near "
                        f"x={float(_sh_yi.x):.0f} (endpoint x={x1:.0f} ignored, "
                        f"dist={dist:.0f}px)"
                    )
                    continue  # skip normal interpolation for this gap

                # Distance-aware mode selection: smoothstep holds near origin
                # which works for short-distance flights (the ball stays
                # nearby), but for large-distance passes/crosses (>500px)
                # the camera MUST follow immediately or it falls behind and
                # the ball leaves the crop.  Use ease-out (quadratic
                # deceleration) for big moves — this front-loads the ball
                # motion to match kicked-ball physics (fast start, gradual
                # deceleration) instead of spreading motion evenly (linear).
                _LARGE_DIST_PX = 500.0  # ~quarter-field on 1920px
                if dist > _LARGE_DIST_PX:
                    _interp_mode = "ease_out"
                elif STEP_THRESHOLD <= gap <= _BASE_LONG_INTERP_GAP:
                    _interp_mode = "smoothstep"
                else:
                    _interp_mode = "linear"
                # (long_flight_info appended after centroid-guide setup below)

            # Use smoothstep ease for moderate-length gaps where the ball
            # stays nearby and the camera has time to settle.
            # For shorter flights, linear interpolation lets the
            # Gaussian absorb rapid direction changes naturally.
            # For very long gaps (> base 90 frames), revert to linear
            # so the camera gradually follows the ball across the field.
            # For large-distance flights (>500px), use ease-out to match
            # kicked-ball physics (fast start, deceleration).
            _flight_dist = math.hypot(x1 - x0, y1 - y0) if is_long else 0.0
            use_step = (is_long
                        and gap >= STEP_THRESHOLD
                        and gap <= _BASE_LONG_INTERP_GAP
                        and _flight_dist <= 500.0)
            use_ease_out = (is_long and _flight_dist > 500.0)

            # --- Centroid-guided interpolation for very long linear gaps ---
            # Pure linear interpolation assumes constant ball speed, which
            # diverges from reality during shots/passes where the ball
            # decelerates, curves, or bounces.  For gaps beyond the base
            # LONG_INTERP_GAP (enabled by the sparse-YOLO extension), extract
            # the centroid tracker's motion shape (deviation from the centroid's
            # own linear trend) and add it to the YOLO linear path.  This
            # preserves YOLO endpoint anchoring while curving the interpolated
            # path to follow on-field motion patterns.
            _cg_offsets: dict[int, float] = {}
            if is_long and not use_step and gap > _BASE_LONG_INTERP_GAP:
                _CG_BLEND_BASE = 0.40   # fraction of centroid residual to apply
                _CG_MAX_PX_BASE = 200.0 # max offset per frame (px)
                # Scale down centroid-guide influence when YOLO is sparse:
                # at 5% density the centroid is unreliable (tracking player
                # clusters, not the ball), so halve the blend and cap.
                if _yolo_density < 0.15:
                    _cg_trust = max(0.35, _yolo_density / 0.15)
                else:
                    _cg_trust = 1.0
                _CG_BLEND = _CG_BLEND_BASE * _cg_trust
                _CG_MAX_PX = _CG_MAX_PX_BASE * _cg_trust
                _CG_MIN_COV = 0.50    # min centroid coverage to enable guiding
                _CG_SMOOTH = max(3, int(7 * _fps_scale))  # smoothing window (~0.23s)

                # Collect raw centroid x-positions for frames in this gap
                _cg_raw: dict[int, float] = {}
                for k in range(fi + 1, fj):
                    cs = centroid_by_frame.get(k)
                    if cs is not None and math.isfinite(cs.x):
                        _cg_raw[k] = float(cs.x)

                _cg_cov = len(_cg_raw) / max(1, gap - 1)
                if _cg_cov >= _CG_MIN_COV and len(_cg_raw) >= 3:
                    # Smooth centroid x to reduce frame-to-frame jitter
                    _cg_keys = sorted(_cg_raw.keys())
                    _cg_sm: dict[int, float] = {}
                    _hw = _CG_SMOOTH // 2
                    for k in _cg_keys:
                        _nbrs = [
                            _cg_raw[j]
                            for j in range(k - _hw, k + _hw + 1)
                            if j in _cg_raw
                        ]
                        _cg_sm[k] = sum(_nbrs) / len(_nbrs)

                    # Centroid's own linear trend (first to last smoothed)
                    _cs0 = _cg_sm[_cg_keys[0]]
                    _cs1 = _cg_sm[_cg_keys[-1]]
                    _cspan = float(_cg_keys[-1] - _cg_keys[0])

                    # Safety: only apply if centroid moves in the same
                    # general direction as the ball.  Avoids following
                    # player clusters moving opposite to ball flight.
                    _yolo_dir = x1 - x0
                    _cent_dir = _cs1 - _cs0

                    if _cspan > 0 and _yolo_dir * _cent_dir >= 0:
                        for k in _cg_keys:
                            ct = (k - _cg_keys[0]) / _cspan
                            _cl = _cs0 + ct * (_cs1 - _cs0)
                            residual = _cg_sm[k] - _cl

                            # Taper: 0 at gap endpoints, 1.0 in middle
                            t = (k - fi) / float(gap)
                            taper = min(t, 1.0 - t) * 2.0

                            off = _CG_BLEND * taper * residual
                            off = max(-_CG_MAX_PX, min(_CG_MAX_PX, off))
                            _cg_offsets[k] = off

                        _interp_mode = "linear+cguide"
                        _cg_max = max(abs(v) for v in _cg_offsets.values())
                        _cg_mean = sum(abs(v) for v in _cg_offsets.values()) / len(_cg_offsets)
                        print(
                            f"[FUSION] Centroid-guide: frames {fi}->{fj}, "
                            f"{len(_cg_offsets)} frames guided, "
                            f"max_offset={_cg_max:.0f}px, mean_offset={_cg_mean:.0f}px"
                        )

            if is_long:
                long_flight_info.append((fi, fj, x0, y0, x1, y1, dist, _interp_mode))

            for k in range(fi + 1, fj):
                if use_step:
                    # Smoothstep ease from origin to endpoint over the gap.
                    # Keeps the camera near the shot/pass origin for the
                    # first portion so shots on goal stay visible, then
                    # smoothly pans toward the receiver.  The post-smooth
                    # Gaussian (sigma=5) further softens the transition.
                    t = (k - fi) / float(gap)
                    ease = t * t * (3.0 - 2.0 * t)  # smoothstep: S-curve 0→1
                    interp_x = x0 + ease * (x1 - x0)
                    interp_y = y0 + ease * (y1 - y0)
                elif use_ease_out:
                    # Ease-out (quadratic deceleration) for large-distance
                    # flights (>500px).  A kicked ball moves fast initially
                    # and decelerates — front-loading ~75% of motion into
                    # the first half of the gap.  This prevents the camera
                    # from falling behind the real ball during shots/passes.
                    # Preserves YOLO endpoint anchoring (ease=1.0 at t=1.0).
                    t = (k - fi) / float(gap)
                    ease = 1.0 - (1.0 - t) * (1.0 - t)  # quadratic ease-out
                    interp_x = x0 + ease * (x1 - x0)
                    interp_y = y0 + ease * (y1 - y0)
                else:
                    # Short/medium gaps: linear interpolation
                    t = (k - fi) / float(gap)
                    interp_x = x0 + t * (x1 - x0)
                    interp_y = y0 + t * (y1 - y0)

                    # Apply centroid-guided offset for long gaps
                    if k in _cg_offsets:
                        interp_x += _cg_offsets[k]

                positions[k, 0] = interp_x
                positions[k, 1] = interp_y
                confidence[k] = EASE_OUT_CONF if use_ease_out else INTERP_CONF
                source_labels[k] = FUSE_INTERP
                used_mask[k] = True
                interpolated += 1

        # --- BACKWARD HOLD: fill leading frames before first YOLO ---
        # At clip start (e.g., goal kick setup) the ball is often
        # stationary and YOLO may not detect it for many frames.
        # Without this, the centroid tracks player positioning (wrong
        # spot on the field) and the camera starts misframed.
        # Use a generous limit — 2 seconds at source fps — because
        # the start of a clip is especially important for framing.
        #
        # However: if centroid already placed a position near the first
        # YOLO (within LEAD_HOLD_AGREE px), the centroid is probably
        # tracking real action (e.g. defensive build-up) — keep it
        # so the camera follows that play instead of being parked at
        # the YOLO hold point for the first second.
        LEAD_HOLD_MAX = 60  # ~2s at 30fps — generous start-of-clip allowance
        LEAD_HOLD_AGREE = 300.0  # px: centroid within this of YOLO = "agrees"
        first_yolo = yolo_frames[0]
        if first_yolo > 0:
            yf = yolo_by_frame[first_yolo]
            lead_filled = 0
            lead_kept = 0
            for k in range(first_yolo - 1, max(-1, first_yolo - LEAD_HOLD_MAX - 1), -1):
                if k in yolo_by_frame:
                    break
                # If centroid already filled this frame, check whether it
                # agrees with the first YOLO position.  If it does, the
                # centroid is tracking the right area — preserve it so the
                # camera follows the actual play (defensive work, build-up).
                if source_labels[k] == FUSE_CENTROID and used_mask[k]:
                    _cx_k = float(positions[k, 0])
                    _cy_k = float(positions[k, 1])
                    _hold_dist = math.hypot(_cx_k - float(yf.x), _cy_k - float(yf.y))
                    if _hold_dist <= LEAD_HOLD_AGREE:
                        # Centroid is nearby — trust it; just bump confidence
                        # slightly so downstream smoothing doesn't discount it.
                        confidence[k] = max(confidence[k], 0.25)
                        lead_kept += 1
                        continue
                positions[k, 0] = yf.x
                positions[k, 1] = yf.y
                confidence[k] = max(confidence[k], 0.22)
                source_labels[k] = FUSE_HOLD
                used_mask[k] = True
                lead_filled += 1
            if lead_filled > 0 or lead_kept > 0:
                interpolated += lead_filled
                _kept_msg = f" (kept {lead_kept} centroid frames)" if lead_kept > 0 else ""
                print(
                    f"[FUSION] Backward-hold: filled {lead_filled} leading frames "
                    f"from first YOLO at frame {first_yolo}{_kept_msg}"
                )

        # Hold at last YOLO position for trailing frames.
        # When YOLO is sparse, the trailing hold is extended so the camera
        # doesn't revert to centroid (which tracks player clusters) for the
        # entire tail of the clip.  At very low density the hold covers
        # up to 5 seconds (120 frames at 24fps) — long enough for the
        # bidirectional EMA smoothing to not bleed centroid contamination
        # backward into the held region.
        _trailing_hold_max = SHORT_INTERP_GAP  # already fps-scaled
        if _yolo_density < 0.15:
            # At 5% density → ~4s; at 15% → ~0.5s (unchanged)
            _trailing_hold_max = int(SHORT_INTERP_GAP + 105 * (1.0 - _yolo_density / 0.15) * _fps_scale)
        last_yolo = yolo_frames[-1]
        yl = yolo_by_frame[last_yolo]
        for k in range(last_yolo + 1, min(frame_count, last_yolo + _trailing_hold_max)):
            if k in yolo_by_frame:
                break
            positions[k, 0] = yl.x
            positions[k, 1] = yl.y
            confidence[k] = max(confidence[k], 0.25)
            source_labels[k] = FUSE_HOLD
            used_mask[k] = True
            interpolated += 1

        if interpolated > 0:
            _long_msg = f" (long-flight={long_interp})" if long_interp > 0 else ""
            print(
                f"[FUSION] Interpolated {interpolated} frames between YOLO detections "
                f"(short_gap={SHORT_INTERP_GAP}, long_gap={LONG_INTERP_GAP}){_long_msg}"
            )

        # --- VELOCITY EXTRAPOLATION FOR SHOT GAPS ---
        # When the ball is moving fast at the end of a YOLO segment (shot/pass)
        # and then YOLO drops out (motion blur), the centroid fallback tracks
        # player clusters — not the ball.  The camera stays on the shooter
        # while the ball flies into the net, missing the goal entirely.
        #
        # Fix: detect high-speed YOLO segments and extrapolate the ball's
        # trajectory forward with deceleration for centroid-only frames that
        # follow.  This keeps the camera following the ball flight path toward
        # the goal instead of snapping back to the player group.
        _EXTRAP_SPEED_THR = 4.0 * (30.0 / _fps)  # px/frame: fps-corrected speed threshold
        _EXTRAP_MAX_FRAMES = max(1, int(45 * _fps_scale))  # ~1.5s at any fps
        _EXTRAP_DECEL = 0.92 ** (30.0 / _fps)  # per-frame velocity decay (fps-corrected)
        _EXTRAP_CONF = 0.32       # confidence for extrapolated frames
        _EXTRAP_MIN_YOLO = 3      # need at least 3 YOLO frames to estimate velocity
        _extrap_count = 0

        for seg_idx in range(len(yolo_frames) - 1):
            fi = yolo_frames[seg_idx]
            fj = yolo_frames[seg_idx + 1]
            gap = fj - fi
            if gap <= SHORT_INTERP_GAP:
                continue  # already handled by linear interpolation above

            # Compute velocity at the end of the YOLO segment (fi).
            # Use the last few YOLO frames before fi to get a stable estimate.
            _vel_frames = []
            for _vi in range(seg_idx, max(seg_idx - _EXTRAP_MIN_YOLO, -1), -1):
                _vf = yolo_frames[_vi]
                if _vf < fi - 10:
                    break
                _vel_frames.append(_vf)
            _vel_frames.reverse()

            if len(_vel_frames) < 2:
                continue

            # Average velocity over the last few YOLO frames
            _vx_sum = 0.0
            _vy_sum = 0.0
            _vpairs = 0
            for _vk in range(1, len(_vel_frames)):
                _va = yolo_by_frame[_vel_frames[_vk - 1]]
                _vb = yolo_by_frame[_vel_frames[_vk]]
                _vdt = _vel_frames[_vk] - _vel_frames[_vk - 1]
                if _vdt > 0:
                    _vx_sum += (float(_vb.x) - float(_va.x)) / _vdt
                    _vy_sum += (float(_vb.y) - float(_va.y)) / _vdt
                    _vpairs += 1

            if _vpairs == 0:
                continue

            _vx = _vx_sum / _vpairs
            _vy = _vy_sum / _vpairs
            _speed = math.hypot(_vx, _vy)

            if _speed < _EXTRAP_SPEED_THR:
                continue  # not a shot — skip

            # Extrapolate forward from fi with decelerating velocity.
            # Override centroid AND interpolated frames when the
            # physics-based extrapolation diverges significantly
            # from the existing position.  The measured velocity at
            # the gap boundary is more accurate than ease-out's
            # assumed curve shape for the first few frames.
            _yi = yolo_by_frame[fi]
            _ex = float(_yi.x)
            _ey = float(_yi.y)
            _evx = _vx
            _evy = _vy
            _seg_extrap = 0
            for k in range(fi + 1, min(fj, fi + _EXTRAP_MAX_FRAMES + 1)):
                if k in yolo_by_frame:
                    break  # YOLO available — don't override

                _ex += _evx
                _ey += _evy
                _evx *= _EXTRAP_DECEL
                _evy *= _EXTRAP_DECEL

                # Stop extrapolating if the trajectory has left the
                # visible frame — the ball is off-screen and continuing
                # would just clamp at the edge, dragging the camera to
                # the sideline / coach area.
                _EXTRAP_EDGE_PX = width * 0.03  # small inset
                if _ex <= _EXTRAP_EDGE_PX or _ex >= width - _EXTRAP_EDGE_PX:
                    break

                # Clamp to frame bounds (safety)
                _ex_clamped = max(0.0, min(width, _ex))
                _ey_clamped = max(0.0, min(height, _ey))

                # Override existing position if the extrapolation
                # diverges by >100px (position is tracking something
                # else) OR the frame has no data.
                # NEVER override FUSE_INTERP or FUSE_SHOT_HOLD frames:
                # those were set by ease_out / smoothstep interpolation
                # or by shot-hold / pan-hold between YOLO anchors — far
                # more reliable than a velocity estimate from 1-2
                # consecutive frames dominated by detection jitter.
                if source_labels[k] in (FUSE_INTERP, FUSE_SHOT_HOLD):
                    continue  # preserve YOLO-anchored interpolation / hold
                # For FUSE_HOLD (backward-fill), only override in early
                # extrapolation where velocity estimates are still fresh.
                # After ~15 frames, 0.92 decel reduces speed to 29% of
                # initial — hold position is more reliable at that point.
                _extrap_step = k - fi
                if source_labels[k] == FUSE_HOLD and _extrap_step > 10:
                    continue
                _cx_cur = float(positions[k, 0]) if used_mask[k] else _ex_clamped
                _cy_cur = float(positions[k, 1]) if used_mask[k] else _ey_clamped
                _extrap_dist = math.hypot(_ex_clamped - _cx_cur, _ey_clamped - _cy_cur)
                if _extrap_dist > 100.0 or not used_mask[k]:
                    positions[k, 0] = _ex_clamped
                    positions[k, 1] = _ey_clamped
                    confidence[k] = max(confidence[k], _EXTRAP_CONF)
                    source_labels[k] = FUSE_INTERP
                    used_mask[k] = True
                    _seg_extrap += 1

            if _seg_extrap > 0:
                _extrap_count += _seg_extrap

        # Also extrapolate past the LAST YOLO frame if the ball was
        # moving fast (shot toward goal at end of clip).
        if len(yolo_frames) >= _EXTRAP_MIN_YOLO:
            _last_fi = yolo_frames[-1]
            _vel_frames = yolo_frames[max(0, len(yolo_frames) - _EXTRAP_MIN_YOLO - 1):]
            if len(_vel_frames) >= 2:
                _vx_sum = 0.0
                _vy_sum = 0.0
                _vpairs = 0
                for _vk in range(1, len(_vel_frames)):
                    _va = yolo_by_frame[_vel_frames[_vk - 1]]
                    _vb = yolo_by_frame[_vel_frames[_vk]]
                    _vdt = _vel_frames[_vk] - _vel_frames[_vk - 1]
                    if _vdt > 0:
                        _vx_sum += (float(_vb.x) - float(_va.x)) / _vdt
                        _vy_sum += (float(_vb.y) - float(_va.y)) / _vdt
                        _vpairs += 1
                if _vpairs > 0:
                    _vx = _vx_sum / _vpairs
                    _vy = _vy_sum / _vpairs
                    _speed = math.hypot(_vx, _vy)
                    if _speed >= _EXTRAP_SPEED_THR:
                        _yl = yolo_by_frame[_last_fi]
                        _ex = float(_yl.x)
                        _ey = float(_yl.y)
                        _evx = _vx
                        _evy = _vy
                        for k in range(_last_fi + 1, min(frame_count, _last_fi + _EXTRAP_MAX_FRAMES + 1)):
                            if k in yolo_by_frame:
                                break
                            _ex += _evx
                            _ey += _evy
                            _evx *= _EXTRAP_DECEL
                            _evy *= _EXTRAP_DECEL
                            # Stop if trajectory exits visible frame
                            _EXTRAP_EDGE_PX_T = width * 0.03
                            if _ex <= _EXTRAP_EDGE_PX_T or _ex >= width - _EXTRAP_EDGE_PX_T:
                                break
                            _ex_c = max(0.0, min(width, _ex))
                            _ey_c = max(0.0, min(height, _ey))
                            if source_labels[k] in (FUSE_INTERP, FUSE_SHOT_HOLD):
                                continue  # preserve interpolation / hold
                            _extrap_step_t = k - _last_fi
                            if source_labels[k] == FUSE_HOLD and _extrap_step_t > 10:
                                continue
                            _cx_cur = float(positions[k, 0]) if used_mask[k] else _ex_c
                            _cy_cur = float(positions[k, 1]) if used_mask[k] else _ey_c
                            _d = math.hypot(_ex_c - _cx_cur, _ey_c - _cy_cur)
                            if _d > 100.0 or not used_mask[k]:
                                positions[k, 0] = _ex_c
                                positions[k, 1] = _ey_c
                                confidence[k] = max(confidence[k], _EXTRAP_CONF)
                                source_labels[k] = FUSE_INTERP
                                used_mask[k] = True
                                _extrap_count += 1

        if _extrap_count > 0:
            print(
                f"[FUSION] Velocity extrapolation: filled {_extrap_count} frames "
                f"during high-speed ball flight (speed_thr={_EXTRAP_SPEED_THR:.1f} px/f, "
                f"decel={_EXTRAP_DECEL:.3f}, fps_scale={_fps_scale:.2f})"
            )

        # --- SOFT YOLO-INTERPOLATION ANCHOR BLEND ---
        # After hard interpolation and hold passes, remaining centroid-only
        # frames can still wander freely — the centroid tracks the largest
        # motion region (player clusters), not the ball.  This causes the
        # camera to chase players while the ball drifts out of frame.
        #
        # For each remaining centroid-only frame that sits between two YOLO
        # detections, compute the YOLO-interpolated position and blend the
        # centroid toward it.  The blend weight decays with temporal distance
        # from the nearest YOLO, so frames close to a YOLO are strongly
        # anchored while distant frames still allow some centroid influence.
        #
        # This is a soft version of the hard interpolation above — it doesn't
        # replace centroid, just pulls it toward the YOLO-derived trajectory.
        if _yolo_density < 0.15 and len(yolo_frames) >= 2:
            ANCHOR_RANGE = 90       # max frames from nearest YOLO to still anchor
            ANCHOR_BLEND_MAX = 0.55  # peak blend weight toward YOLO-interpolated pos
            # Cap the maximum pixel displacement the anchor can apply.
            # When the centroid tracks genuine ball movement that diverges
            # from the YOLO-interpolated trajectory (e.g., ball lingers on
            # the right while YOLO endpoints are on the left), an uncapped
            # blend drags the position hundreds of pixels away from the
            # actual ball.  Capping at ~1/3 of portrait crop width prevents
            # the anchor from overriding a plausibly-correct centroid.
            MAX_ANCHOR_DISP = 200.0  # px: max displacement per axis
            _anchored = 0
            _capped = 0
            for i in range(frame_count):
                if source_labels[i] != FUSE_CENTROID:
                    continue  # only process centroid-only frames

                # Find the flanking YOLO frames (prev_yolo <= i <= next_yolo)
                _prev_yf = -1
                _next_yf = -1
                for yf in yolo_frames:
                    if yf <= i:
                        _prev_yf = yf
                    if yf >= i and _next_yf < 0:
                        _next_yf = yf
                        break

                if _prev_yf >= 0 and _next_yf >= 0 and _prev_yf != _next_yf:
                    # Interpolate between flanking YOLO detections
                    _gap = _next_yf - _prev_yf
                    _t = (i - _prev_yf) / float(_gap)
                    _yp = yolo_by_frame[_prev_yf]
                    _yn = yolo_by_frame[_next_yf]
                    _interp_x = float(_yp.x) + _t * (float(_yn.x) - float(_yp.x))
                    _interp_y = float(_yp.y) + _t * (float(_yn.y) - float(_yp.y))
                    # Blend weight: strongest near YOLOs, decays with distance.
                    _nearest_dist = min(i - _prev_yf, _next_yf - i)
                    if _nearest_dist > ANCHOR_RANGE:
                        continue
                    _decay = 1.0 - _nearest_dist / ANCHOR_RANGE
                    _w = ANCHOR_BLEND_MAX * _decay
                elif _prev_yf >= 0 and _next_yf < 0:
                    # After last YOLO — anchor toward last YOLO
                    _dist = i - _prev_yf
                    if _dist > ANCHOR_RANGE:
                        continue
                    _yp = yolo_by_frame[_prev_yf]
                    _interp_x = float(_yp.x)
                    _interp_y = float(_yp.y)
                    _decay = 1.0 - _dist / ANCHOR_RANGE
                    _w = ANCHOR_BLEND_MAX * 0.7 * _decay  # weaker for trailing
                elif _next_yf >= 0 and _prev_yf < 0:
                    # Before first YOLO — anchor toward first YOLO
                    _dist = _next_yf - i
                    if _dist > ANCHOR_RANGE:
                        continue
                    _yn = yolo_by_frame[_next_yf]
                    _interp_x = float(_yn.x)
                    _interp_y = float(_yn.y)
                    _decay = 1.0 - _dist / ANCHOR_RANGE
                    _w = ANCHOR_BLEND_MAX * 0.7 * _decay
                else:
                    continue

                if _w > 0.01:
                    _cx = float(positions[i, 0])
                    _cy = float(positions[i, 1])
                    _blended_x = _w * _interp_x + (1.0 - _w) * _cx
                    _blended_y = _w * _interp_y + (1.0 - _w) * _cy
                    # Cap displacement: don't drag the position more than
                    # MAX_ANCHOR_DISP from the original centroid.  This
                    # prevents the anchor from overriding the centroid when
                    # the ball genuinely moved away from the YOLO trajectory.
                    _dx = _blended_x - _cx
                    _dy = _blended_y - _cy
                    _disp = math.hypot(_dx, _dy)
                    if _disp > MAX_ANCHOR_DISP:
                        _scale = MAX_ANCHOR_DISP / _disp
                        _blended_x = _cx + _dx * _scale
                        _blended_y = _cy + _dy * _scale
                        _capped += 1
                    positions[i, 0] = _blended_x
                    positions[i, 1] = _blended_y
                    confidence[i] = max(confidence[i], 0.22 + 0.13 * _decay)
                    _anchored += 1

            if _anchored > 0:
                _cap_msg = f", capped={_capped}" if _capped > 0 else ""
                print(
                    f"[FUSION] YOLO-anchor blend: adjusted {_anchored} centroid frames "
                    f"(density={_yolo_density:.1%}, range={ANCHOR_RANGE}f, "
                    f"max_disp={MAX_ANCHOR_DISP:.0f}px{_cap_msg})"
                )

        _src_fps = 30.0  # source clips are always 30fps
        for _lf in long_flight_info:
            _fi, _fj, _x0, _y0, _x1, _y1, _dist, _mode = _lf
            _t0 = _fi / _src_fps
            _t1 = _fj / _src_fps
            print(
                f"[FUSION] Long-flight: frames {_fi}->{_fj} "
                f"(t={_t0:.1f}s->{_t1:.1f}s, {_fj - _fi} frames), "
                f"ball x={_x0:.0f}->{_x1:.0f} ({_dist:.0f}px) [{_mode}]"
            )
        # Log ball_x at 1-second intervals for the full clip
        _n_pos = len(positions)
        _step = int(_src_fps)
        _sample_frames = list(range(0, _n_pos, _step))
        if _sample_frames and _sample_frames[-1] != _n_pos - 1:
            _sample_frames.append(_n_pos - 1)
        if _sample_frames:
            _parts = [f"f{_sf}={positions[_sf, 0]:.0f}" for _sf in _sample_frames]
            print(f"[FUSION] Ball x timeline (1s steps): {', '.join(_parts)}")

    # --- Final safety net: fill any remaining all-NaN frames ---
    # After all fusion passes, some frames may still have NaN positions
    # (e.g., very short clips where neither YOLO nor centroid had data).
    # Fill them with frame centre so the camera always has a valid target
    # and downstream code (planner, renderer) never encounters NaN.
    _nan_mask = np.isnan(positions[:, 0])
    _nan_count = int(_nan_mask.sum())
    if _nan_count > 0 and _nan_count == frame_count:
        # All frames are NaN — hold at frame centre
        print(f"[FUSION] All {frame_count} frames are NaN after fusion; holding at frame centre")
        positions[:, 0] = width * 0.5
        positions[:, 1] = height * 0.5
        confidence[:] = 0.10
        used_mask[:] = True
        source_labels[:] = FUSE_HOLD
    elif _nan_count > 0:
        # Partial NaN — forward-fill from nearest valid frame
        _last_x, _last_y = width * 0.5, height * 0.5
        for _fi in range(frame_count):
            if not _nan_mask[_fi]:
                _last_x = float(positions[_fi, 0])
                _last_y = float(positions[_fi, 1])
            else:
                positions[_fi, 0] = _last_x
                positions[_fi, 1] = _last_y
                used_mask[_fi] = True
                confidence[_fi] = max(confidence[_fi], 0.10)
                if source_labels[_fi] == FUSE_NONE:
                    source_labels[_fi] = FUSE_HOLD

    return positions, used_mask, confidence, source_labels


# ---------------------------------------------------------------------------
# CSRT visual tracker: fill YOLO detection gaps with real pixel tracking
# ---------------------------------------------------------------------------


def _tracker_cache_path(video_path: Path, model_name: str | None = None) -> Path:
    """Return the canonical tracker cache path for a video."""
    stem = Path(video_path).stem
    # Include YOLO model in tracker cache name so different YOLO models
    # produce different tracker caches (gaps differ per model).
    if model_name:
        _model_stem = Path(model_name).stem
        _default_stem = Path(DEFAULT_MODEL_NAME).stem
        if _model_stem != _default_stem:
            return Path("out") / "telemetry" / f"{stem}.tracker_ball.{_model_stem}.jsonl"
    return Path("out") / "telemetry" / f"{stem}.tracker_ball.jsonl"


def _file_md5(path: Path) -> str:
    """Compute MD5 hex digest of a file for cache invalidation."""
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_tracker_cache(
    path: Path, yolo_hash: str, bbox_size: int
) -> list[dict] | None:
    """Load tracker cache, returning None if invalid/stale."""
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            if not first_line:
                return None
            meta = json.loads(first_line)
            if not meta.get("_meta"):
                return None
            if meta.get("yolo_hash") != yolo_hash:
                print(f"[TRACKER] Cache stale (YOLO hash mismatch), re-tracking")
                return None
            if meta.get("bbox_size") != bbox_size:
                print(f"[TRACKER] Cache stale (bbox_size mismatch), re-tracking")
                return None
            results = []
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
            return results
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[TRACKER] Cache load error: {exc}")
        return None


def _save_tracker_cache(
    path: Path, results: list[dict], yolo_hash: str, bbox_size: int
) -> None:
    """Write tracker results to JSONL cache with metadata header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        meta = {"_meta": True, "yolo_hash": yolo_hash, "bbox_size": bbox_size, "version": 1}
        f.write(json.dumps(meta) + "\n")
        for rec in results:
            f.write(json.dumps(rec) + "\n")
    print(f"[TRACKER] Cached {len(results)} tracker positions to {path}")


def _apply_tracker_cache(
    cached: list[dict],
    positions: np.ndarray,
    source_labels: np.ndarray,
    confidence: np.ndarray,
    used_mask: np.ndarray,
) -> int:
    """Apply cached tracker results to the fusion arrays. Returns count applied."""
    count = 0
    frame_count = len(positions)
    for rec in cached:
        fi = int(rec["frame"])
        if 0 <= fi < frame_count:
            positions[fi, 0] = float(rec["cx"])
            positions[fi, 1] = float(rec["cy"])
            source_labels[fi] = np.uint8(FUSE_TRACKER)
            confidence[fi] = float(rec.get("conf", 0.70))
            used_mask[fi] = True
            count += 1
    return count


def run_csrt_tracker_for_gaps(
    video_path: str | Path,
    positions: np.ndarray,
    source_labels: np.ndarray,
    confidence: np.ndarray,
    used_mask: np.ndarray,
    *,
    fps: float = 30.0,
    bbox_size: int = 48,
    max_jump_px: float = 60.0,
    min_gap: int = 3,
    max_gap: int = 150,
    tracker_conf: float = 0.70,
    cache: bool = True,
    yolo_model_name: str | None = None,
) -> int:
    """Fill FUSE_INTERP gaps with CSRT visual tracker positions.

    Identifies contiguous runs of FUSE_INTERP frames flanked by YOLO/BLENDED
    detections, initializes a CSRT tracker at the leading anchor frame, and
    tracks forward frame-by-frame through the actual video.

    Modifies *positions*, *source_labels*, *confidence*, *used_mask* in-place.
    Returns the number of frames successfully tracked.
    """
    try:
        import cv2
    except ImportError:
        print("[TRACKER] OpenCV not available, skipping visual tracking")
        return 0

    # Verify CSRT is available
    if not hasattr(cv2, "legacy") or not hasattr(cv2.legacy, "TrackerCSRT_create"):
        print("[TRACKER] cv2.legacy.TrackerCSRT not available, skipping")
        return 0

    video_path = Path(video_path)
    frame_count = len(positions)

    # --- Cache check ---
    cache_path = _tracker_cache_path(video_path, yolo_model_name)
    # Find the YOLO cache to compute hash for invalidation
    _yolo_cache = Path(yolo_telemetry_path_for_video(video_path))
    if yolo_model_name:
        _model_stem = Path(yolo_model_name).stem
        _default_stem = Path(DEFAULT_MODEL_NAME).stem
        if _model_stem != _default_stem:
            _yolo_cache = _yolo_cache.with_suffix(f".{_model_stem}.jsonl")
    yolo_hash = _file_md5(_yolo_cache) if _yolo_cache.is_file() else ""

    if cache and cache_path.is_file():
        cached = _load_tracker_cache(cache_path, yolo_hash, bbox_size)
        if cached is not None:
            applied = _apply_tracker_cache(cached, positions, source_labels,
                                           confidence, used_mask)
            print(f"[TRACKER] Loaded {applied} cached tracker positions from {cache_path}")
            return applied

    # --- Identify trackable gaps ---
    # A gap is a contiguous run of FUSE_INTERP (4) frames flanked by
    # FUSE_YOLO (1) or FUSE_BLENDED (3) anchors on both sides.
    _INTERP = 4
    _YOLO_ANCHORS = (1, 3)  # FUSE_YOLO, FUSE_BLENDED

    gaps: list[tuple[int, int, int, int]] = []  # (anchor_before, gap_start, gap_end, anchor_after)
    i = 0
    while i < frame_count:
        if source_labels[i] == _INTERP:
            gap_start = i
            while i < frame_count and source_labels[i] == _INTERP:
                i += 1
            gap_end = i - 1  # inclusive
            gap_len = gap_end - gap_start + 1

            anchor_before = gap_start - 1
            anchor_after = gap_end + 1

            if (gap_len >= min_gap
                    and gap_len <= max_gap
                    and anchor_before >= 0
                    and anchor_after < frame_count
                    and source_labels[anchor_before] in _YOLO_ANCHORS
                    and source_labels[anchor_after] in _YOLO_ANCHORS
                    and not np.isnan(positions[anchor_before]).any()):
                gaps.append((anchor_before, gap_start, gap_end, anchor_after))
        else:
            i += 1

    if not gaps:
        print("[TRACKER] No trackable INTERP gaps found")
        return 0

    total_gap_frames = sum(g[2] - g[1] + 1 for g in gaps)
    print(f"[TRACKER] Found {len(gaps)} trackable gaps ({total_gap_frames} total frames)")

    # --- Open video and track ---
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[TRACKER] Cannot open video: {video_path}")
        return 0

    fps_scale = max(fps, 1.0) / 30.0
    max_jump_scaled = max_jump_px * fps_scale
    half_bbox = bbox_size // 2
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracked_results: list[dict] = []
    total_tracked = 0

    for gap_idx, (anchor_frame, gap_start, gap_end, anchor_after) in enumerate(gaps):
        # Seek to anchor frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, anchor_frame)
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"[TRACKER] Gap {gap_start}-{gap_end}: failed to read anchor frame {anchor_frame}")
            continue

        # Initialize CSRT at anchor position
        ax = float(positions[anchor_frame, 0])
        ay = float(positions[anchor_frame, 1])
        x0 = max(0.0, ax - half_bbox)
        y0 = max(0.0, ay - half_bbox)
        # Clamp bbox to stay within frame
        if x0 + bbox_size > vid_w:
            x0 = max(0.0, vid_w - bbox_size)
        if y0 + bbox_size > vid_h:
            y0 = max(0.0, vid_h - bbox_size)
        init_bbox = (float(x0), float(y0), float(bbox_size), float(bbox_size))

        tracker = cv2.legacy.TrackerCSRT_create()
        tracker.init(frame, init_bbox)

        prev_cx, prev_cy = ax, ay
        gap_tracked: list[tuple[int, float, float]] = []
        tracker_lost = False

        for fi in range(gap_start, gap_end + 1):
            ok, frame = cap.read()
            if not ok or frame is None:
                tracker_lost = True
                break

            ok_track, tracked_bbox = tracker.update(frame)
            if not ok_track:
                tracker_lost = True
                break

            bx, by, bw, bh = tracked_bbox
            cx = bx + bw / 2.0
            cy = by + bh / 2.0

            # Validate: speed check — reject if jumped too far
            jump = math.hypot(cx - prev_cx, cy - prev_cy)
            if jump > max_jump_scaled:
                print(f"[TRACKER] Gap {gap_start}-{gap_end}: speed violation at frame {fi} "
                      f"({jump:.1f}px > {max_jump_scaled:.1f}px limit)")
                tracker_lost = True
                break

            # Validate: frame bounds (reject near-edge positions)
            edge_margin = bbox_size
            if cx < edge_margin or cx > vid_w - edge_margin or \
               cy < edge_margin or cy > vid_h - edge_margin:
                tracker_lost = True
                break

            gap_tracked.append((fi, cx, cy))
            prev_cx, prev_cy = cx, cy

        # Endpoint validation: tracker's last position should converge
        # toward the YOLO anchor after the gap
        if gap_tracked and not tracker_lost:
            last_cx = gap_tracked[-1][1]
            last_cy = gap_tracked[-1][2]
            anchor_cx = float(positions[anchor_after, 0])
            anchor_cy = float(positions[anchor_after, 1])
            endpoint_dist = math.hypot(last_cx - anchor_cx, last_cy - anchor_cy)
            max_endpoint = bbox_size * 3.0
            if endpoint_dist > max_endpoint:
                print(f"[TRACKER] Gap {gap_start}-{gap_end}: endpoint divergence "
                      f"{endpoint_dist:.0f}px > {max_endpoint:.0f}px, discarding "
                      f"{len(gap_tracked)} frames")
                gap_tracked = []

        # Apply results
        for fi, cx, cy in gap_tracked:
            positions[fi, 0] = cx
            positions[fi, 1] = cy
            source_labels[fi] = np.uint8(FUSE_TRACKER)
            confidence[fi] = tracker_conf
            used_mask[fi] = True
            total_tracked += 1
            tracked_results.append({
                "frame": fi,
                "cx": round(cx, 2),
                "cy": round(cy, 2),
                "conf": round(tracker_conf, 4),
                "gap_start": gap_start,
                "gap_end": gap_end,
            })

        status = f"{len(gap_tracked)}/{gap_end - gap_start + 1} frames"
        if tracker_lost and not gap_tracked:
            status += " (tracker lost immediately)"
        elif tracker_lost:
            status += f" (tracker lost at frame {gap_tracked[-1][0] + 1})"
        print(f"[TRACKER] Gap {gap_start}-{gap_end}: tracked {status}")

    cap.release()

    # --- Save cache ---
    if cache and tracked_results:
        _save_tracker_cache(cache_path, tracked_results, yolo_hash, bbox_size)

    print(f"[TRACKER] Total: {total_tracked}/{total_gap_frames} gap frames tracked "
          f"across {len(gaps)} gaps")
    return total_tracked


__all__ = [
    "BallSample",
    "ExcludeZone",
    "load_exclude_zones",
    "load_ball_telemetry",
    "load_ball_telemetry_for_clip",
    "telemetry_path_for_video",
    "yolo_telemetry_path_for_video",
    "save_ball_telemetry_jsonl",
    "load_and_interpolate_telemetry",
    "set_telemetry_frame_bounds",
    "smooth_telemetry",
    "run_yolo_ball_detection",
    "fuse_yolo_and_centroid",
    "run_csrt_tracker_for_gaps",
    "FUSE_TRACKER",
]


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


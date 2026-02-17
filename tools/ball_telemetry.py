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
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Optional, Sequence, Tuple

import numpy as np


DEFAULT_MODEL_NAME = "yolov8n.pt"
DEFAULT_MIN_CONF = 0.35


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
) -> list[BallSample]:
    """Run YOLO ball detection on every frame, returning BallSamples with confidence.

    Results are cached to ``out/telemetry/<stem>.yolo_ball.jsonl``.  If
    a cache file already exists and *cache* is True, the cached result
    is returned without re-running detection.

    Uses the BallTracker from soccer_highlights.ball_tracker which wraps
    ultralytics YOLO with constant-velocity smoothing.
    """
    video_path = Path(video_path)
    cache_path = Path(yolo_telemetry_path_for_video(video_path))

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
        weights_path=None,  # uses default yolov8n.pt
        min_conf=min_conf,
        device="cpu",
        input_size=1280,    # full resolution — critical for small soccer balls
        smooth_alpha=0.25,
        max_gap=12,
    )
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


def fuse_yolo_and_centroid(
    yolo_samples: list[BallSample],
    centroid_samples: list[BallSample],
    frame_count: int,
    width: float,
    height: float,
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
    """
    FUSE_NONE = np.uint8(0)
    FUSE_YOLO = np.uint8(1)
    FUSE_CENTROID = np.uint8(2)
    FUSE_BLENDED = np.uint8(3)
    FUSE_INTERP = np.uint8(4)
    FUSE_HOLD = np.uint8(5)

    positions = np.full((frame_count, 2), np.nan, dtype=np.float32)
    used_mask = np.zeros(frame_count, dtype=bool)
    confidence = np.zeros(frame_count, dtype=np.float32)
    source_labels = np.zeros(frame_count, dtype=np.uint8)

    # Edge margin: YOLO detections within this fraction of the frame edge
    # are almost always false positives (goalposts, shadows, partial views).
    # Filter them at ingestion time so they don't contaminate the centroid
    # gating via _last_yolo_x for many subsequent frames.
    EDGE_MARGIN_FRAC = 0.04  # 4% of frame width ≈ 77px on 1920px source

    # Index YOLO samples by frame, filtering out near-edge detections.
    _edge_px_ingest = width * EDGE_MARGIN_FRAC if width > 0 else 0
    _edge_filtered = 0
    yolo_by_frame: dict[int, BallSample] = {}
    for s in yolo_samples:
        fidx = int(s.frame)
        if 0 <= fidx < frame_count and math.isfinite(s.x) and math.isfinite(s.y):
            if _edge_px_ingest > 0 and (s.x < _edge_px_ingest or s.x > width - _edge_px_ingest):
                _edge_filtered += 1
                continue
            yolo_by_frame[fidx] = s
    if _edge_filtered > 0:
        print(
            f"[FUSION] Filtered {_edge_filtered} near-edge YOLO detections "
            f"(margin={_edge_px_ingest:.0f}px)"
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
    conf_threshold = 0.28  # YOLO conf above which we trust it fully (lowered from 0.40)

    # Track last trusted YOLO position for centroid gating.
    # When centroid-only frames appear far from the last YOLO position,
    # the centroid is probably tracking a player cluster, not the ball.
    # In that case we blend toward the YOLO hold to prevent the camera
    # from snapping to the wrong part of the field.
    _last_yolo_x: Optional[float] = None
    _last_yolo_y: Optional[float] = None
    YOLO_HOLD_DIST = 150.0  # px: centroid beyond this → suspect
    YOLO_HOLD_BLEND = 0.65  # weight for last YOLO when centroid diverges

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
                _last_yolo_x, _last_yolo_y = float(yolo.x), float(yolo.y)
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
                _last_yolo_x, _last_yolo_y = float(yolo.x), float(yolo.y)
            used_mask[i] = True
        elif yolo is not None:
            # Only YOLO
            positions[i, 0] = yolo.x
            positions[i, 1] = yolo.y
            confidence[i] = max(0.0, min(1.0, yolo.conf))
            source_labels[i] = FUSE_YOLO
            used_mask[i] = True
            yolo_used += 1
            _last_yolo_x, _last_yolo_y = float(yolo.x), float(yolo.y)
        elif centroid is not None:
            cx, cy = float(centroid.x), float(centroid.y)
            # Gate centroid against last known YOLO position.
            if _last_yolo_x is not None:
                dist = math.hypot(cx - _last_yolo_x, cy - _last_yolo_y)
                if dist > YOLO_HOLD_DIST:
                    # Centroid diverged — blend toward last YOLO position.
                    w = YOLO_HOLD_BLEND
                    cx = w * _last_yolo_x + (1.0 - w) * cx
                    cy = w * _last_yolo_y + (1.0 - w) * cy
                    confidence[i] = 0.22  # lower: uncertain position
                else:
                    confidence[i] = 0.30  # centroid agrees with YOLO area
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
    SHORT_INTERP_GAP = 15   # always interpolate gaps <= this (~0.5s at 30fps)
    LONG_INTERP_GAP = 90    # max gap for flight interpolation (~3s at 30fps)
    MIN_FLIGHT_DIST = 100.0  # px: long gaps only interpolated if ball clearly traveled
    INTERP_CONF = 0.28       # confidence for interpolated frames (lowered from 0.35)
    # Step-function threshold: only use step (jump to receiver) for gaps
    # >= this many frames.  Shorter flights use linear interpolation so the
    # Gaussian can absorb rapid back-and-forth movements without whipsawing
    # the camera.  45 frames ≈ 1.5s at 30fps — enough for the camera to
    # arrive and settle at the receiver.
    STEP_THRESHOLD = 45

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
                        f"frames {fi}→{fj}, ball x={x0:.0f}→{x1:.0f} "
                        f"(edge margin={_edge_px:.0f}px)"
                    )
                    continue
                long_interp += gap - 1
                _interp_mode = "step" if gap >= STEP_THRESHOLD else "linear"
                long_flight_info.append((fi, fj, x0, y0, x1, y1, dist, _interp_mode))

            # Use step function only for long-enough gaps where the
            # camera has time to arrive and settle at the receiver.
            # For shorter flights, linear interpolation lets the
            # Gaussian absorb rapid direction changes naturally.
            use_step = is_long and gap >= STEP_THRESHOLD
            for k in range(fi + 1, fj):
                if use_step:
                    # Step function: target the receiver for the entire gap.
                    # The post-smooth Gaussian (sigma=5, ±15 frames) turns
                    # this into a smooth cinematic pan.
                    interp_x = x1
                    interp_y = y1
                else:
                    # Short/medium gaps: linear interpolation
                    t = (k - fi) / float(gap)
                    interp_x = x0 + t * (x1 - x0)
                    interp_y = y0 + t * (y1 - y0)

                positions[k, 0] = interp_x
                positions[k, 1] = interp_y
                confidence[k] = INTERP_CONF
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
        LEAD_HOLD_MAX = 60  # ~2s at 30fps — generous start-of-clip allowance
        first_yolo = yolo_frames[0]
        if first_yolo > 0:
            yf = yolo_by_frame[first_yolo]
            lead_filled = 0
            for k in range(first_yolo - 1, max(-1, first_yolo - LEAD_HOLD_MAX - 1), -1):
                if k in yolo_by_frame:
                    break
                positions[k, 0] = yf.x
                positions[k, 1] = yf.y
                confidence[k] = max(confidence[k], 0.22)
                source_labels[k] = FUSE_HOLD
                used_mask[k] = True
                lead_filled += 1
            if lead_filled > 0:
                interpolated += lead_filled
                print(
                    f"[FUSION] Backward-hold: filled {lead_filled} leading frames "
                    f"from first YOLO at frame {first_yolo}"
                )

        # Hold at last YOLO position for trailing frames (up to SHORT_INTERP_GAP)
        last_yolo = yolo_frames[-1]
        yl = yolo_by_frame[last_yolo]
        for k in range(last_yolo + 1, min(frame_count, last_yolo + SHORT_INTERP_GAP)):
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
        _src_fps = 30.0  # source clips are always 30fps
        for _lf in long_flight_info:
            _fi, _fj, _x0, _y0, _x1, _y1, _dist, _mode = _lf
            _t0 = _fi / _src_fps
            _t1 = _fj / _src_fps
            print(
                f"[FUSION] Long-flight: frames {_fi}→{_fj} "
                f"(t={_t0:.1f}s→{_t1:.1f}s, {_fj - _fi} frames), "
                f"ball x={_x0:.0f}→{_x1:.0f} ({_dist:.0f}px) [{_mode}]"
            )
        # Log ball_x at 1-second intervals for the first ~5 seconds
        _n_pos = len(positions)
        _step = int(_src_fps)
        _sample_frames = list(range(0, min(_n_pos, _step * 6), _step))
        if _sample_frames:
            _parts = [f"f{_sf}={positions[_sf, 0]:.0f}" for _sf in _sample_frames]
            print(f"[FUSION] Ball x timeline (1s steps): {', '.join(_parts)}")

    return positions, used_mask, confidence, source_labels


__all__ = [
    "BallSample",
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
]


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


"""Unified ball-lock renderer.

This script consolidates the historical ``render_follow_*`` variants into a single
implementation that reproduces the behaviour of the "good tester clip" while
remaining configurable through presets and CLI overrides.

The module is intentionally self-contained so that the calibration and debug
helpers can import and reuse the building blocks (label loading, camera
planning, etc.).  The implementation is optimised for clarity and predictable
Windows behaviour rather than raw performance.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2  # type: ignore
import numpy as np
import yaml


PRESETS_PATH = Path(__file__).resolve().parent / "render_presets.yaml"
DEFAULT_PRESETS = {
    "cinematic": {
        "fps": 30,
        "portrait": "1080x1920",
        "lookahead": 18,
        "smoothing": 0.65,
        "pad": 0.22,
        "speed_limit": 480,
        "zoom_min": 1.0,
        "zoom_max": 2.2,
        "crf": 19,
        "keyint_factor": 4,
    },
    "gentle": {
        "fps": 30,
        "portrait": "1080x1920",
        "lookahead": 12,
        "smoothing": 0.55,
        "pad": 0.20,
        "speed_limit": 360,
        "zoom_min": 1.0,
        "zoom_max": 1.8,
        "crf": 20,
        "keyint_factor": 4,
    },
    "realzoom": {
        "fps": 30,
        "portrait": "1080x1920",
        "lookahead": 10,
        "smoothing": 0.50,
        "pad": 0.18,
        "speed_limit": 520,
        "zoom_min": 1.0,
        "zoom_max": 2.4,
        "crf": 19,
        "keyint_factor": 4,
    },
}


def ensure_presets_file() -> None:
    """Create the presets file with defaults when missing."""

    if PRESETS_PATH.exists():
        return
    PRESETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PRESETS_PATH.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(DEFAULT_PRESETS, handle, sort_keys=True)


def load_presets() -> dict:
    """Load the preset configuration, creating defaults if required."""

    ensure_presets_file()
    with PRESETS_PATH.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def ffprobe_fps(path: Path) -> float:
    """Return the floating-point FPS using ffprobe."""

    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:  # pragma: no cover - execution context dependant
        raise RuntimeError(
            "Failed to read FPS using ffprobe. Ensure ffmpeg is installed and on PATH."
        ) from exc

    value = result.stdout.strip()
    if not value:
        raise RuntimeError("ffprobe did not return a frame rate value.")

    if "/" in value:
        num, den = value.split("/", 1)
        den_value = float(den)
        if den_value == 0:
            return float(num)
        return float(num) / den_value
    return float(value)


def parse_portrait(value: Optional[str]) -> Optional[Tuple[int, int]]:
    """Convert a WxH string into integers."""

    if not value:
        return None
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Invalid portrait specification: {value}")
    width = int(parts[0])
    height = int(parts[1])
    if width <= 0 or height <= 0:
        raise ValueError("Portrait dimensions must be positive integers.")
    return width, height


def find_label_files(stem: str, labels_root: Path) -> List[Path]:
    """Discover YOLO label shards matching ``<stem>_*.txt``."""

    if not labels_root.exists():
        return []
    pattern = f"**/labels/{stem}_*.txt"
    return sorted(labels_root.glob(pattern))


def load_labels(paths: Sequence[Path], frame_width: int, frame_height: int) -> np.ndarray:
    """Load and merge labels from one or more shards.

    Returns a numpy array with columns ``frame_idx``, ``x`` and ``y`` in pixel space.
    Outliers (z-score > 3) are discarded.
    """

    records: List[Tuple[int, float, float]] = []
    for file_path in paths:
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = stripped.split()
                if len(parts) < 3:
                    continue
                try:
                    frame_idx = int(float(parts[0]))
                    x_value = float(parts[1])
                    y_value = float(parts[2])
                except ValueError:
                    continue
                records.append((frame_idx, x_value, y_value))

    if not records:
        return np.empty((0, 3), dtype=np.float32)

    data = np.array(records, dtype=np.float32)
    xs = data[:, 1]
    ys = data[:, 2]
    if np.nanmax(xs) <= 1.5 and np.nanmax(ys) <= 1.5:
        data[:, 1] = xs * float(frame_width)
        data[:, 2] = ys * float(frame_height)

    # Sort and drop duplicates (keeping the highest frame index entry first in chronological order)
    order = np.argsort(data[:, 0])
    data = data[order]
    _, unique_indices = np.unique(data[:, 0], return_index=True)
    data = data[np.sort(unique_indices)]

    mean_x = float(np.mean(data[:, 1]))
    std_x = float(np.std(data[:, 1]))
    mean_y = float(np.mean(data[:, 2]))
    std_y = float(np.std(data[:, 2]))
    keep_mask = np.ones(len(data), dtype=bool)
    if std_x > 1e-6:
        keep_mask &= np.abs((data[:, 1] - mean_x) / std_x) <= 3.0
    if std_y > 1e-6:
        keep_mask &= np.abs((data[:, 2] - mean_y) / std_y) <= 3.0
    filtered = data[keep_mask]
    return filtered


def _interpolate(values: np.ndarray, times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    """Helper performing linear interpolation with edge clamping."""

    if len(values) == 0:
        return np.full_like(target_times, np.nan, dtype=np.float32)
    left = values[0]
    right = values[-1]
    return np.interp(target_times, times, values, left=left, right=right)


def interp_labels_to_fps(
    labels: np.ndarray,
    frame_count: int,
    src_fps: float,
    dst_fps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample label positions to the render FPS.

    Returns ``(positions, used_mask)`` where ``positions`` is ``Nx2`` with NaNs for
    missing data and ``used_mask`` marks frames that were close to a labelled
    position.
    """

    if len(labels) == 0 or frame_count <= 0:
        positions = np.full((frame_count, 2), np.nan, dtype=np.float32)
        used = np.zeros(frame_count, dtype=bool)
        return positions, used

    times = labels[:, 0] / float(src_fps)
    xs = labels[:, 1]
    ys = labels[:, 2]

    duration = float(frame_count) / float(src_fps)
    target_count = max(1, int(round(duration * float(dst_fps))))
    target_times = np.arange(target_count, dtype=np.float32) / float(dst_fps)

    interp_x = _interpolate(xs, times, target_times)
    interp_y = _interpolate(ys, times, target_times)

    positions = np.stack([interp_x, interp_y], axis=1).astype(np.float32)

    used = np.zeros(target_count, dtype=bool)
    threshold = 1.5 / float(dst_fps)
    for idx, t_value in enumerate(target_times):
        diff = np.abs(times - t_value)
        if diff.size and float(np.min(diff)) <= threshold:
            used[idx] = True

    return positions, used


@dataclass
class CamState:
    frame: int
    cx: float
    cy: float
    zoom: float
    crop_w: float
    crop_h: float
    used_label: bool
    clamp_flags: List[str]


class CameraPlanner:
    """Planner that tracks the ball and produces smoothed camera states."""

    def __init__(
        self,
        width: int,
        height: int,
        fps: float,
        lookahead: int,
        smoothing: float,
        pad: float,
        speed_limit: float,
        zoom_min: float,
        zoom_max: float,
        portrait: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.width = float(width)
        self.height = float(height)
        self.fps = float(fps)
        self.lookahead = max(0, int(lookahead))
        self.smoothing = float(np.clip(smoothing, 0.0, 1.0))
        self.pad = max(0.0, min(0.45, float(pad)))
        self.speed_limit = max(0.0, float(speed_limit))
        self.zoom_min = max(0.1, float(zoom_min))
        self.zoom_max = max(self.zoom_min, float(zoom_max))
        self.portrait = portrait

        base_side = min(self.width, self.height)
        base_side = max(1.0, base_side)
        desired_side = max(base_side * (1.0 - 2.0 * self.pad), base_side * 0.35)
        desired_zoom = base_side / desired_side
        self.base_zoom = float(np.clip(desired_zoom, self.zoom_min, self.zoom_max))

    def plan(self, positions: np.ndarray, used_mask: np.ndarray) -> List[CamState]:
        frame_count = len(positions)
        states: List[CamState] = []
        prev_cx = self.width / 2.0
        prev_cy = self.height / 2.0
        prev_zoom = self.base_zoom
        fallback_center = np.array([prev_cx, self.height * 0.45], dtype=np.float32)
        fallback_alpha = 0.05
        per_frame_speed = self.speed_limit / max(self.fps, 0.001)

        aspect_target = None
        if self.portrait:
            aspect_target = float(self.portrait[0]) / float(self.portrait[1])

        for frame_idx in range(frame_count):
            pos = positions[frame_idx]
            has_position = bool(used_mask[frame_idx]) and not np.isnan(pos).any()

            if has_position:
                target = pos.copy()
            else:
                fallback_target = np.array([self.width / 2.0, self.height * 0.40], dtype=np.float32)
                fallback_center = (
                    fallback_alpha * fallback_target + (1.0 - fallback_alpha) * fallback_center
                )
                target = fallback_center

            # Lookahead bias.
            if self.lookahead > 0 and frame_idx < frame_count - 1:
                max_future = min(frame_count - 1, frame_idx + self.lookahead)
                future_positions = positions[frame_idx + 1 : max_future + 1]
                future_mask = used_mask[frame_idx + 1 : max_future + 1]
                valid_future = future_positions[future_mask]
                if valid_future.size:
                    future_mean = valid_future.mean(axis=0)
                    target = 0.65 * target + 0.35 * future_mean

            target_zoom = self.base_zoom

            smoothed_cx = self.smoothing * target[0] + (1.0 - self.smoothing) * prev_cx
            smoothed_cy = self.smoothing * target[1] + (1.0 - self.smoothing) * prev_cy
            smoothed_zoom = self.smoothing * target_zoom + (1.0 - self.smoothing) * prev_zoom

            clamp_flags: List[str] = []

            # Limit the camera speed.
            dx = smoothed_cx - prev_cx
            dy = smoothed_cy - prev_cy
            distance = math.hypot(dx, dy)
            if per_frame_speed > 0 and distance > per_frame_speed:
                ratio = per_frame_speed / distance
                smoothed_cx = prev_cx + dx * ratio
                smoothed_cy = prev_cy + dy * ratio
                clamp_flags.append("speed")

            smoothed_zoom = float(np.clip(smoothed_zoom, self.zoom_min, self.zoom_max))

            crop_w = self.width / smoothed_zoom
            crop_h = self.height / smoothed_zoom

            if aspect_target:
                current_aspect = crop_w / crop_h
                if current_aspect > aspect_target:
                    crop_w = crop_h * aspect_target
                else:
                    crop_h = crop_w / aspect_target

            # Bias the framing so the ball sits lower in portrait compositions.
            if aspect_target:
                desired_center_y = smoothed_cy + 0.10 * crop_h
                smoothed_cy = desired_center_y

            half_w = crop_w / 2.0
            half_h = crop_h / 2.0

            min_cx = half_w
            max_cx = self.width - half_w
            min_cy = half_h
            max_cy = self.height - half_h

            if smoothed_cx < min_cx:
                smoothed_cx = min_cx
                clamp_flags.append("bounds")
            if smoothed_cx > max_cx:
                smoothed_cx = max_cx
                clamp_flags.append("bounds")
            if smoothed_cy < min_cy:
                smoothed_cy = min_cy
                clamp_flags.append("bounds")
            if smoothed_cy > max_cy:
                smoothed_cy = max_cy
                clamp_flags.append("bounds")

            prev_cx = smoothed_cx
            prev_cy = smoothed_cy
            prev_zoom = smoothed_zoom

            states.append(
                CamState(
                    frame=frame_idx,
                    cx=smoothed_cx,
                    cy=smoothed_cy,
                    zoom=smoothed_zoom,
                    crop_w=crop_w,
                    crop_h=crop_h,
                    used_label=bool(has_position),
                    clamp_flags=clamp_flags,
                )
            )

        return states


def _load_overlay(path: Optional[Path], output_size: Tuple[int, int]) -> Optional[np.ndarray]:
    if not path:
        return None
    if not path.exists():
        logging.warning("Brand overlay %s not found; skipping.", path)
        return None
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        logging.warning("Failed to read brand overlay at %s; skipping.", path)
        return None
    resized = cv2.resize(image, output_size, interpolation=cv2.INTER_LINEAR)
    return resized


def _apply_overlay(frame: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    if overlay.shape[2] < 4:
        return cv2.addWeighted(frame, 0.7, overlay[:, :, :3], 0.3, 0.0)
    alpha = overlay[:, :, 3:] / 255.0
    base = frame.astype(np.float32)
    overlay_rgb = overlay[:, :, :3].astype(np.float32)
    blended = overlay_rgb * alpha + base * (1.0 - alpha)
    return blended.astype(np.uint8)


class Renderer:
    def __init__(
        self,
        input_path: Path,
        output_path: Path,
        temp_dir: Path,
        fps_in: float,
        fps_out: float,
        flip180: bool,
        portrait: Optional[Tuple[int, int]],
        brand_overlay: Optional[Path],
        endcard: Optional[Path],
        telemetry_path: Optional[Path],
    ) -> None:
        self.input_path = input_path
        self.output_path = output_path
        self.temp_dir = temp_dir
        self.fps_in = fps_in
        self.fps_out = fps_out
        self.flip180 = flip180
        self.portrait = portrait
        self.brand_overlay_path = brand_overlay
        self.endcard_path = endcard
        self.telemetry_path = telemetry_path

    def _read_frames(self) -> List[np.ndarray]:
        capture = cv2.VideoCapture(str(self.input_path))
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open input video: {self.input_path}")
        frames: List[np.ndarray] = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if self.flip180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            frames.append(frame)
        capture.release()
        if not frames:
            raise RuntimeError("No frames decoded from the input video.")
        return frames

    def _sample_frame(self, frames: Sequence[np.ndarray], index: int) -> np.ndarray:
        time_sec = float(index) / float(self.fps_out)
        source_index = int(round(time_sec * float(self.fps_in)))
        source_index = max(0, min(source_index, len(frames) - 1))
        return frames[source_index]

    def _compose_frame(
        self,
        frame: np.ndarray,
        state: CamState,
        output_size: Tuple[int, int],
        overlay_image: Optional[np.ndarray],
    ) -> np.ndarray:
        height, width = frame.shape[:2]
        crop_w = min(state.crop_w, float(width))
        crop_h = min(state.crop_h, float(height))
        x1 = int(round(state.cx - crop_w / 2.0))
        y1 = int(round(state.cy - crop_h / 2.0))
        x1 = max(0, min(x1, width - int(round(crop_w))))
        y1 = max(0, min(y1, height - int(round(crop_h))))
        x2 = int(round(x1 + crop_w))
        y2 = int(round(y1 + crop_h))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))

        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            cropped = frame

        resized = cv2.resize(cropped, output_size, interpolation=cv2.INTER_CUBIC)

        if overlay_image is not None:
            resized = _apply_overlay(resized, overlay_image)

        return resized

    def _append_endcard(self, output_size: Tuple[int, int]) -> List[np.ndarray]:
        if not self.endcard_path:
            return []
        if not self.endcard_path.exists():
            logging.warning("Endcard %s not found; skipping.", self.endcard_path)
            return []
        image = cv2.imread(str(self.endcard_path))
        if image is None:
            logging.warning("Failed to read endcard at %s; skipping.", self.endcard_path)
            return []
        resized = cv2.resize(image, output_size, interpolation=cv2.INTER_LINEAR)
        frame_count = int(round(self.fps_out * 2.0))
        return [resized for _ in range(frame_count)]

    def write_frames(self, states: Sequence[CamState]) -> None:
        frames = self._read_frames()
        height, width = frames[0].shape[:2]
        if self.portrait:
            output_size = self.portrait
        else:
            output_size = (width, height)

        overlay_image = _load_overlay(self.brand_overlay_path, output_size)
        telemetry_file = None
        if self.telemetry_path:
            self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
            telemetry_file = self.telemetry_path.open("w", encoding="utf-8")

        for state in states:
            frame = self._sample_frame(frames, state.frame)
            composed = self._compose_frame(frame, state, output_size, overlay_image)
            out_path = self.temp_dir / f"f_{state.frame:06d}.jpg"
            success = cv2.imwrite(str(out_path), composed, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if not success:
                raise RuntimeError(f"Failed to write frame to {out_path}")
            if telemetry_file:
                record = {
                    "t": float(state.frame) / float(self.fps_out),
                    "cx": state.cx,
                    "cy": state.cy,
                    "zoom": state.zoom,
                    "crop_w": state.crop_w,
                    "crop_h": state.crop_h,
                    "used_label": state.used_label,
                    "clamp_flags": state.clamp_flags,
                }
                telemetry_file.write(json.dumps(record) + "\n")

        if telemetry_file:
            telemetry_file.close()

        endcard_frames = self._append_endcard(output_size)
        if endcard_frames:
            start_index = len(states)
            for offset, endcard_frame in enumerate(endcard_frames):
                out_path = self.temp_dir / f"f_{start_index + offset:06d}.jpg"
                cv2.imwrite(str(out_path), endcard_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    def ffmpeg_stitch(
        self,
        crf: int,
        keyint: int,
        log_path: Optional[Path] = None,
    ) -> None:
        pattern = str(self.temp_dir / "f_%06d.jpg")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(self.fps_out),
            "-i",
            pattern,
            "-i",
            str(self.input_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-profile:v",
            "high",
            "-level",
            "4.0",
            "-x264-params",
            f"keyint={keyint}:min-keyint={keyint}:scenecut=0",
            "-movflags",
            "+faststart",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            str(self.output_path),
        ]

        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError("ffmpeg failed during stitching.") from exc

        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(" ".join(command) + "\n")


def _prepare_temp_dir(temp_dir: Path, clean: bool) -> None:
    if clean and temp_dir.exists():
        shutil.rmtree(temp_dir)
    if not temp_dir.exists():
        temp_dir.mkdir(parents=True, exist_ok=True)
        return
    for file in temp_dir.glob("*.jpg"):
        try:
            file.unlink()
        except OSError:
            logging.warning("Failed to remove temp frame %s", file)


def _default_output_path(input_path: Path, preset: str) -> Path:
    suffix = f".__{preset.upper()}.mp4"
    return input_path.with_name(input_path.stem + suffix)


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    input_path = Path(args.in_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    presets = load_presets()
    preset_key = (args.preset or "cinematic").lower()
    if preset_key not in presets:
        raise ValueError(f"Preset '{preset_key}' not found in {PRESETS_PATH}")

    preset_config = presets[preset_key]

    fps_in = ffprobe_fps(input_path)
    fps_out = float(args.fps) if args.fps else float(preset_config.get("fps", fps_in))
    if fps_out <= 0:
        fps_out = fps_in if fps_in > 0 else 30.0

    portrait_str = args.portrait or preset_config.get("portrait")
    portrait = parse_portrait(portrait_str) if portrait_str else None

    lookahead = args.lookahead if args.lookahead is not None else preset_config.get("lookahead", 18)
    smoothing = args.smoothing if args.smoothing is not None else preset_config.get("smoothing", 0.65)
    pad = args.pad if args.pad is not None else preset_config.get("pad", 0.22)
    speed_limit = args.speed_limit if args.speed_limit is not None else preset_config.get("speed_limit", 480)
    zoom_min = args.zoom_min if args.zoom_min is not None else preset_config.get("zoom_min", 1.0)
    zoom_max = args.zoom_max if args.zoom_max is not None else preset_config.get("zoom_max", 2.2)
    crf = int(args.crf) if args.crf is not None else int(preset_config.get("crf", 19))
    keyint_factor = int(args.keyint_factor) if args.keyint_factor is not None else int(preset_config.get("keyint_factor", 4))

    output_path = Path(args.out) if args.out else _default_output_path(input_path, preset_key)
    output_path = output_path.expanduser().resolve()

    labels_root = Path(args.labels_root or "out/yolo").expanduser()
    label_files = find_label_files(input_path.stem, labels_root)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError("Unable to open input video for metadata extraction.")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()

    labels = load_labels(label_files, width, height)
    positions, used_mask = interp_labels_to_fps(labels, frame_count, fps_in, fps_out)

    planner = CameraPlanner(
        width=width,
        height=height,
        fps=fps_out,
        lookahead=lookahead,
        smoothing=smoothing,
        pad=pad,
        speed_limit=speed_limit,
        zoom_min=zoom_min,
        zoom_max=zoom_max,
        portrait=portrait,
    )
    states = planner.plan(positions, used_mask)

    temp_root = Path("out/autoframe_work")
    temp_dir = temp_root / preset_key / input_path.stem
    _prepare_temp_dir(temp_dir, args.clean_temp)

    brand_overlay_path = Path(args.brand_overlay).expanduser() if args.brand_overlay else None
    endcard_path = Path(args.endcard).expanduser() if args.endcard else None
    telemetry_path = Path(args.telemetry).expanduser() if args.telemetry else None

    renderer = Renderer(
        input_path=input_path,
        output_path=output_path,
        temp_dir=temp_dir,
        fps_in=fps_in,
        fps_out=fps_out,
        flip180=args.flip180,
        portrait=portrait,
        brand_overlay=brand_overlay_path,
        endcard=endcard_path,
        telemetry_path=telemetry_path,
    )

    renderer.write_frames(states)

    keyint = max(1, int(round(keyint_factor * fps_out)))
    log_path = Path(args.log).expanduser() if args.log else None
    renderer.ffmpeg_stitch(crf=crf, keyint=keyint, log_path=log_path)

    if log_path:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "input": str(input_path),
                        "output": str(output_path),
                        "fps_in": fps_in,
                        "fps_out": fps_out,
                        "labels_found": len(labels),
                        "preset": preset_key,
                    }
                )
                + "\n"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified cinematic ball-follow renderer")
    parser.add_argument("--in", dest="in_path", required=True, help="Input MP4 path")
    parser.add_argument("--src", dest="src", help="Legacy compatibility input path (ignored)")
    parser.add_argument("--out", dest="out", help="Output MP4 path")
    parser.add_argument("--preset", dest="preset", choices=["cinematic", "gentle", "realzoom"], default="cinematic")
    parser.add_argument("--portrait", dest="portrait", help="Portrait canvas WxH")
    parser.add_argument("--fps", dest="fps", type=float, help="Output FPS")
    parser.add_argument("--flip180", dest="flip180", action="store_true", help="Rotate frames by 180 degrees before processing")
    parser.add_argument("--labels-root", dest="labels_root", help="Root directory containing YOLO label shards")
    parser.add_argument("--clean-temp", dest="clean_temp", action="store_true", help="Remove temporary frame folder before rendering")
    parser.add_argument("--lookahead", dest="lookahead", type=int, help="Frames of lookahead for planning")
    parser.add_argument("--smoothing", dest="smoothing", type=float, help="EMA smoothing factor")
    parser.add_argument("--pad", dest="pad", type=float, help="Edge padding ratio used to derive zoom")
    parser.add_argument("--speed-limit", dest="speed_limit", type=float, help="Maximum pan speed in px/sec")
    parser.add_argument("--zoom-min", dest="zoom_min", type=float, help="Minimum zoom multiplier")
    parser.add_argument("--zoom-max", dest="zoom_max", type=float, help="Maximum zoom multiplier")
    parser.add_argument("--telemetry", dest="telemetry", help="Output JSONL telemetry file")
    parser.add_argument("--brand-overlay", dest="brand_overlay", help="PNG overlay composited on every frame")
    parser.add_argument("--endcard", dest="endcard", help="Optional endcard image displayed for ~2 seconds")
    parser.add_argument("--log", dest="log", help="Optional render log path")
    parser.add_argument("--crf", dest="crf", type=int, help="Override CRF value")
    parser.add_argument("--keyint-factor", dest="keyint_factor", type=int, help="Override keyint factor")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    try:
        main()
    except Exception as exc:  # pylint: disable=broad-except
        logging.error(str(exc))
        sys.exit(1)

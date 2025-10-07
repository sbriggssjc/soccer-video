#!/usr/bin/env python3
"""Unified render-follow / auto-zoom pipeline.

This CLI replaces the historical ``render_follow_*`` variants with a single
entrypoint that supports presets, portrait framing, branding, and YOLO ball
label integration.  The pipeline stays streaming-friendly (no full video loads)
and only depends on Python, NumPy, OpenCV, and ffmpeg being present.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2  # type: ignore
import numpy as np

# ---------------------------------------------------------------------------
# Preset definitions
# ---------------------------------------------------------------------------

PRESETS: Dict[str, Dict[str, float]] = {
    "cinematic": {
        "lookahead": 18.0,
        "smoothing": 0.65,
        "pad": 0.22,
        "speed_limit": 0.035,  # fraction of frame width per frame
        "zoom_speed": 0.08,     # fraction of frame width per frame
    },
    "gentle": {
        "lookahead": 12.0,
        "smoothing": 0.55,
        "pad": 0.28,
        "speed_limit": 0.030,
        "zoom_speed": 0.06,
    },
    "realzoom": {
        "lookahead": 16.0,
        "smoothing": 0.60,
        "pad": 0.18,
        "speed_limit": 0.045,
        "zoom_speed": 0.10,
    },
}

DEFAULT_PRESET = "cinematic"
DEFAULT_FPS = 30.0
DEFAULT_LABELS_ROOT = Path("out") / "yolo"
DEFAULT_LOG_ROOT = Path("out") / "render_logs"
DEFAULT_TEMP_ROOT = Path("out") / "autoframe_work"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_portrait(value: Optional[str]) -> Optional[Tuple[int, int]]:
    if not value:
        return None
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--portrait must be WIDTHxHEIGHT")
    try:
        width = int(parts[0])
        height = int(parts[1])
    except ValueError as exc:  # pragma: no cover - argparse handles message
        raise argparse.ArgumentTypeError("--portrait must be WIDTHxHEIGHT") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("--portrait dimensions must be >0")
    return width, height


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified render-follow / auto-zoom pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--in", "--src", dest="input", required=True,
                        help="Input MP4 clip")
    parser.add_argument("--out", dest="output", help="Output MP4 path")
    parser.add_argument("--preset", choices=sorted(PRESETS.keys()),
                        default=DEFAULT_PRESET,
                        help="Rendering preset")
    parser.add_argument("--portrait", type=parse_portrait,
                        help="Portrait canvas WxH")
    parser.add_argument("--fps", type=float,
                        help="Output frames per second (defaults to input fps)")
    parser.add_argument("--flip180", action="store_true",
                        help="Rotate frames 180 degrees before processing")
    parser.add_argument("--labels-root", default=str(DEFAULT_LABELS_ROOT),
                        help="Root directory for YOLO ball labels")
    parser.add_argument("--clean-temp", action="store_true",
                        help="Clean temp work directory before rendering")
    parser.add_argument("--brand-overlay", dest="brand_overlay",
                        help="PNG overlay composited on top of frames")
    parser.add_argument("--endcard", dest="endcard",
                        help="PNG endcard appended (~1s) to final video")
    parser.add_argument("--lookahead", type=float,
                        help="Override lookahead window (frames)")
    parser.add_argument("--smoothing", type=float,
                        help="Override smoothing factor (0-1)")
    parser.add_argument("--pad", type=float,
                        help="Override fractional padding for camera window")
    parser.add_argument("--log", dest="log_path",
                        help="Optional path for render log")
    parser.add_argument("--jsonl-telemetry", dest="telemetry",
                        help="Optional JSONL debug output for per-frame camera")
    parser.add_argument("--quiet", action="store_true",
                        help="Reduce console output")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# ffprobe helpers
# ---------------------------------------------------------------------------


def run_subprocess(cmd: Sequence[str], capture_output: bool = False) -> subprocess.CompletedProcess:
    if capture_output:
        return subprocess.run(cmd, check=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True)
    return subprocess.run(cmd, check=True)


def ffprobe_stream_info(path: Path) -> Dict[str, Optional[float]]:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames",
        "-of", "json",
        str(path),
    ]
    result = run_subprocess(cmd, capture_output=True)
    data = json.loads(result.stdout or "{}")
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found in {path}")
    stream = streams[0]

    def parse_rate(rate: str) -> Optional[float]:
        if not rate or rate == "0/0":
            return None
        if isinstance(rate, (int, float)):
            return float(rate)
        if "/" in rate:
            num, den = rate.split("/", 1)
            try:
                num_f = float(num)
                den_f = float(den)
                if den_f:
                    return num_f / den_f
            except ValueError:
                return None
        else:
            try:
                return float(rate)
            except ValueError:
                return None
        return None

    width = stream.get("width")
    height = stream.get("height")
    avg_rate = parse_rate(stream.get("avg_frame_rate"))
    real_rate = parse_rate(stream.get("r_frame_rate"))
    fps = avg_rate or real_rate or DEFAULT_FPS
    nb_frames = stream.get("nb_frames")
    try:
        total_frames = int(nb_frames) if nb_frames is not None else None
    except (TypeError, ValueError):
        total_frames = None
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frames": total_frames,
    }


# ---------------------------------------------------------------------------
# Label handling
# ---------------------------------------------------------------------------


def _split_tokens(line: str) -> List[str]:
    line = line.strip()
    if not line:
        return []
    return [part for part in line.replace(",", " ").split() if part]


@dataclass
class LabelEntry:
    frame: int
    x: float
    y: float
    score: Optional[float] = None


class LabelStore:
    def __init__(self, labels_root: Path):
        self.labels_root = labels_root

    def find_label_files(self, stem: str) -> List[Path]:
        if not self.labels_root.exists():
            return []
        pattern = f"{stem}_*.txt"
        matches: List[Path] = []
        for labels_dir in self.labels_root.glob("**/labels"):
            if labels_dir.is_dir():
                matches.extend(sorted(labels_dir.glob(pattern)))
        return sorted(matches)

    def load_labels(self, stem: str) -> List[LabelEntry]:
        label_files = self.find_label_files(stem)
        entries: List[LabelEntry] = []
        for file_path in label_files:
            try:
                with file_path.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        tokens = _split_tokens(line)
                        if len(tokens) < 3:
                            continue
                        try:
                            frame_idx = int(float(tokens[0]))
                            x = float(tokens[1])
                            y = float(tokens[2])
                            score = float(tokens[3]) if len(tokens) > 3 else None
                        except ValueError:
                            continue
                        entries.append(LabelEntry(frame=frame_idx, x=x, y=y, score=score))
            except FileNotFoundError:
                continue
        entries.sort(key=lambda e: (e.frame, -(e.score or 0.0)))
        return entries

    def build_sequence(self, stem: str, frame_count: int) -> List[Optional[Tuple[float, float]]]:
        entries = self.load_labels(stem)
        if frame_count <= 0:
            return []
        sequence: List[Optional[Tuple[float, float]]] = [None] * frame_count
        for entry in entries:
            for candidate in (entry.frame, entry.frame - 1):
                if 0 <= candidate < frame_count:
                    sequence[candidate] = (entry.x, entry.y)
                    break
        if not any(sequence):
            return [None] * frame_count
        # forward / backward fill sparse labels for stability
        last_known: Optional[Tuple[float, float]] = None
        for idx, value in enumerate(sequence):
            if value is None and last_known is not None:
                sequence[idx] = last_known
            elif value is not None:
                last_known = value
        last_known = None
        for idx in range(frame_count - 1, -1, -1):
            value = sequence[idx]
            if value is None and last_known is not None:
                sequence[idx] = last_known
            elif value is not None:
                last_known = value
        return sequence


# ---------------------------------------------------------------------------
# Camera planning
# ---------------------------------------------------------------------------


@dataclass
class CameraState:
    x: float
    y: float
    width: float
    height: float
    zoom: float
    target: Tuple[float, float]


class CameraPlanner:
    def __init__(
        self,
        frame_size: Tuple[int, int],
        canvas_size: Tuple[int, int],
        fps: float,
        lookahead: float,
        smoothing: float,
        pad: float,
        speed_limit: float,
        zoom_speed: float,
        focus_x: float,
        focus_y: float,
    ) -> None:
        self.frame_w, self.frame_h = frame_size
        self.canvas_w, self.canvas_h = canvas_size
        self.aspect = self.canvas_w / self.canvas_h
        self.fps = max(fps, 1.0)
        self.lookahead = max(0, int(round(lookahead)))
        self.smoothing = float(np.clip(smoothing, 0.0, 0.99))
        self.pad = max(0.0, pad)
        self.speed_limit = max(0.0, speed_limit)
        self.zoom_speed = max(0.001, zoom_speed)
        self.focus_x = focus_x
        self.focus_y = focus_y
        self.min_crop_width = float(self.frame_w) / 3.2
        self.max_crop_width = float(self.frame_w)

    def _fallback_position(self) -> Tuple[float, float]:
        # Keep the ball roughly 60% up from the bottom in the portrait output.
        return (self.frame_w / 2.0, self.frame_h * 0.4)

    def _window_stats(self, positions: np.ndarray, idx: int) -> Tuple[Tuple[float, float], float, float]:
        end = min(len(positions), idx + self.lookahead + 1)
        window = positions[idx:end]
        valid_mask = ~np.isnan(window[:, 0])
        valid = window[valid_mask]
        if valid.size == 0:
            target = self._fallback_position()
            span_x = self.min_crop_width * 0.6
            span_y = (self.min_crop_width / self.aspect) * 0.6
        else:
            xs = valid[:, 0]
            ys = valid[:, 1]
            target = (float(xs[-1]), float(ys[-1]))
            span_x = float(xs.max() - xs.min())
            span_y = float(ys.max() - ys.min())
            span_x = max(span_x, 12.0)
            span_y = max(span_y, 8.0)
        span_x *= (1.0 + self.pad)
        span_y *= (1.0 + self.pad)
        return target, span_x, span_y

    def plan(self, positions: Sequence[Optional[Tuple[float, float]]]) -> List[CameraState]:
        frame_count = len(positions)
        if frame_count == 0:
            return []
        arr = np.full((frame_count, 2), np.nan, dtype=float)
        for i, pos in enumerate(positions):
            if pos is not None:
                arr[i, 0] = float(pos[0])
                arr[i, 1] = float(pos[1])
        smoothing_alpha = max(0.01, 1.0 - self.smoothing)
        max_dx = self.speed_limit * self.frame_w
        max_dy = self.speed_limit * self.frame_h
        max_zoom_delta = self.zoom_speed * self.frame_w

        states: List[CameraState] = []
        prev_ball_x, prev_ball_y = self._fallback_position()
        prev_crop_w = min(self.max_crop_width, max(self.min_crop_width, self.frame_w * 0.8))
        prev_crop_h = prev_crop_w / self.aspect
        prev_cam_x = max(0.0, min(prev_ball_x - self.focus_x * prev_crop_w, self.frame_w - prev_crop_w))
        prev_cam_y = max(0.0, min(prev_ball_y - self.focus_y * prev_crop_h, self.frame_h - prev_crop_h))

        for idx in range(frame_count):
            target, span_x, span_y = self._window_stats(arr, idx)
            ball_x = prev_ball_x + smoothing_alpha * (target[0] - prev_ball_x)
            ball_y = prev_ball_y + smoothing_alpha * (target[1] - prev_ball_y)

            desired_crop_w = max(self.min_crop_width, min(self.max_crop_width, max(span_x, self.aspect * span_y)))
            crop_w = prev_crop_w + np.clip(desired_crop_w - prev_crop_w, -max_zoom_delta, max_zoom_delta)
            crop_w = float(np.clip(crop_w, self.min_crop_width, self.max_crop_width))
            crop_h = float(crop_w / self.aspect)

            cam_x = ball_x - self.focus_x * crop_w
            cam_y = ball_y - self.focus_y * crop_h

            cam_x = prev_cam_x + np.clip(cam_x - prev_cam_x, -max_dx, max_dx)
            cam_y = prev_cam_y + np.clip(cam_y - prev_cam_y, -max_dy, max_dy)

            cam_x = float(np.clip(cam_x, 0.0, max(0.0, self.frame_w - crop_w)))
            cam_y = float(np.clip(cam_y, 0.0, max(0.0, self.frame_h - crop_h)))
            zoom = float(self.frame_w / crop_w) if crop_w else 1.0

            states.append(CameraState(x=cam_x, y=cam_y, width=crop_w, height=crop_h,
                                      zoom=zoom, target=(ball_x, ball_y)))

            prev_ball_x, prev_ball_y = ball_x, ball_y
            prev_crop_w, prev_crop_h = crop_w, crop_h
            prev_cam_x, prev_cam_y = cam_x, cam_y

        return states


# ---------------------------------------------------------------------------
# Rendering pipeline
# ---------------------------------------------------------------------------


class Renderer:
    def __init__(self, args: argparse.Namespace, video_info: Dict[str, Optional[float]]) -> None:
        self.args = args
        self.input_path = Path(args.input).resolve()
        self.video_info = video_info
        self.preset = PRESETS[args.preset]
        self.labels_root = Path(args.labels_root).resolve()
        self.temp_root = DEFAULT_TEMP_ROOT / args.preset / self.input_path.stem
        self.frames_dir = self.temp_root / "frames"
        self.frames_pattern = self.frames_dir / "f_%06d.jpg"
        self.clean_temp = bool(args.clean_temp)
        self.overlay_path = Path(args.brand_overlay).resolve() if args.brand_overlay else None
        self.endcard_path = Path(args.endcard).resolve() if args.endcard else None
        self.telemetry_path = Path(args.telemetry).resolve() if args.telemetry else None
        self.overlay_image: Optional[np.ndarray] = None
        self.overlay_alpha: Optional[np.ndarray] = None

        if args.output:
            self.output_path = Path(args.output).resolve()
        else:
            suffix = f".__{args.preset.upper()}.mp4"
            self.output_path = self.input_path.with_name(self.input_path.name + suffix)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        if args.log_path:
            self.log_path = Path(args.log_path).resolve()
        else:
            DEFAULT_LOG_ROOT.mkdir(parents=True, exist_ok=True)
            log_name = f"{self.input_path.stem}__{args.preset}.log"
            self.log_path = (DEFAULT_LOG_ROOT / log_name).resolve()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        if args.portrait:
            self.canvas_size = args.portrait
        else:
            width = int(video_info.get("width") or 0)
            height = int(video_info.get("height") or 0)
            if width <= 0 or height <= 0:
                raise RuntimeError("Unable to determine input resolution")
            self.canvas_size = (width, height)

    # ------------------------------
    def extract_frames(self, fps: float, quiet: bool = False) -> None:
        if self.clean_temp and self.temp_root.exists():
            shutil.rmtree(self.temp_root)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        cached = sorted(self.frames_dir.glob("f_*.jpg"))
        if cached and not self.clean_temp:
            if not quiet:
                print(f"Reusing {len(cached)} cached frames in {self.frames_dir}")
            return
        if not quiet:
            print("Extracting source frames...")
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(self.input_path),
            "-vf", f"fps={fps:.6f}",
            "-start_number", "0",
            str(self.frames_pattern),
        ]
        run_subprocess(cmd)

    def prepare_overlay(self) -> None:
        if not self.overlay_path:
            return
        if not self.overlay_path.exists():
            print(f"Warning: overlay not found: {self.overlay_path}")
            self.overlay_path = None
            return
        overlay = cv2.imread(str(self.overlay_path), cv2.IMREAD_UNCHANGED)
        if overlay is None:
            print(f"Warning: failed to load overlay {self.overlay_path}")
            self.overlay_path = None
            return
        overlay = cv2.resize(overlay, (self.canvas_size[0], self.canvas_size[1]), interpolation=cv2.INTER_AREA)
        if overlay.shape[2] == 4:
            alpha = overlay[:, :, 3].astype(np.float32) / 255.0
            self.overlay_alpha = alpha[..., None]
            self.overlay_image = overlay[:, :, :3].astype(np.float32)
        else:
            self.overlay_alpha = np.ones((overlay.shape[0], overlay.shape[1], 1), dtype=np.float32)
            self.overlay_image = overlay.astype(np.float32)

    def apply_overlay(self, frame: np.ndarray) -> np.ndarray:
        if self.overlay_image is None or self.overlay_alpha is None:
            return frame
        overlay = self.overlay_image
        alpha = self.overlay_alpha
        if overlay.shape[0] != frame.shape[0] or overlay.shape[1] != frame.shape[1]:
            overlay = cv2.resize(overlay, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_AREA)
            alpha = cv2.resize(alpha, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_AREA)[..., None]
        blended = overlay * alpha + frame.astype(np.float32) * (1.0 - alpha)
        return np.clip(blended, 0, 255).astype(np.uint8)

    def compose_frames(self, states: Sequence[CameraState], quiet: bool = False) -> int:
        frame_files = sorted(self.frames_dir.glob("f_*.jpg"))
        if len(frame_files) != len(states):
            raise RuntimeError("Frame count mismatch between extracted frames and camera plan")
        count = 0
        telemetry_fh = None
        if self.telemetry_path:
            self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
            telemetry_fh = self.telemetry_path.open("w", encoding="utf-8")
        try:
            for idx, frame_file in enumerate(frame_files):
                bgr = cv2.imread(str(frame_file), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise RuntimeError(f"Failed to load frame {frame_file}")
                if self.args.flip180:
                    bgr = cv2.rotate(bgr, cv2.ROTATE_180)
                state = states[idx]
                xi = int(round(state.x))
                yi = int(round(state.y))
                wi = int(round(state.width))
                hi = int(round(state.height))
                xi = max(0, min(xi, bgr.shape[1] - wi))
                yi = max(0, min(yi, bgr.shape[0] - hi))
                crop = bgr[yi:yi + hi, xi:xi + wi]
                if crop.shape[0] != hi or crop.shape[1] != wi:
                    pad_bottom = max(0, hi - crop.shape[0])
                    pad_right = max(0, wi - crop.shape[1])
                    crop = cv2.copyMakeBorder(crop, 0, pad_bottom, 0, pad_right, cv2.BORDER_REPLICATE)
                resized = cv2.resize(crop, self.canvas_size, interpolation=cv2.INTER_LANCZOS4)
                composited = self.apply_overlay(resized)
                cv2.imwrite(str(frame_file), composited, [int(cv2.IMWRITE_JPEG_QUALITY), 97])
                count += 1
                if telemetry_fh:
                    record = {
                        "frame": idx,
                        "camera": {
                            "x": state.x,
                            "y": state.y,
                            "w": state.width,
                            "h": state.height,
                            "zoom": state.zoom,
                        },
                        "target": {
                            "x": state.target[0],
                            "y": state.target[1],
                        },
                    }
                    telemetry_fh.write(json.dumps(record) + "\n")
            if not quiet:
                print(f"Rendered {count} frames into {self.frames_dir}")
        finally:
            if telemetry_fh:
                telemetry_fh.close()
        return count

    def ffmpeg_stitch(self, fps: float, quiet: bool = False) -> None:
        tmp_main = self.temp_root / "__main_video.mp4"
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-framerate", f"{fps:.6f}",
            "-i", str(self.frames_pattern),
            "-i", str(self.input_path),
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "19",
            "-pix_fmt", "yuv420p",
            "-profile:v", "high",
            "-g", str(int(round(fps * 4))),
            "-shortest",
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            str(tmp_main),
        ]
        run_subprocess(cmd)
        if not self.endcard_path or not self.endcard_path.exists():
            shutil.move(str(tmp_main), str(self.output_path))
            if not quiet:
                print(f"Wrote {self.output_path}")
            return

        tmp_endcard = self.temp_root / "__endcard.mp4"
        cmd_endcard = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-loop", "1", "-t", "1.0",
            "-i", str(self.endcard_path),
            "-f", "lavfi", "-t", "1.0", "-i",
            "anullsrc=channel_layout=stereo:sample_rate=48000",
            "-vf", f"scale={self.canvas_size[0]}:{self.canvas_size[1]}",
            "-r", f"{fps:.6f}",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-profile:v", "high",
            "-crf", "19",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            str(tmp_endcard),
        ]
        run_subprocess(cmd_endcard)

        concat_file = self.temp_root / "__concat.ffconcat"
        concat_file.parent.mkdir(parents=True, exist_ok=True)
        with concat_file.open("w", encoding="utf-8") as fh:
            fh.write("ffconcat version 1.0\n")
            fh.write(f"file '{tmp_main.as_posix()}'\n")
            fh.write(f"file '{tmp_endcard.as_posix()}'\n")

        cmd_concat = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_file),
            "-c", "copy",
            str(self.output_path),
        ]
        run_subprocess(cmd_concat)
        if not quiet:
            print(f"Wrote {self.output_path}")

    def render(self, fps: float, states: Sequence[CameraState], quiet: bool = False) -> int:
        self.prepare_overlay()
        frame_count = self.compose_frames(states, quiet=quiet)
        self.ffmpeg_stitch(fps, quiet=quiet)
        return frame_count


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input clip not found: {input_path}")

    video_info = ffprobe_stream_info(input_path)
    fps = float(args.fps or video_info.get("fps") or DEFAULT_FPS)
    if fps <= 1e-3:
        fps = DEFAULT_FPS

    renderer = Renderer(args, video_info)
    renderer.extract_frames(fps=fps, quiet=args.quiet)
    frame_files = sorted(renderer.frames_dir.glob("f_*.jpg"))
    frame_count = len(frame_files)
    if frame_count == 0:
        raise RuntimeError("No frames extracted for processing")

    label_store = LabelStore(Path(args.labels_root).expanduser().resolve())
    try:
        labels_sequence = label_store.build_sequence(renderer.input_path.stem, frame_count)
    except Exception as exc:  # pragma: no cover - defensive
        if not args.quiet:
            print(f"Warning: failed to load labels ({exc})")
        labels_sequence = [None] * frame_count
    if not labels_sequence:
        labels_sequence = [None] * frame_count
    label_files_used = label_store.find_label_files(renderer.input_path.stem)

    preset = PRESETS[args.preset]
    lookahead = args.lookahead if args.lookahead is not None else preset["lookahead"]
    smoothing = args.smoothing if args.smoothing is not None else preset["smoothing"]
    pad = args.pad if args.pad is not None else preset["pad"]

    focus_y = 0.4 if args.portrait else 0.5
    planner = CameraPlanner(
        frame_size=(int(video_info.get("width") or 0), int(video_info.get("height") or 0)),
        canvas_size=renderer.canvas_size,
        fps=fps,
        lookahead=lookahead,
        smoothing=smoothing,
        pad=pad,
        speed_limit=preset["speed_limit"],
        zoom_speed=preset["zoom_speed"],
        focus_x=0.5,
        focus_y=focus_y,
    )
    states = planner.plan(labels_sequence if labels_sequence else [None] * frame_count)

    start_time = time.time()
    rendered_frames = renderer.render(fps=fps, states=states, quiet=args.quiet)
    elapsed = time.time() - start_time

    summary_line = (
        f"preset={args.preset} frames={rendered_frames} fps={fps:.3f} "
        f"labels={len(label_files_used)} output={renderer.output_path}"
    )
    if not args.quiet:
        print(summary_line)

    log_json = {
        "input": str(renderer.input_path),
        "output": str(renderer.output_path),
        "preset": args.preset,
        "fps": fps,
        "frames": rendered_frames,
        "labels_present": bool(any(labels_sequence)),
        "label_files": [str(p) for p in label_files_used],
        "pad": pad,
        "lookahead": lookahead,
        "smoothing": smoothing,
        "flip180": bool(args.flip180),
        "portrait": renderer.canvas_size,
        "brand_overlay": str(renderer.overlay_path) if renderer.overlay_path else None,
        "endcard": str(renderer.endcard_path) if renderer.endcard_path else None,
        "temp_root": str(renderer.temp_root),
        "time_sec": elapsed,
    }
    renderer.log_path.write_text(summary_line + "\n", encoding="utf-8")
    renderer.log_path.with_suffix(renderer.log_path.suffix + ".json").write_text(
        json.dumps(log_json, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:  # pragma: no cover - CLI
        if exc.stderr:
            print(exc.stderr, file=sys.stderr)
        raise SystemExit(exc.returncode)

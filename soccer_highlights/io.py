"""Video/audio IO helpers built on ffmpeg/ffprobe."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from ._loguru import logger


@dataclass
class VideoStreamInfo:
    path: Path
    duration: float
    fps: float
    width: int
    height: int
    time_base: Fraction


@dataclass
class AudioStreamInfo:
    path: Path
    duration: float
    sample_rate: int
    channels: int


class FFmpegError(RuntimeError):
    pass


def run_command(cmd: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess:
    """Run a subprocess command logging the invocation."""

    logger.debug("Running command: {}", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise FFmpegError(f"Command failed with code {result.returncode}: {' '.join(cmd)}\n{result.stderr}")
    if result.stderr:
        logger.debug(result.stderr.strip())
    return result


def ffprobe_json(path: Path) -> Dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    result = run_command(cmd)
    if result.stdout:
        return json.loads(result.stdout)
    raise FFmpegError(f"ffprobe produced no output for {path}")


def _parse_fraction(value: str) -> Fraction:
    num, _, den = value.partition("/")
    if den:
        return Fraction(int(num), int(den))
    return Fraction(float(value)).limit_denominator()


def video_stream_info(path: Path) -> VideoStreamInfo:
    data = ffprobe_json(path)
    streams = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
    if not streams:
        raise FFmpegError(f"No video streams found in {path}")
    stream = streams[0]
    duration = float(stream.get("duration") or data["format"].get("duration") or 0.0)
    width = int(stream.get("width"))
    height = int(stream.get("height"))
    r_frame_rate = stream.get("r_frame_rate", "0/0")
    fps = float(Fraction(r_frame_rate)) if "0/0" not in r_frame_rate else float(stream.get("avg_frame_rate", 0.0))
    time_base = _parse_fraction(stream.get("time_base", "1/1"))
    return VideoStreamInfo(path=path, duration=duration, fps=fps, width=width, height=height, time_base=time_base)


def audio_stream_info(path: Path) -> AudioStreamInfo:
    data = ffprobe_json(path)
    streams = [s for s in data.get("streams", []) if s.get("codec_type") == "audio"]
    if not streams:
        raise FFmpegError(f"No audio streams found in {path}")
    stream = streams[0]
    duration = float(stream.get("duration") or data["format"].get("duration") or 0.0)
    sample_rate = int(stream.get("sample_rate", 48000))
    channels = int(stream.get("channels", 2))
    return AudioStreamInfo(path=path, duration=duration, sample_rate=sample_rate, channels=channels)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def seconds_to_timestamp(value: float) -> str:
    return f"{value:.3f}"


def list_from_paths(paths: Iterable[Path]) -> List[str]:
    return [str(p) for p in paths]

"""Clip exporter using ffmpeg with frame-accurate trimming."""
from __future__ import annotations

import concurrent.futures
from pathlib import Path
from typing import List

from ._loguru import logger
from ._tqdm import tqdm

from .config import AppConfig
from .io import VideoStreamInfo, run_command, seconds_to_timestamp, video_stream_info
from .utils import HighlightWindow, read_highlights


def _seek_mode(start: float, fps: float) -> str:
    frac = abs(start * fps - round(start * fps))
    if start < 3.0 or frac > 0.01:
        return "output"
    return "input"


def _build_command(video_path: Path, out_path: Path, start: float, end: float, info: VideoStreamInfo, config: AppConfig) -> List[str]:
    duration = max(0.05, end - start)
    fade = min(0.1, duration / 2)
    afilter = (
        f"afade=t=in:st=0:d={fade:.3f},"
        f"afade=t=out:st={max(duration - fade, 0):.3f}:d={fade:.3f},"
        "asetpts=N/SR/TB,aresample=async=1"
    )
    seek_mode = _seek_mode(start, info.fps)
    cmd: List[str] = ["ffmpeg", "-hide_banner", "-y"]
    if seek_mode == "input":
        cmd += ["-ss", seconds_to_timestamp(start)]
    cmd += ["-i", str(video_path)]
    if seek_mode == "output":
        cmd += ["-ss", seconds_to_timestamp(start)]
    cmd += [
        "-t",
        seconds_to_timestamp(duration),
        "-avoid_negative_ts",
        "make_zero",
        "-c:v",
        "libx264",
        "-preset",
        config.clips.preset,
        "-crf",
        str(config.clips.crf),
        "-c:a",
        "aac",
        "-b:a",
        config.clips.audio_bitrate,
        "-af",
        afilter,
        "-shortest",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    return cmd


def _export_one(video_path: Path, info: VideoStreamInfo, task: HighlightWindow, out_path: Path, config: AppConfig, overwrite: bool) -> bool:
    if out_path.exists() and not overwrite:
        logger.debug("Skipping existing %s", out_path)
        return False
    cmd = _build_command(video_path, out_path, task.start, task.end, info, config)
    run_command(cmd)
    return True


def export_clips(config: AppConfig, video_path: Path, csv_path: Path, out_dir: Path) -> List[Path]:
    info = video_stream_info(video_path)
    windows = [w for w in read_highlights(csv_path) if w.end - w.start >= config.clips.min_duration]
    if not windows:
        logger.warning("No clips to export from %s", csv_path)
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for idx, win in enumerate(windows, start=1):
        out_path = out_dir / f"clip_{idx:04d}.mp4"
        tasks.append((idx, win, out_path))

    exported: List[Path] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=config.clips.workers) as executor:
        future_map = {
            executor.submit(_export_one, video_path, info, win, out_path, config, config.clips.overwrite): (idx, out_path)
            for idx, win, out_path in tasks
        }
        for future in tqdm(concurrent.futures.as_completed(future_map), total=len(future_map), desc="clips", unit="clip", leave=False):
            idx, out_path = future_map[future]
            try:
                result = future.result()
            except Exception as exc:
                logger.error("Clip %d failed: %s", idx, exc)
            else:
                if result:
                    exported.append(out_path)
    return exported


def run_clips(config: AppConfig, video_path: Path, csv_path: Path, out_dir: Path) -> List[Path]:
    return export_clips(config, video_path, csv_path, out_dir)

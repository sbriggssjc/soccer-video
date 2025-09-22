"""Helpers for building ``ffconcat`` playlists used by ``ffmpeg``."""
from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Iterable, List, Sequence


@dataclasses.dataclass
class Clip:
    """A single clip reference inside an ``ffconcat`` playlist."""

    start: float
    end: float
    label: str = ""
    score: float = 0.0

    def duration(self) -> float:
        return max(0.0, float(self.end) - float(self.start))


def _validate_sorted_non_overlapping(clips: Sequence[Clip]) -> None:
    """Validate that ``clips`` are sorted, non-overlapping and well formed."""

    prev_end = None
    for idx, clip in enumerate(clips):
        if clip.end <= clip.start:
            raise ValueError(f"Clip #{idx} has non-positive duration: {clip}")
        if prev_end is not None and clip.start < prev_end - 1e-6:
            raise ValueError(
                f"Clip #{idx} at {clip.start:.3f}s overlaps previous end {prev_end:.3f}s"
            )
        prev_end = clip.end


def sort_clips(clips: Iterable[Clip]) -> List[Clip]:
    """Return clips sorted by their ``start`` timestamp."""

    return sorted((Clip(float(c.start), float(c.end), c.label, c.score) for c in clips), key=lambda c: c.start)


def write_ffconcat(clips: Sequence[Clip], video_path: Path, output_path: Path) -> None:
    """Write an ``ffconcat`` file describing ``clips``.

    Parameters
    ----------
    clips:
        Ordered clips to be written. They must already be validated as
        chronologically sorted and non-overlapping.
    video_path:
        Path to the source video. The file is referenced with an absolute
        POSIX-style path as required by ``ffmpeg``.
    output_path:
        Destination file. Parent directories are created automatically.
    """

    ordered = sort_clips(clips)
    _validate_sorted_non_overlapping(ordered)

    video_abs = Path(video_path).resolve()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["ffconcat version 1.0\n"]
    for clip in ordered:
        lines.append(f"file '{video_abs.as_posix()}'\n")
        lines.append(f"inpoint {clip.start:.3f}\n")
        lines.append(f"outpoint {clip.end:.3f}\n")

    output_path.write_text("".join(lines), encoding="utf-8")
    logging.debug("Wrote ffconcat with %d clips to %s", len(ordered), output_path)


def total_duration(clips: Iterable[Clip]) -> float:
    """Return the sum of clip durations."""

    return float(sum(max(0.0, c.end - c.start) for c in clips))


def ensure_duration_cap(clips: Sequence[Clip], target_seconds: float) -> List[Clip]:
    """Greedy selection of clips to satisfy a duration cap.

    Clips are processed in descending score order, with ties broken by start
    time. The resulting playlist preserves chronological order while limiting
    the total duration to ``target_seconds``.
    """

    if target_seconds <= 0:
        return list(clips)

    ordered = list(clips)
    ordered.sort(key=lambda c: (-c.score, c.start))
    picked: List[Clip] = []
    accumulated = 0.0
    for clip in ordered:
        duration = clip.duration()
        if accumulated + duration > target_seconds + 1e-6:
            continue
        picked.append(clip)
        accumulated += duration

    picked.sort(key=lambda c: c.start)
    try:
        _validate_sorted_non_overlapping(picked)
    except ValueError as exc:  # pragma: no cover - defensive logging
        logging.warning("Duration-cap selection produced overlapping clips: %s", exc)
    return picked


if __name__ == "__main__":  # pragma: no cover - smoke test
    demo_clips = [
        Clip(0.0, 5.0, label="BUILDUP", score=10.0),
        Clip(7.0, 12.5, label="GOAL", score=100.0),
    ]
    tmp = Path("/tmp/ffconcat_demo.ffconcat")
    try:
        write_ffconcat(demo_clips, Path("/tmp/video.mp4"), tmp)
        print(tmp.read_text())
    finally:
        if tmp.exists():
            tmp.unlink()


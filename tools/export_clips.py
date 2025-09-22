"""Export per-event highlight clips and playlists."""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

def _ffprobe_fps(video: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    if "/" in out:
        num, denom = out.split("/")
        fps = float(num) / float(denom) if float(denom) else float(num)
    else:
        fps = float(out)
    return fps


def _sanitize_label(label: str) -> str:
    cleaned = re.sub(r"[^0-9A-Z]+", "_", label.upper())
    return cleaned.strip("_") or "EVENT"


def _format_time(value: float) -> str:
    return f"{value:.2f}"


def _build_concat_file(video: Path, start: float, end: float) -> Path:
    tmp = tempfile.NamedTemporaryFile("w", suffix=".ffconcat", delete=False)
    try:
        tmp.write("ffconcat version 1.0\n")
        escaped = str(video).replace("'", "'\\''")
        tmp.write(f"file '{escaped}'\n")
        tmp.write(f"inpoint {start:.3f}\n")
        tmp.write(f"outpoint {end:.3f}\n")
        tmp.flush()
    finally:
        tmp.close()
    return Path(tmp.name)


def _run_ffmpeg(cmd: List[str]) -> None:
    process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd)}\n{process.stderr}")


def _anchor_time(label: str, start: float, end: float) -> float:
    upper = label.upper()
    if any(token in upper for token in ["GOAL", "SHOT", "CROSS", "SAVE", "GK"]):
        return end
    if any(token in upper for token in ["BUILD", "OFFENSE", "ATTACK", "PASS", "COMBINE", "DEFENSE", "TACKLE", "INTERCEPT", "BLOCK", "CLEAR"]):
        return (start + end) / 2.0
    return (start + end) / 2.0


def _write_ffconcat(path: Path, files: Iterable[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("ffconcat version 1.0\n")
        for file_path in files:
            escaped = str(file_path).replace("'", "'\\''")
            fh.write(f"file '{escaped}'\n")


def _write_srt(path: Path, events: pd.DataFrame) -> None:
    def format_srt_time(seconds: float) -> str:
        millis = int(round(seconds * 1000))
        hrs, rem = divmod(millis, 3600_000)
        mins, rem = divmod(rem, 60_000)
        secs, ms = divmod(rem, 1000)
        return f"{hrs:02}:{mins:02}:{secs:02},{ms:03}"

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for idx, row in enumerate(events.itertuples(index=False), start=1):
            fh.write(f"{idx}\n")
            fh.write(f"{format_srt_time(getattr(row, 't0'))} --> {format_srt_time(getattr(row, 't1'))}\n")
            label = getattr(row, "label", "EVENT")
            fh.write(f"{label}\n\n")


def export_clips(
    video: Path,
    events: pd.DataFrame,
    clips_dir: Path,
    reels_dir: Path,
    overlay_path: Optional[Path] = None,
) -> Dict[str, List[Path]]:
    clips_dir.mkdir(parents=True, exist_ok=True)
    reels_dir.mkdir(parents=True, exist_ok=True)

    fps = _ffprobe_fps(video)
    lists: Dict[str, List[Path]] = {"goals": [], "shots": [], "saves": [], "defense": [], "offense": [], "all": []}

    for idx, row in enumerate(events.itertuples(index=False), start=1):
        start = float(getattr(row, "t0"))
        end = float(getattr(row, "t1"))
        label = str(getattr(row, "label", "EVENT"))
        safe_label = _sanitize_label(label)
        clip_name = f"{idx:03d}__{safe_label}__t{_format_time(start)}-t{_format_time(end)}.mp4"
        clip_path = clips_dir / clip_name

        concat_file = _build_concat_file(video, start, end)
        try:
            cmd = [
                "ffmpeg",
                "-nostdin",
                "-y",
                "-safe",
                "0",
                "-f",
                "concat",
                "-i",
                str(concat_file),
                "-vsync",
                "cfr",
                "-fps_mode",
                "cfr",
                "-r",
                f"{fps:.6f}",
                "-af",
                "aresample=async=1:first_pts=0",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "160k",
                "-movflags",
                "+faststart",
                str(clip_path),
            ]
            _run_ffmpeg(cmd)
        finally:
            concat_file.unlink(missing_ok=True)

        anchor = min(max(_anchor_time(label, start, end), start), end)
        thumb_path = clips_dir / f"{idx:03d}__thumb.jpg"
        thumb_cmd = [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-ss",
            f"{anchor:.3f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(thumb_path),
        ]
        _run_ffmpeg(thumb_cmd)

        label_upper = label.upper()
        clip_abs = clip_path.resolve()
        lists["all"].append(clip_abs)
        if "GOAL" in label_upper:
            lists["goals"].append(clip_abs)
        if any(token in label_upper for token in ["SHOT", "CROSS"]):
            lists["shots"].append(clip_abs)
        if any(token in label_upper for token in ["SAVE", "GK"]):
            lists["saves"].append(clip_abs)
        if any(token in label_upper for token in ["DEFENSE", "TACKLE", "INTERCEPT", "BLOCK", "CLEAR"]):
            lists["defense"].append(clip_abs)
        if any(token in label_upper for token in ["BUILD", "OFFENSE", "ATTACK", "PASS", "COMBINE"]):
            lists["offense"].append(clip_abs)

    for name, files in lists.items():
        if name == "all" or files:
            _write_ffconcat(reels_dir / f"{name}.ffconcat", files)

    if overlay_path is not None:
        _write_srt(overlay_path, events)

    return lists


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export highlight clips")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--clips-dir", type=Path, required=True)
    parser.add_argument("--reels-dir", type=Path, required=True)
    parser.add_argument("--write-overlay", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    events = pd.read_csv(args.events)
    export_clips(
        args.video,
        events,
        clips_dir=args.clips_dir,
        reels_dir=args.reels_dir,
        overlay_path=args.write_overlay,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


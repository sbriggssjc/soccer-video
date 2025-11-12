"""Build highlight reels from ffconcat playlists."""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class ReelItem:
    order: int
    path: Path
    duration: float
    priority: int


PRIORITY_RULES = [
    (5, ["GOAL"]),
    (4, ["SHOT", "CROSS"]),
    (3, ["SAVE", "GK"]),
    (2, ["BUILD", "OFFENSE", "ATTACK", "PASS", "COMBINE"]),
    (1, ["DEFENSE", "TACKLE", "INTERCEPT", "BLOCK", "CLEAR"]),
]


def _probe_fps(video: Path) -> float:
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
    output = subprocess.check_output(cmd, text=True).strip()
    if "/" in output:
        num, denom = output.split("/")
        denom_val = float(denom)
        return float(num) / denom_val if denom_val else float(num)
    return float(output)


def _parse_file_line(line: str) -> Optional[Path]:
    line = line.strip()
    if not line or not line.lower().startswith("file"):
        return None
    _, value = line.split(" ", 1)
    value = value.strip()
    if value.startswith("'") and value.endswith("'"):
        value = value[1:-1].replace("\\'", "'")
    elif value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    return Path(value)


def _compute_priority(label: str) -> int:
    upper = label.upper()
    for score, tokens in PRIORITY_RULES:
        if any(token in upper for token in tokens):
            return score
    return 0


def _parse_item(order: int, path: Path) -> ReelItem:
    name = path.name
    match = re.search(r"__t(?P<start>\d+(?:\.\d+)?)\-t(?P<end>\d+(?:\.\d+)?)", name)
    if match:
        start = float(match.group("start"))
        end = float(match.group("end"))
        duration = max(0.0, end - start)
    else:
        duration = 0.0
    parts = name.split("__")
    label = parts[1] if len(parts) > 1 else name
    priority = _compute_priority(label)
    return ReelItem(order=order, path=path, duration=duration, priority=priority)


def _read_playlist(path: Path) -> List[ReelItem]:
    items: List[ReelItem] = []
    with path.open("r", encoding="utf-8") as fh:
        order = 0
        for line in fh:
            file_path = _parse_file_line(line)
            if file_path is None:
                continue
            items.append(_parse_item(order, file_path))
            order += 1
    return items


def _select_items(items: List[ReelItem], cap_frac: Optional[float]) -> List[ReelItem]:
    if cap_frac is None or not items:
        return items
    total_duration = sum(item.duration for item in items)
    if total_duration <= 0:
        return items
    limit = total_duration * cap_frac
    ranked = sorted(items, key=lambda item: (-item.priority, item.order))
    selected = []
    accumulated = 0.0
    for item in ranked:
        if item.duration == 0 and not selected:
            selected.append(item)
            continue
        if accumulated + item.duration <= limit or not selected:
            selected.append(item)
            accumulated += item.duration
        if accumulated >= limit:
            break
    selected = sorted(selected, key=lambda item: item.order)
    return selected


def _write_concat(items: List[ReelItem]) -> Path:
    tmp = tempfile.NamedTemporaryFile("w", suffix=".ffconcat", delete=False)
    try:
        tmp.write("ffconcat version 1.0\n")
        for item in items:
            escaped = str(item.path).replace("'", "'\\''")
            tmp.write(f"file '{escaped}'\n")
        tmp.flush()
    finally:
        tmp.close()
    return Path(tmp.name)


def _build_reel(list_path: Path, out_path: Path, cap_frac: Optional[float]) -> None:
    items = _read_playlist(list_path)
    if not items:
        raise ValueError(f"Playlist {list_path} is empty")
    selected = _select_items(items, cap_frac)
    concat_path = list_path
    if cap_frac is not None:
        concat_path = _write_concat(selected)
    fps = _probe_fps(selected[0].path if selected else items[0].path)
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
            str(concat_path),
            "-vsync",
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
            "-g",
            "48",
            "-sc_threshold",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-movflags",
            "+faststart",
            str(out_path),
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr)
    finally:
        if concat_path != list_path:
            concat_path.unlink(missing_ok=True)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build highlight reels from playlists")
    parser.add_argument("--list", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cap-frac", type=float, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _build_reel(args.list, out_path, args.cap_frac)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


#!/usr/bin/env python3
"""Export highlight clips using FFmpeg.

Reads ``out/highlights.csv`` and writes individual clips to
``out/clips/clip_####.mp4``. Existing clips are skipped unless
``--overwrite`` is given."""
from __future__ import annotations

import argparse
import csv
import subprocess as sp
from pathlib import Path


def safe_name(idx: int) -> str:
    return f"clip_{idx:04d}.mp4"


def run_ffmpeg(src: str, start: float, end: float, out: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        src,
        "-ss",
        f"{start}",
        "-to",
        f"{end}",
        "-c",
        "copy",
        str(out),
    ]
    sp.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", default="full_game_stabilized.mp4")
    p.add_argument("--csv", default="out/highlights.csv")
    p.add_argument("--outdir", default="out/clips")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.csv) as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, 1):
            start = float(row["start"])
            end = float(row["end"])
            out = outdir / safe_name(idx)
            if out.exists() and not args.overwrite:
                continue
            run_ffmpeg(args.video, start, end, out)


if __name__ == "__main__":
    main()

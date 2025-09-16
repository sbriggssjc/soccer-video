#!/usr/bin/env python3
"""Export highlight clips using FFmpeg.

Reads ``out/highlights.csv`` and writes individual clips to
``out/clips/clip_####.mp4``. Existing clips are skipped unless
``--overwrite`` is given."""
from __future__ import annotations

import argparse
import csv
import os
import subprocess as sp
from pathlib import Path


def safe_name(idx: int) -> str:
    return f"clip_{idx:04d}.mp4"


def run_ffmpeg(src: str, start: float, end: float, out: Path) -> None:
    """Re-encode a clip with audio fades and timestamp cleanup."""

    # Duration is needed to position the fade-out filter.
    dur = max(0.0, end - start)
    fade = min(0.05, dur / 2)  # tiny fades at head/tail
    af = (
        f"afade=t=in:st=0:d={fade:.3f},"
        f"afade=t=out:st={max(dur - fade, 0):.3f}:d={fade:.3f},"
        "asetpts=N/SR/TB,aresample=async=1"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start}",
        "-to",
        f"{end}",
        "-i",
        src,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-af",
        af,
        "-shortest",
        "-movflags",
        "+faststart",
        str(out),
    ]
    sp.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--min-dur", type=float, default=3.0, help="skip clips shorter than this")
    p.add_argument("--pre-roll", type=float, default=1.8, help="seconds before start")
    p.add_argument("--post-roll", type=float, default=2.4, help="seconds after end")
    args = p.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(args.csv):
        raise SystemExit(f"[make_highlights] CSV not found: {args.csv}")

    rows = []
    with open(args.csv, newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            start_raw = row.get("start") or row.get("t0")
            end_raw = row.get("end") or row.get("t1")

            if start_raw in (None, "") or end_raw in (None, ""):
                raise SystemExit(
                    f"[make_highlights] Missing start/end values in row {idx} of {args.csv}."
                )

            try:
                start = float(start_raw)
                end = float(end_raw)
            except (TypeError, ValueError) as exc:
                raise SystemExit(
                    f"[make_highlights] Invalid start/end values in row {idx} of {args.csv}."
                ) from exc

            clip_start = max(0.0, start - args.pre_roll)
            clip_end = max(clip_start + 0.80, end + args.post_roll)

            rows.append((clip_start, clip_end))

    if not rows:
        raise SystemExit(f"[make_highlights] No rows in {args.csv}.")

    print(f"[make_highlights] Using CSV: {args.csv}  rows={len(rows)}")

    idx = 1
    for start, end in rows:
        if end - start < args.min_dur:
            continue
        out = outdir / safe_name(idx)
        if out.exists() and not args.overwrite:
            idx += 1
            continue
        run_ffmpeg(args.video, start, end, out)
        idx += 1


if __name__ == "__main__":
    main()

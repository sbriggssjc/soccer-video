#!/usr/bin/env python3
"""Post-process selected events into a tightly capped highlight reel.

This utility mirrors the PowerShell snippet shared in the engineering notes.
It keeps only the high-signal action labels, widens the pre-roll for goals, and
trims everything else around the primary action moment.  After merging tiny
gaps and capping the total runtime, we render a CFR highlight reel via FFmpeg.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


@dataclass
class Span:
    """Simple time span helper."""

    t0: float
    t1: float

    @property
    def duration(self) -> float:
        return max(0.0, self.t1 - self.t0)


DEFAULT_KEEP = ("GOAL", "SHOT", "SAVE", "CROSS")

LABEL_CONFIG: Dict[str, Dict[str, float | str]] = {
    "GOAL": {"pre": 12.0, "post": 8.0, "anchor": "end", "extra_post": 1.5},
    "SHOT": {"pre": 6.0, "post": 4.0, "anchor": "end", "extra_post": 0.5},
    "SAVE": {"pre": 6.0, "post": 4.0, "anchor": "end", "extra_post": 0.5},
    "CROSS": {"pre": 5.0, "post": 3.0, "anchor": "end", "extra_post": 0.0},
    "DEFAULT": {"pre": 2.5, "post": 2.5, "anchor": "mid", "extra_post": 0.0},
}


def parse_float(val: object) -> float | None:
    """Parse a CSV field into a float."""

    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    text = str(val).strip()
    if not text:
        return None
    text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


def run_cmd(cmd: Sequence[str]) -> str:
    """Run a command and capture stdout (stripping whitespace)."""

    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return proc.stdout.strip()


def probe_duration(video: Path) -> float:
    output = run_cmd(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nk=1:nw=1",
            str(video),
        ]
    )
    try:
        return float(output)
    except ValueError as exc:  # pragma: no cover - ffprobe should give a float
        raise RuntimeError(f"Failed to parse duration from ffprobe output: {output!r}") from exc


def probe_fps(video: Path) -> str:
    output = run_cmd(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate",
            "-of",
            "default=nk=1:nw=1",
            str(video),
        ]
    )
    output = output.strip()
    if not output:
        raise RuntimeError("ffprobe returned empty avg_frame_rate")
    return output


def parse_events(events_path: Path, keep: Iterable[str]) -> List[dict]:
    keep_upper = {label.upper() for label in keep}

    with events_path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        rows = [dict(row) for row in reader]

    filtered = []
    for row in rows:
        label = str(row.get("label", "")).strip().upper()
        if label in keep_upper:
            filtered.append(row)
    return filtered


def write_core_csv(core_path: Path, rows: List[dict]) -> None:
    if not rows:
        core_path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with core_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def select_anchor(start: float, end: float, anchor: str) -> float:
    if anchor == "start":
        return start
    if anchor == "end":
        return end
    return (start + end) / 2.0


def build_spans(rows: Iterable[dict], video_dur: float) -> List[Span]:
    spans: List[Span] = []
    for row in rows:
        start = parse_float(row.get("start", row.get("t0")))
        end = parse_float(row.get("end", row.get("t1")))
        if start is None or end is None or not end > start:
            continue

        label = str(row.get("label", "")).strip().upper()
        config = LABEL_CONFIG.get(label, LABEL_CONFIG["DEFAULT"])
        pre = float(config["pre"])
        post = float(config["post"])
        extra_post = float(config.get("extra_post", 0.0))
        anchor = select_anchor(start, end, str(config.get("anchor", "mid")))

        t0 = max(0.0, anchor - pre)
        t1 = min(video_dur, anchor + post + extra_post)
        if t1 <= t0:
            continue
        spans.append(Span(t0=t0, t1=t1))
    return spans


def merge_spans(spans: List[Span], merge_gap: float) -> List[Span]:
    if not spans:
        return []
    spans = sorted(spans, key=lambda sp: (sp.t0, sp.t1))
    merged: List[Span] = [Span(spans[0].t0, spans[0].t1)]
    for span in spans[1:]:
        last = merged[-1]
        if span.t0 <= last.t1 + merge_gap:
            last.t1 = max(last.t1, span.t1)
        else:
            merged.append(Span(span.t0, span.t1))
    return merged


def cap_spans(spans: List[Span], budget: float) -> List[Span]:
    picked: List[Span] = []
    total = 0.0
    for span in spans:
        length = span.duration
        if length <= 0:
            continue
        if total + length <= budget + 1e-6:
            picked.append(span)
            total += length
        else:
            break
    return picked


def write_ffconcat(ffconcat_path: Path, video: Path, spans: Iterable[Span]) -> None:
    lines = ["ffconcat version 1.0"]
    video_str = video.resolve().as_posix()
    for span in spans:
        lines.append(f"file '{video_str}'")
        lines.append(f"inpoint {span.t0:.3f}")
        lines.append(f"outpoint {span.t1:.3f}")
    ffconcat_path.write_text("\n".join(lines) + "\n", encoding="ascii")


def run_ffmpeg(concat_path: Path, fps: str, out_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-stats",
        "-safe",
        "0",
        "-fflags",
        "+genpts",
        "-f",
        "concat",
        "-i",
        str(concat_path),
        "-vsync",
        "cfr",
        "-r",
        fps,
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
    subprocess.run(cmd, check=True)


def format_duration(seconds: float) -> str:
    minutes, secs = divmod(max(seconds, 0.0), 60.0)
    return f"{int(minutes):02d}:{secs:04.1f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=Path("out/full_game_stabilized.mp4"))
    parser.add_argument("--events", type=Path, default=Path("out/events_selected.csv"))
    parser.add_argument("--core-out", type=Path, default=Path("out/events_core.csv"))
    parser.add_argument("--ffconcat", type=Path, default=Path("out/segments_tight.ffconcat"))
    parser.add_argument("--out", type=Path, default=Path("out/top_highlights_tight.mp4"))
    parser.add_argument(
        "--keep",
        nargs="*",
        default=list(DEFAULT_KEEP),
        help="Labels to keep (case-insensitive).",
    )
    parser.add_argument("--cap-frac", type=float, default=0.07, help="Fraction of match to keep.")
    parser.add_argument(
        "--merge-gap",
        type=float,
        default=0.20,
        help="Merge spans that are this close together (seconds).",
    )
    args = parser.parse_args()

    if not args.events.exists():
        raise SystemExit(f"Events CSV not found: {args.events}")
    if not args.video.exists():
        raise SystemExit(f"Video not found: {args.video}")

    filtered_rows = parse_events(args.events, args.keep)
    if not filtered_rows:
        raise SystemExit("No events remain after filtering; nothing to do.")

    args.core_out.parent.mkdir(parents=True, exist_ok=True)
    write_core_csv(args.core_out, filtered_rows)

    video_duration = probe_duration(args.video)
    fps = probe_fps(args.video)

    spans = build_spans(filtered_rows, video_duration)
    if not spans:
        raise SystemExit("No valid spans after tightening windows.")

    merged_spans = merge_spans(spans, args.merge_gap)
    budget = video_duration * args.cap_frac
    capped_spans = cap_spans(merged_spans, budget)
    if not capped_spans:
        raise SystemExit("No spans remain after applying duration cap.")

    args.ffconcat.parent.mkdir(parents=True, exist_ok=True)
    write_ffconcat(args.ffconcat, args.video, capped_spans)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(args.ffconcat, fps, args.out)

    total_duration = sum(span.duration for span in capped_spans)
    print(
        f"[tighten_highlights] done -> {args.out} | spans={len(capped_spans)} "
        f"total={total_duration:.1f}s ({format_duration(total_duration)})"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Match individual DJI sideline clips to master highlight events by wall-clock time.

Given a folder of DJI clips (timestamps encoded in filenames) and a set of
highlight events (master_start/master_end in seconds from master recording
start), this tool:

1. Determines the master video's recording start time (via ffprobe creation_time)
2. Converts each event's master timestamps to wall-clock times
3. Finds DJI clips that overlap each event's time window
4. Extracts the overlapping portion into the atomic_clips sideline folder

Usage:
    # Dry-run preview for one game
    python tools/match_sideline_dji.py ^
        --game 2026-03-01__TSC_vs_OK_Celtic ^
        --sideline-dir "D:\\Projects\\soccer-video\\out\\masters\\2026-03-01__TSC_vs_OK_Celtic\\sideline_raw" ^
        --dry-run

    # With manual master start time override (local wall-clock)
    python tools/match_sideline_dji.py ^
        --game 2026-03-01__TSC_vs_OK_Celtic ^
        --sideline-dir "path\\to\\sideline_raw" ^
        --master-start-local "11:05:00"
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
CATALOG_DIR = ROOT / "out" / "catalog"
GAMES_DIR = ROOT / "out" / "games"
ATOMIC_DIR = ROOT / "out" / "atomic_clips"


# ---------------------------------------------------------------------------
# DJI filename parsing
# ---------------------------------------------------------------------------

_DJI_RE = re.compile(r"DJI_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})_\d+_video\.MP4",
                      re.IGNORECASE)


def parse_dji_wallclock(filename: str) -> Optional[datetime]:
    """Extract wall-clock start time from a DJI filename."""
    m = _DJI_RE.match(filename)
    if not m:
        return None
    y, mo, d, h, mi, s = (int(g) for g in m.groups())
    return datetime(y, mo, d, h, mi, s)


# ---------------------------------------------------------------------------
# ffprobe helpers
# ---------------------------------------------------------------------------

def ffprobe_creation_time(video: Path) -> Optional[datetime]:
    """Get creation_time tag from video metadata.  Returns timezone-aware UTC."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format_tags=creation_time",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video),
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.PIPE).strip()
        if not out:
            return None
        out = out.replace("Z", "+00:00")
        return datetime.fromisoformat(out)
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
        return None


def ffprobe_duration(video: Path) -> Optional[float]:
    """Get video duration in seconds."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video),
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.PIPE).strip()
        return float(out)
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
        return None


# ---------------------------------------------------------------------------
# Event loading
# ---------------------------------------------------------------------------

def load_events(game_label: str) -> list[dict]:
    """Load events from events_selected.csv (preferred) or plays_manual.csv."""
    events = []

    # Try events_selected.csv first
    csv_path = CATALOG_DIR / game_label / "events_selected.csv"
    if not csv_path.exists():
        csv_path = GAMES_DIR / game_label / "plays_manual.csv"
    if not csv_path.exists():
        return events

    with csv_path.open("r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            clip_id = (row.get("id") or row.get("clip_id", "")).strip().strip('"')
            if not clip_id:
                continue
            label = (row.get("playtag") or row.get("label", "")).strip().strip('"')
            start = row.get("master_start", "").strip()
            end = row.get("master_end", "").strip()
            if not start or not end:
                continue
            events.append({
                "id": clip_id,
                "label": label or "CLIP",
                "master_start": float(start),
                "master_end": float(end),
            })
    print(f"  Loaded {len(events)} events from {csv_path.name}")
    return events


# ---------------------------------------------------------------------------
# Sanitize for filenames
# ---------------------------------------------------------------------------

def _sanitize_label(label: str) -> str:
    cleaned = label.upper().replace("&", "AND").replace(" ", "_")
    cleaned = re.sub(r"[,'\"\(\)]+", "", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "CLIP"


# ---------------------------------------------------------------------------
# Core matching
# ---------------------------------------------------------------------------

def determine_master_start(
    master_path: Path,
    utc_offset_hours: float,
    manual_override: Optional[str],
    game_date: datetime,
) -> Optional[datetime]:
    """Determine wall-clock (local) time when master recording started."""

    if manual_override:
        parts = manual_override.split(":")
        h, m = int(parts[0]), int(parts[1])
        s = int(parts[2]) if len(parts) > 2 else 0
        return game_date.replace(hour=h, minute=m, second=s, microsecond=0)

    creation = ffprobe_creation_time(master_path)
    if creation:
        local_tz = timezone(timedelta(hours=utc_offset_hours))
        local_dt = creation.astimezone(local_tz).replace(tzinfo=None)
        print(f"  Master creation_time (UTC): {creation.isoformat()}")
        print(f"  Master creation_time (local, UTC{utc_offset_hours:+.0f}): "
              f"{local_dt.strftime('%H:%M:%S')}")
        return local_dt

    return None


def scan_dji_clips(sideline_dir: Path, verbose: bool = False) -> list[dict]:
    """Scan a directory for DJI clips, get their wall-clock times and durations."""
    clips = []
    files = sorted(sideline_dir.glob("DJI_*_video.MP4"))
    if not files:
        # Try lowercase
        files = sorted(sideline_dir.glob("DJI_*_video.mp4"))

    total = len(files)
    for i, f in enumerate(files):
        wall = parse_dji_wallclock(f.name)
        if wall is None:
            continue
        dur = ffprobe_duration(f)
        if dur is None:
            continue
        clips.append({
            "path": f,
            "name": f.name,
            "start": wall,
            "end": wall + timedelta(seconds=dur),
            "duration": dur,
        })
        if verbose or (i + 1) % 25 == 0 or (i + 1) == total:
            print(f"    Scanned {i + 1}/{total} DJI clips...", end="\r")

    print(f"  Scanned {len(clips)} DJI clips in {sideline_dir.name}        ")
    if clips:
        print(f"    Earliest: {clips[0]['start'].strftime('%H:%M:%S')}  "
              f"Latest end: {clips[-1]['end'].strftime('%H:%M:%S')}")
    return clips


def match_and_extract(
    events: list[dict],
    dji_clips: list[dict],
    master_start: datetime,
    game_label: str,
    out_dir: Path,
    *,
    pre_pad: float = 2.0,
    post_pad: float = 2.0,
    half_break_master: float = 0.0,
    halftime_gap: float = 0.0,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict:
    """Match events to DJI clips and extract overlapping portions.

    When the master video has halftime edited out, second-half events need
    a wall-clock correction.  Provide half_break_master (the master offset
    in seconds where the 2nd half begins) and halftime_gap (the real-world
    seconds of halftime that were cut).  For events at or past the break
    point, halftime_gap is added to the computed wall-clock time.
    """
    stats = {"matched": 0, "unmatched": 0, "clips_written": 0, "failed": 0}

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    for event in events:
        # For second-half events, add the halftime gap that was cut from
        # the master but elapsed in real (wall-clock) time.
        ht_offset = 0.0
        if half_break_master > 0 and event["master_start"] >= half_break_master:
            ht_offset = halftime_gap
        evt_start_wc = master_start + timedelta(seconds=event["master_start"] + ht_offset - pre_pad)
        evt_end_wc = master_start + timedelta(seconds=event["master_end"] + ht_offset + post_pad)

        # Find overlapping DJI clips
        overlapping = [
            c for c in dji_clips
            if c["start"] < evt_end_wc and c["end"] > evt_start_wc
        ]

        safe_label = _sanitize_label(event["label"])
        eid = event["id"]
        half_tag = ""
        if half_break_master > 0:
            half_tag = " [2H]" if ht_offset > 0 else " [1H]"

        if not overlapping:
            print(f"    [{eid:>3s}] {event['label']:<30s} "
                  f"({evt_start_wc.strftime('%H:%M:%S')}-"
                  f"{evt_end_wc.strftime('%H:%M:%S')})  -> NO MATCH{half_tag}")
            stats["unmatched"] += 1
            continue

        stats["matched"] += 1

        for i, clip in enumerate(overlapping):
            suffix = f"_part{i + 1}" if len(overlapping) > 1 else ""
            clip_name = (
                f"{eid}__{game_label}__SIDELINE__{safe_label}{suffix}.mp4"
            )
            clip_path = out_dir / clip_name

            # How much of the DJI clip to extract
            trim_start = max(0.0, (evt_start_wc - clip["start"]).total_seconds())
            trim_end = min(clip["duration"],
                           (evt_end_wc - clip["start"]).total_seconds())
            trim_dur = trim_end - trim_start

            if trim_dur <= 0:
                continue

            if dry_run:
                print(f"    [{eid:>3s}] {event['label']:<30s} "
                      f"({evt_start_wc.strftime('%H:%M:%S')}-"
                      f"{evt_end_wc.strftime('%H:%M:%S')})  "
                      f"-> {clip['name']}  "
                      f"[{trim_start:.1f}s-{trim_end:.1f}s] ({trim_dur:.1f}s)"
                      f"{suffix}")
                stats["clips_written"] += 1
                continue

            if clip_path.exists() and not overwrite:
                stats["clips_written"] += 1
                continue

            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-ss", f"{trim_start:.3f}",
                "-i", str(clip["path"]),
                "-t", f"{trim_dur:.3f}",
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                str(clip_path),
            ]

            try:
                subprocess.run(cmd, capture_output=True, check=True, text=True)
                size_kb = clip_path.stat().st_size / 1024
                print(f"    [{eid:>3s}] {event['label']:<30s} "
                      f"-> {clip_name} ({size_kb:.0f} KB)")
                stats["clips_written"] += 1
            except subprocess.CalledProcessError as exc:
                msg = exc.stderr[:200] if exc.stderr else "unknown error"
                print(f"    [{eid:>3s}] FAIL {clip_name}: {msg}")
                stats["failed"] += 1

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--game", required=True,
                   help="Game label, e.g. 2026-03-01__TSC_vs_OK_Celtic")
    p.add_argument("--sideline-dir", required=True, type=Path,
                   help="Directory containing DJI_*.MP4 clips")
    p.add_argument("--master", type=Path, default=None,
                   help="Path to master video (for auto-detecting start time)")
    p.add_argument("--master-start-local", type=str, default=None,
                   help="Override master start time in local wall-clock (HH:MM:SS)")
    p.add_argument("--utc-offset", type=float, default=-6,
                   help="Local timezone UTC offset in hours (default: -6 for CST)")
    p.add_argument("--pre-pad", type=float, default=2.0,
                   help="Extra seconds before each event (default: 2.0)")
    p.add_argument("--post-pad", type=float, default=2.0,
                   help="Extra seconds after each event (default: 2.0)")

    # Halftime gap correction (same concept as extract_sideline_angles.py)
    p.add_argument("--half-break-master", type=float, default=0.0,
                   help="Master offset (seconds) where 2nd half starts. "
                        "Required when master has halftime edited out.")
    p.add_argument("--halftime-gap", type=float, default=0.0,
                   help="Real-world seconds of halftime that were cut from "
                        "the master. For events past the break point, this "
                        "is added to the wall-clock time.")
    p.add_argument("--dry-run", action="store_true",
                   help="Preview matches without extracting")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing sideline clips")
    p.add_argument("--outdir", type=Path, default=None,
                   help="Override output directory")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    print()
    print("=" * 78)
    print("  SIDELINE DJI CLIP MATCHING")
    print("=" * 78)

    # 1. Load events
    events = load_events(args.game)
    if not events:
        print("  ERROR: No events found.")
        return 1

    # 2. Determine master start time
    # Extract date from game label (assumes YYYY-MM-DD__ prefix)
    date_match = re.match(r"(\d{4})-(\d{2})-(\d{2})", args.game)
    if not date_match:
        print("  ERROR: Cannot parse date from game label.")
        return 1
    game_date = datetime(int(date_match.group(1)),
                         int(date_match.group(2)),
                         int(date_match.group(3)))

    master_path = args.master
    if master_path is None:
        # Default master path
        master_path = (ROOT / "out" / "masters" / args.game / "MASTER.mp4")

    master_start = determine_master_start(
        master_path, args.utc_offset, args.master_start_local, game_date,
    )
    if master_start is None:
        print("  ERROR: Could not determine master start time.")
        print("  Use --master-start-local HH:MM:SS to provide it manually.")
        return 1

    print(f"  Master recording start (local): {master_start.strftime('%H:%M:%S')}")

    # Halftime gap info
    half_break_master = args.half_break_master
    halftime_gap = args.halftime_gap
    if half_break_master > 0:
        first_half = sum(1 for e in events if e["master_start"] < half_break_master)
        second_half = len(events) - first_half
        print(f"  Halftime break at master t={half_break_master:.0f}s, "
              f"gap={halftime_gap:.0f}s")
        print(f"  1st half events: {first_half}, 2nd half events: {second_half}")

    # 3. Scan DJI clips
    dji_clips = scan_dji_clips(args.sideline_dir)
    if not dji_clips:
        print("  ERROR: No DJI clips found.")
        return 1

    # 4. Match and extract
    out_dir = args.outdir or (ATOMIC_DIR / args.game / "sideline")
    tag = "[DRY-RUN] " if args.dry_run else ""
    print(f"\n  {tag}Matching {len(events)} events against "
          f"{len(dji_clips)} DJI clips...")
    print(f"  Padding: {args.pre_pad:.1f}s before, {args.post_pad:.1f}s after")
    print(f"  Output: {out_dir}")
    print()

    stats = match_and_extract(
        events, dji_clips, master_start, args.game, out_dir,
        pre_pad=args.pre_pad, post_pad=args.post_pad,
        half_break_master=half_break_master, halftime_gap=halftime_gap,
        dry_run=args.dry_run, overwrite=args.overwrite,
    )

    # Summary
    print()
    print("-" * 78)
    print(f"  {tag}Events matched: {stats['matched']} | "
          f"Unmatched: {stats['unmatched']} | "
          f"Clips written: {stats['clips_written']} | "
          f"Failed: {stats['failed']}")
    print()

    return 1 if stats["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())

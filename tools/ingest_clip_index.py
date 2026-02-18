"""Ingest clip_index files (CSV or TXT) into the atomic clips catalog.

Some game folders contain a clip_index.csv (or clip_index.txt) that lists
manually-identified highlight clips with timestamps and event labels. This
tool parses those files and adds entries to atomic_index.csv so they can
flow through the normal pipeline (extract -> upscale -> portrait render -> brand).

Preferred format: clip_index.csv
    Columns: clip_num, description, start, end
    Timestamps: H:MM:SS, MM:SS, or seconds (e.g. 750.5)

Legacy format: clip_index.txt (one clip per line)
    TIMESTAMP_RANGE  EVENT_TYPE  [optional description]

Supported timestamp formats (both CSV and TXT):
    H:MM:SS           (hours:minutes:seconds)
    MM:SS             (minutes:seconds)
    SS or SS.ss       (plain seconds)
    H_MM_SS           (underscore-separated, from filenames)
    tSS.ss            (t-prefixed seconds, TXT only)

Usage:
    python tools/ingest_clip_index.py --scan                     # find all clip_index files
    python tools/ingest_clip_index.py --all                      # ingest all found indexes
    python tools/ingest_clip_index.py --game 2025-11-05__TSC_vs_TSC_Navy
    python tools/ingest_clip_index.py --file out/games/GAME/clip_index.csv --game GAME
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from timestamp_parse import parse_timestamp_pair

ROOT = Path(__file__).resolve().parents[1]
GAMES_DIR = ROOT / "out" / "games"
CATALOG_DIR = ROOT / "out" / "catalog"
ATOMIC_INDEX = CATALOG_DIR / "atomic_index.csv"
ATOMIC_DIR = ROOT / "out" / "atomic_clips"

ATOMIC_HEADERS = [
    "clip_id", "clip_name", "clip_path", "clip_rel", "clip_stem",
    "created_at_utc", "duration_s", "fps", "height", "master_path",
    "master_rel", "sha1_64", "tags", "t_end_s", "t_start_s", "width",
]

# Regex for tSS.ss-tSS.ss format
TS_SECONDS_RE = re.compile(
    r"t(-?\d+(?:\.\d+)?)\s*[-–]\s*t?(-?\d+(?:\.\d+)?)"
)

# Regex for MM:SS or H:MM:SS format (colon-separated)
TS_COLON_RE = re.compile(
    r"(\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?)\s*[-–]\s*(\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?)"
)

# Regex for H_MM_SS or MM_SS format (underscore-separated, from filenames)
TS_UNDERSCORE_RE = re.compile(
    r"(\d{1,2}_\d{2}_\d{2})\s*[-–]\s*(\d{1,2}_\d{2}_\d{2})"
)


def _colon_to_seconds(ts: str) -> float:
    """Convert H:MM:SS, MM:SS, or SS to seconds."""
    parts = ts.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(parts[0])


def _underscore_to_seconds(ts: str) -> float:
    """Convert H_MM_SS format to seconds."""
    parts = ts.split("_")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(parts[0])


def parse_clip_line(line: str) -> Optional[dict]:
    """Parse a single line from clip_index.txt.

    Returns dict with keys: t_start_s, t_end_s, event_type, description
    or None if line can't be parsed.
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    t_start = t_end = None

    # Try tSS-tSS format first
    m = TS_SECONDS_RE.search(line)
    if m:
        t_start = float(m.group(1))
        t_end = float(m.group(2))
        remainder = line[:m.start()] + line[m.end():]
    else:
        # Try colon format (H:MM:SS)
        m = TS_COLON_RE.search(line)
        if m:
            t_start = _colon_to_seconds(m.group(1))
            t_end = _colon_to_seconds(m.group(2))
            remainder = line[:m.start()] + line[m.end():]
        else:
            # Try underscore format (H_MM_SS from filenames)
            m = TS_UNDERSCORE_RE.search(line)
            if m:
                t_start = _underscore_to_seconds(m.group(1))
                t_end = _underscore_to_seconds(m.group(2))
                remainder = line[:m.start()] + line[m.end():]
            else:
                return None

    # Parse event type and description from remainder
    remainder = remainder.strip()
    # Strip leading clip number prefix (e.g. "001__" or "013__")
    remainder = re.sub(r"^\d{1,4}__", "", remainder).strip()
    # Known event types, ordered by priority (most important first).
    # When multiple keywords appear in a description, the highest-priority
    # match becomes the event_type.
    events = [
        "GOAL", "PENALTY", "SAVE", "SHOT", "DRIBBLING", "DRIBBLE",
        "BUILD_UP_PLAY", "BUILD_UP", "BUILD UP", "FREE_KICK", "FREE KICK",
        "CORNER", "CROSS", "HEADER", "TACKLE", "PRESSURE",
        "THROUGH_BALL", "THROUGH BALL", "INTERCEPTION", "CLEARANCE",
        "ASSIST", "PASS", "CELEBRATION",
    ]

    event_type = "HIGHLIGHT"
    description = remainder
    remainder_upper = remainder.upper()

    for ev in events:
        if ev in remainder_upper:
            event_type = ev.replace(" ", "_")
            break
    # Use full remainder as description (human-readable context)
    description = remainder.strip(" \t-–,:__")

    return {
        "t_start_s": round(t_start, 2),
        "t_end_s": round(t_end, 2),
        "event_type": event_type,
        "description": description,
    }


def parse_clip_index(path: Path) -> List[dict]:
    """Parse a clip_index.txt file and return list of clip dicts."""
    clips = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_clip_line(line)
            if parsed:
                clips.append(parsed)
    return clips


def _ts_to_seconds(value: str) -> Optional[float]:
    """Convert a timestamp string to seconds.

    Accepts H:MM:SS, MM:SS, H_MM_SS, MM_SS, or plain seconds.
    """
    value = value.strip()
    if not value:
        return None
    # Colon format: H:MM:SS or MM:SS
    if ":" in value:
        return _colon_to_seconds(value)
    # Underscore format: H_MM_SS or MM_SS
    if "_" in value:
        return _underscore_to_seconds(value)
    # Plain seconds
    try:
        return float(value)
    except ValueError:
        return None


def parse_clip_csv(path: Path) -> List[dict]:
    """Parse a clip_index.csv file and return list of clip dicts.

    Expected columns: clip_num, description, start, end

    Uses ``parse_timestamp_pair`` for MM:SS:cs vs H:MM:SS disambiguation
    when both start and end are three-part timestamps.  Falls back to
    ``_ts_to_seconds`` for underscore-format timestamps (H_MM_SS).
    """
    clips = []
    with path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            start_raw = row.get("start", "").strip()
            end_raw = row.get("end", "").strip()
            if not start_raw or not end_raw:
                continue

            # Underscore format (H_MM_SS) is filename-specific; handle locally
            if "_" in start_raw or "_" in end_raw:
                t_start = _ts_to_seconds(start_raw)
                t_end = _ts_to_seconds(end_raw)
            else:
                # Use shared module for colon/plain-seconds disambiguation
                t_start, t_end = parse_timestamp_pair(start_raw, end_raw)

            if t_start is None or t_end is None:
                continue

            description = row.get("description", "").strip()

            # Detect event type from description
            events = [
                "GOAL", "PENALTY", "SAVE", "SHOT", "DRIBBLING", "DRIBBLE",
                "BUILD_UP_PLAY", "BUILD_UP", "BUILD UP", "FREE_KICK", "FREE KICK",
                "CORNER", "CROSS", "HEADER", "TACKLE", "PRESSURE",
                "THROUGH_BALL", "THROUGH BALL", "INTERCEPTION", "CLEARANCE",
                "ASSIST", "PASS", "CELEBRATION",
            ]
            event_type = "HIGHLIGHT"
            desc_upper = description.upper()
            for ev in events:
                if ev in desc_upper:
                    event_type = ev.replace(" ", "_")
                    break

            clips.append({
                "t_start_s": round(t_start, 2),
                "t_end_s": round(t_end, 2),
                "event_type": event_type,
                "description": description,
            })
    return clips


def _load_existing_index() -> List[dict]:
    """Load existing atomic_index.csv rows."""
    if not ATOMIC_INDEX.exists():
        return []
    with ATOMIC_INDEX.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _existing_timestamps(rows: List[dict], game_name: str) -> set:
    """Get set of (t_start, t_end) tuples already in the index for a game."""
    ts = set()
    for row in rows:
        rel = row.get("clip_rel", "")
        if game_name in rel:
            start = row.get("t_start_s", "").strip()
            end = row.get("t_end_s", "").strip()
            if start and end:
                try:
                    ts.add((round(float(start), 2), round(float(end), 2)))
                except ValueError:
                    pass
    return ts


def _next_clip_number(rows: List[dict], game_name: str) -> int:
    """Find the next available clip number for a game."""
    max_num = 0
    for row in rows:
        rel = row.get("clip_rel", "")
        if game_name in rel:
            name = row.get("clip_name", "")
            m = re.match(r"^(\d+)__", name)
            if m:
                max_num = max(max_num, int(m.group(1)))
    return max_num + 1


def ingest_game(
    game_name: str,
    clip_index_path: Optional[Path] = None,
    *,
    dry_run: bool = False,
) -> dict:
    """Ingest clip_index.txt entries for a game into atomic_index.csv.

    Returns dict with counts: added, skipped (already exist), failed.
    """
    # Find clip_index file (CSV preferred, TXT fallback)
    if clip_index_path is None:
        ci_csv = GAMES_DIR / game_name / "clip_index.csv"
        ci_txt = GAMES_DIR / game_name / "clip_index.txt"
        if ci_csv.exists():
            clip_index_path = ci_csv
        elif ci_txt.exists():
            clip_index_path = ci_txt
        else:
            print(f"  No clip_index.csv or clip_index.txt found for {game_name}")
            return {"added": 0, "skipped": 0, "failed": 0}

    if not clip_index_path.exists():
        print(f"  File not found: {clip_index_path}")
        return {"added": 0, "skipped": 0, "failed": 0}

    if clip_index_path.suffix == ".csv":
        clips = parse_clip_csv(clip_index_path)
    else:
        clips = parse_clip_index(clip_index_path)
    if not clips:
        print(f"  No valid clips parsed from {clip_index_path}")
        return {"added": 0, "skipped": 0, "failed": 0}

    print(f"  Parsed {len(clips)} clips from {clip_index_path.name}")

    # Load existing index
    existing_rows = _load_existing_index()
    existing_ts = _existing_timestamps(existing_rows, game_name)
    next_num = _next_clip_number(existing_rows, game_name)

    added = 0
    skipped = 0
    new_rows = []

    master_rel = f"out/games/{game_name}/MASTER.mp4"

    for clip in clips:
        ts_key = (clip["t_start_s"], clip["t_end_s"])

        # Check for overlap with existing clips (within 1 second tolerance)
        is_duplicate = False
        for (es, ee) in existing_ts:
            if abs(es - clip["t_start_s"]) < 1.0 and abs(ee - clip["t_end_s"]) < 1.0:
                is_duplicate = True
                break

        if is_duplicate:
            print(f"    SKIP (exists): t{clip['t_start_s']}-t{clip['t_end_s']} {clip['event_type']}")
            skipped += 1
            continue

        # Build clip name and paths
        clip_num = f"{next_num:03d}"
        event = clip["event_type"]
        t_start = clip["t_start_s"]
        t_end = clip["t_end_s"]
        clip_stem = f"{clip_num}__{game_name}__{event}__t{t_start:.2f}-t{t_end:.2f}"
        clip_name = f"{clip_stem}.mp4"
        clip_rel = f"out/atomic_clips/{game_name}/{clip_name}"
        duration = round(t_end - t_start, 3)

        row = {
            "clip_id": f":{clip_stem}",  # no SHA until file exists
            "clip_name": clip_name,
            "clip_path": "",
            "clip_rel": clip_rel,
            "clip_stem": clip_stem,
            "created_at_utc": "",
            "duration_s": f"{duration:.3f}",
            "fps": "",
            "height": "",
            "master_path": "",
            "master_rel": master_rel,
            "sha1_64": "",
            "tags": clip["description"],
            "t_end_s": str(t_end),
            "t_start_s": str(t_start),
            "width": "",
        }

        new_rows.append(row)
        existing_ts.add(ts_key)
        next_num += 1
        added += 1
        print(f"    ADD: {clip_name}" + (f"  ({clip['description']})" if clip['description'] else ""))

    if dry_run:
        print(f"  [DRY-RUN] Would add {added} clips, skip {skipped}")
        return {"added": added, "skipped": skipped, "failed": 0}

    if new_rows:
        # Append to existing index
        CATALOG_DIR.mkdir(parents=True, exist_ok=True)
        all_rows = existing_rows + new_rows
        with ATOMIC_INDEX.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ATOMIC_HEADERS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"  Wrote {len(all_rows)} total rows to atomic_index.csv (+{added} new)")

    return {"added": added, "skipped": skipped, "failed": 0}


def scan_for_clip_indexes() -> List[Tuple[str, Path]]:
    """Find all clip_index files (CSV preferred, TXT fallback) in game folders."""
    found = []
    if not GAMES_DIR.is_dir():
        return found
    for d in sorted(GAMES_DIR.iterdir()):
        if d.is_dir():
            ci_csv = d / "clip_index.csv"
            ci_txt = d / "clip_index.txt"
            if ci_csv.exists():
                found.append((d.name, ci_csv))
            elif ci_txt.exists():
                found.append((d.name, ci_txt))
    return found


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scan", action="store_true", help="Find all clip_index files (CSV or TXT)")
    group.add_argument("--all", action="store_true", help="Ingest all found clip indexes")
    group.add_argument("--game", help="Ingest clip index for a specific game")

    parser.add_argument("--file", type=Path, help="Path to a specific clip_index.csv or .txt")
    parser.add_argument("--dry-run", action="store_true", help="Preview without modifying catalog")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.scan:
        found = scan_for_clip_indexes()
        if not found:
            print(f"No clip_index.txt files found in {GAMES_DIR}")
            return 0
        print(f"Found {len(found)} clip_index.txt files:")
        for game_name, path in found:
            clips = parse_clip_index(path)
            print(f"  {game_name}: {len(clips)} clips")
        return 0

    if args.game:
        result = ingest_game(args.game, args.file, dry_run=args.dry_run)
        print(f"\nResults: {result['added']} added, {result['skipped']} skipped")
        return 0

    # --all
    found = scan_for_clip_indexes()
    if not found:
        print(f"No clip_index.txt files found in {GAMES_DIR}")
        return 0

    total_added = 0
    total_skipped = 0
    for game_name, path in found:
        print(f"\n{game_name}:")
        result = ingest_game(game_name, path, dry_run=args.dry_run)
        total_added += result["added"]
        total_skipped += result["skipped"]

    print(f"\nTotal: {total_added} added, {total_skipped} skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())

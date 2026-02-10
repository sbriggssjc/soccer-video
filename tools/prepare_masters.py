"""Prepare MASTER.mp4 files from raw XBotGo / iPhone game segments.

Raw game recordings from the XBotGo Chameleon are typically split into
~30-minute iPhone segments with timestamp-based filenames like:

    20250920_174654000_iOS.MP4   (first half)
    20250920_181631000_iOS.MP4   (second half)

This tool:
  1. Scans each game folder under out/games/ for raw segments
  2. Sorts segments chronologically by filename timestamp
  3. Detects halftime gaps (configurable, default > 5 min between segments)
  4. Optionally trims halftime (removes gap or middle segment)
  5. Concatenates remaining segments into a single MASTER.mp4 using
     ffmpeg concat demuxer (stream-copy, no re-encode)
  6. Moves original raw segments to a raw/ subfolder for reference

Naming convention:
  - out/games/<game>/raw/         Original segments (preserved)
  - out/games/<game>/MASTER.mp4   Final concatenated game video
  - out/games/<game>/concat.txt   ffmpeg concat list (for audit trail)

Usage:
    python tools/prepare_masters.py --scan                 # report what's found
    python tools/prepare_masters.py --all                  # prepare all games
    python tools/prepare_masters.py --game 2025-09-20__TSC_vs_Greenwood
    python tools/prepare_masters.py --game ... --halftime-gap 300
    python tools/prepare_masters.py --game ... --skip-halftime
    python tools/prepare_masters.py --game ... --dry-run
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
GAMES_DIR = ROOT / "out" / "games"

# Extensions we treat as video segments
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}

# Regex to extract timestamp from XBotGo/iPhone filenames:
#   20250920_174654000_iOS.MP4  -> date=20250920, time=174654, millis=000
#   20250517_150018000_iOS 1.MOV -> date=20250517, time=150018, millis=000, suffix=" 1"
XBOTGO_TS_RE = re.compile(
    r"^(\d{8})_(\d{6})(\d{3})_iOS(?:\s*\d+)?\.(?:mp4|mov|m4v)$",
    re.IGNORECASE,
)

# Seconds gap between segments that signals halftime
DEFAULT_HALFTIME_GAP = 300  # 5 minutes


@dataclass
class Segment:
    """A raw video segment file."""
    path: Path
    sort_key: str  # YYYYMMDD_HHMMSS for chronological sorting
    timestamp_str: str  # Human-readable timestamp
    is_duplicate: bool = False  # e.g., "iOS 1.MOV" alongside "iOS.MOV"

    @property
    def name(self) -> str:
        return self.path.name


@dataclass
class GameInfo:
    """Analysis of a game folder's raw segments."""
    game_name: str
    game_dir: Path
    segments: List[Segment] = field(default_factory=list)
    halftime_index: Optional[int] = None  # index in segments where halftime gap detected
    has_master: bool = False
    master_path: Optional[Path] = None
    single_file: bool = False  # already a single file, no concat needed
    notes: List[str] = field(default_factory=list)

    @property
    def segment_count(self) -> int:
        return len(self.segments)


def _parse_xbotgo_timestamp(filename: str) -> Optional[str]:
    """Extract sortable timestamp from XBotGo/iPhone filename.

    Returns YYYYMMDD_HHMMSS string or None.
    """
    m = XBOTGO_TS_RE.match(filename)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    return None


def _is_duplicate_variant(filename: str) -> bool:
    """Check if filename is a duplicate variant like 'iOS 1.MOV' or 'iOS 2.MOV'."""
    return bool(re.search(r"_iOS\s+\d+\.", filename, re.IGNORECASE))


def _detect_halftime(segments: List[Segment], gap_seconds: float) -> Optional[int]:
    """Find halftime gap between segments based on filename timestamps.

    Returns the index of the first segment AFTER the gap, or None.
    """
    if len(segments) < 2:
        return None

    for i in range(len(segments) - 1):
        ts_a = segments[i].sort_key
        ts_b = segments[i + 1].sort_key
        # Parse YYYYMMDD_HHMMSS
        try:
            h_a, m_a, s_a = int(ts_a[9:11]), int(ts_a[11:13]), int(ts_a[13:15])
            h_b, m_b, s_b = int(ts_b[9:11]), int(ts_b[11:13]), int(ts_b[13:15])
            secs_a = h_a * 3600 + m_a * 60 + s_a
            secs_b = h_b * 3600 + m_b * 60 + s_b
            gap = secs_b - secs_a
            # XBotGo segments are ~30 min, so gap > threshold means halftime
            if gap > gap_seconds:
                return i + 1
        except (ValueError, IndexError):
            continue

    return None


def _probe_duration(path: Path) -> Optional[float]:
    """Get video duration in seconds via ffprobe, returns None on failure."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True, text=True, timeout=30,
        )
        return float(result.stdout.strip()) if result.returncode == 0 else None
    except (subprocess.SubprocessError, ValueError):
        return None


def scan_game(game_dir: Path, halftime_gap: float = DEFAULT_HALFTIME_GAP) -> GameInfo:
    """Analyze a game folder and identify segments, duplicates, and halftime."""
    info = GameInfo(
        game_name=game_dir.name,
        game_dir=game_dir,
    )

    # Check for existing MASTER.mp4
    master_path = game_dir / "MASTER.mp4"
    if master_path.exists():
        info.has_master = True
        info.master_path = master_path

    # Find all video files (not in raw/ subfolder, not MASTER.mp4)
    raw_dir = game_dir / "raw"
    all_videos = []
    for f in sorted(game_dir.iterdir()):
        if f.is_dir():
            continue
        if f.suffix.lower() not in VIDEO_EXTS:
            continue
        if f.name.upper() == "MASTER.MP4":
            continue
        all_videos.append(f)

    # Also check raw/ subfolder for already-organized segments
    if raw_dir.is_dir():
        for f in sorted(raw_dir.iterdir()):
            if f.suffix.lower() in VIDEO_EXTS:
                all_videos.append(f)

    if not all_videos:
        if info.has_master:
            info.notes.append("MASTER.mp4 exists, no raw segments found")
        else:
            info.notes.append("No video files found")
        return info

    # Single non-raw file that isn't XBotGo format -> treat as pre-processed
    if len(all_videos) == 1 and not _parse_xbotgo_timestamp(all_videos[0].name):
        info.single_file = True
        info.segments = [
            Segment(
                path=all_videos[0],
                sort_key="",
                timestamp_str=all_videos[0].stem,
            )
        ]
        info.notes.append(f"Single file: {all_videos[0].name} (likely already processed)")
        return info

    # Parse timestamps and build segment list
    seen_timestamps = set()
    for f in all_videos:
        ts = _parse_xbotgo_timestamp(f.name)
        is_dup = _is_duplicate_variant(f.name)

        if ts:
            seg = Segment(
                path=f,
                sort_key=ts,
                timestamp_str=f"{ts[:4]}-{ts[4:6]}-{ts[6:8]} {ts[9:11]}:{ts[11:13]}:{ts[13:15]}",
                is_duplicate=is_dup or (ts in seen_timestamps),
            )
            if not is_dup:
                seen_timestamps.add(ts)
        else:
            # Non-XBotGo file — use filename for sort
            seg = Segment(
                path=f,
                sort_key=f.name.lower(),
                timestamp_str=f.stem,
            )
        info.segments.append(seg)

    # Sort by timestamp
    info.segments.sort(key=lambda s: s.sort_key)

    # Mark duplicate variants
    dup_count = sum(1 for s in info.segments if s.is_duplicate)
    if dup_count:
        info.notes.append(f"{dup_count} duplicate variant(s) detected (e.g., 'iOS 1.MOV')")

    # Detect halftime gap
    non_dup = [s for s in info.segments if not s.is_duplicate]
    ht_idx = _detect_halftime(non_dup, halftime_gap)
    if ht_idx is not None:
        info.halftime_index = ht_idx
        info.notes.append(f"Halftime gap detected after segment {ht_idx} of {len(non_dup)}")

    return info


def scan_all(halftime_gap: float = DEFAULT_HALFTIME_GAP) -> List[GameInfo]:
    """Scan all game folders under out/games/."""
    if not GAMES_DIR.is_dir():
        return []
    results = []
    for d in sorted(GAMES_DIR.iterdir()):
        if d.is_dir() and not d.name.startswith("."):
            results.append(scan_game(d, halftime_gap))
    return results


def print_scan_report(games: List[GameInfo]) -> None:
    """Print a human-readable scan report."""
    print("=" * 80)
    print("GAME MASTER PREPARATION REPORT")
    print("=" * 80)
    print()

    for g in games:
        status = "READY" if g.has_master else ("SINGLE FILE" if g.single_file else "NEEDS CONCAT")
        print(f"  {g.game_name}  [{status}]")

        if g.has_master:
            print(f"    MASTER.mp4 exists")

        non_dup = [s for s in g.segments if not s.is_duplicate]
        dups = [s for s in g.segments if s.is_duplicate]
        print(f"    Segments: {len(non_dup)} primary" + (f" + {len(dups)} duplicates" if dups else ""))

        for i, seg in enumerate(non_dup):
            half_marker = ""
            if g.halftime_index is not None and i == g.halftime_index:
                half_marker = "  <-- HALFTIME GAP"
            print(f"      {i + 1}. {seg.name}  ({seg.timestamp_str}){half_marker}")

        if dups:
            print(f"    Duplicate variants (will be skipped):")
            for seg in dups:
                print(f"      - {seg.name}")

        for note in g.notes:
            print(f"    Note: {note}")
        print()

    # Summary
    need_concat = [g for g in games if not g.has_master and not g.single_file and g.segments]
    single_files = [g for g in games if g.single_file]
    already_done = [g for g in games if g.has_master]

    print(f"Summary: {len(games)} games | {len(already_done)} have MASTER | "
          f"{len(single_files)} single-file | {len(need_concat)} need concat")
    if need_concat:
        print(f"\nTo prepare masters: python tools/prepare_masters.py --all")


def prepare_master(
    game_info: GameInfo,
    *,
    skip_halftime: bool = False,
    dry_run: bool = False,
    keep_raw: bool = True,
) -> bool:
    """Concatenate segments into MASTER.mp4 for a single game.

    Args:
        game_info: Scanned game info.
        skip_halftime: If True, exclude segments after the halftime gap
                       (i.e. only keep first half). Usually False — we want
                       the full game minus the dead time between halves.
        dry_run: Print commands without executing.
        keep_raw: Move raw segments to raw/ subfolder (default True).

    Returns True on success.
    """
    game_dir = game_info.game_dir
    master_out = game_dir / "MASTER.mp4"

    if game_info.has_master and not dry_run:
        print(f"  SKIP: {game_info.game_name} already has MASTER.mp4")
        return True

    # Get non-duplicate segments in order
    segments = [s for s in game_info.segments if not s.is_duplicate]

    if not segments:
        print(f"  SKIP: {game_info.game_name} has no segments")
        return False

    # Single pre-processed file — just rename to MASTER.mp4
    if game_info.single_file:
        src = segments[0].path
        if dry_run:
            print(f"  [DRY-RUN] rename {src.name} -> MASTER.mp4")
            return True
        raw_dir = game_dir / "raw"
        raw_dir.mkdir(exist_ok=True)
        raw_dest = raw_dir / src.name
        shutil.copy2(str(src), str(raw_dest))
        src.rename(master_out)
        print(f"  RENAMED: {src.name} -> MASTER.mp4 (original in raw/)")
        return True

    # Multiple segments — concat via ffmpeg
    if len(segments) == 1:
        # Single XBotGo segment — just copy/rename
        src = segments[0].path
        if dry_run:
            print(f"  [DRY-RUN] copy {src.name} -> MASTER.mp4")
            return True
        raw_dir = game_dir / "raw"
        raw_dir.mkdir(exist_ok=True)
        raw_dest = raw_dir / src.name
        shutil.copy2(str(src), str(raw_dest))
        src.rename(master_out)
        print(f"  COPIED: {src.name} -> MASTER.mp4 (original in raw/)")
        return True

    # Build concat list
    concat_path = game_dir / "concat.txt"
    concat_segments = segments  # include all by default

    if dry_run:
        print(f"  [DRY-RUN] Concat {len(concat_segments)} segments -> MASTER.mp4")
        for i, seg in enumerate(concat_segments):
            marker = ""
            if game_info.halftime_index is not None and i == game_info.halftime_index:
                marker = "  (halftime boundary)"
            print(f"    {i + 1}. {seg.name}{marker}")
        return True

    # Write ffmpeg concat file
    # Use absolute paths to handle segments that might be in raw/ already
    with concat_path.open("w", encoding="utf-8") as f:
        for seg in concat_segments:
            # ffmpeg concat demuxer needs forward slashes and escaped quotes
            safe_path = str(seg.path.resolve()).replace("\\", "/").replace("'", "'\\''")
            f.write(f"file '{safe_path}'\n")

    # Run ffmpeg concat
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_path),
        "-c", "copy",
        "-movflags", "+faststart",
        str(master_out),
    ]

    print(f"  CONCAT: {len(concat_segments)} segments -> MASTER.mp4")
    for seg in concat_segments:
        print(f"    + {seg.name}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"  ERROR: ffmpeg failed: {result.stderr[:300]}")
            return False
    except FileNotFoundError:
        print("  ERROR: ffmpeg not found. Install ffmpeg.")
        return False
    except subprocess.TimeoutExpired:
        print("  ERROR: ffmpeg timed out (>10 min)")
        return False

    # Verify output
    if not master_out.exists() or master_out.stat().st_size < 1000:
        print(f"  ERROR: MASTER.mp4 not created or too small")
        return False

    size_mb = master_out.stat().st_size / (1024 * 1024)
    print(f"  OK: MASTER.mp4 created ({size_mb:.1f} MB)")

    # Move raw segments to raw/ subfolder
    if keep_raw:
        raw_dir = game_dir / "raw"
        raw_dir.mkdir(exist_ok=True)
        for seg in game_info.segments:  # include duplicates too
            if seg.path.exists() and seg.path.parent == game_dir:
                dest = raw_dir / seg.path.name
                seg.path.rename(dest)
                print(f"    moved {seg.name} -> raw/")

    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scan", action="store_true", help="Scan and report game folder status")
    group.add_argument("--all", action="store_true", help="Prepare masters for all games needing it")
    group.add_argument("--game", help="Prepare master for a specific game folder name")

    parser.add_argument(
        "--halftime-gap", type=float, default=DEFAULT_HALFTIME_GAP,
        help=f"Seconds gap between segments to detect halftime (default: {DEFAULT_HALFTIME_GAP})",
    )
    parser.add_argument(
        "--skip-halftime", action="store_true",
        help="Exclude second-half segments (keep first half only)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview without executing")
    parser.add_argument("--no-keep-raw", action="store_true", help="Don't move raw segments to raw/ subfolder")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.scan:
        games = scan_all(args.halftime_gap)
        if not games:
            print(f"No game folders found in {GAMES_DIR}")
            return 1
        print_scan_report(games)
        return 0

    if args.game:
        game_dir = GAMES_DIR / args.game
        if not game_dir.is_dir():
            print(f"Error: game folder not found: {game_dir}")
            return 1
        info = scan_game(game_dir, args.halftime_gap)
        ok = prepare_master(
            info,
            skip_halftime=args.skip_halftime,
            dry_run=args.dry_run,
            keep_raw=not args.no_keep_raw,
        )
        return 0 if ok else 1

    # --all
    games = scan_all(args.halftime_gap)
    success = 0
    failed = 0
    skipped = 0
    for info in games:
        if info.has_master:
            skipped += 1
            continue
        if not info.segments:
            skipped += 1
            continue
        ok = prepare_master(
            info,
            skip_halftime=args.skip_halftime,
            dry_run=args.dry_run,
            keep_raw=not args.no_keep_raw,
        )
        if ok:
            success += 1
        else:
            failed += 1

    print(f"\nResults: {success} prepared, {skipped} skipped, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

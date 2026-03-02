#!/usr/bin/env python
"""Extract sideline-camera clips at the same timestamps as XBot Go highlights.

Reads a plays_manual.csv (or events_selected.csv) and a sideline sources
config, applies a time offset to account for the different recording start
times between the XBot Go master and the sideline stabilizer, then extracts
matching clips from the sideline footage using ffmpeg stream-copy.

The key concept is the **sync offset**: the number of seconds to ADD to a
master timestamp to get the corresponding sideline timestamp.  For example,
if the XBot Go started recording 30 seconds before the stabilizer, the
offset is -30 (master t=30 maps to sideline t=0).  You can determine this
by finding the same visible event (kickoff whistle, a goal, etc.) in both
videos and computing: offset = sideline_time - master_time.

Usage:
    # Preview what would be extracted
    python tools/extract_sideline_angles.py \\
        --game 2026-02-23__TSC_vs_NEOFC \\
        --sideline /path/to/sideline_NEOFC.mp4 \\
        --offset -12.5 \\
        --dry-run

    # Extract sideline angles
    python tools/extract_sideline_angles.py \\
        --game 2026-02-23__TSC_vs_NEOFC \\
        --sideline /path/to/sideline_NEOFC.mp4 \\
        --offset -12.5

    # Use a sideline_sources.csv config for batch processing
    python tools/extract_sideline_angles.py --from-config

    # Extract for all games listed in sideline_sources.csv
    python tools/extract_sideline_angles.py --from-config --all
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from timestamp_parse import parse_timestamp_pair

ROOT = Path(__file__).resolve().parents[1]
GAMES_DIR = ROOT / "out" / "games"
CATALOG_DIR = ROOT / "out" / "catalog"
ATOMIC_DIR = ROOT / "out" / "atomic_clips"
SIDELINE_CONFIG = ROOT / "sideline_sources.csv"


@dataclass
class SidelineSource:
    """A sideline video source and its sync offset for a game."""
    game_label: str
    sideline_path: Path
    sync_offset: float  # seconds to ADD to master timestamp
    pre_pad: float = 1.0  # extra seconds before the highlight
    post_pad: float = 1.0  # extra seconds after the highlight


@dataclass
class PlayEntry:
    """A single play from plays_manual.csv."""
    clip_id: int
    label: str
    master_start: float
    master_end: float


def load_plays_manual(game_dir: Path) -> list[PlayEntry]:
    """Load plays_manual.csv for a game."""
    plays_path = game_dir / "plays_manual.csv"
    if not plays_path.exists():
        return []
    entries = []
    with plays_path.open("r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            raw_id = row.get("clip_id", "").strip().strip('"')
            if not raw_id:
                continue
            try:
                num = int(raw_id)
            except ValueError:
                continue
            label = row.get("label", "").strip().strip('"')
            t0, t1 = parse_timestamp_pair(
                row.get("master_start", ""),
                row.get("master_end", ""),
            )
            if t0 is None or t1 is None or t1 <= t0:
                continue
            entries.append(PlayEntry(clip_id=num, label=label,
                                     master_start=t0, master_end=t1))
    return entries


def load_sideline_config(config_path: Path) -> list[SidelineSource]:
    """Load sideline_sources.csv."""
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return []
    sources = []
    with config_path.open("r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            game = row.get("game_label", "").strip()
            path = row.get("sideline_path", "").strip()
            offset = row.get("sync_offset", "0").strip()
            pre = row.get("pre_pad", "1.0").strip()
            post = row.get("post_pad", "1.0").strip()
            if not game or not path:
                continue
            sources.append(SidelineSource(
                game_label=game,
                sideline_path=Path(path),
                sync_offset=float(offset),
                pre_pad=float(pre) if pre else 1.0,
                post_pad=float(post) if post else 1.0,
            ))
    return sources


def _sanitize_label(label: str) -> str:
    """Clean label for use in filenames."""
    cleaned = label.upper().replace("&", "AND").replace(" ", "_")
    cleaned = cleaned.replace(",", "").replace("__", "_").strip("_")
    return cleaned or "CLIP"


def _ffprobe_duration(video: Path) -> Optional[float]:
    """Get video duration in seconds via ffprobe."""
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


def extract_sideline_clips(
    plays: list[PlayEntry],
    source: SidelineSource,
    out_dir: Path,
    *,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict:
    """Extract sideline clips for all plays using the sync offset."""
    stats = {"extracted": 0, "skipped": 0, "failed": 0, "out_of_range": 0}

    if not source.sideline_path.exists():
        print(f"  ERROR: Sideline video not found: {source.sideline_path}")
        stats["failed"] = len(plays)
        return stats

    sideline_duration = _ffprobe_duration(source.sideline_path)
    if sideline_duration:
        print(f"  Sideline video duration: {sideline_duration:.1f}s")

    out_dir.mkdir(parents=True, exist_ok=True)

    for play in plays:
        # Apply sync offset to convert master timestamps to sideline timestamps
        sl_start = play.master_start + source.sync_offset - source.pre_pad
        sl_end = play.master_end + source.sync_offset + source.post_pad

        # Clamp to valid range
        if sl_start < 0:
            sl_start = 0
        if sideline_duration and sl_end > sideline_duration:
            sl_end = sideline_duration

        # Check if the window is valid in the sideline video
        if sl_end <= sl_start:
            print(f"    SKIP #{play.clip_id:03d} {play.label:<30s} "
                  f"out of sideline range (offset={source.sync_offset:+.1f}s)")
            stats["out_of_range"] += 1
            continue

        if sideline_duration and sl_start >= sideline_duration:
            print(f"    SKIP #{play.clip_id:03d} {play.label:<30s} "
                  f"starts beyond sideline end ({sl_start:.1f}s > {sideline_duration:.1f}s)")
            stats["out_of_range"] += 1
            continue

        safe_label = _sanitize_label(play.label)
        clip_name = (
            f"{play.clip_id:03d}__{source.game_label}__SIDELINE__{safe_label}"
            f"__t{play.master_start:.2f}-t{play.master_end:.2f}.mp4"
        )
        clip_path = out_dir / clip_name

        if clip_path.exists() and not overwrite:
            stats["skipped"] += 1
            continue

        duration = sl_end - sl_start

        if dry_run:
            print(f"    [DRY-RUN] #{play.clip_id:03d} {play.label:<30s} "
                  f"master t={play.master_start:.1f}-{play.master_end:.1f}s "
                  f"-> sideline t={sl_start:.1f}-{sl_end:.1f}s ({duration:.1f}s)")
            stats["extracted"] += 1
            continue

        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{sl_start:.3f}",
            "-i", str(source.sideline_path),
            "-t", f"{duration:.3f}",
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            str(clip_path),
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True, text=True)
            size_kb = clip_path.stat().st_size / 1024
            print(f"    OK #{play.clip_id:03d} {play.label:<30s} "
                  f"({duration:.1f}s, {size_kb:.0f} KB)")
            stats["extracted"] += 1
        except FileNotFoundError:
            print("    ERROR: ffmpeg not found.")
            stats["failed"] += 1
        except subprocess.CalledProcessError as exc:
            msg = exc.stderr[:200] if exc.stderr else "unknown error"
            print(f"    FAIL #{play.clip_id:03d} {play.label}: {msg}")
            stats["failed"] += 1

    return stats


def process_game(
    game_label: str,
    source: SidelineSource,
    *,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict:
    """Process a single game: load plays and extract sideline clips."""
    game_dir = GAMES_DIR / game_label
    if not game_dir.exists():
        print(f"  Game directory not found: {game_dir}")
        return {"game": game_label, "error": "game dir not found"}

    plays = load_plays_manual(game_dir)
    if not plays:
        print(f"  No plays_manual.csv found for {game_label}")
        return {"game": game_label, "error": "no plays_manual.csv"}

    print(f"\n  {game_label}")
    print(f"    Plays: {len(plays)}")
    print(f"    Sideline: {source.sideline_path}")
    print(f"    Sync offset: {source.sync_offset:+.1f}s "
          f"(pre_pad={source.pre_pad:.1f}s, post_pad={source.post_pad:.1f}s)")

    out_dir = ATOMIC_DIR / game_label / "sideline"
    stats = extract_sideline_clips(
        plays, source, out_dir,
        dry_run=dry_run, overwrite=overwrite,
    )
    stats["game"] = game_label
    stats["plays_count"] = len(plays)
    return stats


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Single-game mode
    p.add_argument("--game", help="Game label (e.g. 2026-02-23__TSC_vs_NEOFC)")
    p.add_argument("--sideline", type=Path,
                   help="Path to sideline stabilizer video")
    p.add_argument("--offset", type=float, default=0.0,
                   help="Sync offset in seconds (sideline_time - master_time)")
    p.add_argument("--pre-pad", type=float, default=1.0,
                   help="Extra seconds before each highlight (default: 1.0)")
    p.add_argument("--post-pad", type=float, default=1.0,
                   help="Extra seconds after each highlight (default: 1.0)")

    # Config-based batch mode
    p.add_argument("--from-config", action="store_true",
                   help="Read sideline_sources.csv for batch processing")
    p.add_argument("--config", type=Path, default=SIDELINE_CONFIG,
                   help="Path to sideline_sources.csv")
    p.add_argument("--all", action="store_true",
                   help="Process all games in the config")

    # Common flags
    p.add_argument("--dry-run", action="store_true",
                   help="Preview extractions without writing files")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing sideline clips")
    p.add_argument("--outdir", type=Path, default=None,
                   help="Override output directory")

    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    print()
    print("=" * 78)
    print("  SIDELINE ANGLE EXTRACTION")
    print("=" * 78)

    all_stats = []

    if args.from_config:
        sources = load_sideline_config(args.config)
        if not sources:
            print("  No sideline sources configured.")
            return 1

        for source in sources:
            if not args.all and args.game:
                if args.game.upper() not in source.game_label.upper():
                    continue
            elif not args.all:
                print("  Use --all to process all games, or --game LABEL to filter.")
                return 1

            stats = process_game(
                source.game_label, source,
                dry_run=args.dry_run, overwrite=args.overwrite,
            )
            all_stats.append(stats)

    elif args.game and args.sideline:
        source = SidelineSource(
            game_label=args.game,
            sideline_path=args.sideline,
            sync_offset=args.offset,
            pre_pad=args.pre_pad,
            post_pad=args.post_pad,
        )
        stats = process_game(
            args.game, source,
            dry_run=args.dry_run, overwrite=args.overwrite,
        )
        all_stats.append(stats)

    else:
        print("  Provide --game and --sideline, or use --from-config.")
        return 1

    # Summary
    total_extracted = sum(s.get("extracted", 0) for s in all_stats)
    total_skipped = sum(s.get("skipped", 0) for s in all_stats)
    total_failed = sum(s.get("failed", 0) for s in all_stats)
    total_oor = sum(s.get("out_of_range", 0) for s in all_stats)

    print()
    print("-" * 78)
    tag = "[DRY-RUN] " if args.dry_run else ""
    print(f"  {tag}Extracted: {total_extracted} | Skipped: {total_skipped} | "
          f"Out-of-range: {total_oor} | Failed: {total_failed}")
    print()

    if total_extracted > 0 and not args.dry_run:
        print("  Sideline clips saved under out/atomic_clips/<game>/sideline/")
        print()
        print("  Next steps:")
        print("    1. Review clips and adjust sync offset if needed")
        print("    2. Run portrait pipeline on sideline clips if desired:")
        print("       python tools/batch_pipeline.py --preset cinematic --game <GAME>")
        print()

    return 1 if total_failed else 0


if __name__ == "__main__":
    sys.exit(main())

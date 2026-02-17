#!/usr/bin/env python
"""Bootstrap atomic clips from game indexes that are not yet in the catalog.

Reads plays_manual.csv or clip_index.csv for each game, determines which
clips are missing from out/atomic_clips/, and extracts them from MASTER.mp4
using ffmpeg stream-copy.  After extraction, run:

    python tools/catalog.py --rebuild-atomic-index
    python tools/batch_pipeline.py --preset cinematic --upscale --game GAME

Usage:
    python tools/bootstrap_clips.py --report            # show what's missing
    python tools/bootstrap_clips.py --all --dry-run     # preview extraction
    python tools/bootstrap_clips.py --all               # extract everything
    python tools/bootstrap_clips.py --game SLSG         # one game only
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
GAMES_DIR = ROOT / "out" / "games"
ATOMIC_DIR = ROOT / "out" / "atomic_clips"
CATALOG_DIR = ROOT / "out" / "catalog"
ATOMIC_INDEX = CATALOG_DIR / "atomic_index.csv"


# ---------------------------------------------------------------------------
# Timestamp parsing  (MM:SS, MM:SS:cs, H:MM:SS, and plain seconds with commas)
# ---------------------------------------------------------------------------

_THREE_PART_RE = re.compile(r"^(\d+):(\d+):(\d+(?:\.\d+)?)$")  # A:B:C
_MMSS_RE = re.compile(r"^(\d+):(\d+(?:\.\d+)?)$")              # MM:SS or MM:SS.f
_SEC_RE = re.compile(r"^[\d,]+(?:\.\d+)?$")                      # plain seconds


def _parse_three_part_as_mmss_cs(a: float, b: float, c: float) -> float:
    """Interpret A:B:C as MM:SS:centiseconds (e.g., "5:15:00" = 5m 15.00s)."""
    return a * 60 + b + c / 100


def _parse_three_part_as_hmmss(a: float, b: float, c: float) -> float:
    """Interpret A:B:C as H:MM:SS (e.g., "0:06:37" = 0h 6m 37s)."""
    return a * 3600 + b * 60 + c


def parse_timestamp(raw: str) -> Optional[float]:
    """Parse a timestamp string to seconds."""
    raw = raw.strip().strip('"')
    if not raw:
        return None

    # Three-part: could be MM:SS:cs or H:MM:SS — resolved by parse_timestamp_pair
    m = _THREE_PART_RE.match(raw)
    if m:
        a, b, c = float(m.group(1)), float(m.group(2)), float(m.group(3))
        # Default to MM:SS:cs; pair-level disambiguation overrides when needed
        return _parse_three_part_as_mmss_cs(a, b, c)

    # Two-part: MM:SS or MM:SS.f  (e.g., "6:27" = 6m 27s = 387s)
    m = _MMSS_RE.match(raw)
    if m:
        return float(m.group(1)) * 60 + float(m.group(2))

    # Plain seconds with optional comma separators  (e.g., "1,247.80")
    m = _SEC_RE.match(raw)
    if m:
        return float(raw.replace(",", ""))

    return None


def parse_timestamp_pair(raw_start: str, raw_end: str) -> tuple[Optional[float], Optional[float]]:
    """Parse a start/end timestamp pair, disambiguating three-part formats.

    When both timestamps match A:B:C, tries MM:SS:cs first.  If that gives
    a clip shorter than 2 seconds, retries as H:MM:SS (needed for TSC Navy
    timestamps like "0:06:37" = 6 min 37 sec, not 6.37 sec).
    """
    raw_start = raw_start.strip().strip('"')
    raw_end = raw_end.strip().strip('"')
    if not raw_start or not raw_end:
        return None, None

    ms = _THREE_PART_RE.match(raw_start)
    me = _THREE_PART_RE.match(raw_end)

    if ms and me:
        sa, sb, sc = float(ms.group(1)), float(ms.group(2)), float(ms.group(3))
        ea, eb, ec = float(me.group(1)), float(me.group(2)), float(me.group(3))

        t0_mmss = _parse_three_part_as_mmss_cs(sa, sb, sc)
        t1_mmss = _parse_three_part_as_mmss_cs(ea, eb, ec)
        t0_hms = _parse_three_part_as_hmmss(sa, sb, sc)
        t1_hms = _parse_three_part_as_hmmss(ea, eb, ec)

        dur_mmss = t1_mmss - t0_mmss
        dur_hms = t1_hms - t0_hms

        # If MM:SS:cs gives a clip < 2 seconds but H:MM:SS gives a reasonable
        # duration, use H:MM:SS  (handles Navy-style "0:06:37-0:07:10")
        if dur_mmss < 2.0 and dur_hms >= 2.0:
            return t0_hms, t1_hms
        return t0_mmss, t1_mmss

    # Fall back to individual parsing
    return parse_timestamp(raw_start), parse_timestamp(raw_end)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ClipDef:
    """A clip defined in a game index."""
    clip_num: int
    label: str
    t_start_s: float
    t_end_s: float

    @property
    def duration_s(self) -> float:
        return self.t_end_s - self.t_start_s

    def canonical_name(self, game_folder: str) -> str:
        """Build the canonical filename: NNN__GAME__LABEL__tSTART-tEND.mp4"""
        label_clean = (
            self.label.upper()
            .replace("&", "AND")
            .replace(" ", "_")
            .replace("__", "_")
            .strip("_")
        )
        return (
            f"{self.clip_num:03d}__{game_folder}__{label_clean}"
            f"__t{self.t_start_s:.2f}-t{self.t_end_s:.2f}.mp4"
        )


# ---------------------------------------------------------------------------
# Index loaders
# ---------------------------------------------------------------------------

def load_plays_manual(path: Path) -> list[ClipDef]:
    """Load plays_manual.csv: clip_id, label, master_start, master_end, notes."""
    clips = []
    seen_ids = set()
    with path.open("r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            raw_id = row.get("clip_id", "").strip().strip('"')
            if not raw_id:
                continue
            try:
                num = int(raw_id)
            except ValueError:
                continue
            # Skip duplicate clip_id rows (some games have dual labels)
            if num in seen_ids:
                continue
            seen_ids.add(num)
            label = row.get("label", "").strip().strip('"')
            t0, t1 = parse_timestamp_pair(
                row.get("master_start", ""),
                row.get("master_end", ""),
            )
            if t0 is None or t1 is None or t1 <= t0:
                continue
            clips.append(ClipDef(clip_num=num, label=label, t_start_s=t0, t_end_s=t1))
    return clips


def load_clip_index(path: Path) -> list[ClipDef]:
    """Load clip_index.csv: clip_num, description, start, end."""
    clips = []
    with path.open("r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            raw_num = row.get("clip_num", "").strip()
            if not raw_num:
                continue
            try:
                num = int(raw_num)
            except ValueError:
                continue
            label = row.get("description", "").strip()
            t0, t1 = parse_timestamp_pair(
                row.get("start", ""),
                row.get("end", ""),
            )
            if t0 is None or t1 is None or t1 <= t0:
                continue
            clips.append(ClipDef(clip_num=num, label=label, t_start_s=t0, t_end_s=t1))
    return clips


def load_game_clips(game_dir: Path) -> tuple[list[ClipDef], str]:
    """Load the authoritative clip list for a game.  Prefers plays_manual.csv."""
    plays = game_dir / "plays_manual.csv"
    clip_csv = game_dir / "clip_index.csv"
    if plays.exists():
        return load_plays_manual(plays), "plays_manual.csv"
    if clip_csv.exists():
        return load_clip_index(clip_csv), "clip_index.csv"
    return [], ""


# ---------------------------------------------------------------------------
# Master resolution
# ---------------------------------------------------------------------------

def find_master(game_dir: Path) -> Optional[Path]:
    """Find MASTER.mp4 or full_game*.mp4 in a game directory."""
    master = game_dir / "MASTER.mp4"
    if master.exists():
        return master
    for p in game_dir.glob("full_game*.*"):
        if p.suffix.lower() in (".mp4", ".mov"):
            return p
    return None


# ---------------------------------------------------------------------------
# Existing clip detection
# ---------------------------------------------------------------------------

def load_existing_clips(game_folder: str) -> set[int]:
    """Return clip numbers already present on disk for this game."""
    clip_dir = ATOMIC_DIR / game_folder
    existing = set()
    if clip_dir.exists():
        for f in clip_dir.glob("*.mp4"):
            m = re.match(r"^(\d{3})__", f.name)
            if m:
                existing.add(int(m.group(1)))
    return existing


def load_indexed_clips(game_folder: str) -> set[int]:
    """Return clip numbers present in atomic_index.csv for this game."""
    indexed = set()
    if not ATOMIC_INDEX.exists():
        return indexed
    with ATOMIC_INDEX.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel = row.get("clip_rel", "").replace("\\", "/")
            name = row.get("clip_name", "")
            if game_folder.lower() in rel.lower() or game_folder.lower() in name.lower():
                m = re.match(r"^(\d{3})__", name)
                if m:
                    indexed.add(int(m.group(1)))
    return indexed


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_clip(master: Path, out_path: Path, clip: ClipDef, *, dry_run: bool) -> bool:
    """Extract a single clip from the master video using ffmpeg."""
    if clip.duration_s <= 0:
        print(f"    SKIP #{clip.clip_num:03d}: invalid duration {clip.duration_s:.1f}s")
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{clip.t_start_s:.3f}",
        "-i", str(master),
        "-t", f"{clip.duration_s:.3f}",
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        str(out_path),
    ]

    if dry_run:
        print(f"    [DRY-RUN] #{clip.clip_num:03d} {clip.label:<25s} "
              f"t={clip.t_start_s:.1f}-{clip.t_end_s:.1f}s ({clip.duration_s:.1f}s) -> {out_path.name}")
        return True

    try:
        subprocess.run(cmd, capture_output=True, check=True, text=True)
        size_kb = out_path.stat().st_size / 1024
        print(f"    OK #{clip.clip_num:03d} {clip.label:<25s} "
              f"({clip.duration_s:.1f}s, {size_kb:.0f} KB)")
        return True
    except FileNotFoundError:
        print("    ERROR: ffmpeg not found. Install ffmpeg first.")
        return False
    except subprocess.CalledProcessError as exc:
        print(f"    FAIL #{clip.clip_num:03d} {clip.label}: {exc.stderr[:200] if exc.stderr else 'unknown error'}")
        return False


# ---------------------------------------------------------------------------
# Per-game processing
# ---------------------------------------------------------------------------

def process_game(
    game_dir: Path,
    *,
    dry_run: bool,
    overwrite: bool,
) -> dict:
    """Bootstrap clips for one game.  Returns stats dict."""
    game_folder = game_dir.name
    clips, source = load_game_clips(game_dir)
    if not clips:
        return {"game": game_folder, "skip_reason": "no clip index"}

    master = find_master(game_dir)
    on_disk = load_existing_clips(game_folder)
    in_index = load_indexed_clips(game_folder)

    # Determine what's missing
    need_extract = []
    for clip in clips:
        if not overwrite and clip.clip_num in on_disk:
            continue
        need_extract.append(clip)

    stats = {
        "game": game_folder,
        "source": source,
        "expected": len(clips),
        "on_disk": len(on_disk),
        "in_index": len(in_index),
        "need_extract": len(need_extract),
        "master_found": master is not None,
        "extracted": 0,
        "failed": 0,
    }

    if not need_extract:
        return stats

    print(f"\n  {game_folder}  ({source}, {len(need_extract)} clips to extract)")

    if not master:
        print(f"    MASTER NOT FOUND — place MASTER.mp4 in {game_dir}")
        for clip in need_extract:
            print(f"    PENDING #{clip.clip_num:03d} {clip.label:<25s} "
                  f"t={clip.t_start_s:.1f}-{clip.t_end_s:.1f}s")
        stats["failed"] = len(need_extract)
        return stats

    print(f"    Master: {master.name}")
    for clip in need_extract:
        out_name = clip.canonical_name(game_folder)
        out_path = ATOMIC_DIR / game_folder / out_name
        ok = extract_clip(master, out_path, clip, dry_run=dry_run)
        if ok:
            stats["extracted"] += 1
        else:
            stats["failed"] += 1

    return stats


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report() -> None:
    """Show bootstrap status for every game."""
    print()
    print("=" * 78)
    print("  CLIP BOOTSTRAP STATUS")
    print("=" * 78)
    print()

    total_expected = 0
    total_on_disk = 0
    total_need = 0

    for game_dir in sorted(GAMES_DIR.iterdir()):
        if not game_dir.is_dir():
            continue
        game_folder = game_dir.name
        clips, source = load_game_clips(game_dir)
        if not clips:
            print(f"  {game_folder}: no clip index found")
            continue

        on_disk = load_existing_clips(game_folder)
        in_index = load_indexed_clips(game_folder)
        master = find_master(game_dir)

        missing = [c for c in clips if c.clip_num not in on_disk]
        total_expected += len(clips)
        total_on_disk += len(on_disk)
        total_need += len(missing)

        status = "READY" if not missing else f"NEED {len(missing)} CLIPS"
        master_tag = "master found" if master else "NO MASTER"

        print(f"  {game_folder}")
        print(f"    {source}: {len(clips)} clips | on disk: {len(on_disk)} | "
              f"indexed: {len(in_index)} | {master_tag}")
        print(f"    Status: {status}")

        if missing and len(missing) <= 8:
            for c in missing:
                print(f"      - #{c.clip_num:03d} {c.label} "
                      f"t={c.t_start_s:.1f}-{c.t_end_s:.1f}s ({c.duration_s:.1f}s)")
        elif missing:
            print(f"      {len(missing)} clips pending extraction")
        print()

    print("-" * 78)
    print(f"  Total: {total_expected} expected | {total_on_disk} on disk | "
          f"{total_need} need extraction")
    print()
    if total_need > 0:
        print("  Next steps:")
        print("    1. Place MASTER.mp4 in each game folder that needs extraction")
        print("    2. Run: python tools/bootstrap_clips.py --all")
        print("    3. Run: python tools/catalog.py --rebuild-atomic-index")
        print("    4. Run: python tools/batch_pipeline.py --preset cinematic --upscale")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--report", action="store_true", help="Show bootstrap status")
    group.add_argument("--all", action="store_true", help="Extract all missing clips")
    group.add_argument("--game", help="Extract clips for one game (substring match)")

    p.add_argument("--dry-run", action="store_true", help="Preview without extraction")
    p.add_argument("--overwrite", action="store_true", help="Re-extract existing clips")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.report:
        print_report()
        return 0

    print()
    print("=" * 78)
    print("  BOOTSTRAP CLIP EXTRACTION")
    print("=" * 78)

    all_stats = []
    for game_dir in sorted(GAMES_DIR.iterdir()):
        if not game_dir.is_dir():
            continue
        if args.game and args.game.upper() not in game_dir.name.upper():
            continue
        stats = process_game(game_dir, dry_run=args.dry_run, overwrite=args.overwrite)
        all_stats.append(stats)

    # Summary
    total_extracted = sum(s.get("extracted", 0) for s in all_stats)
    total_failed = sum(s.get("failed", 0) for s in all_stats)
    total_need = sum(s.get("need_extract", 0) for s in all_stats)

    print()
    print("-" * 78)
    tag = "[DRY-RUN] " if args.dry_run else ""
    print(f"  {tag}Extracted: {total_extracted} | Failed: {total_failed} | "
          f"Total needed: {total_need}")
    print()

    if total_extracted > 0 and not args.dry_run:
        print("  Next steps:")
        print("    1. python tools/catalog.py --rebuild-atomic-index")
        print("    2. python tools/batch_pipeline.py --preset cinematic --upscale")

    return 1 if total_failed else 0


if __name__ == "__main__":
    sys.exit(main())

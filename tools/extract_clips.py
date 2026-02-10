"""Extract atomic clips from master game videos using catalog timestamps.

Usage:
    python tools/extract_clips.py --master path/to/MASTER.mp4 --game "2025-09-13__TSC_vs_NEOFC"
    python tools/extract_clips.py --master-dir out/games/ --all
    python tools/extract_clips.py --report

This script reads the atomic_index.csv and sidecar JSONs to determine which
clips need to be extracted, then uses ffmpeg to cut them from master videos.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
CATALOG_DIR = ROOT / "out" / "catalog"
ATOMIC_INDEX = CATALOG_DIR / "atomic_index.csv"
SIDECAR_DIR = CATALOG_DIR / "sidecar"
ATOMIC_OUT = ROOT / "out" / "atomic_clips"


def _load_index() -> List[dict]:
    if not ATOMIC_INDEX.exists():
        print(f"Error: {ATOMIC_INDEX} not found. Run catalog.py --rebuild-atomic-index first.")
        sys.exit(1)
    with ATOMIC_INDEX.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_sidecar(stem: str) -> dict:
    path = SIDECAR_DIR / f"{stem}.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _game_from_rel(clip_rel: str) -> str:
    """Extract game folder name from clip_rel like 'out/atomic_clips/2025-09-13__TSC/clip.mp4'."""
    parts = clip_rel.replace("\\", "/").split("/")
    # Expected: out/atomic_clips/GAME_NAME/clip.mp4  or  atomic_clips/GAME_NAME/clip.mp4
    for i, p in enumerate(parts):
        if p == "atomic_clips" and i + 1 < len(parts):
            return parts[i + 1]
    return ""


def _resolve_master(row: dict, master_dir: Optional[Path], explicit_master: Optional[Path]) -> Optional[Path]:
    """Find the master video for a clip row."""
    if explicit_master and explicit_master.exists():
        return explicit_master

    master_rel = row.get("master_rel", "")
    if master_rel:
        candidate = ROOT / master_rel
        if candidate.exists():
            return candidate

    if master_dir:
        game = _game_from_rel(row.get("clip_rel", ""))
        if game:
            for name in ["MASTER.mp4", f"full_game_{game}.mp4"]:
                candidate = master_dir / game / name
                if candidate.exists():
                    return candidate
            # Try loose match
            for mp4 in master_dir.rglob("MASTER.mp4"):
                if game.lower() in mp4.parent.name.lower():
                    return mp4

    return None


def _clip_exists(row: dict) -> bool:
    """Check if the atomic clip already exists on disk."""
    clip_rel = row.get("clip_rel", "")
    if clip_rel:
        candidate = ROOT / clip_rel
        if candidate.exists() and candidate.stat().st_size > 1000:
            return True
    return False


def _extract_clip(master: Path, out_path: Path, t_start: float, t_end: float, *, dry_run: bool = False) -> bool:
    """Extract a clip from master video using ffmpeg."""
    duration = t_end - t_start
    if duration <= 0:
        print(f"  SKIP: invalid duration {duration:.2f}s for {out_path.name}")
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{t_start:.3f}",
        "-i", str(master),
        "-t", f"{duration:.3f}",
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        str(out_path),
    ]

    if dry_run:
        print(f"  [DRY-RUN] ffmpeg -ss {t_start:.1f} -i {master.name} -t {duration:.1f} -> {out_path.name}")
        return True

    try:
        subprocess.run(cmd, capture_output=True, check=True, text=True)
        size_kb = out_path.stat().st_size / 1024
        print(f"  EXTRACTED: {out_path.name} ({duration:.1f}s, {size_kb:.0f} KB)")
        return True
    except FileNotFoundError:
        print("  ERROR: ffmpeg not found. Install ffmpeg to extract clips.")
        return False
    except subprocess.CalledProcessError as exc:
        print(f"  ERROR: ffmpeg failed for {out_path.name}: {exc.stderr[:200]}")
        return False


def report() -> None:
    """Print a summary of clip status by game."""
    rows = _load_index()

    games: Dict[str, dict] = {}
    for row in rows:
        game = _game_from_rel(row.get("clip_rel", ""))
        if not game:
            continue
        if game not in games:
            games[game] = {
                "total": 0, "on_disk": 0, "missing": 0,
                "master_rel": row.get("master_rel", ""),
                "master_exists": False,
                "events": {},
            }
        g = games[game]
        g["total"] += 1

        if _clip_exists(row):
            g["on_disk"] += 1
        else:
            g["missing"] += 1

        # Count events
        name = row.get("clip_name", "").upper()
        for tag in ["GOAL", "SHOT", "BUILD", "SAVE", "DRIBBL", "PRESSURE"]:
            if tag in name:
                g["events"][tag] = g["events"].get(tag, 0) + 1
                break
        else:
            g["events"]["OTHER"] = g["events"].get("OTHER", 0) + 1

        # Check master
        master_rel = row.get("master_rel", "")
        if master_rel:
            g["master_rel"] = master_rel
            if (ROOT / master_rel).exists():
                g["master_exists"] = True

    print("=" * 80)
    print("ATOMIC CLIPS RECOVERY REPORT")
    print("=" * 80)
    total_clips = sum(g["total"] for g in games.values())
    total_on_disk = sum(g["on_disk"] for g in games.values())
    total_missing = sum(g["missing"] for g in games.values())
    print(f"\nTotal: {total_clips} clips | On disk: {total_on_disk} | Missing: {total_missing}")
    print()

    for game in sorted(games.keys()):
        g = games[game]
        status = "READY" if g["on_disk"] == g["total"] else "NEEDS EXTRACTION"
        master_status = "FOUND" if g["master_exists"] else "MISSING"
        print(f"  {game}")
        print(f"    Clips: {g['total']} ({g['on_disk']} on disk, {g['missing']} missing)")
        print(f"    Events: {', '.join(f'{v} {k}' for k, v in sorted(g['events'].items(), key=lambda x: -x[1]))}")
        print(f"    Master: {g['master_rel'] or 'unknown'} [{master_status}]")
        print(f"    Status: {status}")
        print()

    if total_missing > 0:
        print("TO RECOVER MISSING CLIPS:")
        print("  1. Copy master game videos to out/games/<game_name>/MASTER.mp4")
        print("  2. Run: python tools/extract_clips.py --master-dir out/games/ --all")
        print("  Or per-game: python tools/extract_clips.py --master out/games/<game>/MASTER.mp4 --game <game>")


def extract(
    *,
    game_filter: Optional[str] = None,
    master: Optional[Path] = None,
    master_dir: Optional[Path] = None,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict:
    """Extract missing clips from master videos."""
    rows = _load_index()
    extracted = 0
    skipped = 0
    failed = 0
    no_master = 0

    for row in rows:
        game = _game_from_rel(row.get("clip_rel", ""))
        if game_filter and game != game_filter:
            continue

        clip_rel = row.get("clip_rel", "")
        if not clip_rel:
            continue
        out_path = ROOT / clip_rel

        if _clip_exists(row) and not overwrite:
            skipped += 1
            continue

        t_start = row.get("t_start_s", "").strip()
        t_end = row.get("t_end_s", "").strip()
        if not t_start or not t_end:
            # Try sidecar
            sc = _load_sidecar(row.get("clip_stem", ""))
            t_start = t_start or str(sc.get("t_start_s", ""))
            t_end = t_end or str(sc.get("t_end_s", ""))

        if not t_start or not t_end:
            print(f"  SKIP: no timestamps for {row.get('clip_name', '?')}")
            failed += 1
            continue

        master_path = _resolve_master(row, master_dir, master)
        if not master_path:
            print(f"  NO MASTER: {row.get('clip_name', '?')} (need {row.get('master_rel', '?')})")
            no_master += 1
            continue

        ok = _extract_clip(master_path, out_path, float(t_start), float(t_end), dry_run=dry_run)
        if ok:
            extracted += 1
        else:
            failed += 1

    return {
        "extracted": extracted,
        "skipped": skipped,
        "failed": failed,
        "no_master": no_master,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--report", action="store_true", help="Print recovery status report")
    group.add_argument("--all", action="store_true", help="Extract all missing clips")
    group.add_argument("--game", help="Extract clips for a specific game (e.g. '2025-09-13__TSC_vs_NEOFC')")

    parser.add_argument("--master", type=Path, help="Path to a specific master video")
    parser.add_argument("--master-dir", type=Path, help="Directory containing game folders with MASTER.mp4 files")
    parser.add_argument("--dry-run", action="store_true", help="Preview extraction commands without running them")
    parser.add_argument("--overwrite", action="store_true", help="Re-extract even if clip already exists")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.report:
        report()
        return 0

    result = extract(
        game_filter=args.game,
        master=args.master,
        master_dir=args.master_dir or (ROOT / "out" / "games"),
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )

    print(f"\nResults: {result['extracted']} extracted, {result['skipped']} skipped (already exist), "
          f"{result['failed']} failed, {result['no_master']} missing master")

    return 1 if result["failed"] or result["no_master"] else 0


if __name__ == "__main__":
    sys.exit(main())

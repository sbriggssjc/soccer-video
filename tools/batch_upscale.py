#!/usr/bin/env python
"""Batch upscale portrait renders that haven't been upscaled yet.

Reads pipeline_status.csv to find clips with a portrait render but no upscale,
then runs upscale_video() on each portrait output.

This is a standalone complement to batch_pipeline.py --upscale, which only
upscales clips it renders in the same run.  batch_upscale.py handles the
common case where clips were rendered previously and just need the upscale
pass applied.

Usage:
    python tools/batch_upscale.py --dry-run          # preview what would run
    python tools/batch_upscale.py                     # upscale all pending
    python tools/batch_upscale.py --game SLSG         # one game only
    python tools/batch_upscale.py --limit 5           # test with 5 clips
    python tools/batch_upscale.py --method realesrgan # use AI upscaler
    python tools/batch_upscale.py --remap "C:/Users/scott/OneDrive/SoccerVideoMedia=D:/Projects/soccer-video"
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PIPELINE_STATUS_PATH = REPO_ROOT / "out" / "catalog" / "pipeline_status.csv"
SIDECAR_DIR = REPO_ROOT / "out" / "catalog" / "sidecar"

# Tags that identify rendering-variant or test rows to skip
_VARIANT_TAGS = ("__CINEMATIC", "__DEBUG", "__OVERLAY", "portrait_POST",
                 "portrait_FINAL", "nonexistent")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch upscale portrait renders missing upscale output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be upscaled without processing")
    p.add_argument("--game", help="Filter to a specific game (substring match)")
    p.add_argument("--limit", type=int, default=0,
                   help="Max clips to process (0 = no limit)")
    p.add_argument("--scale", type=int, default=2,
                   help="Upscale factor (default: 2)")
    p.add_argument("--method", default="lanczos",
                   choices=["lanczos", "realesrgan"],
                   help="Upscale method (default: lanczos)")
    p.add_argument("--force", action="store_true",
                   help="Re-upscale even if output already exists")
    p.add_argument("--remap", metavar="OLD=NEW",
                   help="Remap portrait path prefix (e.g. 'C:/old/path=D:/new/path')")
    return p.parse_args(argv)


def _parse_remap(remap_arg: str | None) -> tuple[str, str] | None:
    if not remap_arg:
        return None
    if "=" not in remap_arg:
        print(f"[ERROR] --remap must be OLD=NEW, got: {remap_arg!r}")
        return None
    old, new = remap_arg.split("=", 1)
    return old.replace("\\", "/"), new.replace("\\", "/")


def _apply_remap(path: str, remap: tuple[str, str] | None) -> str:
    if not remap:
        return path
    old_prefix, new_prefix = remap
    normalized = path.replace("\\", "/")
    if normalized.startswith(old_prefix):
        return new_prefix + normalized[len(old_prefix):]
    return path


def _portrait_stem(portrait_path: str) -> str:
    """Extract portrait filename stem, normalizing Windows backslashes."""
    fname = portrait_path.replace("\\", "/").rsplit("/", 1)[-1]
    return fname.rsplit(".", 1)[0] if "." in fname else fname


def load_upscale_work(args: argparse.Namespace) -> list[dict]:
    """Load clips needing upscale from pipeline_status.csv.

    Two-pass approach:
      1. Collect all portrait stems that already have upscale (from ANY entry).
      2. Build work list from entries without upscale, deduplicating by stem.

    This handles the common case where the same clip has pipeline entries
    from both old (OneDrive) and new (D:) paths — if either entry has
    upscale recorded, the portrait is considered done.
    """
    if not PIPELINE_STATUS_PATH.exists():
        print(f"[ERROR] Pipeline status not found: {PIPELINE_STATUS_PATH}")
        return []

    remap = _parse_remap(args.remap)

    # Pass 1: read all rows, identify which portrait stems are already upscaled
    all_rows: list[dict] = []
    upscaled_stems: set[str] = set()

    with PIPELINE_STATUS_PATH.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cp = row.get("clip_path", "")
            if any(tag in cp for tag in _VARIANT_TAGS):
                continue
            portrait = row.get("portrait_path", "").strip()
            if not portrait:
                continue
            all_rows.append(row)
            if row.get("upscaled_path", "").strip():
                upscaled_stems.add(_portrait_stem(portrait))

    # Pass 2: build work list, dedup by portrait stem, skip already-upscaled
    work = []
    seen_stems: set[str] = set()

    for row in all_rows:
        cp = row.get("clip_path", "")
        portrait = row.get("portrait_path", "").strip()
        stem = _portrait_stem(portrait)

        # Skip if this portrait is already upscaled (unless --force)
        if not args.force and stem in upscaled_stems:
            continue

        # Skip already-seen stems (dedup old vs new path entries)
        if stem in seen_stems:
            continue

        # Apply game filter
        if args.game and args.game.lower() not in cp.lower():
            continue

        seen_stems.add(stem)

        # Apply path remap
        portrait_remapped = _apply_remap(portrait, remap)
        clip_remapped = _apply_remap(cp, remap)

        work.append({
            "clip_path": clip_remapped,
            "portrait_path": portrait_remapped,
            "clip_name": stem.replace("__CINEMATIC_portrait_FINAL", ""),
            "game": _extract_game(cp),
        })

    # Pass 3: scan sidecars for clips with portrait render done but not in
    # the pipeline-based work list (handles clips rendered on-disk without
    # pipeline_status tracking, e.g. Greenwood/RVFC reconciled renders).
    if SIDECAR_DIR.exists():
        import json
        for f in sorted(SIDECAR_DIR.iterdir()):
            if f.suffix != ".json":
                continue
            sc_stem = f.stem
            if any(tag in sc_stem for tag in _VARIANT_TAGS):
                continue
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            steps = data.get("steps", {})
            pr = steps.get("portrait_render", {})
            up = steps.get("upscale", {})
            if not pr.get("done"):
                continue
            if not args.force and up.get("done"):
                continue
            portrait_out = pr.get("out", "")
            if not portrait_out:
                continue
            p_stem = _portrait_stem(portrait_out)
            if p_stem in seen_stems:
                continue
            # Apply game filter
            cp = data.get("clip_path", "")
            if args.game and args.game.lower() not in cp.lower():
                continue
            seen_stems.add(p_stem)
            portrait_remapped = _apply_remap(portrait_out, remap)
            work.append({
                "clip_path": _apply_remap(cp, remap),
                "portrait_path": portrait_remapped,
                "clip_name": p_stem.replace("__CINEMATIC_portrait_FINAL", ""),
                "game": _extract_game(cp),
            })

    return work


def _extract_game(clip_path: str) -> str:
    """Extract the game folder name from a clip path."""
    parts = clip_path.replace("\\", "/").split("/")
    for i, part in enumerate(parts):
        if part == "atomic_clips" and i + 1 < len(parts):
            return parts[i + 1]
    return ""


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    work = load_upscale_work(args)

    if args.limit and args.limit > 0:
        work = work[:args.limit]

    print()
    print("=" * 60)
    print("  BATCH UPSCALE — Portrait Renders")
    print("=" * 60)
    print(f"  Method:      {args.method} {args.scale}x")
    print(f"  To upscale:  {len(work)}")
    if args.game:
        print(f"  Game filter: {args.game}")
    if args.remap:
        print(f"  Path remap:  {args.remap}")
    print("=" * 60)
    print()

    if not work:
        print("[OK] Nothing to upscale — all portrait renders already have upscale output.")
        return 0

    if args.dry_run:
        print("[DRY-RUN] Would upscale:")
        by_game: dict[str, list] = {}
        for item in work:
            by_game.setdefault(item["game"], []).append(item)
        for game in sorted(by_game):
            clips = by_game[game]
            print(f"\n  {game} ({len(clips)} clips):")
            for item in clips[:5]:
                print(f"    - {item['clip_name']}")
            if len(clips) > 5:
                print(f"    ... and {len(clips) - 5} more")
        return 0

    # Import upscale here so dry-run works without ffmpeg
    from tools.upscale import upscale_video

    ok = 0
    failed = 0
    skipped = 0
    t_start = time.monotonic()

    for i, item in enumerate(work, 1):
        portrait_path = Path(item["portrait_path"])
        elapsed = time.monotonic() - t_start
        eta = ""
        if i > 1 and (ok + failed) > 0:
            avg = elapsed / (ok + failed)
            remaining = avg * (len(work) - i + 1)
            eta = f"  ETA: {int(remaining // 60)}m{int(remaining % 60):02d}s"

        print(f"\n[{i}/{len(work)}] {item['clip_name']}{eta}")

        if not portrait_path.exists():
            print(f"  [SKIP] Portrait not found: {portrait_path}")
            skipped += 1
            continue

        try:
            result = upscale_video(
                str(portrait_path),
                scale=args.scale,
                method=args.method,
                force=args.force,
                track=True,
            )
            print(f"  [OK] {Path(result).name}")
            ok += 1
        except Exception as exc:
            print(f"  [FAIL] {exc}")
            failed += 1

    elapsed = time.monotonic() - t_start
    print()
    print("=" * 60)
    print("  BATCH UPSCALE COMPLETE")
    print("=" * 60)
    print(f"  Upscaled:  {ok}")
    print(f"  Skipped:   {skipped}")
    print(f"  Failed:    {failed}")
    print(f"  Time:      {int(elapsed // 60)}m{int(elapsed % 60):02d}s")
    print("=" * 60)
    print()

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())

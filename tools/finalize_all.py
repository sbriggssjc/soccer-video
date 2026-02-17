#!/usr/bin/env python
"""Finalize all games: bootstrap missing clips, rebuild catalog, and render.

Orchestrates the full pipeline in the correct order:

  Tier 1  Re-render fully-rendered games with --upscale --force
  Tier 2  Bootstrap missing clips from MASTER.mp4 via game indexes
  Tier 3  Rebuild atomic index so new clips are cataloged
  Tier 4  Render all unprocessed clips with upscale

Usage:
    python tools/finalize_all.py --dry-run          # preview everything
    python tools/finalize_all.py                     # run full pipeline
    python tools/finalize_all.py --game SLSG         # one game only
    python tools/finalize_all.py --skip-render       # bootstrap + catalog only
    python tools/finalize_all.py --tier 1            # only re-render existing
    python tools/finalize_all.py --tier 2            # only bootstrap clips
    python tools/finalize_all.py --tier 3            # only rebuild catalog
    python tools/finalize_all.py --tier 4            # only render unprocessed
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

# Games that are fully rendered but missing the upscale step (Tier 1)
TIER1_GAMES = [
    "Route_66",
    "Arkansas",
    "2025-10-14__TSC_vs_BASC",
    "OK_Sporting",
    "2025-11-23__TSC_vs_BASC",
]


def run(cmd: list[str], *, dry_run: bool, label: str) -> int:
    """Run a command, or print it in dry-run mode."""
    if dry_run:
        print(f"  [DRY-RUN] {' '.join(cmd)}")
        return 0
    print(f"  >> {label}")
    print(f"     {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"  !! Exited with code {result.returncode}")
    return result.returncode


def tier1_rerender_with_upscale(*, dry_run: bool, game_filter: str | None) -> None:
    """Re-render fully-rendered games with upscale (fixes blurry output)."""
    print()
    print("=" * 78)
    print("  TIER 1: Re-render completed games with 2x upscale")
    print("=" * 78)

    games = TIER1_GAMES
    if game_filter:
        games = [g for g in games if game_filter.upper() in g.upper()]

    if not games:
        print("  (no Tier 1 games match filter)")
        return

    for game in games:
        print(f"\n  --- {game} ---")
        cmd = [
            PYTHON, "tools/batch_pipeline.py",
            "--preset", "cinematic",
            "--upscale",
            "--force",
            "--game", game,
        ]
        run(cmd, dry_run=dry_run, label=f"Re-render {game} with upscale")


def tier2_bootstrap_clips(*, dry_run: bool, game_filter: str | None) -> None:
    """Extract missing clips from MASTER.mp4 using game indexes."""
    print()
    print("=" * 78)
    print("  TIER 2: Bootstrap missing clips from masters")
    print("=" * 78)

    cmd = [PYTHON, "tools/bootstrap_clips.py"]
    if game_filter:
        cmd += ["--game", game_filter]
    else:
        cmd += ["--all"]
    if dry_run:
        cmd += ["--dry-run"]

    run(cmd, dry_run=False, label="Bootstrap clips")  # bootstrap handles its own dry-run


def tier3_rebuild_catalog(*, dry_run: bool) -> None:
    """Rebuild atomic_index.csv to pick up new clips."""
    print()
    print("=" * 78)
    print("  TIER 3: Rebuild atomic index")
    print("=" * 78)

    cmd = [PYTHON, "tools/catalog.py", "--rebuild-atomic-index"]
    run(cmd, dry_run=dry_run, label="Rebuild catalog")


def tier4_render_all(*, dry_run: bool, game_filter: str | None) -> None:
    """Render all unprocessed clips with cinematic preset + upscale."""
    print()
    print("=" * 78)
    print("  TIER 4: Render all unprocessed clips with upscale")
    print("=" * 78)

    cmd = [
        PYTHON, "tools/batch_pipeline.py",
        "--preset", "cinematic",
        "--upscale",
    ]
    if game_filter:
        cmd += ["--game", game_filter]
    if dry_run:
        cmd += ["--dry-run"]

    run(cmd, dry_run=False, label="Batch render with upscale")


def tier_audit(*, game_filter: str | None) -> None:
    """Run the audit to show final status."""
    print()
    print("=" * 78)
    print("  FINAL AUDIT")
    print("=" * 78)

    cmd = [PYTHON, "tools/audit_pipeline.py"]
    if game_filter:
        cmd += ["--game", game_filter]

    run(cmd, dry_run=False, label="Pipeline audit")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dry-run", action="store_true",
                    help="Preview all commands without executing")
    p.add_argument("--game", help="Filter to one game (substring match)")
    p.add_argument("--tier", type=int, choices=[1, 2, 3, 4],
                    help="Run only a specific tier")
    p.add_argument("--skip-render", action="store_true",
                    help="Skip rendering tiers (bootstrap + catalog only)")
    p.add_argument("--skip-audit", action="store_true",
                    help="Skip the final audit report")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    print()
    print("#" * 78)
    print("#  FINALIZE ALL GAMES — Soccer Video Pipeline")
    print("#" * 78)

    tiers_to_run = {1, 2, 3, 4} if args.tier is None else {args.tier}

    if args.skip_render:
        tiers_to_run -= {1, 4}

    if 1 in tiers_to_run:
        tier1_rerender_with_upscale(dry_run=args.dry_run, game_filter=args.game)

    if 2 in tiers_to_run:
        tier2_bootstrap_clips(dry_run=args.dry_run, game_filter=args.game)

    if 3 in tiers_to_run:
        tier3_rebuild_catalog(dry_run=args.dry_run)

    if 4 in tiers_to_run:
        tier4_render_all(dry_run=args.dry_run, game_filter=args.game)

    if not args.skip_audit:
        tier_audit(game_filter=args.game)

    print()
    print("#" * 78)
    print("#  DONE")
    print("#" * 78)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

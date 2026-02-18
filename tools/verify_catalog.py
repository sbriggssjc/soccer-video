#!/usr/bin/env python
"""Verify that every clip in every game's authoritative index is cataloged.

Cross-checks each game's plays_manual.csv / clip_index.csv against
atomic_index.csv, printing a per-game pass/fail report.

Usage:
    python tools/verify_catalog.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

# Re-use the audit module's loading and matching logic
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_pipeline import (
    GAMES_DIR,
    find_matching_clip,
    load_atomic_index,
    load_expected_clips,
)


def main() -> int:
    if not GAMES_DIR.exists():
        print(f"Games directory not found: {GAMES_DIR}", file=sys.stderr)
        return 1

    atomic_by_master = load_atomic_index()
    all_ok = True
    total_expected = 0
    total_matched = 0

    print()
    print("=" * 70)
    print("  CATALOG VERIFICATION — All Games")
    print("=" * 70)
    print()

    for game_dir in sorted(GAMES_DIR.iterdir()):
        if not game_dir.is_dir():
            continue

        expected, source_file = load_expected_clips(game_dir)
        if not expected:
            continue

        game_folder = game_dir.name
        master_rel = f"out/games/{game_folder}/MASTER.mp4"
        indexed = atomic_by_master.get(master_rel, [])

        matched = 0
        missing = []
        for clip in expected:
            match = find_matching_clip(clip, indexed)
            if match:
                matched += 1
            else:
                missing.append(clip)

        n = len(expected)
        total_expected += n
        total_matched += matched
        status = "PASS" if matched == n else "FAIL"
        if status == "FAIL":
            all_ok = False

        print(f"  [{status}] {game_folder}")
        print(f"         Source: {source_file}  |  {matched}/{n} matched")
        if missing:
            for clip in missing:
                print(
                    f"         MISSING: #{clip.clip_num:03d} {clip.label} "
                    f"t={clip.start_s:.0f}-{clip.end_s:.0f}s"
                )
        print()

    print("=" * 70)
    overall = "ALL CATALOGED" if all_ok else "GAPS FOUND"
    print(f"  {overall}: {total_matched}/{total_expected} clips verified")
    print("=" * 70)
    print()
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

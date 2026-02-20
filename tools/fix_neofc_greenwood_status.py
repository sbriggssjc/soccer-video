"""One-shot repair for NEOFC and Greenwood clip entries.

Fixes two problems found in the pipeline_status.csv and atomic_index.csv:

1. NEOFC clips — stacked ``.__CINEMATIC`` tags in pipeline_status clip_path
   entries (e.g. ``....__CINEMATIC.__CINEMATIC.__CINEMATIC.mp4``).  These rows
   are reset so the pipeline can re-discover and re-process them cleanly.

2. Greenwood clips 002–013 — filenames embed wrong HH:MM:SS-derived timestamps
   (e.g. ``t20400.00`` instead of ``t340.00``).  The ``t_start_s`` / ``t_end_s``
   columns in atomic_index are already correct; this script rebuilds the
   ``clip_name``, ``clip_path``, and ``clip_rel`` to match.

Usage:
    python tools/fix_neofc_greenwood_status.py [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_STATUS = ROOT / "out" / "catalog" / "pipeline_status.csv"
ATOMIC_INDEX = ROOT / "out" / "catalog" / "atomic_index.csv"

# ── Greenwood MM:SS:00 → seconds corrections ────────────────────────────
# plays_manual.csv times that were mis-parsed as HH:MM:SS instead of MM:SS:00.
# Map: clip_id → (correct_t_start, correct_t_end)  (already in atomic_index t_start_s/t_end_s)
# These clip ids had wrong timestamps baked into their filenames.
_GW_BAD_CLIP_IDS = {f"{i:03d}" for i in range(2, 14)}  # 002 .. 013


def _strip_stacked_cinematic(path_str: str) -> str:
    """Remove stacked .__CINEMATIC and _portrait_FINAL / _portrait_POST tags."""
    # Strip all occurrences of .__CINEMATIC (dot-prefixed)
    cleaned = re.sub(r"(?:\.__CINEMATIC)+", "", path_str, flags=re.IGNORECASE)
    # Strip .__DEBUG_FINAL
    cleaned = re.sub(r"\.__DEBUG_FINAL", "", cleaned, flags=re.IGNORECASE)
    # Strip _portrait_FINAL / _portrait_POST suffixes (may be stacked)
    cleaned = re.sub(r"(?:_portrait_(?:FINAL|POST))+", "", cleaned, flags=re.IGNORECASE)
    return cleaned


def fix_pipeline_status(*, dry_run: bool = False) -> int:
    """Reset NEOFC rows with stacked tags and Greenwood rows with wrong filenames."""
    if not PIPELINE_STATUS.exists():
        print(f"SKIP: {PIPELINE_STATUS} not found")
        return 0

    with PIPELINE_STATUS.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    changed = 0
    for row in rows:
        clip_path = row.get("clip_path", "")

        # ── Any clip with stacked .__CINEMATIC tags ──
        if ".__CINEMATIC" in clip_path:
            cleaned = _strip_stacked_cinematic(clip_path)
            if cleaned != clip_path:
                if dry_run:
                    print(f"  [DRY] strip tags: {Path(clip_path).name}")
                    print(f"    -> {Path(cleaned).name}")
                row["clip_path"] = cleaned
                row["portrait_path"] = ""
                row["render_done_at"] = ""
                row["last_error"] = ""
                changed += 1

        # ── Greenwood wrong timestamps in filename ──
        if "Greenwood" in clip_path:
            # Check for the bad HH:MM:SS timestamps (values > 10000)
            m = re.search(r"__t(\d+(?:\.\d+)?)-t(\d+(?:\.\d+)?)", clip_path)
            if m:
                t_start_in_name = float(m.group(1))
                if t_start_in_name > 3000:  # > master duration → bad parse
                    cleaned = _strip_stacked_cinematic(clip_path)
                    if cleaned != clip_path:
                        row["clip_path"] = cleaned
                    row["portrait_path"] = ""
                    row["render_done_at"] = ""
                    row["last_error"] = ""
                    if dry_run:
                        print(f"  [DRY] Greenwood reset: {Path(clip_path).name}")
                    changed += 1

            # Also clear "source not found" errors on Greenwood clips so they retry
            if row.get("last_error") == "source not found":
                row["last_error"] = ""
                changed += 1
                if dry_run:
                    print(f"  [DRY] Greenwood clear error: {Path(clip_path).name}")

    if changed and not dry_run:
        with PIPELINE_STATUS.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"pipeline_status.csv: {changed} rows updated" + (" (dry-run)" if dry_run else ""))
    return changed


def _rebuild_greenwood_filename(
    old_name: str, clip_id: str, t_start: float, t_end: float
) -> str:
    """Rebuild a Greenwood filename replacing the wrong timestamps."""
    # Pattern: NNN__GAME__LABEL__tSTART-tEND[_portrait_POST].mp4
    # Use non-greedy match and anchor before a non-digit-or-dot character
    # to avoid consuming the '.' before '.mp4'.
    new_ts = f"t{t_start:.2f}-t{t_end:.2f}"
    result = re.sub(r"__t\d+(?:\.\d+)?-t\d+(?:\.\d+)?", f"__{new_ts}", old_name)
    return result


def fix_atomic_index(*, dry_run: bool = False) -> int:
    """Fix Greenwood clips 002-013 filenames in atomic_index.csv."""
    if not ATOMIC_INDEX.exists():
        print(f"SKIP: {ATOMIC_INDEX} not found")
        return 0

    with ATOMIC_INDEX.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    changed = 0
    for row in rows:
        # Check any column for Greenwood + bad timestamps
        row_text = row.get("clip_id", "") + row.get("clip_name", "")
        if "Greenwood" not in row_text:
            continue

        # Extract clip_id from clip_name or clip_id column
        clip_name = row.get("clip_name", "")
        m = re.match(r"^(\d{3})__", clip_name) or re.search(r":(\d{3})__", row.get("clip_id", ""))
        if not m:
            continue
        clip_id = m.group(1)
        if clip_id not in _GW_BAD_CLIP_IDS:
            continue

        t_start_s = row.get("t_start_s", "").strip()
        t_end_s = row.get("t_end_s", "").strip()
        if not t_start_s or not t_end_s:
            continue

        t_start = float(t_start_s)
        t_end = float(t_end_s)

        # Check if ANY column still has wrong timestamps (> 3000)
        needs_fix = False
        for col in ("clip_id", "clip_name", "clip_stem", "clip_path", "clip_rel"):
            fn_match = re.search(r"__t(\d+(?:\.\d+)?)-t", row.get(col, ""))
            if fn_match and float(fn_match.group(1)) > 3000:
                needs_fix = True
                break
        if not needs_fix:
            continue

        old_clip_name = row["clip_name"]
        new_clip_name = _rebuild_greenwood_filename(old_clip_name, clip_id, t_start, t_end)

        if dry_run:
            print(f"  [DRY] {clip_id}: {old_clip_name}")
            print(f"    -> {new_clip_name}")

        # Update all filename-bearing columns (including clip_id which embeds the name)
        for col in ("clip_id", "clip_name", "clip_stem"):
            if row.get(col):
                row[col] = _rebuild_greenwood_filename(row[col], clip_id, t_start, t_end)

        for col in ("clip_path", "clip_rel"):
            old_val = row.get(col, "")
            if old_val:
                # Replace the filename portion (last path component)
                old_fn = Path(old_val).name if col == "clip_path" else old_val.split("/")[-1]
                new_fn = _rebuild_greenwood_filename(old_fn, clip_id, t_start, t_end)
                row[col] = old_val.replace(old_fn, new_fn)

        changed += 1

    if changed and not dry_run:
        with ATOMIC_INDEX.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"atomic_index.csv: {changed} Greenwood clips updated" + (" (dry-run)" if dry_run else ""))
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    args = parser.parse_args()

    n1 = fix_pipeline_status(dry_run=args.dry_run)
    n2 = fix_atomic_index(dry_run=args.dry_run)
    total = n1 + n2
    print(f"\nTotal: {total} changes" + (" (dry-run)" if args.dry_run else ""))
    return 0 if total > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

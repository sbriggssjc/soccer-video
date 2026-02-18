#!/usr/bin/env python
"""Cross-game pipeline audit: compare expected clips with actual processing state.

Loads each game's authoritative clip list (clip_index.csv or plays_manual.csv),
cross-references with atomic_index.csv, pipeline_status.csv, and sidecar JSONs
to identify gaps in cataloging, upscaling, and portrait rendering.

Usage:
    python tools/audit_pipeline.py              # Full audit report
    python tools/audit_pipeline.py --game SLSG  # Audit one game (substring match)
    python tools/audit_pipeline.py --json       # Machine-readable output
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from timestamp_parse import parse_timestamp as parse_time_to_seconds
from timestamp_parse import parse_timestamp_pair as parse_time_pair

ROOT = Path(__file__).resolve().parents[1]
GAMES_DIR = ROOT / "out" / "games"
CATALOG_DIR = ROOT / "out" / "catalog"
SIDECAR_DIR = CATALOG_DIR / "sidecar"
ATOMIC_INDEX_PATH = CATALOG_DIR / "atomic_index.csv"
PIPELINE_STATUS_PATH = CATALOG_DIR / "pipeline_status.csv"
PORTRAIT_REELS_DIR = ROOT / "out" / "portrait_reels" / "clean"

# Regex to strip rendering-variant suffixes from portrait output filenames
_PORTRAIT_SUFFIX_RE = re.compile(
    r"__(?:CINEMATIC|DEBUG|OVERLAY)_portrait_(?:FINAL|POST)$"
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ExpectedClip:
    """A clip listed in a game's authoritative index."""
    clip_num: int
    label: str
    start_s: float
    end_s: float
    duration_s: float = 0.0

    def __post_init__(self):
        self.duration_s = round(self.end_s - self.start_s, 3)


@dataclass
class GameAudit:
    """Processing state for one game."""
    game_folder: str
    game_date: str
    expected_clips: list[ExpectedClip] = field(default_factory=list)
    in_atomic_index: int = 0
    has_sidecar: int = 0
    upscaled: int = 0
    portrait_rendered: int = 0
    errors: int = 0
    error_details: list[str] = field(default_factory=list)
    missing_from_index: list[str] = field(default_factory=list)
    missing_upscale: list[str] = field(default_factory=list)
    missing_portrait: list[str] = field(default_factory=list)
    index_source: str = ""  # which file the expected clips came from


# ---------------------------------------------------------------------------
# Load game clip indexes
# ---------------------------------------------------------------------------

def load_clip_index_csv(path: Path) -> list[ExpectedClip]:
    """Load a clip_index.csv (clip_num, description, start, end)."""
    clips = []
    with path.open("r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            num_raw = row.get("clip_num", "").strip()
            if not num_raw:
                continue
            try:
                num = int(num_raw)
            except ValueError:
                continue
            label = row.get("description", "").strip()
            start, end = parse_time_pair(row.get("start", ""), row.get("end", ""))
            if start is None or end is None:
                continue
            clips.append(ExpectedClip(
                clip_num=num,
                label=label.upper().replace(" ", "_").replace("&", "").replace("__", "_").strip("_"),
                start_s=round(start, 1),
                end_s=round(end, 1),
            ))
    return clips


def load_plays_manual_csv(path: Path) -> list[ExpectedClip]:
    """Load a plays_manual.csv (clip_id, label, master_start, master_end, notes)."""
    clips = []
    with path.open("r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            num_raw = row.get("clip_id", "").strip().strip('"')
            if not num_raw:
                continue
            try:
                num = int(num_raw)
            except ValueError:
                continue
            label = row.get("label", "").strip().strip('"')
            start, end = parse_time_pair(
                row.get("master_start", ""), row.get("master_end", ""),
            )
            if start is None or end is None:
                continue
            clips.append(ExpectedClip(
                clip_num=num,
                label=label.upper().replace(" ", "_").replace("&", "").replace("__", "_").strip("_"),
                start_s=round(start, 1),
                end_s=round(end, 1),
            ))
    return clips


def load_expected_clips(game_dir: Path) -> tuple[list[ExpectedClip], str]:
    """Load the authoritative clip list for a game directory."""
    plays = game_dir / "plays_manual.csv"
    clip_csv = game_dir / "clip_index.csv"
    # Prefer plays_manual.csv when both exist (it has more detail)
    if plays.exists():
        return load_plays_manual_csv(plays), "plays_manual.csv"
    if clip_csv.exists():
        return load_clip_index_csv(clip_csv), "clip_index.csv"
    return [], ""


# ---------------------------------------------------------------------------
# Load pipeline state
# ---------------------------------------------------------------------------

def load_atomic_index() -> dict[str, dict]:
    """Load atomic_index.csv keyed by master_rel -> list of clip rows."""
    if not ATOMIC_INDEX_PATH.exists():
        return {}
    rows = {}
    with ATOMIC_INDEX_PATH.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            master_rel = row.get("master_rel", "")
            rows.setdefault(master_rel, []).append(row)
    return rows


def load_pipeline_status() -> dict[str, dict]:
    """Load pipeline_status.csv keyed by clip_path."""
    if not PIPELINE_STATUS_PATH.exists():
        return {}
    result = {}
    with PIPELINE_STATUS_PATH.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cp = row.get("clip_path", "")
            if cp:
                result[cp] = row
    return result


def load_sidecars_for_game(game_key: str) -> dict[str, dict]:
    """Load all sidecar JSONs whose master path references the given game key."""
    result = {}
    if not SIDECAR_DIR.exists():
        return result
    for f in sorted(SIDECAR_DIR.iterdir()):
        if f.suffix != ".json":
            continue
        # Skip rendering variants
        stem = f.stem
        if any(tag in stem for tag in ("__CINEMATIC", "__DEBUG", "__OVERLAY", "portrait_POST", "portrait_FINAL")):
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        master = data.get("master_rel", "") or data.get("master_path", "")
        if game_key in master:
            result[stem] = data
    return result


def load_rendered_stems() -> set[str]:
    """Return clip stems that have rendered portrait outputs on disk."""
    if not PORTRAIT_REELS_DIR.exists():
        return set()
    stems: set[str] = set()
    for f in PORTRAIT_REELS_DIR.iterdir():
        if f.suffix.lower() not in {".mp4", ".mov"}:
            continue
        stem = _PORTRAIT_SUFFIX_RE.sub("", f.stem)
        stems.add(stem)
    return stems


# ---------------------------------------------------------------------------
# Matching logic
# ---------------------------------------------------------------------------

def timestamps_overlap(a_start: float, a_end: float, b_start: float, b_end: float,
                       tolerance: float = 5.0) -> bool:
    """Check if two timestamp ranges overlap or are within tolerance."""
    # Expand ranges by tolerance for matching
    return (a_start - tolerance) < b_end and (a_end + tolerance) > b_start


def find_matching_clip(expected: ExpectedClip, indexed_clips: list[dict]) -> Optional[dict]:
    """Find the best match for an expected clip in the atomic index."""
    best = None
    best_overlap = -1.0
    for row in indexed_clips:
        try:
            t0 = float(row.get("t_start_s", ""))
            t1 = float(row.get("t_end_s", ""))
        except (ValueError, TypeError):
            continue
        if timestamps_overlap(expected.start_s, expected.end_s, t0, t1):
            overlap = min(expected.end_s, t1) - max(expected.start_s, t0)
            if overlap > best_overlap:
                best_overlap = overlap
                best = row
    return best


def find_matching_sidecar(expected: ExpectedClip, sidecars: dict[str, dict]) -> Optional[dict]:
    """Find the best matching sidecar for an expected clip."""
    best = None
    best_overlap = -1.0
    for stem, data in sidecars.items():
        t0 = data.get("t_start_s")
        t1 = data.get("t_end_s")
        if t0 is None or t1 is None:
            continue
        if timestamps_overlap(expected.start_s, expected.end_s, t0, t1):
            overlap = min(expected.end_s, t1) - max(expected.start_s, t0)
            if overlap > best_overlap:
                best_overlap = overlap
                best = data
    return best


# ---------------------------------------------------------------------------
# Per-game audit
# ---------------------------------------------------------------------------

def audit_game(game_dir: Path, atomic_by_master: dict, pipeline_status: dict,
               rendered_stems: set[str] | None = None) -> Optional[GameAudit]:
    """Audit a single game directory against the pipeline state."""
    game_folder = game_dir.name
    # Extract date from folder name
    date_match = re.match(r"(\d{4}-\d{2}-\d{2})__", game_folder)
    game_date = date_match.group(1) if date_match else ""

    expected, source_file = load_expected_clips(game_dir)
    if not expected:
        return None

    # Determine master_rel key for this game
    master_rel = f"out/games/{game_folder}/MASTER.mp4"

    # Load indexed clips for this master
    indexed = atomic_by_master.get(master_rel, [])

    # Load sidecars for this game
    game_key = game_folder.split("__", 1)[1] if "__" in game_folder else game_folder
    sidecars = load_sidecars_for_game(game_key)

    audit = GameAudit(
        game_folder=game_folder,
        game_date=game_date,
        expected_clips=expected,
        index_source=source_file,
    )

    for clip in expected:
        # Check atomic index
        match = find_matching_clip(clip, indexed)
        if match:
            audit.in_atomic_index += 1
        else:
            audit.missing_from_index.append(
                f"#{clip.clip_num:03d} {clip.label} t={clip.start_s:.0f}-{clip.end_s:.0f}s"
            )

        # Check sidecars (broader search including old naming)
        sidecar = find_matching_sidecar(clip, sidecars)
        rendered = False

        if sidecar:
            audit.has_sidecar += 1
            steps = sidecar.get("steps", {})
            if steps.get("upscale", {}).get("done"):
                audit.upscaled += 1
            else:
                audit.missing_upscale.append(
                    f"#{clip.clip_num:03d} {clip.label}"
                )
            if steps.get("portrait_render", {}).get("done"):
                rendered = True
            elif rendered_stems:
                # Fallback: check if rendered output file exists on disk
                clip_stem = Path(sidecar.get("clip_path", "")).stem
                if clip_stem and clip_stem in rendered_stems:
                    rendered = True

        # Pipeline-status fallback — works whether sidecar found or not,
        # handles clips renamed after rendering (different label, same timestamps)
        if not rendered:
            ts_tag = f"t{clip.start_s:.0f}"
            for cp, status in pipeline_status.items():
                if game_folder not in cp:
                    continue
                if any(tag in cp for tag in ("__CINEMATIC", "__DEBUG", "__OVERLAY",
                                             "portrait_POST", "portrait_FINAL")):
                    continue
                if status.get("portrait_path") and ts_tag in cp:
                    rendered = True
                    break

        if rendered:
            audit.portrait_rendered += 1
        elif sidecar:
            audit.missing_portrait.append(
                f"#{clip.clip_num:03d} {clip.label}"
            )
        else:
            audit.missing_upscale.append(
                f"#{clip.clip_num:03d} {clip.label}"
            )
            audit.missing_portrait.append(
                f"#{clip.clip_num:03d} {clip.label}"
            )

    # Build set of clip filenames with successful renders for this game
    _VARIANT_TAGS = ("__CINEMATIC", "__DEBUG", "__OVERLAY", "portrait_POST", "portrait_FINAL")
    rendered_filenames: set[str] = set()
    for cp, status in pipeline_status.items():
        if game_folder not in cp:
            continue
        if status.get("portrait_path", "").strip() and status.get("render_done_at", "").strip():
            fname = cp.replace("\\", "/").rsplit("/", 1)[-1]
            rendered_filenames.add(fname)

    # Count errors from pipeline_status (skip resolved & variant entries)
    for cp, status in pipeline_status.items():
        if game_folder not in cp:
            continue
        # Skip rendering-variant paths (same filter as sidecar loader)
        if any(tag in cp for tag in _VARIANT_TAGS):
            continue
        err = status.get("last_error", "").strip()
        if not err:
            continue
        # Skip stale errors — render succeeded after the error was logged
        if status.get("portrait_path", "").strip() and status.get("render_done_at", "").strip():
            continue
        # Skip stale errors — same clip rendered via a different path
        clip_fname = cp.replace("\\", "/").rsplit("/", 1)[-1]
        if clip_fname in rendered_filenames:
            continue
        # Skip stale errors — output file exists on disk
        if rendered_stems:
            clip_stem = Path(clip_fname).stem
            if clip_stem in rendered_stems:
                continue
        audit.errors += 1
        audit.error_details.append(f"{clip_fname[:60]}: {err}")

    return audit


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_bar(filled: int, total: int, width: int = 20) -> str:
    """Simple ASCII progress bar."""
    if total == 0:
        return "[" + " " * width + "]"
    pct = min(filled / total, 1.0)
    n = int(pct * width)
    return "[" + "#" * n + "-" * (width - n) + f"] {filled}/{total}"


def print_report(audits: list[GameAudit]) -> None:
    """Print a formatted audit report to stdout."""
    sep = "=" * 78
    print()
    print(sep)
    print("  SOCCER VIDEO PIPELINE AUDIT — ALL GAMES")
    print(sep)
    print()

    total_expected = 0
    total_indexed = 0
    total_upscaled = 0
    total_rendered = 0
    total_errors = 0

    for audit in sorted(audits, key=lambda a: a.game_date):
        n = len(audit.expected_clips)
        total_expected += n
        total_indexed += audit.in_atomic_index
        total_upscaled += audit.upscaled
        total_rendered += audit.portrait_rendered
        total_errors += audit.errors

        # Game header
        print(f"  {audit.game_folder}")
        print(f"  Source: {audit.index_source}  |  Expected clips: {n}")
        print()

        # Progress bars
        print(f"    Cataloged:  {format_bar(audit.in_atomic_index, n)}")
        print(f"    Upscaled:   {format_bar(audit.upscaled, n)}")
        print(f"    Rendered:   {format_bar(audit.portrait_rendered, n)}")
        if audit.errors:
            print(f"    Errors:     {audit.errors}")

        # Status determination
        if audit.portrait_rendered == n and audit.upscaled == n:
            status = "COMPLETE"
        elif audit.portrait_rendered == n and audit.upscaled == 0:
            status = "RENDERED (no upscale)"
        elif audit.portrait_rendered > 0:
            status = "PARTIAL"
        elif audit.errors > 0:
            status = "FAILED"
        else:
            status = "NOT STARTED"
        print(f"    Status:     {status}")

        # Details for incomplete games
        if audit.missing_from_index and len(audit.missing_from_index) <= 10:
            print(f"    Missing from catalog ({len(audit.missing_from_index)}):")
            for item in audit.missing_from_index:
                print(f"      - {item}")
        elif audit.missing_from_index:
            print(f"    Missing from catalog: {len(audit.missing_from_index)} clips")

        if audit.error_details and len(audit.error_details) <= 5:
            print(f"    Error details:")
            for err in audit.error_details:
                print(f"      - {err}")
        elif audit.error_details:
            print(f"    Errors: {len(audit.error_details)} clips with errors")

        print()
        print("-" * 78)
        print()

    # Summary
    print(sep)
    print("  SUMMARY")
    print(sep)
    print()
    print(f"    Games:              {len(audits)}")
    print(f"    Expected clips:     {total_expected}")
    print(f"    In atomic index:    {total_indexed}  {format_bar(total_indexed, total_expected)}")
    print(f"    Upscaled:           {total_upscaled}  {format_bar(total_upscaled, total_expected)}")
    print(f"    Portrait rendered:  {total_rendered}  {format_bar(total_rendered, total_expected)}")
    print(f"    Pipeline errors:    {total_errors}")
    print()

    # Action items
    print(sep)
    print("  ACTION ITEMS")
    print(sep)
    print()

    actions = []
    for audit in sorted(audits, key=lambda a: a.game_date):
        n = len(audit.expected_clips)
        missing_idx = n - audit.in_atomic_index
        missing_up = n - audit.upscaled
        missing_rend = n - audit.portrait_rendered

        if missing_idx > 0:
            actions.append(
                f"  [{audit.game_date}] {audit.game_folder}: "
                f"catalog {missing_idx} missing clip(s) into atomic_index"
            )
        if missing_up > 0:
            actions.append(
                f"  [{audit.game_date}] {audit.game_folder}: "
                f"upscale {missing_up} clip(s) (2x lanczos)"
            )
        if missing_rend > 0:
            actions.append(
                f"  [{audit.game_date}] {audit.game_folder}: "
                f"render {missing_rend} portrait reel(s)"
            )
        if audit.errors > 0:
            actions.append(
                f"  [{audit.game_date}] {audit.game_folder}: "
                f"investigate {audit.errors} pipeline error(s)"
            )

    if actions:
        for action in actions:
            print(action)
    else:
        print("  No action items — all games fully processed!")
    print()


def to_json(audits: list[GameAudit]) -> str:
    """Serialize audit results as JSON."""
    result = []
    for a in sorted(audits, key=lambda x: x.game_date):
        n = len(a.expected_clips)
        result.append({
            "game_folder": a.game_folder,
            "game_date": a.game_date,
            "index_source": a.index_source,
            "expected": n,
            "in_atomic_index": a.in_atomic_index,
            "has_sidecar": a.has_sidecar,
            "upscaled": a.upscaled,
            "portrait_rendered": a.portrait_rendered,
            "errors": a.errors,
            "missing_from_index": a.missing_from_index,
            "missing_upscale": a.missing_upscale,
            "missing_portrait": a.missing_portrait,
            "error_details": a.error_details,
        })
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Reconcile sidecars from output files
# ---------------------------------------------------------------------------

def _reconcile(rendered_stems: set[str], dry_run: bool = False) -> int:
    """Update sidecars for clips with output files but missing render status.

    Matches by exact stem first, then falls back to matching by game folder
    + clip number + timestamps (handles clips renamed after rendering).
    """
    import datetime as _dt

    if not PORTRAIT_REELS_DIR.exists():
        print("No portrait reels directory found — nothing to reconcile.")
        return 0

    # Build two indexes for output files:
    # 1. Exact stem match (fast path)
    # 2. (game_folder, clip_num, timestamps) match (handles renames)
    output_by_stem: dict[str, Path] = {}
    output_by_id: dict[tuple[str, str, str], Path] = {}
    _ts_re = re.compile(r"__t([\d.]+)-t?([\d.]+)")
    _num_re = re.compile(r"^(\d+)__(\d{4}-\d{2}-\d{2}__[^_]+(?:_[^_]+)*?)__")

    for f in PORTRAIT_REELS_DIR.iterdir():
        if f.suffix.lower() not in {".mp4", ".mov"}:
            continue
        stem = _PORTRAIT_SUFFIX_RE.sub("", f.stem)
        output_by_stem[stem] = f
        # Parse (game_folder, clip_num, timestamps)
        ts_m = _ts_re.search(stem)
        num_m = _num_re.match(stem)
        if ts_m and num_m:
            key = (num_m.group(2), num_m.group(1), f"t{ts_m.group(1)}-t{ts_m.group(2)}")
            output_by_id[key] = f

    if not SIDECAR_DIR.exists():
        print("No sidecar directory found — nothing to reconcile.")
        return 0

    updated = 0
    for sidecar_file in sorted(SIDECAR_DIR.iterdir()):
        if sidecar_file.suffix != ".json":
            continue
        stem = sidecar_file.stem
        # Skip rendering variants
        if any(tag in stem for tag in ("__CINEMATIC", "__DEBUG", "__OVERLAY",
                                       "portrait_POST", "portrait_FINAL")):
            continue

        try:
            data = json.loads(sidecar_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        steps = data.get("steps", {})
        if steps.get("portrait_render", {}).get("done"):
            continue  # Already marked

        # Try exact stem match first
        output_file = output_by_stem.get(stem)
        if not output_file:
            # Fallback: match by game_folder + clip_num + timestamps
            ts_m = _ts_re.search(stem)
            num_m = _num_re.match(stem)
            if ts_m and num_m:
                key = (num_m.group(2), num_m.group(1), f"t{ts_m.group(1)}-t{ts_m.group(2)}")
                output_file = output_by_id.get(key)
        if not output_file:
            continue

        if dry_run:
            print(f"  [DRY-RUN] Would mark rendered: {stem}")
            print(f"            Output: {output_file.name}")
        else:
            steps["portrait_render"] = {
                "done": True,
                "out": str(output_file),
                "at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                "preset": "cinematic",
                "portrait": "1080x1920",
                "reconciled": True,
            }
            data["steps"] = steps
            sidecar_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
            print(f"  [OK] Marked rendered: {stem}")
        updated += 1

    if updated == 0:
        print("All sidecars are up to date — nothing to reconcile.")
    else:
        verb = "Would update" if dry_run else "Updated"
        print(f"\n{verb} {updated} sidecar(s).")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cross-game pipeline audit.")
    parser.add_argument("--game", help="Filter to games matching this substring")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--reconcile", action="store_true",
                        help="Update sidecars for clips with rendered outputs but missing sidecar status")
    parser.add_argument("--dry-run", action="store_true",
                        help="With --reconcile, show what would change without writing")
    args = parser.parse_args(argv)

    if not GAMES_DIR.exists():
        print(f"Games directory not found: {GAMES_DIR}", file=sys.stderr)
        return 1

    rendered_stems = load_rendered_stems()

    if args.reconcile:
        return _reconcile(rendered_stems, dry_run=args.dry_run)

    atomic_by_master = load_atomic_index()
    pipeline_status = load_pipeline_status()

    audits: list[GameAudit] = []
    for game_dir in sorted(GAMES_DIR.iterdir()):
        if not game_dir.is_dir():
            continue
        if args.game and args.game.upper() not in game_dir.name.upper():
            continue
        audit = audit_game(game_dir, atomic_by_master, pipeline_status, rendered_stems)
        if audit is not None:
            audits.append(audit)

    if not audits:
        print("No games found to audit.", file=sys.stderr)
        return 1

    if args.json:
        print(to_json(audits))
    else:
        print_report(audits)

    return 0


if __name__ == "__main__":
    sys.exit(main())

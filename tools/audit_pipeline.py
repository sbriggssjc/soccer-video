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

ROOT = Path(__file__).resolve().parents[1]
GAMES_DIR = ROOT / "out" / "games"
CATALOG_DIR = ROOT / "out" / "catalog"
SIDECAR_DIR = CATALOG_DIR / "sidecar"
ATOMIC_INDEX_PATH = CATALOG_DIR / "atomic_index.csv"
PIPELINE_STATUS_PATH = CATALOG_DIR / "pipeline_status.csv"


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
# Timestamp parsing helpers
# ---------------------------------------------------------------------------

_THREE_PART_RE = re.compile(r"^(\d+):(\d+):(\d+(?:\.\d+)?)$")  # A:B:C
_MS_RE = re.compile(r"^(\d+):(\d+(?:\.\d+)?)$")                 # MM:SS.f
_SEC_RE = re.compile(r"^[\d,]+(?:\.\d+)?$")                       # plain seconds (with commas)


def _parse_three_part_as_mmss_cs(a: float, b: float, c: float) -> float:
    """Interpret A:B:C as MM:SS:centiseconds (e.g., "2:02:00" = 2m 2s)."""
    return a * 60 + b + c / 100


def _parse_three_part_as_hmmss(a: float, b: float, c: float) -> float:
    """Interpret A:B:C as H:MM:SS (e.g., "0:06:37" = 0h 6m 37s)."""
    return a * 3600 + b * 60 + c


def parse_time_to_seconds(raw: str) -> Optional[float]:
    """Parse various timestamp formats to seconds."""
    raw = raw.strip().strip('"')
    if not raw:
        return None

    # Three-part: default to MM:SS:cs; pair-level disambiguation overrides
    m = _THREE_PART_RE.match(raw)
    if m:
        a, b, c = float(m.group(1)), float(m.group(2)), float(m.group(3))
        return _parse_three_part_as_mmss_cs(a, b, c)

    # MM:SS.f
    m = _MS_RE.match(raw)
    if m:
        mn, s = float(m.group(1)), float(m.group(2))
        return mn * 60 + s

    # Plain seconds (possibly with comma thousands separator)
    m = _SEC_RE.match(raw)
    if m:
        return float(raw.replace(",", ""))

    return None


def parse_time_pair(raw_start: str, raw_end: str) -> tuple[Optional[float], Optional[float]]:
    """Parse a start/end timestamp pair, disambiguating three-part formats.

    When both timestamps match A:B:C, tries MM:SS:cs first.  If that gives
    a clip shorter than 2 seconds, retries as H:MM:SS (needed for TSC Navy
    timestamps like "0:06:37" = 6 min 37 sec, not 6.37 sec).
    """
    raw_start = (raw_start or "").strip().strip('"')
    raw_end = (raw_end or "").strip().strip('"')
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
    return parse_time_to_seconds(raw_start), parse_time_to_seconds(raw_end)


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

def audit_game(game_dir: Path, atomic_by_master: dict, pipeline_status: dict) -> Optional[GameAudit]:
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
                audit.portrait_rendered += 1
            else:
                audit.missing_portrait.append(
                    f"#{clip.clip_num:03d} {clip.label}"
                )
        else:
            # Also check pipeline_status for any matching entries
            found_in_pipeline = False
            for cp, status in pipeline_status.items():
                if game_key.replace("TSC_vs_", "") in cp or game_folder in cp:
                    # Rough timestamp match from filename
                    if status.get("portrait_path") and f"t{clip.start_s:.0f}" in cp:
                        found_in_pipeline = True
                        audit.portrait_rendered += 1
                        break
            if not found_in_pipeline:
                audit.missing_upscale.append(
                    f"#{clip.clip_num:03d} {clip.label}"
                )
                audit.missing_portrait.append(
                    f"#{clip.clip_num:03d} {clip.label}"
                )

    # Count errors from pipeline_status
    for cp, status in pipeline_status.items():
        if game_key.replace("TSC_vs_", "") in cp or game_folder in cp:
            err = status.get("last_error", "").strip()
            if err:
                audit.errors += 1
                clip_name = cp.split("\\")[-1] if "\\" in cp else cp.split("/")[-1]
                audit.error_details.append(f"{clip_name[:60]}: {err}")

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
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cross-game pipeline audit.")
    parser.add_argument("--game", help="Filter to games matching this substring")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args(argv)

    if not GAMES_DIR.exists():
        print(f"Games directory not found: {GAMES_DIR}", file=sys.stderr)
        return 1

    atomic_by_master = load_atomic_index()
    pipeline_status = load_pipeline_status()

    audits: list[GameAudit] = []
    for game_dir in sorted(GAMES_DIR.iterdir()):
        if not game_dir.is_dir():
            continue
        if args.game and args.game.upper() not in game_dir.name.upper():
            continue
        audit = audit_game(game_dir, atomic_by_master, pipeline_status)
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

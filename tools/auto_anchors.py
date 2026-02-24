"""Auto-generate manual anchor CSVs from diagnostic data.

Analyzes diagnostic CSVs from rendered clips to identify gaps in
real ball detection coverage and generates manual anchor files using
the fused ball position data. This gives the PCHIP spline additional
control points to avoid camera freezes during long gaps.

Three gap types are handled:
  1. TRAILING: After last real detection — dense sampling every N frames
  2. LEADING: Before first real detection — sparse sampling
  3. INTERIOR: Between real detections with gap > threshold — midpoint
     + quartile sampling (avoids adding too many INTERP points which
     would downgrade PCHIP interpolation to linear)

Usage:
    # Single clip:
    python tools/auto_anchors.py \\
        --diag out/portrait_reels/.../001__fix_v8_spline.diag.csv \\
        --out  out/portrait_reels/.../001__manual_anchors.csv

    # Batch mode (all clips in a match directory):
    python tools/auto_anchors.py \\
        --match-dir out/portrait_reels/2026-02-23__TSC_vs_NEOFC/ \\
        --diag-pattern "*_spline.diag.csv"

    # Merge with existing manual anchors:
    python tools/auto_anchors.py \\
        --diag <diag.csv> --out <anchors.csv> --merge <existing_anchors.csv>
"""
import argparse
import csv
import sys
from pathlib import Path

# Real detection sources that the spline uses as anchors
ANCHOR_SOURCES = {"yolo", "blended", "tracker"}

# Preferred sources for auto-anchor positions (ranked by quality)
# centroid tracks player mass — reasonable for general camera following
# interp is linear interpolation — OK for short gaps
# hold is stale data — last resort
SOURCE_QUALITY = {"yolo": 5, "tracker": 4, "blended": 3, "centroid": 2, "interp": 1, "hold": 0}


def load_diag(path: Path) -> list[dict]:
    """Load diagnostic CSV into list of dicts."""
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_existing_anchors(path: Path) -> dict[int, tuple[float, float]]:
    """Load existing manual anchors as {frame: (x, y)}."""
    anchors = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fi = int(row["frame"])
            bx = float(row["ball_x"])
            by = float(row["ball_y"])
            anchors[fi] = (bx, by)
    return anchors


def find_anchor_frames(rows: list[dict]) -> list[int]:
    """Return sorted list of frame indices that have real detections."""
    anchors = []
    for row in rows:
        src = row.get("source", "").strip().lower()
        if src in ANCHOR_SOURCES:
            fi = int(row["frame"])
            bx = row.get("ball_x", "")
            by = row.get("ball_y", "")
            try:
                x = float(bx)
                y = float(by)
                if x > 0 and y > 0:
                    anchors.append(fi)
            except (ValueError, TypeError):
                pass
    return sorted(set(anchors))


def generate_auto_anchors(
    rows: list[dict],
    *,
    interior_gap_threshold: int = 20,
    trailing_threshold: int = 15,
    leading_threshold: int = 25,
    trailing_interval: int = 12,
    interior_max_samples: int = 5,
    existing: dict[int, tuple[float, float]] | None = None,
) -> list[dict]:
    """Generate auto-anchor positions for coverage gaps.

    Args:
        rows: Diagnostic CSV rows
        interior_gap_threshold: Min gap (frames) between anchors to trigger auto-fill
        trailing_threshold: Min trailing frames after last anchor to trigger
        leading_threshold: Min leading frames before first anchor to trigger
        trailing_interval: Frame interval for trailing anchor sampling
        interior_max_samples: Max anchor samples per interior gap
        existing: Existing manual anchors to preserve (frame -> (x, y))

    Returns:
        List of dicts with 'frame', 'ball_x', 'ball_y', 'reason'
    """
    if existing is None:
        existing = {}

    n = len(rows)
    if n == 0:
        return []

    # Build frame -> row lookup
    frame_map = {}
    for row in rows:
        fi = int(row["frame"])
        frame_map[fi] = row

    real_anchor_frames = find_anchor_frames(rows)

    # Merge existing manual anchor frames into anchor list for gap analysis
    # This prevents the tool from adding auto-anchors in gaps already covered
    # by manually-placed anchors
    anchor_frames = sorted(set(real_anchor_frames) | set(existing.keys()))

    if not anchor_frames:
        # No real detections at all — sample every 20 frames
        results = []
        for i in range(0, n, 20):
            fi = int(rows[i]["frame"])
            if fi in existing:
                continue
            bx = rows[i].get("ball_x", "")
            by = rows[i].get("ball_y", "")
            try:
                x = float(bx)
                y = float(by)
                if x > 0 and y > 0:
                    results.append({
                        "frame": fi, "ball_x": x, "ball_y": y,
                        "reason": "no_anchors",
                    })
            except (ValueError, TypeError):
                pass
        return results

    results = []

    # 1) Leading hold — before first anchor
    first_anchor = anchor_frames[0]
    if first_anchor > leading_threshold:
        # Sample a couple of points in the leading section
        mid = first_anchor // 2
        q1 = first_anchor // 4
        for fi in sorted(set([q1, mid])):
            if fi <= 0 or fi >= first_anchor or fi in existing:
                continue
            row = frame_map.get(fi)
            if row is None:
                continue
            try:
                x = float(row["ball_x"])
                y = float(row["ball_y"])
                if x > 0 and y > 0:
                    results.append({
                        "frame": fi, "ball_x": x, "ball_y": y,
                        "reason": f"leading_hold_{first_anchor}f",
                    })
            except (ValueError, TypeError):
                pass

    # 2) Interior gaps between anchors
    for gi in range(len(anchor_frames) - 1):
        gap_start = anchor_frames[gi]
        gap_end = anchor_frames[gi + 1]
        gap_len = gap_end - gap_start

        if gap_len <= interior_gap_threshold:
            continue

        # Sample strategy: midpoint + quartiles, up to interior_max_samples
        # For very large gaps, add more samples
        if gap_len > 60:
            n_samples = min(interior_max_samples, max(3, gap_len // 15))
        elif gap_len > 40:
            n_samples = min(interior_max_samples, 3)
        else:
            n_samples = min(interior_max_samples, 2)

        # Generate evenly-spaced sample points within the gap
        import numpy as np
        sample_frames = np.linspace(
            gap_start + max(3, gap_len // 8),
            gap_end - max(3, gap_len // 8),
            n_samples,
            dtype=int,
        )

        for fi in sample_frames:
            fi = int(fi)
            if fi <= gap_start or fi >= gap_end or fi in existing:
                continue
            row = frame_map.get(fi)
            if row is None:
                continue
            src = row.get("source", "").strip().lower()
            try:
                x = float(row["ball_x"])
                y = float(row["ball_y"])
                if x > 0 and y > 0:
                    results.append({
                        "frame": fi, "ball_x": x, "ball_y": y,
                        "reason": f"gap_{gap_len}f_src={src}",
                    })
            except (ValueError, TypeError):
                pass

    # 3) Trailing hold — after last anchor
    last_anchor = anchor_frames[-1]
    last_frame = int(rows[-1]["frame"])
    trailing_len = last_frame - last_anchor

    if trailing_len > trailing_threshold:
        # Dense sampling at regular intervals
        fi = last_anchor + trailing_interval
        while fi <= last_frame:
            if fi not in existing:
                row = frame_map.get(fi)
                if row is None:
                    # Try adjacent frames
                    for offset in [1, -1, 2, -2]:
                        row = frame_map.get(fi + offset)
                        if row is not None:
                            fi = fi + offset
                            break
                if row is not None:
                    try:
                        x = float(row["ball_x"])
                        y = float(row["ball_y"])
                        if x > 0 and y > 0:
                            results.append({
                                "frame": fi, "ball_x": x, "ball_y": y,
                                "reason": f"trailing_hold_{trailing_len}f",
                            })
                    except (ValueError, TypeError):
                        pass
            fi += trailing_interval

    # De-duplicate and sort
    seen = set()
    unique = []
    for r in sorted(results, key=lambda x: x["frame"]):
        if r["frame"] not in seen:
            seen.add(r["frame"])
            unique.append(r)

    return unique


def write_anchors(anchors: list[dict], out_path: Path, existing: dict[int, tuple[float, float]] | None = None):
    """Write manual anchors CSV, merging with any existing anchors."""
    merged = {}

    # Add existing anchors first
    if existing:
        for fi, (x, y) in existing.items():
            merged[fi] = (x, y)

    # Add auto-generated anchors (don't overwrite existing)
    for a in anchors:
        fi = a["frame"]
        if fi not in merged:
            merged[fi] = (a["ball_x"], a["ball_y"])

    # Sort by frame and write
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "ball_x", "ball_y"])
        for fi in sorted(merged.keys()):
            x, y = merged[fi]
            writer.writerow([fi, f"{x:.1f}", f"{y:.1f}"])

    return len(merged)


def process_single(diag_path: Path, out_path: Path, merge_path: Path | None = None, **kwargs) -> dict:
    """Process a single diagnostic CSV and generate anchors.

    Returns dict with summary stats.
    """
    rows = load_diag(diag_path)
    n = len(rows)
    if n == 0:
        return {"status": "empty", "frames": 0, "anchors_added": 0}

    existing = {}
    if merge_path and merge_path.exists():
        existing = load_existing_anchors(merge_path)

    anchor_frames = find_anchor_frames(rows)
    coverage = len(anchor_frames) / n * 100 if n > 0 else 0
    # Note: coverage is based on real detections only (YOLO/TRACKER/BLENDED),
    # not including existing manual anchors

    auto_anchors = generate_auto_anchors(rows, existing=existing, **kwargs)
    total = write_anchors(auto_anchors, out_path, existing=existing)

    # Summary
    reasons = {}
    for a in auto_anchors:
        r = a["reason"].split("_")[0]  # leading, trailing, gap, no
        reasons[r] = reasons.get(r, 0) + 1

    return {
        "status": "ok",
        "frames": n,
        "real_anchors": len(anchor_frames),
        "coverage_pct": coverage,
        "auto_added": len(auto_anchors),
        "existing_kept": len(existing),
        "total_manual": total,
        "reasons": reasons,
    }


def process_batch(match_dir: Path, diag_pattern: str = "*_spline.diag.csv", **kwargs):
    """Process all diagnostic CSVs in a match directory."""
    diag_files = sorted(match_dir.glob(diag_pattern))
    if not diag_files:
        print(f"No diagnostic files matching '{diag_pattern}' in {match_dir}")
        return []

    results = []
    for diag_path in diag_files:
        # Derive clip number from filename (e.g., "001__fix_v8_spline.diag.csv" -> "001")
        stem = diag_path.stem.replace(".diag", "")
        clip_num = stem.split("__")[0]

        # Output path: {clip_num}__manual_anchors.csv
        out_path = match_dir / f"{clip_num}__manual_anchors.csv"

        # Check for existing manual anchors to merge
        merge_path = out_path if out_path.exists() else None

        print(f"\n--- {clip_num} ({diag_path.name}) ---")
        stats = process_single(diag_path, out_path, merge_path, **kwargs)
        stats["clip"] = clip_num
        stats["diag"] = diag_path.name
        results.append(stats)

        if stats["status"] == "ok":
            print(f"  Frames: {stats['frames']}, Real anchors: {stats['real_anchors']} "
                  f"({stats['coverage_pct']:.0f}%)")
            print(f"  Auto-generated: {stats['auto_added']}, "
                  f"Existing kept: {stats['existing_kept']}, "
                  f"Total manual: {stats['total_manual']}")
            if stats['reasons']:
                print(f"  Breakdown: {stats['reasons']}")
        else:
            print(f"  {stats['status']}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Batch complete: {len(results)} clips processed")
    total_auto = sum(r.get("auto_added", 0) for r in results)
    total_manual = sum(r.get("total_manual", 0) for r in results)
    clips_needing = sum(1 for r in results if r.get("auto_added", 0) > 0)
    print(f"  {clips_needing}/{len(results)} clips needed auto-anchors")
    print(f"  {total_auto} auto-anchors generated, {total_manual} total manual anchors")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Auto-generate manual anchors from diagnostic CSVs"
    )

    # Single mode
    parser.add_argument("--diag", type=str, help="Path to diagnostic CSV")
    parser.add_argument("--out", type=str, help="Output path for manual_anchors.csv")
    parser.add_argument("--merge", type=str, default=None,
                        help="Existing manual_anchors.csv to merge with (preserves existing)")

    # Batch mode
    parser.add_argument("--match-dir", type=str,
                        help="Match directory to process all diag CSVs")
    parser.add_argument("--diag-pattern", type=str, default="*_spline.diag.csv",
                        help="Glob pattern for diag files in batch mode (default: *_spline.diag.csv)")

    # Tuning
    parser.add_argument("--interior-gap", type=int, default=20,
                        help="Min interior gap (frames) to auto-fill (default: 20)")
    parser.add_argument("--trailing-threshold", type=int, default=15,
                        help="Min trailing hold (frames) to auto-fill (default: 15)")
    parser.add_argument("--leading-threshold", type=int, default=25,
                        help="Min leading hold (frames) to auto-fill (default: 25)")
    parser.add_argument("--trailing-interval", type=int, default=12,
                        help="Frame interval for trailing anchors (default: 12)")

    args = parser.parse_args()

    kwargs = {
        "interior_gap_threshold": args.interior_gap,
        "trailing_threshold": args.trailing_threshold,
        "leading_threshold": args.leading_threshold,
        "trailing_interval": args.trailing_interval,
    }

    if args.match_dir:
        # Batch mode
        match_dir = Path(args.match_dir)
        if not match_dir.is_dir():
            print(f"ERROR: Not a directory: {match_dir}")
            sys.exit(1)
        process_batch(match_dir, args.diag_pattern, **kwargs)

    elif args.diag:
        # Single mode
        diag_path = Path(args.diag)
        if not diag_path.exists():
            print(f"ERROR: Not found: {diag_path}")
            sys.exit(1)

        out_path = Path(args.out) if args.out else diag_path.parent / (
            diag_path.stem.split("__")[0] + "__manual_anchors.csv"
        )
        merge_path = Path(args.merge) if args.merge else None

        stats = process_single(diag_path, out_path, merge_path, **kwargs)
        if stats["status"] == "ok":
            print(f"Frames: {stats['frames']}, Real anchors: {stats['real_anchors']} "
                  f"({stats['coverage_pct']:.0f}%)")
            print(f"Auto-generated: {stats['auto_added']}, "
                  f"Existing kept: {stats['existing_kept']}, "
                  f"Total manual: {stats['total_manual']}")
            print(f"Output: {out_path}")
        else:
            print(f"Status: {stats['status']}")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

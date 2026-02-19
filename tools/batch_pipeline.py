#!/usr/bin/env python
"""Unified batch pipeline: render portrait reels, upscale, and track in catalog.

Usage examples:

    # Render all clips with auto-tuning (default — one command, no tuning needed)
    python tools/batch_pipeline.py

    # Dry run — show what would be processed (no rendering)
    python tools/batch_pipeline.py --dry-run

    # Render with a specific preset instead of auto
    python tools/batch_pipeline.py --preset cinematic

    # Render + upscale with progress tracking
    python tools/batch_pipeline.py --upscale

    # Limit to first 5 clips for testing
    python tools/batch_pipeline.py --limit 5

    # Rebuild catalog, detect duplicates, and clean up before rendering
    python tools/batch_pipeline.py --rebuild-catalog --cleanup-dupes

    # Generate status report only
    python tools/batch_pipeline.py --report
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.catalog import (
    ATOMIC_INDEX_PATH,
    DUPLICATES_PATH,
    PIPELINE_STATUS_PATH,
    ensure_catalog_dirs,
    generate_report,
    load_duplicates,
    load_existing_atomic_rows,
    load_pipeline_status_table,
    mark_rendered,
    mark_upscaled,
    normalize_tree,
    rebuild_atomic_index,
    write_duplicates_from_index,
)
from tools.path_naming import build_output_name


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified batch pipeline: render, upscale, track, and clean.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Pipeline steps
    p.add_argument(
        "--preset",
        default="auto",
        help="Render preset (default: auto — analyses each clip and auto-tunes)",
    )
    p.add_argument(
        "--portrait",
        default="1080x1920",
        help="Portrait output geometry WxH (default: 1080x1920)",
    )
    p.add_argument(
        "--upscale",
        action="store_true",
        help="Upscale portrait renders after rendering (2x lanczos)",
    )
    p.add_argument(
        "--upscale-scale",
        type=int,
        default=2,
        help="Upscale factor (default: 2)",
    )
    p.add_argument(
        "--upscale-method",
        default="lanczos",
        choices=["lanczos", "realesrgan"],
        help="Upscale method (default: lanczos)",
    )

    # Clip selection
    p.add_argument(
        "--src-dir",
        default="out/atomic_clips",
        help="Root directory for atomic clips (default: out/atomic_clips)",
    )
    p.add_argument(
        "--out-dir",
        default="out/portrait_reels/clean",
        help="Output directory for portrait renders (default: out/portrait_reels/clean)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max clips to process (0 = no limit)",
    )
    p.add_argument(
        "--pattern",
        default="*.mp4",
        help="Glob pattern for scanning clips (default: *.mp4)",
    )
    p.add_argument(
        "--game",
        help="Filter to a specific game folder name (substring match)",
    )
    p.add_argument(
        "--clip",
        action="append",
        help="Process specific clip path(s) instead of scanning (repeatable)",
    )
    p.add_argument(
        "--remap",
        metavar="OLD=NEW",
        help=(
            "Remap clip path prefix from catalog. "
            "E.g. --remap 'C:/Users/scott/OneDrive/SoccerVideoMedia=D:/Projects/soccer-video'"
        ),
    )

    # Behavior
    p.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip clips with existing output (default: True)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-render even if output exists",
    )
    p.add_argument(
        "--skip-duplicates",
        action="store_true",
        default=True,
        help="Skip duplicate clips (default: True)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without rendering",
    )

    # Catalog operations
    p.add_argument(
        "--rebuild-catalog",
        action="store_true",
        help="Rebuild atomic_index.csv and duplicates.csv before processing",
    )
    p.add_argument(
        "--cleanup-dupes",
        action="store_true",
        help="Move duplicate clips to trash after processing",
    )
    p.add_argument(
        "--purge-trash",
        action="store_true",
        help="Permanently delete trashed duplicates",
    )

    # Output cleanup
    p.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove render output files not tracked in pipeline_status.csv",
    )

    # Reporting
    p.add_argument(
        "--report",
        action="store_true",
        help="Print pipeline status report and exit",
    )

    # Render options
    p.add_argument(
        "--keep-scratch",
        action="store_true",
        help="Keep scratch artifacts after rendering",
    )
    p.add_argument(
        "--scratch-root",
        help="Root directory for scratch artifacts",
    )
    p.add_argument(
        "--diagnostics",
        action="store_true",
        help="Write per-frame diagnostic CSV alongside each render",
    )

    return p.parse_args(argv)


def _parse_remap(remap_arg: str | None) -> tuple[str, str] | None:
    """Parse --remap OLD=NEW into (old_prefix, new_prefix)."""
    if not remap_arg:
        return None
    if "=" not in remap_arg:
        print(f"[ERROR] --remap must be OLD=NEW, got: {remap_arg!r}")
        return None
    old, new = remap_arg.split("=", 1)
    # Normalize to forward slashes for consistent matching
    return old.replace("\\", "/"), new.replace("\\", "/")


def _apply_remap(clip_path: str, remap: tuple[str, str] | None) -> str:
    """Replace old prefix with new prefix in a clip path."""
    if not remap:
        return clip_path
    old_prefix, new_prefix = remap
    normalized = clip_path.replace("\\", "/")
    if normalized.startswith(old_prefix):
        return new_prefix + normalized[len(old_prefix):]
    return clip_path


def _load_duplicate_set() -> set[str]:
    """Return set of clip_rel paths that are duplicates (not canonical)."""
    dupes = set()
    try:
        records = load_duplicates()
        for rec in records:
            if rec.duplicate_rel:
                dupes.add(rec.duplicate_rel.replace("\\", "/"))
    except Exception:
        pass
    return dupes


def _scan_directory(src_dir: str, pattern: str, game_filter: str | None) -> list[dict]:
    """Scan src_dir for clip files, applying optional game filter."""
    clips: list[dict] = []
    src_root = REPO_ROOT / src_dir
    if not src_root.exists():
        return clips
    for mp4 in sorted(src_root.rglob(pattern)):
        rel = str(mp4.relative_to(REPO_ROOT)).replace("\\", "/")
        if game_filter and game_filter.lower() not in rel.lower():
            continue
        clips.append({
            "clip_path": str(mp4),
            "clip_rel": rel,
            "clip_stem": mp4.stem,
        })
    return clips


def _clips_from_catalog(src_dir: str, pattern: str, game_filter: str | None,
                        remap: tuple[str, str] | None = None) -> list[dict]:
    """Load clips from atomic_index.csv, falling back to directory scan."""
    clips = []
    rows = load_existing_atomic_rows()
    if rows:
        for rel, row in rows.items():
            clip_path = row.get("clip_path", "")
            if not clip_path:
                continue
            if game_filter and game_filter.lower() not in rel.lower():
                continue
            # Apply path remap if provided
            if remap:
                row = dict(row)  # don't mutate the original
                row["clip_path"] = _apply_remap(row["clip_path"], remap)
                row["clip_rel"] = _apply_remap(row.get("clip_rel", ""), remap)
            clips.append(row)
        if clips:
            return clips
        # Catalog had rows but none matched the filter — show diagnostics
        if game_filter:
            # Collect unique game folder names from catalog for suggestion
            game_names: set[str] = set()
            for rel in rows:
                parts = rel.replace("\\", "/").split("/")
                # Expect pattern: out/atomic_clips/<game>/clip.mp4
                if len(parts) >= 3:
                    game_names.add(parts[-2])
            print(f"[INFO] Catalog has {len(rows)} clips but none match --game {game_filter!r}.")
            if game_names:
                print(f"[INFO] Available games: {', '.join(sorted(game_names))}")
        else:
            print(f"[INFO] Catalog has {len(rows)} rows but all have empty clip_path.")
    else:
        print(f"[INFO] Catalog is empty or missing ({ATOMIC_INDEX_PATH}).")

    # Fallback: scan directory
    clips = _scan_directory(src_dir, pattern, game_filter)
    if clips:
        print(f"[INFO] Found {len(clips)} clips via directory scan of {src_dir}/.")
    return clips


def _output_path_for_clip(clip_path: str, preset: str, portrait: str, out_dir: Path) -> Path:
    """Compute deterministic output path for a clip."""
    output_name = build_output_name(
        input_path=clip_path,
        preset=preset.upper(),
        portrait=portrait,
        follow=None,
        is_final=True,
        extra_tags=[],
    )
    return out_dir / output_name


def _render_clip(clip_path: Path, out_path: Path, preset: str, portrait: str,
                 *, keep_scratch: bool = False,
                 scratch_root: str | None = None,
                 diagnostics: bool = False) -> tuple[bool, str, dict]:
    """Invoke render_follow_unified.py for a single clip.

    Returns (success, error_message, fusion_stats).
    fusion_stats keys: yolo_total, edge_filtered, yolo_used, ball_in_crop_pct
    """
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "render_follow_unified.py"),
        "--preset", preset,
        "--in", str(clip_path),
        "--out", str(out_path),
        "--portrait", portrait,
        "--no-draw-ball",
    ]
    if keep_scratch:
        cmd.append("--keep-scratch")
    if diagnostics:
        cmd.append("--diagnostics")
    if scratch_root:
        cmd.extend(["--scratch-root", scratch_root])

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # Pass stdout through so user sees per-clip logs
    if result.stdout:
        print(result.stdout, end="")

    # Parse fusion stats from stdout
    stats: dict = {}
    if result.stdout:
        # [YOLO] Loaded 175 cached YOLO detections ...
        m = re.search(r"\[YOLO\] Loaded (\d+) cached YOLO detections", result.stdout)
        if m:
            stats["yolo_total"] = int(m.group(1))
        # [FUSION] Filtered 19 near-edge YOLO detections (margin=154px)
        m = re.search(r"\[FUSION\] Filtered (\d+) near-edge YOLO detections", result.stdout)
        stats["edge_filtered"] = int(m.group(1)) if m else 0
        # [FUSION] ... yolo=137, centroid=380, blended=19 ...
        m = re.search(r"yolo=(\d+), centroid=(\d+), blended=(\d+)", result.stdout)
        if m:
            stats["yolo_used"] = int(m.group(1))
            stats["centroid_used"] = int(m.group(2))
            stats["blended"] = int(m.group(3))
        # [DIAG] Ball in crop: 536/536 (100.0%)
        m = re.search(r"Ball in crop: \d+/\d+ \(([0-9.]+)%\)", result.stdout)
        if m:
            stats["ball_in_crop_pct"] = float(m.group(1))
        # [DIAG] ... Outside: N frames | Max escape: Npx
        m = re.search(r"Outside: (\d+) frames \| Max escape: (\d+)px", result.stdout)
        if m:
            stats["ball_outside_frames"] = int(m.group(1))
            stats["ball_max_escape_px"] = int(m.group(2))

    if result.returncode == 0:
        return True, "", stats
    # Extract last non-empty line of stderr for a concise error
    stderr_lines = [ln for ln in (result.stderr or "").strip().splitlines() if ln.strip()]
    err_msg = stderr_lines[-1][:200] if stderr_lines else f"exit code {result.returncode}"
    return False, err_msg, stats


def _upscale_clip(portrait_path: Path, scale: int, method: str) -> str | None:
    """Upscale a portrait render. Returns output path or None on failure."""
    try:
        from tools.upscale import upscale_video
        return upscale_video(str(portrait_path), scale=scale, method=method, track=True)
    except Exception as exc:
        print(f"[ERROR] Upscale failed for {portrait_path}: {exc}")
        return None


def run_report() -> None:
    """Print a comprehensive pipeline status report."""
    ensure_catalog_dirs()
    report = generate_report()

    print("=" * 60)
    print("PIPELINE STATUS REPORT")
    print("=" * 60)
    print(f"  Total atomic clips:    {report['total_clips']}")
    print(f"  Portrait renders done: {report['rendered']}")
    print(f"  Upscaled:              {report['upscaled']}")
    print(f"  Branded:               {report['branded']}")
    print(f"  Duplicate groups:      {report['dup_groups']}")
    print(f"  Orphans (no master):   {report['orphans']}")
    print(f"  Sidecars with errors:  {report['sidecars_with_errors']}")
    print()

    # Show per-step completion
    total = report["total_clips"]
    if total > 0:
        pct_rendered = 100 * report["rendered"] / total
        pct_upscaled = 100 * report["upscaled"] / total
        print(f"  Render progress: {report['rendered']}/{total} ({pct_rendered:.0f}%)")
        print(f"  Upscale progress: {report['upscaled']}/{total} ({pct_upscaled:.0f}%)")
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ensure_catalog_dirs()

    # --- Report mode ---
    if args.report:
        run_report()
        return 0

    # --- Clean output directory ---
    if args.clean_output:
        out_dir = REPO_ROOT / args.out_dir
        _clean_output_dir(out_dir, dry_run=args.dry_run)
        if not args.rebuild_catalog and not args.clip and args.dry_run:
            return 0

    # --- Rebuild catalog ---
    if args.rebuild_catalog:
        print("[CATALOG] Rebuilding atomic index and duplicates...")
        result = rebuild_atomic_index()
        print(
            f"[CATALOG] Scanned: {result.scanned} | Changed: {result.changed} | "
            f"Hard dupes: {result.hard_dupes} | Soft dupes: {result.soft_dupes}"
        )

    # --- Parse remap ---
    remap = _parse_remap(getattr(args, "remap", None))
    if remap:
        print(f"[REMAP] {remap[0]}  ->  {remap[1]}")

    # --- Load clips ---
    if args.clip:
        clips = []
        for c in args.clip:
            p = Path(c)
            clips.append({
                "clip_path": str(p.resolve()),
                "clip_rel": str(p.relative_to(REPO_ROOT)).replace("\\", "/") if p.is_absolute() else c,
                "clip_stem": p.stem,
            })
    else:
        clips = _clips_from_catalog(args.src_dir, args.pattern, args.game, remap=remap)

    if not clips:
        print("[WARN] No clips found to process.")
        src_root = REPO_ROOT / args.src_dir
        if not src_root.exists():
            print(f"[HINT] Source directory does not exist: {src_root}")
        elif not list(src_root.rglob(args.pattern)):
            print(f"[HINT] No {args.pattern} files found under {src_root}")
        if not ATOMIC_INDEX_PATH.exists():
            print("[HINT] Catalog file missing. Run --rebuild-catalog with clips present.")
        return 0

    # --- Filter duplicates ---
    dup_set = _load_duplicate_set() if args.skip_duplicates else set()
    pipeline_table = load_pipeline_status_table()

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build work list
    work = []
    skipped_dup = 0
    skipped_done = 0
    skipped_output = 0
    preset_upper = args.preset.strip().upper()
    # Detect rendered outputs: stem contains dot-prefixed preset tag like .__CINEMATIC
    _output_tag_re = re.compile(
        rf"\.__{re.escape(preset_upper)}(?:\b|__|\.|$)",
        flags=re.IGNORECASE,
    )
    for row in clips:
        clip_rel = row.get("clip_rel", "")
        clip_path_str = row.get("clip_path", "")

        # Skip clips that are rendered outputs (stacked preset tags from previous runs)
        if _output_tag_re.search(Path(clip_path_str).stem):
            skipped_output += 1
            continue

        # Skip duplicates
        if clip_rel in dup_set:
            skipped_dup += 1
            continue

        clip_path = Path(clip_path_str)
        out_path = _output_path_for_clip(clip_path_str, args.preset, args.portrait, out_dir)

        # Check if already rendered
        already_rendered = False
        status = pipeline_table.get(str(clip_path.resolve()), {})
        if status.get("portrait_path") and not args.force:
            portrait_p = Path(status["portrait_path"])
            if portrait_p.exists() and portrait_p.stat().st_size > 0:
                already_rendered = True

        if not already_rendered and not args.force:
            if out_path.exists() and out_path.stat().st_size > 0:
                already_rendered = True

        if already_rendered and args.skip_existing and not args.force:
            skipped_done += 1
            continue

        # Check upscale status
        already_upscaled = False
        if args.upscale and status.get("upscaled_path"):
            up_p = Path(status["upscaled_path"])
            if up_p.exists() and up_p.stat().st_size > 0 and not args.force:
                already_upscaled = True

        work.append({
            "clip_path": clip_path,
            "clip_rel": clip_rel,
            "out_path": out_path,
            "already_rendered": already_rendered,
            "already_upscaled": already_upscaled,
        })

    if args.limit and args.limit > 0:
        work = work[:args.limit]

    # Summary
    total_clips = len(clips)
    print(f"\n{'=' * 60}")
    print(f"BATCH PIPELINE — {args.preset} preset")
    print(f"{'=' * 60}")
    print(f"  Total clips in catalog: {total_clips}")
    print(f"  Skipped (rendered outputs): {skipped_output}")
    print(f"  Skipped (duplicates):   {skipped_dup}")
    print(f"  Skipped (already done): {skipped_done}")
    print(f"  To process:             {len(work)}")
    if args.upscale:
        needs_upscale = sum(1 for w in work if not w["already_upscaled"])
        print(f"  Needs upscale:          {needs_upscale}")
    print(f"  Output directory:       {out_dir}")
    print(f"{'=' * 60}\n")

    if args.dry_run:
        print("[DRY-RUN] Would process:")
        for i, item in enumerate(work, 1):
            status_parts = []
            if item["already_rendered"]:
                status_parts.append("render:skip")
            else:
                status_parts.append("render:TODO")
            if args.upscale:
                if item["already_upscaled"]:
                    status_parts.append("upscale:skip")
                else:
                    status_parts.append("upscale:TODO")
            status_str = ", ".join(status_parts)
            print(f"  {i:3d}. {item['clip_rel']}  [{status_str}]")
        return 0

    if not work:
        print("[OK] Nothing to process — all clips are up to date.")
        if args.cleanup_dupes:
            _run_cleanup(args)
        return 0

    # --- Process clips ---
    ok = 0
    failed = 0
    upscaled = 0
    clip_stats: list[tuple[str, dict]] = []  # (clip_name, fusion_stats)
    t_start = time.monotonic()

    for i, item in enumerate(work, 1):
        clip_path = item["clip_path"]
        out_path = item["out_path"]
        clip_rel = item["clip_rel"]
        elapsed = time.monotonic() - t_start
        eta = ""
        if i > 1 and ok + failed > 0:
            avg = elapsed / (ok + failed)
            remaining = avg * (len(work) - i + 1)
            eta = f"  ETA: {int(remaining // 60)}m{int(remaining % 60):02d}s"

        print(f"\n[{i}/{len(work)}] {clip_rel}{eta}")

        # --- Render ---
        if not item["already_rendered"]:
            if not clip_path.exists():
                print(f"  [SKIP] Source not found: {clip_path}")
                failed += 1
                mark_rendered(clip_path, None, preset=args.preset, error="source not found")
                continue

            out_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"  [RENDER] {clip_path.name} -> {out_path.name}")
            success, err_msg, stats = _render_clip(
                clip_path, out_path, args.preset, args.portrait,
                keep_scratch=args.keep_scratch,
                scratch_root=args.scratch_root,
                diagnostics=getattr(args, "diagnostics", False),
            )
            if stats:
                clip_stats.append((clip_path.name, stats))

            if success and out_path.exists():
                mark_rendered(clip_path, out_path, preset=args.preset, portrait=args.portrait)
                print(f"  [OK] Rendered: {out_path.name}")
                ok += 1
            else:
                mark_rendered(clip_path, None, preset=args.preset, error=err_msg or "render failed")
                print(f"  [FAIL] Render failed: {clip_path.name}")
                if err_msg:
                    print(f"         {err_msg}")
                failed += 1
                continue
        else:
            out_path = item["out_path"]
            # Check if pipeline has the path
            status = pipeline_table.get(str(clip_path.resolve()), {})
            if status.get("portrait_path"):
                out_path = Path(status["portrait_path"])
            ok += 1

        # --- Upscale ---
        if args.upscale and not item["already_upscaled"]:
            if out_path.exists():
                print(f"  [UPSCALE] {out_path.name}")
                result = _upscale_clip(out_path, args.upscale_scale, args.upscale_method)
                if result:
                    upscaled += 1
                    print(f"  [OK] Upscaled: {Path(result).name}")
                else:
                    print(f"  [FAIL] Upscale failed")

    # --- Summary ---
    elapsed = time.monotonic() - t_start
    print(f"\n{'=' * 60}")
    print(f"BATCH COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Rendered:  {ok}")
    print(f"  Failed:    {failed}")
    if args.upscale:
        print(f"  Upscaled:  {upscaled}")
    print(f"  Time:      {int(elapsed // 60)}m{int(elapsed % 60):02d}s")
    print(f"{'=' * 60}")

    # --- Fusion health summary ---
    if clip_stats:
        # Flag clips where edge filter removed >25% of YOLO detections
        # or ball escaped the crop — these may need spot-checking.
        flagged: list[tuple[str, str]] = []
        for name, st in clip_stats:
            total = st.get("yolo_total", 0)
            filtered = st.get("edge_filtered", 0)
            if total > 0 and filtered / total > 0.25:
                pct = 100.0 * filtered / total
                flagged.append((name, f"edge-filtered {filtered}/{total} ({pct:.0f}%) YOLO detections"))
            outside = st.get("ball_outside_frames", 0)
            if outside > 0:
                esc = st.get("ball_max_escape_px", 0)
                flagged.append((name, f"ball outside crop {outside} frames (max {esc}px)"))
            crop_pct = st.get("ball_in_crop_pct", 100.0)
            if crop_pct < 95.0:
                flagged.append((name, f"ball in crop only {crop_pct:.1f}%"))
        if flagged:
            print(f"\n{'=' * 60}")
            print(f"FUSION HEALTH — {len(flagged)} clip(s) flagged for review")
            print(f"{'=' * 60}")
            for name, reason in flagged:
                print(f"  [!] {name}")
                print(f"      {reason}")
            print(f"{'=' * 60}")
        else:
            print(f"\n[FUSION HEALTH] All {len(clip_stats)} clips OK — no edge-filter or crop issues.")

    # --- Cleanup duplicates ---
    if args.cleanup_dupes:
        _run_cleanup(args)

    return 1 if failed > 0 else 0


def _clean_output_dir(out_dir: Path, dry_run: bool = False) -> int:
    """Remove render output files not tracked in pipeline_status.csv.

    Returns count of files removed (or that would be removed in dry-run).
    """
    if not out_dir.is_dir():
        print(f"[CLEAN] Output directory does not exist: {out_dir}")
        return 0

    # Build set of tracked portrait paths (resolved)
    table = load_pipeline_status_table()
    tracked_paths: set[str] = set()
    for _key, status in table.items():
        pp = status.get("portrait_path", "")
        if pp:
            try:
                tracked_paths.add(str(Path(pp).resolve()))
            except Exception:
                tracked_paths.add(pp)

    removed = 0
    for mp4 in sorted(out_dir.rglob("*.mp4")):
        resolved = str(mp4.resolve())
        if resolved not in tracked_paths:
            if dry_run:
                print(f"  [CLEAN] Would remove: {mp4.name}")
            else:
                print(f"  [CLEAN] Removing: {mp4.name}")
                mp4.unlink()
            removed += 1

    if removed == 0:
        print("[CLEAN] No stale output files found.")
    elif dry_run:
        print(f"[CLEAN] Would remove {removed} stale file(s). Run without --dry-run to delete.")
    else:
        print(f"[CLEAN] Removed {removed} stale file(s).")
    return removed


def _run_cleanup(args: argparse.Namespace) -> None:
    """Run duplicate cleanup after processing."""
    print("\n[CLEANUP] Detecting and cleaning up duplicates...")

    # Refresh duplicates from current catalog
    if not DUPLICATES_PATH.exists():
        print("[CLEANUP] Computing duplicate groups...")
        hard, soft = write_duplicates_from_index()
        print(f"[CLEANUP] Found {hard} hard + {soft} soft duplicate groups")

    try:
        result = normalize_tree(
            dry_run=False,
            force=True,
            purge=args.purge_trash,
        )
        print(
            f"[CLEANUP] Moved: {result['moved']} | "
            f"Upscale redirects: {result['upscaled_redirected']} | "
            f"Brand redirects: {result['branded_redirected']} | "
            f"Dirs removed: {result['removed_dirs']}"
        )
        if result["trash_purged"]:
            print("[CLEANUP] Trash purged.")
    except Exception as exc:
        print(f"[CLEANUP] Error: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())

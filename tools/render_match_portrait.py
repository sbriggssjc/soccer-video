#!/usr/bin/env python3
"""Production batch renderer: portrait reels for all clips in a match.

Usage:
    python tools/render_match_portrait.py "2026-02-23__TSC_vs_NEOFC"
    python tools/render_match_portrait.py "2026-02-23__TSC_vs_NEOFC" --resume
    python tools/render_match_portrait.py --all               # all matches, newest first
    python tools/render_match_portrait.py --all --resume      # skip already-rendered

Outputs per match:
    out/portrait_reels/{match}/
        001__clip_name__portrait.mp4        rendered reel
        001__clip_name__portrait.diag.csv   per-frame diagnostic CSV
    out/portrait_reels/{match}/_summary.txt  match-level metrics report
    out/portrait_reels/{match}/_summary.csv  machine-readable metrics

Diagnostic trail enables easy re-analysis without re-rendering.
"""

import os
import re
import subprocess
import sys
import csv
import time
from pathlib import Path
from datetime import datetime

# Handle Windows cp1252 encoding for Unicode output
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass

REPO = Path(__file__).resolve().parent.parent
ATOMIC_DIR = REPO / "out" / "atomic_clips"
OUT_ROOT = REPO / "out" / "portrait_reels"
RENDERER = REPO / "tools" / "render_follow_unified.py"

# Map event tags in filenames -> --event-type values
def extract_event_type(clip_name: str) -> str | None:
    upper = clip_name.upper()
    if "GOAL" in upper:
        return "GOAL"
    if "CROSS" in upper:
        return "CROSS"
    if "FREE_KICK" in upper:
        return "FREE_KICK"
    if "SAVE" in upper:
        return "SAVE"
    if "SHOT" in upper:
        return "SHOT"
    return None


def list_matches(newest_first: bool = True) -> list[str]:
    """Return match directory names sorted by date."""
    matches = []
    for d in ATOMIC_DIR.iterdir():
        if d.is_dir() and d.name.startswith("20"):
            clips = list(d.glob("*.mp4"))
            if clips:  # skip empty dirs
                matches.append(d.name)
    matches.sort(reverse=newest_first)
    return matches


def list_clips(match_name: str) -> list[str]:
    """Return sorted list of .mp4 clip filenames in a match dir."""
    match_dir = ATOMIC_DIR / match_name
    clips = sorted(f.name for f in match_dir.glob("*.mp4"))
    return clips


def render_clip(match_name: str, clip_name: str, out_dir: Path, resume: bool = False, yolo_model: str | None = None, use_ball_telemetry: bool = True) -> dict:
    """Render one clip and return parsed metrics."""
    clip_path = ATOMIC_DIR / match_name / clip_name
    stem = Path(clip_name).stem
    out_path = out_dir / f"{stem}__portrait.mp4"
    event_type = extract_event_type(clip_name)

    result = {
        "clip": clip_name,
        "stem": stem,
        "event_type": event_type,
        "ball_in_crop_pct": None,
        "ball_in_crop_frac": None,
        "max_escape_px": None,
        "yolo_confirmed_pct": None,
        "yolo_density_pct": None,
        "validation": None,
        "shot_hold": False,
        "pan_hold": False,
        "goal_infer": False,
        "hold_exit": False,
        "warnings": [],
        "exit_code": None,
        "render_time_s": None,
        "skipped": False,
    }

    # Skip if already rendered and --resume
    if resume and out_path.exists() and out_path.stat().st_size > 0:
        result["skipped"] = True
        result["exit_code"] = 0
        # Try to read metrics from existing summary if available
        diag_csv = out_path.with_suffix(".diag.csv")
        if diag_csv.exists():
            result["validation"] = "SKIPPED (resume)"
        return result

    cmd = [
        sys.executable,
        str(RENDERER),
        "--in", str(clip_path),
        "--src", str(clip_path),
        "--out", str(out_path),
        "--portrait", "1080x1920",
        "--preset", "cinematic",
        "--fps", "24",
        "--diagnostics",
    ]
    if use_ball_telemetry:
        cmd.append("--use-ball-telemetry")
    if event_type:
        cmd.extend(["--event-type", event_type])
    if yolo_model:
        cmd.extend(["--yolo-model", yolo_model])

    # Auto-detect manual anchors: look for {clip_num}__manual_anchors.csv
    clip_num = stem.split("__")[0]  # e.g. "001"
    anchors_path = out_dir / f"{clip_num}__manual_anchors.csv"
    if anchors_path.exists() and anchors_path.stat().st_size > 20:
        cmd.extend(["--manual-anchors", str(anchors_path)])

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=900,
            encoding="utf-8",
            errors="replace",
        )
        result["exit_code"] = proc.returncode
        result["render_time_s"] = round(time.time() - t0, 1)
        output = proc.stdout + "\n" + proc.stderr

        # Parse metrics from output
        m = re.search(r"Ball in crop:\s*(\d+)/(\d+)\s*\(([\d.]+)%\)", output)
        if m:
            result["ball_in_crop_pct"] = float(m.group(3))
            result["ball_in_crop_frac"] = f"{m.group(1)}/{m.group(2)}"

        m = re.search(r"Max escape:\s*([\d.]+)\s*px", output)
        if m:
            result["max_escape_px"] = float(m.group(1))

        m = re.search(r"YOLO-confirmed in crop:\s*([\d.]+)%", output)
        if m:
            result["yolo_confirmed_pct"] = float(m.group(1))

        # YOLO density (how many frames have YOLO detections)
        m = re.search(r"YOLO density:\s*([\d.]+)%", output)
        if m:
            result["yolo_density_pct"] = float(m.group(1))
        else:
            # Try to infer from source breakdown
            m2 = re.search(r"yolo=(\d+)", output)
            m3 = re.search(r"Ball in crop:\s*\d+/(\d+)", output)
            if m2 and m3:
                yolo_n = int(m2.group(1))
                total_n = int(m3.group(1))
                if total_n > 0:
                    result["yolo_density_pct"] = round(100.0 * yolo_n / total_n, 1)

        if "PASSED" in output:
            result["validation"] = "PASSED"
        elif "FAILED" in output:
            result["validation"] = "FAILED"

        if "Shot-hold:" in output or "shot_hold" in output.lower():
            result["shot_hold"] = True
        if "Pan-hold:" in output or "pan_hold" in output.lower():
            result["pan_hold"] = True
        if "GOAL-INFER" in output or "goal_event_inferred" in output:
            result["goal_infer"] = True
        if "hold_exit=" in output:
            result["hold_exit"] = True

        for line in output.split("\n"):
            if "WARNING" in line:
                result["warnings"].append(line.strip())

    except subprocess.TimeoutExpired:
        result["exit_code"] = -1
        result["render_time_s"] = round(time.time() - t0, 1)
    except Exception as e:
        result["exit_code"] = -2
        result["render_time_s"] = round(time.time() - t0, 1)
        result["warnings"].append(f"EXCEPTION: {e}")

    return result


def write_summary(results: list[dict], out_dir: Path, match_name: str):
    """Write match-level summary report and CSV."""
    # --- Text summary ---
    summary_path = out_dir / "_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"PORTRAIT REEL RENDER SUMMARY\n")
        f.write(f"Match: {match_name}\n")
        f.write(f"Date:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Clips: {len(results)}\n")
        f.write(f"{'='*100}\n\n")

        # Classification buckets
        excellent = []  # >= 98%
        good = []       # >= 90%
        marginal = []   # >= 80%
        poor = []       # < 80%
        failed = []     # render error
        skipped = []

        for r in results:
            if r["skipped"]:
                skipped.append(r)
            elif r["exit_code"] != 0:
                failed.append(r)
            elif r["ball_in_crop_pct"] is None:
                failed.append(r)
            elif r["ball_in_crop_pct"] >= 98.0:
                excellent.append(r)
            elif r["ball_in_crop_pct"] >= 90.0:
                good.append(r)
            elif r["ball_in_crop_pct"] >= 80.0:
                marginal.append(r)
            else:
                poor.append(r)

        f.write(f"QUALITY BREAKDOWN:\n")
        f.write(f"  Excellent (>=98%): {len(excellent)}\n")
        f.write(f"  Good     (>=90%):  {len(good)}\n")
        f.write(f"  Marginal (>=80%):  {len(marginal)}\n")
        f.write(f"  Poor     (<80%):   {len(poor)}\n")
        f.write(f"  Failed:            {len(failed)}\n")
        f.write(f"  Skipped (resume):  {len(skipped)}\n")
        f.write(f"\n")

        # Overall stats (excluding skipped/failed)
        valid = [r for r in results if r["ball_in_crop_pct"] is not None and not r["skipped"]]
        if valid:
            avg_bic = sum(r["ball_in_crop_pct"] for r in valid) / len(valid)
            min_bic = min(r["ball_in_crop_pct"] for r in valid)
            max_esc = max((r["max_escape_px"] or 0) for r in valid)
            total_time = sum((r["render_time_s"] or 0) for r in valid)
            f.write(f"OVERALL STATS:\n")
            f.write(f"  Avg ball-in-crop: {avg_bic:.1f}%\n")
            f.write(f"  Min ball-in-crop: {min_bic:.1f}%\n")
            f.write(f"  Max escape:       {max_esc:.0f}px\n")
            f.write(f"  Total render:     {total_time:.0f}s ({total_time/60:.1f}min)\n")
            f.write(f"\n")

        # Per-clip detail table
        f.write(f"{'Clip':<8} {'Event':<10} {'BIC':>7} {'MaxEsc':>8} {'YOLO%':>7} {'YOLOConf':>9} {'Time':>6} {'Status':<10} {'Flags'}\n")
        f.write(f"{'-'*8} {'-'*10} {'-'*7} {'-'*8} {'-'*7} {'-'*9} {'-'*6} {'-'*10} {'-'*20}\n")

        for r in results:
            clip_id = r["stem"][:7]
            evt = r["event_type"] or "-"
            bic = f"{r['ball_in_crop_pct']:.1f}%" if r["ball_in_crop_pct"] is not None else "N/A"
            esc = f"{r['max_escape_px']:.0f}px" if r["max_escape_px"] is not None else "N/A"
            yd = f"{r['yolo_density_pct']:.0f}%" if r["yolo_density_pct"] is not None else "?"
            yc = f"{r['yolo_confirmed_pct']:.1f}%" if r["yolo_confirmed_pct"] is not None else "N/A"
            tm = f"{r['render_time_s']:.0f}s" if r["render_time_s"] is not None else "-"
            flags = []
            if r["shot_hold"]: flags.append("SH")
            if r["pan_hold"]: flags.append("PH")
            if r["goal_infer"]: flags.append("GI")
            if r["hold_exit"]: flags.append("HE")

            if r["skipped"]:
                status = "SKIP"
            elif r["exit_code"] != 0:
                status = "FAIL"
            elif r["ball_in_crop_pct"] is not None and r["ball_in_crop_pct"] >= 98.0:
                status = "EXCELLENT"
            elif r["ball_in_crop_pct"] is not None and r["ball_in_crop_pct"] >= 90.0:
                status = "GOOD"
            elif r["ball_in_crop_pct"] is not None and r["ball_in_crop_pct"] >= 80.0:
                status = "MARGINAL"
            else:
                status = "POOR"

            f.write(f"{clip_id:<8} {evt:<10} {bic:>7} {esc:>8} {yd:>7} {yc:>9} {tm:>6} {status:<10} {', '.join(flags)}\n")

        # Clips that need attention
        attention = poor + failed
        if attention:
            f.write(f"\n{'='*100}\n")
            f.write(f"CLIPS NEEDING ATTENTION:\n")
            for r in attention:
                f.write(f"\n  {r['stem']}\n")
                f.write(f"    Event: {r['event_type'] or 'none'}\n")
                f.write(f"    BIC:   {r['ball_in_crop_pct']}%\n")
                f.write(f"    Escape: {r['max_escape_px']}px\n")
                f.write(f"    Exit code: {r['exit_code']}\n")
                if r["warnings"]:
                    for w in r["warnings"][:5]:
                        f.write(f"    WARN: {w}\n")

    # --- CSV summary (machine-readable) ---
    csv_path = out_dir / "_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "clip_id", "stem", "event_type", "ball_in_crop_pct",
            "max_escape_px", "yolo_density_pct", "yolo_confirmed_pct",
            "validation", "shot_hold", "pan_hold", "goal_infer",
            "hold_exit", "render_time_s", "exit_code", "skipped",
            "warning_count",
        ])
        for r in results:
            clip_id = r["stem"][:3] if len(r["stem"]) >= 3 else r["stem"]
            w.writerow([
                clip_id, r["stem"], r["event_type"] or "",
                r["ball_in_crop_pct"] or "",
                r["max_escape_px"] or "",
                r["yolo_density_pct"] or "",
                r["yolo_confirmed_pct"] or "",
                r["validation"] or "",
                int(r["shot_hold"]), int(r["pan_hold"]),
                int(r["goal_infer"]), int(r["hold_exit"]),
                r["render_time_s"] or "",
                r["exit_code"],
                int(r["skipped"]),
                len(r["warnings"]),
            ])

    return summary_path, csv_path


def render_match(match_name: str, resume: bool = False, yolo_model: str | None = None, use_ball_telemetry: bool = True):
    """Render all clips in a match."""
    clips = list_clips(match_name)
    if not clips:
        print(f"No clips found for {match_name}")
        return

    out_dir = OUT_ROOT / match_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*80}")
    print(f"# MATCH: {match_name}")
    print(f"# Clips: {len(clips)}")
    print(f"# Output: {out_dir}")
    print(f"# Resume: {resume}")
    if yolo_model:
        print(f"# YOLO model: {yolo_model}")
    if use_ball_telemetry:
        print(f"# Ball telemetry: ON (YOLO+Tracker+Spline)")
    print(f"{'#'*80}\n")

    results = []
    for i, clip_name in enumerate(clips, 1):
        print(f"\n[{i}/{len(clips)}] {clip_name}")
        r = render_clip(match_name, clip_name, out_dir, resume=resume, yolo_model=yolo_model, use_ball_telemetry=use_ball_telemetry)
        results.append(r)

        if r["skipped"]:
            print(f"  -> SKIPPED (already rendered)")
        elif r["exit_code"] == 0:
            bic = f"{r['ball_in_crop_pct']:.1f}%" if r["ball_in_crop_pct"] is not None else "N/A"
            esc = f"{r['max_escape_px']:.0f}px" if r["max_escape_px"] is not None else "N/A"
            tm = f"{r['render_time_s']:.0f}s" if r["render_time_s"] is not None else "?"
            print(f"  -> BIC={bic}  MaxEsc={esc}  Time={tm}")
        else:
            print(f"  -> FAILED (exit code {r['exit_code']})")

    # Write summaries
    summary_path, csv_path = write_summary(results, out_dir, match_name)
    print(f"\n{'='*80}")
    print(f"MATCH COMPLETE: {match_name}")
    print(f"Summary: {summary_path}")
    print(f"CSV:     {csv_path}")

    # Quick console summary
    valid = [r for r in results if r["ball_in_crop_pct"] is not None and not r["skipped"]]
    skipped = sum(1 for r in results if r["skipped"])
    failed = sum(1 for r in results if r["exit_code"] != 0 and not r["skipped"])
    if valid:
        avg = sum(r["ball_in_crop_pct"] for r in valid) / len(valid)
        worst = min(valid, key=lambda x: x["ball_in_crop_pct"])
        print(f"Avg BIC: {avg:.1f}%  |  Worst: {worst['stem'][:20]} @ {worst['ball_in_crop_pct']:.1f}%")
    print(f"Rendered: {len(valid)}  Skipped: {skipped}  Failed: {failed}")
    print(f"{'='*80}\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch render portrait reels per match")
    parser.add_argument("match", nargs="?", help="Match directory name (e.g. '2026-02-23__TSC_vs_NEOFC')")
    parser.add_argument("--all", action="store_true", help="Render all matches, newest first")
    parser.add_argument("--resume", action="store_true", help="Skip clips that already have output")
    parser.add_argument("--yolo-model", dest="yolo_model", default=None,
                        help="YOLO model weights (e.g. yolov8m.pt, yolov8x.pt)")
    parser.add_argument("--no-ball-telemetry", dest="use_ball_telemetry",
                        action="store_false", default=True,
                        help="Disable ball telemetry (YOLO+Tracker+Spline). Default: ON")
    parser.add_argument("--list", action="store_true", help="List available matches and exit")
    args = parser.parse_args()

    if args.list:
        matches = list_matches()
        print("Available matches (newest first):")
        for m in matches:
            clips = list_clips(m)
            print(f"  {m}  ({len(clips)} clips)")
        return

    if args.all:
        matches = list_matches()
        print(f"Rendering {len(matches)} matches, newest first...\n")
        for m in matches:
            render_match(m, resume=args.resume, yolo_model=args.yolo_model, use_ball_telemetry=args.use_ball_telemetry)
    elif args.match:
        # Accept partial match name
        if not (ATOMIC_DIR / args.match).exists():
            matches = list_matches(newest_first=False)
            candidates = [m for m in matches if args.match.lower() in m.lower()]
            if len(candidates) == 1:
                args.match = candidates[0]
            elif candidates:
                print(f"Ambiguous match name '{args.match}'. Candidates:")
                for c in candidates:
                    print(f"  {c}")
                return
            else:
                print(f"No match found for '{args.match}'")
                return
        render_match(args.match, resume=args.resume, yolo_model=args.yolo_model, use_ball_telemetry=args.use_ball_telemetry)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

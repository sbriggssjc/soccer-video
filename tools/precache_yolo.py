#!/usr/bin/env python3
"""Pre-cache YOLO ball detections for all clips in a match.

Runs ball detection only (no rendering) using the specified YOLO model.
Cached detections are saved to out/telemetry/ and will be reused by
render_follow_unified.py, making subsequent renders fast.

Usage:
    python tools/precache_yolo.py "2026-02-23__TSC_vs_NEOFC" --model yolov8x.pt
    python tools/precache_yolo.py --all --model yolov8m.pt
    python tools/precache_yolo.py --all --model yolov8x.pt --resume
"""

import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass

REPO = Path(__file__).resolve().parent.parent
ATOMIC_DIR = REPO / "out" / "atomic_clips"

# Import detection function
sys.path.insert(0, str(REPO))
from tools.ball_telemetry import (
    run_yolo_ball_detection,
    yolo_telemetry_path_for_video,
    DEFAULT_MODEL_NAME,
)


def list_matches(newest_first: bool = True) -> list[str]:
    matches = []
    for d in ATOMIC_DIR.iterdir():
        if d.is_dir() and d.name.startswith("20"):
            clips = list(d.glob("*.mp4"))
            if clips:
                matches.append(d.name)
    matches.sort(reverse=newest_first)
    return matches


def list_clips(match_name: str) -> list[str]:
    match_dir = ATOMIC_DIR / match_name
    return sorted(f.name for f in match_dir.glob("*.mp4"))


def cache_path_for(video_path: Path, model_name: str) -> Path:
    """Compute the cache path for a given model."""
    base = Path(yolo_telemetry_path_for_video(video_path))
    model_stem = Path(model_name).stem
    default_stem = Path(DEFAULT_MODEL_NAME).stem
    if model_stem == default_stem:
        return base
    return base.with_suffix(f".{model_stem}.jsonl")


def precache_match(match_name: str, model_name: str, resume: bool = False):
    """Run YOLO detection on all clips in a match."""
    clips = list_clips(match_name)
    if not clips:
        print(f"No clips found for {match_name}")
        return

    print(f"\n{'#'*70}")
    print(f"# PRE-CACHE: {match_name}")
    print(f"# Model: {model_name}")
    print(f"# Clips: {len(clips)}")
    print(f"{'#'*70}\n")

    total_time = 0.0
    cached = 0
    skipped = 0
    failed = 0

    for i, clip_name in enumerate(clips, 1):
        clip_path = ATOMIC_DIR / match_name / clip_name
        cp = cache_path_for(clip_path, model_name)

        if resume and cp.is_file() and cp.stat().st_size > 0:
            print(f"[{i}/{len(clips)}] SKIP (cached): {clip_name[:60]}")
            skipped += 1
            continue

        print(f"[{i}/{len(clips)}] {clip_name[:60]}...", end=" ", flush=True)
        t0 = time.time()
        try:
            samples = run_yolo_ball_detection(
                clip_path,
                min_conf=0.20,
                cache=True,
                model_name=model_name,
            )
            elapsed = time.time() - t0
            total_time += elapsed
            cached += 1
            n_dets = len(samples)
            avg_conf = sum(s.conf for s in samples) / n_dets if n_dets > 0 else 0
            print(f"{n_dets} dets, avg_conf={avg_conf:.2f}, {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - t0
            total_time += elapsed
            failed += 1
            print(f"FAILED ({e})")

    print(f"\n{'='*70}")
    print(f"PRE-CACHE COMPLETE: {match_name}")
    print(f"Cached: {cached} | Skipped: {skipped} | Failed: {failed}")
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f}min)")
    if cached > 0:
        print(f"Avg per clip: {total_time/cached:.1f}s")
    print(f"{'='*70}\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pre-cache YOLO ball detections")
    parser.add_argument("match", nargs="?", help="Match directory name")
    parser.add_argument("--model", default="yolov8x.pt", help="YOLO model (default: yolov8x.pt)")
    parser.add_argument("--all", action="store_true", help="Process all matches, newest first")
    parser.add_argument("--resume", action="store_true", help="Skip clips with existing cache")
    parser.add_argument("--list", action="store_true", help="List matches and exit")
    args = parser.parse_args()

    if args.list:
        matches = list_matches()
        print("Available matches (newest first):")
        for m in matches:
            clips = list_clips(m)
            # Check cache status for each match
            clip_paths = [ATOMIC_DIR / m / c for c in clips]
            cached = sum(1 for cp in clip_paths if cache_path_for(cp, args.model).is_file())
            print(f"  {m}  ({len(clips)} clips, {cached} cached for {args.model})")
        return

    if args.all:
        matches = list_matches()
        for m in matches:
            precache_match(m, args.model, resume=args.resume)
    elif args.match:
        if not (ATOMIC_DIR / args.match).exists():
            matches = list_matches(newest_first=False)
            candidates = [m for m in matches if args.match.lower() in m.lower()]
            if len(candidates) == 1:
                args.match = candidates[0]
            elif candidates:
                print(f"Ambiguous. Candidates: {candidates}")
                return
            else:
                print(f"No match found for '{args.match}'")
                return
        precache_match(args.match, args.model, resume=args.resume)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

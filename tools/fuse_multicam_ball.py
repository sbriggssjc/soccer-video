#!/usr/bin/env python
"""Fuse ball telemetry from two camera angles (XBotGo + sideline) for improved tracking.

Given a game with both XBotGo (overhead) and sideline (ground-level) atomic clips,
this tool:

1. Pairs clips by clip number (e.g., 007__Cross & Shot matches
   007__...__SIDELINE__CROSS_AND_SHOT)
2. Loads ball telemetry from both cameras
3. Fits a linear mapping from sideline X -> XBotGo X using high-confidence
   frames where both cameras see the ball
4. Fuses the two telemetry streams: when one camera loses the ball, the other
   can fill in the gap
5. Writes fused telemetry as .ball_fused.jsonl alongside the XBotGo telemetry

The fused telemetry is compatible with the existing portrait planner pipeline.
The render_follow_unified.py and offline_portrait_planner.py can consume it
as a drop-in replacement for .ball.jsonl telemetry.

Usage:
    # Preview what would be fused (dry-run)
    python tools/fuse_multicam_ball.py --game 2026-03-01__TSC_vs_OK_Celtic --dry-run

    # Run fusion for a game
    python tools/fuse_multicam_ball.py --game 2026-03-01__TSC_vs_OK_Celtic

    # Run fusion for a specific clip pair
    python tools/fuse_multicam_ball.py \\
        --xbotgo out/atomic_clips/.../007__Cross_Shot__390-401.mp4 \\
        --sideline out/atomic_clips/.../sideline/007__...__SIDELINE__CROSS_AND_SHOT.mp4

    # Re-render with fused telemetry
    python tools/render_follow_unified.py --in clip.mp4 --ball-telemetry clip.ball_fused.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.ball_telemetry import (
    BallSample,
    load_ball_telemetry,
    telemetry_path_for_video,
)


# ---------------------------------------------------------------------------
# Data structures


@dataclass
class ClipPair:
    """A paired XBotGo + sideline clip covering the same play."""

    clip_number: int
    xbotgo_clip: Path
    sideline_clip: Path
    xbotgo_telemetry: Optional[Path] = None
    sideline_telemetry: Optional[Path] = None


@dataclass
class FusionResult:
    """Result of fusing telemetry from two cameras."""

    clip_number: int
    xbotgo_clip: str
    sideline_clip: str
    xbotgo_samples: int
    sideline_samples: int
    fused_samples: int
    # How many frames were filled in from the other camera
    xbotgo_filled: int  # frames filled from sideline when xbotgo lost ball
    sideline_filled: int  # frames filled from xbotgo when sideline lost ball
    # Mapping quality
    mapping_r2: float  # R-squared of the linear X mapping
    high_conf_pairs: int  # frames with high confidence in both cameras
    output_path: Optional[str] = None


@dataclass
class LinearMapping:
    """Linear mapping: y = slope * x + intercept."""

    slope: float = 1.0
    intercept: float = 0.0
    r_squared: float = 0.0
    n_points: int = 0

    def predict(self, x: float) -> float:
        return self.slope * x + self.intercept


# ---------------------------------------------------------------------------
# Clip pairing


def _extract_clip_number(filename: str) -> Optional[int]:
    """Extract leading clip number from a filename like '007__Cross...'."""
    m = re.match(r"^(\d{1,4})(?:__|_)", filename)
    if m:
        return int(m.group(1))
    return None


def _is_sideline(path: Path) -> bool:
    """Check if a clip path is a sideline clip."""
    return "sideline" in str(path).lower() or "SIDELINE" in path.stem


def discover_clip_pairs(game_dir: Path) -> list[ClipPair]:
    """Find matching XBotGo + sideline clips in a game directory."""
    xbotgo_clips: dict[int, Path] = {}
    sideline_clips: dict[int, list[Path]] = {}

    # Scan all mp4 files
    for mp4 in sorted(game_dir.rglob("*.mp4")):
        clip_num = _extract_clip_number(mp4.name)
        if clip_num is None:
            continue

        if _is_sideline(mp4):
            sideline_clips.setdefault(clip_num, []).append(mp4)
        else:
            # Prefer the first non-sideline clip found for each number
            if clip_num not in xbotgo_clips:
                xbotgo_clips[clip_num] = mp4

    # Build pairs — only where we have both cameras
    pairs = []
    for clip_num in sorted(set(xbotgo_clips.keys()) & set(sideline_clips.keys())):
        xbotgo = xbotgo_clips[clip_num]
        # If multiple sideline parts, use the first (longest) one
        sideline = sideline_clips[clip_num][0]

        # Find telemetry files
        xb_tele = _find_telemetry(xbotgo)
        sl_tele = _find_telemetry(sideline)

        pairs.append(ClipPair(
            clip_number=clip_num,
            xbotgo_clip=xbotgo,
            sideline_clip=sideline,
            xbotgo_telemetry=xb_tele,
            sideline_telemetry=sl_tele,
        ))

    return pairs


def _find_telemetry(clip_path: Path) -> Optional[Path]:
    """Find ball telemetry file for a clip, checking multiple locations."""
    candidates = [
        # Canonical location: out/telemetry/<stem>.ball.jsonl
        Path(telemetry_path_for_video(clip_path)),
        # Next to clip
        clip_path.with_suffix(".ball.jsonl"),
        clip_path.with_suffix(".ball_path.jsonl"),
        clip_path.with_suffix(".telemetry.jsonl"),
    ]

    # Also check the REPO_ROOT out/telemetry/ with absolute path
    tele_dir = ROOT / "out" / "telemetry"
    if tele_dir.is_dir():
        candidates.append(tele_dir / f"{clip_path.stem}.ball.jsonl")
        candidates.append(tele_dir / f"{clip_path.stem}.ball_path.jsonl")

    for p in candidates:
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


# ---------------------------------------------------------------------------
# Time alignment and mapping


def _samples_to_time_series(
    samples: list[BallSample], fps: float = 30.0
) -> dict[int, BallSample]:
    """Convert samples to a dict keyed by frame number."""
    by_frame: dict[int, BallSample] = {}
    for s in samples:
        f = s.frame
        # Keep highest-confidence detection per frame
        if f not in by_frame or s.conf > by_frame[f].conf:
            by_frame[f] = s
    return by_frame


def fit_horizontal_mapping(
    xbotgo_samples: dict[int, BallSample],
    sideline_samples: dict[int, BallSample],
    min_conf: float = 0.4,
) -> LinearMapping:
    """Fit a linear mapping from sideline X -> XBotGo X.

    Uses frames where both cameras have high-confidence detections
    to fit: xbotgo_x = slope * sideline_x + intercept.

    This works because both cameras view the field along roughly the same
    horizontal axis (lengthwise), so horizontal position correlates strongly.
    """
    # Collect paired X values from high-confidence frames
    xb_xs = []
    sl_xs = []

    common_frames = set(xbotgo_samples.keys()) & set(sideline_samples.keys())
    for f in sorted(common_frames):
        xb = xbotgo_samples[f]
        sl = sideline_samples[f]
        if xb.conf >= min_conf and sl.conf >= min_conf:
            if math.isfinite(xb.x) and math.isfinite(sl.x):
                xb_xs.append(xb.x)
                sl_xs.append(sl.x)

    n = len(xb_xs)
    if n < 3:
        # Not enough data for a reliable mapping
        return LinearMapping(n_points=n)

    # Linear regression: xbotgo_x = slope * sideline_x + intercept
    xb_arr = np.array(xb_xs, dtype=np.float64)
    sl_arr = np.array(sl_xs, dtype=np.float64)

    sl_mean = np.mean(sl_arr)
    xb_mean = np.mean(xb_arr)

    ss_xy = np.sum((sl_arr - sl_mean) * (xb_arr - xb_mean))
    ss_xx = np.sum((sl_arr - sl_mean) ** 2)

    if ss_xx < 1e-10:
        # All sideline X values are the same — can't fit
        return LinearMapping(intercept=float(xb_mean), n_points=n)

    slope = float(ss_xy / ss_xx)
    intercept = float(xb_mean - slope * sl_mean)

    # R-squared
    predicted = slope * sl_arr + intercept
    ss_res = np.sum((xb_arr - predicted) ** 2)
    ss_tot = np.sum((xb_arr - xb_mean) ** 2)
    r_squared = 1.0 - (ss_res / max(ss_tot, 1e-10)) if ss_tot > 1e-10 else 0.0

    return LinearMapping(
        slope=slope,
        intercept=intercept,
        r_squared=float(r_squared),
        n_points=n,
    )


# ---------------------------------------------------------------------------
# Fusion


def fuse_telemetry(
    xbotgo_samples: list[BallSample],
    sideline_samples: list[BallSample],
    mapping: LinearMapping,
    *,
    xbotgo_width: int = 1920,
    xbotgo_height: int = 1080,
    conf_threshold: float = 0.3,
    mapping_conf_discount: float = 0.6,
) -> list[dict]:
    """Fuse two telemetry streams into a single timeline.

    Strategy:
    - When XBotGo has high-confidence detection: use it directly
    - When only sideline sees the ball: map sideline X to XBotGo X,
      use XBotGo Y center (since sideline Y doesn't map well), apply
      confidence discount
    - When both see it but XBotGo is low-conf: weighted blend
    - When neither sees it: emit no sample (gap)

    Returns list of dicts compatible with save_ball_telemetry_jsonl().
    """
    xb_by_frame = _samples_to_time_series(xbotgo_samples)
    sl_by_frame = _samples_to_time_series(sideline_samples)

    # Determine frame range
    all_frames = set(xb_by_frame.keys()) | set(sl_by_frame.keys())
    if not all_frames:
        return []

    min_frame = min(all_frames)
    max_frame = max(all_frames)

    fused: list[dict] = []
    xbotgo_filled = 0
    sideline_filled = 0

    for f in range(min_frame, max_frame + 1):
        xb = xb_by_frame.get(f)
        sl = sl_by_frame.get(f)

        xb_good = (
            xb is not None
            and xb.conf >= conf_threshold
            and math.isfinite(xb.x)
            and math.isfinite(xb.y)
        )
        sl_good = (
            sl is not None
            and sl.conf >= conf_threshold
            and math.isfinite(sl.x)
            and mapping.n_points >= 3
        )

        if xb_good and sl_good:
            # Both cameras see the ball — weighted average
            xb_weight = xb.conf
            # Map sideline X to XBotGo X space
            sl_mapped_x = mapping.predict(sl.x)
            sl_weight = sl.conf * mapping_conf_discount * mapping.r_squared

            total_weight = xb_weight + sl_weight
            if total_weight > 0:
                fused_x = (xb.x * xb_weight + sl_mapped_x * sl_weight) / total_weight
                fused_y = xb.y  # Y from XBotGo only (sideline Y doesn't map)
                fused_conf = min(1.0, (xb.conf + sl.conf * mapping.r_squared) / 2)
            else:
                fused_x = xb.x
                fused_y = xb.y
                fused_conf = xb.conf

            t_val = xb.t if math.isfinite(xb.t) else (sl.t if sl and math.isfinite(sl.t) else 0.0)

            fused.append({
                "frame": f,
                "t": round(t_val, 4),
                "cx": round(float(np.clip(fused_x, 0, xbotgo_width)), 1),
                "cy": round(float(np.clip(fused_y, 0, xbotgo_height)), 1),
                "conf": round(fused_conf, 3),
                "src": "both",
            })

        elif xb_good:
            # Only XBotGo sees it
            fused.append({
                "frame": f,
                "t": round(xb.t, 4),
                "cx": round(float(np.clip(xb.x, 0, xbotgo_width)), 1),
                "cy": round(float(np.clip(xb.y, 0, xbotgo_height)), 1),
                "conf": round(xb.conf, 3),
                "src": "xbotgo",
            })

        elif sl_good:
            # Only sideline sees it — map to XBotGo coordinates
            sl_mapped_x = mapping.predict(sl.x)
            # For Y, use a reasonable default (field center or last known Y)
            last_y = xbotgo_height * 0.55  # default: slightly below center
            # Look back for last known XBotGo Y
            for prev_f in range(f - 1, max(f - 30, min_frame - 1), -1):
                prev_xb = xb_by_frame.get(prev_f)
                if prev_xb and math.isfinite(prev_xb.y) and prev_xb.conf >= conf_threshold:
                    last_y = prev_xb.y
                    break

            t_val = sl.t if math.isfinite(sl.t) else 0.0
            mapped_conf = sl.conf * mapping_conf_discount * mapping.r_squared

            fused.append({
                "frame": f,
                "t": round(t_val, 4),
                "cx": round(float(np.clip(sl_mapped_x, 0, xbotgo_width)), 1),
                "cy": round(float(np.clip(last_y, 0, xbotgo_height)), 1),
                "conf": round(mapped_conf, 3),
                "src": "sideline_mapped",
            })
            xbotgo_filled += 1

        # else: neither camera sees the ball — gap (no sample emitted)

    return fused


# ---------------------------------------------------------------------------
# Per-pair processing


def process_clip_pair(
    pair: ClipPair,
    out_dir: Path,
    *,
    dry_run: bool = False,
    verbose: bool = False,
) -> Optional[FusionResult]:
    """Process a single clip pair and write fused telemetry."""
    # Load telemetry
    if not pair.xbotgo_telemetry or not pair.sideline_telemetry:
        missing = []
        if not pair.xbotgo_telemetry:
            missing.append("xbotgo")
        if not pair.sideline_telemetry:
            missing.append("sideline")
        print(f"  [SKIP] Clip {pair.clip_number:03d}: missing telemetry for {', '.join(missing)}")
        return None

    try:
        xb_samples = load_ball_telemetry(pair.xbotgo_telemetry)
        sl_samples = load_ball_telemetry(pair.sideline_telemetry)
    except Exception as e:
        print(f"  [ERROR] Clip {pair.clip_number:03d}: failed to load telemetry: {e}")
        return None

    if not xb_samples and not sl_samples:
        print(f"  [SKIP] Clip {pair.clip_number:03d}: both telemetry files are empty")
        return None

    # Build frame-indexed maps
    xb_by_frame = _samples_to_time_series(xb_samples)
    sl_by_frame = _samples_to_time_series(sl_samples)

    # Fit horizontal mapping
    mapping = fit_horizontal_mapping(xb_by_frame, sl_by_frame)

    if verbose:
        print(f"  [MAP] Clip {pair.clip_number:03d}: "
              f"slope={mapping.slope:.3f}, intercept={mapping.intercept:.1f}, "
              f"R²={mapping.r_squared:.3f}, n={mapping.n_points}")

    # Fuse
    fused_samples = fuse_telemetry(xb_samples, sl_samples, mapping)

    # Count gap fills
    xbotgo_filled = sum(1 for s in fused_samples if s.get("src") == "sideline_mapped")
    both_frames = sum(1 for s in fused_samples if s.get("src") == "both")

    result = FusionResult(
        clip_number=pair.clip_number,
        xbotgo_clip=str(pair.xbotgo_clip),
        sideline_clip=str(pair.sideline_clip),
        xbotgo_samples=len(xb_samples),
        sideline_samples=len(sl_samples),
        fused_samples=len(fused_samples),
        xbotgo_filled=xbotgo_filled,
        sideline_filled=0,  # we only fuse into XBotGo space for now
        mapping_r2=mapping.r_squared,
        high_conf_pairs=mapping.n_points,
    )

    if dry_run:
        return result

    # Write fused telemetry
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"{pair.xbotgo_clip.stem}.ball_fused.jsonl"
    out_path = out_dir / out_name

    with out_path.open("w", encoding="utf-8") as fh:
        for rec in fused_samples:
            # Strip the src field for compatibility with existing loaders
            clean = {k: v for k, v in rec.items() if k != "src"}
            fh.write(json.dumps(clean) + "\n")

    result.output_path = str(out_path)

    # Also write a fusion report alongside
    report_path = out_path.with_suffix(".fusion_report.json")
    report = {
        "clip_number": pair.clip_number,
        "xbotgo_clip": str(pair.xbotgo_clip.name),
        "sideline_clip": str(pair.sideline_clip.name),
        "xbotgo_samples": len(xb_samples),
        "sideline_samples": len(sl_samples),
        "fused_samples": len(fused_samples),
        "xbotgo_filled_from_sideline": xbotgo_filled,
        "both_cameras_agreed": both_frames,
        "mapping": {
            "slope": round(mapping.slope, 4),
            "intercept": round(mapping.intercept, 2),
            "r_squared": round(mapping.r_squared, 4),
            "calibration_points": mapping.n_points,
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return result


# ---------------------------------------------------------------------------
# CLI


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fuse ball telemetry from XBotGo + sideline cameras.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--game",
        help="Game folder name (e.g., 2026-03-01__TSC_vs_OK_Celtic)",
    )
    p.add_argument(
        "--xbotgo",
        help="Specific XBotGo clip path (for single-pair mode)",
    )
    p.add_argument(
        "--sideline",
        help="Specific sideline clip path (for single-pair mode)",
    )
    p.add_argument(
        "--src-dir",
        default="out/atomic_clips",
        help="Root directory for atomic clips (default: out/atomic_clips)",
    )
    p.add_argument(
        "--out-dir",
        default="out/telemetry",
        help="Output directory for fused telemetry (default: out/telemetry)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fused without writing files",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed fusion statistics",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = ROOT / args.out_dir

    # --- Single-pair mode ---
    if args.xbotgo and args.sideline:
        xb_path = Path(args.xbotgo)
        sl_path = Path(args.sideline)
        clip_num = _extract_clip_number(xb_path.name) or 0

        pair = ClipPair(
            clip_number=clip_num,
            xbotgo_clip=xb_path,
            sideline_clip=sl_path,
            xbotgo_telemetry=_find_telemetry(xb_path),
            sideline_telemetry=_find_telemetry(sl_path),
        )

        result = process_clip_pair(pair, out_dir, dry_run=args.dry_run, verbose=True)
        if result:
            _print_result(result)
        return 0

    # --- Game mode ---
    if not args.game:
        print("[ERROR] Specify --game or --xbotgo + --sideline")
        return 1

    src_root = ROOT / args.src_dir
    game_dir = src_root / args.game
    if not game_dir.is_dir():
        # Try substring match
        candidates = [d for d in src_root.iterdir() if d.is_dir() and args.game.lower() in d.name.lower()]
        if len(candidates) == 1:
            game_dir = candidates[0]
        elif candidates:
            print(f"[ERROR] Ambiguous game match. Candidates:")
            for c in candidates:
                print(f"  {c.name}")
            return 1
        else:
            print(f"[ERROR] Game directory not found: {game_dir}")
            return 1

    print(f"[FUSE] Scanning {game_dir.name} for XBotGo + sideline pairs...")
    pairs = discover_clip_pairs(game_dir)

    if not pairs:
        print("[WARN] No matching clip pairs found.")
        return 0

    print(f"[FUSE] Found {len(pairs)} clip pairs\n")

    # Summary header
    print(f"{'Clip':>5}  {'XBotGo':>8}  {'Sideline':>10}  {'Fused':>7}  "
          f"{'Filled':>7}  {'R²':>6}  {'Pairs':>6}  Status")
    print("-" * 78)

    results: list[FusionResult] = []
    for pair in pairs:
        tele_status = []
        if not pair.xbotgo_telemetry:
            tele_status.append("no-xbotgo-tele")
        if not pair.sideline_telemetry:
            tele_status.append("no-sideline-tele")

        result = process_clip_pair(
            pair, out_dir,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )

        if result:
            results.append(result)
            status = "OK" if not args.dry_run else "DRY-RUN"
            if result.mapping_r2 < 0.3 and result.high_conf_pairs >= 3:
                status += " (weak mapping)"
            elif result.high_conf_pairs < 3:
                status += " (insufficient calibration)"

            print(f"{result.clip_number:5d}  "
                  f"{result.xbotgo_samples:8d}  "
                  f"{result.sideline_samples:10d}  "
                  f"{result.fused_samples:7d}  "
                  f"{result.xbotgo_filled:7d}  "
                  f"{result.mapping_r2:6.3f}  "
                  f"{result.high_conf_pairs:6d}  "
                  f"{status}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"FUSION SUMMARY — {game_dir.name}")
    print(f"{'=' * 60}")
    print(f"  Pairs found:           {len(pairs)}")
    print(f"  Pairs fused:           {len(results)}")

    if results:
        avg_r2 = sum(r.mapping_r2 for r in results) / len(results)
        total_filled = sum(r.xbotgo_filled for r in results)
        total_fused = sum(r.fused_samples for r in results)
        good_mapping = sum(1 for r in results if r.mapping_r2 >= 0.5)

        print(f"  Avg mapping R²:        {avg_r2:.3f}")
        print(f"  Good mappings (R²≥0.5): {good_mapping}/{len(results)}")
        print(f"  Total fused samples:   {total_fused}")
        print(f"  XBotGo gaps filled:    {total_filled}")

    print(f"{'=' * 60}")

    if not args.dry_run and results:
        print(f"\nFused telemetry written to: {out_dir}/")
        print("To use with portrait rendering:")
        print("  python tools/render_follow_unified.py --in clip.mp4 "
              "--ball-telemetry out/telemetry/<stem>.ball_fused.jsonl")

    return 0


def _print_result(result: FusionResult) -> None:
    """Print a single fusion result."""
    print(f"\n{'=' * 60}")
    print(f"FUSION RESULT — Clip {result.clip_number:03d}")
    print(f"{'=' * 60}")
    print(f"  XBotGo clip:       {Path(result.xbotgo_clip).name}")
    print(f"  Sideline clip:     {Path(result.sideline_clip).name}")
    print(f"  XBotGo samples:    {result.xbotgo_samples}")
    print(f"  Sideline samples:  {result.sideline_samples}")
    print(f"  Fused samples:     {result.fused_samples}")
    print(f"  Gaps filled:       {result.xbotgo_filled} (from sideline)")
    print(f"  Mapping R²:        {result.mapping_r2:.3f}")
    print(f"  Calibration pts:   {result.high_conf_pairs}")
    if result.output_path:
        print(f"  Output:            {result.output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    raise SystemExit(main())

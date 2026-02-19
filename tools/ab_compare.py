#!/usr/bin/env python
"""A/B comparison: render the same clips with two presets side by side.

Picks a small set of clips (diverse play types) and renders each with both
presets, placing results in out/ab_compare/ for easy review.

Usage:
    python tools/ab_compare.py --dry-run              # preview clip selection
    python tools/ab_compare.py                         # render defaults (cinematic vs tight-action)
    python tools/ab_compare.py --preset-a cinematic --preset-b tight-action
    python tools/ab_compare.py --game SLSG --count 3   # 3 clips from one game
    python tools/ab_compare.py --clips clip1.mp4 clip2.mp4  # explicit clips
"""

from __future__ import annotations

import argparse
import csv
import random
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ATOMIC_INDEX = REPO_ROOT / "out" / "catalog" / "atomic_index.csv"
RENDER_SCRIPT = REPO_ROOT / "tools" / "render_follow_unified.py"
AB_OUT_DIR = REPO_ROOT / "out" / "ab_compare"

# Play types that best expose tracking differences (fast ball movement)
_PREFERRED_TYPES = ["GOAL", "SHOT", "CROSS", "DRIBBLING", "PRESSURE"]

# Skip CINEMATIC / variant rows
_SKIP_TAGS = ("__CINEMATIC", "__DEBUG", "__OVERLAY", "portrait_POST",
              "portrait_FINAL")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="A/B render comparison between two presets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--preset-a", default="cinematic",
                   help="First preset (default: cinematic)")
    p.add_argument("--preset-b", default="tight-action",
                   help="Second preset (default: tight-action)")
    p.add_argument("--count", type=int, default=5,
                   help="Number of clips to compare (default: 5)")
    p.add_argument("--game", help="Filter to a specific game (substring)")
    p.add_argument("--clips", nargs="+", metavar="PATH",
                   help="Explicit clip paths instead of auto-selection")
    p.add_argument("--dry-run", action="store_true",
                   help="Show selected clips without rendering")
    p.add_argument("--remap", metavar="OLD=NEW",
                   help="Remap clip path prefix (e.g. 'D:/Projects=C:/Work')")
    return p.parse_args(argv)


def _apply_remap(path: str, remap: tuple[str, str] | None) -> str:
    if not remap:
        return path
    old, new = remap
    normalized = path.replace("\\", "/")
    if normalized.startswith(old):
        return new + normalized[len(old):]
    return path


def _parse_remap(remap_arg: str | None) -> tuple[str, str] | None:
    if not remap_arg:
        return None
    if "=" not in remap_arg:
        return None
    old, new = remap_arg.split("=", 1)
    return old.replace("\\", "/"), new.replace("\\", "/")


def _play_type(clip_stem: str) -> str:
    """Extract play type from clip stem like ...GOAL__t48.50-t58.00."""
    parts = clip_stem.split("__")
    for part in reversed(parts):
        if part and not part.startswith("t") and not part[0].isdigit():
            return part.upper().replace(" ", "_")
    return "UNKNOWN"


def select_clips(args: argparse.Namespace) -> list[dict]:
    """Pick a diverse set of clips from atomic_index.csv."""
    if args.clips:
        return [{"path": c, "stem": Path(c).stem, "type": _play_type(Path(c).stem)}
                for c in args.clips]

    if not ATOMIC_INDEX.exists():
        print(f"[ERROR] Atomic index not found: {ATOMIC_INDEX}")
        return []

    remap = _parse_remap(args.remap)
    candidates: list[dict] = []

    with ATOMIC_INDEX.open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            stem = row.get("clip_stem", "")
            # Skip variant/render rows
            if any(tag in stem for tag in _SKIP_TAGS):
                continue
            # Skip portrait-size clips (already rendered)
            w = int(row.get("width", 0) or 0)
            h = int(row.get("height", 0) or 0)
            if h > w:
                continue
            clip_path = row.get("clip_path", "")
            if args.game and args.game.lower() not in clip_path.lower():
                continue
            clip_path = _apply_remap(clip_path, remap)
            play_type = _play_type(stem)
            candidates.append({
                "path": clip_path,
                "stem": stem,
                "type": play_type,
                "duration": float(row.get("duration_s", 0) or 0),
            })

    if not candidates:
        print("[ERROR] No matching clips found")
        return []

    # Prefer diverse play types, favor action-heavy types
    selected: list[dict] = []
    used_types: set[str] = set()

    # First pass: one clip per preferred type
    for ptype in _PREFERRED_TYPES:
        if len(selected) >= args.count:
            break
        matches = [c for c in candidates
                   if ptype in c["type"] and c["type"] not in used_types]
        if matches:
            pick = random.choice(matches)
            selected.append(pick)
            used_types.add(pick["type"])

    # Second pass: fill remaining from any type
    remaining = [c for c in candidates if c not in selected]
    random.shuffle(remaining)
    for c in remaining:
        if len(selected) >= args.count:
            break
        if c["type"] not in used_types:
            selected.append(c)
            used_types.add(c["type"])

    # Third pass: if still short, allow duplicates
    for c in remaining:
        if len(selected) >= args.count:
            break
        if c not in selected:
            selected.append(c)

    return selected


def render_clip(clip_path: str, out_path: Path, preset: str) -> bool:
    """Render one clip with a given preset. Returns True on success."""
    cmd = [
        sys.executable,
        str(RENDER_SCRIPT),
        "--in", clip_path,
        "--out", str(out_path),
        "--preset", preset,
        "--clean-temp",
    ]
    print(f"    {preset}: rendering...", end="", flush=True)
    t0 = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.monotonic() - t0
    if result.returncode == 0:
        print(f" done ({dt:.0f}s)")
        return True
    else:
        print(f" FAILED ({dt:.0f}s)")
        # Show last few lines of stderr for debugging
        for line in (result.stderr or "").strip().splitlines()[-5:]:
            print(f"      {line}")
        return False


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    clips = select_clips(args)

    print()
    print("=" * 65)
    print("  A/B PRESET COMPARISON")
    print("=" * 65)
    print(f"  Preset A:  {args.preset_a}")
    print(f"  Preset B:  {args.preset_b}")
    print(f"  Clips:     {len(clips)}")
    if args.game:
        print(f"  Game:      {args.game}")
    print(f"  Output:    {AB_OUT_DIR}")
    print("=" * 65)

    if not clips:
        return 1

    print()
    for i, clip in enumerate(clips, 1):
        print(f"  [{i}] {clip['type']:20s}  {clip['stem'][:60]}")

    if args.dry_run:
        print("\n[DRY-RUN] Would render each clip with both presets.")
        return 0

    print()
    AB_OUT_DIR.mkdir(parents=True, exist_ok=True)

    ok_a = 0
    ok_b = 0
    t_start = time.monotonic()

    for i, clip in enumerate(clips, 1):
        stem = clip["stem"]
        short_name = stem[:60]
        print(f"\n[{i}/{len(clips)}] {clip['type']} — {short_name}")

        out_a = AB_OUT_DIR / f"{stem}__{args.preset_a}.mp4"
        out_b = AB_OUT_DIR / f"{stem}__{args.preset_b}.mp4"

        if render_clip(clip["path"], out_a, args.preset_a):
            ok_a += 1
        if render_clip(clip["path"], out_b, args.preset_b):
            ok_b += 1

    elapsed = time.monotonic() - t_start
    print()
    print("=" * 65)
    print("  A/B COMPARISON COMPLETE")
    print("=" * 65)
    print(f"  {args.preset_a}:  {ok_a}/{len(clips)} rendered")
    print(f"  {args.preset_b}:  {ok_b}/{len(clips)} rendered")
    print(f"  Time:        {int(elapsed // 60)}m{int(elapsed % 60):02d}s")
    print(f"  Output dir:  {AB_OUT_DIR}")
    print()
    print("  Compare side-by-side by opening pairs:")
    for clip in clips[:3]:
        stem = clip["stem"][:45]
        print(f"    {stem}__cinematic.mp4")
        print(f"    {stem}__tight-action.mp4")
        if clip != clips[min(2, len(clips) - 1)]:
            print()
    print("=" * 65)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Re-render all clips using manual-anchors CSVs from _csv_to_anchors.py.

Clips WITH an anchors file get cinematic + manual anchor override.
Clips WITHOUT an anchors file get cinematic with default auto-follow.

Usage: python _rerender_with_anchors.py
       python _rerender_with_anchors.py --clip 1    (single clip)
"""
import os, subprocess, sys, argparse
from pathlib import Path

# ===== CONFIGURE =====
GAME = "2026-03-21__TSC_vs_NEOFC_2017_Blue"
# ======================

os.chdir(r"D:\Projects\soccer-video")

clips_dir = Path(f"out/atomic_clips/{GAME}")
anchors_dir = Path(f"_tmp/anchors/{GAME}")
out_dir = Path(f"out/portrait_reels/{GAME}")
out_dir.mkdir(parents=True, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--clip", type=int, default=None,
                help="Render only this clip number")
run_args = ap.parse_args()

clip_files = sorted(clips_dir.glob("*.mp4"),
                    key=lambda p: int(p.stem.split("__")[0]))

rendered = 0
failed = 0
anchored = 0

for clip_file in clip_files:
    stem = clip_file.stem
    clip_num = int(stem.split("__")[0])

    # If --clip specified, skip others
    if run_args.clip is not None and clip_num != run_args.clip:
        continue

    out_path = out_dir / f"{stem}_portrait_FINAL.mp4"
    anchors_path = anchors_dir / f"anchors_{clip_num}.csv"

    cmd = [
        sys.executable, "tools/render_follow_unified.py",
        "--preset", "cinematic",
        "--portrait", "1080x1920",
        "--in", str(clip_file),
        "--out", str(out_path),
    ]

    if anchors_path.exists():
        cmd.extend(["--manual-anchors", str(anchors_path)])
        print(f"\n{'='*60}")
        print(f"Clip {clip_num}: rendering with MANUAL ANCHORS")
        print(f"{'='*60}")
        anchored += 1
    else:
        print(f"\n{'='*60}")
        print(f"Clip {clip_num}: rendering with auto-follow (no anchors)")
        print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        rendered += 1
        print(f"  -> OK: {out_path.name}")
    else:
        failed += 1
        print(f"  -> FAILED (exit code {result.returncode})")

print(f"\n{'='*60}")
print(f"BATCH COMPLETE")
print(f"  Rendered:     {rendered}")
print(f"  With anchors: {anchored}")
print(f"  Failed:       {failed}")
print(f"  Output dir:   {out_dir}")
print(f"{'='*60}")

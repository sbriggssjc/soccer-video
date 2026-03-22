"""Re-render all clips using JSON plan files from _csv_to_plan.py.

Clips WITH a plan file get cinematic + plan override.
Clips WITHOUT a plan file get cinematic with default auto-follow.

Usage: python _rerender_with_plans.py
"""
import os, subprocess, sys
from pathlib import Path

# ===== CONFIGURE =====
GAME = "2026-03-21__TSC_vs_NEOFC_2017_Blue"
# ======================

os.chdir(r"D:\Projects\soccer-video")

clips_dir = Path(f"out/atomic_clips/{GAME}")
plans_dir = Path(f"_tmp/plans/{GAME}")
out_dir = Path(f"out/portrait_reels/{GAME}")
out_dir.mkdir(parents=True, exist_ok=True)

clip_files = sorted(clips_dir.glob("*.mp4"),
                    key=lambda p: int(p.stem.split("__")[0]))

rendered = 0
failed = 0
auto_follow = 0

for clip_file in clip_files:
    stem = clip_file.stem
    clip_num = stem.split("__")[0]
    out_path = out_dir / f"{stem}_portrait_FINAL.mp4"
    plan_path = plans_dir / f"plan_{clip_num}.json"

    cmd = [
        sys.executable, "tools/render_follow_unified.py",
        "--preset", "cinematic",
        "--portrait", "1080x1920",
        "--in", str(clip_file),
        "--out", str(out_path),
    ]

    if plan_path.exists():
        cmd.extend(["--plan", str(plan_path)])
        print(f"\n{'='*60}")
        print(f"Clip {clip_num}: rendering with corrected plan")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print(f"Clip {clip_num}: rendering with auto-follow (no plan edits)")
        print(f"{'='*60}")
        auto_follow += 1

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        rendered += 1
        print(f"  -> OK: {out_path.name}")
    else:
        failed += 1
        print(f"  -> FAILED (exit code {result.returncode})")

print(f"\n{'='*60}")
print(f"BATCH COMPLETE")
print(f"  Rendered:    {rendered}")
print(f"  Auto-follow: {auto_follow} (no plan edits)")
print(f"  Failed:      {failed}")
print(f"  Output dir:  {out_dir}")
print(f"{'='*60}")

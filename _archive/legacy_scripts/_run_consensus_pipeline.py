"""Re-run the pipeline with consensus YOLO detections.
Delete stale caches so the pipeline rebuilds from our new YOLO data.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(r"D:\Projects\soccer-video")
TEL = REPO / "out" / "telemetry"
STEM = "002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00"

# Files to delete so pipeline rebuilds them
STALE = [
    f"{STEM}.tracker_ball.jsonl",      # CSRT tracker - must rebuild from new YOLO
    f"{STEM}.ball.jsonl",              # Fused telemetry - must rebuild
    f"{STEM}.ball.follow.jsonl",       # Follow path - must rebuild
    f"{STEM}.ball.follow__smooth.jsonl", # Smoothed follow - must rebuild
]

print("=== Deleting stale caches ===")
for name in STALE:
    p = TEL / name
    if p.exists():
        # Backup first
        bak = TEL / (name + ".pre_consensus_backup")
        if not bak.exists():
            shutil.copy2(p, bak)
            print(f"  Backed up: {name}")
        os.remove(p)
        print(f"  Deleted: {name}")
    else:
        print(f"  Not found (ok): {name}")

# Verify consensus YOLO is in place
yolo_path = TEL / f"{STEM}.yolo_ball.jsonl"
with open(yolo_path) as f:
    lines = f.readlines()
print(f"\n=== YOLO telemetry: {len(lines)} consensus detections ===")

# Find the clip
CLIPS_DIR = REPO / "out" / "atomic_clips"
clip_path = CLIPS_DIR / "2026-02-23__TSC_vs_NEOFC" / f"{STEM}.mp4"
if not clip_path.exists():
    # Try finding it
    import glob
    candidates = glob.glob(str(CLIPS_DIR / "**" / "002__*NEOFC*BUILD*.mp4"), recursive=True)
    if candidates:
        clip_path = Path(candidates[0])

print(f"Clip: {clip_path}")
print(f"Exists: {clip_path.exists()}")

out_path = REPO / "out" / "portrait_reels" / "2026-02-23__TSC_vs_NEOFC" / "002__consensus_v10.mp4"

cmd = [
    sys.executable,
    str(REPO / "tools" / "render_follow_unified.py"),
    "--preset", "cinematic",
    "--in", str(clip_path),
    "--out", str(out_path),
    "--portrait", "1080x1920",
    "--no-draw-ball",
    "--diagnostics",
]

print(f"\n=== Running pipeline ===")
print(f"Command: {' '.join(cmd)}")
print(f"Output: {out_path}")
print()

result = subprocess.run(cmd, cwd=str(REPO), capture_output=False, text=True)
print(f"\nExit code: {result.returncode}")

"""Build full game highlight reel from all FINAL portrait clips in order."""
import os, subprocess, time, glob
from pathlib import Path

GAME = "2026-02-23__TSC_vs_NEOFC"
reels_dir = Path(rf"D:\Projects\soccer-video\out\portrait_reels\{GAME}")
output = reels_dir / f"{GAME}__GAME_REEL.mp4"
concat_list = Path(r"D:\Projects\soccer-video\_tmp\concat_list.txt")
FFMPEG = "ffmpeg"

# Find all FINAL clips, sorted by clip number
finals = sorted(reels_dir.glob("*__portrait__FINAL.mp4"), key=lambda p: p.name)
print(f"Found {len(finals)} FINAL clips")
for f in finals:
    print(f"  {f.name}")

# Write concat list
with open(concat_list, "w") as cl:
    for f in finals:
        cl.write(f"file '{f}'\n")

# Concatenate with re-encode to ensure consistent format
t0 = time.time()
if output.exists():
    output.unlink()

r = subprocess.run([
    FFMPEG, "-y", "-f", "concat", "-safe", "0",
    "-i", str(concat_list),
    "-c:v", "libx264", "-preset", "slow", "-crf", "17",
    "-c:a", "aac", "-b:a", "128k",
    "-pix_fmt", "yuv420p",
    str(output)
], capture_output=True, text=True, timeout=1800)

elapsed = time.time() - t0
if r.returncode != 0:
    print(f"FAILED: {(r.stderr or '')[-500:]}")
else:
    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"\nCOMPLETE: {elapsed/60:.1f} minutes, {size_mb:.1f} MB")
    print(f"Saved to: {output}")

# Cleanup
if concat_list.exists():
    concat_list.unlink()

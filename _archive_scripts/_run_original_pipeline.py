"""Run the original render_follow_unified pipeline on clip 002.

This uses the full fusion pipeline (YOLO + tracker + centroid + interp + hold)
which produced the working renders from 2/25.
"""
import subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
CLIP = (REPO / "out" / "atomic_clips" / "2026-02-23__TSC_vs_NEOFC"
        / "002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.mp4")
OUT = (REPO / "out" / "portrait_reels" / "2026-02-23__TSC_vs_NEOFC"
       / "002__pipeline_v9.mp4")
DIAG = OUT.with_suffix(".diag.csv")
RENDERER = REPO / "tools" / "render_follow_unified.py"

cmd = [
    sys.executable, str(RENDERER),
    "--preset", "cinematic",
    "--in", str(CLIP),
    "--out", str(OUT),
    "--portrait", "1080x1920",
    "--no-draw-ball",
    "--diagnostics",
]

print(f"Running: {' '.join(cmd)}")
result = subprocess.run(cmd, cwd=str(REPO), capture_output=False)
print(f"\nExit code: {result.returncode}")
if OUT.is_file():
    print(f"Output: {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")

"""Launch render as a child process, wait for completion, write log."""
import subprocess
import sys
import os

os.chdir(r"D:\Projects\soccer-video")

log_path = r"out\portrait_reels\2026-02-23__TSC_vs_NEOFC\002__v22_fixes3.log"
with open(log_path, "w") as log_f:
    log_f.write("LAUNCHER STARTING\n")
    log_f.flush()

    p = subprocess.Popen(
        [
            sys.executable,
            "tools/render_follow_unified.py",
            "--in", r"out\atomic_clips\2026-02-23__TSC_vs_NEOFC\002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.mp4",
            "--src", r"out\atomic_clips\2026-02-23__TSC_vs_NEOFC\002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.mp4",
            "--out", r"out\portrait_reels\2026-02-23__TSC_vs_NEOFC\002__v22_fixes.mp4",
            "--portrait", "1080x1920",
            "--preset", "cinematic",
            "--fps", "24",
            "--diagnostics",
            "--use-ball-telemetry",
        ],
        stdout=log_f,
        stderr=subprocess.STDOUT,
    )
    print(f"Child PID={p.pid}", flush=True)
    rc = p.wait()
    log_f.write(f"\nEXIT_CODE={rc}\n")
    print(f"EXIT_CODE={rc}", flush=True)

import subprocess, sys, traceback, os
os.chdir(r"D:\Projects\soccer-video")
logpath = r"D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_NEOFC\002__v22_fixes2.log"
try:
    with open(logpath, "w") as f:
        f.write("WRAPPER STARTING\n")
        f.flush()
    result = subprocess.run(
        [sys.executable, "tools/render_follow_unified.py",
         "--in", r"out\atomic_clips\2026-02-23__TSC_vs_NEOFC\002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.mp4",
         "--src", r"out\atomic_clips\2026-02-23__TSC_vs_NEOFC\002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.mp4",
         "--out", r"out\portrait_reels\2026-02-23__TSC_vs_NEOFC\002__v22_fixes.mp4",
         "--portrait", "1080x1920",
         "--preset", "cinematic",
         "--fps", "24",
         "--diagnostics",
         "--use-ball-telemetry"],
        capture_output=True, text=True, timeout=3600,
        encoding="utf-8", errors="replace"
    )
    with open(logpath, "w") as f:
        f.write("=== STDOUT ===\n")
        f.write(result.stdout or "(empty)")
        f.write("\n=== STDERR ===\n")
        f.write(result.stderr or "(empty)")
        f.write(f"\n=== EXIT CODE: {result.returncode} ===\n")
except Exception as e:
    with open(logpath, "a") as f:
        f.write(f"WRAPPER EXCEPTION:\n{traceback.format_exc()}\n")

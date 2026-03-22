import glob, os
REEL_DIR = r"D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_NEOFC"
LOG = r"D:\Projects\soccer-video\_tmp\batch_status.txt"

pattern = os.path.join(REEL_DIR, "*__portrait.mp4")
portraits = sorted(glob.glob(pattern))

with open(LOG, "w") as f:
    f.write(f"Total portrait clips: {len(portraits)}\n")
    final_count = 0
    need_processing = []
    for p in portraits:
        base = p.replace("__portrait.mp4", "")
        final = base + "__portrait__FINAL.mp4"
        if os.path.exists(final):
            final_count += 1
        else:
            need_processing.append(os.path.basename(p))
    f.write(f"Already finalized: {final_count}\n")
    f.write(f"Need processing: {len(need_processing)}\n")
    for n in need_processing:
        f.write(f"  {n}\n")

import glob, os, sys
sys.stdout.reconfigure(line_buffering=True)
pattern = os.path.join(r"D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_NEOFC", "*__portrait.mp4")
print(f"Pattern: {pattern}", flush=True)
files = sorted(glob.glob(pattern))
print(f"Found: {len(files)}", flush=True)
for f in files[:3]:
    print(f"  {os.path.basename(f)}", flush=True)

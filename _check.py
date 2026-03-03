import os
d = r"D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_Greenwood"
files = [f for f in os.listdir(d) if "FINAL" in f]
files.sort()
for f in files:
    sz = os.path.getsize(os.path.join(d, f)) / (1024*1024)
    print(f"{f}  ({sz:.1f} MB)")

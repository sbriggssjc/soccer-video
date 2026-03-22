import os
d = r"D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_Greenwood"
for f in sorted(os.listdir(d)):
    if "007" in f:
        sz = os.path.getsize(os.path.join(d, f)) / (1024*1024)
        print(f"{f}  ({sz:.1f} MB)")
if not any("007" in f for f in os.listdir(d)):
    print("No 007 files found")

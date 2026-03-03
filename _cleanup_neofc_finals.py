import os, glob
d = "D:/Projects/soccer-video/out/portrait_reels/2026-02-23__TSC_vs_NEOFC"
files = []
for i in range(6, 35):
    files.extend(glob.glob(os.path.join(d, f"{i:03d}__*FINAL*")))
print(f"Found {len(files)} FINAL files to delete")
for f in files:
    os.remove(f)
    print(f"DEL {os.path.basename(f)}")
print("Done")

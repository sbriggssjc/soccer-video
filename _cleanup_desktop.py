import os, glob

# 1. Delete old NEOFC 001-005 FINAL files (will be re-rendered)
d = "D:/Projects/soccer-video/out/portrait_reels/2026-02-23__TSC_vs_NEOFC"
for i in range(1, 6):
    for f in glob.glob(os.path.join(d, f"{i:03d}__*FINAL*")):
        os.remove(f)
        print(f"DEL FINAL: {os.path.basename(f)}")

# 2. Clean up desktop: remove NEOFC CSVs for 006-034 (already rendered)
#    and filmstrips for 006-034
desk = "C:/Users/scott/Desktop"
removed = 0
for i in range(6, 35):
    for pattern in [f"review_neofc_{i:03d}.csv", f"filmstrip_neofc_{i:03d}.jpg"]:
        p = os.path.join(desk, pattern)
        if os.path.exists(p):
            os.remove(p)
            print(f"DEL Desktop: {pattern}")
            removed += 1

# Also remove North OKC CSVs 001-019 (all rendered) and filmstrips
for i in range(1, 20):
    for pattern in [f"review_{i:03d}.csv", f"filmstrip_{i:03d}.jpg"]:
        p = os.path.join(desk, pattern)
        if os.path.exists(p):
            os.remove(p)
            print(f"DEL Desktop: {pattern}")
            removed += 1

# Remove Greenwood CSVs 001-015 (rendered, except 010 which was empty anyway)
for i in range(1, 16):
    for pattern in [f"review_gw_{i:03d}.csv", f"filmstrip_gw_{i:03d}.jpg"]:
        p = os.path.join(desk, pattern)
        if os.path.exists(p):
            os.remove(p)
            print(f"DEL Desktop: {pattern}")
            removed += 1

print(f"\nTotal removed from Desktop: {removed}")
print("Done")

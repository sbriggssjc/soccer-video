import shutil, os
src = r"D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_NEOFC\004__2026-02-23__TSC_vs_NEOFC__PRESSURE_AND_SHOT__t136.00-t153.00__portrait__FINAL.mp4"
dst_desktop = r"C:\Users\scott\Desktop\clip_004_v3.mp4"
dst_clean = r"D:\Projects\soccer-video\out\portrait_reels\clean\004__2026-02-23__TSC_vs_NEOFC__PRESSURE_AND_SHOT__t136.00-t153.00__CINEMATIC_portrait_FINAL.mp4"

for dst in [dst_desktop, dst_clean]:
    shutil.copy2(src, dst)
    sz = os.path.getsize(dst) / (1024*1024)
    print(f"Copied to {dst} ({sz:.1f} MB)")

# Clean up old desktop review files for clip 004
for f in ["review_004.csv", "filmstrip_004.png", "filmstrip_004_dense.png"]:
    p = os.path.join(r"C:\Users\scott\Desktop", f)
    if os.path.exists(p):
        os.remove(p)
        print(f"Removed {f}")
print("DONE")

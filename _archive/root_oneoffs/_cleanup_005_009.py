"""Clean up clips 005-009: fix stale copies, remove intermediates, tidy Desktop."""
import shutil, os
from pathlib import Path

reels = Path("out/portrait_reels/2026-02-23__TSC_vs_NEOFC")
clean = Path("out/portrait_reels/clean")
desktop = Path(r"C:\Users\scott\Desktop")
telem = Path("out/telemetry")
GAME = "2026-02-23__TSC_vs_NEOFC"

cleaned = []
errors = []

for c in ["005", "006", "007", "008", "009"]:
    print(f"\n=== Clip {c} ===")
    
    # 1. Ensure FINAL exists in reels dir
    finals = list(reels.glob(f"{c}__*__portrait__FINAL.mp4"))
    if not finals:
        errors.append(f"{c}: No FINAL in reels dir!")
        continue
    final = finals[0]
    stem = final.name.replace("__portrait__FINAL.mp4", "")
    
    # 2. Copy/update clean folder
    clean_name = f"{stem}__CINEMATIC_portrait_FINAL.mp4"
    clean_dst = clean / clean_name
    shutil.copy2(final, clean_dst)
    print(f"  Updated clean: {clean_name} ({final.stat().st_size/1024/1024:.1f} MB)")
    cleaned.append(f"{c}: clean copy updated")
    
    # 3. Remove intermediate portrait.mp4 (keep diag.csv and FINAL)
    portrait = reels / f"{stem}__portrait.mp4"
    if portrait.exists():
        sz = portrait.stat().st_size / (1024*1024)
        portrait.unlink()
        print(f"  Removed portrait.mp4 ({sz:.1f} MB)")
        cleaned.append(f"{c}: removed portrait.mp4")
    
    # 4. Remove render logs
    for log_file in reels.glob(f"{c}__rerender.log"):
        log_file.unlink()
        print(f"  Removed {log_file.name}")
    
    # 5. Remove .tmp portrait scratch files
    for tmp in reels.glob(f".tmp.{c}__*"):
        tmp.unlink()
        print(f"  Removed scratch {tmp.name}")
    
    # 6. Clean Desktop: remove review CSVs and filmstrips
    for pattern in [f"review_{c}.csv", f"filmstrip_{c}.png", f"filmstrip_{c}_dense.png", f"clip_{c}.mp4", f"clip_{c}_v*.mp4"]:
        for f in desktop.glob(pattern):
            sz = f.stat().st_size / (1024*1024)
            f.unlink()
            print(f"  Desktop removed: {f.name} ({sz:.1f} MB)")
            cleaned.append(f"{c}: desktop {f.name}")

# 7. Summary
print(f"\n{'='*50}")
print(f"Cleanup complete: {len(cleaned)} actions")
if errors:
    print(f"ERRORS: {errors}")

# 8. Verify final state
print(f"\nFinal verification:")
for c in ["005", "006", "007", "008", "009"]:
    finals = list(reels.glob(f"{c}__*FINAL*"))
    cleans = list(clean.glob(f"{c}__*{GAME}*CINEMATIC*"))
    f_sz = finals[0].stat().st_size/1024/1024 if finals else 0
    c_sz = cleans[0].stat().st_size/1024/1024 if cleans else 0
    match = "OK" if abs(f_sz - c_sz) < 0.1 else "MISMATCH!"
    print(f"  {c}: FINAL={f_sz:.1f}MB, CLEAN={c_sz:.1f}MB [{match}]")

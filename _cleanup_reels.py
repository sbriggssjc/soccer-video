"""Clean up NEOFC reels folder: keep only __portrait__FINAL.mp4 for locked clips.
Remove intermediate portraits, diag CSVs, render logs, scratch files.
"""
from pathlib import Path
import os

reels = Path("out/portrait_reels/2026-02-23__TSC_vs_NEOFC")
locked = ["001","002","003","004","005","006","007","008","009","012","032"]

total_removed = 0
total_freed = 0

# First, show what's currently in the folder (non-archive, non-FINAL)
print("=== AUDIT: files to remove ===\n")
for f in sorted(reels.iterdir()):
    if f.is_dir():
        continue
    name = f.name
    clip_num = name.split("__")[0]
    
    # Skip non-locked clips entirely
    if clip_num not in locked:
        continue
    
    # Keep FINAL mp4s
    if "__portrait__FINAL.mp4" in name:
        sz = f.stat().st_size / (1024*1024)
        print(f"  KEEP:   {name} ({sz:.1f} MB)")
        continue
    
    # Everything else for locked clips gets removed
    sz = f.stat().st_size / (1024*1024)
    print(f"  REMOVE: {name} ({sz:.1f} MB)")
    total_removed += 1
    total_freed += sz

print(f"\n--- Will remove {total_removed} files, freeing {total_freed:.0f} MB ---")
print("Proceeding...\n")

# Now actually remove
removed = 0
for f in sorted(reels.iterdir()):
    if f.is_dir():
        continue
    name = f.name
    clip_num = name.split("__")[0]
    if clip_num not in locked:
        continue
    if "__portrait__FINAL.mp4" in name:
        continue
    f.unlink()
    removed += 1

print(f"Removed {removed} files.")

# Also clean up the .tmp scratch file if present
for f in reels.glob(".tmp.*"):
    print(f"Removing scratch: {f.name}")
    f.unlink()

# Verify: show what remains for locked clips
print(f"\n=== VERIFICATION: locked clips in reels folder ===\n")
for c in locked:
    finals = list(reels.glob(f"{c}__*FINAL*"))
    if finals:
        sz = finals[0].stat().st_size / (1024*1024)
        print(f"  {c}: {finals[0].name} ({sz:.1f} MB)")
    else:
        print(f"  {c}: *** MISSING FINAL! ***")

# Count unlocked clips still intact
unlocked_files = [f for f in reels.iterdir() if f.is_file() and f.name.split("__")[0] not in locked and not f.name.startswith(".") and not f.name.startswith("_")]
print(f"\nUnlocked clip files remaining: {len(unlocked_files)} (untouched)")

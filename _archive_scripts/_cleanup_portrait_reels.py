"""Move legacy iteration files from portrait_reels to _archive.

Keeps:
  - *__portrait.mp4 / *__portrait.diag.csv  (baseline pipeline outputs)
  - *__portrait__FINAL.mp4 / *__portrait__FINAL.diag.csv  (approved finals)
  - _summary.csv, _summary.txt

Moves everything else to _archive/ subfolder.
"""
import os, shutil, re

REEL_DIR = r"D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_NEOFC"
ARCHIVE = os.path.join(REEL_DIR, "_archive")
os.makedirs(ARCHIVE, exist_ok=True)

keep_patterns = [
    r"__portrait\.mp4$",
    r"__portrait\.diag\.csv$",
    r"__portrait__FINAL\.mp4$",
    r"__portrait__FINAL\.diag\.csv$",
    r"^_summary\.",
]

moved = 0
kept = 0
for entry in os.listdir(REEL_DIR):
    full = os.path.join(REEL_DIR, entry)
    if entry == "_archive":
        continue
    # Check if this matches any keep pattern
    dominated = False
    for pat in keep_patterns:
        if re.search(pat, entry):
            dominated = True
            break
    if dominated:
        kept += 1
        print(f"  KEEP: {entry}")
        continue
    # Move to archive
    dest = os.path.join(ARCHIVE, entry)
    if os.path.isdir(full):
        shutil.move(full, dest)
    else:
        shutil.move(full, dest)
    moved += 1
    print(f"  MOVE: {entry}")

print(f"\nDone. Kept {kept} files, moved {moved} files/dirs to _archive/")

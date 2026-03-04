"""Update atomic index with new FINAL paths for Celtic 011-020, then clean desktop CSVs."""
import csv, os, re
from pathlib import Path

INDEX = r"D:\Projects\soccer-video\AtomicClips.All.csv"
REEL_DIR = Path(r"D:\Projects\soccer-video\out\portrait_reels\2026-03-01__TSC_vs_OK_Celtic")
DESKTOP = Path(r"C:\Users\scott\Desktop")

# Update index: set stabilized_exists=True and stabilized_path for clips 011-020
with open(INDEX, "r", newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

updated = 0
for row in rows:
    if row["date"] == "2026-03-01" and row["away"] == "OK_Celtic":
        idx = int(row["idx"])
        if 11 <= idx <= 20:
            # Find the FINAL file
            finals = list(REEL_DIR.glob(f"{row['idx']}__*__portrait__FINAL.mp4"))
            if finals:
                row["stabilized_exists"] = "True"
                row["stabilized_path"] = str(finals[0])
                updated += 1

with open(INDEX, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
    writer.writeheader()
    writer.writerows(rows)

print(f"Updated {updated} index entries with FINAL paths")

# Clean desktop CSVs
removed = 0
for i in range(11, 21):
    csv_file = DESKTOP / f"review_celtic_{i:03d}.csv"
    if csv_file.exists():
        try:
            os.remove(csv_file)
            removed += 1
            print(f"  Removed {csv_file.name}")
        except Exception as e:
            print(f"  FAILED {csv_file.name}: {e}")

# Also check for filmstrip PNGs
for i in range(11, 21):
    for pat in [f"filmstrip_celtic_{i:03d}*", f"celtic_{i:03d}*filmstrip*"]:
        for f in DESKTOP.glob(pat):
            try:
                os.remove(f)
                removed += 1
                print(f"  Removed {f.name}")
            except Exception as e:
                print(f"  FAILED {f.name}: {e}")

print(f"\nCleaned {removed} files from Desktop")

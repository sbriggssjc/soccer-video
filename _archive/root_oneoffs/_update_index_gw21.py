"""Update Feb 21 Greenwood index entries to reflect new FINAL renders."""
import csv, re
from pathlib import Path

INDEX = r"D:\Projects\soccer-video\AtomicClips.All.csv"
REEL_DIR = Path(r"D:\Projects\soccer-video\out\portrait_reels\2026-02-21__TSC_vs_Greenwood")

# Build map of FINAL renders by clip number
finals = {}
for f in REEL_DIR.glob("*__portrait__FINAL.mp4"):
    m = re.match(r'^(\d{3})__', f.name)
    if m:
        finals[m.group(1)] = str(f)

print(f"Found {len(finals)} FINAL renders for Feb 21 Greenwood")

with open(INDEX, "r", newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

updated = 0
for row in rows:
    if row["date"] == "2026-02-21" and row["away"] == "Greenwood":
        idx = row["idx"]
        if idx in finals:
            if row["stabilized_exists"] != "True" or row["stabilized_path"] != finals[idx]:
                row["stabilized_exists"] = "True"
                row["stabilized_path"] = finals[idx]
                updated += 1

if updated:
    with open(INDEX, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Updated {updated} entries")
else:
    print("All entries already up to date")

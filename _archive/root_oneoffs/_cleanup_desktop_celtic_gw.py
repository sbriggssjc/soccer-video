"""Clean up Desktop: Celtic 001-010 CSVs/filmstrips and Greenwood 011-015 CSVs/filmstrips."""
import os
from pathlib import Path

DESKTOP = Path(r"C:\Users\scott\Desktop")
removed = 0

# Celtic 001-010
for i in range(1, 11):
    for pat in [f"review_celtic_{i:03d}.csv", f"filmstrip_celtic_{i:03d}.jpg"]:
        f = DESKTOP / pat
        if f.exists():
            try:
                os.remove(f)
                removed += 1
                print(f"  Removed {pat}")
            except Exception as e:
                print(f"  FAILED {pat}: {e}")

# Greenwood 011-015
for i in range(11, 16):
    for pat in [f"review_gw_{i:03d}.csv", f"filmstrip_gw_{i:03d}.jpg"]:
        f = DESKTOP / pat
        if f.exists():
            try:
                os.remove(f)
                removed += 1
                print(f"  Removed {pat}")
            except Exception as e:
                print(f"  FAILED {pat}: {e}")

print(f"\nCleaned {removed} files from Desktop")

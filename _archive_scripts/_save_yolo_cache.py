"""Extract v6 YOLO detections from diag CSV and save as cache for v7."""
import json, csv
from pathlib import Path

diag = Path(r"D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_NEOFC\002__fresh_yolo_v6.diag.csv")
cache = Path(r"D:\Projects\soccer-video\_v7_yolo_cache.json")

detections = {}
with open(diag) as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["yolo_x"] and row["yolo_conf"]:
            fi = int(row["frame"])
            cx = float(row["yolo_x"])
            conf = float(row["yolo_conf"])
            # We don't have cy in the diag, need to re-detect
            # Actually the diag only has x, not y. We need y for world-space filtering.
            detections[str(fi)] = [cx, 0.0, conf]  # placeholder y

print(f"Found {len(detections)} detections in diag CSV")
print("NOTE: diag CSV doesn't have Y coords. Need to re-run YOLO or use a different approach.")

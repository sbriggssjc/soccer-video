"""Compare all ball position sources side-by-side to find where they diverge"""
import json
import numpy as np
from pathlib import Path

base = Path(r"D:\Projects\soccer-video")

# Load all three original data sources
def load_jsonl(path, x_key="cx", y_key="cy"):
    data = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            rec = json.loads(line)
            if rec.get("_meta"): continue
            fr = int(rec["frame"])
            cx = rec.get(x_key) or rec.get("ball_x")
            cy = rec.get(y_key) or rec.get("ball_y")
            if cx is not None and cy is not None:
                data[fr] = (float(cx), float(cy))
    return data

centroid = load_jsonl(base / "out/telemetry/002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.ball.jsonl.orig_backup")
yolo = load_jsonl(base / "out/telemetry/002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.yolo_ball.jsonl.orig_backup")
optflow = load_jsonl(base / "_optflow_track/optflow_ball_path_v3.jsonl")

# Also load tracker data
tracker_path = base / "out/telemetry/002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.tracker_ball.jsonl.orig_backup"
tracker = {}
if tracker_path.exists():
    tracker = load_jsonl(tracker_path)

print(f"Data coverage:")
print(f"  Centroid:  {len(centroid)} frames")
print(f"  YOLO:      {len(yolo)} frames") 
print(f"  Tracker:   {len(tracker)} frames")
print(f"  OptFlow:   {len(optflow)} frames")

# Show all sources every 10 frames
print(f"\n{'frame':>5}  {'centroid_x':>10}  {'yolo_x':>10}  {'tracker_x':>10}  {'optflow_x':>10}  {'cent-of':>8}")
print("-" * 75)
for fr in range(0, 496, 10):
    cx = f"{centroid[fr][0]:10.1f}" if fr in centroid else f"{'---':>10}"
    yx = f"{yolo[fr][0]:10.1f}" if fr in yolo else f"{'---':>10}"
    tx = f"{tracker[fr][0]:10.1f}" if fr in tracker else f"{'---':>10}"
    ox = f"{optflow[fr][0]:10.1f}" if fr in optflow else f"{'---':>10}"
    
    # Calculate centroid vs optflow divergence
    div = ""
    if fr in centroid and fr in optflow:
        d = abs(centroid[fr][0] - optflow[fr][0])
        div = f"{d:8.1f}"
        if d > 100:
            div += " !!!"
    
    print(f"{fr:5d}  {cx}  {yx}  {tx}  {ox}  {div}")

# Identify big divergence zones
print(f"\n\nZONES WHERE CENTROID AND OPTFLOW DIVERGE >200px:")
print("-" * 60)
in_diverge = False
start_fr = 0
for fr in range(496):
    if fr in centroid and fr in optflow:
        d = abs(centroid[fr][0] - optflow[fr][0])
        if d > 200:
            if not in_diverge:
                start_fr = fr
                in_diverge = True
        else:
            if in_diverge:
                print(f"  f{start_fr}-f{fr-1}: centroid x={centroid[start_fr][0]:.0f}→{centroid[fr-1][0]:.0f}, optflow x={optflow[start_fr][0]:.0f}→{optflow[fr-1][0]:.0f}")
                in_diverge = False
if in_diverge:
    print(f"  f{start_fr}-f495")

# Show centroid trajectory summary
print(f"\n\nCENTROID X-POSITION ZONES:")
zones = {}
for fr in sorted(centroid.keys()):
    x = centroid[fr][0]
    zone = int(x // 200) * 200
    zones.setdefault(zone, []).append(fr)
for zone in sorted(zones.keys()):
    frames = zones[zone]
    pct = 100.0 * len(frames) / len(centroid)
    print(f"  x={zone:4d}-{zone+199:4d}: {len(frames):3d} frames ({pct:5.1f}%)")

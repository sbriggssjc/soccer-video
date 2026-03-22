"""Audit: compare original YOLO detections vs optical flow v3 trajectory"""
import json
from pathlib import Path

base = Path(r"D:\Projects\soccer-video")

# Load original YOLO backup
yolo_path = base / "out/telemetry/002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.yolo_ball.jsonl.orig_backup"
yolo = {}
with open(yolo_path) as f:
    for line in f:
        line = line.strip()
        if not line: continue
        rec = json.loads(line)
        if rec.get("_meta"): continue
        fr = int(rec["frame"])
        yolo[fr] = (float(rec["cx"]), float(rec["cy"]), float(rec.get("conf", 0)))

# Load original centroid/ball backup
cent_path = base / "out/telemetry/002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.ball.jsonl.orig_backup"
cent = {}
with open(cent_path) as f:
    for line in f:
        line = line.strip()
        if not line: continue
        rec = json.loads(line)
        if rec.get("_meta"): continue
        fr = int(rec["frame"])
        cx = rec.get("cx") or rec.get("ball_x")
        cy = rec.get("cy") or rec.get("ball_y")
        if cx is not None and cy is not None:
            cent[fr] = (float(cx), float(cy))

# Load optical flow v3
of_path = base / "_optflow_track/optflow_ball_path_v3.jsonl"
ofv3 = {}
with open(of_path) as f:
    for line in f:
        line = line.strip()
        if not line: continue
        rec = json.loads(line)
        if rec.get("_meta"): continue
        fr = int(rec["frame"])
        ofv3[fr] = (float(rec["cx"]), float(rec["cy"]))

print("="*80)
print("ORIGINAL YOLO DETECTIONS (ground truth)")
print("="*80)
for fr in sorted(yolo.keys()):
    x, y, c = yolo[fr]
    print(f"  f{fr:3d}  x={x:7.1f}  y={y:7.1f}  conf={c:.3f}")
print(f"\nTotal YOLO frames: {len(yolo)}")

print("\n" + "="*80)
print("OPTICAL FLOW v3 TRAJECTORY - sampled every 20 frames")
print("="*80)
max_fr = max(ofv3.keys())
for fr in range(0, max_fr+1, 20):
    if fr in ofv3:
        ox, oy = ofv3[fr]
        yl = yolo.get(fr)
        yolo_str = f"  YOLO: x={yl[0]:.1f}" if yl else ""
        print(f"  f{fr:3d}  x={ox:7.1f}  y={oy:7.1f}{yolo_str}")

print("\n" + "="*80)
print("TRAJECTORY X-POSITION ZONES")
print("="*80)
# Bucket by 100px zones
zones = {}
for fr in sorted(ofv3.keys()):
    x = ofv3[fr][0]
    zone = int(x // 100) * 100
    zones.setdefault(zone, []).append(fr)

for zone in sorted(zones.keys()):
    frames = zones[zone]
    pct = 100.0 * len(frames) / len(ofv3)
    print(f"  x={zone:4d}-{zone+99:4d}: {len(frames):3d} frames ({pct:5.1f}%)  f{frames[0]}-f{frames[-1]}")

print("\n" + "="*80)
print("CENTROID vs OPTFLOW DIVERGENCE CHECK")
print("="*80)
diverged = []
for fr in sorted(cent.keys()):
    if fr in ofv3:
        cx, cy = cent[fr]
        ox, oy = ofv3[fr]
        dx = abs(cx - ox)
        if dx > 50:
            diverged.append((fr, cx, ox, dx))
if diverged:
    print(f"Frames where centroid and optflow differ by >50px in X: {len(diverged)}")
    for fr, cx, ox, dx in diverged[:20]:
        print(f"  f{fr:3d}  centroid_x={cx:7.1f}  optflow_x={ox:7.1f}  delta={dx:.1f}")
else:
    print("No significant divergence found")

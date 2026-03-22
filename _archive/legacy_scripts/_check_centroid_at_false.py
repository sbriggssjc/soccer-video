"""Check if centroid data exists at the 13 false YOLO frames."""
import json

YOLO_FILE = r"D:\Projects\soccer-video\out\telemetry\002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.yolo_ball.jsonl"
BALL_FILE = r"D:\Projects\soccer-video\out\telemetry\002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.ball.jsonl"

# Load centroid/ball telemetry
centroid = {}
with open(BALL_FILE, "r") as f:
    for line in f:
        d = json.loads(line.strip())
        fidx = int(d.get("frame", d.get("frame_idx", 0)))
        centroid[fidx] = d

# Load YOLO
yolo = []
with open(YOLO_FILE, "r") as f:
    for line in f:
        yolo.append(json.loads(line.strip()))

false_frames = [25, 45, 69, 120, 121, 186, 198, 199, 268, 269, 332, 466, 494]

print(f"Centroid frames: {len(centroid)}/496", flush=True)
print(f"\nFalse YOLO vs Centroid at those frames:", flush=True)
print(f"{'Frame':>6} {'YOLO_x':>8} {'YOLO_y':>8} {'YOLO_conf':>9} | {'Cent_x':>8} {'Cent_y':>8} {'Cent_conf':>9} {'Dist':>7}", flush=True)
print("-" * 80, flush=True)

import math
for fidx in false_frames:
    yd = next((d for d in yolo if int(d.get("frame", d.get("frame_idx", 0))) == fidx), None)
    cd = centroid.get(fidx)
    if yd and cd:
        yx = float(yd.get("x", yd.get("cx", 0)))
        yy = float(yd.get("y", yd.get("cy", 0)))
        yc = float(yd.get("confidence", yd.get("conf", 0)))
        cx = float(cd.get("x", cd.get("cx", 0)))
        cy = float(cd.get("y", cd.get("cy", 0)))
        cc = float(cd.get("confidence", cd.get("conf", 0)))
        dist = math.hypot(yx - cx, yy - cy)
        print(f"{fidx:>6} {yx:>8.1f} {yy:>8.1f} {yc:>9.3f} | {cx:>8.1f} {cy:>8.1f} {cc:>9.3f} {dist:>7.0f}", flush=True)
    elif yd:
        yx = float(yd.get("x", yd.get("cx", 0)))
        yy = float(yd.get("y", yd.get("cy", 0)))
        yc = float(yd.get("confidence", yd.get("conf", 0)))
        print(f"{fidx:>6} {yx:>8.1f} {yy:>8.1f} {yc:>9.3f} | {'NO CENTROID':>30}", flush=True)

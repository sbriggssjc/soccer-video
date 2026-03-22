"""Preview which false YOLO the spatial consistency filter keeps/drops."""
import json, math

YOLO_FILE = r"D:\Projects\soccer-video\out\telemetry\002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.yolo_ball.jsonl"
CONF_FLOOR = 0.25
SPATIAL_THRESH = 200.0

detections = []
with open(YOLO_FILE) as f:
    for line in f:
        detections.append(json.loads(line.strip()))

by_frame = {}
for d in detections:
    fi = int(d.get("frame", d.get("frame_idx", 0)))
    by_frame[fi] = d

real_frames = sorted(fi for fi, d in by_frame.items()
                     if d.get("confidence", d.get("conf", 0)) > CONF_FLOOR)
false_frames = sorted(fi for fi, d in by_frame.items()
                      if d.get("confidence", d.get("conf", 0)) <= CONF_FLOOR)

print(f"Real YOLO: {len(real_frames)}, False YOLO: {len(false_frames)}")
print(f"\n{'Frame':>6} {'Conf':>6} {'X':>7} {'ExpX':>7} {'Dev':>6} {'Action':>8}")
print("-" * 50)

for fi in false_frames:
    d = by_frame[fi]
    x = float(d.get("x", d.get("cx", 0)))
    conf = d.get("confidence", d.get("conf", 0))
    prev_real = next((rf for rf in reversed(real_frames) if rf < fi), None)
    next_real = next((rf for rf in real_frames if rf > fi), None)
    if prev_real is not None and next_real is not None:
        px = float(by_frame[prev_real].get("x", by_frame[prev_real].get("cx", 0)))
        nx = float(by_frame[next_real].get("x", by_frame[next_real].get("cx", 0)))
        t = (fi - prev_real) / max(next_real - prev_real, 1)
        exp_x = px + t * (nx - px)
    elif prev_real is not None:
        exp_x = float(by_frame[prev_real].get("x", by_frame[prev_real].get("cx", 0)))
    elif next_real is not None:
        exp_x = float(by_frame[next_real].get("x", by_frame[next_real].get("cx", 0)))
    else:
        exp_x = 0
    dev = abs(x - exp_x)
    action = "KEEP" if dev <= SPATIAL_THRESH else "DROP"
    print(f"{fi:>6} {conf:>6.3f} {x:>7.1f} {exp_x:>7.1f} {dev:>6.0f} {action:>8}")

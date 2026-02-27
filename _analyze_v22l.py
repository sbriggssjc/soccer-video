"""Analyze which YOLO detections were dropped by the 0.25 conf floor."""
import json, sys

yolo_file = r"D:\Projects\soccer-video\out\telemetry\002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.yolo_ball.jsonl"

detections = []
with open(yolo_file, "r") as f:
    for line in f:
        d = json.loads(line.strip())
        detections.append(d)

print(f"Total YOLO detections: {len(detections)}")
print(f"\nAll detections sorted by confidence:")
print(f"{'Frame':>6} {'Conf':>6} {'X':>7} {'Y':>7} {'Status':>10}")
print("-" * 45)

for d in sorted(detections, key=lambda x: x.get("confidence", x.get("conf", 0))):
    frame = d.get("frame", d.get("frame_idx", "?"))
    conf = d.get("confidence", d.get("conf", 0))
    x = d.get("x", d.get("cx", 0))
    y = d.get("y", d.get("cy", 0))
    status = "DROPPED" if conf < 0.25 else "KEPT"
    print(f"{frame:>6} {conf:>6.3f} {x:>7.1f} {y:>7.1f} {status:>10}")

dropped = [d for d in detections if d.get("confidence", d.get("conf", 0)) < 0.25]
kept = [d for d in detections if d.get("confidence", d.get("conf", 0)) >= 0.25]
print(f"\nDropped: {len(dropped)}, Kept: {len(kept)}")

# Show dropped detections with context
print(f"\nDropped detections detail:")
for d in sorted(dropped, key=lambda x: x.get("frame", x.get("frame_idx", 0))):
    frame = d.get("frame", d.get("frame_idx", "?"))
    conf = d.get("confidence", d.get("conf", 0))
    x = d.get("x", d.get("cx", 0))
    y = d.get("y", d.get("cy", 0))
    print(f"  f{frame}: conf={conf:.3f}, x={x:.1f}, y={y:.1f}")

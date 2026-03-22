"""Check where CSRT tracker data exists relative to key zones."""
import json

TRACKER_FILE = r"D:\Projects\soccer-video\out\telemetry\002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.tracker_ball.jsonl"
YOLO_FILE = r"D:\Projects\soccer-video\out\telemetry\002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.yolo_ball.jsonl"
BALL_FILE = r"D:\Projects\soccer-video\out\telemetry\002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.ball.jsonl"

tracker = {}
with open(TRACKER_FILE) as f:
    for line in f:
        d = json.loads(line.strip())
        fi = int(d.get("frame", d.get("frame_idx", 0)))
        tracker[fi] = d

yolo = {}
with open(YOLO_FILE) as f:
    for line in f:
        d = json.loads(line.strip())
        fi = int(d.get("frame", d.get("frame_idx", 0)))
        yolo[fi] = d

centroid = {}
with open(BALL_FILE) as f:
    for line in f:
        d = json.loads(line.strip())
        fi = int(d.get("frame", d.get("frame_idx", 0)))
        centroid[fi] = d

print(f"Tracker: {len(tracker)} frames", flush=True)
print(f"YOLO: {len(yolo)} frames", flush=True)
print(f"Centroid: {len(centroid)} frames", flush=True)

# Key zones
zones = [
    ("Beginning", 0, 95),
    ("Winger", 250, 314),
    ("Shot/Save", 314, 425),
    ("Restart", 425, 496),
]

for name, start, end in zones:
    print(f"\n=== {name} (f{start}-f{end}) ===", flush=True)
    t_frames = [fi for fi in range(start, end) if fi in tracker]
    y_frames = [fi for fi in range(start, end) if fi in yolo]
    print(f"  Tracker: {len(t_frames)} frames", flush=True)
    if t_frames:
        for fi in t_frames[:5]:
            d = tracker[fi]
            x = float(d.get("x", d.get("cx", 0)))
            print(f"    f{fi}: x={x:.0f}", flush=True)
        if len(t_frames) > 5:
            print(f"    ... ({len(t_frames)} total)", flush=True)
        for fi in t_frames[-3:]:
            d = tracker[fi]
            x = float(d.get("x", d.get("cx", 0)))
            print(f"    f{fi}: x={x:.0f}", flush=True)
    print(f"  YOLO: {len(y_frames)} frames", flush=True)
    if y_frames:
        for fi in y_frames:
            d = yolo[fi]
            x = float(d.get("x", d.get("cx", 0)))
            c = float(d.get("confidence", d.get("conf", 0)))
            print(f"    f{fi}: x={x:.0f}, conf={c:.3f}", flush=True)

import json, os

clip = '002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00'
yolo_path = f'out/telemetry/{clip}.yolo_ball.jsonl.orig_backup'
tracker_path = f'out/telemetry/{clip}.tracker_ball.jsonl.orig_backup'

# Load original YOLO
yolo = {}
with open(yolo_path) as f:
    for line in f:
        d = json.loads(line)
        if 'frame' in d:
            yolo[d['frame']] = d

# Load original tracker
tracker = {}
with open(tracker_path) as f:
    for line in f:
        d = json.loads(line)
        if 'frame' in d:
            tracker[d['frame']] = d

print('=== YOLO detections f380-f496 ===')
for fi in sorted(yolo.keys()):
    if fi >= 380:
        y = yolo[fi]
        print(f'  f{fi}: cx={y["cx"]:.1f} cy={y["cy"]:.1f} conf={y.get("conf",0):.3f}')

print('\n=== Tracker entries f380-f496 (sampled) ===')
for fi in range(380, 497, 5):
    if fi in tracker:
        t = tracker[fi]
        print(f'  f{fi}: cx={t["cx"]:.1f} cy={t["cy"]:.1f}')

print(f'\nTotal YOLO entries: {len(yolo)}')
print(f'Total tracker entries: {len(tracker)}')
print(f'Last YOLO frame: {max(yolo.keys())}')
print(f'Last tracker frame: {max(tracker.keys())}')

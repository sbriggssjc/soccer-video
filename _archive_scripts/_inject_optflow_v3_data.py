"""
Inject optical flow v3 trajectory into all telemetry channels.
Reads from optflow_ball_path_v3.jsonl, writes to centroid + YOLO channels.
Tracker channel stays empty (already emptied by v3 injection).
"""
import json, os, shutil

os.chdir(r'D:\Projects\soccer-video')

CLIP = '002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00'
OPTFLOW = '_optflow_track/optflow_ball_path_v3.jsonl'
CENTROID = f'out/telemetry/{CLIP}.ball.jsonl'
YOLO = f'out/telemetry/{CLIP}.yolo_ball.jsonl'
TRACKER = f'out/telemetry/{CLIP}.tracker_ball.jsonl'

# Load optical flow v3
entries = []
with open(OPTFLOW) as f:
    for line in f:
        entries.append(json.loads(line))
print(f"Loaded {len(entries)} optical flow v3 entries")

# Write centroid channel (conf=0.90)
with open(CENTROID, 'w') as fh:
    for e in entries:
        fh.write(json.dumps({
            'frame': e['frame'], 't': e['t'],
            'cx': e['cx'], 'cy': e['cy'],
            'conf': max(e['conf'], 0.90),
        }) + '\n')
print(f"Wrote {CENTROID}")

# Write YOLO channel (conf=0.85, fake w/h)
with open(YOLO, 'w') as fh:
    for e in entries:
        fh.write(json.dumps({
            'frame': e['frame'], 't': e['t'],
            'cx': e['cx'], 'cy': e['cy'],
            'conf': max(e['conf'], 0.85),
            'w': 20.0, 'h': 20.0,
        }) + '\n')
print(f"Wrote {YOLO}")

# Ensure tracker is empty (just _meta)
with open(TRACKER, 'w') as fh:
    fh.write(json.dumps({'_meta': True, 'note': 'emptied for optflow v3'}) + '\n')
print(f"Wrote {TRACKER} (empty)")
print("Done! All channels injected with optflow v3.")

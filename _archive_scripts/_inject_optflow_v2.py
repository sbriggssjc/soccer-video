"""
Inject optical flow into BOTH the centroid (.ball.jsonl) AND YOLO
(.yolo_ball.jsonl) channels so the fusion sees consistent data and
doesn't override our pixel-accurate tracking.
"""
import json, os, shutil
os.chdir(r'D:\Projects\soccer-video')

CLIP = '002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00'
TELEM = 'out/telemetry'
BALL_JSONL = f'{TELEM}/{CLIP}.ball.jsonl'
YOLO_JSONL = f'{TELEM}/{CLIP}.yolo_ball.jsonl'
OPTFLOW = '_optflow_track/optflow_ball_path_v2.jsonl'

# Load optical flow
optflow = []
with open(OPTFLOW) as fh:
    for line in fh:
        optflow.append(json.loads(line))
print(f'Optical flow: {len(optflow)} entries')

# Backup originals (only if not already backed up)
for path, label in [(BALL_JSONL, 'ball'), (YOLO_JSONL, 'yolo')]:
    backup = f'{path}.orig_backup'
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
        print(f'Backed up {label}: {path} -> {backup}')
    else:
        print(f'Backup exists for {label}: {backup}')

# Write centroid channel (.ball.jsonl)
with open(BALL_JSONL, 'w') as fh:
    for e in optflow:
        fh.write(json.dumps({
            'frame': e['frame'],
            't': e['t'],
            'cx': e['cx'],
            'cy': e['cy'],
            'conf': max(e['conf'], 0.90),  # high conf
        }) + '\n')
print(f'Wrote {len(optflow)} entries to {BALL_JSONL}')

# Write YOLO channel (.yolo_ball.jsonl)
# Format: frame, t, cx, cy, conf, w, h
with open(YOLO_JSONL, 'w') as fh:
    for e in optflow:
        fh.write(json.dumps({
            'frame': e['frame'],
            't': e['t'],
            'cx': e['cx'],
            'cy': e['cy'],
            'conf': max(e['conf'], 0.85),
            'w': 20.0,  # approx ball size
            'h': 20.0,
        }) + '\n')
print(f'Wrote {len(optflow)} entries to {YOLO_JSONL}')
print('\nBoth channels injected. Ready to render!')

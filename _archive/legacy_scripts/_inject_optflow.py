"""
Inject optical flow ball path into the telemetry pipeline.
Backs up the existing .ball.jsonl, then writes the optical flow data
in the exact same format so the render pipeline picks it up.
"""
import json, os, shutil
os.chdir(r'D:\Projects\soccer-video')

CLIP = '002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00'
TELEM_DIR = 'out/telemetry'
BALL_JSONL = f'{TELEM_DIR}/{CLIP}.ball.jsonl'
OPTFLOW_PATH = '_optflow_track/optflow_ball_path_v2.jsonl'

# Step 1: Backup existing
backup = f'{BALL_JSONL}.fusion_backup'
if not os.path.exists(backup):
    shutil.copy2(BALL_JSONL, backup)
    print(f'Backed up: {BALL_JSONL} -> {backup}')
else:
    print(f'Backup already exists: {backup}')

# Step 2: Read optical flow data
optflow = []
with open(OPTFLOW_PATH) as fh:
    for line in fh:
        optflow.append(json.loads(line))
print(f'Optical flow entries: {len(optflow)}')

# Step 3: Write in telemetry format
# The render pipeline reads: frame, t, cx, cy, conf
with open(BALL_JSONL, 'w') as fh:
    for entry in optflow:
        row = {
            'frame': entry['frame'],
            't': entry['t'],
            'cx': entry['cx'],
            'cy': entry['cy'],
            'conf': entry['conf'],
        }
        fh.write(json.dumps(row) + '\n')

print(f'Wrote {len(optflow)} entries to {BALL_JSONL}')
print('Ready to render!')

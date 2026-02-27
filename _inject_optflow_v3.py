"""Also inject into tracker channel and clear scratch."""
import json, os, shutil
os.chdir(r'D:\Projects\soccer-video')

CLIP = '002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00'
TELEM = 'out/telemetry'
TRACKER_JSONL = f'{TELEM}/{CLIP}.tracker_ball.jsonl'
OPTFLOW = '_optflow_track/optflow_ball_path_v2.jsonl'

optflow = []
with open(OPTFLOW) as fh:
    for line in fh:
        optflow.append(json.loads(line))

# Backup tracker
backup = f'{TRACKER_JSONL}.orig_backup'
if not os.path.exists(backup):
    shutil.copy2(TRACKER_JSONL, backup)
    print(f'Backed up tracker: {backup}')
else:
    print(f'Tracker backup exists')

# Write tracker (empty — no gaps to fill since YOLO covers every frame)
# Actually write _meta only so the file is valid but tracker adds nothing
with open(TRACKER_JSONL, 'w') as fh:
    fh.write(json.dumps({'_meta': True, 'version': 'optflow_override'}) + '\n')
print(f'Wrote empty tracker to {TRACKER_JSONL}')

# Clean scratch for previous render
import glob
scratch_dirs = glob.glob('out/_scratch/*6180b164*')
for d in scratch_dirs:
    print(f'Removing scratch: {d}')
    shutil.rmtree(d, ignore_errors=True)
print('Done')

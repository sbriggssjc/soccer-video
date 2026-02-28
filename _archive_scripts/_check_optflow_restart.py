import json

path = '_optflow_track/optflow_ball_path_v2.jsonl'
data = {}
with open(path) as f:
    for line in f:
        d = json.loads(line)
        data[d['frame']] = d

print('=== Optical flow trajectory f410-f470 ===')
for fi in range(410, 471):
    if fi in data:
        d = data[fi]
        src = d.get('source', '?')
        print(f'  f{fi}: x={d["cx"]:.1f} y={d["cy"]:.1f} conf={d["conf"]:.2f} src={src}')
    else:
        print(f'  f{fi}: MISSING')

# Check what anchors exist near this zone
print('\n=== Check YOLO anchors in restart zone ===')
clip = '002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00'
yolo_path = f'out/telemetry/{clip}.yolo_ball.jsonl.orig_backup'
import os
if not os.path.exists(yolo_path):
    yolo_path = f'out/telemetry/{clip}.yolo_ball.jsonl'
print(f'Reading: {yolo_path}')

yolo = {}
with open(yolo_path) as f:
    for line in f:
        d = json.loads(line)
        if 'frame' in d:
            yolo[d['frame']] = d

for fi in range(400, 475):
    if fi in yolo:
        y = yolo[fi]
        print(f'  f{fi}: YOLO cx={y["cx"]:.1f} cy={y["cy"]:.1f} conf={y.get("conf",0):.2f}')

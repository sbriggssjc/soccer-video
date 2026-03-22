import json, os

# Load centroid data
cent_path = None
for f in os.listdir('out/telemetry'):
    if '002__' in f and 'ball.jsonl' in f and 'yolo' not in f and 'tracker' not in f:
        cent_path = os.path.join('out/telemetry', f)
        break

print(f'Centroid file: {cent_path}')
centroid = {}
with open(cent_path) as fh:
    for line in fh:
        d = json.loads(line)
        centroid[d['frame']] = d

# Show centroid in shot/save zone
print('\n=== Centroid in Shot/Save zone (f314-f425) ===')
for fi in range(314, 426):
    if fi in centroid:
        c = centroid[fi]
        marker = ''
        if fi == 314: marker = ' <-- last real YOLO (x=1607)'
        if fi == 332: marker = ' <-- false YOLO (shadow)'
        if fi == 425: marker = ' <-- next real YOLO (x=95, restart)'
        print(f'  f{fi}: x={c["x"]:.0f}, y={c["y"]:.0f}{marker}')
    else:
        print(f'  f{fi}: NO CENTROID')

# Also check beginning zone centroid vs tracker
print('\n=== Beginning: Centroid vs Tracker (f0-f95) ===')
tracker_path = None
for f in os.listdir('out/telemetry'):
    if '002__' in f and 'tracker_ball' in f:
        tracker_path = os.path.join('out/telemetry', f)
        break
tracker = {}
with open(tracker_path) as fh:
    for line in fh:
        d = json.loads(line)
        tracker[d['frame']] = d

for fi in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]:
    cx = centroid[fi]['x'] if fi in centroid else None
    tx = tracker[fi]['x'] if fi in tracker else None
    cx_str = f'{cx:.0f}' if cx else 'N/A'
    tx_str = f'{tx:.0f}' if tx else 'N/A'
    print(f'  f{fi}: centroid_x={cx_str}, tracker_x={tx_str}')

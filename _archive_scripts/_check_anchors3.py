import json, os
os.chdir(r'D:\Projects\soccer-video')

CLIP = '002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00'

# YOLO
yp = f'out/telemetry/{CLIP}.yolo_ball.jsonl'
print(f'YOLO file: {yp} (exists={os.path.exists(yp)})')
entries = []
with open(yp) as fh:
    for line in fh:
        d = json.loads(line)
        if d.get('_meta'): continue
        entries.append(d)

frames = set(e['frame'] for e in entries)
print(f'Total: {len(entries)} entries, {len(frames)} unique frames')

# Key frames
for target in [28, 95, 250, 293, 314, 332, 425, 450, 480]:
    hits = [e for e in entries if e['frame'] == target]
    if hits:
        for h in hits:
            print(f'  YOLO f{target}: cx={h["cx"]:.1f}, cy={h["cy"]:.1f}, conf={h["conf"]:.3f}')
    else:
        print(f'  YOLO f{target}: NONE')

# Tracker
tp = f'out/telemetry/{CLIP}.tracker_ball.jsonl'
print(f'\nTracker file: {tp} (exists={os.path.exists(tp)})')
tracker = {}
with open(tp) as fh:
    for line in fh:
        d = json.loads(line)
        if d.get('_meta'): continue
        if 'frame' in d:
            tracker[d['frame']] = d
print(f'Tracker entries: {len(tracker)}')

# Show tracker key names from first entry
if tracker:
    first_f = min(tracker.keys())
    print(f'Tracker keys (f{first_f}): {list(tracker[first_f].keys())}')

for target in [28, 95, 250, 293, 314, 332, 425, 450, 480]:
    if target in tracker:
        t = tracker[target]
        tx = t.get('cx', t.get('x'))
        ty = t.get('cy', t.get('y'))
        print(f'  Tracker f{target}: x={tx}, y={ty}')
    else:
        print(f'  Tracker f{target}: NOT PRESENT')

# Also check centroid
cp = f'out/telemetry/{CLIP}.ball.jsonl'
print(f'\nCentroid file: {cp} (exists={os.path.exists(cp)})')
centroid = {}
with open(cp) as fh:
    for line in fh:
        d = json.loads(line)
        if d.get('_meta'): continue
        if 'frame' in d:
            centroid[d['frame']] = d
print(f'Centroid entries: {len(centroid)}')
if centroid:
    min_f = min(centroid.keys())
    max_f = max(centroid.keys())
    print(f'Centroid range: f{min_f}-f{max_f}')
    print(f'Centroid keys: {list(centroid[min_f].keys())}')

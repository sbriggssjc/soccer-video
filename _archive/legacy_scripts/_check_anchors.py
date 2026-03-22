import json, os
os.chdir(r'D:\Projects\soccer-video')

# Find YOLO file
yp = None
for f in os.listdir('out/telemetry'):
    if '002__' in f and 'yolo_ball' in f:
        yp = os.path.join('out/telemetry', f)
        break
print(f'YOLO file: {yp}')

entries = []
with open(yp) as fh:
    for line in fh:
        d = json.loads(line)
        if d.get('_meta'):
            continue
        entries.append(d)

frames = set(e['frame'] for e in entries)
print(f'Total: {len(entries)} entries, {len(frames)} unique frames')

# Show all entries at key frames
for target in [28, 293, 314, 332, 425]:
    hits = [e for e in entries if e['frame'] == target]
    print(f'\nFrame {target}: {len(hits)} entries')
    for h in hits:
        print(f'  cx={h["cx"]:.1f}, cy={h["cy"]:.1f}, conf={h["conf"]:.3f}')

# Also load tracker
tp = None
for f in os.listdir('out/telemetry'):
    if '002__' in f and 'tracker_ball' in f:
        tp = os.path.join('out/telemetry', f)
        break

tracker = {}
with open(tp) as fh:
    for line in fh:
        d = json.loads(line)
        if d.get('_meta'):
            continue
        if 'frame' in d:
            tracker[d['frame']] = d

print(f'\nTracker entries: {len(tracker)}')
for target in [28, 293, 314, 332, 425]:
    if target in tracker:
        t = tracker[target]
        keys = [k for k in t.keys() if k not in ('frame', 't', '_meta')]
        print(f'Tracker f{target}: {dict((k, t[k]) for k in keys)}')
    else:
        print(f'Tracker f{target}: NOT PRESENT')

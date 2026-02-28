import json, os
os.chdir(r'D:\Projects\soccer-video')

# List all telemetry files with 002__
print("=== All 002__ telemetry files ===")
for f in sorted(os.listdir('out/telemetry')):
    if '002__' in f:
        print(f'  {f}')

# Find the RIGHT YOLO file (2026-02-23)
yp = None
for f in os.listdir('out/telemetry'):
    if '002__' in f and '2026-02-23' in f and 'yolo_ball' in f:
        yp = os.path.join('out/telemetry', f)
        break

if not yp:
    print("\nNo 2026-02-23 YOLO ball file found!")
    # Try broader search
    for f in os.listdir('out/telemetry'):
        if 'yolo_ball' in f:
            print(f'  Found: {f}')
else:
    print(f'\nYOLO file: {yp}')
    entries = []
    with open(yp) as fh:
        for line in fh:
            d = json.loads(line)
            if d.get('_meta'):
                continue
            entries.append(d)
    
    frames = set(e['frame'] for e in entries)
    print(f'Total: {len(entries)} entries, {len(frames)} unique frames')
    
    for target in [28, 293, 314, 332, 425]:
        hits = [e for e in entries if e['frame'] == target]
        print(f'\nFrame {target}: {len(hits)} entries')
        for h in hits:
            print(f'  cx={h["cx"]:.1f}, cy={h["cy"]:.1f}, conf={h["conf"]:.3f}')

# Find the RIGHT tracker file
tp = None
for f in os.listdir('out/telemetry'):
    if '002__' in f and '2026-02-23' in f and 'tracker_ball' in f:
        tp = os.path.join('out/telemetry', f)
        break

if not tp:
    print("\nNo 2026-02-23 tracker file found!")
    for f in os.listdir('out/telemetry'):
        if 'tracker_ball' in f:
            print(f'  Found: {f}')
else:
    print(f'\nTracker file: {tp}')
    tracker = {}
    with open(tp) as fh:
        for line in fh:
            d = json.loads(line)
            if d.get('_meta'):
                continue
            if 'frame' in d:
                tracker[d['frame']] = d
    print(f'Tracker entries: {len(tracker)}')
    for target in [28, 293, 314, 332, 425]:
        if target in tracker:
            t = tracker[target]
            keys = [k for k in sorted(t.keys()) if k not in ('_meta',)]
            print(f'Tracker f{target}: {dict((k, t[k]) for k in keys[:6])}')
        else:
            print(f'Tracker f{target}: NOT PRESENT')

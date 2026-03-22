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

frames = sorted(centroid.keys())
print(f'Total centroid samples: {len(frames)}')
print(f'Frame range: {frames[0]} - {frames[-1]}')

# Find gaps in centroid coverage
gaps = []
for i in range(len(frames)-1):
    if frames[i+1] - frames[i] > 1:
        gap_start = frames[i] + 1
        gap_end = frames[i+1] - 1
        gaps.append((gap_start, gap_end, gap_end - gap_start + 1))

print(f'\nCentroid gaps (missing frames):')
for gs, ge, gl in gaps:
    print(f'  f{gs}-f{ge} ({gl} frames)')

# Load tracker - check keys
tracker_path = None
for f in os.listdir('out/telemetry'):
    if '002__' in f and 'tracker_ball' in f:
        tracker_path = os.path.join('out/telemetry', f)
        break

print(f'\nTracker file: {tracker_path}')
with open(tracker_path) as fh:
    first_line = json.loads(fh.readline())
    print(f'Tracker keys: {list(first_line.keys())}')
    print(f'First entry: {first_line}')

# Load tracker properly
tracker = {}
with open(tracker_path) as fh:
    for line in fh:
        d = json.loads(line)
        # Try 'frame' first, then 'f', then index
        fkey = d.get('frame', d.get('f', None))
        if fkey is not None:
            tracker[fkey] = d

if not tracker:
    # Maybe it uses sequential indexing
    with open(tracker_path) as fh:
        for i, line in enumerate(fh):
            d = json.loads(line)
            tracker[i] = d
    print('Tracker uses sequential indexing')

tframes = sorted(tracker.keys())
print(f'Tracker samples: {len(tframes)}')
print(f'Tracker frame range: {tframes[0]} - {tframes[-1]}')

# What data exists in each zone?
zones = [
    ('Beginning', 0, 95),
    ('Winger', 250, 314),
    ('Shot/Save', 314, 425),
    ('Restart', 425, 496),
]

print('\n=== DATA COVERAGE SUMMARY ===')
for name, start, end in zones:
    c_count = sum(1 for f in range(start, end+1) if f in centroid)
    t_count = sum(1 for f in range(start, end+1) if f in tracker)
    total = end - start + 1
    print(f'{name} (f{start}-f{end}, {total} frames):')
    print(f'  Centroid: {c_count}/{total} ({100*c_count/total:.0f}%)')
    print(f'  Tracker:  {t_count}/{total} ({100*t_count/total:.0f}%)')

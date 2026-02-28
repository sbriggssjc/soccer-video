import json

# Load all detection sources
def load_jsonl(path):
    dets = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            dets[d['frame']] = (d['cx'], d['cy'], d['conf'])
    return dets

# yolov8n (original pipeline - 59 detections)
nano = load_jsonl(r'D:\Projects\soccer-video\out\telemetry\002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.yolo_ball.jsonl')

# yolov8x (168 detections from earlier run)
xlarge = load_jsonl(r'D:\Projects\soccer-video\out\telemetry\002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.yolo_ball.yolov8x.jsonl')

# v7 filtered cache (123 detections from BallTracker nano at conf>=0.10, trajectory-filtered)
with open(r'D:\Projects\soccer-video\_v7_yolo_cache.json') as f:
    v7_raw = json.load(f)
v7 = {}
for fstr, vals in v7_raw.items():
    v7[int(fstr)] = (vals[0], vals[1], vals[2])

print(f'Detection counts: nano={len(nano)}, v8x={len(xlarge)}, v7_filtered={len(v7)}')

# Frame coverage overlap
nano_frames = set(nano.keys())
xlarge_frames = set(xlarge.keys())
v7_frames = set(v7.keys())

print(f'\nFrame overlap:')
print(f'  nano & v8x: {len(nano_frames & xlarge_frames)}')
print(f'  nano & v7:  {len(nano_frames & v7_frames)}')
print(f'  v8x & v7:   {len(xlarge_frames & v7_frames)}')
print(f'  all three:  {len(nano_frames & xlarge_frames & v7_frames)}')
print(f'  any source: {len(nano_frames | xlarge_frames | v7_frames)}')

# Frames only in one source
only_nano = nano_frames - xlarge_frames - v7_frames
only_xlarge = xlarge_frames - nano_frames - v7_frames
only_v7 = v7_frames - nano_frames - xlarge_frames
print(f'\n  only nano: {len(only_nano)}')
print(f'  only v8x: {len(only_xlarge)}')
print(f'  only v7:  {len(only_v7)}')

# Position agreement where both detect
shared = nano_frames & xlarge_frames
if shared:
    diffs = []
    for f in sorted(shared):
        nx, ny, nc = nano[f]
        xx, xy, xc = xlarge[f]
        dist = ((nx-xx)**2 + (ny-xy)**2)**0.5
        diffs.append(dist)
    print(f'\nPosition agreement (nano vs v8x on {len(shared)} shared frames):')
    print(f'  mean dist: {sum(diffs)/len(diffs):.1f}px')
    print(f'  max dist:  {max(diffs):.1f}px')
    print(f'  <50px:     {sum(1 for d in diffs if d < 50)}')
    print(f'  >200px:    {sum(1 for d in diffs if d > 200)} (likely false positives)')

# Check big disagreements
print('\nBig disagreements (>200px) between nano and v8x:')
for f in sorted(shared):
    nx, ny, nc = nano[f]
    xx, xy, xc = xlarge[f]
    dist = ((nx-xx)**2 + (ny-xy)**2)**0.5
    if dist > 200:
        print(f'  frame {f}: nano=({nx:.0f},{ny:.0f} c={nc:.2f}) v8x=({xx:.0f},{xy:.0f} c={xc:.2f}) dist={dist:.0f}')

# Coverage gaps - where NO detection exists
all_detected = nano_frames | xlarge_frames | v7_frames
gaps = []
gap_start = None
for f in range(496):
    if f not in all_detected:
        if gap_start is None:
            gap_start = f
    else:
        if gap_start is not None:
            gaps.append((gap_start, f-1, f-gap_start))
            gap_start = None
if gap_start is not None:
    gaps.append((gap_start, 495, 496-gap_start))

gaps.sort(key=lambda x: -x[2])
print(f'\nLargest detection gaps (even combining all sources):')
for start, end, length in gaps[:10]:
    print(f'  frames {start}-{end} ({length} frames, {length/30:.1f}s)')

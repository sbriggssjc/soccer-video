"""
Optical-flow ball trajectory builder v3.
Changes from v2:
- Reads from .orig_backup files (originals were overwritten by injection)
- Adds f466 to FALSE_YOLO (static false detection at x=1713, conf=0.20)
"""
import cv2
import json
import os
import numpy as np
import sys
import time

os.chdir(r'D:\Projects\soccer-video')

CLIP = '002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00'
VIDEO = f'out/atomic_clips/2026-02-23__TSC_vs_NEOFC/{CLIP}.mp4'
# Read from ORIGINAL backups (current files have injected optical flow data)
YOLO_PATH = f'out/telemetry/{CLIP}.yolo_ball.jsonl.orig_backup'
TRACKER_PATH = f'out/telemetry/{CLIP}.tracker_ball.jsonl.orig_backup'
OUT_DIR = '_optflow_track'

os.makedirs(OUT_DIR, exist_ok=True)

# ── Step 1: Load anchors ──
print("=== Loading anchor data ===")

yolo = {}
with open(YOLO_PATH) as fh:
    for line in fh:
        d = json.loads(line)
        if d.get('_meta'): continue
        yolo[d['frame']] = d
print(f"YOLO entries: {len(yolo)}")

tracker = {}
with open(TRACKER_PATH) as fh:
    for line in fh:
        d = json.loads(line)
        if d.get('_meta'): continue
        if 'frame' in d:
            tracker[d['frame']] = d
print(f"Tracker entries: {len(tracker)}")

# Known false detections:
# f332: shadow/false detection
# f466: static spot at x=1713, conf=0.20 - NOT the ball
FALSE_YOLO = {332, 466}

anchors = {}
for f, d in sorted(yolo.items()):
    if f in FALSE_YOLO:
        print(f"  Skipping false YOLO f{f}")
        continue
    cx, cy, conf = d['cx'], d['cy'], d['conf']
    if f in tracker:
        tx, ty = tracker[f]['cx'], tracker[f]['cy']
        dist = ((cx - tx)**2 + (cy - ty)**2)**0.5
        if dist < 100:
            anchors[f] = (cx, cy, 'yolo+tracker', conf)
        else:
            anchors[f] = (tx, ty, 'tracker_override', tracker[f].get('conf', 0.5))
            print(f"  f{f}: YOLO/tracker disagree (dist={dist:.0f}), using tracker")
    else:
        anchors[f] = (cx, cy, 'yolo', conf)

for f, d in tracker.items():
    if f not in anchors:
        anchors[f] = (d['cx'], d['cy'], 'tracker', d.get('conf', 0.5))

anchor_frames = sorted(anchors.keys())
n_yolo = sum(1 for _, _, s, _ in anchors.values() if 'yolo' in s)
n_tracker = sum(1 for _, _, s, _ in anchors.values() if s == 'tracker')
n_override = sum(1 for _, _, s, _ in anchors.values() if s == 'tracker_override')
print(f"\nAnchors: {len(anchors)} total (yolo={n_yolo}, tracker={n_tracker}, override={n_override})")
print(f"Range: f{anchor_frames[0]}-f{anchor_frames[-1]}")

# ── Step 2: Load video frames ──
print("\n=== Loading video frames ===")
cap = cv2.VideoCapture(VIDEO)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video: {total_frames} frames @ {fps:.1f} fps")

frames_gray = []
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
for i in range(total_frames):
    ret, frame = cap.read()
    if not ret:
        break
    frames_gray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
cap.release()
print(f"Loaded {len(frames_gray)} frames into memory")

# ── Step 3: Optical flow tracking ──
lk_params = dict(
    winSize=(21, 21),
    maxLevel=4,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.003),
)
FB_THRESHOLD = 4.0
MAX_JUMP = 30.0

def track_direction(start_frame, start_x, start_y, direction, max_frames=200):
    """Track ball from start_frame. direction: +1=forward, -1=backward."""
    results = []
    offsets = [(0, 0), (-6, 0), (6, 0), (0, -6), (0, 6)]
    pts = np.array([[[start_x + dx, start_y + dy]] for dx, dy in offsets], dtype=np.float32)
    cur_frame = start_frame

    for step in range(1, max_frames + 1):
        nf = cur_frame + direction
        if nf < 0 or nf >= len(frames_gray):
            break
        new_pts, status, err = cv2.calcOpticalFlowPyrLK(
            frames_gray[cur_frame], frames_gray[nf], pts, None, **lk_params)
        back_pts, bstatus, berr = cv2.calcOpticalFlowPyrLK(
            frames_gray[nf], frames_gray[cur_frame], new_pts, None, **lk_params)
        good = []
        for i in range(len(pts)):
            if status[i][0] == 1 and bstatus[i][0] == 1:
                fb_dist = np.linalg.norm(back_pts[i][0] - pts[i][0])
                if fb_dist < FB_THRESHOLD:
                    good.append(new_pts[i][0])
        if len(good) < 2:
            break
        arr = np.array(good)
        mx, my = float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))
        prev_x = results[-1][1] if results else start_x
        prev_y = results[-1][2] if results else start_y
        if ((mx - prev_x)**2 + (my - prev_y)**2)**0.5 > MAX_JUMP:
            break
        results.append((nf, mx, my, len(good) / len(pts)))
        pts = np.array([[[mx + dx, my + dy]] for dx, dy in offsets], dtype=np.float32)
        cur_frame = nf
    return results

# ── Step 4: Track from every anchor ──
print("\n=== Tracking from anchors ===")
t0 = time.time()
all_tracks = {}

for i, af in enumerate(anchor_frames):
    ax, ay, src, aconf = anchors[af]
    if af not in all_tracks:
        all_tracks[af] = []
    all_tracks[af].append((ax, ay, 1.0, 0, af))

    fwd = track_direction(af, ax, ay, +1)
    for f, x, y, c in fwd:
        if f not in all_tracks:
            all_tracks[f] = []
        all_tracks[f].append((x, y, c, abs(f - af), af))

    bwd = track_direction(af, ax, ay, -1)
    for f, x, y, c in bwd:
        if f not in all_tracks:
            all_tracks[f] = []
        all_tracks[f].append((x, y, c, abs(f - af), af))

    if (i + 1) % 25 == 0 or i == len(anchor_frames) - 1:
        elapsed = time.time() - t0
        print(f"  Anchor {i+1}/{len(anchor_frames)}: f{af} ({src}) fwd={len(fwd)}f bwd={len(bwd)}f [{elapsed:.1f}s]")

# ── Step 5: Merge — prefer closest to anchor ──
trajectory = {}
for f in range(total_frames):
    if f in all_tracks:
        best = min(all_tracks[f], key=lambda c: c[3])
        trajectory[f] = (best[0], best[1], best[2])

covered = len(trajectory)
print(f"\nOptical flow coverage: {covered}/{total_frames} ({100*covered/total_frames:.1f}%)")

# ── Step 6: Fill gaps with linear interpolation ──
full_trajectory = {}
for f in range(total_frames):
    if f in trajectory:
        full_trajectory[f] = trajectory[f]
    else:
        prev_f = next((pf for pf in range(f-1, -1, -1) if pf in trajectory), None)
        next_f = next((nf for nf in range(f+1, total_frames) if nf in trajectory), None)
        if prev_f is not None and next_f is not None:
            t = (f - prev_f) / (next_f - prev_f)
            px, py, pc = trajectory[prev_f]
            nx, ny, nc = trajectory[next_f]
            full_trajectory[f] = (px + t*(nx-px), py + t*(ny-py), min(pc, nc) * 0.5)
        elif prev_f is not None:
            full_trajectory[f] = trajectory[prev_f]
        elif next_f is not None:
            full_trajectory[f] = trajectory[next_f]

print(f"Full trajectory: {len(full_trajectory)}/{total_frames} frames")

# ── Step 7: Save ──
ball_path = os.path.join(OUT_DIR, 'optflow_ball_path_v3.jsonl')
with open(ball_path, 'w') as fh:
    for f in range(total_frames):
        if f in full_trajectory:
            x, y, c = full_trajectory[f]
            src = 'tracked' if f in trajectory else 'interpolated'
            fh.write(json.dumps({
                'frame': f, 't': round(f / fps, 4),
                'cx': round(x, 2), 'cy': round(y, 2),
                'conf': round(c, 3), 'source': src
            }) + '\n')
print(f"Saved: {ball_path}")

# ── Step 8: Analysis ──
zones = [
    ('Beginning', 0, 95),
    ('Mid-field', 95, 250),
    ('Winger', 250, 314),
    ('Shot/Save', 314, 425),
    ('Restart', 425, 496),
]

print(f"\n=== ZONE ANALYSIS ===")
for name, s, e in zones:
    of_count = sum(1 for f in range(s, e) if f in trajectory)
    total = e - s
    if of_count > 0:
        xs = [trajectory[f][0] for f in range(s, e) if f in trajectory]
        print(f"  {name} (f{s}-f{e}): {of_count}/{total} tracked ({100*of_count/total:.0f}%), x=[{min(xs):.0f}, {max(xs):.0f}]")
    else:
        print(f"  {name} (f{s}-f{e}): 0/{total} tracked (0%)")

print(f"\n=== RESTART ZONE DETAIL (f410-f496) ===")
for f in range(410, 496):
    if f in full_trajectory:
        x, y, c = full_trajectory[f]
        src = 'T' if f in trajectory else 'I'
        anchor_info = ''
        if f in all_tracks:
            best = min(all_tracks[f], key=lambda c: c[3])
            anchor_info = f' (anchor f{best[4]}, dist={best[3]})'
        print(f"  f{f}: x={x:.0f}, y={y:.0f} [{src}]{anchor_info}")

print(f"\n=== TIMELINE (every 30 frames) ===")
for f in range(0, total_frames, 30):
    if f in full_trajectory:
        x, y, c = full_trajectory[f]
        src = 'T' if f in trajectory else 'I'
        print(f"  f{f}: x={x:.0f} [{src}, conf={c:.2f}]")

elapsed = time.time() - t0
print(f"\nTotal time: {elapsed:.1f}s")
print(f"Done! Ball path: {ball_path}")

"""
Full optical-flow ball trajectory builder.

Strategy: start from CONFIRMED YOLO ball positions, track the actual ball
pixels forward and backward using Lucas-Kanade optical flow, merge all
tracks into a single pixel-accurate trajectory covering every frame.

Output: a ball path file that can be injected directly into the render
pipeline, bypassing the fusion layer entirely.
"""
import cv2
import json
import os
import numpy as np
import sys

VIDEO = r"out\atomic_clips\2026-02-23__TSC_vs_NEOFC\002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.mp4"

# ── Step 1: Load the YOLO detections that ball_telemetry selects ──
# ball_telemetry uses BallSample objects from yolo_ball.jsonl.
# It keeps one detection per frame (highest conf).
# We need the SAME positions it uses.
YOLO_PATH = None
for f in os.listdir('out/telemetry'):
    if '002__' in f and 'yolo_ball' in f:
        YOLO_PATH = os.path.join('out/telemetry', f)
        break

# Load raw YOLO (keep highest-conf per frame, matching ball_telemetry)
raw_yolo = {}
with open(YOLO_PATH) as fh:
    for line in fh:
        d = json.loads(line)
        if d.get('_meta'):
            continue
        f = d['frame']
        if f not in raw_yolo or d['conf'] > raw_yolo[f]['conf']:
            raw_yolo[f] = d

# Also load tracker positions as additional anchors
TRACKER_PATH = None
for f in os.listdir('out/telemetry'):
    if '002__' in f and 'tracker_ball' in f:
        TRACKER_PATH = os.path.join('out/telemetry', f)
        break

tracker_data = {}
with open(TRACKER_PATH) as fh:
    for line in fh:
        d = json.loads(line)
        if d.get('_meta'):
            continue
        if 'frame' in d:
            tracker_data[d['frame']] = d

# The confirmed YOLO for ball_telemetry are the ones it would select.
# From our analysis: all 59 entries deduplicated to one per frame.
# The 13 with conf <= 0.25 are false.
# But for optical flow, we want the REAL ball at each frame.
# Problem: some YOLO entries at the same frame might be different objects.
# Solution: use tracker data to validate which YOLO entry is the ball.

# For frames with tracker data, pick YOLO entry closest to tracker position
# For frames without tracker, use ball_telemetry's default (highest conf)
anchors = {}  # frame -> (x, y, source, conf)

for f, d in sorted(raw_yolo.items()):
    cx, cy, conf = d['cx'], d['cy'], d['conf']
    
    # Check if tracker exists at this frame for validation
    if f in tracker_data:
        tx = tracker_data[f].get('cx', tracker_data[f].get('x', None))
        ty = tracker_data[f].get('cy', tracker_data[f].get('y', None))
        if tx is not None:
            # Multiple YOLO at this frame? Check all entries
            frame_entries = []
            with open(YOLO_PATH) as fh2:
                for line2 in fh2:
                    d2 = json.loads(line2)
                    if d2.get('_meta'):
                        continue
                    if d2['frame'] == f:
                        dist = ((d2['cx'] - tx)**2 + (d2['cy'] - ty)**2)**0.5
                        frame_entries.append((dist, d2))
            if frame_entries:
                frame_entries.sort(key=lambda x: x[0])
                best = frame_entries[0][1]
                anchors[f] = (best['cx'], best['cy'], 'yolo+tracker', best['conf'])
                continue
    
    anchors[f] = (cx, cy, 'yolo', conf)

# Add tracker-only frames as secondary anchors
for f, d in tracker_data.items():
    if f not in anchors:
        tx = d.get('cx', d.get('x', None))
        ty = d.get('cy', d.get('y', None))
        if tx is not None:
            anchors[f] = (float(tx), float(ty), 'tracker', 0.50)

anchor_frames = sorted(anchors.keys())
print(f"Total anchors: {len(anchor_frames)} (YOLO={sum(1 for _,_,s,_ in anchors.values() if 'yolo' in s)}, tracker={sum(1 for _,_,s,_ in anchors.values() if s=='tracker')})")
print(f"Anchor frame range: {anchor_frames[0]}-{anchor_frames[-1]}")

# ── Step 2: Optical flow tracking ──
cap = cv2.VideoCapture(VIDEO)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video: {total_frames} frames @ {fps:.1f} fps")

# Pre-read all frames into memory for bidirectional tracking
print("Loading all frames into memory...")
frames_gray = []
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
for i in range(total_frames):
    ret, frame = cap.read()
    if not ret:
        break
    frames_gray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
cap.release()
print(f"Loaded {len(frames_gray)} frames")

lk_params = dict(
    winSize=(21, 21),
    maxLevel=4,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.003),
)

FB_THRESHOLD = 4.0  # forward-backward error threshold

def track_direction(start_frame, start_x, start_y, direction, max_frames=200):
    """Track ball from start_frame in given direction (+1=forward, -1=backward).
    Returns list of (frame, x, y, confidence)."""
    results = []
    
    # Use 5-point grid around the ball for robust tracking
    offsets = [(0,0), (-6,0), (6,0), (0,-6), (0,6)]
    pts = np.array([[[start_x+dx, start_y+dy]] for dx,dy in offsets], dtype=np.float32)
    
    cur_frame = start_frame
    for step in range(1, max_frames + 1):
        next_frame = cur_frame + direction
        if next_frame < 0 or next_frame >= len(frames_gray):
            break
        
        # Forward flow
        new_pts, status, err = cv2.calcOpticalFlowPyrLK(
            frames_gray[cur_frame], frames_gray[next_frame], pts, None, **lk_params
        )
        
        # Backward check
        back_pts, back_status, back_err = cv2.calcOpticalFlowPyrLK(
            frames_gray[next_frame], frames_gray[cur_frame], new_pts, None, **lk_params
        )
        
        # Check consistency
        good_pts = []
        for i in range(len(pts)):
            if status[i][0] == 1 and back_status[i][0] == 1:
                fb_dist = np.linalg.norm(back_pts[i][0] - pts[i][0])
                if fb_dist < FB_THRESHOLD:
                    good_pts.append(new_pts[i][0])
        
        if len(good_pts) < 2:
            break  # lost tracking
        
        good_arr = np.array(good_pts)
        mx = float(np.median(good_arr[:, 0]))
        my = float(np.median(good_arr[:, 1]))
        conf = len(good_pts) / len(pts)
        
        # Sanity: ball shouldn't jump more than 30px/frame
        prev_x = results[-1][1] if results else start_x
        prev_y = results[-1][2] if results else start_y
        jump = ((mx - prev_x)**2 + (my - prev_y)**2)**0.5
        if jump > 30.0:
            break  # unrealistic jump
        
        results.append((next_frame, mx, my, conf))
        
        # Update points for next iteration
        pts = np.array([[[mx+dx, my+dy]] for dx,dy in offsets], dtype=np.float32)
        cur_frame = next_frame
    
    return results

# ── Step 3: Track from every anchor, both directions ──
# Store: frame -> list of (x, y, conf, distance_from_anchor)
all_tracks = {}  # frame -> [(x, y, conf, dist_from_anchor, anchor_frame), ...]

for i, af in enumerate(anchor_frames):
    ax, ay, src, aconf = anchors[af]
    
    # Add the anchor itself
    if af not in all_tracks:
        all_tracks[af] = []
    all_tracks[af].append((ax, ay, 1.0, 0, af))
    
    # Track forward
    fwd = track_direction(af, ax, ay, +1)
    for f, x, y, c in fwd:
        if f not in all_tracks:
            all_tracks[f] = []
        dist = abs(f - af)
        all_tracks[f].append((x, y, c, dist, af))
    
    # Track backward
    bwd = track_direction(af, ax, ay, -1)
    for f, x, y, c in bwd:
        if f not in all_tracks:
            all_tracks[f] = []
        dist = abs(f - af)
        all_tracks[f].append((x, y, c, dist, af))
    
    fwd_len = len(fwd)
    bwd_len = len(bwd)
    if (i + 1) % 20 == 0 or i == len(anchor_frames) - 1:
        print(f"  Anchor {i+1}/{len(anchor_frames)}: f{af} (x={ax:.0f}) fwd={fwd_len}f bwd={bwd_len}f")

# ── Step 4: Merge tracks — prefer closer-to-anchor positions ──
trajectory = {}  # frame -> (x, y, conf)
for f in range(total_frames):
    if f in all_tracks:
        candidates = all_tracks[f]
        # Pick the candidate with smallest distance from its anchor
        # (closer to anchor = more reliable tracking)
        best = min(candidates, key=lambda c: c[3])  # sort by dist_from_anchor
        trajectory[f] = (best[0], best[1], best[2])
    # else: gap (will interpolate)

covered = len(trajectory)
print(f"\nOptical flow coverage: {covered}/{total_frames} ({100*covered/total_frames:.1f}%)")

# ── Step 5: Fill gaps with linear interpolation ──
traj_frames = sorted(trajectory.keys())
full_trajectory = {}

for f in range(total_frames):
    if f in trajectory:
        full_trajectory[f] = trajectory[f]
    else:
        # Find nearest covered frames before and after
        prev_f = None
        next_f = None
        for pf in range(f-1, -1, -1):
            if pf in trajectory:
                prev_f = pf
                break
        for nf in range(f+1, total_frames):
            if nf in trajectory:
                next_f = nf
                break
        
        if prev_f is not None and next_f is not None:
            t = (f - prev_f) / (next_f - prev_f)
            px, py, pc = trajectory[prev_f]
            nx, ny, nc = trajectory[next_f]
            x = px + t * (nx - px)
            y = py + t * (ny - py)
            c = min(pc, nc) * 0.5  # lower conf for interpolated
            full_trajectory[f] = (x, y, c)
        elif prev_f is not None:
            full_trajectory[f] = trajectory[prev_f]
        elif next_f is not None:
            full_trajectory[f] = trajectory[next_f]

print(f"Full trajectory: {len(full_trajectory)}/{total_frames} frames")

# ── Step 6: Save as ball path JSONL (compatible with --ball-path) ──
OUT_DIR = "_optflow_track"
os.makedirs(OUT_DIR, exist_ok=True)

# Save in ball_telemetry BallSample format
ball_path = os.path.join(OUT_DIR, 'optflow_ball_path.jsonl')
with open(ball_path, 'w') as fh:
    for f in range(total_frames):
        if f in full_trajectory:
            x, y, c = full_trajectory[f]
            entry = {
                'frame': f,
                't': f / fps,
                'x': round(x, 2),
                'y': round(y, 2),
                'conf': round(c, 3),
            }
            fh.write(json.dumps(entry) + '\n')

print(f"Ball path saved: {ball_path}")

# ── Step 7: Analysis ──
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
        print(f"{name} (f{s}-f{e}): {of_count}/{total} tracked ({100*of_count/total:.0f}%), x=[{min(xs):.0f}, {max(xs):.0f}]")
    else:
        print(f"{name} (f{s}-f{e}): 0/{total} tracked (0%)")

# Timeline
print(f"\n=== BALL X TIMELINE (optical flow) ===")
for f in range(0, total_frames, 30):
    if f in full_trajectory:
        x, y, c = full_trajectory[f]
        src = "tracked" if f in trajectory else "interp"
        print(f"  f{f}: x={x:.0f} [{src}, conf={c:.2f}]")

# Shot/save zone detail
print(f"\n=== SHOT/SAVE ZONE (f314-f425) ===")
for f in range(314, 426, 10):
    if f in full_trajectory:
        x, y, c = full_trajectory[f]
        src = "tracked" if f in trajectory else "interp"
        print(f"  f{f}: x={x:.0f}, y={y:.0f} [{src}, conf={c:.2f}]")

print(f"\nDone! Use --ball-path {ball_path} to inject into render.")

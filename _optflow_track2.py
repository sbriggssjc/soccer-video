"""
Optical-flow ball tracker v2 — uses ball_telemetry's verified YOLO positions.

Specifically tracks from the last real YOLO before the shot/save data desert
(f313/f314 area) forward through the entire shot zone to see where the ball
actually goes pixel-by-pixel.
"""
import cv2
import json
import os
import numpy as np

VIDEO = r"out\atomic_clips\2026-02-23__TSC_vs_NEOFC\002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.mp4"

# Load the YOLO detections that ball_telemetry uses (from BallSample format)
# These are the cx/cy positions from the yolo_ball file, but we need the
# ones that match ball_telemetry's selection. Use the known verified positions.
# From our earlier analysis, the real YOLO in the winger/shot zone:
VERIFIED_YOLO = {
    253: (1379, 450), 254: (1379, 450),
    272: (1727, 450), 276: (1702, 450), 280: (1683, 450),
    285: (1683, 450), 286: (1705, 450), 287: (1727, 450),
    288: (1746, 450), 289: (1760, 450), 290: (1771, 450),
    291: (1776, 450), 292: (1777, 450), 293: (1776, 450),
    295: (1774, 450), 296: (1774, 450), 297: (1774, 450),
    302: (1792, 450), 306: (1792, 450), 307: (1782, 450),
    308: (1771, 450), 309: (1707, 450), 313: (1615, 450),
    314: (1607, 450),
    # After shot zone:
    425: (95, 450), 426: (96, 450),
}

# We don't have exact y coords from the earlier printout, so let's load
# from the actual YOLO file and match by frame+x
YOLO_PATH = None
for f in os.listdir('out/telemetry'):
    if '002__' in f and 'yolo_ball' in f:
        YOLO_PATH = os.path.join('out/telemetry', f)
        break

# Load ALL yolo entries (multiple per frame possible)
all_yolo = []
with open(YOLO_PATH) as fh:
    for line in fh:
        d = json.loads(line)
        if d.get('_meta'):
            continue
        all_yolo.append(d)

print(f"Total YOLO entries in file: {len(all_yolo)}")

# For each verified frame, find the matching YOLO entry by x proximity
verified_with_y = {}
for frame, (vx, _) in VERIFIED_YOLO.items():
    candidates = [d for d in all_yolo if d['frame'] == frame]
    if candidates:
        # Pick the one closest to our verified x
        best = min(candidates, key=lambda d: abs(d['cx'] - vx))
        verified_with_y[frame] = (best['cx'], best['cy'])
    else:
        verified_with_y[frame] = (vx, 450.0)  # fallback

print(f"\nVerified YOLO positions with actual y-coordinates:")
for f in sorted(verified_with_y.keys()):
    x, y = verified_with_y[f]
    print(f"  f{f}: x={x:.0f}, y={y:.0f}")

# Open video
cap = cv2.VideoCapture(VIDEO)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"\nVideo: {total_frames} frames @ {fps:.1f} fps")

# Lucas-Kanade parameters - tuned for ball tracking
lk_params = dict(
    winSize=(21, 21),
    maxLevel=4,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.003),
)

# Track forward from key anchor points
# We want to track from the last few YOLO before the shot zone
track_starts = [293, 302, 308, 313, 314]  # multiple anchors for redundancy

all_tracks = {}
for start_f in track_starts:
    if start_f not in verified_with_y:
        continue
    sx, sy = verified_with_y[start_f]
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    ret, prev_frame = cap.read()
    if not ret:
        continue
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Use a grid of points around the ball for more robust tracking
    offsets = [(0,0), (-8,0), (8,0), (0,-8), (0,8)]
    pts = np.array([[[sx+dx, sy+dy]] for dx,dy in offsets], dtype=np.float32)
    
    track = [(start_f, sx, sy, 'yolo', 1.0)]
    
    max_frames = min(200, total_frames - start_f)
    for offset in range(1, max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Forward flow
        new_pts, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, pts, None, **lk_params
        )
        
        # Backward check
        back_pts, back_status, back_err = cv2.calcOpticalFlowPyrLK(
            gray, prev_gray, new_pts, None, **lk_params
        )
        
        # Check forward-backward consistency for each point
        good_pts = []
        for i in range(len(pts)):
            if status[i][0] == 1 and back_status[i][0] == 1:
                fb_dist = np.linalg.norm(back_pts[i][0] - pts[i][0])
                if fb_dist < 3.0:
                    good_pts.append(new_pts[i][0])
        
        if len(good_pts) < 2:
            # Lost tracking
            track.append((start_f + offset, float(pts[0][0][0]), float(pts[0][0][1]), 'lost', 0.0))
            break
        
        # Median of good points (robust to outliers)
        good_arr = np.array(good_pts)
        mx, my = float(np.median(good_arr[:, 0])), float(np.median(good_arr[:, 1]))
        confidence = len(good_pts) / len(pts)
        
        track.append((start_f + offset, mx, my, 'tracked', confidence))
        
        # Update points centered on new position
        pts = np.array([[[mx+dx, my+dy]] for dx,dy in offsets], dtype=np.float32)
        prev_gray = gray
    
    all_tracks[start_f] = track
    end_f = track[-1][0]
    end_status = track[-1][3]
    end_x = track[-1][1]
    print(f"\n  From f{start_f} (x={sx:.0f}): tracked {len(track)-1} frames to f{end_f} (x={end_x:.0f}, {end_status})")
    
    # Show trajectory at key frames
    for f, x, y, st, conf in track:
        if f in [314, 330, 350, 370, 390, 410, 425] or st == 'lost':
            print(f"    f{f}: x={x:.0f}, y={y:.0f} [{st}, conf={conf:.2f}]")

cap.release()

# Composite trajectory (prefer later anchors as more recent)
trajectory = {}
for start_f in sorted(all_tracks.keys()):
    for f, x, y, st, conf in all_tracks[start_f]:
        if f not in trajectory or conf > trajectory[f][4]:
            trajectory[f] = (f, x, y, st, conf, start_f)

# Save
OUT_DIR = "_optflow_track"
os.makedirs(OUT_DIR, exist_ok=True)
traj_path = os.path.join(OUT_DIR, 'trajectory_v2.jsonl')
with open(traj_path, 'w') as fh:
    for f in sorted(trajectory.keys()):
        t = trajectory[f]
        fh.write(json.dumps({
            'frame': t[0], 'x': round(t[1], 1), 'y': round(t[2], 1),
            'status': t[3], 'confidence': round(t[4], 2), 'source_yolo': t[5]
        }) + '\n')

print(f"\n=== SHOT/SAVE ZONE COMPARISON ===")
print(f"{'Frame':>6} {'OptFlow_x':>10} {'ShotHold_x':>11} {'Diff':>6}")
for f in range(314, 426, 5):
    if f in trajectory:
        of_x = trajectory[f][1]
        sh_x = 1357 if f > 329 else 1607  # approximate shot-hold position
        print(f"f{f:>5} {of_x:>10.0f} {sh_x:>11.0f} {of_x-sh_x:>+6.0f}")

# Plot
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    for start_f, track in all_tracks.items():
        frames = [t[0] for t in track]
        xs = [t[1] for t in track]
        label = f'From YOLO f{start_f}'
        ax.plot(frames, xs, '-', linewidth=1.5, alpha=0.7, label=label)
    
    # Mark zones
    ax.axvspan(314, 425, alpha=0.15, color='red', label='Shot/Save zone')
    ax.axhline(y=1607, color='green', linestyle='--', alpha=0.5, label='Shot-hold x=1607')
    ax.axhline(y=1357, color='orange', linestyle='--', alpha=0.5, label='Shot-hold decel x=1357')
    
    ax.set_xlabel('Frame')
    ax.set_ylabel('Ball X position (px)')
    ax.set_title('Optical Flow Ball Tracking v2 (verified YOLO anchors)')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plot_path = os.path.join(OUT_DIR, 'trajectory_v2_plot.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {plot_path}")
except ImportError:
    print("matplotlib not available")

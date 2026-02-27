"""
Optical-flow ball tracker — independent verification of ball position.

For each confirmed YOLO detection (conf > 0.25), uses Lucas-Kanade sparse
optical flow to track the ball's pixel patch forward frame-by-frame until:
  - The next YOLO detection is reached, OR
  - Tracking confidence drops below threshold, OR
  - Max frames exceeded

Outputs:
  - _optflow_track/trajectory.jsonl  (per-frame x,y positions)
  - _optflow_track/summary.txt       (human-readable analysis)
  - _optflow_track/trajectory_plot.png (visual plot)
"""
import cv2
import json
import os
import numpy as np

# Paths
VIDEO = r"out\atomic_clips\2026-02-23__TSC_vs_NEOFC\002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.mp4"
YOLO_PATH = None
for f in os.listdir('out/telemetry'):
    if '002__' in f and 'yolo_ball' in f:
        YOLO_PATH = os.path.join('out/telemetry', f)
        break

OUT_DIR = "_optflow_track"
os.makedirs(OUT_DIR, exist_ok=True)

# Load YOLO detections (keep highest-conf per frame)
yolo = {}
with open(YOLO_PATH) as fh:
    for line in fh:
        d = json.loads(line)
        if d.get('_meta'):
            continue
        f = d['frame']
        if f not in yolo or d['conf'] > yolo[f]['conf']:
            yolo[f] = d

# Filter to real YOLO only (conf > 0.25)
real_yolo = {f: d for f, d in yolo.items() if d['conf'] > 0.25}
real_frames = sorted(real_yolo.keys())
print(f"Real YOLO detections: {len(real_frames)} (conf > 0.25)")
print(f"Frame range: {real_frames[0]} - {real_frames[-1]}")

# Open video
cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO}")
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video: {total_frames} frames @ {fps:.1f} fps")

# Lucas-Kanade parameters
lk_params = dict(
    winSize=(31, 31),      # larger window for ball tracking
    maxLevel=3,            # pyramid levels
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

# Track from each YOLO detection forward
all_tracks = []  # list of (start_frame, [(frame, x, y, status), ...])

for idx, start_f in enumerate(real_frames):
    det = real_yolo[start_f]
    start_x, start_y = float(det['cx']), float(det['cy'])
    
    # Find next real YOLO frame (or end of clip)
    next_f = total_frames
    for nf in real_frames:
        if nf > start_f:
            next_f = nf
            break
    
    max_track = min(next_f - start_f, 150)  # cap at 150 frames
    if max_track <= 1:
        continue
    
    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    ret, prev_frame = cap.read()
    if not ret:
        continue
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Initial point
    pt = np.array([[[start_x, start_y]]], dtype=np.float32)
    track = [(start_f, start_x, start_y, 'yolo')]
    
    lost = False
    for offset in range(1, max_track):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Forward optical flow
        new_pt, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, pt, None, **lk_params
        )
        
        # Backward check (for robustness)
        if status[0][0] == 1:
            back_pt, back_status, back_err = cv2.calcOpticalFlowPyrLK(
                gray, prev_gray, new_pt, None, **lk_params
            )
            if back_status[0][0] == 1:
                fb_dist = np.linalg.norm(back_pt[0][0] - pt[0][0])
                if fb_dist > 5.0:  # forward-backward error too high
                    track.append((start_f + offset, float(new_pt[0][0][0]), float(new_pt[0][0][1]), 'lost_fb'))
                    lost = True
                    break
            else:
                track.append((start_f + offset, float(new_pt[0][0][0]), float(new_pt[0][0][1]), 'lost_back'))
                lost = True
                break
        else:
            track.append((start_f + offset, float(pt[0][0][0]), float(pt[0][0][1]), 'lost_fwd'))
            lost = True
            break
        
        nx, ny = float(new_pt[0][0][0]), float(new_pt[0][0][1])
        track.append((start_f + offset, nx, ny, 'tracked'))
        pt = new_pt
        prev_gray = gray
    
    all_tracks.append((start_f, track))
    end_f = track[-1][0]
    end_status = track[-1][3]
    print(f"  YOLO f{start_f} (x={start_x:.0f}) -> tracked to f{end_f} ({len(track)-1} frames, {end_status})")

cap.release()

# Build per-frame trajectory from all tracks
trajectory = {}
for start_f, track in all_tracks:
    for f, x, y, status in track:
        if f not in trajectory or status in ('yolo', 'tracked'):
            trajectory[f] = {'frame': f, 'x': x, 'y': y, 'status': status, 'source_yolo': start_f}

# Save trajectory
traj_path = os.path.join(OUT_DIR, 'trajectory.jsonl')
with open(traj_path, 'w') as fh:
    for f in sorted(trajectory.keys()):
        fh.write(json.dumps(trajectory[f]) + '\n')
print(f"\nTrajectory saved: {traj_path} ({len(trajectory)} frames)")

# Analysis summary
summary_lines = []
summary_lines.append(f"Optical Flow Ball Tracking Summary")
summary_lines.append(f"=" * 50)
summary_lines.append(f"Real YOLO detections: {len(real_frames)}")
summary_lines.append(f"Tracks generated: {len(all_tracks)}")
summary_lines.append(f"Total tracked frames: {len(trajectory)}")
summary_lines.append(f"Coverage: {len(trajectory)}/{total_frames} ({100*len(trajectory)/total_frames:.1f}%)")

# Zone analysis
zones = [
    ('Beginning', 0, 95),
    ('Mid-field', 95, 250),
    ('Winger', 250, 314),
    ('Shot/Save', 314, 425),
    ('Restart', 425, 496),
]

summary_lines.append(f"\nZone Coverage:")
for name, s, e in zones:
    count = sum(1 for f in range(s, e) if f in trajectory)
    total = e - s
    if count > 0:
        xs = [trajectory[f]['x'] for f in range(s, e+1) if f in trajectory]
        summary_lines.append(f"  {name} (f{s}-f{e}): {count}/{total} ({100*count/total:.0f}%), x=[{min(xs):.0f}, {max(xs):.0f}]")
    else:
        summary_lines.append(f"  {name} (f{s}-f{e}): {count}/{total} (0%)")

# Shot/save zone detail
summary_lines.append(f"\nShot/Save Zone Detail (f314-f425):")
for f in range(314, 426):
    if f in trajectory:
        t = trajectory[f]
        summary_lines.append(f"  f{f}: x={t['x']:.0f}, y={t['y']:.0f} [{t['status']}, from YOLO f{t['source_yolo']}]")

summary_path = os.path.join(OUT_DIR, 'summary.txt')
with open(summary_path, 'w') as fh:
    fh.write('\n'.join(summary_lines))
print(f"Summary saved: {summary_path}")

# Print key results
for line in summary_lines:
    print(line)

# Try to create plot
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    frames_sorted = sorted(trajectory.keys())
    xs = [trajectory[f]['x'] for f in frames_sorted]
    ys = [trajectory[f]['frame'] for f in frames_sorted]
    statuses = [trajectory[f]['status'] for f in frames_sorted]
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Color by status
    colors = {'yolo': 'green', 'tracked': 'blue', 'lost_fb': 'red', 'lost_fwd': 'red', 'lost_back': 'red'}
    for status in ['tracked', 'yolo', 'lost_fb', 'lost_fwd', 'lost_back']:
        mask = [s == status for s in statuses]
        fx = [frames_sorted[i] for i in range(len(mask)) if mask[i]]
        xx = [xs[i] for i in range(len(mask)) if mask[i]]
        if fx:
            ax.scatter(fx, xx, c=colors.get(status, 'gray'), s=3, alpha=0.6, label=status)
    
    # Mark zones
    for name, s, e in zones:
        ax.axvspan(s, e, alpha=0.1, label=name)
    
    ax.set_xlabel('Frame')
    ax.set_ylabel('Ball X position (px)')
    ax.set_title('Optical Flow Ball Tracking (independent verification)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plot_path = os.path.join(OUT_DIR, 'trajectory_plot.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {plot_path}")
except ImportError:
    print("matplotlib not available, skipping plot")

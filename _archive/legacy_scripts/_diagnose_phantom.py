"""Diagnose the two-ball phantom problem.

A phantom = static object (cone, marking) that YOLO keeps detecting.
In frame-space, it sits at ~same position across frames.
The real ball MOVES across the frame.

Also: load camera shifts to convert to world-space and see if
phantom detections trace the camera motion (static object signature).
"""
import json
import numpy as np
from pathlib import Path

REPO = Path(r"D:\Projects\soccer-video")
TEL = REPO / "out" / "telemetry"
STEM = "002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00"

# Load consensus YOLO detections
dets = {}
with open(TEL / f"{STEM}.yolo_ball.consensus.jsonl") as f:
    for line in f:
        d = json.loads(line)
        dets[d["frame"]] = (d["cx"], d["cy"], d["conf"])

# Load camera shifts
cam_shifts = np.load(TEL / f"{STEM}.ball.cam_shifts.npy")
print(f"Camera shifts shape: {cam_shifts.shape}")
print(f"Total detections: {len(dets)}")

# Convert to world-space (add cumulative camera displacement)
cum_dx = np.cumsum(cam_shifts[:, 0]) if cam_shifts.shape[1] >= 1 else np.zeros(496)
cum_dy = np.cumsum(cam_shifts[:, 1]) if cam_shifts.shape[1] >= 2 else np.zeros(496)

frames = sorted(dets.keys())
frame_x = []  # frame-space x
frame_y = []
world_x = []  # world-space x
world_y = []
confs = []

for f in frames:
    cx, cy, conf = dets[f]
    frame_x.append(cx)
    frame_y.append(cy)
    world_x.append(cx + cum_dx[f])
    world_y.append(cy + cum_dy[f])
    confs.append(conf)

frame_x = np.array(frame_x)
frame_y = np.array(frame_y)
world_x = np.array(world_x)
world_y = np.array(world_y)
frames = np.array(frames)
confs = np.array(confs)

print(f"\nFrame-space x range: {frame_x.min():.0f} - {frame_x.max():.0f}")
print(f"Frame-space y range: {frame_y.min():.0f} - {frame_y.max():.0f}")
print(f"World-space x range: {world_x.min():.0f} - {world_x.max():.0f}")
print(f"World-space y range: {world_y.min():.0f} - {world_y.max():.0f}")

# --- ANALYSIS 1: Look for frame-space clusters ---
# If a phantom is a static object, multiple detections will cluster
# at similar frame-space coordinates
print("\n=== Frame-space clustering ===")
from collections import defaultdict

# Group detections by spatial proximity in frame-space
CLUSTER_RADIUS = 80  # pixels
clusters = []
assigned = set()

for i in range(len(frames)):
    if i in assigned:
        continue
    cluster = [i]
    assigned.add(i)
    for j in range(i+1, len(frames)):
        if j in assigned:
            continue
        dist = ((frame_x[i] - frame_x[j])**2 + (frame_y[i] - frame_y[j])**2)**0.5
        if dist < CLUSTER_RADIUS:
            cluster.append(j)
            assigned.add(j)
    clusters.append(cluster)

# Sort clusters by size
clusters.sort(key=len, reverse=True)
print(f"Found {len(clusters)} frame-space clusters (radius={CLUSTER_RADIUS}px)")
for ci, cluster in enumerate(clusters[:10]):
    cl_frames = frames[cluster]
    cl_x = frame_x[cluster]
    cl_y = frame_y[cluster]
    cl_conf = confs[cluster]
    # Check temporal spread - real ball has spread-out frames, phantom has clustered
    frame_span = cl_frames.max() - cl_frames.min() if len(cl_frames) > 1 else 0
    print(f"  Cluster {ci}: {len(cluster)} detections, "
          f"frames {cl_frames.min()}-{cl_frames.max()} (span={frame_span}), "
          f"x={cl_x.mean():.0f}+/-{cl_x.std():.0f}, "
          f"y={cl_y.mean():.0f}+/-{cl_y.std():.0f}, "
          f"conf={cl_conf.mean():.2f}")

# --- ANALYSIS 2: Jump detection ---
# Look for large frame-to-frame jumps in world-space
print("\n=== World-space jump analysis ===")
for i in range(1, len(frames)):
    dt = frames[i] - frames[i-1]
    if dt <= 3:  # only check consecutive or near-consecutive frames
        dx = world_x[i] - world_x[i-1]
        dy = world_y[i] - world_y[i-1]
        jump = (dx**2 + dy**2)**0.5
        if jump > 200:  # big jump
            print(f"  Jump at f{frames[i-1]}->{frames[i]}: "
                  f"world ({world_x[i-1]:.0f},{world_y[i-1]:.0f}) -> "
                  f"({world_x[i]:.0f},{world_y[i]:.0f}) = {jump:.0f}px, "
                  f"frame ({frame_x[i-1]:.0f},{frame_y[i-1]:.0f}) -> "
                  f"({frame_x[i]:.0f},{frame_y[i]:.0f})")

# --- ANALYSIS 3: Static object signature ---
# A static object moves in world-space at exactly -camera_velocity
# Check if any detection's world-position is suspiciously correlated
# with camera position (i.e., stays fixed in frame-space)
print("\n=== Potential static phantom objects ===")
# Group consecutive detections and check frame-space movement
groups = []
current_group = [0]
for i in range(1, len(frames)):
    if frames[i] - frames[i-1] <= 5:  # gap <= 5 frames
        current_group.append(i)
    else:
        groups.append(current_group)
        current_group = [i]
groups.append(current_group)

for gi, group in enumerate(groups):
    if len(group) < 2:
        continue
    gf = frames[group]
    gx = frame_x[group]
    gy = frame_y[group]
    gc = confs[group]
    # Frame-space velocity (low = potentially static phantom)
    frame_velocities = ((np.diff(gx)**2 + np.diff(gy)**2)**0.5) / np.maximum(np.diff(gf), 1)
    mean_vel = frame_velocities.mean()
    
    # World-space velocity (should be different for real ball)
    gwx = world_x[group]
    gwy = world_y[group]
    world_velocities = ((np.diff(gwx)**2 + np.diff(gwy)**2)**0.5) / np.maximum(np.diff(gf), 1)
    world_mean_vel = world_velocities.mean()
    
    is_static = mean_vel < 5.0 and len(group) >= 3  # very low frame-space movement
    marker = " *** LIKELY PHANTOM ***" if is_static else ""
    
    print(f"  Group {gi}: frames {gf[0]}-{gf[-1]} ({len(group)} dets), "
          f"frame_vel={mean_vel:.1f}px/f, world_vel={world_mean_vel:.1f}px/f, "
          f"frame_x={gx.mean():.0f}, conf={gc.mean():.2f}{marker}")

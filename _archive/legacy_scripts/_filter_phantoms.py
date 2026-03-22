"""Automatic phantom ball filter using trajectory graph analysis.

Strategy:
1. Convert all YOLO detections to world-space
2. Build a DAG: edge from det_i to det_j if they could be the same ball
   (frame gap <= MAX_GAP, world-space velocity <= MAX_SPEED)
3. Find the longest weighted path through the DAG (dynamic programming)
4. Keep detections on the dominant trajectory + nearby ones
5. Reject everything else as phantoms

This is automatic and works across clips - no manual tuning needed.
The real ball follows a physically plausible trajectory; phantoms don't.
"""
import json
import numpy as np
from pathlib import Path
import shutil

REPO = Path(r"D:\Projects\soccer-video")
TEL = REPO / "out" / "telemetry"
STEM = "002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00"

# Parameters (designed to generalize across clips)
MAX_GAP = 15       # max frame gap to connect two detections
MAX_SPEED = 80.0   # max world-space px/frame for real ball
PROXIMITY = 250    # accept detections within this distance of main trajectory
MIN_PATH_LEN = 5   # minimum detections to form a valid trajectory
FPS = 30.0

# Load consensus YOLO detections
dets = {}
with open(TEL / f"{STEM}.yolo_ball.consensus.jsonl") as f:
    for line in f:
        d = json.loads(line)
        dets[d["frame"]] = (d["cx"], d["cy"], d["conf"])

# Load camera shifts for world-space conversion
cam_shifts = np.load(TEL / f"{STEM}.ball.cam_shifts.npy")
cum_dx = np.cumsum(cam_shifts[:, 0])
cum_dy = np.cumsum(cam_shifts[:, 1])

# Build detection list sorted by frame
det_list = []
for f in sorted(dets.keys()):
    cx, cy, conf = dets[f]
    wx = cx + cum_dx[f]
    wy = cy + cum_dy[f]
    det_list.append({
        "frame": f, "cx": cx, "cy": cy, "conf": conf,
        "wx": wx, "wy": wy
    })

N = len(det_list)
print(f"Input: {N} consensus detections")

# === Phase 1: Build trajectory DAG and find longest path ===
# dp[i] = (length of longest path ending at detection i, predecessor index)
dp_len = np.ones(N)  # each detection starts with length 1
dp_pred = np.full(N, -1, dtype=int)
dp_score = np.array([d["conf"] for d in det_list])  # weighted by confidence

for j in range(N):
    for i in range(j-1, -1, -1):
        frame_gap = det_list[j]["frame"] - det_list[i]["frame"]
        if frame_gap > MAX_GAP:
            break  # no more candidates (sorted by frame)
        if frame_gap <= 0:
            continue
        
        # World-space velocity check
        dx = det_list[j]["wx"] - det_list[i]["wx"]
        dy = det_list[j]["wy"] - det_list[i]["wy"]
        dist = (dx**2 + dy**2)**0.5
        vel = dist / frame_gap
        
        if vel <= MAX_SPEED:
            # Edge exists - check if this extends a better path
            new_len = dp_len[i] + 1
            new_score = dp_score[i] + det_list[j]["conf"]
            if new_score > dp_score[j]:
                dp_len[j] = new_len
                dp_pred[j] = i
                dp_score[j] = new_score

# Trace back from best endpoint
best_end = np.argmax(dp_score)
path_indices = []
idx = best_end
while idx >= 0:
    path_indices.append(idx)
    idx = dp_pred[idx]
path_indices.reverse()

path_frames = [det_list[i]["frame"] for i in path_indices]
path_wx = [det_list[i]["wx"] for i in path_indices]
path_wy = [det_list[i]["wy"] for i in path_indices]

print(f"\n=== Dominant trajectory ===")
print(f"Path length: {len(path_indices)} detections")
print(f"Frames: {path_frames[0]} - {path_frames[-1]}")
print(f"World-x range: {min(path_wx):.0f} - {max(path_wx):.0f}")

# === Phase 2: Accept nearby detections ===
# For detections NOT on the main path, check if they're close to the
# interpolated trajectory in world-space
from scipy.interpolate import PchipInterpolator

if len(path_frames) >= 2:
    interp_wx = PchipInterpolator(path_frames, path_wx, extrapolate=True)
    interp_wy = PchipInterpolator(path_frames, path_wy, extrapolate=True)

kept = set(path_indices)
rejected = []
nearby_accepted = []

for i in range(N):
    if i in kept:
        continue
    f = det_list[i]["frame"]
    # Only accept if within the frame range of the main trajectory
    if f < path_frames[0] or f > path_frames[-1]:
        # Outside trajectory range - check distance to nearest endpoint
        if f < path_frames[0]:
            ref_wx, ref_wy = path_wx[0], path_wy[0]
        else:
            ref_wx, ref_wy = path_wx[-1], path_wy[-1]
    else:
        ref_wx = float(interp_wx(f))
        ref_wy = float(interp_wy(f))
    
    dist = ((det_list[i]["wx"] - ref_wx)**2 + (det_list[i]["wy"] - ref_wy)**2)**0.5
    if dist < PROXIMITY:
        kept.add(i)
        nearby_accepted.append((i, f, dist))
    else:
        rejected.append((i, f, dist, det_list[i]["cx"], det_list[i]["cy"], det_list[i]["conf"]))

print(f"\n=== Filtering results ===")
print(f"On main path: {len(path_indices)}")
print(f"Nearby accepted: {len(nearby_accepted)}")
print(f"Rejected (phantoms): {len(rejected)}")
print(f"Total kept: {len(kept)}")

print(f"\nRejected detections (phantoms):")
for idx, f, dist, cx, cy, conf in sorted(rejected, key=lambda x: x[1]):
    print(f"  frame {f}: frame=({cx:.0f},{cy:.0f}) conf={conf:.2f} dist_from_traj={dist:.0f}px")

# === Phase 3: Flag for human review ===
# Flag clips where we rejected > 30% of detections
reject_pct = len(rejected) / N * 100
needs_review = reject_pct > 30
print(f"\nPhantom rejection rate: {reject_pct:.1f}%")
if needs_review:
    print(f"*** FLAG FOR REVIEW: High phantom rate ({reject_pct:.0f}%) - "
          f"may need manual verification ***")

# === Phase 4: Save cleaned detections ===
clean_path = TEL / f"{STEM}.yolo_ball.clean.jsonl"
with open(clean_path, "w") as f:
    for i in sorted(kept):
        d = det_list[i]
        f.write(json.dumps({
            "frame": d["frame"],
            "t": round(d["frame"] / FPS, 6),
            "cx": round(d["cx"], 2),
            "cy": round(d["cy"], 2),
            "conf": round(d["conf"], 4)
        }) + "\n")

print(f"\nSaved {len(kept)} clean detections to {clean_path.name}")

# Replace main YOLO telemetry with clean version
main_path = TEL / f"{STEM}.yolo_ball.jsonl"
shutil.copy2(clean_path, main_path)
print(f"Replaced {main_path.name} with clean detections")

# Also clean up stale caches so pipeline rebuilds
for suffix in [".tracker_ball.jsonl", ".ball.jsonl", 
               ".ball.follow.jsonl", ".ball.follow__smooth.jsonl"]:
    p = TEL / f"{STEM}{suffix}"
    if p.exists():
        bak = TEL / f"{STEM}{suffix}.pre_clean_backup"
        if not bak.exists():
            shutil.copy2(p, bak)
        p.unlink()
        print(f"Deleted stale: {p.name}")

# Show the clean trajectory summary
clean_frames = sorted(det_list[i]["frame"] for i in kept)
print(f"\nClean trajectory: {len(clean_frames)} detections, "
      f"frames {clean_frames[0]}-{clean_frames[-1]}")

# Show gaps in clean data
gaps = []
for i in range(1, len(clean_frames)):
    gap = clean_frames[i] - clean_frames[i-1]
    if gap > 5:
        gaps.append((clean_frames[i-1], clean_frames[i], gap))
gaps.sort(key=lambda x: -x[2])
print(f"Largest gaps:")
for start, end, length in gaps[:8]:
    print(f"  frames {start}-{end} ({length} frames, {length/30:.1f}s)")

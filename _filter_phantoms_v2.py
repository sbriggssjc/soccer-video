"""Phantom filter v2: Multi-segment trajectory with camera correlation.

Improvements over v1:
- Much wider gap tolerance (up to 120 frames = 4s)
- Velocity limit scales with gap length (tighter for short gaps)
- Finds multiple trajectory segments and merges them
- Uses camera pan direction as tiebreaker (real ball = camera follows it)
"""
import json
import numpy as np
from pathlib import Path
import shutil

REPO = Path(r"D:\Projects\soccer-video")
TEL = REPO / "out" / "telemetry"
STEM = "002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00"

# Parameters
SHORT_GAP = 10      # frames - tight velocity for nearby detections
LONG_GAP = 120      # frames - max gap to try bridging
SHORT_SPEED = 60.0  # px/frame for short gaps (strict)
LONG_SPEED = 25.0   # px/frame for long gaps (more lenient avg speed)
PROXIMITY = 300     # accept detections within this of trajectory
MIN_SEGMENT = 3     # min detections to form a segment
FPS = 30.0

def max_distance(gap):
    """Velocity-scaled max distance between two detections."""
    if gap <= SHORT_GAP:
        return SHORT_SPEED * gap
    else:
        # Short-range strict + long-range lenient
        return SHORT_SPEED * SHORT_GAP + LONG_SPEED * (gap - SHORT_GAP)

# Load detections
dets = {}
with open(TEL / f"{STEM}.yolo_ball.consensus.jsonl") as f:
    for line in f:
        d = json.loads(line)
        dets[d["frame"]] = (d["cx"], d["cy"], d["conf"])

# Load camera shifts
cam_shifts = np.load(TEL / f"{STEM}.ball.cam_shifts.npy")
cum_dx = np.cumsum(cam_shifts[:, 0])
cum_dy = np.cumsum(cam_shifts[:, 1])

# Build detection list
det_list = []
for f in sorted(dets.keys()):
    cx, cy, conf = dets[f]
    det_list.append({
        "frame": f, "cx": cx, "cy": cy, "conf": conf,
        "wx": cx + cum_dx[f], "wy": cy + cum_dy[f]
    })

N = len(det_list)
print(f"Input: {N} consensus detections")

# === DP: longest weighted path with gap-scaled velocity ===
dp_score = np.array([d["conf"] for d in det_list])
dp_len = np.ones(N, dtype=int)
dp_pred = np.full(N, -1, dtype=int)

for j in range(N):
    for i in range(j-1, -1, -1):
        gap = det_list[j]["frame"] - det_list[i]["frame"]
        if gap > LONG_GAP:
            break
        if gap <= 0:
            continue
        
        dx = det_list[j]["wx"] - det_list[i]["wx"]
        dy = det_list[j]["wy"] - det_list[i]["wy"]
        dist = (dx**2 + dy**2)**0.5
        
        if dist <= max_distance(gap):
            # Bonus for confidence and trajectory length
            new_score = dp_score[i] + det_list[j]["conf"] + 0.5  # +0.5 per link
            if new_score > dp_score[j]:
                dp_score[j] = new_score
                dp_len[j] = dp_len[i] + 1
                dp_pred[j] = i

# Find the best trajectory
best_end = np.argmax(dp_score)
path = []
idx = best_end
while idx >= 0:
    path.append(idx)
    idx = dp_pred[idx]
path.reverse()

print(f"\n=== Primary trajectory ===")
print(f"Length: {len(path)} detections")
print(f"Frames: {det_list[path[0]]['frame']} - {det_list[path[-1]]['frame']}")

# === Find additional consistent segments not on primary path ===
# Remove primary path, find secondary trajectories
on_primary = set(path)
remaining = [i for i in range(N) if i not in on_primary]

# Re-run DP on remaining to find secondary trajectories
secondary_paths = []
while len(remaining) >= MIN_SEGMENT:
    r_score = np.array([det_list[i]["conf"] for i in remaining])
    r_pred = np.full(len(remaining), -1, dtype=int)
    
    for jj in range(len(remaining)):
        j = remaining[jj]
        for ii in range(jj-1, -1, -1):
            i = remaining[ii]
            gap = det_list[j]["frame"] - det_list[i]["frame"]
            if gap > LONG_GAP:
                break
            if gap <= 0:
                continue
            dx = det_list[j]["wx"] - det_list[i]["wx"]
            dy = det_list[j]["wy"] - det_list[i]["wy"]
            dist = (dx**2 + dy**2)**0.5
            if dist <= max_distance(gap):
                new_score = r_score[ii] + det_list[j]["conf"] + 0.5
                if new_score > r_score[jj]:
                    r_score[jj] = new_score
                    r_pred[jj] = ii
    
    best = np.argmax(r_score)
    sec_path = []
    idx = best
    while idx >= 0:
        sec_path.append(remaining[idx])
        idx = r_pred[idx]
    sec_path.reverse()
    
    if len(sec_path) < MIN_SEGMENT:
        break
    
    secondary_paths.append(sec_path)
    sec_set = set(sec_path)
    remaining = [i for i in remaining if i not in sec_set]
    
    print(f"Secondary trajectory: {len(sec_path)} dets, "
          f"frames {det_list[sec_path[0]]['frame']}-{det_list[sec_path[-1]]['frame']}")

# === Camera correlation to identify real vs phantom trajectories ===
# The real ball is what the camera follows. Check correlation between
# trajectory world-x velocity and camera pan velocity.
def trajectory_camera_correlation(path_indices):
    """How well does camera pan correlate with ball world-x movement?"""
    if len(path_indices) < 3:
        return 0.0
    
    scores = []
    for k in range(1, len(path_indices)):
        i, j = path_indices[k-1], path_indices[k]
        fi, fj = det_list[i]["frame"], det_list[j]["frame"]
        gap = fj - fi
        if gap < 1:
            continue
        
        # Camera pan in this segment (positive = panning right)
        cam_dx = cum_dx[fj] - cum_dx[fi]
        
        # Ball frame-space movement (real ball should move AGAINST camera pan
        # or independently; phantom moves AT camera pan rate)
        ball_frame_dx = det_list[j]["cx"] - det_list[i]["cx"]
        
        # For a STATIC phantom: ball_frame_dx ≈ -cam_dx (it stays fixed in world)
        # For a REAL ball: ball_frame_dx is independent of cam_dx
        # Score: how much does the ball move in world-space (= frame movement + camera pan)?
        ball_world_dx = ball_frame_dx + cam_dx  # should be non-trivial for real ball
        scores.append(abs(ball_world_dx) / max(gap, 1))
    
    return np.mean(scores) if scores else 0.0

# Score all trajectories
all_trajectories = [("primary", path)] + [(f"secondary_{i}", sp) for i, sp in enumerate(secondary_paths)]

print(f"\n=== Camera correlation scores ===")
for name, traj in all_trajectories:
    corr = trajectory_camera_correlation(traj)
    f0 = det_list[traj[0]]["frame"]
    f1 = det_list[traj[-1]]["frame"]
    mean_conf = np.mean([det_list[i]["conf"] for i in traj])
    print(f"  {name}: {len(traj)} dets, frames {f0}-{f1}, "
          f"world_motion={corr:.1f}px/f, avg_conf={mean_conf:.2f}")

# === Merge all trajectories (they're all potentially real ball segments) ===
# Instead of picking just one, keep all trajectory segments that show
# real motion (world_motion > threshold)
MIN_WORLD_MOTION = 3.0  # px/frame - below this, likely static phantom

kept_indices = set()
for name, traj in all_trajectories:
    corr = trajectory_camera_correlation(traj)
    if corr >= MIN_WORLD_MOTION or len(traj) >= 8:
        kept_indices.update(traj)
        print(f"  KEEPING {name} ({len(traj)} dets, motion={corr:.1f})")
    else:
        print(f"  REJECTING {name} ({len(traj)} dets, motion={corr:.1f}) - too static")

# Also add isolated high-confidence detections
for i in range(N):
    if i not in kept_indices and det_list[i]["conf"] >= 0.55:
        kept_indices.add(i)

print(f"\n=== Final results ===")
print(f"Kept: {len(kept_indices)} detections")
print(f"Rejected: {N - len(kept_indices)} phantoms")

# Show kept vs rejected
kept_frames = sorted([det_list[i]["frame"] for i in kept_indices])
rejected_frames = sorted([det_list[i]["frame"] for i in range(N) if i not in kept_indices])
print(f"Kept frames: {kept_frames}")
print(f"Rejected frames: {rejected_frames}")

# Save cleaned detections
clean_path = TEL / f"{STEM}.yolo_ball.clean.jsonl"
with open(clean_path, "w") as f:
    for i in sorted(kept_indices):
        d = det_list[i]
        f.write(json.dumps({
            "frame": d["frame"],
            "t": round(d["frame"] / FPS, 6),
            "cx": round(d["cx"], 2),
            "cy": round(d["cy"], 2),
            "conf": round(d["conf"], 4)
        }) + "\n")
print(f"Saved to {clean_path.name}")

# Replace main YOLO file
main_path = TEL / f"{STEM}.yolo_ball.jsonl"
shutil.copy2(clean_path, main_path)
print(f"Replaced {main_path.name}")

# Clean stale caches
for suffix in [".tracker_ball.jsonl", ".ball.jsonl",
               ".ball.follow.jsonl", ".ball.follow__smooth.jsonl"]:
    p = TEL / f"{STEM}{suffix}"
    if p.exists():
        p.unlink()
        print(f"Deleted stale: {p.name}")

# Gaps
print(f"\nClean detection gaps:")
gaps = []
for i in range(1, len(kept_frames)):
    gap = kept_frames[i] - kept_frames[i-1]
    if gap > 5:
        gaps.append((kept_frames[i-1], kept_frames[i], gap))
gaps.sort(key=lambda x: -x[2])
for s, e, g in gaps[:8]:
    print(f"  frames {s}-{e} ({g} frames, {g/30:.1f}s)")

# Flag for review
reject_pct = (N - len(kept_indices)) / N * 100
if reject_pct > 30:
    print(f"\n*** FLAG FOR REVIEW: {reject_pct:.0f}% phantom rate ***")

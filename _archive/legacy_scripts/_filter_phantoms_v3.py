"""
Phantom ball filter v3 - Oscillation-aware approach.

Strategy:
1. Load consensus YOLO detections (86 detections)
2. For each detection, compute consistency with temporal neighbors
   - Look at K nearest detections in time (before and after)
   - If this detection is far from the median trajectory of neighbors, flag it
3. Iteratively remove worst outliers and re-score
4. Also detect "alternating" patterns where detections ping-pong between two positions

This handles the key failure modes:
- v1 was too aggressive (short gap tolerance broke trajectories)
- v2 was too lenient (long gap tolerance connected everything)
- v3 uses local consistency: a real ball has neighbors that agree with it
"""

import json, sys, os, shutil
import numpy as np

# === Config ===
CONSENSUS_FILE = r"D:\Projects\soccer-video\out\telemetry\002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.yolo_ball.consensus.jsonl"
OUTPUT_FILE    = r"D:\Projects\soccer-video\out\telemetry\002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.yolo_ball.jsonl"
CLEAN_FILE     = r"D:\Projects\soccer-video\out\telemetry\002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.yolo_ball.clean_v3.jsonl"
CAM_SHIFTS_FILE= r"D:\Projects\soccer-video\out\telemetry\002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.ball.cam_shifts.npy"

# How many temporal neighbors to consider on each side
K_NEIGHBORS = 4
# Max frames apart to consider as neighbor
MAX_NEIGHBOR_GAP = 40
# Outlier threshold: if detection is > this many px from neighbor median, flag it
OUTLIER_THRESHOLD_PX = 200
# For alternating detection: if 3 consecutive detections go A-B-A with |A-B|>thresh
ALTERNATING_THRESHOLD_PX = 300
# Minimum trajectory length to keep (detections)
MIN_TRAJ_LENGTH = 5

# === Load data ===
print("Loading consensus detections...")
dets = []
with open(CONSENSUS_FILE) as f:
    for line in f:
        d = json.loads(line)
        dets.append(d)
dets.sort(key=lambda d: d["frame"])
print(f"  Loaded {len(dets)} detections, frames {dets[0]['frame']}-{dets[-1]['frame']}")

# Load camera shifts for world-space analysis
cam_shifts = np.load(CAM_SHIFTS_FILE)  # (496, 2)
cum_cam = np.cumsum(cam_shifts, axis=0)
print(f"  Camera shifts: {cam_shifts.shape}")

# === Convert to world space ===
for d in dets:
    f = d["frame"]
    if f < len(cum_cam):
        d["wx"] = d["cx"] + cum_cam[f, 0]
        d["wy"] = d["cy"] + cum_cam[f, 1]
    else:
        d["wx"] = d["cx"]
        d["wy"] = d["cy"]

# === Step 1: Flag alternating detections ===
print("\n=== Alternating detection analysis ===")
alt_flags = set()
for i in range(1, len(dets) - 1):
    prev, curr, nxt = dets[i-1], dets[i], dets[i+1]
    if curr["frame"] - prev["frame"] > MAX_NEIGHBOR_GAP:
        continue
    if nxt["frame"] - curr["frame"] > MAX_NEIGHBOR_GAP:
        continue
    dist_prev_next = abs(prev["cx"] - nxt["cx"])
    dist_curr_prev = abs(curr["cx"] - prev["cx"])
    dist_curr_next = abs(curr["cx"] - nxt["cx"])
    if (dist_curr_prev > ALTERNATING_THRESHOLD_PX and
        dist_curr_next > ALTERNATING_THRESHOLD_PX and
        dist_prev_next < ALTERNATING_THRESHOLD_PX):
        alt_flags.add(curr["frame"])
        print(f"  ALT f{curr['frame']}: cx={curr['cx']:.0f}, prev(f{prev['frame']})={prev['cx']:.0f}, next(f{nxt['frame']})={nxt['cx']:.0f}")

print(f"  Flagged {len(alt_flags)} alternating detections")

# === Step 2: Neighbor median consistency ===
print("\n=== Neighbor median consistency ===")
frames = [d["frame"] for d in dets]
cxs = [d["cx"] for d in dets]

outlier_scores = {}
for i, d in enumerate(dets):
    neighbors_cx = []
    for j in range(max(0, i - K_NEIGHBORS), min(len(dets), i + K_NEIGHBORS + 1)):
        if j == i:
            continue
        if abs(dets[j]["frame"] - d["frame"]) <= MAX_NEIGHBOR_GAP:
            neighbors_cx.append(dets[j]["cx"])

    if len(neighbors_cx) < 2:
        outlier_scores[d["frame"]] = 0
        continue

    median_cx = np.median(neighbors_cx)
    deviation = abs(d["cx"] - median_cx)
    outlier_scores[d["frame"]] = deviation

    if deviation > OUTLIER_THRESHOLD_PX:
        print(f"  OUTLIER f{d['frame']}: cx={d['cx']:.0f}, neighbor_median={median_cx:.0f}, dev={deviation:.0f}px")

# === Step 3: Static object detection ===
print("\n=== Static object detection (frame-space stability) ===")
static_flags = set()
for i in range(len(dets)):
    for j in range(i+1, len(dets)):
        frame_gap = dets[j]["frame"] - dets[i]["frame"]
        if frame_gap < 50:
            continue
        if frame_gap > 400:
            break
        fs_dist = ((dets[i]["cx"] - dets[j]["cx"])**2 + (dets[i]["cy"] - dets[j]["cy"])**2)**0.5
        if fs_dist < 60:
            print(f"  STATIC pair: f{dets[i]['frame']}({dets[i]['cx']:.0f},{dets[i]['cy']:.0f}) <-> f{dets[j]['frame']}({dets[j]['cx']:.0f},{dets[j]['cy']:.0f}), gap={frame_gap}f, dist={fs_dist:.0f}px")

# === Step 4: Build clean trajectory ===
print("\n=== Building clean trajectory ===")
reject_frames = set()
reject_frames.update(alt_flags)
for frame, score in outlier_scores.items():
    if score > OUTLIER_THRESHOLD_PX:
        reject_frames.add(frame)

# === Step 5: Iterative refinement ===
print("\n=== Iterative refinement ===")
for iteration in range(3):
    remaining = [d for d in dets if d["frame"] not in reject_frames]
    new_rejects = set()

    for i, d in enumerate(remaining):
        neighbors_cx = []
        for j in range(max(0, i - K_NEIGHBORS), min(len(remaining), i + K_NEIGHBORS + 1)):
            if j == i:
                continue
            if abs(remaining[j]["frame"] - d["frame"]) <= MAX_NEIGHBOR_GAP:
                neighbors_cx.append(remaining[j]["cx"])

        if len(neighbors_cx) < 2:
            continue

        median_cx = np.median(neighbors_cx)
        deviation = abs(d["cx"] - median_cx)
        thresh = OUTLIER_THRESHOLD_PX * (0.8 ** iteration)
        if deviation > thresh:
            new_rejects.add(d["frame"])

    if not new_rejects:
        print(f"  Iteration {iteration+1}: no new rejects, done")
        break

    print(f"  Iteration {iteration+1}: {len(new_rejects)} new rejects (thresh={thresh:.0f}px)")
    reject_frames.update(new_rejects)

# === Step 6: Trajectory continuity check ===
remaining = [d for d in dets if d["frame"] not in reject_frames]
print(f"\n=== Final trajectory ===")
print(f"  Kept: {len(remaining)}/{len(dets)} detections")
print(f"  Rejected: {len(reject_frames)} detections")
print(f"  Rejected frames: {sorted(reject_frames)}")

if remaining:
    print(f"  Frame range: {remaining[0]['frame']}-{remaining[-1]['frame']}")
    gaps = []
    for i in range(1, len(remaining)):
        gap = remaining[i]["frame"] - remaining[i-1]["frame"]
        if gap > 20:
            gaps.append((remaining[i-1]["frame"], remaining[i]["frame"], gap))
    if gaps:
        print(f"  Large gaps (>20f):")
        for a, b, g in gaps:
            print(f"    f{a} -> f{b}: {g} frames")

# === Step 7: Velocity profile of clean trajectory ===
print(f"\n=== Clean trajectory velocity profile ===")
for i in range(1, len(remaining)):
    d0, d1 = remaining[i-1], remaining[i]
    dt = d1["frame"] - d0["frame"]
    if dt > 0:
        vel = (d1["cx"] - d0["cx"]) / dt
        if abs(vel) > 60:
            print(f"  HIGH VEL f{d0['frame']}->f{d1['frame']}: {vel:.1f}px/f (dt={dt})")

# === Save results ===
print(f"\n=== Saving ===")
with open(CLEAN_FILE, 'w') as f:
    for d in remaining:
        out = {k: v for k, v in d.items() if k not in ("wx", "wy")}
        f.write(json.dumps(out) + "\n")
print(f"  Clean detections -> {CLEAN_FILE}")

backup = OUTPUT_FILE + ".pre_v3_backup"
if os.path.exists(OUTPUT_FILE):
    shutil.copy2(OUTPUT_FILE, backup)
    print(f"  Backup -> {backup}")

shutil.copy2(CLEAN_FILE, OUTPUT_FILE)
print(f"  Main file updated -> {OUTPUT_FILE}")

# Delete stale caches
telemetry_dir = os.path.dirname(OUTPUT_FILE)
stem = "002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00"
stale_suffixes = [".tracker_ball.jsonl", ".ball.jsonl", ".ball.follow.jsonl", ".ball.follow__smooth.jsonl"]
for suffix in stale_suffixes:
    cache_file = os.path.join(telemetry_dir, stem + suffix)
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print(f"  Deleted stale cache: {os.path.basename(cache_file)}")

print("\nDone! Run pipeline to regenerate with clean detections.")

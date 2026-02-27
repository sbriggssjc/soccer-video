"""
Polish v12 - Two-phase approach:
1. RECOVER: Add back xlarge detections that are consistent with the clean trajectory
2. SCRUB: Remove any remaining high-velocity outliers

Goal: increase YOLO coverage from 47 detections while keeping phantom-free.
"""

import json, os, shutil
import numpy as np

# === Paths ===
TELEMETRY = r"D:\Projects\soccer-video\out\telemetry"
STEM = "002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00"

CLEAN_V3    = os.path.join(TELEMETRY, STEM + ".yolo_ball.clean_v3.jsonl")
XLARGE      = os.path.join(TELEMETRY, STEM + ".yolo_ball.yolov8x.jsonl")
MEDIUM      = os.path.join(TELEMETRY, STEM + ".yolo_ball.yolov8m.jsonl")
CAM_SHIFTS  = os.path.join(TELEMETRY, STEM + ".ball.cam_shifts.npy")
OUTPUT      = os.path.join(TELEMETRY, STEM + ".yolo_ball.jsonl")
POLISHED    = os.path.join(TELEMETRY, STEM + ".yolo_ball.polished_v12.jsonl")
DIAG_V11    = r"D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_NEOFC\002__cleaned_v11.diag.csv"

# === Config ===
RECOVERY_MAX_DIST = 150       # max px from interpolated trajectory to recover a detection
RECOVERY_MIN_CONF = 0.30      # minimum confidence for recovered detections
MAX_VELOCITY = 80             # px/frame - reject anything faster
MAX_FIELD_Y = 650             # reject detections below this (bottom of frame junk)
TOTAL_FRAMES = 496

def load_jsonl(path):
    dets = []
    with open(path) as f:
        for line in f:
            dets.append(json.loads(line))
    dets.sort(key=lambda d: d["frame"])
    return dets

def save_jsonl(dets, path):
    with open(path, 'w') as f:
        for d in dets:
            f.write(json.dumps(d) + "\n")

# === Load everything ===
print("=== Loading data ===")
clean = load_jsonl(CLEAN_V3)
xlarge = load_jsonl(XLARGE)
medium = load_jsonl(MEDIUM)
cam_shifts = np.load(CAM_SHIFTS)
cum_cam = np.cumsum(cam_shifts, axis=0)

clean_frames = {d["frame"] for d in clean}
print(f"  Clean v3: {len(clean)} detections, frames {min(clean_frames)}-{max(clean_frames)}")
print(f"  Xlarge: {len(xlarge)} detections")
print(f"  Medium: {len(medium)} detections")

# === Phase 0: Analyze v11 diag to find problem areas ===
print("\n=== Analyzing v11 diagnostics ===")
import csv
diag_rows = []
with open(DIAG_V11) as f:
    reader = csv.DictReader(f)
    for row in reader:
        diag_rows.append(row)

# Find the biggest camera jumps in v11
cam_cxs = []
for r in diag_rows:
    try:
        cam_cxs.append(float(r["cam_cx"]))
    except:
        cam_cxs.append(0)

print("  Biggest remaining camera jumps in v11:")
jumps = []
for i in range(1, len(cam_cxs)):
    delta = abs(cam_cxs[i] - cam_cxs[i-1])
    if delta > 20:
        jumps.append((i, delta, cam_cxs[i-1], cam_cxs[i]))
jumps.sort(key=lambda x: -x[1])
for f, d, a, b in jumps[:15]:
    src = diag_rows[f].get("source", "?")
    print(f"    f{f}: {d:.0f}px ({a:.0f}->{b:.0f}), src={src}")

# Find frames with no YOLO coverage (gaps)
print("\n  Coverage gaps (no YOLO within 10 frames):")
for f in range(TOTAL_FRAMES):
    nearest = min((abs(f - cf) for cf in clean_frames), default=999)
    if nearest > 30 and f % 30 == 0:
        src = diag_rows[f].get("source", "?") if f < len(diag_rows) else "?"
        print(f"    f{f}: nearest YOLO is {nearest} frames away, using src={src}")

# === Phase 1: Build interpolated trajectory from clean detections ===
print("\n=== Building interpolated trajectory ===")
# Linear interpolation between clean detections
interp_cx = np.full(TOTAL_FRAMES, np.nan)
interp_cy = np.full(TOTAL_FRAMES, np.nan)

# Set known points
for d in clean:
    f = d["frame"]
    if f < TOTAL_FRAMES:
        interp_cx[f] = d["cx"]
        interp_cy[f] = d["cy"]

# Interpolate between known points
clean_sorted = sorted(clean, key=lambda d: d["frame"])
for i in range(len(clean_sorted) - 1):
    f0, f1 = clean_sorted[i]["frame"], clean_sorted[i+1]["frame"]
    cx0, cx1 = clean_sorted[i]["cx"], clean_sorted[i+1]["cx"]
    cy0, cy1 = clean_sorted[i]["cy"], clean_sorted[i+1]["cy"]
    for f in range(f0 + 1, f1):
        if f < TOTAL_FRAMES:
            t = (f - f0) / (f1 - f0)
            interp_cx[f] = cx0 + t * (cx1 - cx0)
            interp_cy[f] = cy0 + t * (cy1 - cy0)

# Extrapolate before first and after last
first_f = clean_sorted[0]["frame"]
last_f = clean_sorted[-1]["frame"]
for f in range(0, first_f):
    interp_cx[f] = clean_sorted[0]["cx"]
    interp_cy[f] = clean_sorted[0]["cy"]
for f in range(last_f + 1, TOTAL_FRAMES):
    interp_cx[f] = clean_sorted[-1]["cx"]
    interp_cy[f] = clean_sorted[-1]["cy"]

valid = np.sum(~np.isnan(interp_cx))
print(f"  Interpolated trajectory covers {valid}/{TOTAL_FRAMES} frames")

# === Phase 2: Recover consistent xlarge/medium detections ===
print("\n=== Recovering consistent detections ===")
recovered = []
all_candidates = []

# Combine xlarge and medium, preferring xlarge
for d in xlarge:
    d["_model"] = "xlarge"
    all_candidates.append(d)
for d in medium:
    d["_model"] = "medium"
    all_candidates.append(d)

# Group by frame, prefer higher conf
by_frame = {}
for d in all_candidates:
    f = d["frame"]
    if f not in by_frame or d.get("conf", 0) > by_frame[f].get("conf", 0):
        by_frame[f] = d

for frame, d in sorted(by_frame.items()):
    if frame in clean_frames:
        continue  # Already in clean set
    if frame >= TOTAL_FRAMES:
        continue
    if np.isnan(interp_cx[frame]):
        continue  # No trajectory reference
    if d.get("cy", 0) > MAX_FIELD_Y:
        continue  # Bottom of frame junk
    if d.get("conf", 0) < RECOVERY_MIN_CONF:
        continue

    # Distance from interpolated trajectory
    dist = ((d["cx"] - interp_cx[frame])**2 + (d["cy"] - interp_cy[frame])**2)**0.5

    if dist <= RECOVERY_MAX_DIST:
        recovered.append(d)
        if dist > 80:
            print(f"  RECOVERED f{frame}: cx={d['cx']:.0f}, interp={interp_cx[frame]:.0f}, dist={dist:.0f}px, conf={d.get('conf',0):.2f}, model={d['_model']}")

print(f"  Recovered {len(recovered)} detections from xlarge/medium models")

# === Phase 3: Merge clean + recovered ===
merged = list(clean)
for d in recovered:
    # Remove internal fields
    out = {k: v for k, v in d.items() if not k.startswith("_")}
    merged.append(out)
merged.sort(key=lambda d: d["frame"])

# Deduplicate by frame (keep higher confidence)
deduped = {}
for d in merged:
    f = d["frame"]
    if f not in deduped or d.get("conf", 0) > deduped[f].get("conf", 0):
        deduped[f] = d
merged = sorted(deduped.values(), key=lambda d: d["frame"])

print(f"\n  Merged: {len(merged)} detections (was {len(clean)} clean + {len(recovered)} recovered)")

# === Phase 4: Scrub high-velocity outliers ===
print("\n=== Scrubbing high-velocity outliers ===")
scrubbed = set()
for iteration in range(3):
    active = [d for d in merged if d["frame"] not in scrubbed]
    new_scrubs = set()

    for i in range(1, len(active)):
        d0, d1 = active[i-1], active[i]
        dt = d1["frame"] - d0["frame"]
        if dt == 0:
            continue
        vel = abs(d1["cx"] - d0["cx"]) / dt

        if vel > MAX_VELOCITY and dt <= 5:
            # Which one is the outlier? Check against neighbors
            # Look at the detection before d0 and after d1
            if i >= 2:
                d_prev = active[i-2]
                vel_prev = abs(d0["cx"] - d_prev["cx"]) / max(1, d0["frame"] - d_prev["frame"])
            else:
                vel_prev = 0

            if i < len(active) - 1:
                d_next = active[i+1]
                vel_next = abs(d1["cx"] - d_next["cx"]) / max(1, d_next["frame"] - d1["frame"])
            else:
                vel_next = 0

            # The outlier is the one with high velocity on BOTH sides
            if vel_prev > MAX_VELOCITY * 0.5:
                new_scrubs.add(d0["frame"])
                print(f"  iter{iteration+1} SCRUB f{d0['frame']}: vel_in={vel_prev:.0f}, vel_out={vel:.0f}px/f")
            elif vel_next > MAX_VELOCITY * 0.5:
                new_scrubs.add(d1["frame"])
                print(f"  iter{iteration+1} SCRUB f{d1['frame']}: vel_in={vel:.0f}, vel_out={vel_next:.0f}px/f")

    if not new_scrubs:
        print(f"  Iteration {iteration+1}: no new scrubs, done")
        break
    scrubbed.update(new_scrubs)

final = [d for d in merged if d["frame"] not in scrubbed]
print(f"\n  Scrubbed {len(scrubbed)} outliers: {sorted(scrubbed)}")
print(f"  Final: {len(final)} detections")

# === Phase 5: Final trajectory analysis ===
print("\n=== Final trajectory ===")
final_frames = {d["frame"] for d in final}
print(f"  Detections: {len(final)}")
print(f"  Frame range: {min(final_frames)}-{max(final_frames)}")
print(f"  Coverage: {len(final)/TOTAL_FRAMES*100:.1f}%")

# Gaps
gaps = []
fs = sorted(final_frames)
for i in range(1, len(fs)):
    gap = fs[i] - fs[i-1]
    if gap > 15:
        gaps.append((fs[i-1], fs[i], gap))
if gaps:
    print(f"  Gaps >15f:")
    for a, b, g in gaps:
        print(f"    f{a} -> f{b}: {g} frames")

# Velocity profile
print(f"\n  Velocity check:")
for i in range(1, len(final)):
    d0, d1 = final[i-1], final[i]
    dt = d1["frame"] - d0["frame"]
    if dt > 0:
        vel = abs(d1["cx"] - d0["cx"]) / dt
        if vel > 50:
            print(f"    f{d0['frame']}->f{d1['frame']}: {vel:.1f}px/f (dt={dt})")

# === Save ===
print(f"\n=== Saving ===")
save_jsonl(final, POLISHED)
print(f"  Polished -> {POLISHED}")

backup = OUTPUT + ".pre_v12_backup"
if os.path.exists(OUTPUT):
    shutil.copy2(OUTPUT, backup)
shutil.copy2(POLISHED, OUTPUT)
print(f"  Main file updated -> {OUTPUT}")

# Delete stale caches
stale = [".tracker_ball.jsonl", ".ball.jsonl", ".ball.follow.jsonl", ".ball.follow__smooth.jsonl"]
for suffix in stale:
    p = os.path.join(TELEMETRY, STEM + suffix)
    if os.path.exists(p):
        os.remove(p)
        print(f"  Deleted stale: {os.path.basename(p)}")

print(f"\nDone! {len(clean)} clean -> {len(final)} polished detections.")
print("Run pipeline to regenerate v12.")

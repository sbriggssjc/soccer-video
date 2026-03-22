"""
Appearance-based ball detection filter.

Strategy:
1. Extract ball crops from highest-confidence YOLO detections (reference set)
2. Build a color histogram signature of the real ball
3. Score ALL candidate detections from all 3 models against the signature
4. Keep matches, reject false positives (cones, cleats, markings)
5. Generate a review strip image for manual verification
"""

import json, csv, os, sys, shutil, math
import numpy as np

try:
    import cv2
except ImportError:
    print("ERROR: cv2 required"); sys.exit(1)

# === Paths ===
CLIP = r"D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.mp4"
TELEMETRY = r"D:\Projects\soccer-video\out\telemetry"
STEM = "002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00"
OUTPUT_DIR = r"D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_NEOFC"

# Detection files
NANO   = os.path.join(TELEMETRY, STEM + ".yolo_ball.nano_original.jsonl")
MEDIUM = os.path.join(TELEMETRY, STEM + ".yolo_ball.yolov8m.jsonl")
XLARGE = os.path.join(TELEMETRY, STEM + ".yolo_ball.yolov8x.jsonl")
MERGED_V13 = os.path.join(TELEMETRY, STEM + ".yolo_ball.merged_v13.jsonl")
MAIN_OUTPUT = os.path.join(TELEMETRY, STEM + ".yolo_ball.jsonl")
APPEARANCE_OUTPUT = os.path.join(TELEMETRY, STEM + ".yolo_ball.appearance_v14.jsonl")

# Review strip output
REVIEW_STRIP = os.path.join(OUTPUT_DIR, "002__ball_review_strip.png")

# === Config ===
CROP_RADIUS = 40        # px around detection center to crop for analysis
REF_TOP_N = 10          # Number of top-confidence detections to use as reference
HIST_BINS = 16          # Color histogram bins per channel
HIST_THRESH = 0.35      # Minimum histogram correlation to keep a detection
MAX_FIELD_Y = 650       # Reject detections below this y
TOTAL_FRAMES = 496

def load_jsonl(path):
    if not os.path.exists(path):
        return []
    dets = []
    with open(path) as f:
        for line in f:
            dets.append(json.loads(line))
    return sorted(dets, key=lambda d: d["frame"])

def save_jsonl(dets, path):
    with open(path, 'w') as f:
        for d in dets:
            f.write(json.dumps(d) + "\n")

def extract_crop(cap, frame_idx, cx, cy, radius=CROP_RADIUS):
    """Extract a square crop around (cx, cy) from the given frame."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        return None
    h, w = frame.shape[:2]
    x0 = max(0, int(cx - radius))
    y0 = max(0, int(cy - radius))
    x1 = min(w, int(cx + radius))
    y1 = min(h, int(cy + radius))
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return crop

def compute_color_hist(crop):
    """Compute normalized HSV color histogram for a crop."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # Use H and S channels (ignore V for lighting robustness)
    hist = cv2.calcHist([hsv], [0, 1], None, [HIST_BINS, HIST_BINS],
                        [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist

def hist_similarity(h1, h2):
    """Compute histogram correlation (-1 to 1, higher is better)."""
    return cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)

# === Load all candidate detections ===
print("=== Loading all candidate detections ===")
nano_dets = load_jsonl(NANO)
medium_dets = load_jsonl(MEDIUM)
xlarge_dets = load_jsonl(XLARGE)
v13_dets = load_jsonl(MERGED_V13)

print(f"  Nano: {len(nano_dets)}")
print(f"  Medium: {len(medium_dets)}")
print(f"  Xlarge: {len(xlarge_dets)}")
print(f"  V13 merged: {len(v13_dets)}")

# Combine all candidates, tag with source model
all_candidates = []
for d in nano_dets:
    d["_model"] = "nano"
    all_candidates.append(d)
for d in medium_dets:
    d["_model"] = "medium"
    all_candidates.append(d)
for d in xlarge_dets:
    d["_model"] = "xlarge"
    all_candidates.append(d)

# Deduplicate by frame: keep highest confidence per frame
by_frame = {}
for d in all_candidates:
    f = d["frame"]
    if f not in by_frame or d.get("conf", 0) > by_frame[f].get("conf", 0):
        by_frame[f] = d

# Also add any v13 detections for frames not in the 3-model set
for d in v13_dets:
    f = d["frame"]
    if f not in by_frame or d.get("conf", 0) > by_frame[f].get("conf", 0):
        d["_model"] = "v13"
        by_frame[f] = d

candidates = sorted(by_frame.values(), key=lambda d: d["frame"])
print(f"  Total unique frame candidates: {len(candidates)}")

# === Step 1: Build reference ball signature ===
print(f"\n=== Building reference ball signature ===")
# Use top-N highest confidence detections from v13 (already validated)
v13_by_conf = sorted(v13_dets, key=lambda d: -d.get("conf", 0))
ref_dets = v13_by_conf[:REF_TOP_N]
print(f"  Using top {len(ref_dets)} v13 detections as reference:")
for d in ref_dets:
    print(f"    f{d['frame']}: cx={d['cx']:.0f}, cy={d['cy']:.0f}, conf={d.get('conf',0):.3f}")

cap = cv2.VideoCapture(CLIP)
if not cap.isOpened():
    print(f"ERROR: Cannot open {CLIP}")
    sys.exit(1)

ref_hists = []
ref_crops = []
for d in ref_dets:
    crop = extract_crop(cap, d["frame"], d["cx"], d["cy"])
    if crop is not None and crop.shape[0] > 5 and crop.shape[1] > 5:
        hist = compute_color_hist(crop)
        ref_hists.append(hist)
        ref_crops.append((d["frame"], crop))
        print(f"    f{d['frame']}: crop {crop.shape}, hist computed")

print(f"  Reference histograms: {len(ref_hists)}")

if not ref_hists:
    print("ERROR: No reference crops extracted!")
    cap.release()
    sys.exit(1)

# Build average reference histogram
ref_avg = ref_hists[0].copy()
for h in ref_hists[1:]:
    ref_avg += h
ref_avg /= len(ref_hists)
cv2.normalize(ref_avg, ref_avg, 0, 1, cv2.NORM_MINMAX)

# === Step 2: Score ALL candidates against reference ===
print(f"\n=== Scoring {len(candidates)} candidates against ball signature ===")
scored = []
for d in candidates:
    if d.get("cy", 0) > MAX_FIELD_Y:
        continue

    crop = extract_crop(cap, d["frame"], d["cx"], d["cy"])
    if crop is None or crop.shape[0] < 5 or crop.shape[1] < 5:
        continue

    hist = compute_color_hist(crop)

    # Score against average reference
    sim_avg = hist_similarity(hist, ref_avg)

    # Also score against individual references (take max)
    sim_max = max(hist_similarity(hist, rh) for rh in ref_hists)

    # Combined score: weight average and max
    score = 0.5 * sim_avg + 0.5 * sim_max

    d["_sim_avg"] = round(sim_avg, 4)
    d["_sim_max"] = round(sim_max, 4)
    d["_score"] = round(score, 4)
    d["_crop"] = crop  # Keep for review strip
    scored.append(d)

print(f"  Scored: {len(scored)} candidates")

# Analyze score distribution
scores = [d["_score"] for d in scored]
print(f"  Score range: {min(scores):.3f} - {max(scores):.3f}")
print(f"  Score mean: {np.mean(scores):.3f}, median: {np.median(scores):.3f}")

# Show score histogram
bins = [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
hist_counts, _ = np.histogram(scores, bins=bins)
print(f"  Score distribution:")
for i in range(len(bins)-1):
    bar = "#" * hist_counts[i]
    print(f"    {bins[i]:.1f}-{bins[i+1]:.1f}: {hist_counts[i]:3d} {bar}")

# === Step 3: Filter by appearance score ===
print(f"\n=== Filtering by appearance score (threshold={HIST_THRESH}) ===")
accepted = [d for d in scored if d["_score"] >= HIST_THRESH]
rejected = [d for d in scored if d["_score"] < HIST_THRESH]
print(f"  Accepted: {len(accepted)}")
print(f"  Rejected: {len(rejected)}")

# Show some rejected detections for context
if rejected:
    print(f"\n  Sample rejected detections:")
    for d in sorted(rejected, key=lambda d: d["_score"])[:10]:
        print(f"    f{d['frame']}: cx={d['cx']:.0f}, cy={d['cy']:.0f}, conf={d.get('conf',0):.2f}, score={d['_score']:.3f}, model={d.get('_model','?')}")

# === Step 4: Generate review strip ===
print(f"\n=== Generating review strip ===")
# Sort accepted and rejected by frame for visual review
review_items = []
for d in accepted:
    review_items.append(("KEEP", d))
for d in rejected:
    review_items.append(("REJECT", d))
review_items.sort(key=lambda x: x[1]["frame"])

# Build image strip: rows of thumbnails with labels
THUMB_SIZE = 80
COLS = 16
n_items = min(len(review_items), COLS * 20)  # Max 20 rows
ROWS = math.ceil(n_items / COLS)

strip_w = COLS * (THUMB_SIZE + 4) + 4
strip_h = ROWS * (THUMB_SIZE + 24) + 4
strip = np.ones((strip_h, strip_w, 3), dtype=np.uint8) * 40  # dark gray background

for idx, (label, d) in enumerate(review_items[:n_items]):
    row = idx // COLS
    col = idx % COLS
    x0 = col * (THUMB_SIZE + 4) + 4
    y0 = row * (THUMB_SIZE + 24) + 4

    crop = d.get("_crop")
    if crop is not None:
        thumb = cv2.resize(crop, (THUMB_SIZE, THUMB_SIZE))
    else:
        thumb = np.zeros((THUMB_SIZE, THUMB_SIZE, 3), dtype=np.uint8)

    # Border color: green for KEEP, red for REJECT
    color = (0, 200, 0) if label == "KEEP" else (0, 0, 200)
    cv2.rectangle(thumb, (0, 0), (THUMB_SIZE-1, THUMB_SIZE-1), color, 2)

    strip[y0:y0+THUMB_SIZE, x0:x0+THUMB_SIZE] = thumb

    # Label below thumbnail
    text = f"f{d['frame']} {d['_score']:.2f}"
    cv2.putText(strip, text, (x0, y0+THUMB_SIZE+14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

cv2.imwrite(REVIEW_STRIP, strip)
print(f"  Review strip saved -> {REVIEW_STRIP}")
print(f"  ({n_items} thumbnails: green=KEEP, red=REJECT)")

cap.release()

# === Step 5: Build final detection set ===
print(f"\n=== Building final detection set ===")
# Remove internal fields
final = []
for d in accepted:
    clean = {k: v for k, v in d.items() if not k.startswith("_")}
    final.append(clean)
final.sort(key=lambda d: d["frame"])

# Deduplicate by frame
deduped = {}
for d in final:
    f = d["frame"]
    if f not in deduped or d.get("conf", 0) > deduped[f].get("conf", 0):
        deduped[f] = d
final = sorted(deduped.values(), key=lambda d: d["frame"])

print(f"  Final: {len(final)} detections ({100*len(final)/TOTAL_FRAMES:.1f}% coverage)")

# Gap analysis
frames = [d["frame"] for d in final]
gaps = []
for i in range(1, len(frames)):
    gap = frames[i] - frames[i-1]
    if gap > 10:
        gaps.append((frames[i-1], frames[i], gap))
if gaps:
    print(f"  Gaps >10f:")
    for a, b, g in gaps:
        print(f"    f{a} -> f{b}: {g} frames")

# Velocity check
print(f"\n  Velocity check:")
high_vel = 0
for i in range(1, len(final)):
    d0, d1 = final[i-1], final[i]
    dt = d1["frame"] - d0["frame"]
    if dt > 0 and dt <= 5:
        vel = abs(d1["cx"] - d0["cx"]) / dt
        if vel > 80:
            high_vel += 1
            print(f"    f{d0['frame']}->f{d1['frame']}: {vel:.0f}px/f")
if high_vel == 0:
    print(f"    No high-velocity pairs!")

# === Save ===
print(f"\n=== Saving ===")
save_jsonl(final, APPEARANCE_OUTPUT)
print(f"  Appearance-filtered -> {APPEARANCE_OUTPUT}")

backup = MAIN_OUTPUT + ".pre_v14_backup"
if os.path.exists(MAIN_OUTPUT):
    shutil.copy2(MAIN_OUTPUT, backup)
shutil.copy2(APPEARANCE_OUTPUT, MAIN_OUTPUT)
print(f"  Main file updated -> {MAIN_OUTPUT}")

# Delete stale caches
stale = [".tracker_ball.jsonl", ".ball.jsonl", ".ball.follow.jsonl", ".ball.follow__smooth.jsonl"]
for suffix in stale:
    p = os.path.join(TELEMETRY, STEM + suffix)
    if os.path.exists(p):
        os.remove(p)
        print(f"  Deleted stale: {os.path.basename(p)}")

print(f"\n=== Done ===")
print(f"  v13: {len(v13_dets)} detections")
print(f"  Candidates scored: {len(scored)}")
print(f"  v14 appearance-filtered: {len(final)} detections")
print(f"  Review strip: {REVIEW_STRIP}")
print(f"\nRun pipeline to regenerate v14.")

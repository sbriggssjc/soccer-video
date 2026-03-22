"""
Second-pass ball detection on the portrait (zoomed) video.

The ball is ~1.8x larger in the portrait crop, so YOLO should detect it
more reliably. We then map detections back to original landscape coordinates
and merge with the existing polished detections.
"""

import json, csv, os, sys, shutil
import numpy as np

# Try importing dependencies
try:
    import cv2
except ImportError:
    print("ERROR: cv2 not available")
    sys.exit(1)

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not available")
    sys.exit(1)

# === Paths ===
PORTRAIT_VIDEO = r"D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_NEOFC\002__polished_v12.mp4"
DIAG_CSV       = r"D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_NEOFC\002__polished_v12.diag.csv"
TELEMETRY      = r"D:\Projects\soccer-video\out\telemetry"
STEM           = "002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00"
POLISHED_V12   = os.path.join(TELEMETRY, STEM + ".yolo_ball.polished_v12.jsonl")
OUTPUT         = os.path.join(TELEMETRY, STEM + ".yolo_ball.jsonl")
SECOND_PASS    = os.path.join(TELEMETRY, STEM + ".yolo_ball.portrait_pass.jsonl")
MERGED_V13     = os.path.join(TELEMETRY, STEM + ".yolo_ball.merged_v13.jsonl")

# === Config ===
YOLO_MODELS = ["yolov8x.pt", "yolov8m.pt"]  # Try xlarge first, then medium
MIN_CONF = 0.20        # Lower threshold since portrait view should give cleaner detections
INPUT_SIZE = 1920      # Full resolution - preserves the zoom advantage of portrait crop
BALL_CLASS = 32        # COCO sports ball
MAX_FIELD_Y_SRC = 650  # Reject detections mapping to bottom of source frame
MERGE_PREFER_DIST = 100  # If portrait detection is within this of existing, keep existing

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

# === Load crop geometry from diag CSV ===
print("=== Loading crop geometry ===")
crop_info = {}
with open(DIAG_CSV) as f:
    reader = csv.DictReader(f)
    for row in reader:
        frame = int(row["frame"])
        crop_info[frame] = {
            "crop_x0": float(row["crop_x0"]),
            "crop_y0": float(row["crop_y0"]),
            "crop_w": float(row["crop_w"]),
            "crop_h": float(row["crop_h"]),
        }
print(f"  Loaded crop info for {len(crop_info)} frames")

# Check portrait video dimensions
cap = cv2.VideoCapture(PORTRAIT_VIDEO)
portrait_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
portrait_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"  Portrait video: {portrait_w}x{portrait_h}, {total_frames} frames, {fps}fps")

# === Run YOLO on portrait video ===
print(f"\n=== Running YOLO ball detection on portrait video ===")
# Try to load model
model = None
for model_name in YOLO_MODELS:
    try:
        model = YOLO(model_name)
        print(f"  Loaded model: {model_name}")
        break
    except Exception as e:
        print(f"  Failed to load {model_name}: {e}")

if model is None:
    print("ERROR: No YOLO model available")
    sys.exit(1)

portrait_dets = []
frame_idx = 0
det_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO
    results = model.predict(frame, conf=MIN_CONF, iou=0.45, imgsz=INPUT_SIZE, verbose=False)

    if results and results[0].boxes is not None:
        boxes = results[0].boxes
        cls = boxes.cls.cpu().numpy() if hasattr(boxes.cls, 'cpu') else np.asarray(boxes.cls)
        conf = boxes.conf.cpu().numpy() if hasattr(boxes.conf, 'cpu') else np.asarray(boxes.conf)
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else np.asarray(boxes.xyxy)

        # Find ball detections
        best_ball = None
        best_conf = 0
        for c, cf, box in zip(cls, conf, xyxy):
            if int(c) != BALL_CLASS:
                continue
            cf_val = float(cf)
            if cf_val < MIN_CONF:
                continue
            x1, y1, x2, y2 = box.astype(float)
            bw = x2 - x1
            bh = y2 - y1
            if bw <= 1 or bh <= 1:
                continue
            if bw > portrait_w * 0.3 or bh > portrait_h * 0.3:
                continue
            cx = x1 + bw / 2
            cy = y1 + bh / 2
            if cf_val > best_conf:
                best_conf = cf_val
                best_ball = {"cx": cx, "cy": cy, "conf": cf_val, "w": bw, "h": bh}

        if best_ball:
            # Map portrait coords to source landscape coords
            ci = crop_info.get(frame_idx)
            if ci:
                scale_x = ci["crop_w"] / portrait_w
                scale_y = ci["crop_h"] / portrait_h
                src_cx = ci["crop_x0"] + best_ball["cx"] * scale_x
                src_cy = ci["crop_y0"] + best_ball["cy"] * scale_y

                portrait_dets.append({
                    "frame": frame_idx,
                    "cx": round(src_cx, 1),
                    "cy": round(src_cy, 1),
                    "conf": round(best_ball["conf"], 4),
                    "w": round(best_ball["w"] * scale_x, 1),
                    "h": round(best_ball["h"] * scale_y, 1),
                    "portrait_cx": round(best_ball["cx"], 1),
                    "portrait_cy": round(best_ball["cy"], 1),
                    "source": "portrait_yolo",
                })
                det_count += 1

    frame_idx += 1
    if frame_idx % 50 == 0:
        print(f"  Processed {frame_idx}/{total_frames} frames, {det_count} detections so far")

cap.release()
print(f"\n  Total: {det_count} portrait detections from {frame_idx} frames ({100*det_count/max(1,frame_idx):.1f}%)")

# === Filter portrait detections ===
print(f"\n=== Filtering portrait detections ===")
filtered = []
for d in portrait_dets:
    # Reject if mapping to bottom of source frame
    if d["cy"] > MAX_FIELD_Y_SRC:
        continue
    # Reject very low confidence
    if d["conf"] < 0.25:
        continue
    filtered.append(d)
print(f"  Kept {len(filtered)}/{len(portrait_dets)} after filtering")

# Save portrait-only detections
save_jsonl(filtered, SECOND_PASS)
print(f"  Portrait detections saved -> {SECOND_PASS}")

# === Merge with existing polished detections ===
print(f"\n=== Merging with polished v12 detections ===")
existing = load_jsonl(POLISHED_V12)
existing_frames = {d["frame"]: d for d in existing}
print(f"  Existing polished: {len(existing)} detections")
print(f"  Portrait pass: {len(filtered)} detections")

# Merge strategy:
# - For frames with existing v12 detection: keep existing (it's already validated)
# - For frames WITHOUT existing detection: add portrait detection if consistent
# - Consistency check: if portrait det is wildly different from interpolated
#   trajectory of existing dets, skip it

# Build interpolated trajectory from existing
existing_sorted = sorted(existing, key=lambda d: d["frame"])
interp_cx = np.full(total_frames, np.nan)
for d in existing:
    interp_cx[d["frame"]] = d["cx"]
for i in range(len(existing_sorted) - 1):
    f0, f1 = existing_sorted[i]["frame"], existing_sorted[i+1]["frame"]
    cx0, cx1 = existing_sorted[i]["cx"], existing_sorted[i+1]["cx"]
    for f in range(f0 + 1, f1):
        if f < total_frames:
            t = (f - f0) / (f1 - f0)
            interp_cx[f] = cx0 + t * (cx1 - cx0)
# Extrapolate edges
if existing_sorted:
    for f in range(0, existing_sorted[0]["frame"]):
        interp_cx[f] = existing_sorted[0]["cx"]
    for f in range(existing_sorted[-1]["frame"] + 1, total_frames):
        interp_cx[f] = existing_sorted[-1]["cx"]

merged = list(existing)  # Start with all existing
added = 0
replaced = 0
skipped_consistency = 0
skipped_existing = 0

for d in filtered:
    f = d["frame"]
    if f in existing_frames:
        # Existing detection already validated - keep it
        skipped_existing += 1
        continue

    # Check consistency with interpolated trajectory
    if not np.isnan(interp_cx[f]):
        dist = abs(d["cx"] - interp_cx[f])
        if dist > 250:
            skipped_consistency += 1
            continue

    # Add this portrait detection (remove portrait-specific fields)
    clean_det = {
        "frame": d["frame"],
        "cx": d["cx"],
        "cy": d["cy"],
        "conf": d["conf"],
        "w": d.get("w", 20),
        "h": d.get("h", 20),
    }
    merged.append(clean_det)
    added += 1

merged.sort(key=lambda d: d["frame"])

# Deduplicate by frame (keep higher conf)
deduped = {}
for d in merged:
    f = d["frame"]
    if f not in deduped or d.get("conf", 0) > deduped[f].get("conf", 0):
        deduped[f] = d
merged = sorted(deduped.values(), key=lambda d: d["frame"])

print(f"  Added: {added} new portrait detections")
print(f"  Skipped (existing): {skipped_existing}")
print(f"  Skipped (consistency): {skipped_consistency}")
print(f"  Final merged: {len(merged)} detections")
print(f"  Coverage: {len(merged)}/{total_frames} = {100*len(merged)/total_frames:.1f}%")

# === Velocity scrub on merged set ===
print(f"\n=== Final velocity scrub ===")
scrubbed = set()
for iteration in range(3):
    active = [d for d in merged if d["frame"] not in scrubbed]
    new_scrubs = set()
    for i in range(1, len(active)):
        d0, d1 = active[i-1], active[i]
        dt = d1["frame"] - d0["frame"]
        if dt == 0 or dt > 10:
            continue
        vel = abs(d1["cx"] - d0["cx"]) / dt
        if vel > 100:
            # Which is the outlier?
            if i >= 2:
                vel_in = abs(d0["cx"] - active[i-2]["cx"]) / max(1, d0["frame"] - active[i-2]["frame"])
            else:
                vel_in = 0
            if i < len(active) - 1:
                vel_out = abs(d1["cx"] - active[i+1]["cx"]) / max(1, active[i+1]["frame"] - d1["frame"])
            else:
                vel_out = 0
            if vel_in > 50:
                new_scrubs.add(d0["frame"])
            elif vel_out > 50:
                new_scrubs.add(d1["frame"])
    if not new_scrubs:
        break
    print(f"  Iter {iteration+1}: scrubbed {len(new_scrubs)} outliers: {sorted(new_scrubs)}")
    scrubbed.update(new_scrubs)

final = [d for d in merged if d["frame"] not in scrubbed]
print(f"  Final after scrub: {len(final)} detections ({100*len(final)/total_frames:.1f}% coverage)")

# === Save ===
print(f"\n=== Saving ===")
save_jsonl(final, MERGED_V13)
print(f"  Merged v13 -> {MERGED_V13}")

backup = OUTPUT + ".pre_v13_backup"
if os.path.exists(OUTPUT):
    shutil.copy2(OUTPUT, backup)
shutil.copy2(MERGED_V13, OUTPUT)
print(f"  Main file updated -> {OUTPUT}")

# Delete stale caches
stale = [".tracker_ball.jsonl", ".ball.jsonl", ".ball.follow.jsonl", ".ball.follow__smooth.jsonl"]
for suffix in stale:
    p = os.path.join(TELEMETRY, STEM + suffix)
    if os.path.exists(p):
        os.remove(p)
        print(f"  Deleted stale: {os.path.basename(p)}")

# === Summary ===
print(f"\n=== Summary ===")
print(f"  v12 polished: {len(existing)} YOLO detections (16.3% coverage)")
print(f"  Portrait pass: {len(filtered)} raw detections")
print(f"  v13 merged: {len(final)} detections ({100*len(final)/total_frames:.1f}% coverage)")

# Show gap analysis
gaps = []
frames = sorted(d["frame"] for d in final)
for i in range(1, len(frames)):
    gap = frames[i] - frames[i-1]
    if gap > 10:
        gaps.append((frames[i-1], frames[i], gap))
if gaps:
    print(f"\n  Remaining gaps >10f:")
    for a, b, g in gaps:
        print(f"    f{a} -> f{b}: {g} frames")
else:
    print(f"\n  No gaps >10 frames!")

print(f"\nDone! Run pipeline to regenerate v13.")

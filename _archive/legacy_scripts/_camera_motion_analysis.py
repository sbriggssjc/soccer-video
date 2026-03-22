"""
Camera Motion + YOLO Ball Position Analysis
=============================================
1. Estimate physical camera pan/tilt using sparse optical flow + RANSAC
2. Build cumulative camera displacement (how far camera has panned)
3. Plot confirmed YOLO ball positions in both frame coords and world coords
4. Design optimal portrait crop path
"""
import json, cv2, math, sys
import numpy as np
from pathlib import Path

base = Path(r"D:\Projects\soccer-video")
src_video = base / r"out\atomic_clips\2026-02-23__TSC_vs_NEOFC\002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.mp4"

SRC_W, SRC_H = 1920, 1080
CROP_W = 608  # portrait crop width

# ── 1. Load confirmed YOLO ball detections ──────────────────────────
FALSE_YOLO = {332, 466}
yolo_path = base / "out/telemetry/002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.yolo_ball.jsonl"
yolo = {}
with open(yolo_path) as f:
    for line in f:
        line = line.strip()
        if not line: continue
        rec = json.loads(line)
        if rec.get("_meta"): continue
        fr = int(rec["frame"])
        if fr in FALSE_YOLO: continue
        yolo[fr] = (float(rec["cx"]), float(rec["cy"]), float(rec.get("conf", 0)))

# Also load tracker data  
tracker_path = base / "out/telemetry/002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.tracker_ball.jsonl"
tracker = {}
with open(tracker_path) as f:
    for line in f:
        line = line.strip()
        if not line: continue
        rec = json.loads(line)
        if rec.get("_meta"): continue
        fr = int(rec["frame"])
        tracker[fr] = (float(rec["cx"]), float(rec["cy"]))

print(f"Confirmed YOLO: {len(yolo)} detections")
print(f"Tracker: {len(tracker)} detections")

# ── 2. Estimate camera motion frame-to-frame ────────────────────────
print("\n[CAMERA] Estimating frame-to-frame motion...")
cap = cv2.VideoCapture(str(src_video))
n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Read first frame
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Camera displacement per frame (dx, dy)
cam_dx = np.zeros(n_total)
cam_dy = np.zeros(n_total)
# Cumulative camera position (world coords of frame origin)
cam_cum_x = np.zeros(n_total)
cam_cum_y = np.zeros(n_total)

# Feature detection params
feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=7)
lk_params = dict(winSize=(21, 21), maxLevel=3, 
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

fr = 0
scene_cuts = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    fr += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect features in previous frame
    pts0 = cv2.goodFeaturesToTrack(prev_gray, **feature_params)
    if pts0 is not None and len(pts0) >= 10:
        pts1, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts0, None, **lk_params)
        good_old = pts0[status.ravel() == 1]
        good_new = pts1[status.ravel() == 1]
        
        if len(good_old) >= 6:
            # RANSAC affine to estimate camera motion (ignoring moving objects)
            M, inliers = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC, ransacReprojThreshold=3.0)
            if M is not None:
                dx = M[0, 2]  # translation X
                dy = M[1, 2]  # translation Y
                
                # Detect scene cuts (huge jumps)
                if abs(dx) > 100 or abs(dy) > 100:
                    scene_cuts.append(fr)
                    dx, dy = 0, 0
                
                cam_dx[fr] = dx
                cam_dy[fr] = dy
    
    cam_cum_x[fr] = cam_cum_x[fr-1] + cam_dx[fr]
    cam_cum_y[fr] = cam_cum_y[fr-1] + cam_dy[fr]
    
    prev_gray = gray
    if fr % 100 == 0:
        print(f"  frame {fr}/{n_total}")

cap.release()
print(f"[CAMERA] Done. {len(scene_cuts)} scene cuts detected at: {scene_cuts}")

# ── 3. Convert YOLO positions to world coordinates ──────────────────
# World X = frame_x + cumulative_camera_x
# This tells us where the ball ACTUALLY is on the pitch
print("\n[WORLD] YOLO ball positions in world coordinates:")
print(f"{'frame':>5} {'frame_x':>8} {'frame_y':>8} {'cam_cum_x':>10} {'world_x':>10} {'conf':>6}")
print("-" * 55)

yolo_world = {}
for fr in sorted(yolo.keys()):
    bx, by, conf = yolo[fr]
    wx = bx + cam_cum_x[fr]
    wy = by + cam_cum_y[fr]
    yolo_world[fr] = (wx, wy, conf)
    print(f"{fr:5d} {bx:8.1f} {by:8.1f} {cam_cum_x[fr]:10.1f} {wx:10.1f} {conf:6.3f}")

# Also tracker in world coords
tracker_world = {}
for fr in sorted(tracker.keys()):
    tx, ty = tracker[fr]
    wx = tx + cam_cum_x[fr]
    wy = ty + cam_cum_y[fr]
    tracker_world[fr] = (wx, wy)

# ── 4. Camera motion summary ────────────────────────────────────────
print(f"\n[CAMERA MOTION SUMMARY]")
print(f"  Total X displacement: {cam_cum_x[-1]:.1f}px")
print(f"  Total Y displacement: {cam_cum_y[-1]:.1f}px")
print(f"  X range: [{cam_cum_x.min():.1f}, {cam_cum_x.max():.1f}]")
print(f"  Y range: [{cam_cum_y.min():.1f}, {cam_cum_y.max():.1f}]")

# Show camera motion every 30 frames
print(f"\n  Camera cumulative displacement every 30 frames:")
for i in range(0, n_total, 30):
    print(f"    f{i:3d}: dx={cam_cum_x[i]:7.1f}  dy={cam_cum_y[i]:7.1f}")

# ── 5. Identify segments between scene cuts ─────────────────────────
# Split the clip at scene cuts - each segment has continuous camera motion
cuts = [0] + scene_cuts + [n_total - 1]
segments = []
for i in range(len(cuts) - 1):
    start = cuts[i]
    end = cuts[i + 1]
    seg_yolo = {f: yolo[f] for f in yolo if start <= f <= end}
    seg_tracker = {f: tracker[f] for f in tracker if start <= f <= end}
    segments.append({
        "start": start, "end": end, "frames": end - start + 1,
        "yolo_count": len(seg_yolo), "tracker_count": len(seg_tracker),
        "cam_pan_x": cam_cum_x[end] - cam_cum_x[start]
    })

print(f"\n[SEGMENTS]")
for i, seg in enumerate(segments):
    print(f"  Seg {i}: f{seg['start']}-f{seg['end']} ({seg['frames']} frames, "
          f"{seg['yolo_count']} YOLO, {seg['tracker_count']} tracker, "
          f"cam_pan={seg['cam_pan_x']:.0f}px)")

# ── 6. For each YOLO detection, compute where portrait crop should be
# to center the ball ─────────────────────────────────────────────
half_crop = CROP_W / 2.0
print(f"\n[CROP ANALYSIS] Portrait crop width={CROP_W}px, half={half_crop:.0f}px")
print(f"  Crop center X must be in [{half_crop:.0f}, {SRC_W - half_crop:.0f}]")
print(f"\n  For each YOLO detection, ideal crop center vs constraints:")
for fr in sorted(yolo.keys()):
    bx, by, conf = yolo[fr]
    ideal_cx = bx
    clamped_cx = max(half_crop, min(SRC_W - half_crop, ideal_cx))
    offset = abs(ideal_cx - clamped_cx)
    flag = " CLAMPED" if offset > 1 else ""
    print(f"    f{fr:3d}: ball_x={bx:7.1f}  ideal_crop_cx={ideal_cx:7.1f}  "
          f"clamped={clamped_cx:7.1f}  offset={offset:5.1f}{flag}")

# ── 7. Save analysis data for the path planner ──────────────────────
import pickle
analysis = {
    "n_frames": n_total,
    "cam_cum_x": cam_cum_x,
    "cam_cum_y": cam_cum_y,
    "cam_dx": cam_dx,
    "cam_dy": cam_dy,
    "scene_cuts": scene_cuts,
    "yolo": yolo,
    "yolo_world": yolo_world,
    "tracker": tracker,
    "tracker_world": tracker_world,
    "segments": segments,
}
analysis_path = base / "_camera_analysis.pkl"
with open(analysis_path, "wb") as f:
    pickle.dump(analysis, f)
print(f"\n[SAVED] Analysis data to {analysis_path}")

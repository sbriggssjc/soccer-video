"""Extract 80x80 crops at kept YOLO ball detections from v21f diagnostic CSV."""

import csv
import cv2
import numpy as np
import os

# Paths
DIAG_CSV = r"D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_NEOFC\002__v21f.diag.csv"
VIDEO    = r"D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.mp4"
OUT_DIR  = r"D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_NEOFC\v21_snapshots\v21f_crops"

CROP_SIZE = 80
CENTER_PATCH = 20  # for HSV analysis

os.makedirs(OUT_DIR, exist_ok=True)

# 1. Load YOLO rows from diagnostic CSV
yolo_rows = []
with open(DIAG_CSV, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["source"] == "yolo":
            yolo_rows.append({
                "frame": int(row["frame"]),
                "ball_x": float(row["ball_x"]),
                "ball_y": float(row["ball_y"]),
                "conf": float(row["confidence"]),
                "cam_cx": float(row["cam_cx"]),
            })

print(f"Found {len(yolo_rows)} YOLO detections")

# 2. Open video
cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO}")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video: {vid_w}x{vid_h}, {total_frames} frames")

# Sort by frame for sequential reading
yolo_rows.sort(key=lambda r: r["frame"])

# 3-4. Extract crops and compute metrics
results = []
prev_frame_idx = -1

for row in yolo_rows:
    fnum = row["frame"]
    bx = row["ball_x"]
    by = row["ball_y"]
    conf = row["conf"]
    cam_cx = row["cam_cx"]

    # Seek to frame
    if fnum != prev_frame_idx + 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
    ret, frame = cap.read()
    if not ret:
        print(f"  WARNING: cannot read frame {fnum}")
        continue
    prev_frame_idx = fnum

    h, w = frame.shape[:2]
    half = CROP_SIZE // 2

    # Compute crop bounds (clamp to image)
    cx, cy = int(round(bx)), int(round(by))
    x0 = max(cx - half, 0)
    y0 = max(cy - half, 0)
    x1 = min(cx + half, w)
    y1 = min(cy + half, h)

    crop = frame[y0:y1, x0:x1]

    # Pad if needed (near edges)
    if crop.shape[0] < CROP_SIZE or crop.shape[1] < CROP_SIZE:
        padded = np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
        py = (CROP_SIZE - crop.shape[0]) // 2
        px = (CROP_SIZE - crop.shape[1]) // 2
        padded[py:py+crop.shape[0], px:px+crop.shape[1]] = crop
        crop = padded

    # Save crop
    fname = f"f{fnum:03d}_x{bx:.0f}_y{by:.0f}_c{conf:.2f}.png"
    cv2.imwrite(os.path.join(OUT_DIR, fname), crop)

    # Mean brightness (grayscale)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())

    # HSV analysis of center 20x20 patch
    ch = CENTER_PATCH // 2
    center_crop = crop[half-ch:half+ch, half-ch:half+ch]
    hsv = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
    mean_h = float(hsv[:, :, 0].mean())
    mean_s = float(hsv[:, :, 1].mean())
    mean_v = float(hsv[:, :, 2].mean())

    # Simple "ball visible" heuristic: bright region in center
    center_gray = gray[half-ch:half+ch, half-ch:half+ch]
    center_bright = float(center_gray.mean())
    ball_visible = center_bright > 140  # white/bright ball

    results.append({
        "frame": fnum,
        "x": bx,
        "y": by,
        "conf": conf,
        "cam_cx": cam_cx,
        "brightness": brightness,
        "ball_visible": ball_visible,
        "mean_H": mean_h,
        "mean_S": mean_s,
        "mean_V": mean_v,
    })

cap.release()

# 5. Print summary table
print()
header = f"{'frame':>5}  {'x':>7}  {'y':>7}  {'conf':>5}  {'cam_cx':>7}  {'bright':>6}  {'vis':>3}  {'H':>5}  {'S':>5}  {'V':>5}"
print(header)
print("-" * len(header))
for r in results:
    vis_str = "YES" if r["ball_visible"] else " no"
    print(f"{r['frame']:5d}  {r['x']:7.1f}  {r['y']:7.1f}  {r['conf']:5.2f}  {r['cam_cx']:7.1f}  {r['brightness']:6.1f}  {vis_str}  {r['mean_H']:5.1f}  {r['mean_S']:5.1f}  {r['mean_V']:5.1f}")

print(f"\n{len(results)} crops saved to {OUT_DIR}")

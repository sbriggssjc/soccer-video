"""
Generate a visual review strip of all v13 detections.
Each thumbnail shows the ball region with frame number and confidence.
User can quickly identify phantom detections to remove.
"""

import json, os, sys, shutil, math
import numpy as np

try:
    import cv2
except ImportError:
    print("ERROR: cv2 required"); sys.exit(1)

# === Paths ===
CLIP = r"D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.mp4"
TELEMETRY = r"D:\Projects\soccer-video\out\telemetry"
STEM = "002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00"
MERGED_V13 = os.path.join(TELEMETRY, STEM + ".yolo_ball.merged_v13.jsonl")
MAIN_OUTPUT = os.path.join(TELEMETRY, STEM + ".yolo_ball.jsonl")
OUTPUT_DIR = r"D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_NEOFC"
REVIEW_IMG = os.path.join(OUTPUT_DIR, "002__detection_review.png")

# === Config ===
CROP_RADIUS = 50  # Larger crop so you can see context
THUMB_SIZE = 120
COLS = 10
LABEL_HEIGHT = 36  # Space for text below each thumbnail

def load_jsonl(path):
    dets = []
    with open(path) as f:
        for line in f:
            dets.append(json.loads(line))
    return sorted(dets, key=lambda d: d["frame"])

# === Restore v13 to main output ===
print("=== Restoring v13 detections ===")
shutil.copy2(MERGED_V13, MAIN_OUTPUT)
print(f"  Main file restored from v13 ({MERGED_V13})")

# Delete stale caches
for suffix in [".tracker_ball.jsonl", ".ball.jsonl", ".ball.follow.jsonl", ".ball.follow__smooth.jsonl"]:
    p = os.path.join(TELEMETRY, STEM + suffix)
    if os.path.exists(p):
        os.remove(p)

# === Load v13 detections ===
dets = load_jsonl(MERGED_V13)
print(f"  V13 detections: {len(dets)}")

# === Extract crops from source video ===
print("\n=== Extracting detection crops ===")
cap = cv2.VideoCapture(CLIP)
if not cap.isOpened():
    print(f"ERROR: Cannot open {CLIP}")
    sys.exit(1)

frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

crops = []
for d in dets:
    cap.set(cv2.CAP_PROP_POS_FRAMES, d["frame"])
    ret, frame = cap.read()
    if not ret:
        crops.append(None)
        continue

    cx, cy = int(d["cx"]), int(d["cy"])
    r = CROP_RADIUS
    x0 = max(0, cx - r)
    y0 = max(0, cy - r)
    x1 = min(frame_w, cx + r)
    y1 = min(frame_h, cy + r)
    crop = frame[y0:y1, x0:x1].copy()

    # Draw crosshair at ball center
    local_cx = cx - x0
    local_cy = cy - y0
    cv2.circle(crop, (local_cx, local_cy), 8, (0, 255, 255), 1)
    cv2.line(crop, (local_cx - 12, local_cy), (local_cx + 12, local_cy), (0, 255, 255), 1)
    cv2.line(crop, (local_cx, local_cy - 12), (local_cx, local_cy + 12), (0, 255, 255), 1)

    crops.append(crop)

cap.release()
print(f"  Extracted {sum(1 for c in crops if c is not None)} crops")

# === Build review image ===
print("\n=== Building review image ===")
n = len(dets)
ROWS = math.ceil(n / COLS)
cell_w = THUMB_SIZE + 6
cell_h = THUMB_SIZE + LABEL_HEIGHT + 6
img_w = COLS * cell_w + 6
img_h = ROWS * cell_h + 40  # Extra space for title

img = np.ones((img_h, img_w, 3), dtype=np.uint8) * 30

# Title
cv2.putText(img, f"Ball Detection Review - {n} detections - Mark phantom frame numbers to remove",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

for idx, (d, crop) in enumerate(zip(dets, crops)):
    row = idx // COLS
    col = idx % COLS
    x0 = col * cell_w + 6
    y0 = row * cell_h + 40

    if crop is not None:
        thumb = cv2.resize(crop, (THUMB_SIZE, THUMB_SIZE))
    else:
        thumb = np.zeros((THUMB_SIZE, THUMB_SIZE, 3), dtype=np.uint8)
        cv2.putText(thumb, "NO FRAME", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)

    # Green border
    cv2.rectangle(thumb, (0, 0), (THUMB_SIZE-1, THUMB_SIZE-1), (0, 180, 0), 2)

    img[y0:y0+THUMB_SIZE, x0:x0+THUMB_SIZE] = thumb

    # Labels
    line1 = f"f{d['frame']}"
    line2 = f"c={d.get('conf',0):.2f} x={d['cx']:.0f}"
    cv2.putText(img, line1, (x0, y0 + THUMB_SIZE + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
    cv2.putText(img, line2, (x0, y0 + THUMB_SIZE + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

cv2.imwrite(REVIEW_IMG, img)
print(f"  Review image saved -> {REVIEW_IMG}")
print(f"\n  Image size: {img_w}x{img_h}")
print(f"  {n} detections in {ROWS} rows x {COLS} cols")
print(f"\nOpen the review image and tell me which frame numbers are phantom balls to remove.")
print(f"Look for: cones, ground markings, cleats, or anything that's NOT the actual game ball.")

"""Generate frame strip from the f170-f396 gap for manual ball marking.
Shows full-width frames at intervals so the user can identify ball positions.
Each frame gets a number label and crosshair grid to help locate coordinates.
"""
import cv2
import numpy as np
import os

SRC = r"D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.mp4"
OUT = r"D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_NEOFC\002__gap_frames.png"

# Sample every 15 frames in the gap
GAP_START = 170
GAP_END = 396
STEP = 15
frames_to_sample = list(range(GAP_START, GAP_END + 1, STEP))
# Make sure we include the endpoints
if GAP_END not in frames_to_sample:
    frames_to_sample.append(GAP_END)

print(f"Sampling {len(frames_to_sample)} frames: {frames_to_sample}")

cap = cv2.VideoCapture(SRC)
if not cap.isOpened():
    print("ERROR: Cannot open video")
    exit(1)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Source: {src_w}x{src_h}, {total_frames} frames")

# Read frames
frame_imgs = {}
for f in frames_to_sample:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ret, img = cap.read()
    if ret:
        frame_imgs[f] = img
    else:
        print(f"WARNING: Could not read frame {f}")
cap.release()

# Layout: scale each frame to fit in a grid
# Full width at ~640px wide for readability
THUMB_W = 640
scale = THUMB_W / src_w
THUMB_H = int(src_h * scale)
COLS = 2
ROWS = (len(frames_to_sample) + COLS - 1) // COLS
LABEL_H = 30
CELL_H = THUMB_H + LABEL_H

canvas_w = COLS * THUMB_W
canvas_h = ROWS * CELL_H
canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

# Draw grid lines on each frame to help locate coordinates
# Grid at every 200px in source space
for idx, f in enumerate(frames_to_sample):
    if f not in frame_imgs:
        continue
    row = idx // COLS
    col = idx % COLS
    x0 = col * THUMB_W
    y0 = row * CELL_H

    img = frame_imgs[f].copy()

    # Draw vertical grid lines every 200px with labels
    for gx in range(0, src_w, 200):
        cv2.line(img, (gx, 0), (gx, src_h), (128, 128, 128), 1)
        cv2.putText(img, str(gx), (gx + 3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Draw horizontal center line
    cy = src_h // 2
    cv2.line(img, (0, cy), (src_w, cy), (128, 128, 128), 1)

    # Scale and place
    thumb = cv2.resize(img, (THUMB_W, THUMB_H))
    canvas[y0 + LABEL_H:y0 + CELL_H, x0:x0 + THUMB_W] = thumb

    # Frame label
    label = f"Frame {f} (t={f/30:.1f}s)"
    cv2.putText(canvas, label, (x0 + 10, y0 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

cv2.imwrite(OUT, canvas)
print(f"Saved gap frame strip: {OUT}")
print(f"Grid size: {canvas_w}x{canvas_h}")
print(f"\nFor each frame, estimate the ball x-coordinate using the grid lines.")
print(f"Grid lines are at x=0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800")
print(f"If ball is not visible, mark as 'none'")

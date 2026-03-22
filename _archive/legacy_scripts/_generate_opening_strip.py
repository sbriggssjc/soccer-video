"""Generate frame strip for f0-f155 so user can mark ball positions at clip start."""
import cv2
import numpy as np

SRC = r"D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.mp4"
OUT = r"D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_NEOFC\002__opening_frames.png"

# Sample every 15 frames from f0 to f155
frames_to_sample = list(range(0, 156, 15))
if 155 not in frames_to_sample:
    frames_to_sample.append(155)

print(f"Sampling {len(frames_to_sample)} frames: {frames_to_sample}")

cap = cv2.VideoCapture(SRC)
src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frame_imgs = {}
for f in frames_to_sample:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ret, img = cap.read()
    if ret:
        frame_imgs[f] = img
cap.release()

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

for idx, f in enumerate(frames_to_sample):
    if f not in frame_imgs:
        continue
    row = idx // COLS
    col = idx % COLS
    x0 = col * THUMB_W
    y0 = row * CELL_H
    img = frame_imgs[f].copy()
    # Draw 3 vertical grid lines at 25%, 50%, 75%
    for pct in [25, 50, 75]:
        gx = int(src_w * pct / 100)
        cv2.line(img, (gx, 0), (gx, src_h), (0, 255, 255), 2)
        cv2.putText(img, str(pct), (gx + 5, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    thumb = cv2.resize(img, (THUMB_W, THUMB_H))
    canvas[y0 + LABEL_H:y0 + CELL_H, x0:x0 + THUMB_W] = thumb
    label = f"Frame {f} (t={f/30:.1f}s)"
    cv2.putText(canvas, label, (x0 + 10, y0 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

cv2.imwrite(OUT, canvas)
print(f"Saved: {OUT}")
print(f"Grid lines at 25, 50, 75 (same scale as before)")
print(f"Mark ball x as percentage: 0=left edge, 25/50/75=grid lines, 100=right edge")

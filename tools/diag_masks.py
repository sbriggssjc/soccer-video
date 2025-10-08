import cv2
import argparse
import os
import numpy as np


def field_mask_bgr(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    grass = (H >= 35) & (H <= 95) & (S >= 40) & (V >= 40)
    m = (grass.astype(np.uint8)) * 255
    m = cv2.medianBlur(m, 5)
    m = cv2.morphologyEx(
        m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    m = cv2.morphologyEx(
        m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    )
    m = cv2.erode(m, np.ones((9, 9), np.uint8), 1)
    return m


def motion_strength(prev, cur):
    d = cv2.absdiff(prev, cur)
    g = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (7, 7), 0)
    return g


ap = argparse.ArgumentParser()
ap.add_argument("--in", required=True)
ap.add_argument("--frames", default="0,60,120,180")
ap.add_argument("--outdir", default="out\\diag_masks")
args = ap.parse_args()

os.makedirs(args.outdir, exist_ok=True)
cap = cv2.VideoCapture(args.in)
assert cap.isOpened()
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

# grab first frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ok, f0 = cap.read()
assert ok
prev = f0.copy()
mask0 = field_mask_bgr(f0)
cv2.imwrite(os.path.join(args.outdir, "000_fieldmask.png"), mask0)

for s in [int(x.strip()) for x in args.frames.split(",") if x.strip()]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, s)
    ok, fr = cap.read()
    if not ok:
        continue
    cv2.imwrite(os.path.join(args.outdir, f"{s:06d}_frame.png"), fr)
    cv2.imwrite(os.path.join(args.outdir, f"{s:06d}_fieldmask.png"), field_mask_bgr(fr))
    cv2.imwrite(os.path.join(args.outdir, f"{s:06d}_motion.png"), motion_strength(prev, fr))
    prev = fr
print("Wrote masks:", args.outdir)

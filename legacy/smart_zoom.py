# smart_zoom.py
import argparse, cv2, numpy as np, os, subprocess, sys

p = argparse.ArgumentParser()
p.add_argument('--inp', required=True, help='input mp4 (with audio)')
p.add_argument('--out', required=True, help='output mp4 (video-only temp)')
p.add_argument('--zoom', type=float, default=1.45, help='zoom factor for crop window')
p.add_argument('--smooth', type=float, default=0.85, help='EMA smoothing 0..1 (higher = steadier)')
args = p.parse_args()

cap = cv2.VideoCapture(args.inp)
if not cap.isOpened():
    print("Failed to open input", file=sys.stderr); sys.exit(1)

w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

# crop size (keep aspect)
cw = max(2, int((w / args.zoom) // 2 * 2))
ch = max(2, int((h / args.zoom) // 2 * 2))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # widely supported
tmpv = args.out
vw = cv2.VideoWriter(tmpv, fourcc, fps, (w, h))
if not vw.isOpened():
    print("Failed to open writer", file=sys.stderr); sys.exit(1)

prev_gray = None
cx, cy = w // 2, h // 2  # start centered

def clamp(v, lo, hi): return max(lo, min(hi, v))

while True:
    ok, frame = cap.read()
    if not ok: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_gray is None:
        prev_gray = gray
        # write first frame as simple center crop
        x1 = clamp(cx - cw//2, 0, w - cw)
        y1 = clamp(cy - ch//2, 0, h - ch)
        crop = frame[y1:y1+ch, x1:x1+cw]
        out  = cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC)
        vw.write(out)
        continue

    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                        0.5, 3, 25, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    # blur to reduce noise, emphasize real motion
    mag_blur = cv2.GaussianBlur(mag, (0,0), sigmaX=3, sigmaY=3)

    total = mag_blur.sum()
    if total > 1e-6:
        ys, xs = np.indices(mag_blur.shape)
        tx = (mag_blur * xs).sum() / total
        ty = (mag_blur * ys).sum() / total
        # EMA smoothing
        cx = int(args.smooth * cx + (1.0 - args.smooth) * tx)
        cy = int(args.smooth * cy + (1.0 - args.smooth) * ty)

    x1 = clamp(cx - cw//2, 0, w - cw)
    y1 = clamp(cy - ch//2, 0, h - ch)

    crop = frame[y1:y1+ch, x1:x1+cw]
    out  = cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC)
    vw.write(out)

    prev_gray = gray

vw.release(); cap.release()

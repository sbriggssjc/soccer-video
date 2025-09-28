#!/usr/bin/env python3
import argparse, os, math, json
import numpy as np
import cv2
from ultralytics import YOLO

# ------------------------- utils -------------------------
def hsv_orange_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # two orange-ish bands; tune as needed
    m1 = cv2.inRange(hsv, (5, 120, 90), (20, 255, 255))
    m2 = cv2.inRange(hsv, (0, 120, 90), (5, 255, 255))
    mask = cv2.bitwise_or(m1, m2)
    mask = cv2.medianBlur(mask, 5)
    return mask

def find_orange_centroid(bgr):
    mask = hsv_orange_mask(bgr)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None; best_score = 1e9
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 20 or area > 7000:  # tune for your footage
            continue
        (x,y), r = cv2.minEnclosingCircle(c)
        if r < 2: 
            continue
        circ = (cv2.contourArea(c) / (math.pi*r*r + 1e-6))
        score = abs(1.0 - circ) + 0.0001*area
        if score < best_score:
            best_score = score
            best = (float(x), float(y))
    return best

# ------------------------- Kalman (2D const-vel) -------------------------
class Kalman2D:
    def __init__(self, dt, q=5.0, r=8.0):
        # state: [x, vx, y, vy]
        self.dt = dt
        self.x = np.zeros((4,1), dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 1e3

        self.F = np.array([[1, dt, 0,  0],
                           [0,  1, 0,  0],
                           [0,  0, 1, dt],
                           [0,  0, 0,  1]], dtype=np.float32)

        self.H = np.array([[1, 0, 0, 0],
                           [0, 0, 1, 0]], dtype=np.float32)

        self.Q = np.array([[dt**4/4, dt**3/2,      0,       0],
                           [dt**3/2,    dt**2,      0,       0],
                           [0,              0, dt**4/4, dt**3/2],
                           [0,              0, dt**3/2,    dt**2]], dtype=np.float32) * q

        self.R = np.eye(2, dtype=np.float32) * r

    def init(self, px, py):
        self.x[:] = np.array([[px],[0],[py],[0]], dtype=np.float32)
        self.P[:] = np.eye(4, dtype=np.float32) * 5.0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0,0]), float(self.x[2,0])

    def update(self, z):
        z = np.array(z, dtype=np.float32).reshape(2,1)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

# ------------------------- main -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp",      required=True, help="input clip (.mp4)")
    ap.add_argument("--out_csv",  required=True, help="output CSV path")
    ap.add_argument("--yolo_conf", type=float, default=0.15, help="YOLO confidence")
    ap.add_argument("--roi_pad",   type=int,   default=160,  help="base ROI pad around last position")
    ap.add_argument("--roi_pad_max", type=int, default=480,  help="max ROI pad while missing")
    ap.add_argument("--max_miss",  type=int,   default=36,   help="#frames allowed missing before hard reset")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.inp)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open input: {args.inp}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    model = YOLO("yolov8n.pt")
    dt = 1.0/float(fps)
    kf = Kalman2D(dt, q=12.0, r=9.0)

    rows = []
    frame = 0
    have_init = False
    miss_count = 0
    last_xy = None

    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        t = frame / fps

        # ROI selection
        pad = args.roi_pad if miss_count == 0 else min(args.roi_pad + miss_count*12, args.roi_pad_max)
        if last_xy is None:
            x1, y1, x2, y2 = 0, 0, W, H
            roi_bgr = bgr
            roi_off = (0,0)
        else:
            cx, cy = last_xy
            x1 = max(0, int(cx - pad)); y1 = max(0, int(cy - pad))
            x2 = min(W, int(cx + pad)); y2 = min(H, int(cy + pad))
            roi_bgr = bgr[y1:y2, x1:x2]
            roi_off = (x1, y1)

        det_cx, det_cy = None, None

        # YOLO detect in ROI (faster, more precise)
        try:
            res = model.predict(roi_bgr, verbose=False, conf=args.yolo_conf, classes=[32])
            if res and len(res[0].boxes):
                boxes = res[0].boxes.xyxy.cpu().numpy()
                confs = res[0].boxes.conf.cpu().numpy()
                # pick smallest likely "ball" box, bias by size & conf
                areas = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
                rank  = areas + (1.0-confs)*1e4  # prefer small + confident
                i = int(np.argmin(rank))
                x1b,y1b,x2b,y2b = boxes[i]
                det_cx = float((x1b+x2b)/2.0) + roi_off[0]
                det_cy = float((y1b+y2b)/2.0) + roi_off[1]
        except Exception:
            pass

        # Fallback: HSV orange in ROI if YOLO missed
        if det_cx is None:
            c = find_orange_centroid(roi_bgr)
            if c is not None:
                det_cx = float(c[0]) + roi_off[0]
                det_cy = float(c[1]) + roi_off[1]

        # Kalman + bookkeeping
        if det_cx is not None and det_cy is not None:
            if not have_init:
                kf.init(det_cx, det_cy)
                have_init = True
            else:
                kf.predict()
                kf.update((det_cx, det_cy))
            est_x, est_y = det_cx, det_cy
            last_xy = (est_x, est_y)
            miss_count = 0
        else:
            # no detection: predict forward if initialized
            if have_init:
                est_x, est_y = kf.predict()
                last_xy = (est_x, est_y)
                miss_count += 1
                if miss_count > args.max_miss:
                    # hard reset: forget ROI for a bit
                    have_init = False
                    last_xy = None
                    miss_count = 0
            else:
                est_x, est_y = (np.nan, np.nan)

        rows.append((frame, t, est_x, est_y))
        frame += 1

    cap.release()

    # Save CSV with short-gap interpolation
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    import pandas as pd
    df = pd.DataFrame(rows, columns=["frame","time","cx","cy"])
    for col in ["cx","cy"]:
        s = pd.to_numeric(df[col], errors="coerce")
        s = s.ffill(limit=10).bfill(limit=10)
        df[col] = s
    df.to_csv(args.out_csv, index=False)
    print(f"Saved {args.out_csv}")

if __name__ == "__main__":
    main()

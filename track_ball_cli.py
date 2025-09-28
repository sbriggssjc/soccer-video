# -*- coding: utf-8 -*-
import argparse, os, math, csv
import numpy as np
import cv2
from ultralytics import YOLO

class Kalman1D:
    def __init__(self, q=3.0, r=30.0):
        self.x = np.array([0.0, 0.0])  # [pos, vel]
        self.P = np.eye(2) * 1e6       # start uncertain
        self.q = float(q); self.r = float(r)

    def predict(self, dt):
        F = np.array([[1.0, dt],[0.0, 1.0]])
        Q = np.array([[self.q*dt**3/3, self.q*dt**2/2],
                      [self.q*dt**2/2, self.q*dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z):
        H = np.array([[1.0, 0.0]])
        S = H @ self.P @ H.T + self.r
        K = (self.P @ H.T) / S
        y = z - H @ self.x
        self.x = self.x + (K.flatten() * y)
        self.P = (np.eye(2) - K @ H) @ self.P

def hsv_orange_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (5, 120, 90), (20, 255, 255))
    m2 = cv2.inRange(hsv, (0, 120, 90), (5, 255, 255))
    mask = cv2.bitwise_or(m1, m2)
    mask = cv2.medianBlur(mask, 5)
    return mask

def find_orange_centroid(bgr):
    cnts, _ = cv2.findContours(hsv_orange_mask(bgr), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    best, best_score = None, 1e9
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 20 or area > 5000:
            continue
        (x,y), r = cv2.minEnclosingCircle(c)
        circ = cv2.contourArea(c) / (math.pi*r*r + 1e-6)
        score = abs(1-circ) + 0.0001*area
        if score < best_score:
            best_score, best = score, (float(x), float(y))
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp",      required=True, help="input clip (.mp4)")
    ap.add_argument("--out_csv",  required=True, help="output CSV path")
    ap.add_argument("--yolo_conf", type=float, default=0.15)
    ap.add_argument("--roi_pad",   type=int,   default=220)
    ap.add_argument("--roi_pad_max", type=int, default=560)
    ap.add_argument("--max_miss",  type=int,   default=45)
    args = ap.parse_args()

    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(args.inp)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    rows = []
    frame = 0
    miss = 0
    last_xy = None

    kf_x = Kalman1D(q=8.0, r=80.0)
    kf_y = Kalman1D(q=8.0, r=80.0)
    dt = 1.0 / max(fps, 1e-6)
    initialized = False
    last_good = None

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame","time","cx","cy","ppl"])  # NEW: ppl column

        while True:
            ok, bgr = cap.read()
            if not ok: break
            t = frame / fps

            # dynamic ROI to speed/steady detection
            if last_xy is not None:
                pad = min(
                    args.roi_pad_max,
                    args.roi_pad + min(miss, args.max_miss) * (args.roi_pad_max - args.roi_pad) / max(1, args.max_miss),
                )
                x0 = max(0, int(last_xy[0] - pad))
                y0 = max(0, int(last_xy[1] - pad))
                x1 = min(W, int(last_xy[0] + pad))
                y1 = min(H, int(last_xy[1] + pad))
                roi = bgr[y0:y1, x0:x1]
                roi_off = (x0, y0)
            else:
                roi = bgr
                roi_off = (0, 0)

            cx, cy = None, None
            ppl_centers = []

            try:
                # one pass; we’ll read both classes from results
                res = model.predict(roi, verbose=False, conf=args.yolo_conf, classes=[0, 32])  # 0=person, 32=sports ball
                if res and len(res[0].boxes):
                    boxes = res[0].boxes
                    xyxy = boxes.xyxy.cpu().numpy()
                    cls  = boxes.cls.cpu().numpy().astype(int)

                    # BALL: pick smallest "ball-like" box
                    ball_boxes = xyxy[cls==32]
                    if ball_boxes.size:
                        areas = (ball_boxes[:,2]-ball_boxes[:,0])*(ball_boxes[:,3]-ball_boxes[:,1])
                        i = int(np.argmin(areas))
                        x1,y1,x2,y2 = ball_boxes[i]
                        cx = float((x1+x2)/2) + roi_off[0]
                        cy = float((y1+y2)/2) + roi_off[1]

                    # PEOPLE: keep up to 10 centers
                    person_boxes = xyxy[cls==0]
                    for bx in person_boxes[:]:
                        x1,y1,x2,y2 = bx
                        ppl_centers.append((float((x1+x2)/2)+roi_off[0], float((y1+y2)/2)+roi_off[1]))
            except Exception:
                pass

            # color fallback for ball
            if cx is None:
                cen = find_orange_centroid(roi)
                if cen is not None:
                    cx, cy = cen[0] + roi_off[0], cen[1] + roi_off[1]

            kf_x.predict(dt)
            kf_y.predict(dt)

            if cx is not None and cy is not None:
                if not initialized:
                    kf_x.x[0] = cx
                    kf_y.x[0] = cy
                    initialized = True
                kf_x.update(cx)
                kf_y.update(cy)
                last_good = (float(kf_x.x[0]), float(kf_y.x[0]))
                sx, sy = last_good
                miss = 0
            else:
                sx = float(kf_x.x[0])
                sy = float(kf_y.x[0])
                if initialized:
                    miss += 1
                if miss > args.max_miss:
                    initialized = False
                    last_good = None
                    last_xy = None
                    kf_x = Kalman1D(q=8.0, r=80.0)
                    kf_y = Kalman1D(q=8.0, r=80.0)
                    miss = 0

            if initialized:
                last_xy = last_good if cx is None and last_good is not None else (sx, sy)

            ppl_str = "|".join(f"{px:.1f}:{py:.1f}" for px,py in ppl_centers[:10])
            wr.writerow([frame, f"{t:.3f}",
                         f"{sx:.1f}" if initialized else "",
                         f"{sy:.1f}" if initialized else "",
                         ppl_str])
            frame += 1

    cap.release()
    print(f"Saved {args.out_csv}")

if __name__ == "__main__":
    main()

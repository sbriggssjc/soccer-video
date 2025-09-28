# -*- coding: utf-8 -*-
import argparse, os, csv
import cv2, numpy as np, math
from ultralytics import YOLO

PERSON_CLS = 0  # YOLO 'person'
BALL_CLS = 32   # YOLO 'sports ball'

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
    last_ball = None  # (x,y)
    lk_prev_gray = None
    conf = 0.0       # 0..1, how sure we are about the ball this frame
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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
        f.write('frame,time,cx,cy,px,py,conf\n')
        wr = csv.writer(f)

        while True:
            ok, bgr = cap.read()
            if not ok: break
            t = frame / fps

            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

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
            px, py = None, None
            det_conf = 0.0

            try:
                res = model.predict(
                    roi,
                    verbose=False,
                    conf=args.yolo_conf,
                    classes=[PERSON_CLS, BALL_CLS],
                )
                boxes = res[0].boxes if res else None
                if boxes is not None and len(boxes):
                    xyxy = boxes.xyxy.cpu().numpy()
                    cls = boxes.cls.cpu().numpy()
                    conf_arr = boxes.conf.cpu().numpy()

                    ball_ix = np.where(cls == BALL_CLS)[0]
                    if len(ball_ix):
                        bxy = xyxy[ball_ix]
                        areas = (bxy[:, 2] - bxy[:, 0]) * (bxy[:, 3] - bxy[:, 1])
                        i = ball_ix[int(np.argmin(areas))]
                        x1, y1, x2, y2 = xyxy[i]
                        cx = float((x1 + x2) / 2.0) + roi_off[0]
                        cy = float((y1 + y2) / 2.0) + roi_off[1]
                        det_conf = float(conf_arr[i])

                    ppl_ix = np.where(cls == PERSON_CLS)[0]
                    if len(ppl_ix):
                        pxy = xyxy[ppl_ix]
                        centers = np.stack(
                            [
                                (pxy[:, 0] + pxy[:, 2]) / 2.0,
                                (pxy[:, 1] + pxy[:, 3]) / 2.0,
                            ],
                            axis=1,
                        )
                        if cx is not None:
                            d2 = (centers[:, 0] - (cx - roi_off[0])) ** 2 + (
                                centers[:, 1] - (cy - roi_off[1])
                            ) ** 2
                            j = int(np.argmin(d2))
                        else:
                            areas = (pxy[:, 2] - pxy[:, 0]) * (pxy[:, 3] - pxy[:, 1])
                            j = int(np.argmax(areas))
                        px = float(centers[j, 0]) + roi_off[0]
                        py = float(centers[j, 1]) + roi_off[1]
            except Exception:
                pass

            # color fallback for ball
            if cx is None:
                cen = find_orange_centroid(roi)
                if cen is not None:
                    cx, cy = cen[0] + roi_off[0], cen[1] + roi_off[1]

            flow_xy, flow_ok = None, False
            if last_ball is not None and lk_prev_gray is not None:
                p0 = np.array([[last_ball]], dtype=np.float32)  # shape (1,1,2)
                p1, st, err = cv2.calcOpticalFlowPyrLK(
                    lk_prev_gray,
                    gray,
                    p0,
                    None,
                    winSize=(21, 21),
                    maxLevel=2,
                    criteria=(
                        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                        20,
                        0.03,
                    ),
                )
                if st is not None and st[0, 0] == 1:
                    flow_xy = (float(p1[0, 0, 0]), float(p1[0, 0, 1]))
                    flow_ok = True

            conf = 0.0
            use_yolo = (cx is not None)
            use_flow = flow_ok

            if use_yolo and use_flow:
                alpha = 0.65  # YOLO weight
                fx, fy = flow_xy
                cx = alpha * cx + (1 - alpha) * fx
                cy = alpha * cy + (1 - alpha) * fy
                conf = min(1.0, 0.5 + 0.5 * det_conf)
            elif use_yolo:
                conf = det_conf * 0.9 + 0.1
            elif use_flow:
                cx, cy = flow_xy
                conf = 0.45  # moderate confidence from flow only
            else:
                conf = 0.0

            if cx is not None:
                cx = float(np.clip(cx, 0, W - 1))
                cy = float(np.clip(cy, 0, H - 1))
                last_ball = (cx, cy)
            lk_prev_gray = gray.copy()

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
                    last_ball = None

            if initialized:
                last_xy = last_good if cx is None and last_good is not None else (sx, sy)

            row_cx = float(kf_x.x[0]) if initialized else ""
            row_cy = float(kf_y.x[0]) if initialized else ""
            row_px = float(px) if px is not None else ""
            row_py = float(py) if py is not None else ""

            wr.writerow(
                [
                    frame,
                    f"{t:.3f}",
                    row_cx,
                    row_cy,
                    row_px,
                    row_py,
                    round(conf, 3),
                ]
            )
            frame += 1

    cap.release()
    print(f"Saved {args.out_csv}")

if __name__ == "__main__":
    main()

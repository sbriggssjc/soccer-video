import argparse, os, math, json
import cv2, numpy as np
from ultralytics import YOLO

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inp', required=True, help='input clip (.mp4)')
    ap.add_argument('--out_csv', required=True, help='output CSV path')
    ap.add_argument('--yolo_conf', type=float, default=0.12, help='YOLO confidence')
    ap.add_argument('--roi_pad', type=int, default=220, help='base ROI pad around last position')
    ap.add_argument('--roi_pad_max', type=int, default=560, help='max ROI pad while missing')
    ap.add_argument('--max_miss', type=int, default=45, help='#frames allowed missing before reset')
    ap.add_argument('--model', default='yolov8s.pt', help='YOLO model file')
    return ap.parse_args()

# simple constant-velocity Kalman filter on (x,y)
class CVKalman:
    def __init__(self, x, y, dt=1/24, q=5.0, r=9.0):
        self.x = np.array([x, y, 0.0, 0.0], dtype=float)  # [x,y,vx,vy]
        self.dt = dt
        self.P = np.eye(4)*50
        self.Q = np.eye(4)*q
        self.R = np.eye(2)*r
        self.H = np.array([[1,0,0,0],[0,1,0,0]], float)

    def predict(self):
        dt = self.dt
        F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], float)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
        return self.x[:2]

    def update(self, z):
        z = np.asarray(z, float)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P

def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.inp)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    model = YOLO(args.model)

    # LK flow params
    lk_win = (15,15)
    lk_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)

    rows = []
    frame = 0
    last_xy = None
    last_gray = None
    kf = None
    miss = 0

    # speed sanity (px/s)
    vmax_ok = 1800.0

    def pick_ball_xy(res):
        if not res or len(res[0].boxes)==0: return None
        boxes = res[0].boxes.xyxy.cpu().numpy()
        cls   = res[0].boxes.cls.cpu().numpy()
        conf  = res[0].boxes.conf.cpu().numpy()
        m = cls==32
        if not np.any(m): return None
        boxes = boxes[m]; conf = conf[m]
        # Prefer smallish boxes with good conf (balls are small)
        areas = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
        score = areas*0.002 - conf  # lower is better
        i = int(np.argmin(score))
        x1,y1,x2,y2 = boxes[i]
        return float((x1+x2)/2), float((y1+y2)/2)

    while True:
        ok, bgr = cap.read()
        if not ok: break
        t = frame / fps

        # ROI around last
        if last_xy is None:
            roi = (0,0,W,H)
        else:
            pad = min(args.roi_pad * (1.5**max(0,miss-1)), args.roi_pad_max)
            cx, cy = last_xy
            x1 = int(max(0, cx - pad)); x2 = int(min(W, cx + pad))
            y1 = int(max(0, cy - pad)); y2 = int(min(H, cy + pad))
            roi = (x1,y1,x2,y2)

        rx1,ry1,rx2,ry2 = roi
        crop = bgr[ry1:ry2, rx1:rx2]
        # YOLO in ROI (if big enough)
        yolo_xy = None
        if crop.size>0:
            res = model.predict(crop, verbose=False, conf=args.yolo_conf, iou=0.4, classes=[32])
            if res: 
                cxy = pick_ball_xy(res)
                if cxy is not None:
                    yolo_xy = (cxy[0]+rx1, cxy[1]+ry1)

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Optical flow fallback from last_xy
        flow_xy = None
        if last_xy is not None and last_gray is not None:
            p0 = np.array([[last_xy]], dtype=np.float32)
            p1, st, err = cv2.calcOpticalFlowPyrLK(last_gray, gray, p0, None, winSize=lk_win, maxLevel=3,
                                                   criteria=lk_crit)
            if st is not None and st[0,0]==1:
                flow_xy = (float(p1[0,0,0]), float(p1[0,0,1]))

        # Fuse: prefer YOLO, else flow
        meas = yolo_xy if yolo_xy is not None else flow_xy

        # Initialize KF if needed
        if kf is None:
            if meas is None:
                rows.append((frame, t, '', ''))
                last_gray = gray
                frame += 1
                continue
            kf = CVKalman(meas[0], meas[1], dt=1.0/fps, q=8.0, r=9.0)

        # Predict
        pred = kf.predict()
        use = None

        if meas is not None:
            # sanity check velocity
            if last_xy is not None:
                vx = (meas[0]-last_xy[0]) * fps
                vy = (meas[1]-last_xy[1]) * fps
                if abs(vx)>vmax_ok or abs(vy)>vmax_ok:
                    # suspicious single-frame jump → only accept if still missing
                    if miss==0:
                        meas = None

        if meas is not None:
            kf.update(meas)
            use = (kf.x[0], kf.x[1])
            last_xy = (use[0], use[1])
            miss = 0
        else:
            # no measurement → use prediction but count miss
            use = (pred[0], pred[1])
            miss += 1
            if miss > args.max_miss:
                kf = None
                last_xy = None

        rows.append((frame, t, use[0], use[1]))
        last_gray = gray
        frame += 1

    cap.release()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, 'w', newline='') as f:
        f.write('frame,time,cx,cy\n')
        for r in rows: f.write(','.join(map(str,r))+'\n')
    print(f"Saved {args.out_csv}")

if __name__=='__main__':
    main()

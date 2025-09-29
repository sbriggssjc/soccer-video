# track_ball_cli.py
import argparse, os, math, json, csv
import numpy as np
import cv2
from ultralytics import YOLO

def hsv_orange_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (5, 110, 80), (22, 255, 255))
    m2 = cv2.inRange(hsv, (0, 110, 80), (5, 255, 255))
    mask = cv2.bitwise_or(m1, m2)
    mask = cv2.medianBlur(mask, 5)
    return mask

def find_orange_centroid(bgr, roi=None):
    if roi is not None:
        H, W = bgr.shape[:2]
        x0 = max(0, min(int(roi[0]), W-1))
        y0 = max(0, min(int(roi[1]), H-1))
        x1 = max(0, min(int(roi[2]), W))
        y1 = max(0, min(int(roi[3]), H))

        if x1 <= x0 or y1 <= y0:
            return None

        bgr_roi = bgr[y0:y1, x0:x1]
        if bgr_roi.size == 0:
            return None
        mask = hsv_orange_mask(bgr_roi)
        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best=None; best_s=1e9; best_xy=None
        for c in cnts:
            a=cv2.contourArea(c)
            if a<20 or a>6000: continue
            (x,y), r = cv2.minEnclosingCircle(c)
            circ = (cv2.contourArea(c) / (math.pi*r*r+1e-6))
            s = abs(1-circ) + 0.0001*a
            if s<best_s:
                best_s=s; best=(x,y); best_xy=(x0+x, y0+y)
        return best_xy
    else:
        mask=hsv_orange_mask(bgr)
        cnts,_=cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return None
        best=None; best_s=1e9; best_xy=None
        for c in cnts:
            a=cv2.contourArea(c)
            if a<20 or a>6000: continue
            (x,y), r = cv2.minEnclosingCircle(c)
            circ = (cv2.contourArea(c) / (math.pi*r*r+1e-6))
            s = abs(1-circ) + 0.0001*a
            if s<best_s:
                best_s=s; best=(x,y); best_xy=(x, y)
        return best_xy

def make_kalman(dt=1/24.0, accel_var=4000.0):
    # Constant-velocity Kalman on (x,y,vx,vy)
    kf = cv2.KalmanFilter(4,2)
    kf.transitionMatrix = np.array([[1,0,dt,0],
                                    [0,1,0,dt],
                                    [0,0,1,0 ],
                                    [0,0,0,1 ]], np.float32)
    kf.measurementMatrix = np.array([[1,0,0,0],
                                     [0,1,0,0]], np.float32)
    kf.processNoiseCov  = np.array([[dt**4/4,0,dt**3/2,0],
                                    [0,dt**4/4,0,dt**3/2],
                                    [dt**3/2,0,dt**2,0],
                                    [0,dt**3/2,0,dt**2]], np.float32) * (accel_var)
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32)*20.0
    kf.errorCovPost = np.eye(4, dtype=np.float32)*1e2
    return kf

def clip_roi(x0,y0,x1,y1, W,H):
    x0=max(0,x0); y0=max(0,y0); x1=min(W,x1); y1=min(H,y1)
    if x1<=x0: x1=min(W, x0+1)
    if y1<=y0: y1=min(H, y0+1)
    return (int(x0),int(y0),int(x1),int(y1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True, help="input clip (.mp4)")
    ap.add_argument("--out_csv", required=True, help="output CSV path")
    ap.add_argument("--yolo_conf", type=float, default=0.12)
    ap.add_argument("--roi_pad", type=int, default=220)
    ap.add_argument("--roi_pad_max", type=int, default=560)
    ap.add_argument("--max_miss", type=int, default=45)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.inp)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    model = YOLO('yolov8n.pt')  # classes: 32=ball, 0=person

    rows = []
    frame=0

    # Kalman
    kf = make_kalman(1.0/fps)

    # Optical flow buffers
    prev_gray = None
    pts = None

    last_xy = None
    miss = 0
    pad = args.roi_pad

    while True:
        ok, bgr = cap.read()
        if not ok: break
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        t = frame / fps

        # ROI around last_xy when available
        if last_xy is not None:
            cx,cy = last_xy
            x0,y0 = int(cx-pad), int(cy-pad)
            x1,y1 = int(cx+pad), int(cy+pad)
            roi = clip_roi(x0,y0,x1,y1,W,H)
        else:
            roi = None

        # 1) YOLO (ball + persons in ROI if present)
        yolo_xy = None
        persons = []
        try:
            y_inp = bgr if roi is None else bgr[roi[1]:roi[3], roi[0]:roi[2]]
            res = model.predict(y_inp, verbose=False, conf=args.yolo_conf, classes=[32,0])
            if res and len(res[0].boxes):
                for b in res[0].boxes:
                    cls = int(b.cls.item())
                    # force pure Python floats
                    x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].cpu().tolist()]

                    if roi is not None:
                        x1 += roi[0]; x2 += roi[0]; y1 += roi[1]; y2 += roi[1]

                    if cls == 32:  # ball
                        area = (x2 - x1) * (y2 - y1)
                        if yolo_xy is None or area < yolo_xy[2]:
                            yolo_xy = ( (x1 + x2) / 2.0, (y1 + y2) / 2.0, area )
                    elif cls == 0:  # person
                        w = float(x2 - x1)
                        h = float(y2 - y1)
                        persons.append({
                            'id': -1,
                            'x': float((x1 + x2) / 2.0),
                            'y': float((y1 + y2) / 2.0),
                            'w': w,
                            'h': h
                        })
        except Exception:
            pass

        # 2) Color fallback (only when YOLO missed)
        if yolo_xy is None:
            cxy = find_orange_centroid(bgr, roi)
            if cxy is not None:
                yolo_xy = (float(cxy[0]), float(cxy[1]), 400.0)

        # 3) Optical flow fallback (track a tiny patch around last_xy)
        flow_xy = None
        if yolo_xy is None and prev_gray is not None and last_xy is not None:
            p0 = np.array([[last_xy]], dtype=np.float32)
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, winSize=(21,21), maxLevel=3,
                                                   criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 20, 0.03))
            if st is not None and st[0,0]==1:
                flow_xy = (float(p1[0,0,0]), float(p1[0,0,1]))

        # 4) Choose measurement
        meas = None
        meas_src = ''
        if yolo_xy is not None:
            meas = np.array([[yolo_xy[0]],[yolo_xy[1]]], dtype=np.float32); meas_src='yolo_orange'
        elif flow_xy is not None:
            meas = np.array([[flow_xy[0]],[flow_xy[1]]], dtype=np.float32); meas_src='flow'
        else:
            meas = None

        # 5) Kalman predict
        pred = kf.predict()
        px, py = float(pred[0,0]), float(pred[1,0])

        # 6) Update or coast
        if meas is not None:
            kf.correct(meas)
            cx, cy = float(kf.statePost[0,0]), float(kf.statePost[1,0])
            last_xy = (cx,cy)
            miss = 0
            pad = max(args.roi_pad, pad*0.85)  # tighten back towards base
        else:
            # coast on prediction
            cx,cy = px,py
            last_xy = (cx,cy)
            miss += 1
            pad = min(args.roi_pad_max, int(pad*1.15))
            # periodic global re-scan if we've been missing too long
            if miss>args.max_miss and (frame % int(fps//2 or 12)==0):
                try:
                    res = model.predict(bgr, verbose=False, conf=args.yolo_conf, classes=[32])
                    if res and len(res[0].boxes):
                        # pick smallest
                        boxes = res[0].boxes.xyxy.cpu().numpy()
                        areas = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
                        i = int(np.argmin(areas))
                        x1,y1,x2,y2 = boxes[i]
                        cx,cy = float((x1+x2)/2), float((y1+y2)/2)
                        kf.correct(np.array([[cx],[cy]], np.float32))
                        last_xy=(cx,cy); miss=0; pad=args.roi_pad
                        meas_src='yolo_rescue'
                except Exception:
                    pass

        conf = 0.0
        if meas_src == 'yolo_orange':
            conf = 1.0
        elif meas_src == 'yolo_rescue':
            conf = 0.9
        elif meas_src == 'flow':
            conf = 0.5

        if roi is not None:
            roi_x0, roi_y0, roi_x1, roi_y1 = roi
        else:
            roi_x0, roi_y0, roi_x1, roi_y1 = 0, 0, W, H
        roi_w = int(roi_x1 - roi_x0)
        roi_h = int(roi_y1 - roi_y0)

        persons_slim = []
        if persons:
            for p in persons:
                persons_slim.append({
                    'id': int(p.get('id', -1)),
                    'x': float(p.get('x', 0.0)),
                    'y': float(p.get('y', 0.0)),
                    'w': float(p.get('w', 0.0)),
                    'h': float(p.get('h', 0.0))
                })
        persons_json = json.dumps(persons_slim, separators=(',',':')) if persons_slim else ""

        rows.append({
            'frame': int(frame),
            'time': float(t),
            'cx': float(cx) if cx is not None else float('nan'),
            'cy': float(cy) if cy is not None else float('nan'),
            'conf': float(conf),
            'roi_x': int(roi_x0),
            'roi_y': int(roi_y0),
            'roi_w': int(roi_w),
            'roi_h': int(roi_h),
            'miss': int(miss),
            'pad': float(pad),
            'src': meas_src,
            'persons_json': persons_json
        })

        prev_gray = gray
        frame += 1

    cap.release()
    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    header = ['frame','time','cx','cy','conf','roi_x','roi_y','roi_w','roi_h','miss','pad','src','persons_json']
    with open(args.out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(header)
        for r in rows:
            w.writerow([
                r['frame'], r['time'], r['cx'], r['cy'], r['conf'],
                r['roi_x'], r['roi_y'], r['roi_w'], r['roi_h'],
                r['miss'], r['pad'], r['src'], r['persons_json']
            ])
    print(f"Saved {args.out_csv}")

if __name__ == "__main__":
    main()

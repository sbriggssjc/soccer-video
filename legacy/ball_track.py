import sys, csv, os, cv2, numpy as np

# Usage: ball_track.py <in> <out_csv> [weights_or_NONE] [conf]
args = sys.argv[1:]
if len(args) < 2:
    raise SystemExit("usage: ball_track.py <in> <out_csv> [weights_or_NONE] [conf]")
in_path, out_csv = args[0], args[1]
weights_arg = args[2] if len(args) >= 3 else "NONE"
try: conf_min = float(args[3]) if len(args) >= 4 else 0.35
except: conf_min = 0.35

cap = cv2.VideoCapture(in_path)
if not cap.isOpened(): raise SystemExit(f"Cannot open {in_path}")
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

def write_csv(rows):
    with open(out_csv,"w",newline="") as f:
        wr=csv.writer(f); wr.writerow(["n","cx","cy","conf","w","h","fps"])
        for r in rows: wr.writerow(r)

# ---------- Prefer YOLO if weights exist ----------
USE_YOLO = False
if weights_arg and weights_arg.upper()!="NONE" and os.path.exists(weights_arg):
    try:
        from ultralytics import YOLO
        model = YOLO(weights_arg)
        USE_YOLO = True
    except Exception:
        USE_YOLO = False

rows=[]

if USE_YOLO:
    n = 0; last = None
    imgsz = max(640, ((max(w, h) + 31) // 32) * 32)
    for r in model.predict(source=in_path, conf=conf_min, stream=True, imgsz=imgsz, verbose=False):
        cx = cy = None; conf = 0.0
        if r.boxes is not None and len(r.boxes)>0:
            b=r.boxes; i=int(b.conf.argmax().item())
            xyxy=b.xyxy[i].tolist(); conf=float(b.conf[i].item())
            cx=0.5*(xyxy[0]+xyxy[2]); cy=0.5*(xyxy[1]+xyxy[3])
        if cx is None:
            if last is not None: cx,cy=last
            else: cx,cy=w/2.0,h/2.0
            conf=0.0
        else:
            last=(cx,cy)
        rows.append([n, f"{np.clip(cx,0,w-1):.4f}", f"{np.clip(cy,0,h-1):.4f}", f"{conf:.4f}", w, h, fps]); n+=1
    write_csv(rows); sys.exit(0)

# ---------- Fallback: LK optical flow + Kalman + periodic global scan ----------
# Kalman (cx,cy,vx,vy)
kf = cv2.KalmanFilter(4,2)
kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float32)
dt = 1.0/max(fps,1)
kf.transitionMatrix  = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], dtype=np.float32)
kf.processNoiseCov   = np.diag([1e-2,1e-2,5e-1,5e-1]).astype(np.float32)
kf.measurementNoiseCov = np.diag([2.0,2.0]).astype(np.float32)
kf.errorCovPost = np.eye(4, dtype=np.float32)
kf.statePost = np.array([[w/2],[h/2],[0],[0]], dtype=np.float32)

feature_params = dict(maxCorners=150, qualityLevel=0.03, minDistance=7, blockSize=7)
lk_params      = dict(winSize=(21,21), maxLevel=3,
                      criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 30, 0.01))

def global_reacquire(frame_gray):
    # tile scan: pick tile with highest median gradient ? centroid
    H,W = frame_gray.shape
    tx,ty = 8,4
    best=None; best_val=-1; best_box=None
    for j in range(ty):
        for i in range(tx):
            x0 = int(i*W/tx); x1 = int((i+1)*W/tx)
            y0 = int(j*H/ty); y1 = int((j+1)*H/ty)
            roi = frame_gray[y0:y1, x0:x1]
            val = np.median(cv2.Laplacian(roi, cv2.CV_16S).astype(np.float32))
            if val>best_val:
                best_val = val; best=(x0+(x1-x0)//2, y0+(y1-y0)//2); best_box=(x0,y0,x1,y1)
    return best, best_box

ret, prev = cap.read()
if not ret: write_csv(rows); sys.exit(0)
prevG = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

# seed features around center
p0 = cv2.goodFeaturesToTrack(prevG, mask=None, **feature_params)
last_meas = (w/2.0, h/2.0)
conf = 0.2
n = 0; missing = 0

while True:
    ok, frame = cap.read()
    if not ok: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # predict
    pred = kf.predict()
    cxp, cyp = float(pred[0]), float(pred[1])

    use_meas = False; meas=(cxp,cyp); conf_inst=0.0

    if p0 is not None and len(p0)>0:
        p1, st, err = cv2.calcOpticalFlowPyrLK(prevG, gray, p0, None, **lk_params)
        good_new = p1[st==1] if p1 is not None else np.empty((0,2))
        good_old = p0[st==1] if p0 is not None else np.empty((0,2))
        if len(good_new) >= 6:
            # cluster by proximity around predicted point
            d = np.linalg.norm(good_new - np.array([[cxp,cyp]]), axis=1)
            idx = np.argsort(d)[:int(max(6,0.3*len(good_new)))]
            pts = good_new[idx]
            meas = (float(np.median(pts[:,0])), float(np.median(pts[:,1])))
            use_meas = True
            conf_inst = float(max(0.1, min(0.9, 0.9 - np.median(d[idx]) / (0.20*max(w,h)+1e-6))))
        else:
            missing += 1
    else:
        missing += 1

    # periodic global reacquire if too many misses or low confidence
    if missing >= 4 or conf_inst < 0.18:
        cand, box = global_reacquire(gray)
        if cand is not None:
            meas = (float(cand[0]), float(cand[1]))
            use_meas = True
            conf_inst = 0.3
            missing = 0
            # refresh features around candidate
            x0,y0,x1,y1 = box
            mask = np.zeros_like(gray); mask[y0:y1, x0:x1] = 255
            p0 = cv2.goodFeaturesToTrack(gray, mask=mask, **feature_params)

    # update Kalman
    if use_meas:
        kf.correct(np.array([[meas[0]],[meas[1]]], dtype=np.float32))
        last_meas = meas
    else:
        # keep last measurement as fallback
        meas = last_meas
        conf_inst = max(0.05, conf_inst*0.5)

    # save row
    cx_out = np.clip(meas[0], 0, w-1)
    cy_out = np.clip(meas[1], 0, h-1)
    rows.append([n, f"{cx_out:.4f}", f"{cy_out:.4f}", f"{conf_inst:.4f}", w, h, fps])

    # prepare next
    prevG = gray.copy()
    # refresh features each frame near current meas
    r=48
    x0=int(max(0, cx_out-r)); y0=int(max(0, cy_out-r))
    x1=int(min(w, cx_out+r)); y1=int(min(h, cy_out+r))
    mask = np.zeros_like(gray); mask[y0:y1, x0:x1] = 255
    p0 = cv2.goodFeaturesToTrack(gray, mask=mask, **feature_params)

    n+=1

write_csv(rows)

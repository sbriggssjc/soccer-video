import sys, csv, math, cv2, numpy as np

if len(sys.argv)<3: raise SystemExit("usage: track_ball_plus.py <in> <out_csv>")
in_path, out_csv = sys.argv[1], sys.argv[2]

cap = cv2.VideoCapture(in_path)
if not cap.isOpened(): raise SystemExit(f"Cannot open {in_path}")
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

def clamp(v, lo, hi): return max(lo, min(hi, v))

# --- helpers ---
def small_ball_detect(gray):
    # sharpen then OTSU to find small, bright-ish blobs; choose roundest
    g = cv2.GaussianBlur(gray,(5,5),0)
    hp = cv2.addWeighted(gray, 1.7, g, -0.7, 0)
    th = cv2.threshold(hp,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    th = cv2.medianBlur(th,3)
    cnts,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best=None; best_score=0
    for c in cnts:
        a = cv2.contourArea(c)
        if 5 <= a <= 1600:
            (x,y), r = cv2.minEnclosingCircle(c)
            if 3 <= r <= 26:
                score = float(a)/(r*r+1e-6)
                if score > best_score: best_score, best = score, (x,y,int(max(20,min(64,r*3))))
    return best

# Kalman (x,y,vx,vy)
KF = cv2.KalmanFilter(4,2)
dt = 1.0/max(1.0,fps)
KF.transitionMatrix = np.array([[1,0,dt,0],
                                [0,1,0,dt],
                                [0,0,1,0],
                                [0,0,0,1]], np.float32)
KF.measurementMatrix = np.array([[1,0,0,0],
                                 [0,1,0,0]], np.float32)
KF.processNoiseCov  = np.eye(4, dtype=np.float32)*1e-2
KF.measurementNoiseCov = np.eye(2, dtype=np.float32)*6e-1
KF.errorCovPost = np.eye(4, dtype=np.float32)
KF.statePost   = np.array([W/2,H/2,0,0], dtype=np.float32)

# init
ok, f0 = cap.read()
if not ok: raise SystemExit("empty video")
g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
seed = small_ball_detect(g0)
if seed is None: seed = (W/2.0, H/2.0, 48)
x0,y0,box = seed
x0 = clamp(x0,0,W-1); y0 = clamp(y0,0,H-1)
w0 = h0 = max(28, min(72, box))
bbox = (float(x0-w0/2), float(y0-h0/2), float(w0), float(h0))
tracker = cv2.legacy.TrackerCSRT_create(); tracker.init(f0, bbox)

prev_gray = g0
last_tmpl = cv2.getRectSubPix(g0, (int(w0),int(h0)), (x0,y0))
last_pt   = (x0,y0)
lost_ctr  = 0

def ncc_redetect(gray, pred, win=64, sz=(40,40)):
    # fast template match near predicted location
    px,py = int(round(pred[0])), int(round(pred[1]))
    x0 = clamp(px-win, 0, W-1); y0 = clamp(py-win, 0, H-1)
    x1 = clamp(px+win, 0, W-1); y1 = clamp(py+win, 0, H-1)
    if x1<=x0 or y1<=y0 or last_tmpl is None: return None
    roi = gray[y0:y1, x0:x1]
    tmpl = cv2.resize(last_tmpl, sz)
    if roi.shape[0]<sz[1] or roi.shape[1]<sz[0]: return None
    res = cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED)
    _, maxv, _, maxp = cv2.minMaxLoc(res)
    if maxv < 0.45: return None
    cx = x0 + maxp[0] + sz[0]/2.0; cy = y0 + maxp[1] + sz[1]/2.0
    return (cx,cy, maxv)

rows=[]; n=0; prev_xy=(x0,y0)
while True:
    ok, fr = cap.read()
    if not ok: break
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)

    # predict
    pred = KF.predict().ravel()
    pxy = (float(pred[0]), float(pred[1]))

    # CSRT
    ok_t, bb = tracker.update(fr)
    meas=None; conf=0.0
    if ok_t:
        x,y,w,h = bb
        meas = (x+w/2.0, y+h/2.0); conf = 0.6

    # LK refine
    base = meas if meas is not None else pxy
    p0 = np.array([[base]], dtype=np.float32)
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None,
                                           winSize=(21,21), maxLevel=2,
                                           criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 20, 0.03))
    if st is not None and st[0][0]==1 and (err is None or err[0][0] < 40):
        lk = (float(p1[0,0,0]), float(p1[0,0,1]))
        if meas is None: meas = lk; conf = 0.5
        else:
            meas = (0.6*meas[0] + 0.4*lk[0], 0.6*meas[1] + 0.4*lk[1]); conf = max(conf, 0.6)

    # NCC re-detect when low conf
    if conf < 0.45:
        pr = ncc_redetect(gray, pxy, win=72, sz=(44,44))
        if pr is not None:
            meas = (pr[0], pr[1]); conf = max(conf, 0.55)

    # re-init CSRT when we have a good measurement after being low/confused
    if conf >= 0.55:
        lost_ctr = 0
        sx = int(clamp(meas[0], 10, W-10)); sy = int(clamp(meas[1], 10, H-10))
        last_tmpl = cv2.getRectSubPix(gray, (int(w0),int(h0)), (sx,sy))
        bbox = (float(sx-w0/2), float(sy-h0/2), float(w0), float(h0))
        tracker = cv2.legacy.TrackerCSRT_create(); tracker.init(fr, bbox)
    else:
        lost_ctr += 1

    # Kalman update / choose output
    if meas is None:
        mx,my = pxy; conf = 0.20
    else:
        KF.correct(np.array([[np.float32(meas[0])],[np.float32(meas[1])]], dtype=np.float32))
        mx,my = meas

    mx = clamp(mx,0,W-1); my = clamp(my,0,H-1)
    speed = math.hypot(mx - prev_xy[0], my - prev_xy[1]) * (fps)   # px/sec
    rows.append([n, f"{mx:.4f}", f"{my:.4f}", f"{conf:.4f}", f"{speed:.3f}", W, H, fps])
    prev_xy = (mx,my)
    prev_gray = gray
    n += 1

with open(out_csv,"w",newline="") as f:
    wr = csv.writer(f); wr.writerow(["n","cx","cy","conf","speed","w","h","fps"])
    for r in rows: wr.writerow(r)

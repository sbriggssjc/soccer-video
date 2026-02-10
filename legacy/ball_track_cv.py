import sys, csv, cv2, numpy as np

# usage: ball_track_cv.py <in> <out_csv>
if len(sys.argv) < 3: raise SystemExit("usage: ball_track_cv.py <in> <out_csv>")
in_path, out_csv = sys.argv[1], sys.argv[2]

cap = cv2.VideoCapture(in_path)
if not cap.isOpened(): raise SystemExit(f"Cannot open {in_path}")
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

# --- Kalman (x,y,vx,vy) ---
kf = cv2.KalmanFilter(4, 2)
dt = 1.0 / float(fps)
# state: [x, y, vx, vy]^T
kf.transitionMatrix = np.array([[1,0,dt,0],
                                [0,1,0,dt],
                                [0,0,1,0],
                                [0,0,0,1]], dtype=np.float32)
kf.measurementMatrix = np.array([[1,0,0,0],
                                 [0,1,0,0]], dtype=np.float32)
kf.processNoiseCov  = np.eye(4, dtype=np.float32) * 1e-2
kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
kf.errorCovPost = np.eye(4, dtype=np.float32)

# helper
def clamp(v, lo, hi): return max(lo, min(hi, v))

# Seed by small round-ish moving blob
def initial_detect(gray):
    # highpass-ish to emphasize small bright details
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    hp   = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    # threshold by local contrast
    thr  = cv2.threshold(hp, 0, 255, cv2.THRESH_OTSU)[1]
    thr  = cv2.medianBlur(thr, 3)
    cnts,_ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best=None; best_score=0
    for c in cnts:
        area = cv2.contourArea(c)
        if 6 <= area <= 1200:
            (x,y),r = cv2.minEnclosingCircle(c)
            if r < 24:  # ball small
                # roundness score
                score = float(area) / float(r*r + 1e-6)
                if score > best_score: best_score, best = score, (x,y)
    return best

# LK params
lk_win = (21,21)
lk_crit = (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 20, 0.03)

ok, prev = cap.read()
if not ok: raise SystemExit("empty video")
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

seed = initial_detect(prev_gray)
if seed is None: seed = (w/2.0, h/2.0)
kf.statePost = np.array([[seed[0]],[seed[1]],[0.0],[0.0]], dtype=np.float32)

rows=[]
n=0
last_xy = seed
miss = 0

while True:
    ok, frame = cap.read()
    if not ok: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # predict
    pred = kf.predict()
    px, py = float(pred[0]), float(pred[1])

    # LK track around last_xy (or predicted)
    x0, y0 = last_xy if last_xy is not None else (px, py)
    x0 = clamp(x0, 8, w-9); y0 = clamp(y0, 8, h-9)
    p0 = np.array([[[x0, y0]]], dtype=np.float32)
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, winSize=lk_win, maxLevel=2, criteria=lk_crit)
    measured = None
    if st is not None and st[0][0]==1 and err is not None and err[0][0] < 30:
        mx, my = float(p1[0,0,0]), float(p1[0,0,1])
        measured = (mx, my)

    # re-detect if we lost LK for a few frames
    if measured is None:
        miss += 1
        if miss >= 5:
            re = initial_detect(gray)
            if re is not None:
                measured = re
                miss = 0
    else:
        miss = 0

    if measured is None:
        # no measurement -> just use prediction, low confidence
        mx, my = px, py
        conf = 0.10
        meas = np.array([[np.float32(mx)], [np.float32(my)]])
        kf.correct(meas)
    else:
        # fuse measurement
        mx, my = measured
        conf = 0.80
        meas = np.array([[np.float32(mx)], [np.float32(my)]])
        kf.correct(meas)

    mx = clamp(mx, 0, w-1); my = clamp(my, 0, h-1)
    rows.append([n, f"{mx:.4f}", f"{my:.4f}", f"{conf:.4f}", w, h, fps])
    last_xy = (mx, my)
    prev_gray = gray
    n += 1

with open(out_csv, "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["n","cx","cy","conf","w","h","fps"])
    for r in rows: wr.writerow(r)

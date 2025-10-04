import sys, csv, cv2, numpy as np

# usage: ball_track_hq.py <in> <out_csv>
if len(sys.argv) < 3: raise SystemExit("usage: ball_track_hq.py <in> <out_csv>")
in_path, out_csv = sys.argv[1], sys.argv[2]

cap = cv2.VideoCapture(in_path)
if not cap.isOpened(): raise SystemExit(f"Cannot open {in_path}")
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

def clamp(v, lo, hi): return max(lo, min(hi, v))

# LK params
lk_win = (21,21); lk_crit = (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 20, 0.03)

# quick small-round re-detector (used at init & when lost)
def small_round_detect(gray):
    # emphasize small high-contrast blobs
    blur = cv2.GaussianBlur(gray,(5,5),0)
    hp   = cv2.addWeighted(gray,1.5,blur,-0.5,0)
    thr  = cv2.threshold(hp,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    thr  = cv2.medianBlur(thr,3)
    cnts,_ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best=None; best_score=0
    for c in cnts:
        area = cv2.contourArea(c)
        if 6 <= area <= 1500:
            (x,y), r = cv2.minEnclosingCircle(c)
            if r < 26:
                score = float(area) / float(r*r + 1e-6)  # roundness preference
                if score > best_score: best_score, best = score, (x,y, int(max(12, min(36, r*3))))
    return best  # (x,y, box)

ok, frame0 = cap.read()
if not ok: raise SystemExit("empty video")
gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)

seed = small_round_detect(gray0)
if seed is None:
    # fall back: search center region
    seed = (W/2.0, H/2.0, 48)
x0,y0,box = seed
x0 = clamp(x0,0,W-1); y0 = clamp(y0,0,H-1)

# CSRT tracker box around seed
w0 = h0 = max(32, min(72, box))
bbox = (float(x0 - w0/2), float(y0 - h0/2), float(w0), float(h0))
tracker = cv2.legacy.TrackerCSRT_create()
tracker.init(frame0, bbox)

rows=[]
n=0
last_xy=(x0,y0)
lost_frames=0

prev_gray = gray0

while True:
    ok, frame = cap.read()
    if not ok: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1) Try CSRT
    ok_csrt, bb = tracker.update(frame)
    measured=None; csrt_conf = 0.0
    if ok_csrt:
        x,y,w,h = bb
        cx = x + w/2.0; cy = y + h/2.0
        measured = (cx,cy); csrt_conf = 0.7

    # 2) Refine with LK around last (or CSRT)
    base = measured if measured is not None else last_xy
    p0 = np.array([[[clamp(base[0],8,W-9), clamp(base[1],8,H-9)]]], dtype=np.float32)
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, winSize=lk_win, maxLevel=2, criteria=lk_crit)
    if st is not None and st[0][0]==1 and err is not None and err[0][0] < 30:
        lkx,lky = float(p1[0,0,0]), float(p1[0,0,1])
        if measured is None: measured=(lkx,lky); csrt_conf=0.5
        else:
            # fuse CSRT+LK
            measured = (0.6*measured[0] + 0.4*lkx, 0.6*measured[1] + 0.4*lky)
            csrt_conf = max(csrt_conf, 0.6)

    # 3) If confidence low for a few frames → re-detect & re-init CSRT
    if measured is None:
        lost_frames += 1
        if lost_frames >= 5:
            red = small_round_detect(gray)
            if red is not None:
                x,y,box = red
                w = h = max(32, min(72, box))
                bbox = (float(x - w/2), float(y - h/2), float(w), float(h))
                tracker = cv2.legacy.TrackerCSRT_create()
                tracker.init(frame, bbox)
                measured = (x,y); csrt_conf = 0.6
                lost_frames = 0
    else:
        lost_frames = 0

    if measured is None:
        # still nothing → hold last
        mx,my = last_xy; conf = 0.10
    else:
        mx,my = measured; conf = csrt_conf

    mx = clamp(mx,0,W-1); my = clamp(my,0,H-1)
    rows.append([n, f"{mx:.4f}", f"{my:.4f}", f"{conf:.4f}", W, H, fps])
    last_xy=(mx,my)
    prev_gray = gray
    n += 1

with open(out_csv, "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["n","cx","cy","conf","w","h","fps"])
    for r in rows: wr.writerow(r)

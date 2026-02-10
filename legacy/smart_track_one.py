import sys, os, csv
import numpy as np, cv2, math, time


def hsv_orange_score(bgr):
    if bgr is None or bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    masks = []
    masks.append(cv2.inRange(hsv, (5, 90, 70), (25, 255, 255)))
    masks.append(cv2.inRange(hsv, (0, 80, 60), (5, 255, 255)))
    masks.append(cv2.inRange(hsv, (170, 80, 60), (180, 255, 255)))
    mask = masks[0]
    for m in masks[1:]:
        mask = cv2.bitwise_or(mask, m)
    return float(np.mean(mask > 0))


def box_size_score(w, h, expect_min=10, expect_max=60):
    s = max(w, h)
    if s < expect_min:
        return s / expect_min * 0.6
    if s > expect_max:
        return max(0.0, 1.0 - (s - expect_max) / (2.0 * expect_max))
    m = 0.5 * (expect_min + expect_max)
    return max(0.0, 1.0 - abs(s - m) / (m))


def motion_score(cx, cy, px, py):
    if px is None or py is None:
        return 0.5
    d = math.hypot(cx - px, cy - py)
    return max(0.0, 1.0 - d / 140.0)


class RunningVar:
    def __init__(self, k=20):
        self.k = k
        self.buf = []

    def push(self, v):
        self.buf.append(float(v))
        if len(self.buf) > self.k:
            self.buf.pop(0)

    def var(self):
        if len(self.buf) < 3:
            return 1e3
        a = np.array(self.buf)
        return float(np.var(a))

# --- argv / io ---
args=sys.argv[1:]
if len(args)<2: raise SystemExit("usage: smart_track_one.py <in> <out_csv> [weights_or_NONE] [conf]")
in_path,out_csv=args[0],args[1]
weights=args[2] if len(args)>=3 else "NONE"
try: conf_min=float(args[3]) if len(args)>=4 else 0.35
except: conf_min=0.35

cap=cv2.VideoCapture(in_path)
if not cap.isOpened(): raise SystemExit("Cannot open "+in_path)
W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS=cap.get(cv2.CAP_PROP_FPS) or 24.0; dt=1.0/max(FPS,1.0)

# --- YOLO (optional) ---
use_yolo=False
if weights and weights.upper()!="NONE" and os.path.exists(weights):
    try:
        from ultralytics import YOLO
        yolo=YOLO(weights); use_yolo=True
    except Exception as e:
        print("YOLO load failed:", e)

def clamp(v,a,b): return max(a,min(b,v))

# --- color masks: tight + wide (lost-mode) ---
RED1=((0,120,80),(8,255,255)); RED2=((165,120,80),(179,255,255)); ORNG=((8,120,80),(20,255,255))
RED1W=((0,100,70),(8,255,255)); RED2W=((165,100,70),(179,255,255)); ORNGW=((8,100,70),(22,255,255))
GREEN=((35,25,40),(95,255,255))

def mask_red(img, wide=False):
    r1,r2,orng = (RED1W,RED2W,ORNGW) if wide else (RED1,RED2,ORNG)
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    m=cv2.inRange(hsv,np.array(r1[0]),np.array(r1[1]))
    m|=cv2.inRange(hsv,np.array(r2[0]),np.array(r2[1]))
    m|=cv2.inRange(hsv,np.array(orng[0]),np.array(orng[1]))
    g=cv2.inRange(hsv,np.array(GREEN[0]),np.array(GREEN[1]))
    m=cv2.bitwise_and(m, cv2.bitwise_not(g))
    m=cv2.medianBlur(m,5)
    m=cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3,3),np.uint8), iterations=1)
    return m

# --- helper: redness score near (cx,cy) ---
def redness_at(img, cx, cy, rad=8):
    x0=max(0,int(cx-rad)); x1=min(img.shape[1], int(cx+rad))
    y0=max(0,int(cy-rad)); y1=min(img.shape[0], int(cy+rad))
    if x1<=x0 or y1<=y0: return 0.0
    patch=img[y0:y1,x0:x1]
    b,g,r=cv2.split(patch)
    R=float(r.mean()); G=float(g.mean()); B=float(b.mean())
    return max(0.0, (R - max(G,B)) / (R+G+B + 1e-3))

# --- candidates on grass only, tighter shape ---
def color_candidates(bgr, roi=None, wide=False):
    Hh,Wh=bgr.shape[0],bgr.shape[1]
    y_floor = int(0.33*Hh)
    if roi is None: x0=y0=0; x1=Wh; y1=Hh; patch=bgr
    else: x0,y0,x1,y1=roi; patch=bgr[y0:y1, x0:x1]
    m=mask_red(patch, wide=wide)
    cnts,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    out=[]
    if 0<len(cnts)<=100:
        for c in cnts:
            area=cv2.contourArea(c)
            if area<40 or area>2000: continue
            peri=cv2.arcLength(c,True);  circ=(4*np.pi*area)/(peri*peri) if peri>0 else 0
            if circ<0.74: continue
            (x,y),r=cv2.minEnclosingCircle(c)
            gy=int(y0+y)
            if gy<y_floor or gy>Hh-4: continue
            out.append((x0+x, y0+y, float(area*circ)))
    return out

def hough_candidates(bgr, roi=None, wide=False):
    Hh,Wh=bgr.shape[0],bgr.shape[1]
    y_floor = int(0.33*Hh)
    if roi is None: x0=y0=0; x1=Wh; y1=Hh; patch=bgr
    else: x0,y0,x1,y1=roi; patch=bgr[y0:y1, x0:x1]
    m=mask_red(patch, wide=wide)
    nz=cv2.countNonZero(m)
    if nz==0 or nz>20000: return []
    gry=cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY)
    gry=cv2.bitwise_and(gry, gry, mask=m)
    gry=cv2.GaussianBlur(gry,(7,7),1.4)
    param2 = 26 if not wide else 22
    cir=cv2.HoughCircles(gry, cv2.HOUGH_GRADIENT, dp=1.3, minDist=24,
                         param1=140, param2=param2, minRadius=6, maxRadius=34)
    out=[]
    if cir is not None:
        for x,y,r in np.uint16(np.around(cir))[0,:]:
            gy=int(y0+y)
            if gy<y_floor or gy>Hh-4: continue
            out.append((x0+float(x), y0+float(y), float(r*r)))
    return out

# --- Kalman (constant-velocity) ---
class KCV:
    def __init__(self, dt, q=14.0, r=16.0):
        self.x=np.array([[W/2],[H/2],[0],[0]], dtype=np.float32)
        self.F=np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], dtype=np.float32)
        self.H=np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float32)
        self.P=np.eye(4, dtype=np.float32)*120.0
        self.Q=np.array([[dt**4/4,0,dt**3/2,0],
                         [0,dt**4/4,0,dt**3/2],
                         [dt**3/2,0,dt**2,0],
                         [0,dt**3/2,0,dt**2]], dtype=np.float32) * q
        self.R=np.eye(2, dtype=np.float32)*r
    def predict(self):
        self.x=self.F@self.x
        self.P=self.F@self.P@self.F.T + self.Q
    def update(self, z):
        z=np.array(z,dtype=np.float32).reshape(2,1)
        y=z - (self.H@self.x)
        S=self.H@self.P@self.H.T + self.R
        K=self.P@self.H.T@np.linalg.inv(S)
        self.x=self.x + K@y
        I=np.eye(4,dtype=np.float32)
        self.P=(I-K@self.H)@self.P

kf=KCV(dt)

def roi_from(cx,cy, lost):
    rad = int(min(280, 140 + 40*min(lost,7)))
    x0=int(clamp((cx if cx is not None else W/2)-rad,0,W-1)); x1=int(clamp(x0+2*rad,1,W))
    y0=int(clamp((cy if cy is not None else H/2)-rad,0,H-1)); y1=int(clamp(y0+2*rad,1,H))
    return (x0,y0,x1,y1)

def score_and_pick(frame, props, px,py,vx,vy, roi=None):
    best=None; bestS=-1.0
    x0=y0=x1=y1=None
    if roi is not None: x0,y0,x1,y1 = roi
    for cx,cy,raws in props:
        yolo_c = raws if raws<=1.0 else 0.0
        color  = raws if raws> 1.0 else 0.0
        hough  = raws if raws> 1e3 else 0.0
        ex = px + vx*dt; ey = py + vy*dt
        d2 = (cx-ex)**2 + (cy-ey)**2
        mot = 1.0/(1.0 + d2/(140.0**2))
        red = redness_at(frame, cx, cy, 8)
        roi_bonus = 0.6 if (roi is not None and x0<=cx<=x1 and y0<=cy<=y1) else 0.0
        S = 2.0*min(1.0,yolo_c) + 0.8*math.log1p(color/250.0) + 0.55*min(1.0,hough/2500.0) \
            + 2.2*mot + 1.0*red + roi_bonus
        conf = max(min(1.0,yolo_c), 0.55*(color>0) + 0.35*(hough>0))
        if S>bestS:
            bestS=S; best=(cx,cy, conf)
    return best

def propose(frame, px, py, lost, n):
    roi = roi_from(px,py,lost)
    wide = lost >= 6
    props=[]
    props += color_candidates(frame, roi, wide=wide)
    props += hough_candidates(frame, roi, wide=wide)
    if lost>=6 and (n%2==0):
        props += color_candidates(frame, None, wide=True)
        props += hough_candidates(frame, None, wide=True)
    return roi, props

rows=[]; n=0; lost=999
prev_gray=None; prev_pt=None

# Scoring weights (tuneable)
W_DET, W_MO, W_SIZE, W_CLR, W_DYN = 0.55, 0.20, 0.10, 0.12, 0.03

# Track vertical dynamics to avoid flat-line decoys
cy_dyn = RunningVar(k=24)
last_choice = None


def select_best_detection_from_boxes(boxes, frame, offset=(0, 0), prev_pt=None, last_choice=None, cy_dyn=None):
    best = None
    ox, oy = offset
    px = None
    py = None
    if last_choice is not None:
        px, py = last_choice
    elif prev_pt is not None:
        px = float(prev_pt[0, 0, 0])
        py = float(prev_pt[0, 0, 1])
    for i in range(len(boxes)):
        conf_det = float(boxes.conf[i].item())
        x0, y0, x1, y1 = boxes.xyxy[i].cpu().numpy().astype(float)
        x0 += ox
        y0 += oy
        x1 += ox
        y1 += oy
        w = max(1.0, x1 - x0)
        h = max(1.0, y1 - y0)
        cx_cand = 0.5 * (x0 + x1)
        cy_cand = 0.5 * (y0 + y1)
        xi0 = int(max(0, min(W, x0)))
        yi0 = int(max(0, min(H, y0)))
        xi1 = int(max(0, min(W, x1)))
        yi1 = int(max(0, min(H, y1)))
        crop = frame[yi0:yi1, xi0:xi1] if (yi1 > yi0 and xi1 > xi0) else None
        clr = hsv_orange_score(crop) if crop is not None else 0.0
        sz = box_size_score(w, h, expect_min=10, expect_max=60)
        mo = motion_score(cx_cand, cy_cand, px, py)
        dyn_pen = 0.0
        if cy_dyn is not None and cy_dyn.var() < 2.0:
            mean_cy = sum(cy_dyn.buf) / len(cy_dyn.buf) if cy_dyn.buf else None
            if mean_cy is not None and abs(cy_cand - mean_cy) < 6.0:
                dyn_pen = 0.2
        score = (W_DET * conf_det + W_MO * mo + W_SIZE * sz + W_CLR * clr) - W_DYN * dyn_pen
        if best is None or score > best[0]:
            best = (score, cx_cand, cy_cand, conf_det)
    if best is not None and best[0] > 0.25:
        return best
    return None


lk_crit=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,0.03)
last_meas=None
bad_run = 0
REDETECT_EVERY = 1      # force YOLO every frame while bad
BAD_THRESH = 0.22       # confidence threshold
BAD_MAX = 5             # frames under threshold to consider "bad"
ROI_R = 140             # local ROI radius for YOLO when we have a prior
use_roi = True

while True:
    ok, frame = cap.read()
    if not ok: break
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if lost>0:
        kf.x[2]*=0.75; kf.x[3]*=0.75
    kf.predict()
    px,py,vx,vy = float(kf.x[0]),float(kf.x[1]),float(kf.x[2]),float(kf.x[3])

    det=None; conf=0.0

    have_prior = (prev_pt is not None)
    prior = (int(prev_pt[0,0,0]), int(prev_pt[0,0,1])) if have_prior else (W//2, H//2)

    need_force = (bad_run >= BAD_MAX)
    bad_mode = (bad_run > 0)
    do_yolo = False
    if use_yolo:
        if need_force or lost>0:
            do_yolo = True
        elif bad_mode:
            every = max(1, REDETECT_EVERY)
            do_yolo = ((n % every) == 0)
        else:
            do_yolo = (n % 2 == 0)

    if do_yolo:
        det_local=None; best_conf=0.0
        try:
            if use_roi and have_prior and not need_force:
                roi_x0 = max(0, prior[0]-ROI_R); roi_y0 = max(0, prior[1]-ROI_R)
                roi_x1 = min(W, prior[0]+ROI_R); roi_y1 = min(H, prior[1]+ROI_R)
                crop = frame[roi_y0:roi_y1, roi_x0:roi_x1]
                imgsz=max(640,((max(roi_x1-roi_x0,roi_y1-roi_y0)+31)//32)*32)
                rs = yolo.predict(source=crop, conf=conf_min*0.8, stream=False, imgsz=imgsz, verbose=False)
                if len(rs):
                    b=rs[0].boxes
                    if b is not None and len(b)>0:
                        pick = select_best_detection_from_boxes(
                            b,
                            frame,
                            offset=(roi_x0, roi_y0),
                            prev_pt=prev_pt,
                            last_choice=last_choice,
                            cy_dyn=cy_dyn,
                        )
                        if pick is not None:
                            det_local=(pick[1], pick[2])
                            best_conf=pick[3]
            if det_local is None:
                imgsz=max(640,((max(W,H)+31)//32)*32)
                rs = yolo.predict(source=frame, conf=conf_min*0.8, stream=False, imgsz=imgsz, verbose=False)
                if len(rs) and getattr(rs[0],"boxes",None) is not None and len(rs[0].boxes)>0:
                    b=rs[0].boxes
                    pick = select_best_detection_from_boxes(
                        b,
                        frame,
                        offset=(0, 0),
                        prev_pt=prev_pt,
                        last_choice=last_choice,
                        cy_dyn=cy_dyn,
                    )
                    if pick is not None:
                        det_local=(pick[1], pick[2])
                        best_conf=pick[3]
        except Exception:
            det_local=None
        if det_local is not None:
            det=det_local; conf=best_conf

    # Bootstrap small window 0..8
    if n<9:
        roi=(int(W*0.25), int(H*0.25), int(W*0.75), int(H*0.80))
        props  = color_candidates(frame, roi, wide=False)
        props += hough_candidates(frame, roi, wide=False)
        pick = score_and_pick(frame, props, px,py,vx,vy, roi=roi)
        if pick is not None: det=(pick[0],pick[1]); conf=pick[2]

    if det is None:
        roi, props = propose(frame, px, py, lost, n)
        pick = score_and_pick(frame, props, px,py,vx,vy, roi=roi)
        if pick is not None: det=(pick[0],pick[1]); conf=pick[2]

    if det is None and prev_gray is not None and prev_pt is not None:
        p1,st,_=cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pt, None,
                                         winSize=(21,21), maxLevel=3, criteria=lk_crit)
        if st is not None and int(st.ravel()[0])==1:
            p0r,st2,_=cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, None,
                                               winSize=(21,21), maxLevel=3, criteria=lk_crit)
            st2_ok=(st2 is not None and int(st2.ravel()[0])==1)
            fb=float(np.linalg.norm(prev_pt - p0r)) if st2_ok else 9e9
            if fb<2.0:
                x=float(p1[0,0,0]); y=float(p1[0,0,1])
                if 0<=x<W and 0<=y<H: det=(x,y); conf=max(conf,0.45)

    if det is None:
        conf=0.0
        lost+=1
        bad_run += 1
    else:
        lost=0
        bad_run = 0 if conf>=BAD_THRESH else (bad_run+1)

    if det is not None:
        cy_dyn.push(det[1])
        last_choice = (det[0], det[1])
        kf.update([det[0],det[1]])
        last_meas=(det[0],det[1])
    else:
        if last_meas is not None:
            kf.x[0] = 0.92*kf.x[0] + 0.08*last_meas[0]
            kf.x[1] = 0.92*kf.x[1] + 0.08*last_meas[1]

    x,y,vx,vy = float(kf.x[0]),float(kf.x[1]),float(kf.x[2]),float(kf.x[3])
    cx_out=clamp(x,0,W-1); cy_out=clamp(y,0,H-1)
    if det is None:
        if rows:
            cx_out=float(rows[-1][1]); cy_out=float(rows[-1][2])
        else:
            cx_out, cy_out = W/2.0, H/2.0
    rows.append([n, f"{cx_out:.4f}", f"{cy_out:.4f}", f"{conf:.4f}", f"{vx:.4f}", f"{vy:.4f}", W, H, FPS])
    seed_cx, seed_cy = (last_choice if last_choice is not None else (cx_out, cy_out))
    prev_gray=gray; prev_pt=np.array([[[seed_cx,seed_cy]]],dtype=np.float32); n+=1

cap.release()
with open(out_csv,"w",newline="") as f:
    wr=csv.writer(f); wr.writerow(["n","cx","cy","conf","vx","vy","w","h","fps"]); wr.writerows(rows)
print("wrote", out_csv)

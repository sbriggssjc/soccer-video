import sys, os, csv, cv2, numpy as np, math

# --- argv / io ---
args=sys.argv[1:]
if len(args)<2: raise SystemExit("usage: smart_track_one.py <in> <out_csv> [weights_or_NONE] [conf]")
in_path,out_csv=args[0],args[1]
weights=args[2] if len(args)>=3 else "NONE"
conf_min=float(args[3]) if len(args)>=4 else 0.35

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

def color_candidates(bgr, roi=None, wide=False):
    if roi is None: x0=y0=0; x1=bgr.shape[1]; y1=bgr.shape[0]; patch=bgr
    else: x0,y0,x1,y1=roi; patch=bgr[y0:y1, x0:x1]
    m=mask_red(patch, wide=wide)
    cnts,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    out=[]
    if 0<len(cnts)<=120:
        for c in cnts:
            area=cv2.contourArea(c)
            if area<28 or area>4500: continue
            peri=cv2.arcLength(c,True);  circ=(4*np.pi*area)/(peri*peri) if peri>0 else 0
            if circ<0.62: continue
            (x,y),r=cv2.minEnclosingCircle(c)
            # score ~ area*circularity
            out.append((x0+x, y0+y, float(area*circ)))
    return out

def hough_candidates(bgr, roi=None, wide=False):
    if roi is None: x0=y0=0; x1=bgr.shape[1]; y1=bgr.shape[0]; patch=bgr
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
            out.append((x0+float(x), y0+float(y), float(r*r)))  # score ~ r^2
    return out

def yolo_candidates(bgr, roi=None, lost=0):
    if not use_yolo: return []
    if roi is None: crop=bgr; x0=y0=0
    else: x0,y0,x1,y1=roi; crop=bgr[y0:y1, x0:x1]
    try:
        thr = max(0.16, conf_min - 0.10*min(lost,6))  # lower thr while lost
        rs=yolo.predict(source=crop, conf=thr, imgsz=640, verbose=False)
        out=[]
        if len(rs) and getattr(rs[0],"boxes",None) is not None:
            for box in rs[0].boxes:
                x1,y1,x2,y2=box.xyxy[0].cpu().numpy()
                conf=float(box.conf[0].item()); cx=0.5*(x1+x2); cy=0.5*(y1+y2)
                out.append((x0+cx,y0+cy,conf))
        return out
    except Exception: return []

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

def score_and_pick(props, px,py,vx,vy):
    best=None; bestS=-1.0
    for cx,cy,raws in props:
        yolo_c = raws if raws<=1.0 else 0.0
        color  = raws if raws> 1.0 else 0.0
        hough  = raws if raws> 1e3 else 0.0
        ex = px + vx*dt; ey = py + vy*dt
        d2 = (cx-ex)**2 + (cy-ey)**2
        mot = 1.0/(1.0 + d2/(180.0**2))
        S = 2.2*min(1.0,yolo_c) + 0.9*math.log1p(color/250.0) + 0.6*min(1.0,hough/2500.0) + 0.85*mot
        if S>bestS:
            bestS=S; best=(cx,cy, max(min(1.0,yolo_c), 0.55*(color>0) + 0.35*(hough>0)))
    return best

rows=[]; n=0; lost=999
prev_gray=None; prev_pt=None
lk_crit=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,0.03)
last_meas=None

while True:
    ok, frame = cap.read()
    if not ok: break
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Prediction; decay velocity strongly while blind
    if lost>0:
        kf.x[2]*=0.75; kf.x[3]*=0.75
    kf.predict()
    px,py,vx,vy = float(kf.x[0]),float(kf.x[1]),float(kf.x[2]),float(kf.x[3])

    det=None; conf=0.0
    wide = lost>=6

    # --- Bootstrap: frames 0..8 force a central measurement ---
    if n<9:
        cx0,cy0 = W//2, int(H*0.55)
        cx1,cy1 = W//2, int(H*0.35)
        roi = (int(W*0.25), int(H*0.25), int(W*0.75), int(H*0.80))
        props  = color_candidates(frame, roi, wide=False)
        props += hough_candidates(frame, roi, wide=False)
        # gentle YOLO use too
        props += yolo_candidates(frame, roi, lost=0)
        pick = score_and_pick(props, px,py,vx,vy)
        if pick is not None: det=(pick[0],pick[1]); conf=pick[2]

    # --- Normal / lost-mode proposals ---
    if det is None:
        roi = roi_from(px,py,lost)
        props  = yolo_candidates(frame, roi, lost)
        props += color_candidates(frame, roi, wide=wide)
        props += hough_candidates(frame, roi, wide=wide)
        # global re-detect every 2 frames while very lost
        if lost>=6 and (n%2==0):
            props += yolo_candidates(frame, None, lost)
            props += color_candidates(frame, None, wide=True)
            props += hough_candidates(frame, None, wide=True)
        pick = score_and_pick(props, px,py,vx,vy)
        if pick is not None: det=(pick[0],pick[1]); conf=max(conf,pick[2])

    # LK nudge if still none
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

    # Update / tether toward last good position when blind
    if det is not None:
        kf.update([det[0],det[1]])
        lost=0; last_meas=(det[0],det[1])
    else:
        lost+=1
        if last_meas is not None:
            kf.x[0] = 0.92*kf.x[0] + 0.08*last_meas[0]
            kf.x[1] = 0.92*kf.x[1] + 0.08*last_meas[1]

    x,y,vx,vy = float(kf.x[0]),float(kf.x[1]),float(kf.x[2]),float(kf.x[3])
    rows.append([n, f"{clamp(x,0,W-1):.4f}", f"{clamp(y,0,H-1):.4f}", f"{conf:.4f}", f"{vx:.4f}", f"{vy:.4f}", W, H, FPS])
    prev_gray=gray; prev_pt=np.array([[[x,y]]],dtype=np.float32); n+=1

cap.release()
with open(out_csv,"w",newline="") as f:
    wr=csv.writer(f); wr.writerow(["n","cx","cy","conf","vx","vy","w","h","fps"]); wr.writerows(rows)
print("wrote", out_csv)


Why this should help

We force a real measurement right at the start (the red ball is clearly visible early in your frames), so the state doesn’t begin on grass.

When detections drop, we don’t sprint to the edge: velocity is damped, prediction is gently pulled toward the last measured ball, and re-detect runs frequently with a looser mask.

Once re-acquired, the Kalman snaps back quickly (higher process noise q=14).

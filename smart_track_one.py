import sys, os, csv, cv2, numpy as np
try:
    from numpy import RankWarning
except Exception:
    class RankWarning(UserWarning): pass
import warnings
warnings.filterwarnings("ignore", category=RankWarning)

args = sys.argv[1:]
if len(args) < 2:
    raise SystemExit("usage: smart_track_one.py <in> <out_csv> [weights_or_NONE] [conf]")
in_path, out_csv = args[0], args[1]
weights = args[2] if len(args)>=3 else "NONE"
try: conf_min = float(args[3]) if len(args)>=4 else 0.35
except: conf_min = 0.35

cap = cv2.VideoCapture(in_path)
if not cap.isOpened(): raise SystemExit(f"Cannot open {in_path}")
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = cap.get(cv2.CAP_PROP_FPS) or 24.0
dt  = 1.0/max(FPS,1.0)

use_yolo=False; yolo=None
if weights and weights.upper()!="NONE" and os.path.exists(weights):
    try:
        from ultralytics import YOLO
        yolo = YOLO(weights); use_yolo=True
        print("YOLO loaded:", weights)
    except Exception as e:
        print("YOLO load failed:", e)

def clamp(v,a,b): return max(a,min(b,v))

# --- HSV gates for red/orange ball ---
LOW  = [(0,120,80),(165,110,80),(8,120,80)]
HIGH = [(8,255,255),(179,255,255),(22,255,255)]

def redball_candidates(bgr, roi=None):
    if roi is None:
        x0=y0=0; x1=bgr.shape[1]; y1=bgr.shape[0]
    else:
        x0,y0,x1,y1 = roi
    patch = bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    m = np.zeros(hsv.shape[:2], np.uint8)
    for lo,hi in zip(LOW,HIGH):
        m |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    m = cv2.medianBlur(m,5)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3,3),np.uint8), iterations=1)
    cnts,_ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cand=[]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 30 or area > 5000: continue
        (x,y),radius = cv2.minEnclosingCircle(c)
        if radius<=0: continue
        circ = (4*np.pi*area) / ( (cv2.arcLength(c,True)+1e-6)**2 )
        if circ < 0.40: continue
        cx = x0 + x; cy = y0 + y
        score = float(area) * float(circ)
        cand.append((cx,cy,score))
    return cand

def best_color_ball(bgr, last_xy=None, pad=160):
    roi=None
    if last_xy is not None:
        cx,cy = last_xy
        x0=int(clamp(cx-pad,0,W-1)); x1=int(clamp(cx+pad,1,W))
        y0=int(clamp(cy-pad,0,H-1)); y1=int(clamp(cy+pad,1,H))
        roi=(x0,y0,x1,y1)
    cand = redball_candidates(bgr, roi)
    if not cand: return (None,None,0.0)
    if last_xy is not None:
        lx,ly = last_xy
        cand.sort(key=lambda t: (t[0]-lx)**2 + (t[1]-ly)**2)
    cx,cy,score = cand[0]
    return (cx,cy,float(score))

def yolo_ball(bgr, last_xy=None, pad=160):
    if not use_yolo: return (None,None,0.0)
    if last_xy is None:
        crop=bgr; x0=y0=0
    else:
        lx,ly=last_xy
        x0=int(clamp(lx-pad,0,W-1)); x1=int(clamp(lx+pad,1,W))
        y0=int(clamp(ly-pad,0,H-1)); y1=int(clamp(ly+pad,1,H))
        crop = bgr[y0:y1, x0:x1]
    try:
        imgsz=max(512,((max(W,H)+31)//32)*32)
        rs = yolo.predict(source=crop, conf=conf_min, imgsz=imgsz, verbose=False)
        if len(rs):
            r = rs[0]
            if getattr(r,"boxes",None) is not None and len(r.boxes)>0:
                b=r.boxes
                i=int(np.argmax(b.conf.cpu().numpy()))
                xyxy=b.xyxy[i].cpu().numpy()
                conf=float(b.conf[i].item())
                cx=0.5*(float(xyxy[0])+float(xyxy[2])); cy=0.5*(float(xyxy[1])+float(xyxy[3]))
                cx += x0; cy += y0
                return (cx,cy,conf)
    except Exception:
        pass
    return (None,None,0.0)

def fuse_detect(bgr, last_xy, last_v):
    cand=[]
    cand.append( yolo_ball(bgr, last_xy, pad=160) )
    cand.append( best_color_ball(bgr, last_xy, pad=140) )
    if max([c[2] for c in cand if c[0] is not None]+[0.0]) < 0.35:
        cand.append( yolo_ball(bgr, None, pad=0) )
        cand.append( best_color_ball(bgr, None, pad=0) )
    best=None; bestS=-1
    for cx,cy,s in cand:
        if cx is None: continue
        yolo_c = s if s<=1.0 else 0.0
        color  = s if s>1.0 else 0.0
        mot    = 0.0
        if (last_xy is not None) and (last_v is not None):
            ex = last_xy[0]+last_v[0]*dt; ey = last_xy[1]+last_v[1]*dt
            d2 = (cx-ex)**2 + (cy-ey)**2
            mot = 1.0 / (1.0 + d2/(160.0**2))
        score = 2.0*yolo_c + 1.0*(color>0)*np.log10(1.0+color/200.0) + 0.6*mot
        if score>bestS:
            bestS=score; best=(cx,cy,float(min(1.0,max(yolo_c,0.4*(color>0)))))
    return best if best is not None else (None,None,0.0)

rows=[]; n=0
prev_gray=None; prev_pt=None
last_xy=None; last_v=None
lost=999
lk_crit=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,0.03)
early_frames = int(1.0*FPS)

while True:
    ok, frame = cap.read()
    if not ok: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    det=None; conf=0.0
    force_fuse = (n<early_frames) or (lost>0)

    if force_fuse or last_xy is None:
        cx,cy,cf = fuse_detect(frame, last_xy, last_v)
        if cx is not None:
            det=(cx,cy); conf=max(conf,cf); lost=0

    if det is None and prev_gray is not None and prev_pt is not None:
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pt, None, winSize=(21,21), maxLevel=3, criteria=lk_crit)
        if st is not None and int(st.ravel()[0])==1:
            p0r, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, None, winSize=(21,21), maxLevel=3, criteria=lk_crit)
            fb = float(np.linalg.norm(prev_pt - p0r)) if (st2 is not None and int(st2.ravel()[0])==1) else 9e9)
            if fb < 2.0:
                x=float(p1[0,0,0]); y=float(p1[0,0,1])
                if 0<=x<W and 0<=y<H:
                    det=(x,y); conf=max(conf,0.50); lost=0
            else:
                lost += 1

    edge_drift = (last_xy is not None) and (abs(last_xy[0]-(W-20))<40 or abs(last_xy[0]-20)<40)
    if det is None or conf<0.35 or edge_drift:
        cx2,cy2,cf2 = fuse_detect(frame, det if det is not None else last_xy, last_v)
        if cx2 is not None and (det is None or cf2>conf):
            det=(cx2,cy2); conf=cf2; lost=0

    if det is None:
        if last_xy is not None: cx,cy = last_xy
        else: cx,cy = W/2.0, H/2.0
        conf=0.0; lost+=1
    else:
        cx,cy = det

    if last_xy is not None:
        last_v = ((cx-last_xy[0])/dt, (cy-last_xy[1])/dt)
    last_xy = (cx,cy)

    rows.append([n, f"{clamp(cx,0,W-1):.4f}", f"{clamp(cy,0,H-1):.4f}", f"{conf:.4f}", W, H, FPS])
    prev_gray=gray; prev_pt=np.array([[[cx,cy]]],dtype=np.float32); n+=1

cap.release()
with open(out_csv,"w",newline="") as f:
    wr=csv.writer(f); wr.writerow(["n","cx","cy","conf","w","h","fps"]); wr.writerows(rows)

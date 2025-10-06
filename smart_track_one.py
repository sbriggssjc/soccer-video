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

use_yolo = False
yolo = None
if weights and weights.upper()!="NONE" and os.path.exists(weights):
    try:
        from ultralytics import YOLO
        yolo = YOLO(weights); use_yolo = True
        print("YOLO: using", weights)
    except Exception as e:
        print("YOLO load failed:", e); use_yolo = False

def clamp(v,a,b): return max(a,min(b,v))
rows=[]; n=0; prev_gray=None; prev_pt=None; lost=999
lk_crit=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,0.03)

# helper: run yolo on a SMALL crop around last point if available
def yolo_detect(frame, last_xy=None, pad=160):
    h,w=frame.shape[:2]
    crop = frame
    x0=y0=0; x1=w; y1=h
    if last_xy is not None:
        cx,cy = last_xy
        x0 = int(clamp(cx-pad,0,w-1)); x1 = int(clamp(cx+pad,1,w))
        y0 = int(clamp(cy-pad,0,h-1)); y1 = int(clamp(cy+pad,1,h))
        crop = frame[y0:y1, x0:x1]
    try:
        imgsz=max(640,((max(w,h)+31)//32)*32)
        rs = yolo.predict(source=crop, conf=conf_min, imgsz=imgsz, verbose=False)
        if len(rs):
            r = rs[0]
            if getattr(r,"boxes",None) is not None and len(r.boxes)>0:
                b=r.boxes
                i=int(np.argmax(b.conf.cpu().numpy()))
                xyxy=b.xyxy[i].cpu().numpy()
                conf=float(b.conf[i].item())
                cx=0.5*(float(xyxy[0])+float(xyxy[2])); cy=0.5*(float(xyxy[1])+float(xyxy[3]))
                if last_xy is not None:  # offset back to full frame
                    cx += x0; cy += y0
                return (cx,cy,conf)
    except Exception:
        pass
    return (None,None,0.0)

while True:
    ok, frame = cap.read()
    if not ok: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    det=None; conf=0.0

    force_yolo = (n < int(2.0*FPS))  # first 2 seconds: YOLO every frame
    if use_yolo and (force_yolo or lost>0):
        cx,cy,c = yolo_detect(frame, None if (rows==[]) else (float(rows[-1][1]), float(rows[-1][2])))
        if c>0:
            det=(cx,cy); conf=float(c); lost=0

    # LK track if we have a previous point and either YOLO didn't return or to refine
    if det is None and prev_gray is not None and prev_pt is not None:
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pt, None, winSize=(21,21), maxLevel=3, criteria=lk_crit)
        good = st is not None and int(st.ravel()[0])==1
        if good:
            # forward-backward check
            p0r, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, None, winSize=(21,21), maxLevel=3, criteria=lk_crit)
            fb = float(np.linalg.norm(prev_pt - p0r)) if (st2 is not None and int(st2.ravel()[0])==1) else 9e9
            if fb < 2.0:
                x=float(p1[0,0,0]); y=float(p1[0,0,1])
                if 0<=x<W and 0<=y<H:
                    det=(x,y); conf=max(conf,0.55); lost=0
            else:
                lost += 1

    # If confidence still low, try YOLO again with crop around last known or LK guess
    if use_yolo and (det is None or conf<0.40):
        last_xy = None
        if det is not None: last_xy = det
        elif rows: last_xy = (float(rows[-1][1]), float(rows[-1][2]))
        cx2,cy2,c2 = yolo_detect(frame, last_xy, pad=140)
        if c2>conf:
            det=(cx2,cy2); conf=float(c2); lost=0

    if det is None:
        if rows:
            cx,cy=float(rows[-1][1]),float(rows[-1][2])
        else:
            cx,cy = W/2.0, H/2.0
        conf=0.0; lost+=1
    else:
        cx,cy=det

    rows.append([n, f"{clamp(cx,0,W-1):.4f}", f"{clamp(cy,0,H-1):.4f}", f"{conf:.4f}", W, H, FPS])
    prev_gray=gray; prev_pt=np.array([[[cx,cy]]],dtype=np.float32); n+=1

with open(out_csv,"w",newline="") as f:
    wr=csv.writer(f); wr.writerow(["n","cx","cy","conf","w","h","fps"]); wr.writerows(rows)

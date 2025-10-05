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
if weights and weights.upper()!="NONE" and os.path.exists(weights):
    try:
        from ultralytics import YOLO
        yolo = YOLO(weights); use_yolo = True
    except Exception:
        use_yolo = False

def clamp(v,a,b): return max(a,min(b,v))
rows=[]; n=0; prev_gray=None; prev_pt=None; lost=0
lk_crit=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,0.03)

while True:
    ok, frame = cap.read()
    if not ok: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    det=None; conf=0.0
    if use_yolo and (lost>0 or n%2==0):
        try:
            imgsz=max(640,((max(W,H)+31)//32)*32)
            rs = yolo.predict(source=frame, conf=conf_min, imgsz=imgsz, verbose=False)
            if len(rs):
                r = rs[0]
                if getattr(r,"boxes",None) is not None and len(r.boxes)>0:
                    b=r.boxes
                    i=int(np.argmax(b.conf.cpu().numpy()))
                    xyxy=b.xyxy[i].cpu().numpy()
                    conf=float(b.conf[i].item())
                    cx=0.5*(float(xyxy[0])+float(xyxy[2])); cy=0.5*(float(xyxy[1])+float(xyxy[3]))
                    det=(cx,cy)
        except Exception:
            pass

    if det is None and prev_gray is not None and prev_pt is not None:
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pt, None, winSize=(21,21), maxLevel=3, criteria=lk_crit)
        if st is not None and int(st.ravel()[0])==1:
            p0r, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, None, winSize=(21,21), maxLevel=3, criteria=lk_crit)
            if st2 is not None and int(st2.ravel()[0])==1:
                fb = float(np.linalg.norm(prev_pt - p0r))
                if fb < 2.0:
                    x=float(p1[0,0,0]); y=float(p1[0,0,1])
                    if 0<=x<W and 0<=y<H: det=(x,y); conf=max(conf,0.55)

    if det is None:
        if rows: cx,cy=float(rows[-1][1]),float(rows[-1][2])
        else: cx,cy=W/2.0,H/2.0
        conf=0.0; lost+=1
    else:
        cx,cy=det; lost=0

    rows.append([n, f"{clamp(cx,0,W-1):.4f}", f"{clamp(cy,0,H-1):.4f}", f"{conf:.4f}", W, H, FPS])
    prev_gray=gray; prev_pt=np.array([[[cx,cy]]],dtype=np.float32); n+=1

with open(out_csv,"w",newline="") as f:
    wr=csv.writer(f); wr.writerow(["n","cx","cy","conf","w","h","fps"]); wr.writerows(rows)

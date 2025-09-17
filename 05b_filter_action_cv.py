# 05b_filter_action_cv.py
import argparse, csv, math, os, sys
from dataclasses import dataclass
import cv2, numpy as np, pandas as pd

def _to_num(x):
    if pd.isna(x): return None
    s = str(x).strip().replace(',', '.')
    try: return float(s)
    except: return None

def read_candidates(path):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    scol = cols.get('start') or [c for c in df.columns if 'start' in c.lower()][0]
    ecol = cols.get('end')   or [c for c in df.columns if 'end'   in c.lower()][0]
    score_col = None
    for key in ('action_score','score','rank'):
        if key in cols: score_col = cols[key]; break
    rows = []
    for _,r in df.iterrows():
        s = _to_num(r[scol]); e = _to_num(r[ecol])
        if s is None or e is None or e <= s: continue
        sc = _to_num(r[score_col]) if score_col else 0.0
        rows.append(dict(start=float(s), end=float(e), score=float(sc)))
    return rows

@dataclass
class HSVRange:
    h_low:int; s_low:int; v_low:int
    h_high:int; s_high:int; v_high:int
    def low(self):  return np.array([self.h_low, self.s_low, self.v_low], dtype=np.uint8)
    def high(self): return np.array([self.h_high, self.s_high, self.v_high], dtype=np.uint8)

def parse_hsv(arg):
    a = [int(x) for x in arg.split(',')]
    if len(a)!=6: raise ValueError("HSV must be 6 ints: hL,sL,vL,hH,sH,vH")
    return HSVRange(*a)

def optical_flow_metrics(prev_gray, gray):
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 25, 3, 5, 1.1, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1], angleInDegrees=False)
    # Global (camera) direction ~ vector median of flow
    vx = np.median(flow[...,0]); vy = np.median(flow[...,1])
    vmag = math.hypot(float(vx), float(vy)) + 1e-6
    # Pixels that deviate from global direction by > ~35°
    dot = (flow[...,0]*vx + flow[...,1]*vy) / (vmag*np.maximum(1e-6, np.sqrt(flow[...,0]**2+flow[...,1]**2)))
    dev_mask = (dot < math.cos(math.radians(35)))
    # Residual action magnitude (not camera pan)
    residual = mag[dev_mask]
    residual_mag = float(np.median(residual)) if residual.size else 0.0
    return residual_mag

def green_ratio(hsv):
    # wide green band for pitches
    mask = cv2.inRange(hsv, (30, 25, 25), (90, 255, 255))
    return float(np.mean(mask>0))

def find_ball_centroid(bgr):
    # white-ish but not bright lines cluster; small area = ball
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0,0,200), (180,60,255))
    # suppress pitch lines (long, thin): erode a bit
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, k, iterations=1)
    cnts,_ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h,w = white.shape
    best=None; best_score=1e9
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 6 or a > 180:  # tiny to small blobs
            continue
        (x,y), r = cv2.minEnclosingCircle(c)
        circ = 0 if r<=0 else (a/(math.pi*r*r))
        # prefer near-circular small blobs
        score = abs(1.0-circ) + 0.002*a
        if score < best_score:
            best_score = score; best = (int(x),int(y))
    return best

def team_presence_near(hsv, cx, cy, team_hsv:HSVRange, patch=22):
    h,w,_ = hsv.shape
    x1 = max(0, cx-patch); y1 = max(0, cy-patch)
    x2 = min(w, cx+patch); y2 = min(h, cy+patch)
    roi = hsv[y1:y2, x1:x2]
    if roi.size==0: return 0.0
    mask = cv2.inRange(roi, team_hsv.low(), team_hsv.high())
    return float(np.mean(mask>0))

def analyze_window(cap, start, end, fps_sample, team_hsv, att_third_cut=0.18):
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0,start)*1000.0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    step = max(1,int(round(cap.get(cv2.CAP_PROP_FPS)/fps_sample))) if cap.get(cv2.CAP_PROP_FPS)>0 else 4
    idx=0; prev_gray=None; prev_ball=None
    green_rates=[]; flow_resid=[]; ball_speeds=[]; team_near=[]; att_pos=[]
    while True:
        t = start + (idx/fps_sample)
        if t>=end: break
        ok = cap.grab()
        for _ in range(step-1):
            cap.grab()
        ok, frame = cap.retrieve()
        if not ok: break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        g = green_ratio(hsv); green_rates.append(g)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            flow_resid.append(optical_flow_metrics(prev_gray, gray))
        prev_gray = gray
        ball = find_ball_centroid(frame)
        if ball:
            cx,cy = ball
            if prev_ball:
                dx = (cx-prev_ball[0]); dy = (cy-prev_ball[1])
                ball_speeds.append(math.hypot(dx,dy))
            prev_ball = ball
            team_near.append(team_presence_near(hsv,cx,cy,team_hsv))
            # attacking thirds in X (left/right edges)
            att_pos.append( 1.0 if (cx < att_third_cut*w or cx > (1.0-att_third_cut)*w) else 0.0 )
        idx += 1
    # Aggregate
    green_ok = np.mean(green_rates) if green_rates else 0
    flow   = np.median(flow_resid) if flow_resid else 0
    if ball_speeds:
        sp = np.array(ball_speeds)
        # robust stats
        speed_med = float(np.median(sp))
        contig = int(np.max(np.convolve((sp>3.5).astype(int), np.ones(5,dtype=int), 'same')))  # >= ~5 frames
        hits = int(np.sum((sp[1:]-sp[:-1])>2.5))  # acceleration spikes
    else:
        speed_med=0.0; contig=0; hits=0
    team_pres = float(np.mean(team_near)) if team_near else 0.0
    att_frac  = float(np.mean(att_pos))  if att_pos  else 0.0
    return dict(
        green_ok=green_ok, flow=flow,
        speed_med=speed_med, contig=contig, hits=hits,
        team_pres=team_pres, att_frac=att_frac
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--csv',   required=True)
    ap.add_argument('--out',   required=True)
    ap.add_argument('--fps-sample', type=float, default=6.0)
    ap.add_argument('--min-green', type=float, default=0.30)
    ap.add_argument('--min-flow',  type=float, default=0.20)     # motion not due to camera pan
    ap.add_argument('--min-ball-speed', type=float, default=3.8) # px/frame median
    ap.add_argument('--min-contig-frames', type=int, default=6)
    ap.add_argument('--min-ball-hits', type=int, default=1)
    ap.add_argument('--team-hsv', default='105,70,20,130,255,160')
    ap.add_argument('--min-team-pres', type=float, default=0.10)
    ap.add_argument('--team-bias', type=float, default=0.25)
    ap.add_argument('--att-third-cut', type=float, default=0.18)
    args = ap.parse_args()

    team_hsv = parse_hsv(args.team_hsv)
    rows = read_candidates(args.csv)
    if not rows:
        print(f"No valid rows in {args.csv}", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(args.video)
    out_rows=[]
    for r in rows:
        m = analyze_window(cap, r['start'], r['end'], args.fps_sample, team_hsv, args.att_third_cut)
        if m['green_ok'] < args.min_green:  # off-field/bench
            continue
        if m['flow'] < args.min_flow:
            continue
        if m['speed_med'] < args.min_ball_speed or m['contig'] < args.min_contig_frames or m['hits'] < args.min_ball_hits:
            continue
        # action score: flow + ball speed + attacking third + team presence
        action = (0.55*m['flow']
                  + 0.65*(m['speed_med']/8.0)
                  + 0.35*m['att_frac']
                  + (args.team_bias * m['team_pres']))
        out_rows.append(dict(
            start=r['start'], end=r['end'], action_score=round(float(action),4),
            flow=m['flow'], speed_med=m['speed_med'], contig=m['contig'], hits=m['hits'],
            team_pres=m['team_pres'], att_frac=m['att_frac']
        ))
    cap.release()

    if not out_rows:
        print("No clips passed the action filter; try lowering --min-flow to 0.15 or --min-ball-speed to ~3.2", file=sys.stderr)
    with open(args.out,'w',newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()) if out_rows else ['start','end','action_score'])
        w.writeheader()
        for r in out_rows: w.writerow(r)

if __name__=="__main__":
    main()

import argparse, json, math, cv2, numpy as np
from collections import namedtuple
from math import hypot

Cand = namedtuple("Cand", "x y score ncc circ dist grad hsrc")  # hsrc=hough strength

# ---------- utilities ----------
def to_gray(bgr): return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
def eq(g): return cv2.equalizeHist(g)

def stabilize_init(frame):
    g = to_gray(frame); g = eq(g)
    kp = cv2.goodFeaturesToTrack(g, maxCorners=2000, qualityLevel=0.01, minDistance=8, blockSize=7)
    return g, kp

def stabilize_step(prev_gray, cur_bgr):
    g = to_gray(cur_bgr); g = eq(g)
    p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=1000, qualityLevel=0.01, minDistance=8, blockSize=7)
    if p0 is None: return g, np.eye(3, dtype=np.float32)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, g, p0, None, winSize=(21,21), maxLevel=3)
    if p1 is None or st is None: return g, np.eye(3, dtype=np.float32)
    p0v = p0[st==1].reshape(-1,2); p1v = p1[st==1].reshape(-1,2)
    if len(p0v) < 12: return g, np.eye(3, dtype=np.float32)
    M, inliers = cv2.estimateAffinePartial2D(p1v, p0v, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None:
        A = np.eye(3, dtype=np.float32)
    else:
        A = np.eye(3, dtype=np.float32); A[:2,:] = M
    return g, A

def warp_affine(img, A, W, H):
    return cv2.warpAffine(img, A[:2,:], (W,H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def field_mask_bgr(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H,S,V = cv2.split(hsv)
    grass = (H >= 35) & (H <= 95) & (S >= 40) & (V >= 40)
    m = (grass.astype(np.uint8))*255
    m = cv2.medianBlur(m, 5)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7,7),np.uint8))
    m = cv2.erode(m, np.ones((9,9),np.uint8), 1)
    return m

def sideline_penalty(x, y, W, H, margin_xy=(48, 56), heavy=16.0):
    mx,my = margin_xy
    return heavy if (x<mx or x>W-mx or y<my or y>H-my) else 0.0

def circularity(cnt):
    a = cv2.contourArea(cnt); p = cv2.arcLength(cnt, True)
    return 0.0 if p<=0 or a<=0 else float(4*math.pi*a/(p*p))

def ncc_score(win, tpl):
    if win.size==0 or tpl.size==0: return -1.0
    if win.shape[0] < tpl.shape[0] or win.shape[1] < tpl.shape[1]: return -1.0
    g = eq(win); t = eq(tpl)
    r1 = cv2.matchTemplate(g, t, cv2.TM_CCOEFF_NORMED).max()
    h,w = t.shape[:2]; t2 = cv2.resize(t, (max(3,int(w*0.75)), max(3,int(h*0.75))), cv2.INTER_AREA)
    r2 = cv2.matchTemplate(g, t2, cv2.TM_CCOEFF_NORMED).max()
    return float(max(r1, r2))

def radial_gradient_strength(gray, cx, cy, r):
    r = int(max(6, min(24, r)))
    x0=int(max(0,cx-r)); y0=int(max(0,cy-r)); x1=int(min(gray.shape[1], cx+r)); y1=int(min(gray.shape[0], cy+r))
    roi = gray[y0:y1, x0:x1]
    if roi.size==0: return 0.0
    gx = cv2.Sobel(roi, cv2.CV_32F, 1,0,ksize=3); gy = cv2.Sobel(roi, cv2.CV_32F, 0,1,ksize=3)
    mag = np.sqrt(gx*gx + gy*gy)
    return float(np.percentile(mag, 80))

def lk_refine(prev_gray, cur_gray, prev_xy):
    if prev_xy is None: return None, 0.0
    p0 = np.array([[prev_xy]], dtype=np.float32)
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, p0, None,
                                           winSize=(21,21), maxLevel=3,
                                           criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    if st is None or st[0][0]==0: return None,0.0
    conf = 1.0/float(1.0 + (err[0][0] if err is not None else 20.0))
    return (float(p1[0][0][0]), float(p1[0][0][1])), conf

# ---------- candidate fusion ----------
def gen_candidates(stab_bgr, raw_bgr, pred_xy, tpl, field_mask, search_r=300,
                   min_r=6, max_r=22, min_circ=0.58, max_cands=14):
    H,W = stab_bgr.shape[:2]; px,py = pred_xy
    x0=int(max(0,px-search_r)); y0=int(max(0,py-search_r))
    x1=int(min(W,px+search_r)); y1=int(min(H,py+search_r))
    if x1<=x0+2 or y1<=y0+2: return []
    roi = stab_bgr[y0:y1, x0:x1]; roi_raw = raw_bgr[y0:y1, x0:x1]
    field_roi = field_mask[y0:y1, x0:x1]
    gray = to_gray(roi)

    # contour candidates (non-green, bright-ish, low-sat handled by field mask already)
    thr = cv2.adaptiveThreshold(eq(gray), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, -7)
    thr = cv2.medianBlur(thr, 5)
    cnts,_ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    out=[]
    def maybe_add(cx,cy,rad, hsrc=0.0):
        bx = x0+cx; by = y0+cy
        if field_roi[int(np.clip(cy,0,field_roi.shape[0]-1)), int(np.clip(cx,0,field_roi.shape[1]-1))]==0:
            return
        dist = hypot(bx-px, by-py)
        circ = None
        ncc=0.0; grad=0.0
        # NCC window
        side = int(max(16, min(96, rad*6)))
        sx0 = int(max(0,bx-side//2)); sy0=int(max(0,by-side//2))
        sx1 = int(min(W,sx0+side));   sy1=int(min(H,sy0+side))
        win = gray[(sy0-y0):(sy1-y0), (sx0-x0):(sx1-x0)]
        if tpl is not None: ncc = ncc_score(win, tpl)
        grad = radial_gradient_strength(gray, int(cx), int(cy), int(rad))
        base = (-0.02*dist) + (1.6*ncc) + (0.018*grad) + (0.6*hsrc)
        base -= sideline_penalty(bx, by, W, H, margin_xy=(56,64), heavy=18.0)
        out.append(Cand(float(bx), float(by), float(base), float(ncc), float(circ or 0.0), float(dist), float(grad), float(hsrc)))

    # from contours
    for c in cnts:
        a = cv2.contourArea(c)
        if a < (min_r*min_r*0.6) or a > (max_r*max_r*4.0): continue
        circ = circularity(c)
        if circ < min_circ: continue
        (cx,cy),rad = cv2.minEnclosingCircle(c)
        maybe_add(cx,cy,rad, hsrc=0.0)

    # Hough circles (helps on lines)
    hc = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=14,
                          param1=110, param2=18, minRadius=min_r, maxRadius=max_r)
    if hc is not None:
        for x,y,r in np.round(hc[0,:]).astype(int):
            maybe_add(x,y,r, hsrc=1.0)

    out.sort(key=lambda c: c.score, reverse=True)
    return out[:max_cands]

# ---------- DP path + smoothing ----------
def solve_path(cand_lists, start_xy, lam_vel=0.06, miss_penalty=1.25, max_jump=260.0):
    N=len(cand_lists); MISS=Cand(np.nan,np.nan,-999,0,0,0,0,0)
    C=[c+[MISS] for c in cand_lists]; K=[len(c) for c in C]
    INF=1e15; dp=[np.full(k,INF) for k in K]; prv=[np.full(k,-1,int) for k in K]
    sx,sy=start_xy
    for j,c in enumerate(C[0]):
        dp[0][j]= miss_penalty if np.isnan(c.x) else (-c.score + lam_vel*hypot(c.x-sx,c.y-sy))
    for t in range(1,N):
        for j,cj in enumerate(C[t]):
            for i,ci in enumerate(C[t-1]):
                xj,yj = (cj.x,cj.y) if not np.isnan(cj.x) else (ci.x,ci.y)
                xi,yi = (ci.x,ci.y) if not np.isnan(ci.x) else (xj,yj)
                if not (np.isnan(xi) or np.isnan(xj)):
                    jump=hypot(xj-xi,yj-yi)
                    if jump>max_jump: continue
                unary = miss_penalty if np.isnan(cj.x) else (-cj.score)
                pair  = 0.0 if (np.isnan(xi) or np.isnan(xj)) else lam_vel*hypot(xj-xi,yj-yi)
                cost = dp[t-1][i]+unary+pair
                if cost<dp[t][j]: dp[t][j]=cost; prv[t][j]=i
    j=int(np.argmin(dp[-1])); path=[None]*N
    for t in range(N-1,-1,-1):
        c=C[t][j]; path[t]=(c.x,c.y,("miss" if np.isnan(c.x) else "cand")); j=int(prv[t][j]) if t>0 and prv[t][j]>=0 else j
    # fill + tighten
    last=None
    for t in range(N):
        x,y,src=path[t]
        if np.isnan(x) or np.isnan(y):
            if last is None: last=start_xy
            path[t]=(last[0],last[1],"pred")
        else: last=(x,y)
    nxt=None
    for t in range(N-1,-1,-1):
        x,y,src=path[t]
        if src=="pred" and nxt is not None:
            path[t]=(0.6*x+0.4*nxt[0], 0.6*y+0.4*nxt[1], "pred")
        else:
            nxt=(x,y)
    return path

def ema_fb(series, a=0.28):
    s=list(series)
    for i in range(1,len(s)): s[i]= (1-a)*s[i-1] + a*s[i]
    for i in range(len(s)-2,-1,-1): s[i]= (1-a)*s[i+1] + a*s[i]
    return s

def plan_zoom(xs, ys, fps, W, H, zmin=1.06, zmax=1.82, k_speed=0.00105, edge_m=0.14, edge_gain=0.16, rate=0.045):
    N=len(xs); zt=[0]*N; z=[0]*N
    def edge_prox(bx,by):
        dl=bx/max(1,W); dr=(W-bx)/max(1,W); dt=by/max(1,H); db=(H-by)/max(1,H)
        return max(0.0, edge_m - min(dl,dr,dt,db))/max(edge_m,1e-6)
    for t in range(N):
        sp = 0.0 if t==0 else hypot(xs[t]-xs[t-1], ys[t]-ys[t-1]) * fps
        zr = max(zmin, min(zmax, zmax - k_speed*sp))
        zr = max(zmin, zr - edge_gain*edge_prox(xs[t],ys[t]))
        zt[t]=zr
    # smooth + rate-limit
    z = ema_fb(ema_fb(zt, a=0.22), a=0.22)
    for t in range(1,N):
        dz = max(-rate, min(rate, z[t]-z[t-1])); z[t]=z[t-1]+dz
    for t in range(N-2,-1,-1):
        dz = max(-rate, min(rate, z[t]-z[t+1])); z[t]=z[t+1]+dz
    return [float(max(zmin, min(zmax, v))) for v in z]

# ---------- main ----------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--init-manual", action="store_true")
    ap.add_argument("--init-t", type=float, default=0.8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-cands", type=int, default=14)
    ap.add_argument("--search-r", type=int, default=300)
    ap.add_argument("--min-r", type=int, default=6)
    ap.add_argument("--max-r", type=int, default=22)
    args=ap.parse_args()

    cap=cv2.VideoCapture(args.inp)
    if not cap.isOpened(): raise SystemExit("Cannot open input")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # read init frame
    idx0=max(0,int(round(args.init_t*fps))); cap.set(cv2.CAP_PROP_POS_FRAMES, idx0)
    ok, f0 = cap.read()
    if not ok: raise SystemExit("Cannot read init frame")

    if not args.init_manual or not hasattr(cv2, "selectROI"):
        raise SystemExit("Use --init-manual")
    cv2.namedWindow("Select ball", cv2.WINDOW_NORMAL)
    r = cv2.selectROI("Select ball", f0, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select ball")
    if r is None or r[2]<=0 or r[3]<=0: raise SystemExit("No ROI selected")
    bx0=r[0]+r[2]/2.0; by0=r[1]+r[3]/2.0

    tpl = eq(to_gray(f0))[int(by0-32):int(by0+32), int(bx0-32):int(bx0+32)].copy()
    if tpl.size==0: tpl = eq(to_gray(f0))

    # pass 1: stabilize and collect candidates
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, f1 = cap.read()
    if not ok: raise SystemExit("Empty video")
    prev_g, _ = stabilize_init(f1)
    field_full = field_mask_bgr(f1)
    cand_lists=[]; positions=[]
    pred=(bx0,by0); miss_streak=0
    n=0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ok, fr = cap.read()
        if not ok: break
        cur_g, A = stabilize_step(prev_g, fr)
        stab = warp_affine(fr, A, W, H)
        # LK steer
        if positions:
            xy_flow, conf = lk_refine(prev_g, cur_g, positions[-1])
            if xy_flow and conf>0.02:
                pred = (0.7*xy_flow[0]+0.3*positions[-1][0], 0.7*xy_flow[1]+0.3*positions[-1][1])
            else:
                pred = positions[-1]
        # widen search on misses
        sr = args.search_r + min(220, 40*miss_streak)
        cands = gen_candidates(stab, fr, pred, tpl, field_full, search_r=sr, min_r=args.min_r, max_r=args.max_r, max_cands=args.max_cands)
        if not cands and miss_streak>=10:
            # global sweep fallback
            cands = gen_candidates(stab, fr, (W/2.0,H/2.0), tpl, field_full, search_r=max(W,H), max_cands=args.max_cands*2)

        if cands:
            positions.append((cands[0].x, cands[0].y)); miss_streak=0
            cur_tpl = eq(to_gray(fr))[int(cands[0].y-32):int(cands[0].y+32), int(cands[0].x-32):int(cands[0].x+32)]
            if cur_tpl.size and tpl.size and cur_tpl.shape==tpl.shape:
                tpl = cv2.addWeighted(tpl, 0.9, cur_tpl, 0.1, 0)
        else:
            positions.append(positions[-1] if positions else (bx0,by0)); miss_streak+=1

        cand_lists.append(cands)
        prev_g = cur_g; n+=1
    cap.release()

    # DP solve + smoothing
    path = solve_path(cand_lists, (bx0,by0), lam_vel=0.07, miss_penalty=1.3, max_jump=280.0)
    xs = [p[0] for p in path]; ys = [p[1] for p in path]
    xs = ema_fb(xs, a=0.26); ys = ema_fb(ys, a=0.26)
    z  = plan_zoom(xs, ys, fps, W, H, zmin=1.06, zmax=1.84, k_speed=0.0011, edge_m=0.14, edge_gain=0.18, rate=0.042)

    # write JSONL
    with open(args.out, "w", encoding="utf-8") as f:
        for i,(x,y) in enumerate(zip(xs,ys)):
            t = i/float(fps)
            f.write(json.dumps({"t":t,"bx":float(x),"by":float(y),"z":float(z[i])})+"\n")

if __name__=="__main__":
    main()

import argparse, json, math, os
import cv2, numpy as np
from math import hypot

def to_gray(bgr): return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
def eq(g): return cv2.equalizeHist(g)

# ---------- stabilization ----------
def stabilize_step(prev_gray, cur_bgr):
    g = eq(to_gray(cur_bgr))
    p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=1200, qualityLevel=0.01, minDistance=8, blockSize=7)
    if p0 is None: return g, np.eye(3)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, g, p0, None, winSize=(21,21), maxLevel=3)
    if p1 is None or st is None: return g, np.eye(3)
    p0v = p0[st==1].reshape(-1,2); p1v = p1[st==1].reshape(-1,2)
    if len(p0v) < 12: return g, np.eye(3)
    M, _ = cv2.estimateAffinePartial2D(p1v, p0v, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    A = np.eye(3); 
    if M is not None: A[:2,:] = M
    return g, A

def warp_affine(img, A, W, H): 
    return cv2.warpAffine(img, A[:2,:], (W,H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

# ---------- masks & motion ----------
def field_mask_bgr(bgr):
    H, W = bgr.shape[:2]
    pixels = bgr.reshape(-1, 3).astype(np.float32)
    if pixels.size == 0:
        return np.zeros((H, W), dtype=np.uint8)

    sample_size = min(20000, pixels.shape[0])
    if sample_size < 3:
        return np.zeros((H, W), dtype=np.uint8)

    if sample_size < pixels.shape[0]:
        rng = np.random.default_rng(12345)
        idx = rng.choice(pixels.shape[0], size=sample_size, replace=False)
        sample = pixels[idx]
    else:
        sample = pixels

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 25, 0.5)
    try:
        _, labels, centers = cv2.kmeans(sample, 3, None, criteria, 4, cv2.KMEANS_PP_CENTERS)
    except cv2.error:
        return np.zeros((H, W), dtype=np.uint8)

    centers = centers.astype(np.float32)
    counts = np.bincount(labels.flatten(), minlength=3).astype(np.float32)
    area_ratios = counts / max(counts.sum(), 1.0)

    def green_score(center, area_ratio):
        b, g, r = center
        hsv = cv2.cvtColor(center.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2HSV)[0, 0]
        hue = hsv[0] / 180.0
        sat = hsv[1] / 255.0
        hue_dist = min(abs(hue - 1/3), abs(hue - 1/3 + 1), abs(hue - 1/3 - 1))
        hue_score = max(0.0, 1.0 - hue_dist / 0.25)
        green_ratio = g / (r + b + 1.0)
        green_diff = (g - max(r, b)) / 255.0
        return (0.6 * green_ratio + 0.6 * green_diff + 0.4 * sat + 0.4 * hue_score + 0.2 * area_ratio)

    scores = [green_score(c, area_ratios[i]) for i, c in enumerate(centers)]
    green_idx = int(np.argmax(scores))

    dists = np.linalg.norm(pixels[:, None, :] - centers[None, :, :], axis=2)
    labels_full = np.argmin(dists, axis=1)
    mask = (labels_full.reshape(H, W) == green_idx).astype(np.uint8) * 255

    mask = cv2.medianBlur(mask,5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7),np.uint8))
    mask = cv2.erode(mask, np.ones((9,9),np.uint8),1)
    return mask

def motion_strength(prev_stab, cur_stab):
    d = cv2.absdiff(prev_stab, cur_stab)
    g = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g,(7,7),0)
    return g

def sideline_penalty(x,y,W,H,margin_xy=(64,76),heavy=22.0):
    return 0.0

# ---------- scores ----------
def ncc_score(win,tpl):
    if win.size==0 or tpl.size==0: return -1.0
    if win.shape[0]<tpl.shape[0] or win.shape[1]<tpl.shape[1]: return -1.0
    g=eq(win); t=eq(tpl)
    r1=cv2.matchTemplate(g,t,cv2.TM_CCOEFF_NORMED).max()
    h,w=t.shape[:2]
    t2=cv2.resize(t,(max(3,int(w*0.75)),max(3,int(h*0.75))),cv2.INTER_AREA)
    r2=cv2.matchTemplate(g,t2,cv2.TM_CCOEFF_NORMED).max()
    return float(max(r1,r2))

def radial_grad(gray,cx,cy,r):
    r=int(max(6,min(24,r)))
    x0=int(max(0,cx-r)); y0=int(max(0,cy-r)); x1=int(min(gray.shape[1],cx+r)); y1=int(min(gray.shape[0],cy+r))
    roi = gray[y0:y1,x0:x1]
    if roi.size==0: return 0.0
    gx=cv2.Sobel(roi,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(roi,cv2.CV_32F,0,1,ksize=3)
    mag=np.sqrt(gx*gx+gy*gy)
    return float(np.percentile(mag,80))

# ---------- candidate generator with pass-cone ----------
def gen_candidates(stab_bgr, raw_bgr, field_m, mot_map, pred_xy, tpl,
                   search_r=380, cone_dir=None, cone_deg=999, min_r=4, max_r=28, max_cands=16):
    H,W=stab_bgr.shape[:2]; px,py=pred_xy
    x0=int(max(0,px-search_r)); y0=int(max(0,py-search_r))
    x1=int(min(W,px+search_r)); y1=int(min(H,py+search_r))
    if x1<=x0+2 or y1<=y0+2: return []

    roi = stab_bgr[y0:y1,x0:x1]; gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    f_roi = field_m[y0:y1,x0:x1]; m_roi = mot_map[y0:y1,x0:x1]
    out=[]

    def ok_cone(bx,by):
        if cone_dir is None: return True
        vx,vy = cone_dir
        ux,uy = bx-px, by-py
        nu = math.hypot(ux,uy); nv = math.hypot(vx,vy)
        if nu<1e-3 or nv<1e-3: return True
        cosang = (ux*vx + uy*vy)/(nu*nv)
        ang = math.degrees(math.acos(max(-1.0,min(1.0,cosang))))
        return ang <= cone_deg

    # contours
    thr = cv2.adaptiveThreshold(eq(gray),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,-7)
    thr = cv2.medianBlur(thr,5)
    cnts,_=cv2.findContours(thr,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        a=cv2.contourArea(c)
        if a<(min_r*min_r*0.6) or a>(max_r*max_r*4.0): continue
        (cx,cy),rad = cv2.minEnclosingCircle(c)
        bx=x0+cx; by=y0+cy
        if f_roi[int(np.clip(cy,0,f_roi.shape[0]-1)), int(np.clip(cx,0,f_roi.shape[1]-1))]==0: continue
        mot = float(m_roi[int(np.clip(cy,0,m_roi.shape[0]-1)), int(np.clip(cx,0,m_roi.shape[1]-1))])
        if mot < 3.0: continue
        if not ok_cone(bx,by): continue
        dist = hypot(bx-px, by-py)
        side=int(max(16,min(96, rad*6)))
        sx0=int(max(0,bx-side//2)); sy0=int(max(0,by-side//2))
        sx1=int(min(W,sx0+side));    sy1=int(min(H,sy0+side))
        win = cv2.cvtColor(stab_bgr[sy0:sy1,sx0:sx1], cv2.COLOR_BGR2GRAY)
        ncc=ncc_score(win, tpl); grad=radial_grad(gray,int(cx),int(cy),int(rad))
        base = (-0.02*dist) + (1.5*ncc) + (0.02*grad)
        base -= sideline_penalty(bx,by,W,H)
        out.append((bx,by,base))
    # hough
    hc = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=14,
                          param1=110, param2=18, minRadius=min_r, maxRadius=max_r)
    if hc is not None:
        for x,y,r in np.round(hc[0,:]).astype(int):
            bx=x0+x; by=y0+y
            if f_roi[int(np.clip(y,0,f_roi.shape[0]-1)), int(np.clip(x,0,f_roi.shape[1]-1))]==0: continue
            mot = float(m_roi[int(np.clip(y,0,m_roi.shape[0]-1)), int(np.clip(x,0,m_roi.shape[1]-1))])
            if mot < 3.0: continue
            if not ok_cone(bx,by): continue
            dist = hypot(bx-px, by-py)
            side=int(max(16,min(96, r*6)))
            sx0=int(max(0,bx-side//2)); sy0=int(max(0,by-side//2))
            sx1=int(min(W,sx0+side));    sy1=int(min(H,sy0+side))
            win = cv2.cvtColor(stab_bgr[sy0:sy1,sx0:sx1], cv2.COLOR_BGR2GRAY)
            ncc=ncc_score(win, None if tpl is None else tpl)
            grad=radial_grad(gray,int(x),int(y),int(r))
            base = (-0.02*dist) + (1.5*ncc) + (0.02*grad) - sideline_penalty(bx,by,W,H)
            out.append((bx,by,base))
    out.sort(key=lambda t: t[2], reverse=True)
    return out[:max_cands]

# ---------- IMM smoother (CV + CA) ----------
class IMM:
    def __init__(self, dt=1/30, p_switch=0.02):
        # state: [x,y,vx,vy,ax,ay]; CV uses ax=ay=0 with low process noise
        self.dt=dt
        self.mu=np.array([0.5,0.5])  # mode probs
        # models: 0=CV, 1=CA
        self.Q_cv=np.diag([1e-3,1e-3, 2e-2,2e-2, 1e-6,1e-6])
        self.Q_ca=np.diag([1e-3,1e-3, 2e-2,2e-2, 2e-2,2e-2])
        self.R = np.diag([3.0,3.0])  # px meas noise
        self.P0=np.diag([50,50, 50,50, 50,50]).astype(np.float32)
        self.x0=None; self.P_cv=None; self.P_ca=None
        self.pi=np.array([[1-p_switch,p_switch],[p_switch,1-p_switch]])

    def _F(self, model):
        dt=self.dt
        if model==0: # CV
            F=np.eye(6)
            F[0,2]=dt; F[1,3]=dt
            return F
        else:        # CA
            F=np.eye(6)
            F[0,2]=dt; F[1,3]=dt
            F[0,4]=0.5*dt*dt; F[1,5]=0.5*dt*dt
            F[2,4]=dt; F[3,5]=dt
            return F

    def _Q(self, model): return self.Q_cv if model==0 else self.Q_ca

    def init(self, x,y):
        self.x_cv=np.array([x,y,0,0,0,0],dtype=np.float32)
        self.x_ca=self.x_cv.copy()
        self.P_cv=self.P0.copy(); self.P_ca=self.P0.copy()

    def step(self, z):
        # mixing
        c_j = self.pi.T @ self.mu
        mu_cond = (self.pi * self.mu[None,:]) / c_j[None,:]
        x_mix = mu_cond[0,0]*self.x_cv + mu_cond[1,0]*self.x_ca
        y_mix = mu_cond[0,1]*self.x_cv + mu_cond[1,1]*self.x_ca
        P_mix_cv = self.P_cv  # (keep simple)
        P_mix_ca = self.P_ca

        # propagate
        xs=[]; Ps=[]; lik=[]
        for m,(x0,P0) in enumerate([(x_mix,P_mix_cv),(y_mix,P_mix_ca)]):
            F=self._F(m); Q=self._Q(m)
            x_pred=F@x0; P_pred=F@P0@F.T + Q
            # update with z=[x,y]
            H=np.zeros((2,6)); H[0,0]=1; H[1,1]=1
            y=z - H@x_pred
            S=H@P_pred@H.T + self.R
            K=P_pred@H.T@np.linalg.inv(S)
            x_upd=x_pred + K@y
            P_upd=(np.eye(6)-K@H)@P_pred
            # likelihood
            ll = math.exp(-0.5*(y.T@np.linalg.inv(S)@y)) / math.sqrt((2*math.pi)**2 * np.linalg.det(S) + 1e-9)
            xs.append(x_upd); Ps.append(P_upd); lik.append(ll+1e-12)

        # mode prob update
        lik=np.array(lik); c = lik * c_j
        self.mu = c / (c.sum()+1e-12)

        # merge
        self.x = self.mu[0]*xs[0] + self.mu[1]*xs[1]
        self.P = self.mu[0]*(Ps[0]+np.outer(xs[0]-self.x,xs[0]-self.x)) + self.mu[1]*(Ps[1]+np.outer(xs[1]-self.x,xs[1]-self.x))
        self.x_cv, self.P_cv = xs[0], Ps[0]
        self.x_ca, self.P_ca = xs[1], Ps[1]
        return float(self.x[0]), float(self.x[1]), float(math.hypot(self.x[2], self.x[3]))

# ---------- zoom planner (sigmoid, edge-aware, rate-limited) ----------
def plan_zoom(xs, ys, vpx, fps, W,H, zmin=1.08, zmax=1.78, v0=420, k=0.0065, edge_m=0.14, edge_gain=0.18, rate=0.032):
    def edge_prox(bx,by):
        dl=bx/max(1,W); dr=(W-bx)/max(1,W); dt=by/max(1,H); db=(H-by)/max(1,H)
        return max(0.0, edge_m - min(dl,dr,dt,db))/max(edge_m,1e-6)
    def sigmoid(x): return 1.0/(1.0+math.exp(-x))
    zt=[]
    for i,(x,y) in enumerate(zip(xs,ys)):
        v = vpx[i] if i < len(vpx) else 0.0
        # higher speed -> zoom OUT smoothly
        w = sigmoid(k*(v - v0))  # ~0 for slow, ->1 for very fast
        zr = zmax - w*(zmax - zmin)
        zr = max(zmin, zr - edge_gain*edge_prox(x,y))
        zt.append(zr)
    # double EMA + rate limit
    for a in (0.22,0.22):
        for i in range(1,len(zt)): zt[i]=(1-a)*zt[i-1]+a*zt[i]
        for i in range(len(zt)-2,-1,-1): zt[i]=(1-a)*zt[i+1]+a*zt[i]
    for i in range(1,len(zt)):
        dz = max(-rate, min(rate, zt[i]-zt[i-1])); zt[i]=zt[i-1]+dz
    for i in range(len(zt)-2,-1,-1):
        dz = max(-rate, min(rate, zt[i]-zt[i+1])); zt[i]=zt[i+1]+dz
    # quality guard (portrait mapping)
    for i in range(len(zt)):
        crop_w = H / max(1e-6, zt[i]) * (1080/1920)
        min_crop_w = W / 1.90
        if crop_w < min_crop_w:
            zt[i] = H / (min_crop_w * (1080/1920))
    return [float(max(zmin, min(zmax, z))) for z in zt]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--init-manual", action="store_true")
    ap.add_argument("--init-t", type=float, default=0.8)
    ap.add_argument("--out", required=True)
    args=ap.parse_args()

    cap=cv2.VideoCapture(args.inp)
    if not cap.isOpened(): raise SystemExit("Cannot open input")
    fps=cap.get(cv2.CAP_PROP_FPS) or 30.0
    W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # anchors
    anchors={}
    warn_bad_frame=False
    with open(args.anchors,"r",encoding="utf-8") as f:
        for line in f:
            d=json.loads(line)
            frame_from_time=int(round(float(d["t"])*fps))
            stored=d.get("frame")
            if stored is not None:
                try:
                    frame_from_file=int(round(float(stored)))
                except (TypeError, ValueError):
                    frame_from_file=None
                else:
                    if abs(frame_from_file-frame_from_time)>1:
                        warn_bad_frame=True
            anchors[frame_from_time]=(float(d["bx"]), float(d["by"]))

    if warn_bad_frame:
        print("[WARN] Anchor frames differed from time-based index; using t*fps to align.")

    # init frame + template
    idx0=max(0,int(round(args.init_t*fps))); cap.set(cv2.CAP_PROP_POS_FRAMES, idx0)
    ok, f0 = cap.read(); 
    if not ok: raise SystemExit("Cannot read init frame")
    cv2.namedWindow("Select ball", cv2.WINDOW_NORMAL)
    r = cv2.selectROI("Select ball", f0, showCrosshair=True, fromCenter=False) if args.init_manual else None
    cv2.destroyWindow("Select ball")
    if r is None or r[2]<=0 or r[3]<=0: raise SystemExit("Use --init-manual to seed")
    bx0=r[0]+r[2]/2.0; by0=r[1]+r[3]/2.0
    g0 = eq(to_gray(f0))
    _, A0 = stabilize_step(g0, f0)
    stab0 = warp_affine(f0, A0, W, H)
    tpl = eq(to_gray(stab0))[int(by0-32):int(by0+32), int(bx0-32):int(bx0+32)].copy()
    if tpl.size==0: tpl = eq(to_gray(stab0))
    os.makedirs("out/diag_templates", exist_ok=True)
    cv2.imwrite("out/diag_templates/tpl_init.png", tpl)

    # pass 1: stabilized candidates with motion map & cone
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, fr1 = cap.read(); 
    if not ok: raise SystemExit("Empty video")
    prev_g = eq(to_gray(fr1)); prev_stab = fr1.copy()
    pred=(bx0,by0); vel=(0.0,0.0); miss=0
    path_xy=[]; speeds=[]
    trace = None; w = None
    n=0
    while True:
        ok, fr = cap.read()
        if not ok: break
        cur_g, A = stabilize_step(prev_g, fr)
        stab = warp_affine(fr, A, W, H)
        field_m = field_mask_bgr(stab)
        mot = motion_strength(prev_stab, stab)

        # pass-handoff: if speed high recently, use a cone
        speed = hypot(*vel)
        cone_dir = vel if speed>12.0 else None  # px/frame threshold

        # anchor snap
        if n in anchors: pred = anchors[n]

        sr = 240 + min(200, 40*miss)
        cands = gen_candidates(stab, fr, field_m, mot, pred, tpl, search_r=sr, cone_dir=None, cone_deg=999)

        if n==0:
            import os, csv
            os.makedirs("out\\diag_trace", exist_ok=True)
            trace = open("out\\diag_trace\\cands.csv","w",encoding="utf-8",newline="")
            w = csv.writer(trace); w.writerow(["frame","num_cands","top_score","pred_x","pred_y","miss_streak"])

        top = cands[0][2] if cands else ""
        if w is not None:
            w.writerow([n, len(cands), top, pred[0], pred[1], miss])

        if cands:
            bx,by,_ = cands[0]
            # update vel
            if path_xy:
                vx,vy = bx-path_xy[-1][0], by-path_xy[-1][1]
            else:
                vx,vy = 0.0,0.0
            vel = (0.6*vel[0]+0.4*vx, 0.6*vel[1]+0.4*vy)
            pred=(bx,by); miss=0
            # gently refresh template
            g = eq(to_gray(stab))
            cur_tpl = g[int(by-32):int(by+32), int(bx-32):int(bx+32)]
            if cur_tpl.size and tpl.size and cur_tpl.shape==tpl.shape:
                tpl = cv2.addWeighted(tpl, 0.9, cur_tpl, 0.1, 0)
        else:
            # widen globally if long miss
            miss += 1

        # IMM smoothing online (acts like look-ahead when combined with anchors)
        if n==0:
            imm=IMM(dt=1.0/fps, p_switch=0.04); imm.init(pred[0], pred[1])
        x,y,v = imm.step(np.array([pred[0],pred[1]],dtype=np.float32))
        path_xy.append((x,y)); speeds.append(v*fps)  # v is px/frame; convert to px/s
        prev_g=cur_g; prev_stab=stab; n+=1
    cap.release()

    if trace is not None:
        trace.close()

    # planned zoom from smoothed speed
    z = plan_zoom([p[0] for p in path_xy], [p[1] for p in path_xy], speeds, fps, W,H,
                  zmin=1.10, zmax=1.76, v0=480, k=0.0070, edge_m=0.14, edge_gain=0.18, rate=0.030)

    with open(args.out,"w",encoding="utf-8") as f:
        for i,(x,y) in enumerate(path_xy):
            t=i/float(fps)
            f.write(json.dumps({"t":t,"bx":float(x),"by":float(y),"z":float(z[i])})+"\n")

if __name__=="__main__":
    main()

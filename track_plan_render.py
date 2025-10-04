import argparse, json, math, pathlib, traceback, cv2, numpy as np

def clamp(v,a,b): return a if v<a else (b if v>b else v)
def polyfit_expr(ns, vs, deg):
    ns=np.asarray(ns,float); vs=np.asarray(vs,float)
    deg=max(1,min(deg, max(1,len(ns)-1)))
    cs=np.polyfit(ns,vs,deg)
    terms=[]; p=deg
    for c in cs:
        if abs(c)<1e-12: p-=1; continue
        if p==0: terms.append(f"({c:.10f})")
        elif p==1: terms.append(f"({c:.10f})*n")
        else: terms.append(f"({c:.10f})*n^{p}")
        p-=1
    return "(" + "+".join(terms) + ")" if terms else "(0)"

def moving_avg_future(x, L):
    n=len(x); y=np.empty(n)
    for i in range(n):
        j=min(n, i+L)
        y[i]=np.mean(x[i:j]) if j>i else x[i]
    return y

def vel_accel_limit(path, vmax, amax):
    p=np.array(path, float); v=np.zeros_like(p)
    for i in range(1,len(p)):
        dv = p[i]-p[i-1]
        dv = clamp(dv, -vmax, vmax)
        dv = clamp(dv, v[i-1]-amax, v[i-1]+amax)
        v[i]=dv
    out=np.zeros_like(p); out[0]=p[0]
    for i in range(1,len(p)): out[i]=out[i-1]+v[i]
    return out

def hsv_hist(img, box):
    x,y,w,h = [int(round(t)) for t in box]
    x=max(0,x); y=max(0,y); w=max(1,w); h=max(1,h)
    roi = img[y:y+h, x:x+w]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0,1],None,[32,32],[0,180,0,256])
    cv2.normalize(hist,hist,0,1,cv2.NORM_MINMAX)
    return hist

def bhatta(a,b): return cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA)

def iou(a,b):
    ax1,ay1,aw,ah=a; ax2,ay2=ax1+aw,ay1+ah
    bx1,by1,bw,bh=b; bx2,by2=bx1+bw,by1+bh
    ix1=max(ax1,bx1); iy1=max(ay1,by1); ix2=min(ax2,bx2); iy2=min(ay2,by2)
    iw=max(0, ix2-ix1); ih=max(0, iy2-iy1); inter=iw*ih
    u=aw*ah + bw*bh - inter
    return inter/u if u>0 else 0.0

def yolo_persons(frame, conf=0.15):
    try:
        from ultralytics import YOLO
        if not hasattr(yolo_persons, "model"):
            yolo_persons.model = YOLO("yolov8m.pt")
        pred = yolo_persons.model.predict(frame, classes=[0], conf=conf, iou=0.5, verbose=False)
        out=[]
        if pred and pred[0].boxes is not None and len(pred[0].boxes.xyxy)>0:
            for (x1,y1,x2,y2) in pred[0].boxes.xyxy.cpu().numpy():
                out.append( (float(x1),float(y1),float(x2-x1),float(y2-y1)) )
        return out
    except Exception:
        return []

def sample_points(box, step=0.20):
    x,y,w,h = box
    pts=[]
    nx=max(3,int(1/step)); ny=max(4,int(1/step))
    for i in range(nx):
        for j in range(ny):
            pts.append([x + (i+0.5)*w/nx, y + (j+0.5)*h/ny])
    return np.float32(pts).reshape(-1,1,2)

def bbox_from_points(pts):
    xs=pts[:,0,0]; ys=pts[:,0,1]
    x1=float(np.percentile(xs, 10)); x2=float(np.percentile(xs, 90))
    y1=float(np.percentile(ys, 10)); y2=float(np.percentile(ys, 90))
    w=max(8.0, x2-x1); h=max(16.0, y2-y1)
    return (x1,y1,w,h)

def choose_with_identity(cands, prev_box, target_hist):
    # cost = 0.55*appearance + 0.30*(1-IOU) + 0.15*center_distance
    if not cands: return None
    px,py,pw,ph = prev_box
    pcx, pcy = px+pw*0.5, py+ph*0.5
    best=None; best_cost=1e9
    for (box, hist, cx, cy, iouv) in cands:
        app = bhatta(target_hist, hist)           # 0 is best
        di  = 1.0 - iouv
        dc  = math.hypot(cx-pcx, cy-pcy) / 80.0   # ~80px ≈ 1 unit
        cost = 0.55*app + 0.30*di + 0.15*dc
        if cost < best_cost:
            best_cost, best = cost, box
    return best

def zoom_for_subject(W,H,cx,cy,w,h,zmin,zmax,margin):
    half_h_need=(h/2.0)*(1.0+margin)
    half_w_need=(w/2.0)*(1.0+margin)
    z1 = H/(2.0*max(half_h_need,1.0))
    z2 = (H*9/16)/(2.0*max(half_w_need,1.0))
    z_req = min(z1, z2)
    half_h_edge = max(24.0, min(cy, H-cy))
    half_w_edge = max(24.0, min(cx, W-cx))
    z_edge_h = H/(2.0*half_h_edge)
    z_edge_w = (H*9/16)/(2.0*half_w_edge)
    z_low = max(zmin, z_edge_h, z_edge_w, 1.0)
    return float(clamp(z_req, z_low, zmax))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="src", required=True)
    ap.add_argument("--sel", dest="sel_json", required=True)   # from select_player.py
    ap.add_argument("--vars",dest="vars_out", required=True)   # writes $cxExpr/$cyExpr/$zExpr
    ap.add_argument("--debug", dest="debug_mp4", required=True)
    ap.add_argument("--out", dest="out_mp4", required=True)
    # planner knobs
    ap.add_argument("--lookahead", type=int, default=18)
    ap.add_argument("--vmax", type=float, default=72.0)
    ap.add_argument("--amax", type=float, default=12.0)
    ap.add_argument("--edge_margin", type=float, default=0.34)
    ap.add_argument("--z_min", type=float, default=1.00)
    ap.add_argument("--z_max", type=float, default=1.18)
    args = ap.parse_args()

    try:
        sel = json.load(open(args.sel_json,"r",encoding="utf-8"))
    except Exception:
        raise SystemExit(f"Missing/invalid selection JSON: {args.sel_json}")

    cap=cv2.VideoCapture(args.src)
    if not cap.isOpened(): raise SystemExit(f"Cannot open {args.src}")
    W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps=cap.get(cv2.CAP_PROP_FPS) or sel.get("fps",30.0)

    start = int(sel["frame"])
    x0,y0,w0,h0 = sel["x"], sel["y"], sel["w"], sel["h"]
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    ok, frame = cap.read()
    if not ok: raise SystemExit("Cannot read selection frame")
    target_hist = hsv_hist(frame, (x0,y0,w0,h0))
    cur_box = (x0,y0,w0,h0)
    pts = sample_points(cur_box)

    # debug writer (full-res with box)
    fourcc=cv2.VideoWriter_fourcc(*"mp4v")
    dbg = cv2.VideoWriter(args.debug_mp4, fourcc, fps, (W,H))

    lk_params=dict(winSize=(41,41), maxLevel=4,
                   criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    cx_list=[]; cy_list=[]; bw_list=[]; bh_list=[]

    # pad head (before selection frame) by copying the first center
    head_len = start

    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for n in range(start, N):
        if n>start:
            ok, frame = cap.read()
            if not ok: break
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1) propagate with optical flow
        new_pts, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, **lk_params)
        good_new = new_pts[st==1] if new_pts is not None else np.empty((0,2))
        prop_box = cur_box
        if good_new.shape[0] >= 6:
            prop_box = bbox_from_points(new_pts)

        # 2) YOLO candidates
        dets = yolo_persons(frame, conf=0.15)

        # 3) score candidates with identity (appearance) + iou/center
        cands=[]
        px,py,pw,ph = prop_box
        pcx, pcy = px+pw*0.5, py+ph*0.5
        # propagated candidate
        phist = hsv_hist(frame, prop_box)
        cands.append( (prop_box, phist, pcx, pcy, iou(cur_box, prop_box)) )
        # detection candidates
        for d in dets:
            dx,dy,dw,dh = d
            dcx, dcy = dx+dw*0.5, dy+dh*0.5
            dhist = hsv_hist(frame, d)
            cands.append( (d, dhist, dcx, dcy, iou(cur_box, d)) )

        best = choose_with_identity(cands, cur_box, target_hist)
        if best is None:
            best = prop_box

        cur_box = best
        pts = sample_points(cur_box)  # reseed points inside current box for the next LK step

        x,y,w,h = cur_box
        cx = x+w*0.5; cy = y+h*0.5
        cx_list.append(float(clamp(cx,0,W))); cy_list.append(float(clamp(cy,0,H)))
        bw_list.append(float(max(8.0,w)));    bh_list.append(float(max(16.0,h)))

        vis = frame.copy()
        x1,y1,x2,y2 = int(x),int(y),int(x+w),int(y+h)
        cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.circle(vis,(int(cx),int(cy)),4,(0,255,0),-1)
        cv2.putText(vis,f"n={n}",(12,28),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
        dbg.write(vis)

        prev_gray=gray

    cap.release(); dbg.release()

    if len(cx_list)==0:
        # write center fallback
        p=pathlib.Path(args.vars_out); p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w",encoding="utf-8") as f:
            f.write(f"$cxExpr = \"= ({W/2.0:.3f})\"\n")
            f.write(f"$cyExpr = \"= ({H/2.0:.3f})\"\n")
            f.write(f"$zExpr  = \"= (1.00)\"\n")
        print("[WARN] No track; wrote center vars.")
        return

    # prepend head centers
    cx_head = [cx_list[0]]*head_len
    cy_head = [cy_list[0]]*head_len
    bw_head = [bw_list[0]]*head_len
    bh_head = [bh_list[0]]*head_len

    cx = np.array(cx_head + cx_list, float)
    cy = np.array(cy_head + cy_list, float)
    bw = np.array(bw_head + bw_list, float)
    bh = np.array(bh_head + bh_list, float)
    Nfull = len(cx)
    ns = np.arange(Nfull, dtype=float)

    # Path planning
    cx_f = moving_avg_future(cx, args.lookahead)
    cy_f = moving_avg_future(cy, args.lookahead)
    cx_s = vel_accel_limit(cx_f, args.vmax, args.amax)
    cy_s = vel_accel_limit(cy_f, args.vmax, args.amax)

    # compute zoom per-frame
    z_list=[]
    for i in range(Nfull):
        sp = 0.0 if i==0 else min(1.0, math.hypot(cx_s[i]-cx_s[i-1], cy_s[i]-cy_s[i-1])/12.0)
        margin = args.edge_margin*(1.0 + 0.55*sp)
        z = zoom_for_subject(W,H, cx_s[i],cy_s[i], bw[i],bh[i], args.z_min, args.z_max, margin)
        z_list.append(z)
    z=np.array(z_list)
    for i in range(1,Nfull):
        z[i]=0.20*z[i]+0.80*z[i-1]  # gentle zoom smoothing

    cx_expr = polyfit_expr(ns, cx_s, 5)
    cy_expr = polyfit_expr(ns, cy_s, 5)
    z_expr  = polyfit_expr(ns, z,   3)

    p=pathlib.Path(args.vars_out); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w",encoding="utf-8") as f:
        f.write(f"$cxExpr = \"= {cx_expr}\"\n")
        f.write(f"$cyExpr = \"= {cy_expr}\"\n")
        f.write(f"$zExpr  = \"= {z_expr}\"\n")
    print(f"[OK] Vars -> {p}")

    # FFmpeg render (spawn via parent PS; just write a simple done msg)
    print("[OK] Tracking and plan ready.")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise

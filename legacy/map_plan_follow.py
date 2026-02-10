import argparse, os, pathlib, math, cv2, numpy as np, sys, traceback

# ---------- small utils ----------
def clamp(v,a,b): return a if v<a else (b if v>b else v)
def log(x): print(x, flush=True)

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

def zoom_for_subject(W,H,cx,cy,w,h,zmin,zmax,margin):
    # Zoom enough to include subject + margin, but never tighter than 1:1 pixel
    half_h_need=(h/2.0)*(1.0+margin)
    half_w_need=(w/2.0)*(1.0+margin)
    z1 = H/(2.0*max(half_h_need,1.0))
    z2 = (H*9/16)/(2.0*max(half_w_need,1.0))
    z_req = min(z1, z2)
    # guardrails near edges (don’t let crop slam into borders)
    half_h_edge = max(24.0, min(cy, H-cy))
    half_w_edge = max(24.0, min(cx, W-cx))
    z_edge_h = H/(2.0*half_h_edge)
    z_edge_w = (H*9/16)/(2.0*half_w_edge)
    z_low = max(zmin, z_edge_h, z_edge_w, 1.0)
    return float(clamp(z_req, z_low, zmax))

def iou(a, b):
    ax1,ay1,aw,ah=a; ax2,ay2=ax1+aw,ay1+ah
    bx1,by1,bw,bh=b; bx2,by2=bx1+bw,by1+bh
    ix1=max(ax1,bx1); iy1=max(ay1,by1); ix2=min(ax2,bx2); iy2=min(ay2,by2)
    iw=max(0, ix2-ix1); ih=max(0, iy2-iy1); inter=iw*ih
    areaA=max(0,aw)*max(0,ah); areaB=max(0,bw)*max(0,bh)
    union=areaA+areaB-inter
    return inter/union if union>0 else 0.0

# ---------- YOLO helper ----------
def yolo_best_person(frame, guess=None, conf=0.15):
    H,W=frame.shape[:2]
    try:
        from ultralytics import YOLO
        if not hasattr(yolo_best_person, "model"):
            yolo_best_person.model = YOLO("yolov8m.pt")
        m = yolo_best_person.model
        pred = m.predict(frame, classes=[0], conf=conf, iou=0.5, verbose=False)
        if not pred or pred[0].boxes is None or len(pred[0].boxes.xyxy)==0: return None
        cx0,cy0 = (guess[0]+guess[2]*0.5, guess[1]+guess[3]*0.5) if guess else (W/2.0, H/2.0)
        best=None; best_s=-1
        for (x1,y1,x2,y2), sc in zip(pred[0].boxes.xyxy.cpu().numpy(), pred[0].boxes.conf.cpu().numpy()):
            w=float(x2-x1); h=float(y2-y1)
            cx=float(x1+x2)/2.0; cy=float(y1+y2)/2.0
            area = w*h
            d = ((cx-cx0)**2 + (cy-cy0)**2)**0.5
            s = float(sc)*(1.0 + area/(W*H*0.12)) * (1.0/(1.0+d/90.0))
            if guess is not None: s *= (1.0 + 0.60*iou((x1,y1,w,h), guess))
            if s>best_s: best_s=s; best=(float(x1),float(y1),float(w),float(h))
        return best
    except Exception:
        return None

# ---------- LK optical-flow tracker + continuous re-detect ----------
def sample_points(box, step=0.20):
    x,y,w,h = box
    pts=[]
    nx=max(3,int(1/step)); ny=max(4,int(1/step))
    for i in range(nx):
        for j in range(ny):
            px = x + (i+0.5)*w/nx
            py = y + (j+0.5)*h/ny
            pts.append([px,py])
    return np.float32(pts).reshape(-1,1,2)

def bbox_from_points(pts):
    xs=pts[:,0,0]; ys=pts[:,0,1]
    x1=float(np.percentile(xs, 12)); x2=float(np.percentile(xs, 88))
    y1=float(np.percentile(ys, 12)); y2=float(np.percentile(ys, 88))
    w=max(8.0, x2-x1); h=max(16.0, y2-y1)
    return (x1,y1,w,h)

def track_map(src, hint_time=None, hint_xy=None, debug_mp4=None):
    cap=cv2.VideoCapture(src)
    if not cap.isOpened(): raise SystemExit(f"Cannot open {src}")
    W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps=cap.get(cv2.CAP_PROP_FPS) or 30.0

    # start frame
    start_idx=0
    if hint_time is not None: start_idx=int(max(0, min(N-1, round(hint_time*fps))))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    ok, frame = cap.read()
    if not ok: raise SystemExit("Cannot read start frame")

    # initial bbox
    init = yolo_best_person(frame)
    if init is None:
        # fallback center box (safe)
        w=H*0.18; h=H*0.36; init=(W/2-w/2, H/2-h/2, w, h)

    # if user gave hint coordinate, prefer the det that contains it (or the closest)
    if hint_xy is not None:
        hx,hy = hint_xy
        best = yolo_best_person(frame, guess=init)
        if best:
            bx,by,bw,bh = best
            if not (bx<=hx<=bx+bw and by<=hy<=by+bh):
                # choose det closest to hint point
                alt = yolo_best_person(frame, guess=(hx-10,hy-20,20,40))
                if alt: init=alt
        else:
            alt = yolo_best_person(frame, guess=(hx-10,hy-20,20,40))
            if alt: init=alt

    # flow params
    lk_params=dict(winSize=(41,41), maxLevel=4,
                   criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    prev_gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pts=sample_points(init)
    mask_valid=np.ones((pts.shape[0],1), dtype=bool)
    cur_box=init

    # writers
    writer=None
    if debug_mp4:
        fourcc=cv2.VideoWriter_fourcc(*"mp4v")
        writer=cv2.VideoWriter(debug_mp4, fourcc, fps, (W,H))

    cx_list=[]; cy_list=[]; bw_list=[]; bh_list=[]
    for n in range(start_idx, N):
        if n>start_idx:
            ok, frame = cap.read()
            if not ok: break

        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        new_pts, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, **lk_params)
        good_new = new_pts[st==1]; good_old = pts[st==1]
        # if too few points, re-seed inside last box
        if good_new.shape[0] < max(8, pts.shape[0]*0.35):
            # try detector around the current box
            det = yolo_best_person(frame, guess=cur_box, conf=0.12)
            if det is not None: cur_box=det
            pts = sample_points(cur_box); mask_valid=np.ones((pts.shape[0],1), dtype=bool)
            prev_gray=gray
            # compute again on same frame to avoid a dead step
            new_pts, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, **lk_params)
            good_new = new_pts[st==1]; good_old = pts[st==1]

        if good_new.shape[0] >= 6:
            pts = new_pts
            cur_box = bbox_from_points(pts)
        else:
            # last resort: detector only
            det = yolo_best_person(frame, guess=cur_box, conf=0.10)
            if det is not None:
                cur_box=det; pts = sample_points(cur_box)
            # else keep cur_box as-is

        x,y,w,h = cur_box
        cx = x+w*0.5; cy = y+h*0.5
        cx_list.append(float(clamp(cx,0,W))); cy_list.append(float(clamp(cy,0,H)))
        bw_list.append(float(max(4.0,w)));    bh_list.append(float(max(8.0,h)))

        if writer is not None:
            vis=frame.copy()
            x1,y1,x2,y2 = int(x),int(y),int(x+w),int(y+h)
            cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.circle(vis,(int(cx),int(cy)),4,(0,255,0),-1)
            cv2.putText(vis,f"n={n}",(12,28),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
            writer.write(vis)

        prev_gray=gray

    cap.release()
    if writer is not None: writer.release()

    # if started mid-clip, pad head so total length matches N
    head = [cx_list[0]]*(start_idx) if cx_list else []
    cx = np.array(head + cx_list, float)
    cy = np.array(head + cy_list, float)
    bw = np.array(head + bw_list, float)
    bh = np.array(head + bh_list, float)
    return W,H,len(cx),cx,cy,bw,bh

def plan_path(W,H,N, cx,cy,bw,bh, lookahead=18, vmax=64.0, amax=10.0, edge_margin=0.36, zmin=1.00, zmax=1.22):
    # lookahead smoothing + physical limits
    cx_f = moving_avg_future(cx, lookahead)
    cy_f = moving_avg_future(cy, lookahead)
    cx_s = vel_accel_limit(cx_f, vmax, amax)
    cy_s = vel_accel_limit(cy_f, vmax, amax)
    # adaptive margin: move faster ? slightly more margin
    z_list=[]
    for i in range(N):
        sp = 0.0 if i==0 else min(1.0, math.hypot(cx_s[i]-cx_s[i-1], cy_s[i]-cy_s[i-1])/12.0)
        m = edge_margin*(1.0 + 0.55*sp)
        z = zoom_for_subject(W,H, cx_s[i],cy_s[i], bw[i],bh[i], zmin,zmax, m)
        z_list.append(z)
    z=np.array(z_list)
    # lowpass zoom to avoid pump
    for i in range(1,N): z[i]=0.25*z[i]+0.75*z[i-1]
    ns=np.arange(N,dtype=float)
    return polyfit_expr(ns,cx_s,5), polyfit_expr(ns,cy_s,5), polyfit_expr(ns,z,3)

def write_center_vars(path, W,H):
    p=pathlib.Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w",encoding="utf-8") as f:
        f.write(f"$cxExpr = \"= ({W/2.0:.3f})\"\n")
        f.write(f"$cyExpr = \"= ({H/2.0:.3f})\"\n")
        f.write(f"$zExpr  = \"= (1.00)\"\n")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--in",  dest="src", required=True)
    ap.add_argument("--out", dest="vars_out", required=True)
    ap.add_argument("--debug_mp4", default=None)
    # OPTIONAL hint: approximate time (sec) and XY to bias selection toward your mic'd player
    ap.add_argument("--hint_time", type=float, default=None)
    ap.add_argument("--hint_x", type=float, default=None)
    ap.add_argument("--hint_y", type=float, default=None)
    # planner knobs
    ap.add_argument("--lookahead", type=int, default=18)
    ap.add_argument("--vmax", type=float, default=64.0)
    ap.add_argument("--amax", type=float, default=10.0)
    ap.add_argument("--edge_margin", type=float, default=0.36)
    ap.add_argument("--z_min", type=float, default=1.00)
    ap.add_argument("--z_max", type=float, default=1.22)
    args=ap.parse_args()

    try:
        hint_xy = (args.hint_x, args.hint_y) if (args.hint_x is not None and args.hint_y is not None) else None
        W,H,N,cx,cy,bw,bh = track_map(args.src, hint_time=args.hint_time, hint_xy=hint_xy, debug_mp4=args.debug_mp4)
        if N==0:
            write_center_vars(args.vars_out, 1080, 1920)
            log("[WARN] No frames; wrote center vars.")
            return
        if np.std(cx)+np.std(cy) < 1.0:
            write_center_vars(args.vars_out, W,H)
            log("[WARN] Flat tracking; wrote center vars. Check debug MP4.")
            return
        cx_expr, cy_expr, z_expr = plan_path(W,H,N, cx,cy,bw,bh,
                                             lookahead=args.lookahead,
                                             vmax=args.vmax, amax=args.amax,
                                             edge_margin=args.edge_margin,
                                             zmin=args.z_min, zmax=args.z_max)
        p=pathlib.Path(args.vars_out); p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w",encoding="utf-8") as f:
            f.write(f"$cxExpr = \"= {cx_expr}\"\n")
            f.write(f"$cyExpr = \"= {cy_expr}\"\n")
            f.write(f"$zExpr  = \"= {z_expr}\"\n")
        log(f"[OK] Vars -> {p}")
    except Exception:
        traceback.print_exc()
        try:
            write_center_vars(args.vars_out, 1080, 1920)
            log("[FAIL] Exception; wrote center vars.")
        except Exception:
            log("[FAIL] Could not write fallback vars.")

if __name__ == "__main__":
    main()

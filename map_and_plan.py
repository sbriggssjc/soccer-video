import argparse, os, pathlib, math, cv2, numpy as np

# ---------- helpers ----------
def clamp(v,a,b): return a if v<a else (b if v>b else v)

def polyfit_expr(ns, vs, deg):
    ns=np.asarray(ns,float); vs=np.asarray(vs,float)
    deg=max(1,min(deg, max(1,len(ns)-1)))
    cs=np.polyfit(ns,vs,deg)
    # build string high->low: sum c_k * n^(deg-k)
    terms=[]
    p=deg
    for c in cs:
        if abs(c)<1e-12: p-=1; continue
        if p==0: terms.append(f"({c:.10f})")
        elif p==1: terms.append(f"({c:.10f})*n")
        else: terms.append(f"({c:.10f})*n^{p}")
        p-=1
    if not terms: return "(0)"
    return "(" + "+".join(terms) + ")"

def moving_avg_lookahead(x, look, win):
    # simple anticipatory FIR: average over [t, t+look] with window size win
    n=len(x); y=np.zeros(n)
    k=max(1,int(win)); L=max(1,int(look))
    for i in range(n):
        j2=min(n, i+L)
        seg=x[i:j2]
        if len(seg)<k: y[i]=np.mean(seg) if len(seg)>0 else x[i]
        else:
            # rolling mean on future window of length k (clamped to end)
            j2=min(n, i+k)
            y[i]=np.mean(x[i:j2])
    return y

def vel_accel_limit(path, vmax, amax):
    # path: array; vmax px/frame; amax px/frame^2
    p=np.array(path, float)
    v=np.zeros_like(p)
    # forward pass (accel limit)
    for i in range(1,len(p)):
        dv = p[i]-p[i-1]
        dv = clamp(dv, -vmax, vmax)
        # a limit relative to previous v
        dv = clamp(dv, v[i-1]-amax, v[i-1]+amax)
        v[i]=dv
    # rebuild positions
    out=np.zeros_like(p); out[0]=p[0]
    for i in range(1,len(p)): out[i]=out[i-1]+v[i]
    return out

def zoom_for_subject(W,H,cx,cy,w,h,zmin,zmax,margin):
    # 9:16 crop derived from H/z; width=(H*9/16)/z
    half_h_need=(h/2.0)*(1.0+margin)
    half_w_need=(w/2.0)*(1.0+margin)
    # required zoom so crop contains bbox with margin
    z1 = H/(2.0*max(half_h_need,1.0))
    z2 = (H*9/16)/(2.0*max(half_w_need,1.0))
    z_req = min(z1, z2)
    # edge guard: do not zoom so far that crop exceeds edges
    half_h_edge = max(8.0, min(cy, H-cy))
    half_w_edge = max(8.0, min(cx, W-cx))
    z_edge_h = H/(2.0*half_h_edge)
    z_edge_w = (H*9/16)/(2.0*half_w_edge)
    z_low = max(zmin, z_edge_h, z_edge_w, 1.0)
    return float(clamp(z_req, z_low, zmax))

def track_whole_clip(path, interactive=True, init_roi=None, conf=0.25):
    cap=cv2.VideoCapture(path)
    if not cap.isOpened(): raise SystemExit(f"Cannot open {path}")
    W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    ok, frame = cap.read()
    if not ok: raise SystemExit("Cannot read first frame")

    if interactive:
        sel = cv2.selectROI("Select the mic'd player, ENTER to confirm", frame, showCrosshair=True, fromCenter=False)
        cv2.destroyAllWindows()
        x,y,w,h = sel
        if w<3 or h<3: raise SystemExit("Selection too small")
    else:
        if init_roi is None: raise SystemExit("Need --roi or --interactive")
        x,y,w,h = init_roi

    # CSRT gives stickier tracking
    try:
        tracker = cv2.legacy.TrackerCSRT_create()
    except AttributeError:
        tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, (x,y,w,h))

    # YOLO (optional re-acquire)
    yolo=None
    try:
        from ultralytics import YOLO
        for wts in ("yolov8x.pt","yolov8l.pt","yolov8m.pt"):
            if os.path.exists(wts) or True:
                yolo=YOLO(wts); break
    except Exception:
        yolo=None

    cx_list=[]; cy_list=[]; w_list=[]; h_list=[]
    lost=0
    for n in range(N):
        if n>0:
            ok, frame=cap.read()
            if not ok: break

        ok, box = tracker.update(frame)
        if not ok and yolo is not None:
            # re-acquire near last center
            preds=yolo.predict(frame, classes=[0], conf=conf, iou=0.45, verbose=False)
            if preds and preds[0].boxes is not None and preds[0].boxes.xyxy is not None and len(preds[0].boxes.xyxy)>0:
                if len(cx_list)>0: pcx, pcy = cx_list[-1], cy_list[-1]
                else: pcx, pcy = W/2, H/2
                best=None; bestd=1e9
                for (x1,y1,x2,y2) in preds[0].boxes.xyxy.cpu().numpy():
                    cx=(x1+x2)/2; cy=(y1+y2)/2
                    d=(cx-pcx)**2+(cy-pcy)**2
                    if d<bestd: bestd=d; best=(x1,y1,x2-x1,y2-y1)
                if best is not None:
                    try:
                        tracker = (cv2.legacy.TrackerCSRT_create() if hasattr(cv2,'legacy') else cv2.TrackerCSRT_create())
                    except AttributeError:
                        tracker = cv2.TrackerCSRT_create()
                    tracker.init(frame, best)
                    ok, box = True, best

        if ok:
            x,y,w,h = box
            cx = x + w*0.5
            cy = y + h*0.5
            cx_list.append(float(clamp(cx,0,W)))
            cy_list.append(float(clamp(cy,0,H)))
            w_list.append(float(max(2.0,w)))
            h_list.append(float(max(2.0,h)))
            lost=0
        else:
            # still lost: hold last known
            if len(cx_list)>0:
                cx_list.append(cx_list[-1]); cy_list.append(cy_list[-1])
                w_list.append(w_list[-1]);   h_list.append(h_list[-1])
            else:
                cx_list.append(W/2); cy_list.append(H/2)
                w_list.append(min(W,H)*0.1); h_list.append(min(W,H)*0.1)
            lost+=1

    cap.release()
    return W,H,len(cx_list), np.array(cx_list), np.array(cy_list), np.array(w_list), np.array(h_list)

def plan_path(W,H,N, cx,cy, bw,bh,
              lookahead=12, win=10, edge_margin=0.30,
              vmax=28.0, amax=3.0, zmin=1.00, zmax=1.20):
    # Anticipate with lookahead avg, then clamp speed/accel, then compute zoom
    cx_f = moving_avg_lookahead(cx, lookahead, win)
    cy_f = moving_avg_lookahead(cy, lookahead, win)

    cx_s = vel_accel_limit(cx_f, vmax, amax)
    cy_s = vel_accel_limit(cy_f, vmax, amax)

    # zoom per-frame, with slight speed-based margin bump
    z_list=[]
    for i in range(N):
        if i==0: vx=vy=0.0
        else:    vx=cx_s[i]-cx_s[i-1]; vy=cy_s[i]-cy_s[i-1]
        sp=min(1.0, math.hypot(vx,vy)/18.0)
        m = edge_margin*(1.0 + 0.4*sp)
        z = zoom_for_subject(W,H, cx_s[i],cy_s[i], bw[i],bh[i], zmin,zmax, m)
        z_list.append(z)
    z = np.array(z_list, float)

    # Gentle zoom smoothing (EMA-like)
    outz=np.copy(z)
    for i in range(1,N):
        outz[i] = 0.35*z[i] + 0.65*outz[i-1]

    # Fit compact expressions (degree 5 for xy, 3 for z)
    ns = np.arange(N, dtype=float)
    cx_expr = polyfit_expr(ns, cx_s, deg=5)
    cy_expr = polyfit_expr(ns, cy_s, deg=5)
    z_expr  = polyfit_expr(ns, outz, deg=3)
    return cx_expr, cy_expr, z_expr

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--in",  dest="src", required=True)
    ap.add_argument("--out", dest="vars_out", required=True)
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--roi", nargs=4, type=float, default=None)

    ap.add_argument("--lookahead", type=int, default=12)  # frames (~0.4s at 30fps)
    ap.add_argument("--win",       type=int, default=10)
    ap.add_argument("--edge_margin", type=float, default=0.32)
    ap.add_argument("--vmax", type=float, default=28.0)
    ap.add_argument("--amax", type=float, default=3.0)
    ap.add_argument("--z_min", type=float, default=1.00)
    ap.add_argument("--z_max", type=float, default=1.22)
    args=ap.parse_args()

    W,H,N, cx,cy,bw,bh = track_whole_clip(args.src, interactive=args.interactive, init_roi=args.roi)
    cx_expr, cy_expr, z_expr = plan_path(
        W,H,N, cx,cy,bw,bh,
        lookahead=args.lookahead, win=args.win,
        edge_margin=args.edge_margin, vmax=args.vmax, amax=args.amax,
        zmin=args.z_min, zmax=args.z_max
    )
    # shift n so ffmpeg n=0 corresponds to first sample
    # (expressions already use n starting at 0)
    p=pathlib.Path(args.vars_out); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w",encoding="utf-8") as f:
        f.write(f"$cxExpr = \"= {cx_expr}\"\n")
        f.write(f"$cyExpr = \"= {cy_expr}\"\n")
        f.write(f"$zExpr  = \"= {z_expr}\"\n")
    print(f"[OK] Vars -> {p}")

if __name__=="__main__":
    main()

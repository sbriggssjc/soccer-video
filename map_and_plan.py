import argparse, os, pathlib, math, cv2, numpy as np

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
    p=np.array(path, float)
    v=np.zeros_like(p)
    for i in range(1,len(p)):
        dv = p[i]-p[i-1]
        dv = clamp(dv, -vmax, vmax)
        dv = clamp(dv, v[i-1]-amax, v[i-1]+amax)
        v[i]=dv
    out=np.zeros_like(p); out[0]=p[0]
    for i in range(1,len(p)): out[i]=out[i-1]+v[i]
    return out

def zoom_for_subject(W,H,cx,cy,w,h,zmin,zmax,margin):
    half_h_need=(h/2.0)*(1.0+margin)
    half_w_need=(w/2.0)*(1.0+margin)
    z1 = H/(2.0*max(half_h_need,1.0))
    z2 = (H*9/16)/(2.0*max(half_w_need,1.0))
    z_req = min(z1, z2)
    half_h_edge = max(8.0, min(cy, H-cy))
    half_w_edge = max(8.0, min(cx, W-cx))
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

def make_tracker():
    try: return cv2.legacy.TrackerCSRT_create()
    except AttributeError:
        try: return cv2.TrackerCSRT_create()
        except Exception: return cv2.TrackerKCF_create()

def track_with_redetect(src, interactive=True, init_roi=None,
                        redetect_every=10, conf=0.25, iou_keep=0.2,
                        debug_mp4=None):
    cap=cv2.VideoCapture(src)
    if not cap.isOpened(): raise SystemExit(f"Cannot open {src}")
    W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    ok, frame = cap.read()
    if not ok: raise SystemExit("Cannot read first frame")

    if interactive:
        sel=cv2.selectROI("Select the mic'd player, ENTER to confirm", frame, showCrosshair=True, fromCenter=False)
        cv2.destroyAllWindows()
        x,y,w,h=sel
        if w<4 or h<4: raise SystemExit("Selection too small")
    else:
        if init_roi is None: raise SystemExit("Need ROI when not interactive")
        x,y,w,h=init_roi

    tracker=make_tracker(); tracker.init(frame,(x,y,w,h))

    yolo=None
    try:
        from ultralytics import YOLO
        yolo=YOLO("yolov8m.pt")
    except Exception:
        yolo=None

    # Debug writer
    writer=None
    if debug_mp4:
        fourcc=cv2.VideoWriter_fourcc(*"mp4v")
        writer=cv2.VideoWriter(debug_mp4, fourcc, cap.get(cv2.CAP_PROP_FPS) or 30.0, (W,H))

    cx_list=[]; cy_list=[]; bw_list=[]; bh_list=[]
    last=(x,y,w,h); stagnant=0
    for n in range(N):
        if n>0:
            ok, frame = cap.read()
            if not ok: break

        ok, box = tracker.update(frame)
        if ok: last=box
        # decide to redetect: periodic OR stagnant OR near edge
        need_redetect = (n % redetect_every == 0) or (not ok)
        if ok:
            x,y,w,h=box
            cx=x+w*0.5; cy=y+h*0.5
            if n>5:
                dx=abs(cx-cx_list[-1]); dy=abs(cy-cy_list[-1])
                if dx<0.4 and dy<0.4: stagnant+=1
                else: stagnant=0
            if (x<8 or y<8 or x+w>W-8 or y+h>H-8) or stagnant>=6:
                need_redetect=True

        if need_redetect and yolo is not None:
            preds=yolo.predict(frame, classes=[0], conf=conf, iou=0.45, verbose=False)
            best=None; best_score=-1.0
            if preds and preds[0].boxes is not None and len(preds[0].boxes.xyxy)>0:
                # prefer boxes near last center with decent IoU to last bbox
                if ok: guess=box
                else: guess=last
                gcx=(guess[0]+guess[2]*0.5); gcy=(guess[1]+guess[3]*0.5)
                for (x1,y1,x2,y2), score in zip(preds[0].boxes.xyxy.cpu().numpy(),
                                                preds[0].boxes.conf.cpu().numpy()):
                    b=(float(x1),float(y1),float(x2-x1),float(y2-y1))
                    cxx=(b[0]+b[2]*0.5); cyy=(b[1]+b[3]*0.5)
                    d=((cxx-gcx)**2+(cyy-gcy)**2)**0.5
                    s=score* (1.0/(1.0+d/80.0))
                    if iou(b,guess)>iou_keep: s*=1.3
                    if s>best_score: best_score=s; best=b
            if best is not None:
                tracker=make_tracker(); tracker.init(frame, best); ok=True; box=best; last=best; stagnant=0

        if ok:
            x,y,w,h = box
            cx = x + w*0.5; cy = y + h*0.5
            cx_list.append(float(clamp(cx,0,W))); cy_list.append(float(clamp(cy,0,H)))
            bw_list.append(float(max(2.0,w)));    bh_list.append(float(max(2.0,h)))
        else:
            # hold last good
            if cx_list:
                cx_list.append(cx_list[-1]); cy_list.append(cy_list[-1])
                bw_list.append(bw_list[-1]); bh_list.append(bh_list[-1])
            else:
                cx_list.append(W/2); cy_list.append(H/2)
                bw_list.append(min(W,H)*0.1); bh_list.append(min(W,H)*0.1)

        if writer is not None:
            vis=frame.copy()
            bx = (x,y,w,h) if ok else last
            x1,y1,x2,y2 = int(bx[0]),int(bx[1]),int(bx[0]+bx[2]),int(bx[1]+bx[3])
            cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(vis,f"n={n}",(12,28),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
            writer.write(vis)

    cap.release()
    if writer is not None: writer.release()
    return W,H,len(cx_list), np.array(cx_list), np.array(cy_list), np.array(bw_list), np.array(bh_list)

def plan_path(W,H,N, cx,cy,bw,bh, lookahead=12, vmax=36.0, amax=4.0, edge_margin=0.34, zmin=1.00, zmax=1.24):
    cx_f = moving_avg_future(cx, lookahead)
    cy_f = moving_avg_future(cy, lookahead)
    cx_s = vel_accel_limit(cx_f, vmax, amax)
    cy_s = vel_accel_limit(cy_f, vmax, amax)
    z_list=[]
    for i in range(N):
        sp = 0.0 if i==0 else min(1.0, math.hypot(cx_s[i]-cx_s[i-1], cy_s[i]-cy_s[i-1])/18.0)
        m = edge_margin*(1.0 + 0.4*sp)
        z = zoom_for_subject(W,H, cx_s[i],cy_s[i], bw[i],bh[i], zmin,zmax, m)
        z_list.append(z)
    z=np.array(z_list)
    # light temporal smooth
    for i in range(1,N): z[i]=0.35*z[i]+0.65*z[i-1]
    ns=np.arange(N,dtype=float)
    return polyfit_expr(ns,cx_s,5), polyfit_expr(ns,cy_s,5), polyfit_expr(ns,z,3)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--in",  dest="src", required=True)
    ap.add_argument("--out", dest="vars_out", required=True)
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--debug_mp4", default=None)
    ap.add_argument("--lookahead", type=int, default=14)
    ap.add_argument("--vmax", type=float, default=36.0)
    ap.add_argument("--amax", type=float, default=4.5)
    ap.add_argument("--edge_margin", type=float, default=0.34)
    ap.add_argument("--z_min", type=float, default=1.00)
    ap.add_argument("--z_max", type=float, default=1.24)
    args=ap.parse_args()

    W,H,N, cx,cy,bw,bh = track_with_redetect(
        args.src, interactive=args.interactive, init_roi=None,
        redetect_every=10, conf=0.25, iou_keep=0.2, debug_mp4=args.debug_mp4
    )

    # Sanity check: make sure path moves
    if np.std(cx)+np.std(cy) < 2.0:
        p=pathlib.Path(args.vars_out); p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w",encoding="utf-8") as f:
            f.write(f"$cxExpr = \"= ({W/2.0:.3f})\"\n")
            f.write(f"$cyExpr = \"= ({H/2.0:.3f})\"\n")
            f.write(f"$zExpr  = \"= (1.00)\"\n")
        print("[WARN] Tracking looks flat; wrote center vars so render still runs. Check debug MP4.")
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
    print(f"[OK] Vars -> {p}")

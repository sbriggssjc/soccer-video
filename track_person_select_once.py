import argparse, os, pathlib, sys, math, cv2, numpy as np
from typing import Tuple, Optional

def center_xy(b): x1,y1,x2,y2=b; return (0.5*(x1+x2), 0.5*(y1+y2))
def wh(b): x1,y1,x2,y2=b; return (max(0.0,x2-x1), max(0.0,y2-y1))
def iou(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1,iy1=max(ax1,bx1),max(ay1,by1)
    ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw,ih=max(0.0,ix2-ix1),max(0.0,iy2-iy1)
    inter=iw*ih
    ua=(ax2-ax1)*(ay2-ay1); ub=(bx2-bx1)*(by2-by1)
    return inter/max(1e-6, ua+ub-inter)

def ema(vals, a):
    out=[]; s=None
    for v in vals:
        s=v if s is None else a*v+(1-a)*s
        out.append(float(s))
    return np.array(out, float)

def poly_expr(ns, vs, deg=3):
    ns=np.asarray(ns,float); vs=np.asarray(vs,float)
    d=min(deg, len(ns)-1); d=max(1,d)
    cs=np.polyfit(ns,vs,d)
    if d==3: c3,c2,c1,c0=cs; return f"(({c0:.8f})+({c1:.8f})*n+({c2:.8e})*n*n+({c3:.8e})*n*n*n)"
    if d==2: c2,c1,c0=cs;   return f"(({c0:.8f})+({c1:.8f})*n+({c2:.8e})*n*n)"
    c1,c0=cs;               return f"(({c0:.8f})+({c1:.8f})*n)"

def fit_zoom(ns, zs):
    ns=np.asarray(ns,float); zs=np.asarray(zs,float)
    d=min(2, len(ns)-1); d=max(1,d)
    cs=np.polyfit(ns,zs,d)
    if d==2: c2,c1,c0=cs; return f"(({c0:.8f})+({c1:.8f})*n+({c2:.8e})*n*n)"
    c1,c0=cs;             return f"(({c0:.8f})+({c1:.8f})*n)"

def to_xyxy(x,y,w,h): return (x, y, x+w, y+h)

def select_roi_interactive(frame, title="Select player ROI then press ENTER"):
    r=cv2.selectROI(title, frame, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    x,y,w,h=r
    if w<=1 or h<=1: return None
    return to_xyxy(x,y,w,h)

def choose_weights():
    for w in ["yolov8x.pt","yolov8l.pt","yolov8m.pt","yolov8n.pt"]:
        if os.path.exists(w): return w
    return "yolov8l.pt"

def write_vars(path, W,H, cx,cy,z):
    p=pathlib.Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w",encoding="utf-8") as f:
        f.write(f"$cxExpr = \"= {cx}\"\n$cyExpr = \"= {cy}\"\n$zExpr = \"= {z}\"\n$wSrc={W}\n$hSrc={H}\n")
    print(f"[OK] Wrote vars -> {p}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="vars_out", required=True)
    ap.add_argument("--lead", type=int, default=16)
    ap.add_argument("--roi", nargs=4, type=float, default=None, help="x y w h (non-interactive)")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--conf", type=float, default=0.035)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--min_area", type=float, default=0.0012)
    ap.add_argument("--edge_margin", type=float, default=0.30)  # margin fraction (0..0.5)
    ap.add_argument("--z_min", type=float, default=1.00)
    ap.add_argument("--z_max", type=float, default=1.25)
    ap.add_argument("--lost_zoomout", type=float, default=1.12)
    ap.add_argument("--smooth_alpha", type=float, default=0.18)
    ap.add_argument("--device", default="cpu")
    args=ap.parse_args()

    cap=cv2.VideoCapture(args.src)
    if not cap.isOpened(): raise RuntimeError(f"Cannot open {args.src}")
    W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    ok, first=cap.read()
    if not ok: raise RuntimeError("Cannot read first frame")

    # choose initial ROI
    if args.interactive:
        sel=select_roi_interactive(first)
        if sel is None: sel=to_xyxy(0.4*W,0.4*H,0.2*W,0.2*H)
    else:
        if args.roi is None: raise SystemExit("No --roi given and --interactive not set.")
        x,y,w,h=args.roi; sel=to_xyxy(x,y,w,h)

    try:
        from ultralytics import YOLO
    except Exception as e:
        print("[ERR] ultralytics import failed; falling back to center vars.", e)
        write_vars(args.vars_out, W,H, f"(({W/2:.2f})+0*(n+{args.lead}))", f"(({H/2:.2f})+0*(n+{args.lead}))", f"((1.00)+0*(n+{args.lead}))")
        return

    model=YOLO(choose_weights())

    ns=[]; cxs=[]; cys=[]; z_raw=[]
    last_box=np.array(sel, float)
    lost=0

    def want_crop_z(cx,cy,bx, by, bw, bh, z_min, z_max, margin):
        # Ensure full subject box inside a 9:16 crop of height H/z, width (H*9/16)/z with extra margin.
        # Compute max z allowed so that the crop half-sizes still contain (subject_box + margin).
        half_h_needed = (bh/2.0) * (1.0 + margin)
        half_w_needed = (bw/2.0) * (1.0 + margin)

        # Distances from subject center to frame edges
        top,bot = cy, H - cy
        left,right = cx, W - cx

        # The crop half sizes must be <= edge distances and >= needed subject half-size
        # half_h = H/(2z)  <= min(top,bot) and >= half_h_needed  => z >= H/(2*min(top,bot)) and z <= H/(2*half_h_needed)
        # For safety we compute an upper bound from edges:
        z_edge_h = max(1e-6, H / (2.0*max(1.0, min(top,bot))))
        z_edge_w = max(1e-6, (H*9/16) / (2.0*max(1.0, min(left,right))))
        # And a lower bound from subject size (don’t zoom tighter than subject + margin)
        z_sub_h  = max(1e-6, H / (2.0*half_h_needed))
        z_sub_w  = max(1e-6, (H*9/16) / (2.0*half_w_needed))

        # We want a z in [z_min, z_max] but also <= edge bounds and <= subject-size bounds (to keep it inside)
        z_allow = min(z_max, z_edge_h, z_edge_w, z_sub_h, z_sub_w)
        return max(z_min, z_allow)

    frame_idx=0
    while True:
        if frame_idx>0:
            ok, frame = cap.read()
            if not ok: break
        else:
            frame=first

        # detect persons
        res=model.predict(frame, verbose=False, classes=[0], conf=args.conf, iou=0.45, device=args.device)
        boxes=[]
        if res and len(res)>0 and res[0].boxes is not None and res[0].boxes.xyxy is not None:
            xy=res[0].boxes.xyxy.cpu().numpy()
            confs=res[0].boxes.conf.cpu().numpy() if res[0].boxes.conf is not None else np.zeros((xy.shape[0],),float)
            for b,c in zip(xy, confs):
                x1,y1,x2,y2=b
                if (x2-x1)*(y2-y1) >= args.min_area*W*H:
                    boxes.append((float(x1),float(y1),float(x2),float(y2), float(c)))

        # pick the detection with best IoU to last_box (ID continuity)
        best=None; best_iou=-1.0
        for (x1,y1,x2,y2,c) in boxes:
            i = iou(last_box,(x1,y1,x2,y2))
            if i>best_iou:
                best_iou=i; best=(x1,y1,x2,y2,c)

        if best and best_iou>=args.iou:
            last_box=np.array(best[:4], float)
            lost=0
        elif best:
            # if no IoU above threshold, still take the closest by center to avoid drifting
            cx_p,cy_p=center_xy(last_box)
            cand=min(boxes, key=lambda b: (center_xy(b[:4])[0]-cx_p)**2 + (center_xy(b[:4])[1]-cy_p)**2)
            last_box=np.array(cand[:4], float)
            lost=0
        else:
            lost+=1  # keep last_box, but remember we are lost

        cx,cy=center_xy(last_box); bw,bh=wh(last_box)
        ns.append(frame_idx); cxs.append(cx); cys.append(cy)

        # edge-aware zoom; if lost, zoom out a bit more
        z = want_crop_z(cx,cy,last_box[0],last_box[1],bw,bh, args.z_min, args.z_max, args.edge_margin)
        if lost>0: z = max(args.z_min, min(args.z_max, z/args.lost_zoomout))
        z_raw.append(z)

        frame_idx+=1

    cap.release()

    if len(ns)<2:
        write_vars(args.vars_out,W,H, f"(({W/2:.2f})+0*(n+{args.lead}))", f"(({H/2:.2f})+0*(n+{args.lead}))", f"((1.00)+0*(n+{args.lead}))")
        print("[WARN] Too few samples; wrote center fallback.")
        return

    # smoothing
    cx_s=ema(cxs, args.smooth_alpha)
    cy_s=ema(cys, args.smooth_alpha)
    z_s =ema(z_raw, 0.25); z_s=np.clip(z_s, args.z_min, args.z_max)

    cx_expr=poly_expr(ns, cx_s, 3).replace("n", f"(n+{args.lead})")
    cy_expr=poly_expr(ns, cy_s, 3).replace("n", f"(n+{args.lead})")
    z_expr =fit_zoom(ns, z_s   ).replace("n", f"(n+{args.lead})")

    write_vars(args.vars_out, W,H, cx_expr, cy_expr, z_expr)

if __name__=="__main__":
    main()

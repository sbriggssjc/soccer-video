import argparse, os, pathlib, math, cv2, numpy as np

# Small helpers
def ema(prev, val, a): return val if prev is None else (a*val + (1.0-a)*prev)
def poly_expr(ns, vs, deg=3):
    ns=np.asarray(ns,float); vs=np.asarray(vs,float)
    d=max(1,min(deg,len(ns)-1)); cs=np.polyfit(ns,vs,d)
    if d==3: c3,c2,c1,c0=cs; return f"(({c0:.8f})+({c1:.8f})*n+({c2:.8e})*n*n+({c3:.8e})*n*n*n)"
    if d==2: c2,c1,c0=cs;   return f"(({c0:.8f})+({c1:.8f})*n+({c2:.8e})*n*n)"
    c1,c0=cs;               return f"(({c0:.8f})+({c1:.8f})*n)"

def clamp(v,a,b): return max(a, min(b, v))

def zoom_for_subject(W,H,cx,cy,w,h,zmin,zmax,margin,speed):
    # 9:16 crop derived from H/z; width=(H*9/16)/z. Keep the box fully in crop with extra margin.
    m = margin * (1.0 + 0.6*min(1.0, speed))
    need_h = (h/2.0)*(1.0+m)
    need_w = (w/2.0)*(1.0+m)

    # Half-size limits from frame edges
    top,bot = cy, H-cy; left,right = cx, W-cx
    half_h_edge = max(8.0, min(top, bot))
    half_w_edge = max(8.0, min(left, right))

    # crop half-sizes: hh = H/(2z), hw = (H*9/16)/(2z)
    # Ensure hh >= need_h and hw >= need_w => z <= bounds_from_need
    z_from_need_h = H/(2.0*max(need_h,1.0))
    z_from_need_w = (H*9/16)/(2.0*max(need_w,1.0))
    z_upper = min(z_from_need_h, z_from_need_w, zmax)

    # Also don’t zoom in so much that crop half-size exceeds edge (avoid border hits) => z >= bounds_from_edge
    z_from_edge_h = H/(2.0*max(half_h_edge,1.0))
    z_from_edge_w = (H*9/16)/(2.0*max(half_w_edge,1.0))
    z_lower = max(zmin, z_from_edge_h, z_from_edge_w, 1.0)

    return float(clamp(z_upper, z_lower, zmax))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--in",  dest="src", required=True)
    ap.add_argument("--out", dest="vars_out", required=True)
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--roi", nargs=4, type=float, default=None, help="x y w h (if not interactive)")
    # behavior
    ap.add_argument("--lead", type=int, default=20)           # predictive lead frames
    ap.add_argument("--smooth_xy", type=float, default=0.25)  # pan smoothing
    ap.add_argument("--smooth_z",  type=float, default=0.35)  # zoom smoothing
    ap.add_argument("--edge_margin", type=float, default=0.30)# keep subject inside crop
    ap.add_argument("--z_min", type=float, default=1.00)
    ap.add_argument("--z_max", type=float, default=1.18)      # cap zoom to avoid grain
    ap.add_argument("--lost_grace", type=int, default=12)     # frames to try re-acquire before fallback zoom-out
    args=ap.parse_args()

    cap=cv2.VideoCapture(args.src)
    if not cap.isOpened(): raise SystemExit(f"Cannot open {args.src}")
    W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    ok, frame = cap.read()
    if not ok: raise SystemExit("Cannot read first frame")

    # orientation guard: we assume your source is upright; do nothing

    # pick initial ROI
    if args.interactive:
        sel = cv2.selectROI("Select the mic'd player, ENTER to confirm", frame, showCrosshair=True, fromCenter=False)
        cv2.destroyAllWindows()
        x,y,w,h = sel
        if w<2 or h<2: raise SystemExit("Selection too small. Try again.")
    else:
        if args.roi is None: raise SystemExit("Provide --roi x y w h or use --interactive")
        x,y,w,h = args.roi
    box = (x,y,w,h)

    # CSRT tracker (contrib)
    try:
        tracker = cv2.legacy.TrackerCSRT_create()
    except AttributeError:
        tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, box)

    # Optional YOLO re-detect if lost
    use_yolo = True
    yolo = None
    if use_yolo:
        try:
            from ultralytics import YOLO
            weights = "yolov8x.pt" if os.path.exists("yolov8x.pt") else ("yolov8l.pt" if os.path.exists("yolov8l.pt") else "yolov8m.pt")
            yolo = YOLO(weights)
        except Exception:
            yolo = None

    ns=[]; cxs=[]; cys=[]; zs=[]
    prev_cx=prev_cy=None
    last_vx=last_vy=0.0
    lost=0

    def bbox_center(b):
        x,y,w,h=b; return (x+w*0.5, y+h*0.5, w, h)

    n=0
    while True:
        if n>0:
            ok, frame = cap.read()
            if not ok: break

        ok, b = tracker.update(frame)
        if not ok:
            lost += 1
            # Try YOLO near last known area
            if yolo is not None:
                preds = yolo.predict(frame, classes=[0], conf=0.25, iou=0.45, verbose=False)
                if preds and preds[0].boxes is not None and preds[0].boxes.xyxy is not None:
                    xy = preds[0].boxes.xyxy.cpu().numpy()
                    # pick the person closest to previous center
                    if len(cxs)>0:
                        pcx, pcy = cxs[-1], cys[-1]
                    else:
                        pcx, pcy = W/2, H/2
                    best=None; bestd=1e9
                    for x1,y1,x2,y2 in xy:
                        cx=(x1+x2)/2; cy=(y1+y2)/2
                        d=(cx-pcx)**2 + (cy-pcy)**2
                        if d<bestd: bestd=d; best=(x1,y1,x2-x1,y2-y1)
                    if best is not None:
                        tracker = (cv2.legacy.TrackerCSRT_create() if hasattr(cv2,'legacy') else cv2.TrackerCSRT_create())
                        tracker.init(frame, best)
                        b = best; ok = True
                        lost = 0

        if not ok:
            # still lost — widen crop by zooming out a bit, keep last center if we had one
            if len(cxs)>0:
                cx,cy = cxs[-1], cys[-1]
                w_sub = h_sub = min(W,H)*0.1
            else:
                cx,cy = W/2,H/2; w_sub=h_sub=min(W,H)*0.1
        else:
            cx,cy,w_sub,h_sub = bbox_center(b)
            # clamp box inside frame
            cx=clamp(cx,0,W); cy=clamp(cy,0,H)

        # velocity
        if prev_cx is None:
            vx=vy=0.0
        else:
            vx = 0.7*last_vx + 0.3*(cx - prev_cx)
            vy = 0.7*last_vy + 0.3*(cy - prev_cy)
        last_vx,last_vy = vx,vy
        speed = min(1.0, math.sqrt(vx*vx + vy*vy)/12.0)

        # target zoom
        z_tgt = zoom_for_subject(W,H,cx,cy,w_sub,h_sub, args.z_min, args.z_max, args.edge_margin, speed)
        if not ok and lost > 0:
            # if lost, gently zoom out within limits
            z_tgt = max(args.z_min, min(args.z_max, z_tgt*0.94))

        # smoothing + prediction
        cx_s = ema(prev_cx, cx + args.lead*vx, args.smooth_xy); prev_cx=cx_s
        cy_s = ema(prev_cy, cy + args.lead*vy, args.smooth_xy); prev_cy=cy_s
        cx_s = float(clamp(cx_s,0,W)); cy_s=float(clamp(cy_s,0,H))

        if len(zs)==0: z_s=z_tgt
        else: z_s = float(args.smooth_z*z_tgt + (1.0-args.smooth_z)*zs[-1])
        z_s=float(clamp(z_s, args.z_min, args.z_max))

        ns.append(n); cxs.append(cx_s); cys.append(cy_s); zs.append(z_s)

        if ok: lost=0
        elif lost>args.lost_grace and len(zs)>0:
            # after grace, bias to minimal zoom to be safe
            zs[-1] = max(args.z_min, zs[-1]*0.92)

        n+=1

    cap.release()

    if len(ns)<2:
        cx_expr=f"(({W/2:.2f})+0*(n+{args.lead}))"
        cy_expr=f"(({H/2:.2f})+0*(n+{args.lead}))"
        z_expr =f"((1.00)+0*(n+{args.lead}))"
    else:
        cx_expr=poly_expr(ns,cxs,3).replace("n",f"(n+{args.lead})")
        cy_expr=poly_expr(ns,cys,3).replace("n",f"(n+{args.lead})")
        z_expr =poly_expr(ns,zs,2).replace("n",f"(n+{args.lead})")

    p=pathlib.Path(args.vars_out); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w",encoding="utf-8") as f:
        f.write(f"$cxExpr = \"= {cx_expr}\"\n$cyExpr = \"= {cy_expr}\"\n$zExpr = \"= {z_expr}\"\n")
    print(f"[OK] Wrote vars -> {p}")

if __name__=="__main__":
    main()

import argparse, os, pathlib, cv2, numpy as np, math
from typing import Tuple

# ---------- small helpers ----------
def cxcywh(xyxy):
    x1,y1,x2,y2 = xyxy; w=max(1.0,x2-x1); h=max(1.0,y2-y1)
    return (x1+x2)*0.5, (y1+y2)*0.5, w, h

def iou(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1,iy1=max(ax1,bx1), max(ay1,by1); ix2,iy2=min(ax2,bx2), min(ay2,by2)
    iw,ih=max(0.0,ix2-ix1), max(0.0,iy2-iy1)
    inter=iw*ih
    ua=(ax2-ax1)*(ay2-ay1); ub=(bx2-bx1)*(by2-by1)
    return inter/max(1e-6, ua+ub-inter)

def hsv_hist(img, box):
    x1,y1,x2,y2 = [int(round(v)) for v in box]
    x1=max(0,x1); y1=max(0,y1); x2=min(img.shape[1]-1,x2); y2=min(img.shape[0]-1,y2)
    if x2<=x1 or y2<=y1: return None
    crop = img[y1:y2, x1:x2]
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0,1,2],None,[16,16,16],[0,180,0,256,0,256])
    hist = cv2.normalize(hist, None).flatten().astype(np.float32)
    return hist

def hist_sim(h1,h2):  # 1 - Bhattacharyya (higher is better)
    if h1 is None or h2 is None: return 0.0
    d = cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)
    return max(0.0, 1.0 - float(d))

def ema(prev, val, a):
    return val if prev is None else (a*val + (1.0-a)*prev)

def poly_expr(ns, vs, deg=3):
    ns = np.asarray(ns, float); vs=np.asarray(vs, float)
    d = max(1, min(deg, len(ns)-1)); cs=np.polyfit(ns,vs,d)
    if d==3: c3,c2,c1,c0=cs; return f"(({c0:.8f})+({c1:.8f})*n+({c2:.8e})*n*n+({c3:.8e})*n*n*n)"
    if d==2: c2,c1,c0=cs;   return f"(({c0:.8f})+({c1:.8f})*n+({c2:.8e})*n*n)"
    c1,c0=cs;               return f"(({c0:.8f})+({c1:.8f})*n)"

# ---------- main ----------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--in",  dest="src", required=True)
    ap.add_argument("--out", dest="vars_out", required=True)
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--roi", nargs=4, type=float, default=None, help="x y w h (non-interactive)")
    ap.add_argument("--lead", type=int, default=18)           # predictive lead
    ap.add_argument("--conf", type=float, default=0.03)       # YOLO conf
    ap.add_argument("--iou_gate", type=float, default=0.30)   # IoU gate for ID lock
    ap.add_argument("--sim_w", type=float, default=0.55)      # weight for appearance
    ap.add_argument("--iou_w", type=float, default=0.45)      # weight for IoU
    ap.add_argument("--min_area", type=float, default=0.0010) # min person area vs frame
    ap.add_argument("--edge_margin", type=float, default=0.28)# subject margin inside crop
    ap.add_argument("--z_min", type=float, default=1.00)
    ap.add_argument("--z_max", type=float, default=1.22)      # cap zoom to avoid grain
    ap.add_argument("--lost_relax", type=float, default=1.16) # zoom-out factor when lost
    ap.add_argument("--smooth_xy", type=float, default=0.22)  # EMA for cx,cy
    ap.add_argument("--smooth_z",  type=float, default=0.30)  # EMA for z
    ap.add_argument("--device", default="cpu")
    args=ap.parse_args()

    cap=cv2.VideoCapture(args.src)
    if not cap.isOpened(): raise SystemExit(f"Cannot open {args.src}")
    W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    ok, frame = cap.read()
    if not ok: raise SystemExit("Cannot read first frame")

    # pick initial box
    if args.interactive:
        sel = cv2.selectROI("Select mic'd player, ENTER to confirm", frame, showCrosshair=True, fromCenter=False)
        cv2.destroyAllWindows()
        x,y,w,h = sel
        if w<2 or h<2: x,y,w,h = 0.4*W, 0.4*H, 0.2*W, 0.2*H
        box = (x, y, x+w, y+h)
    else:
        if args.roi is None:
            raise SystemExit("Provide --roi x y w h or use --interactive")
        x,y,w,h=args.roi; box=(x,y,x+w,y+h)

    target_hist = hsv_hist(frame, box)

    # YOLO
    from ultralytics import YOLO
    weights = "yolov8x.pt" if os.path.exists("yolov8x.pt") else ("yolov8l.pt" if os.path.exists("yolov8l.pt") else "yolov8m.pt")
    model = YOLO(weights)

    ns=[]; cxs=[]; cys=[]; zs=[]
    prev_cx=prev_cy=None
    last_box = np.array(box, float)
    last_vx=last_vy=0.0
    lost=0

    def choose_det(dets):
        nonlocal last_box, target_hist
        best=None; best_score=-1.0
        for (x1,y1,x2,y2,c,hvec) in dets:
            IoU = iou(last_box, (x1,y1,x2,y2))
            SIM = hist_sim(target_hist, hvec)
            score = args.sim_w*SIM + args.iou_w*IoU
            if score>best_score: best_score=score; best=(x1,y1,x2,y2,c,IoU,SIM)
        return best, best_score

    def zoom_for_subject(cx,cy,w_sub,h_sub, zmin, zmax, margin, speed):
        # crop height = H/z, crop width=(H*9/16)/z
        # keep subject fully inside crop with extra margin, plus velocity-based cushion
        m = margin * (1.0 + 0.6*min(1.0, speed))  # widen when moving fast
        half_h_need = (h_sub/2.0)*(1.0+m)
        half_w_need = (w_sub/2.0)*(1.0+m)
        # edge limits: crop half-size cannot exceed distance to each edge
        top,bot = cy, H-cy; left,right = cx, W-cx
        half_h_edge = max(8.0, min(top,bot))
        half_w_edge = max(8.0, min(left,right))
        # z upper bound so crop half-size >= need and <= edge:
        # half_h = H/(2z) >= half_h_need  -> z <= H/(2*half_h_need)
        # half_h <= half_h_edge           -> z >= H/(2*half_h_edge)  (don’t zoom in more than edges allow)
        # same for width
        z_from_need_h = H/(2.0*max(half_h_need,1.0))
        z_from_need_w = (H*9/16)/(2.0*max(half_w_need,1.0))
        z_from_edge_h = H/(2.0*max(half_h_edge,1.0))
        z_from_edge_w = (H*9/16)/(2.0*max(half_w_edge,1.0))

        # allowable z range intersection
        z_upper = min(z_from_need_h, z_from_need_w, zmax)
        z_lower = max(zmin, z_from_edge_h, z_from_edge_w, 1.0)  # don’t zoom *in* past edges
        z = max(z_lower, min(z_upper, zmax))
        return float(np.clip(z, zmin, zmax))

    # run
    frame_idx=0
    while True:
        if frame_idx>0:
            ok, frame = cap.read()
            if not ok: break

        # detect persons
        pred = model.predict(frame, verbose=False, classes=[0], conf=args.conf, iou=0.45, device=args.device)
        dets=[]
        if pred and len(pred)>0 and pred[0].boxes is not None and pred[0].boxes.xyxy is not None:
            xy = pred[0].boxes.xyxy.cpu().numpy()
            cf = pred[0].boxes.conf.cpu().numpy() if pred[0].boxes.conf is not None else np.zeros((xy.shape[0],))
            for b,c in zip(xy, cf):
                x1,y1,x2,y2=b
                if (x2-x1)*(y2-y1) >= args.min_area*W*H:
                    dets.append( (float(x1),float(y1),float(x2),float(y2), float(c), hsv_hist(frame,(x1,y1,x2,y2))) )

        picked=None; score=-1.0
        if dets:
            picked, score = choose_det(dets)

        if picked is not None:
            x1,y1,x2,y2,c,IoU,SIM = picked
            if IoU < args.iou_gate and SIM < 0.35:
                lost += 1
            else:
                last_box = np.array([x1,y1,x2,y2], float); lost=0
        else:
            lost += 1

        cx,cy,w_sub,h_sub = cxcywh(last_box)

        # velocity (px/frame)
        if prev_cx is None:
            vx=vy=0.0
        else:
            vx = 0.7*last_vx + 0.3*(cx - prev_cx)
            vy = 0.7*last_vy + 0.3*(cy - prev_cy)
        last_vx,last_vy = vx,vy
        speed = min(1.0, math.sqrt(vx*vx + vy*vy)/12.0)

        # zoom logic (zoom out if moving fast or near edges / if lost)
        z_tgt = zoom_for_subject(cx,cy,w_sub,h_sub, args.z_min, args.z_max, args.edge_margin, speed)
        if lost>0:
            z_tgt = max(args.z_min, min(args.z_max, z_tgt/args.lost_relax))

        # smoothing / prediction
        cx_s = ema(prev_cx, cx + args.lead*vx, args.smooth_xy); prev_cx = cx_s
        cy_s = ema(prev_cy, cy + args.lead*vy, args.smooth_xy); prev_cy = cy_s
        # center clamp to frame to avoid FFmpeg crop hitting borders weirdly
        cx_s = float(np.clip(cx_s, 0, W)); cy_s = float(np.clip(cy_s, 0, H))

        if len(zs)==0: z_s=z_tgt
        else: z_s = float(args.smooth_z*z_tgt + (1.0-args.smooth_z)*zs[-1])
        z_s = float(np.clip(z_s, args.z_min, args.z_max))

        ns.append(frame_idx); cxs.append(cx_s); cys.append(cy_s); zs.append(z_s)
        frame_idx += 1

    cap.release()

    if len(ns) < 2:
        # fallback to dead-center
        cx_expr = f"(({W/2:.2f})+0*(n+{args.lead}))"
        cy_expr = f"(({H/2:.2f})+0*(n+{args.lead}))"
        z_expr  = f"((1.00)+0*(n+{args.lead}))"
    else:
        cx_expr = poly_expr(ns, cxs, 3).replace("n", f"(n+{args.lead})")
        cy_expr = poly_expr(ns, cys, 3).replace("n", f"(n+{args.lead})")
        # zoom varies slowly; lower order fit is fine
        z_expr  = poly_expr(ns, zs, 2).replace("n", f"(n+{args.lead})")

    p=pathlib.Path(args.vars_out); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w",encoding="utf-8") as f:
        f.write(f"$cxExpr = \"= {cx_expr}\"\n$cyExpr = \"= {cy_expr}\"\n$zExpr = \"= {z_expr}\"\n")
    print(f"[OK] Wrote vars -> {p}")

if __name__=="__main__":
    main()

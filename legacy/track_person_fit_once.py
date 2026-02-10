import argparse, pathlib, sys, math, os, traceback
import numpy as np, cv2
from collections import defaultdict

def center_xy(b): x1,y1,x2,y2=b; return (0.5*(x1+x2), 0.5*(y1+y2))
def wh(b): x1,y1,x2,y2=b; return (max(0.0,x2-x1), max(0.0,y2-y1))
def area(b): w,h=wh(b); return w*h

def ema(vals, alpha=0.22):
    out=[]; s=None
    for v in vals:
        s=v if s is None else alpha*v+(1-alpha)*s
        out.append(float(s))
    return np.array(out, float)

def poly_fit_expr(ns, vs, max_deg=3):
    ns=np.asarray(ns, float); vs=np.asarray(vs, float)
    deg = 1 if len(ns) < 4 else min(max_deg, 3)
    cs=np.polyfit(ns, vs, deg)
    if deg==3:
        c3,c2,c1,c0=cs; return f"(({c0:.8f})+({c1:.8f})*n+({c2:.8e})*n*n+({c3:.8e})*n*n*n)"
    if deg==2:
        c2,c1,c0=cs; return f"(({c0:.8f})+({c1:.8f})*n+({c2:.8e})*n*n)"
    c1,c0=cs;       return f"(({c0:.8f})+({c1:.8f})*n)"

def choose_weights(pref=None):
    if pref and pref.strip():
        return pref
    for w in ["yolov8l.pt","yolov8x.pt","yolov8m.pt","yolov8n.pt"]:
        if os.path.exists(w): return w
    # Will auto-download if internet; otherwise ultralytics falls back if cached.
    return "yolov8l.pt"

def write_vars(vars_out, W, H, cx_expr, cy_expr, z_expr):
    p=pathlib.Path(vars_out); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        f.write(f"$cxExpr = \"= {cx_expr}\"\n")
        f.write(f"$cyExpr = \"= {cy_expr}\"\n")
        f.write(f"$zExpr  = \"= {z_expr}\"\n")
        f.write(f"$wSrc   = {W}\n$hSrc   = {H}\n")
    print(f"[OK] Wrote vars -> {p}")

def safe_center_vars(vars_out, W, H, lead):
    cxe=f"(({W/2:.3f})+0*n)".replace("n", f"(n+{lead})")
    cye=f"(({H/2:.3f})+0*n)".replace("n", f"(n+{lead})")
    ze =f"((1.00)+0*n)".replace("n", f"(n+{lead})")
    write_vars(vars_out, W, H, cxe, cye, ze)

def main():
    try:
        from ultralytics import YOLO
    except Exception:
        print("[ERR] ultralytics import failed; writing center fallback.")
        raise

    ap=argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="vars_out", required=True)
    ap.add_argument("--lead", type=int, default=16)
    ap.add_argument("--conf", type=float, default=0.035)     # lower -> more recall
    ap.add_argument("--iou",  type=float, default=0.45)
    ap.add_argument("--min_area", type=float, default=0.0018) # accept smaller persons
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--weights", default="")                  # put 'yolov8x.pt' here if you have it
    ap.add_argument("--visualize", type=int, default=0)
    # edge-aware safety (fraction of crop half-size that must remain between subject and edge)
    ap.add_argument("--edge_margin", type=float, default=0.28)  # bigger = zooms out sooner near edges
    ap.add_argument("--z_min", type=float, default=1.00)        # never zoom in beyond fit
    ap.add_argument("--z_max", type=float, default=1.22)        # cap zoom-in (avoid grain)
    ap.add_argument("--lost_zoomout", type=float, default=1.08) # extra zoom-out when lost
    ap.add_argument("--smooth_alpha", type=float, default=0.20) # EMA smoothing for pan
    args=ap.parse_args()

    cap=cv2.VideoCapture(args.src)
    if not cap.isOpened(): raise RuntimeError(f"Cannot open: {args.src}")
    W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS=cap.get(cv2.CAP_FPS) or 30.0; N=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    print(f"[INFO] {args.src}  {W}x{H}  fps={FPS:.3f}  frames={N}")

    weights=choose_weights(args.weights); print(f"[INFO] Using weights: {weights}")
    model=YOLO(weights)

    # track state
    traj_n=[]; traj_cx=[]; traj_cy=[]; traj_w=[]; traj_h=[]
    fallback_cx=[]; fallback_cy=[];  # per-frame nearest detection center if ID breaks
    det_frames=0; id_frames=0; lost_streak=0

    last_cx, last_cy = W/2, H/2
    last_id=None

    n=0
    while True:
        ok, frame=cap.read()
        if not ok: break

        res=model.track(frame, persist=True, verbose=False, classes=[0],
                        conf=args.conf, iou=args.iou, tracker="bytetrack.yaml",
                        device=args.device)
        picked=None
        cur_id=None

        if res and res[0].boxes is not None and res[0].boxes.xyxy is not None:
            xyxy=res[0].boxes.xyxy.cpu().numpy()
            ids =res[0].boxes.id.cpu().numpy().astype(int) if res[0].boxes.id is not None else None

            # pick by (ID continuity first) then center-weighted area
            best_score=-1.0; best_idx=-1
            for i,b in enumerate(xyxy):
                w,h=wh(b); a=(w*h)/(W*H)
                if a < args.min_area: continue
                cx,cy=center_xy(b)
                center_w=1.0 - 0.6*math.hypot((cx-W/2)/(W/2),(cy-H/2)/(H/2))
                cont = 0.35 if (ids is not None and last_id is not None and i < len(ids) and ids[i]==last_id) else 0.0
                s = a*0.65 + max(0.0, center_w)*0.35 + cont
                if s>best_score: best_score=s; best_idx=i

            if best_idx>=0:
                det_frames+=1
                bx=xyxy[best_idx]; w,h=wh(bx); cx,cy=center_xy(bx)
                picked=(cx,cy,w,h); 
                if ids is not None and best_idx < len(ids): 
                    cur_id=int(ids[best_idx]); id_frames+=1

        # bookkeeping
        if picked is not None:
            last_cx, last_cy = picked[0], picked[1]
            last_id = cur_id if cur_id is not None else last_id
            lost_streak=0
            traj_n.append(n); traj_cx.append(last_cx); traj_cy.append(last_cy); traj_w.append(picked[2]); traj_h.append(picked[3])
            fallback_cx.append(last_cx); fallback_cy.append(last_cy)
        else:
            # use previous seen position; mark lost
            lost_streak += 1
            traj_n.append(n); traj_cx.append(last_cx); traj_cy.append(last_cy); traj_w.append(0.0); traj_h.append(0.0)
            fallback_cx.append(last_cx); fallback_cy.append(last_cy)

        n+=1

    cap.release()

    if len(traj_n)==0:
        print("[ERR] No samples. Writing center fallback.")
        safe_center_vars(args.vars_out, W, H, args.lead); return

    # Smooth center (EMA)
    cx_s=ema(traj_cx, alpha=args.smooth_alpha)
    cy_s=ema(traj_cy, alpha=args.smooth_alpha)

    # Edge-aware zoom: ensure subject has margin from edges; when lost, auto zoom-out
    # Compute a per-frame desired zoom z>=1.0 (1.0 = fit height). Higher z = tighter.
    z_raw=[]
    for i in range(len(traj_n)):
        # Desired crop height = H / z; half-height = H/(2z).
        # Enforce margin m * half-height from edges:
        # If subject center near top/bottom, reduce z (zoom out) so margin holds.
        m = args.edge_margin
        # When lost, add extra zoom out
        lost_bonus = args.lost_zoomout if traj_w[i]==0.0 and traj_h[i]==0.0 else 1.0

        # compute z bound from vertical edges
        # Require: cy within [m*H/(2z), H - m*H/(2z)]  => z <= H*(1- m) / (2*min(cy, H-cy))  (derived)
        cy = cy_s[i]
        dist_top = max(1.0, cy)      # avoid zero
        dist_bot = max(1.0, H-cy)
        # bound that keeps margin: half_h = H/(2z); margin = m*half_h => cy >= margin and (H-cy) >= margin
        # => cy >= m*H/(2z) and (H-cy) >= m*H/(2z)  => z <= m*H/(2*min(cy,H-cy))
        z_edge = (m*H)/(2.0*max(1.0, min(dist_top, dist_bot)))
        # But we want at least 1.0 (fit height) and at most z_max; and apply lost_bonus to zoom OUT (i.e., reduce z by factor)
        z = max(args.z_min, min(args.z_max, z_edge))
        z = max(args.z_min, min(args.z_max, z / lost_bonus))
        z_raw.append(z)

    # Smooth zoom (avoid pump) and clamp
    z_s = ema(z_raw, alpha=0.25)
    z_s = np.clip(z_s, args.z_min, args.z_max)

    # Fit cubic to cx,cy and a gentle quadratic to z
    cx_expr = poly_fit_expr(traj_n, cx_s, 3).replace("n", f"(n+{args.lead})")
    cy_expr = poly_fit_expr(traj_n, cy_s, 3).replace("n", f"(n+{args.lead})")
    # Quadratic fit for z for extra stability
    z_expr  = poly_fit_expr(traj_n, z_s, 2).replace("n", f"(n+{args.lead})")

    write_vars(args.vars_out, W, H, cx_expr, cy_expr, z_expr)
    print(f"[STATS] frames={len(traj_n)} det_frames={det_frames} id_frames={id_frames} lost_max={max(0, (np.diff(np.where(np.array(traj_w)==0.0, 1, 0), prepend=0).max() if len(traj_w)>0 else 0))}")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Always leave something renderable + write error log
        try:
            argv=sys.argv[1:]; src = argv[argv.index("--in")+1] if "--in" in argv else ""
            out = argv[argv.index("--out")+1] if "--out" in argv else ""
            lead= int(argv[argv.index("--lead")+1]) if "--lead" in argv else 16
        except Exception:
            src=""; out=""; lead=16
        W=1920; H=1080
        try:
            cap=cv2.VideoCapture(src)
            if cap.isOpened():
                W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
        except Exception: pass
        if out:
            pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                f.write(f"$cxExpr = \"= (({W/2:.3f})+0*(n+{lead}))\"\n")
                f.write(f"$cyExpr = \"= (({H/2:.3f})+0*(n+{lead}))\"\n")
                f.write(f"$zExpr  = \"= ((1.00)+0*(n+{lead}))\"\n")
                f.write(f"$wSrc={W}\n$hSrc={H}\n")
            print(f"[FALLBACK] Wrote center vars -> {out}")
            elog = pathlib.Path(out).with_suffix(".error.log")
            with open(elog, "w", encoding="utf-8") as L:
                L.write("".join(traceback.format_exc()))
            print(f"[ERRLOG] {elog}")
        else:
            print("".join(traceback.format_exc()))
        sys.exit(1)

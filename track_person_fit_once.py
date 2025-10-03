import argparse, pathlib, sys, math, os, traceback
import numpy as np, cv2
from collections import defaultdict, deque

def center_xy(b): x1,y1,x2,y2=b; return (0.5*(x1+x2), 0.5*(y1+y2))
def box_area(b): x1,y1,x2,y2=b; return max(0.0,x2-x1)*max(0.0,y2-y1)

def ema(vals, alpha=0.22):
    out=[]; s=None
    for v in vals:
        s=v if s is None else alpha*v+(1-alpha)*s
        out.append(s)
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

def choose_weights():
    for w in ["yolov8x.pt","yolov8l.pt","yolov8m.pt","yolov8n.pt"]:
        if os.path.exists(w): return w
    return "yolov8n.pt"

def write_vars(vars_out, W, H, cx_expr, cy_expr):
    p=pathlib.Path(vars_out); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        f.write(f"$cxExpr = \"= {cx_expr}\"\n")
        f.write(f"$cyExpr = \"= {cy_expr}\"\n")
        f.write(f"$wSrc   = {W}\n$hSrc   = {H}\n")
    print(f"[OK] Wrote vars -> {p}")

def safe_center_vars(vars_out, W, H, lead):
    cxe=f"(({W/2:.3f})+0*n)".replace("n", f"(n+{lead})")
    cye=f"(({H/2:.3f})+0*n)".replace("n", f"(n+{lead})")
    write_vars(vars_out, W, H, cxe, cye)

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
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--iou",  type=float, default=0.45)
    ap.add_argument("--min_area", type=float, default=0.0030)
    ap.add_argument("--min_frames", type=int, default=40)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--visualize", type=int, default=0)
    ap.add_argument("--errlog", default="")
    args=ap.parse_args()

    cap=cv2.VideoCapture(args.src)
    if not cap.isOpened(): raise RuntimeError(f"Cannot open: {args.src}")
    W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS=cap.get(cv2.CAP_FPS) or 30.0; N=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    print(f"[INFO] {args.src}  {W}x{H}  fps={FPS:.3f}  frames={N}")

    weights=choose_weights(); print(f"[INFO] Using weights: {weights}")
    model=YOLO(weights)

    from collections import defaultdict
    paths=defaultdict(list); score=defaultdict(float)
    frame_pick=[None]*N
    det_frames=0; id_frames=0

    viz=None
    if args.visualize==1:
        outp=pathlib.Path(args.vars_out)
        vpath=str(outp.with_suffix("").as_posix())+"_DEBUG.mp4"
        fourcc=cv2.VideoWriter_fourcc(*"mp4v")
        viz=cv2.VideoWriter(vpath, fourcc, FPS, (W,H))
        print(f"[DBG] Preview -> {vpath}")

    n=0; last_good=None
    while True:
        ok, frame=cap.read()
        if not ok: break
        res=model.track(frame, persist=True, verbose=False, classes=[0],
                        conf=args.conf, iou=args.iou, tracker="bytetrack.yaml",
                        device=args.device)
        best=None
        if res and res[0].boxes is not None and res[0].boxes.xyxy is not None:
            xyxy=res[0].boxes.xyxy.cpu().numpy()
            ids =res[0].boxes.id.cpu().numpy().astype(int) if res[0].boxes.id is not None else None
            best_s=-1.0
            for i,b in enumerate(xyxy):
                a=box_area(b)/(W*H)
                if a < args.min_area: continue
                cx,cy=center_xy(b)
                dx=(cx-W/2)/(W/2); dy=(cy-H/2)/(H/2)
                s=a*max(0.0, 1.0-0.6*math.sqrt(dx*dx+dy*dy))
                if s>best_s: best_s=s; best=(cx,cy,a,i)
            if best is not None:
                det_frames+=1
                frame_pick[n]=(best[0],best[1])
                if ids is not None:
                    i = best[3]
                    if i < len(ids):
                        tid=int(ids[i]); id_frames+=1
                        cx,cy,a=best[0],best[1],best[2]
                        paths[tid].append((n,cx,cy,a))
                        score[tid]+=a
        if frame_pick[n] is None:
            frame_pick[n] = last_good if last_good is not None else (W/2, H/2)
        last_good=frame_pick[n]
        if viz is not None:
            vis=frame.copy(); cx,cy=frame_pick[n]; cv2.circle(vis,(int(cx),int(cy)),6,(0,255,0),-1); viz.write(vis)
        n+=1
    cap.release(); 
    if viz is not None: viz.release()

    # choose best ID or fallback
    traj_ns=[]; traj_cx=[]; traj_cy=[]
    if score:
        best_id=max(score.items(), key=lambda kv: kv[1])[0]
        pts=paths[best_id]
        print(f"[OK] Selected ID={best_id} samples={len(pts)}")
        for (n,cx,cy,a) in pts:
            traj_ns.append(n); traj_cx.append(cx); traj_cy.append(cy)

    if len(traj_ns) < args.min_frames:
        print(f"[WARN] Using detection fallback. id_samples={len(traj_ns)}  det_frames={det_frames}  id_frames={id_frames}")
        traj_ns=list(range(len(frame_pick)))
        fp=np.array(frame_pick, float)
        traj_cx=fp[:,0].tolist(); traj_cy=fp[:,1].tolist()

    # smooth+fit
    if len(traj_ns)==0:
        print("[ERR] No samples after fallback.")
        safe_center_vars(args.vars_out, W, H, args.lead); return

    traj_cx=ema(traj_cx, alpha=0.20); traj_cy=ema(traj_cy, alpha=0.20)
    cx_expr=poly_fit_expr(traj_ns, traj_cx, 3).replace("n", f"(n+{args.lead})")
    cy_expr=poly_fit_expr(traj_ns, traj_cy, 3).replace("n", f"(n+{args.lead})")
    write_vars(args.vars_out, W, H, cx_expr, cy_expr)
    print(f"[STATS] frames={N} det_frames>min_area={det_frames} id_frames={id_frames}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # last-resort: write center vars and log the error
        import argparse, cv2, pathlib, traceback, sys
        try:
            # extract args to locate paths
            argv=sys.argv[1:]
            src = argv[argv.index("--in")+1] if "--in" in argv else ""
            out = argv[argv.index("--out")+1] if "--out" in argv else ""
            lead= int(argv[argv.index("--lead")+1]) if "--lead" in argv else 16
        except Exception:
            src=""; out=""; lead=16
        W=1920; H=1080
        try:
            cap=cv2.VideoCapture(src); 
            if cap.isOpened(): W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); cap.release()
        except Exception: pass
        if out:
            pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
            cxe=f"(({W/2:.3f})+0*n)".replace("n", f"(n+{lead})")
            cye=f"(({H/2:.3f})+0*n)".replace("n", f"(n+{lead})")
            with open(out, "w", encoding="utf-8") as f:
                f.write(f"$cxExpr = \"= {cxe}\"\n$cyExpr = \"= {cye}\"\n$wSrc={W}\n$hSrc={H}\n")
            print(f"[FALLBACK] Wrote center vars -> {out}")
        # write error log next to vars
        if out:
            elog = pathlib.Path(out).with_suffix(".error.log")
            with open(elog, "w", encoding="utf-8") as L:
                L.write("".join(traceback.format_exc()))
            print(f"[ERRLOG] {elog}")
        else:
            print("".join(traceback.format_exc()))
        sys.exit(1)

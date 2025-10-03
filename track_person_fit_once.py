import argparse, pathlib, sys, math, os
import numpy as np, cv2
from ultralytics import YOLO
from collections import defaultdict, deque

def center_xy(b):
    x1,y1,x2,y2=b
    return (0.5*(x1+x2), 0.5*(y1+y2))

def box_area(b):
    x1,y1,x2,y2=b
    return max(0.0, x2-x1)*max(0.0, y2-y1)

def ema(vals, alpha=0.22):
    out=[]; s=None
    for v in vals:
        s=v if s is None else alpha*v+(1-alpha)*s
        out.append(s)
    return np.array(out, float)

def poly_fit_expr(ns, vs, max_deg=3):
    ns=np.asarray(ns, float); vs=np.asarray(vs, float)
    if len(ns) < 4:  # too few points; fall back to linear
        deg=1
    else:
        deg = min(max_deg, max(1, len(ns)-1, 3))
    cs=np.polyfit(ns, vs, deg)
    if deg==3:
        c3,c2,c1,c0=cs; return f"(({c0:.8f})+({c1:.8f})*n+({c2:.8e})*n*n+({c3:.8e})*n*n*n)"
    if deg==2:
        c2,c1,c0=cs; return f"(({c0:.8f})+({c1:.8f})*n+({c2:.8e})*n*n)"
    c1,c0=cs;       return f"(({c0:.8f})+({c1:.8f})*n)"

def choose_weights():
    # prefer a strong local model if present
    if os.path.exists("yolov8x.pt"): return "yolov8x.pt"
    if os.path.exists("yolov8l.pt"): return "yolov8l.pt"
    if os.path.exists("yolov8m.pt"): return "yolov8m.pt"
    return "yolov8n.pt"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="vars_out", required=True)
    ap.add_argument("--lead", type=int, default=16)
    ap.add_argument("--conf", type=float, default=0.06)
    ap.add_argument("--iou",  type=float, default=0.45)
    ap.add_argument("--min_area", type=float, default=0.0035, help="min rel area of box (box_area/(W*H))")
    ap.add_argument("--min_frames", type=int, default=40, help="min samples to accept before fallback")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--visualize", type=int, default=1, help="1 = write debug preview mp4 next to vars")
    args=ap.parse_args()

    cap=cv2.VideoCapture(args.src)
    if not cap.isOpened():
        print(f"[ERR] Cannot open: {args.src}"); sys.exit(2)
    W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS=cap.get(cv2.CAP_FPS) or 30.0; N=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    print(f"[INFO] {args.src}  {W}x{H}  fps={FPS:.3f}  frames={N}")

    weights=choose_weights()
    print(f"[INFO] Using weights: {weights}")
    model=YOLO(weights)

    # Track storage
    paths=defaultdict(list)        # id -> [(n,cx,cy,a)]
    score=defaultdict(float)       # id -> score
    frame_pick=[None]*N            # chosen (cx,cy) per frame (fallback)
    det_counts=0
    id_counts=0

    # debug preview
    viz_writer=None
    if args.visualize==1:
        outp=pathlib.Path(args.vars_out)
        viz_path=str(outp.with_suffix("").as_posix())+"_DEBUG.mp4"
        fourcc=cv2.VideoWriter_fourcc(*"mp4v")
        viz_writer=cv2.VideoWriter(viz_path, fourcc, FPS, (W,H))
        print(f"[DBG] Writing debug preview -> {viz_path}")

    # tracking loop
    n=0
    last_good=None
    hold=deque([], maxlen=10)  # short memory for hold smoothing when missing
    while True:
        ok, frame=cap.read()
        if not ok: break

        res=model.track(
            frame, persist=True, verbose=False, classes=[0],
            conf=args.conf, iou=args.iou, tracker="bytetrack.yaml",
            device=args.device
        )
        best=None
        if res and res[0].boxes is not None and res[0].boxes.xyxy is not None:
            xyxy=res[0].boxes.xyxy.cpu().numpy()
            ids =res[0].boxes.id.cpu().numpy().astype(int) if res[0].boxes.id is not None else None

            # pick best det near center with enough area
            best_s=-1.0
            for i,b in enumerate(xyxy):
                a=box_area(b)/(W*H)
                if a < args.min_area: continue
                cx,cy=center_xy(b)
                dx=(cx-W/2)/(W/2); dy=(cy-H/2)/(H/2)
                s=a*max(0.0, 1.0-0.6*math.sqrt(dx*dx+dy*dy))
                if s>best_s: best_s=s; best=(cx,cy,a,i)
            if best is not None:
                det_counts+=1
                frame_pick[n]=(best[0],best[1])

            if ids is not None and best is not None:
                id_counts+=1
                # accumulate for that ID only (closest to best center)
                i = best[3]
                tid = int(ids[i]) if i < len(ids) else None
                if tid is not None:
                    cx,cy,a = best[0],best[1],best[2]
                    paths[tid].append((n,cx,cy,a))
                    dx=(cx-W/2)/(W/2); dy=(cy-H/2)/(H/2)
                    center_bias=max(0.0, 1.0-0.6*math.sqrt(dx*dx+dy*dy))
                    score[tid]+=a*center_bias

        # fill with last-good if empty for this frame (keeps continuity)
        if frame_pick[n] is None:
            val = last_good if last_good is not None else (W/2, H/2)
            frame_pick[n]=val
        else:
            last_good=frame_pick[n]
        hold.append(frame_pick[n])

        if viz_writer is not None:
            vis = frame.copy()
            # draw chosen point
            cx,cy = frame_pick[n]
            cv2.circle(vis, (int(cx),int(cy)), 6, (0,255,0), -1)
            cv2.putText(vis, f"n={n}", (12,28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            viz_writer.write(vis)

        n+=1

    cap.release()
    if viz_writer is not None: viz_writer.release()

    # select best id if tracked, else fall back to per-frame picks
    traj_ns=[]; traj_cx=[]; traj_cy=[]
    if score:
        best_id=max(score.items(), key=lambda kv: kv[1])[0]
        pts=paths[best_id]
        print(f"[OK] Selected ID={best_id}  samples={len(pts)}")
        for (n,cx,cy,a) in pts:
            traj_ns.append(n); traj_cx.append(cx); traj_cy.append(cy)

    if len(traj_ns) < args.min_frames:
        # fallback to per-frame picks (smoothed)
        print(f"[WARN] ID samples={len(traj_ns)} < min_frames={args.min_frames}; using detection fallback.")
        traj_ns=[i for i in range(len(frame_pick))]
        fp=np.array(frame_pick, float)
        traj_cx=fp[:,0].tolist()
        traj_cy=fp[:,1].tolist()

    # in extreme case (empty), write center so downstream never breaks
    if len(traj_ns)==0:
        print("[ERR] Zero samples even after fallback; writing static center to avoid pipeline break.")
        traj_ns=[0,1,2,3]
        traj_cx=[W/2]*4
        traj_cy=[H/2]*4

    # smooth and fit with lookahead
    traj_cx=ema(traj_cx, alpha=0.20)
    traj_cy=ema(traj_cy, alpha=0.20)
    cx_expr=poly_fit_expr(traj_ns, traj_cx, max_deg=3).replace("n", f"(n+{args.lead})")
    cy_expr=poly_fit_expr(traj_ns, traj_cy, max_deg=3).replace("n", f"(n+{args.lead})")

    out=pathlib.Path(args.vars_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write(f"$cxExpr = \"= {cx_expr}\"\n")
        f.write(f"$cyExpr = \"= {cy_expr}\"\n")
        f.write(f"$wSrc   = {W}\n")
        f.write(f"$hSrc   = {H}\n")

    print(f"[STATS] frames={N} det_frames_with_person>min_area={det_counts} id_frames_used={id_counts}")
    print(f"[OK] Wrote vars -> {out}")

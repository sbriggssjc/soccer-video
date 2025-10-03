import argparse, pathlib, sys, math
import numpy as np, cv2
from ultralytics import YOLO
from collections import defaultdict

def center_xy(b):
    x1,y1,x2,y2 = b
    return (0.5*(x1+x2), 0.5*(y1+y2))

def box_area(b):
    x1,y1,x2,y2 = b
    return max(0.0, x2-x1)*max(0.0, y2-y1)

def ema(vals, alpha=0.22):
    out=[]; s=None
    for v in vals:
        s=v if s is None else alpha*v+(1-alpha)*s
        out.append(s)
    return np.array(out, dtype=float)

def poly_fit_expr(ns, vs, max_deg=3):
    ns=np.asarray(ns, float); vs=np.asarray(vs, float)
    deg = min(max_deg, max(1, len(ns)-1))
    if len(ns)<12: deg=1
    cs=np.polyfit(ns, vs, deg)
    if deg==3:
        c3,c2,c1,c0=cs; return f"(({c0:.8f})+({c1:.8f})*n+({c2:.8e})*n*n+({c3:.8e})*n*n*n)"
    if deg==2:
        c2,c1,c0=cs; return f"(({c0:.8f})+({c1:.8f})*n+({c2:.8e})*n*n)"
    c1,c0=cs;       return f"(({c0:.8f})+({c1:.8f})*n)"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="vars_out", required=True)
    ap.add_argument("--lead", type=int, default=12)
    ap.add_argument("--conf", type=float, default=0.08)
    ap.add_argument("--iou",  type=float, default=0.45)
    ap.add_argument("--device", default="cpu")
    args=ap.parse_args()

    cap=cv2.VideoCapture(args.src)
    if not cap.isOpened():
        print(f"[ERR] Cannot open: {args.src}"); sys.exit(2)
    W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS=cap.get(cv2.CAP_FPS) or 30.0; N=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    print(f"[INFO] {args.src}  {W}x{H}  fps={FPS:.3f}  frames={N}")

    # Prefer a local weight file if present (avoids download issues)
    weights_path = "yolov8n.pt"
    model=YOLO(weights_path)

    # Collect trajectories (if IDs available) and per-frame detections (always)
    paths=defaultdict(list)        # id -> [(n,cx,cy,a)]
    frame_dets=defaultdict(list)   # n -> [(cx,cy,a)]
    score=defaultdict(float)       # id -> score
    n=0

    while True:
        ok, frame=cap.read()
        if not ok: break

        # Tracking attempt
        res=model.track(frame, persist=True, verbose=False,
                        classes=[0], conf=args.conf, iou=args.iou,
                        tracker="bytetrack.yaml", device=args.device)

        got_boxes = (res and res[0].boxes is not None and res[0].boxes.xyxy is not None)
        if not got_boxes:
            n+=1; continue

        boxes=res[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        ids  = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

        # Always record per-frame detections for fallback
        for b in xyxy:
            cx,cy=center_xy(b)
            a = box_area(b)/(W*H)
            frame_dets[n].append((cx,cy,a))

        # If we have IDs, accumulate path + score
        if ids is not None:
            for i, tid in enumerate(ids):
                b=xyxy[i]
                cx,cy=center_xy(b)
                a = box_area(b)/(W*H)
                dx=(cx-W/2)/(W/2); dy=(cy-H/2)/(H/2)
                center_bias=max(0.0, 1.0-0.6*math.sqrt(dx*dx+dy*dy))
                score[tid]+=a*center_bias
                paths[tid].append((n,cx,cy,a))

        n+=1

    cap.release()

    # Pick best ID if available
    best_id=None
    if score:
        best_id=max(score.items(), key=lambda kv: kv[1])[0]
        print(f"[OK] Selected ID={best_id}")
    else:
        print("[WARN] Tracker IDs unavailable; using detection fallback")

    # Build sample series
    ns=[]; cxs=[]; cys=[]

    if best_id is not None and paths[best_id]:
        for (n,cx,cy,a) in paths[best_id]:
            ns.append(n); cxs.append(cx); cys.append(cy)
    else:
        # Fallback: choose the largest near-center person each frame
        for n in sorted(frame_dets.keys()):
            cand=frame_dets[n]
            if not cand: continue
            best=None; best_s=-1
            for (cx,cy,a) in cand:
                dx=(cx-W/2)/(W/2); dy=(cy-H/2)/(H/2)
                s=a*max(0.0, 1.0-0.6*math.sqrt(dx*dx+dy*dy))
                if s>best_s: best_s=s; best=(cx,cy)
            if best:
                ns.append(n); cxs.append(best[0]); cys.append(best[1])

    if len(ns)==0:
        print("[ERR] Zero samples after fallback; no vars written.")
        sys.exit(4)

    # Smooth + lookahead
    cxs=ema(cxs, alpha=0.20)
    cys=ema(cys, alpha=0.20)

    cx_expr=poly_fit_expr(ns, cxs, max_deg=3).replace("n", f"(n+{args.lead})")
    cy_expr=poly_fit_expr(ns, cys, max_deg=3).replace("n", f"(n+{args.lead})")

    out=pathlib.Path(args.vars_out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write(f"$cxExpr = \"= {cx_expr}\"\n")
        f.write(f"$cyExpr = \"= {cy_expr}\"\n")
        f.write(f"$wSrc   = {W}\n")
        f.write(f"$hSrc   = {H}\n")
    print(f"[OK] Wrote vars -> {out}")

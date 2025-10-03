import argparse, pathlib, sys, math
import numpy as np, cv2
from ultralytics import YOLO
from collections import defaultdict

def center_xy(xyxy):
    x1,y1,x2,y2 = xyxy
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def area(xyxy):
    x1,y1,x2,y2 = xyxy
    return max(0.0,(x2-x1))*max(0.0,(y2-y1))

def poly_fit_expr(ns, vs, max_deg=3):
    ns = np.asarray(ns, dtype=np.float64)
    vs = np.asarray(vs, dtype=np.float64)
    deg = int(min(max_deg, max(1, len(ns)-1)))
    if len(ns) < 12:
        deg = 1
    cs = np.polyfit(ns, vs, deg)
    if deg == 3:
        c3,c2,c1,c0 = cs
        return f"(({c0:.8f})+({c1:.8f})*n+({c2:.8e})*n*n+({c3:.8e})*n*n*n)"
    elif deg == 2:
        c2,c1,c0 = cs
        return f"(({c0:.8f})+({c1:.8f})*n+({c2:.8e})*n*n)"
    else:
        c1,c0 = cs
        return f"(({c0:.8f})+({c1:.8f})*n)"

def ema(vals, alpha=0.22):
    out, s = [], None
    for v in vals:
        s = v if s is None else (alpha*v + (1-alpha)*s)
        out.append(s)
    return np.array(out, dtype=np.float64)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="src", required=True)
    ap.add_argument("--out", dest="vars_out", required=True)
    ap.add_argument("--lead", type=int, default=12)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--iou",  type=float, default=0.45)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.src)
    if not cap.isOpened():
        print(f"[ERR] Cannot open video: {args.src}")
        sys.exit(2)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0
    N  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    print(f"[INFO] Opened {args.src}  {W}x{H}  fps={FPS:.3f}  frames={N}")

    model = YOLO("yolov8n.pt")

    # ------- PASS 1: pick best person (size x dwell) with center bias -------
    scores = defaultdict(float)
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        res = model.track(
            frame, persist=True, verbose=False, classes=[0],
            conf=args.conf, iou=args.iou
        )
        if not res or res[0].boxes is None or res[0].boxes.xyxy is None:
            n += 1; continue
        ids = res[0].boxes.id
        xyxy = res[0].boxes.xyxy
        if ids is None or xyxy is None: 
            n += 1; continue
        ids  = ids.cpu().numpy().astype(int)
        xyxy = xyxy.cpu().numpy()
        for i, tid in enumerate(ids):
            b = xyxy[i]
            cx,cy = center_xy(b)
            a = area(b) / float(W*H)
            dx = (cx - W/2.0) / (W/2.0)
            dy = (cy - H/2.0) / (H/2.0)
            center_bias = max(0.0, 1.0 - 0.6*math.sqrt(dx*dx + dy*dy))
            scores[tid] += a * center_bias
        n += 1

    if not scores:
        print("[ERR] No persons detected; aborting.")
        sys.exit(3)

    best_id = max(scores.items(), key=lambda kv: kv[1])[0]
    print(f"[OK] Selected ID={best_id}")

    # ------- PASS 2: follow best subject; hold last when detection is missing -------
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ns, cxs, cys = [], [], []
    last_cx, last_cy = None, None
    n = 0

    while True:
        ok, frame = cap.read()
        if not ok: break
        res = model.track(
            frame, persist=True, verbose=False, classes=[0],
            conf=args.conf, iou=args.iou
        )
        cx, cy = None, None
        if res and res[0].boxes is not None and res[0].boxes.xyxy is not None:
            xyxy = res[0].boxes.xyxy.cpu().numpy()
            ids  = res[0].boxes.id
            ids  = ids.cpu().numpy().astype(int) if ids is not None else None

            if ids is not None and best_id in ids:
                i = list(ids).index(best_id)
                cx, cy = center_xy(xyxy[i])
                last_cx, last_cy = cx, cy
            elif last_cx is not None and xyxy.shape[0] > 0:
                # nearest neighbor re-association
                dmin, best = 1e18, None
                for i in range(xyxy.shape[0]):
                    bx, by = center_xy(xyxy[i])
                    d = (bx-last_cx)**2 + (by-last_cy)**2
                    if d < dmin: dmin, best = d, i
                if best is not None:
                    cx, cy = center_xy(xyxy[best])
                    last_cx, last_cy = cx, cy
                    if ids is not None:
                        best_id = int(ids[best])

        # HOLD-LAST: if nothing detected, keep prior position so we never go empty
        if cx is None and last_cx is not None:
            cx, cy = last_cx, last_cy

        if cx is not None:
            ns.append(n); cxs.append(cx); cys.append(cy)
        n += 1

    cap.release()

    if len(ns) == 0:
        print("[ERR] Zero track samples; aborting.")
        sys.exit(4)

    # Smooth + fit with look-ahead
    cxs = ema(cxs, alpha=0.22)
    cys = ema(cys, alpha=0.22)
    cx_expr = poly_fit_expr(ns, cxs, max_deg=3).replace("n", f"(n+{args.lead})")
    cy_expr = poly_fit_expr(ns, cys, max_deg=3).replace("n", f"(n+{args.lead})")

    out = pathlib.Path(args.vars_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write(f"$cxExpr = \"= {cx_expr}\"\n")
        f.write(f"$cyExpr = \"= {cy_expr}\"\n")
        f.write(f"$wSrc   = {W}\n")
        f.write(f"$hSrc   = {H}\n")

    print(f"[OK] Wrote vars -> {out}")
    sys.exit(0)

if __name__ == "__main__":
    main()

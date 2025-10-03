import argparse, pathlib, sys, math, time
import numpy as np, cv2
from ultralytics import YOLO
from collections import defaultdict

def center_xy(xyxy):
    x1,y1,x2,y2 = xyxy
    return ( (x1+x2)/2.0, (y1+y2)/2.0 )

def area(xyxy):
    x1,y1,x2,y2 = xyxy
    return max(0.0,(x2-x1))*max(0.0,(y2-y1))

def poly_fit_expr(ns, vs, max_deg=3):
    ns = np.asarray(ns, dtype=np.float64)
    vs = np.asarray(vs, dtype=np.float64)
    deg = int(min(max_deg, max(1, len(ns)-1)))
    # Robust fallback: if too few samples, deg=1
    if len(ns) < 12: deg = 1
    cs = np.polyfit(ns, vs, deg)
    # Highest power first from np.polyfit
    if deg == 3:
        c3,c2,c1,c0 = cs
        return f"(({c0:.8f})+({c1:.8f})*n+({c2:.8e})*n*n+({c3:.8e})*n*n*n)"
    elif deg == 2:
        c2,c1,c0 = cs
        return f"(({c0:.8f})+({c1:.8f})*n+({c2:.8e})*n*n)"
    else:
        c1,c0 = cs
        return f"(({c0:.8f})+({c1:.8f})*n)"

def ema_smooth(vals, alpha=0.25):
    out = []
    s = None
    for v in vals:
        s = v if s is None else (alpha*v + (1-alpha)*s)
        out.append(s)
    return np.array(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="src", required=True)
    ap.add_argument("--out", dest="vars_out", required=True)
    ap.add_argument("--lead", type=int, default=12, help="frame lead for slight anticipatory framing")
    ap.add_argument("--log",  dest="logfile", default=None)
    args = ap.parse_args()

    def log(msg):
        print(msg)
        if args.logfile:
            with open(args.logfile, "a", encoding="utf-8") as L:
                L.write(msg + "\n")

    cap = cv2.VideoCapture(args.src)
    if not cap.isOpened():
        log(f"[ERR] Failed to open video: {args.src}")
        sys.exit(2)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0
    N  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    log(f"[INFO] Opened {args.src}  {W}x{H}  fps={FPS:.3f}  frames={N}")

    model = YOLO("yolov8n.pt")
    # Pass 1: build a score per ID (visibility*time * center bias)
    scores = defaultdict(float)
    centers_last = {}  # id -> (cx,cy) last
    n = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = model.track(frame, persist=True, verbose=False, classes=[0])  # class 0 = person
        if not res or res[0].boxes is None or res[0].boxes.id is None or res[0].boxes.xyxy is None:
            n += 1
            continue

        ids  = res[0].boxes.id.cpu().numpy().astype(int)
        xyxy = res[0].boxes.xyxy.cpu().numpy()

        for i, tid in enumerate(ids):
            b = xyxy[i]
            cx,cy = center_xy(b)
            a = area(b) / float(W*H)  # normalize by frame area
            # center bias: higher if near center
            dx = (cx - W/2.0) / (W/2.0)
            dy = (cy - H/2.0) / (H/2.0)
            center_bias = max(0.0, 1.0 - 0.6*math.sqrt(dx*dx + dy*dy))
            scores[tid] += a * center_bias
            centers_last[tid] = (cx,cy)

        n += 1

    if len(scores) == 0:
        log("[ERR] No persons detected in pass-1; cannot select subject.")
        sys.exit(3)

    best_id = max(scores.items(), key=lambda kv: kv[1])[0]
    log(f"[OK] Selected ID={best_id} (max visibility/center score)")

    # Pass 2: track chosen ID with re-association to nearest if ID switches
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ns, cxs, cys = [], [], []
    last_cx, last_cy = None, None
    n = 0
    lost = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        res = model.track(frame, persist=True, verbose=False, classes=[0])
        cx, cy = None, None
        if res and res[0].boxes is not None and res[0].boxes.xyxy is not None:
            xyxy = res[0].boxes.xyxy.cpu().numpy()
            ids  = res[0].boxes.id
            ids  = ids.cpu().numpy().astype(int) if ids is not None else None

            if ids is not None and best_id in ids:
                i = list(ids).index(best_id)
                cx, cy = center_xy(xyxy[i])
                last_cx, last_cy = cx, cy
            else:
                # nearest re-link if we temporarily lost the same ID
                if last_cx is not None and xyxy.shape[0] > 0:
                    dmin, best = 1e18, None
                    for i in range(xyxy.shape[0]):
                        bx, by = center_xy(xyxy[i])
                        d = (bx-last_cx)**2 + (by-last_cy)**2
                        if d < dmin: dmin, best = d, i
                    if best is not None:
                        cx, cy = center_xy(xyxy[best])
                        last_cx, last_cy = cx, cy
                        if ids is not None:
                            best_id = int(ids[best])  # accept switch to keep following the same person
                else:
                    lost += 1

        if cx is not None:
            ns.append(n); cxs.append(cx); cys.append(cy)
        n += 1

    cap.release()

    if len(ns) == 0:
        log("[ERR] Tracked person produced zero samples; aborting.")
        sys.exit(4)

    # Smooth (EMA) before fitting
    cxs = ema_smooth(cxs, alpha=0.25)
    cys = ema_smooth(cys, alpha=0.25)

    cx_expr = poly_fit_expr(ns, cxs, max_deg=3).replace("n", f"(n+{args.lead})")
    cy_expr = poly_fit_expr(ns, cys, max_deg=3).replace("n", f"(n+{args.lead})")

    out = pathlib.Path(args.vars_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write(f"$cxExpr = \"= {cx_expr}\"\n")
        f.write(f"$cyExpr = \"= {cy_expr}\"\n")
        f.write(f"$wSrc   = {W}\n")
        f.write(f"$hSrc   = {H}\n")

    log(f"[OK] Wrote vars -> {out}")
    sys.exit(0)

if __name__ == "__main__":
    main()

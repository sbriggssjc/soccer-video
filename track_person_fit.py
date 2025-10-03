import argparse, pathlib, math, numpy as np, cv2
from ultralytics import YOLO

def choose_track(tracks, W, H):
    # prefer long + center-ish track
    best = None; best_score = -1e18
    cx0, cy0 = W/2, H/2
    for tid, pts in tracks.items():
        if len(pts) < 10:  # ignore very short fragments
            continue
        dur = pts[-1][0]-pts[0][0] + 1
        centers = np.array([[p[1], p[2]] for p in pts], dtype=np.float32)
        d = np.linalg.norm(centers - np.array([cx0,cy0]), axis=1).mean()
        score = 3.0*dur - 0.01*d
        if score > best_score:
            best, best_score = tid, score
    return best

def poly_fit_expr(ns, vs, deg=3):
    # fit polynomial v(n) = c0 + c1*n + c2*n^2 + c3*n^3
    cs = np.polyfit(ns, vs, deg)  # highest power first
    # turn into readable expr string for FFmpeg (n is the frame index)
    # numpy returns [c3, c2, c1, c0]
    if deg == 3:
        c3,c2,c1,c0 = cs
        return f"(({c0:.8f})+({c1:.8f})*n+({c2:.8e})*n*n+({c3:.8e})*n*n*n)"
    elif deg == 2:
        c2,c1,c0 = cs
        return f"(({c0:.8f})+({c1:.8f})*n+({c2:.8e})*n*n)"
    else:
        c1,c0 = cs
        return f"(({c0:.8f})+({c1:.8f})*n)"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="src", required=True)
    ap.add_argument("--out", dest="vars_out", required=True)
    ap.add_argument("--deg", type=int, default=3)
    ap.add_argument("--nlead", type=int, default=10)  # predictive lead (~0.33s @30fps)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.src)
    if not cap.isOpened(): raise SystemExit(f"Failed to open: {args.src}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    model = YOLO("yolov8n.pt")
    tracks = {}  # id -> list of (n, cx, cy)
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        res = model.track(frame, persist=True, verbose=False, classes=[0])  # person=0
        if not res or len(res)==0:
            n += 1; continue
        r = res[0]
        if r.boxes is None or r.boxes.id is None:
            n += 1; continue
        ids = r.boxes.id.cpu().numpy().astype(int)
        xyxy = r.boxes.xyxy.cpu().numpy()
        for i, box in enumerate(xyxy):
            x1,y1,x2,y2 = box
            cx, cy = float((x1+x2)/2), float((y1+y2)/2)
            tid = int(ids[i])
            tracks.setdefault(tid, []).append((n, cx, cy))
        n += 1
    cap.release()

    if not tracks: raise SystemExit("No person tracks found.")

    best_id = choose_track(tracks, W, H)
    pts = tracks[best_id]
    ns  = np.array([p[0] for p in pts], dtype=np.float32)
    cxs = np.array([p[1] for p in pts], dtype=np.float32)
    cys = np.array([p[2] for p in pts], dtype=np.float32)

    # poly fit + predictive lead
    cx_expr = poly_fit_expr(ns, cxs, args.deg).replace("n", f"(n+{args.nlead})")
    cy_expr = poly_fit_expr(ns, cys, args.deg).replace("n", f"(n+{args.nlead})")

    # emit a PowerShell vars file
    out = pathlib.Path(args.vars_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write(f"$cxExpr = \"= {cx_expr}\"\n")
        f.write(f"$cyExpr = \"= {cy_expr}\"\n")
        f.write(f"$wSrc   = {W}\n")
        f.write(f"$hSrc   = {H}\n")

if __name__ == "__main__":
    main()

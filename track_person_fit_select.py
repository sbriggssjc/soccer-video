import argparse, pathlib, math, numpy as np, cv2, sys, time
from ultralytics import YOLO

def poly_fit_expr(ns, vs, deg=3):
    ns = np.array(ns, dtype=np.float32); vs = np.array(vs, dtype=np.float32)
    deg = max(1, min(deg, len(ns)-1))
    cs = np.polyfit(ns, vs, deg)
    # emit up to cubic
    if deg == 3:
        c3,c2,c1,c0 = cs
        return f"(({c0:.8f})+({c1:.8f})*n+({c2:.8e})*n*n+({c3:.8e})*n*n*n)"
    elif deg == 2:
        c2,c1,c0 = cs
        return f"(({c0:.8f})+({c1:.8f})*n+({c2:.8e})*n*n)"
    else:
        c1,c0 = cs
        return f"(({c0:.8f})+({c1:.8f})*n)"

def center(box):
    x1,y1,x2,y2 = box
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def area(box):
    x1,y1,x2,y2 = box
    return max(0.0,(x2-x1))*max(0.0,(y2-y1))

def pick_largest_person(frame, model):
    res = model.predict(frame, verbose=False, classes=[0])
    if not res or res[0].boxes is None or res[0].boxes.xyxy is None:
        return None, None
    xyxy = res[0].boxes.xyxy.cpu().numpy()
    if xyxy.shape[0] == 0: return None, None
    # largest by area
    areas = [area(b) for b in xyxy]
    i = int(np.argmax(areas))
    return i, xyxy[i]

def interactive_pick(cap, model, fps, timeout_sec=5.0):
    """
    Show boxes for ~timeout_sec; user clicks any person box to select id.
    If no click, auto-picks the largest person in the last frame.
    Returns (track_id or None, chosen_frame_index).
    """
    n = 0
    clicked = [None]
    chosen_id, chosen_n = None, None

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked[0] = (x, y)

    cv2.namedWindow("Select Player", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Select Player", on_mouse)

    t0 = time.time()
    last_ids, last_boxes = [], []

    while True:
        ok, frame = cap.read()
        if not ok: break
        res = model.track(frame, persist=True, verbose=False, classes=[0])  # track people
        ids, boxes = [], []
        if res and res[0].boxes is not None and res[0].boxes.id is not None:
            ids  = res[0].boxes.id.cpu().numpy().astype(int).tolist()
            boxes= res[0].boxes.xyxy.cpu().numpy().tolist()

        disp = frame.copy()
        for i, b in enumerate(boxes):
            x1,y1,x2,y2 = map(int, b)
            tid = ids[i]
            cv2.rectangle(disp, (x1,y1), (x2,y2), (0,255,255), 2)
            cv2.putText(disp, f"id {tid}", (x1, max(0,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        msg = "Click your player's box (ESC=cancel). Auto-pick in {:.1f}s".format(
            max(0.0, timeout_sec-(time.time()-t0))
        )
        cv2.putText(disp, msg, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.imshow("Select Player", disp)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

        if clicked[0] is not None and boxes:
            cx, cy = clicked[0]
            # choose first box that contains click; else nearest center
            chosen = None
            for i, b in enumerate(boxes):
                x1,y1,x2,y2 = b
                if cx>=x1 and cx<=x2 and cy>=y1 and cy<=y2:
                    chosen = i; break
            if chosen is None:
                dmin, best = 1e18, None
                for i, b in enumerate(boxes):
                    bx, by = center(b)
                    d = (bx-cx)**2 + (by-cy)**2
                    if d < dmin: dmin, best = d, i
                chosen = best
            if chosen is not None:
                chosen_id = ids[chosen]
                chosen_n  = n
                break

        last_ids, last_boxes = ids, boxes
        n += 1
        if (time.time()-t0) >= timeout_sec:
            # auto-pick largest person from current frame (or last seen)
            if boxes:
                areas = [area(b) for b in boxes]
                i = int(np.argmax(areas))
                chosen_id = ids[i]
                chosen_n  = n
            else:
                # attempt predict (no IDs) on this frame
                idx, b = pick_largest_person(frame, model)
                if idx is not None:
                    chosen_id = -999  # synthetic
                    chosen_n  = n
            break

    cv2.destroyWindow("Select Player")
    return chosen_id, (chosen_n if chosen_n is not None else 0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="src", required=True)
    ap.add_argument("--out", dest="vars_out", required=True)
    ap.add_argument("--deg", type=int, default=3)
    ap.add_argument("--nlead", type=int, default=12)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.src)
    if not cap.isOpened():
        print(f"[ERR] Failed to open: {args.src}", file=sys.stderr); sys.exit(2)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0

    model = YOLO("yolov8n.pt")

    # 1) Interactive (with auto fallback)
    cap_sel = cv2.VideoCapture(args.src)
    tid, chosen_at = interactive_pick(cap_sel, model, FPS, timeout_sec=5.0)
    cap_sel.release()
    if tid is None:
        print("[ERR] No player selected and no auto-pick possible.", file=sys.stderr); sys.exit(3)

    # 2) Full pass with re-association to keep the same person
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ns, cxs, cys = [], [], []
    last_cx, last_cy = None, None
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        res = model.track(frame, persist=True, verbose=False, classes=[0])
        cx, cy = None, None
        if res and res[0].boxes is not None and res[0].boxes.id is not None:
            ids  = res[0].boxes.id.cpu().numpy().astype(int).tolist()
            xyxy = res[0].boxes.xyxy.cpu().numpy()
            if (tid in ids) and xyxy.shape[0] > 0:
                i = ids.index(tid)
                cx, cy = center(xyxy[i])
                last_cx, last_cy = cx, cy
            else:
                # re-link to nearest detection to last center
                if last_cx is not None and xyxy.shape[0] > 0:
                    dmin, best = 1e18, None
                    for i in range(xyxy.shape[0]):
                        bx, by = center(xyxy[i])
                        d = (bx-last_cx)**2 + (by-last_cy)**2
                        if d < dmin: dmin, best = d, i
                    if best is not None:
                        cx, cy = center(xyxy[best])
                        last_cx, last_cy = cx, cy
                        # accept ID switch
                        if res[0].boxes.id is not None:
                            tidc = int(res[0].boxes.id.cpu().numpy()[best])
                            tid = tidc
        if cx is not None:
            ns.append(n); cxs.append(cx); cys.append(cy)
        n += 1
    cap.release()

    if len(ns) == 0:
        print("[ERR] No target samples tracked; lighting/occlusion too heavy.", file=sys.stderr); sys.exit(4)

    # 3) Fit; if samples are few, fall back to linear
    deg = args.deg if len(ns) >= 12 else 1
    cx_expr = poly_fit_expr(ns, cxs, deg).replace("n", f"(n+{args.nlead})")
    cy_expr = poly_fit_expr(ns, cys, deg).replace("n", f"(n+{args.nlead})")

    out = pathlib.Path(args.vars_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write(f"$cxExpr = \"= {cx_expr}\"\n")
        f.write(f"$cyExpr = \"= {cy_expr}\"\n")
        f.write(f"$wSrc   = {W}\n")
        f.write(f"$hSrc   = {H}\n")
    print(f"[OK] Wrote vars -> {out}")

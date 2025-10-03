import argparse, pathlib, math, numpy as np, cv2, time
from ultralytics import YOLO

def poly_fit_expr(ns, vs, deg=3):
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

def box_contains(box, x, y):
    x1,y1,x2,y2 = box
    return (x >= x1 and x <= x2 and y >= y1 and y <= y2)

def center(box):
    x1,y1,x2,y2 = box
    return (float((x1+x2)/2), float((y1+y2)/2))

def pick_id_interactively(cap, model, max_sel_frames=150):
    """
    Show first ~5s of frames with boxes/IDs; user clicks inside the target box once.
    Returns selected track id (int) and the frame index at which it was chosen.
    """
    n = 0
    target_id = None
    clicked = [None]  # mutable closure

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked[0] = (x, y)

    cv2.namedWindow("Select Player", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Select Player", on_mouse)

    chosen_at = None
    while n < max_sel_frames:
        ok, frame = cap.read()
        if not ok: break
        h, w = frame.shape[:2]
        res = model.track(frame, persist=True, verbose=False, classes=[0])  # person
        boxes, ids = [], []
        if res and res[0].boxes is not None and res[0].boxes.id is not None:
            ids  = res[0].boxes.id.cpu().numpy().astype(int).tolist()
            xyxy = res[0].boxes.xyxy.cpu().numpy().tolist()
            boxes = xyxy

        # draw
        disp = frame.copy()
        for i, box in enumerate(boxes):
            x1,y1,x2,y2 = map(int, box)
            tid = ids[i]
            cv2.rectangle(disp, (x1,y1), (x2,y2), (0,255,255), 2)
            cv2.putText(disp, f"id {tid}", (x1, max(0,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        cv2.putText(disp, "Click inside the MIC'D PLAYER box to lock target.",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.imshow("Select Player", disp)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to abort
            break

        if clicked[0] is not None and boxes:
            cx, cy = clicked[0]
            # pick the FIRST box that contains the click
            for i, box in enumerate(boxes):
                if box_contains(box, cx, cy):
                    target_id = ids[i]
                    chosen_at = n
                    break
            # if no box contains the click, pick nearest center
            if target_id is None:
                dmin, best = 1e18, None
                for i, box in enumerate(boxes):
                    bx, by = center(box)
                    d = (bx-cx)**2 + (by-cy)**2
                    if d < dmin: dmin, best = d, ids[i]
                target_id = best
                chosen_at = n
            break
        n += 1

    cv2.destroyWindow("Select Player")
    return target_id, (chosen_at if chosen_at is not None else 0), n

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
    FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0

    model = YOLO("yolov8n.pt")

    # Interactive selection on a copy of the capture (so we can re-read from 0 afterwards)
    cap_sel = cv2.VideoCapture(args.src)
    target_id, chosen_at, seen = pick_id_interactively(cap_sel, model, max_sel_frames=int(5*FPS))
    cap_sel.release()
    if target_id is None:
        raise SystemExit("No player selected; please click on the target player's box.")

    # Now do the full pass, tracking ONLY that id; if it disappears briefly, re-link to nearest box
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ns, cxs, cys = [], [], []
    last_cx, last_cy = None, None
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        res = model.track(frame, persist=True, verbose=False, classes=[0])
        if res and res[0].boxes is not None and res[0].boxes.id is not None:
            ids  = res[0].boxes.id.cpu().numpy().astype(int).tolist()
            xyxy = res[0].boxes.xyxy.cpu().numpy()
            # find target id
            if target_id in ids:
                i = ids.index(target_id)
                cx, cy = center(xyxy[i])
                last_cx, last_cy = cx, cy
            else:
                # re-associate: pick the detection whose center is nearest to last center
                if last_cx is not None:
                    dmin, best = 1e18, None
                    for i in range(len(ids)):
                        cx_, cy_ = center(xyxy[i])
                        d = (cx_-last_cx)**2 + (cy_-last_cy)**2
                        if d < dmin:
                            dmin, best = d, i
                    if best is not None:
                        cx, cy = center(xyxy[best])
                        last_cx, last_cy = cx, cy
                        target_id = ids[best]  # hop ids if tracker changed identity
                    else:
                        cx, cy = last_cx, last_cy
                else:
                    cx, cy = None, None
            if cx is not None:
                ns.append(n); cxs.append(cx); cys.append(cy)
        n += 1
    cap.release()

    if len(ns) < 10:  # not enough samples
        raise SystemExit("Not enough target samples tracked; try clicking again or brighter footage.")

    ns  = np.array(ns, dtype=np.float32)
    cxs = np.array(cxs, dtype=np.float32)
    cys = np.array(cys, dtype=np.float32)

    cx_expr = poly_fit_expr(ns, cxs, args.deg).replace("n", f"(n+{args.nlead})")
    cy_expr = poly_fit_expr(ns, cys, args.deg).replace("n", f"(n+{args.nlead})")

    out = pathlib.Path(args.vars_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write(f"$cxExpr = \"= {cx_expr}\"\n")
        f.write(f"$cyExpr = \"= {cy_expr}\"\n")
        f.write(f"$wSrc   = {W}\n")
        f.write(f"$hSrc   = {H}\n")

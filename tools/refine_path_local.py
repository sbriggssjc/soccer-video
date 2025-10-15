import argparse, json, math, cv2, numpy as np


def read_plan(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            o = json.loads(ln)
            t = o.get("t", 0.0)
            bx = o.get("bx_stab", o.get("bx"))
            by = o.get("by_stab", o.get("by"))
            if bx is None or by is None:
                continue
            rows.append((float(t), float(bx), float(by)))
    return rows


def extract_patch(img, cx, cy, rad):
    h, w = img.shape[:2]
    x0 = max(0, int(round(cx - rad)))
    y0 = max(0, int(round(cy - rad)))
    x1 = min(w, int(round(cx + rad)))
    y1 = min(h, int(round(cy + rad)))
    if x1 <= x0 or y1 <= y0:
        return None, (x0, y0)
    return img[y0:y1, x0:x1], (x0, y0)


def refine_with_template(frame, cx, cy, tpl, rad, max_shift):
    patch, (x0, y0) = extract_patch(frame, cx, cy, rad)
    if patch is None or tpl is None:
        return (cx, cy, 0.0, False)
    if patch.shape[0] < tpl.shape[0] or patch.shape[1] < tpl.shape[1]:
        return (cx, cy, 0.0, False)
    res = cv2.matchTemplate(
        cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY),
        cv2.TM_CCOEFF_NORMED,
    )
    minv, maxv, minl, maxl = cv2.minMaxLoc(res)
    # position in patch coords of top-left
    px, py = maxl
    # centre of best match
    mx = x0 + px + tpl.shape[1] / 2.0
    my = y0 + py + tpl.shape[0] / 2.0
    # limit crazy jumps
    if abs(mx - cx) > max_shift or abs(my - cy) > max_shift:
        return (cx, cy, maxv, False)
    return (mx, my, maxv, True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="video", required=True)
    ap.add_argument("--plan", required=True, help="initial (zero-lag) plan jsonl")
    ap.add_argument("--out", required=True, help="refined plan jsonl")
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--search-r", type=int, default=64)
    ap.add_argument("--tpl-r", type=int, default=18)
    ap.add_argument("--max-shift", type=int, default=40)
    ap.add_argument("--conf-min", type=float, default=0.40)
    ap.add_argument("--update-every", type=int, default=3, help="refresh template every N frames")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit("Cannot open video")

    rows = read_plan(args.plan)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
    # if rows shorter than video, pad last sample
    if len(rows) < n:
        rows = rows + [(i / fps, rows[-1][1], rows[-1][2]) for i in range(len(rows), n)]
    rows = rows[:n]

    refined = []
    tpl = None
    for i, (t, cx, cy) in enumerate(rows):
        ok, frame = cap.read()
        if not ok:
            break

        # establish/update template periodically from the predicted centre
        if tpl is None or (i % args.update_every) == 0:
            tpl, _ = extract_patch(frame, cx, cy, args.tpl_r)
        # snap search
        mx, my, conf, used = refine_with_template(frame, cx, cy, tpl, args.search_r, args.max_shift)
        # fallback: small LK step if match weak (optional simple flow)
        if (not used) or (conf < args.conf_min):
            # try a smaller search to avoid drift
            mx, my, conf2, used2 = refine_with_template(
                frame, mx, my, tpl, max(24, args.search_r // 2), args.max_shift // 2
            )
            if used2 and conf2 > conf:
                conf, used = conf2, True

        # accept snapped if used; else keep prior
        nx, ny = (mx, my) if used else (cx, cy)
        refined.append((t, float(nx), float(ny)))

    cap.release()
    with open(args.out, "w", encoding="utf-8") as f:
        for t, x, y in refined:
            o = {"t": float(t), "bx": x, "by": y, "bx_stab": x, "by_stab": y}
            f.write(json.dumps(o, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()

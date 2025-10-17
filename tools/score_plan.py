import json, sys, cv2, numpy as np


def load_jsonl(path):
    pts = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                o = json.loads(ln)
                t = o.get("t", None)
                bx = o.get("bx_stab", o.get("bx", None))
                by = o.get("by_stab", o.get("by", None))
                if bx is not None and by is not None:
                    pts.append((t, float(bx), float(by)))
    return pts


def main():
    if len(sys.argv) < 3:
        print("usage: score_plan.py <video> <plan.jsonl> [--step 4] [--win 24] [--tpl 15] [--dump residuals.jsonl]")
        sys.exit(2)

    vid, plan = sys.argv[1], sys.argv[2]
    args = sys.argv[3:]
    step = int(args[args.index("--step") + 1]) if "--step" in args else 4
    win = int(args[args.index("--win") + 1]) if "--win" in args else 24
    tpl = int(args[args.index("--tpl") + 1]) if "--tpl" in args else 15
    dump_path = args[args.index("--dump") + 1] if "--dump" in args else None

    cap = cv2.VideoCapture(vid)
    if not cap.isOpened():
        print(json.dumps({"ok": False, "reason": "open_video"}))
        sys.exit(3)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

    pts = load_jsonl(plan)
    if not pts:
        print(json.dumps({"ok": False, "reason": "no_points"}))
        sys.exit(4)

    use_time = all(p[0] is not None for p in pts)
    table = pts if use_time else [(i / fps, x, y) for i, (t, x, y) in enumerate(pts)]

    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    scores, resids, times = [], [], []
    for i, (t, bx, by) in enumerate(table):
        if i % step:
            continue
        fno = int(round(t * fps)) if use_time else i
        cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cx = int(round(bx))
        cy = int(round(by))
        x0 = clamp(cx - win, 0, W - 1)
        x1 = clamp(cx + win, 0, W - 1)
        y0 = clamp(cy - win, 0, H - 1)
        y1 = clamp(cy + win, 0, H - 1)
        roi = gray[y0 : y1 + 1, x0 : x1 + 1]

        tx0 = clamp(cx - tpl, 0, W - 1)
        tx1 = clamp(cx + tpl, 0, W - 1)
        ty0 = clamp(cy - tpl, 0, H - 1)
        ty1 = clamp(cy + tpl, 0, H - 1)
        templ = gray[ty0 : ty1 + 1, tx0 : tx1 + 1]

        if roi.shape[0] < templ.shape[0] + 1 or roi.shape[1] < templ.shape[1] + 1:
            continue

        res = cv2.matchTemplate(roi, templ, cv2.TM_CCOEFF_NORMED)
        _, maxval, _, maxloc = cv2.minMaxLoc(res)

        dx = (x0 + maxloc[0] + templ.shape[1] // 2) - cx
        dy = (y0 + maxloc[1] + templ.shape[0] // 2) - cy
        scores.append(float(maxval))
        resids.append((dx, dy))
        times.append(float(t))

    if not scores:
        print(json.dumps({"ok": False, "reason": "no_samples"}))
        return

    mean_score = float(np.mean(scores))
    out = {
        "ok": True,
        "samples": len(scores),
        "score_mean": mean_score,
        "score_p10": float(np.percentile(scores, 10)),
        "score_p50": float(np.percentile(scores, 50)),
        "score_p90": float(np.percentile(scores, 90)),
        "dx_mean": float(np.mean([r[0] for r in resids])),
        "dy_mean": float(np.mean([r[1] for r in resids])),
        "dx_std": float(np.std([r[0] for r in resids])),
        "dy_std": float(np.std([r[1] for r in resids])),
        "t_min": float(min(times)),
        "t_max": float(max(times)),
    }

    if dump_path:
        with open(dump_path, "w", encoding="utf-8") as f:
            for t, (dx, dy), s in zip(times, resids, scores):
                f.write(json.dumps({"t": t, "dx": float(dx), "dy": float(dy), "score": float(s)}) + "\n")

    print(json.dumps(out))


if __name__ == "__main__":
    main()

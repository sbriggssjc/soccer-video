import json
import sys
import cv2
import numpy as np


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
        print("usage: score_plan.py <video> <plan.jsonl> [--step 4] [--win 24] [--tpl 15]")
        sys.exit(2)

    vid, plan = sys.argv[1], sys.argv[2]
    args = sys.argv[3:]
    step = int(args[args.index("--step") + 1]) if "--step" in args else 4
    win = int(args[args.index("--win") + 1]) if "--win" in args else 24
    tpl = int(args[args.index("--tpl") + 1]) if "--tpl" in args else 15

    cap = cv2.VideoCapture(vid)
    if not cap.isOpened():
        print("ERR open video")
        sys.exit(3)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

    pts = load_jsonl(plan)
    if not pts:
        print("ERR no points")
        sys.exit(4)

    # map time->(bx,by)
    use_time = all(p[0] is not None for p in pts)
    if use_time:
        table = pts
    else:
        # infer times by index
        table = [(i / fps, x, y) for i, (t, x, y) in enumerate(pts)]

    scores = []
    resids = []
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

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
        # template centered on predicted point
        tx0 = clamp(cx - tpl, 0, W - 1)
        tx1 = clamp(cx + tpl, 0, W - 1)
        ty0 = clamp(cy - tpl, 0, H - 1)
        ty1 = clamp(cy + tpl, 0, H - 1)
        templ = gray[ty0 : ty1 + 1, tx0 : tx1 + 1]
        if roi.shape[0] < templ.shape[0] + 1 or roi.shape[1] < templ.shape[1] + 1:
            continue
        res = cv2.matchTemplate(roi, templ, cv2.TM_CCOEFF_NORMED)
        _, maxval, _, maxloc = cv2.minMaxLoc(res)
        # residual from predicted center to best match center
        dx = (x0 + maxloc[0] + templ.shape[1] // 2) - cx
        dy = (y0 + maxloc[1] + templ.shape[0] // 2) - cy
        scores.append(float(maxval))
        resids.append((dx, dy))

    if not scores:
        print(json.dumps({"ok": False, "reason": "no samples"}))
        return

    mean_score = float(np.mean(scores))
    p10 = float(np.percentile(scores, 10))
    p50 = float(np.percentile(scores, 50))
    p90 = float(np.percentile(scores, 90))
    mx = float(np.mean([r[0] for r in resids]))
    my = float(np.mean([r[1] for r in resids]))
    sx = float(np.std([r[0] for r in resids]))
    sy = float(np.std([r[1] for r in resids]))
    print(
        json.dumps(
            {
                "ok": True,
                "samples": len(scores),
                "score_mean": mean_score,
                "score_p10": p10,
                "score_p50": p50,
                "score_p90": p90,
                "dx_mean": mx,
                "dy_mean": my,
                "dx_std": sx,
                "dy_std": sy,
            }
        )
    )


if __name__ == "__main__":
    main()

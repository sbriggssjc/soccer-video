import cv2
import json
import math
import argparse
import numpy as np


def load_plan(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            o = json.loads(ln)
            t = o.get("t", None)
            bx = o.get("bx_stab", o.get("bx"))
            by = o.get("by_stab", o.get("by"))
            if bx is None or by is None:
                rows.append((t, None, None))
            else:
                if isinstance(bx, float) and math.isnan(bx):
                    bx = None
                if isinstance(by, float) and math.isnan(by):
                    by = None
                rows.append((t, bx, by))
    return rows


def write_plan(path, rows):
    with open(path, 'w', encoding='utf-8') as f:
        for t, x, y in rows:
            if x is None or y is None:
                continue
            o = {
                "t": float(t),
                "bx": float(x),
                "by": float(y),
                "bx_stab": float(x),
                "by_stab": float(y),
            }
            f.write(json.dumps(o, separators=(',', ':')) + "\n")


def grab_gray(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def clip(val, lo, hi):
    return max(lo, min(hi, val))


def ncc_score(patchA, patchB):
    a = patchA.astype(np.float32)
    b = patchB.astype(np.float32)
    ma, sa = float(a.mean()), float(a.std() + 1e-6)
    mb, sb = float(b.mean()), float(b.std() + 1e-6)
    return float(((a - ma) * (b - mb)).mean() / (sa * sb))


def best_time_shift(cap, fps, plan_xy, w, h, sample_stride=8, max_shift=6, tpl=25, win=31):
    """Return integer frame shift maximizing average NCC between consecutive patches along the track."""

    track = []
    for i, (t, x, y) in enumerate(plan_xy):
        if x is None or y is None:
            continue
        ti = int(round((i if t is None else t * fps)))
        track.append((ti, float(x), float(y)))
    if len(track) < max(10, 2 * sample_stride):
        return 0

    gray_cache = {}

    def get_gray(idx):
        if idx in gray_cache:
            return gray_cache[idx]
        g = grab_gray(cap, idx)
        gray_cache[idx] = g
        return g

    best_s = 0
    best_avg = -1e9
    half_tpl = tpl // 2
    for s in range(-max_shift, max_shift + 1):
        scores = []
        for k in range(0, len(track) - 1, sample_stride):
            t0, x0, y0 = track[k]
            t1, x1, y1 = track[min(k + 1, len(track) - 1)]
            g0 = get_gray(t0 + s)
            g1 = get_gray(t1 + s)
            if g0 is None or g1 is None:
                continue
            x0i, y0i = int(round(x0)), int(round(y0))
            x1i, y1i = int(round(x1)), int(round(y1))
            x0a = clip(x0i - half_tpl, 0, w - 1)
            x0b = clip(x0i + half_tpl, 0, w - 1)
            y0a = clip(y0i - half_tpl, 0, h - 1)
            y0b = clip(y0i + half_tpl, 0, h - 1)
            x1a = clip(x1i - half_tpl, 0, w - 1)
            x1b = clip(x1i + half_tpl, 0, w - 1)
            y1a = clip(y1i - half_tpl, 0, h - 1)
            y1b = clip(y1i + half_tpl, 0, h - 1)
            pa = g0[y0a : y0b + 1, x0a : x0b + 1]
            pb = g1[y1a : y1b + 1, x1a : x1b + 1]
            if pa.size == 0 or pb.size == 0:
                continue
            scc = ncc_score(pa, pb)
            scores.append(scc)
        if scores:
            avg = float(np.mean(scores))
            if avg > best_avg:
                best_avg = avg
                best_s = s
    return best_s


def refine_track(cap, fps, plan_xy, w, h, shift=0, tpl=21, search_r=28, ema_alpha=0.20):
    """Refine each frame by searching in a window around prior refined point using template NCC."""

    out = []
    half_tpl = tpl // 2
    pts = []
    for i, (t, x, y) in enumerate(plan_xy):
        ti = int(round((i if t is None else t * fps))) + shift
        pts.append((ti, None if x is None else float(x), None if y is None else float(y)))

    seed_idx = next((i for i, (_, x, y) in enumerate(pts) if x is not None and y is not None), None)
    if seed_idx is None:
        return None
    g0 = grab_gray(cap, max(0, pts[seed_idx][0]))
    if g0 is None:
        return None
    sx, sy = int(round(pts[seed_idx][1])), int(round(pts[seed_idx][2]))
    sx = clip(sx, 0, w - 1)
    sy = clip(sy, 0, h - 1)
    txa = clip(sx - half_tpl, 0, w - 1)
    txb = clip(sx + half_tpl, 0, w - 1)
    tya = clip(sy - half_tpl, 0, h - 1)
    tyb = clip(sy + half_tpl, 0, h - 1)
    template = g0[tya : tyb + 1, txa : txb + 1]
    cx, cy = float(sx), float(sy)
    last = None

    for i, (fi, px, py) in enumerate(pts):
        g = grab_gray(cap, max(0, fi))
        if g is None:
            out.append((i / fps, cx, cy))
            continue

        if px is not None and py is not None:
            if last is None:
                predx, predy = px, py
            else:
                predx = ema_alpha * px + (1 - ema_alpha) * last[0]
                predy = ema_alpha * py + (1 - ema_alpha) * last[1]
        else:
            predx, predy = last if last is not None else (cx, cy)

        sx0 = int(round(predx))
        sy0 = int(round(predy))
        xa = clip(sx0 - search_r, 0, w - 1)
        xb = clip(sx0 + search_r, 0, w - 1)
        ya = clip(sy0 - search_r, 0, h - 1)
        yb = clip(sy0 + search_r, 0, h - 1)
        win = g[ya : yb + 1, xa : xb + 1]
        if (
            win.size == 0
            or template.size == 0
            or win.shape[0] < template.shape[0]
            or win.shape[1] < template.shape[1]
        ):
            out.append((i / fps, cx, cy))
            last = (cx, cy)
            continue

        res = cv2.matchTemplate(win, template, cv2.TM_CCOEFF_NORMED)
        _, maxval, _, maxloc = cv2.minMaxLoc(res)
        cx = xa + maxloc[0] + template.shape[1] // 2
        cy = ya + maxloc[1] + template.shape[0] // 2
        cx = float(clip(cx, 0, w - 1))
        cy = float(clip(cy, 0, h - 1))
        last = (cx, cy)

        txa = clip(int(round(cx)) - half_tpl, 0, w - 1)
        txb = clip(int(round(cx)) + half_tpl, 0, w - 1)
        tya = clip(int(round(cy)) - half_tpl, 0, h - 1)
        tyb = clip(int(round(cy)) + half_tpl, 0, h - 1)
        new_tpl = g[tya : tyb + 1, txa : txb + 1]
        if new_tpl.size and new_tpl.shape == template.shape:
            template = (
                0.2 * new_tpl.astype(np.float32) + 0.8 * template.astype(np.float32)
            ).astype(np.uint8)

        out.append((i / fps, cx, cy))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="input video (atomic)")
    ap.add_argument("--plan", required=True, help="ball plan jsonl (.ball.lock.jsonl or similar)")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--tpl", type=int, default=21, help="template size (odd)")
    ap.add_argument("--search-r", type=int, default=28, help="search radius (px)")
    ap.add_argument("--shift-max", type=int, default=6, help="max time shift (frames)")
    ap.add_argument("--out", required=True, help="refined jsonl output")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.inp)
    if not cap.isOpened():
        raise SystemExit("Cannot open video: " + args.inp)
    fps_vid = cap.get(cv2.CAP_PROP_FPS)
    fps = args.fps if args.fps > 0 else (fps_vid if fps_vid > 0 else 24)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    plan = load_plan(args.plan)
    s = best_time_shift(
        cap,
        fps,
        plan,
        w,
        h,
        max_shift=args.shift_max,
        tpl=max(15, args.tpl),
        win=max(23, args.tpl + 10),
    )

    refined = refine_track(
        cap,
        fps,
        plan,
        w,
        h,
        shift=s,
        tpl=args.tpl,
        search_r=args.search_r,
        ema_alpha=0.20,
    )
    if refined is None:
        raise SystemExit("Refinement failed.")

    out_rows = []
    for i, (t, x, y) in enumerate(refined):
        out_rows.append((i / fps, x, y))

    write_plan(args.out, out_rows)
    print(
        f"Refined with shift={s} frames, tpl={args.tpl}, search_r={args.search_r}. Wrote: {args.out}"
    )


if __name__ == "__main__":
    main()


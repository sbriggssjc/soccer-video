import argparse
import json
import math
from collections import namedtuple

import cv2
import numpy as np


def lk_refine(prev_gray, cur_gray, prev_xy):
    if prev_xy is None:
        return None, 0.0
    p0 = np.array([[prev_xy]], dtype=np.float32)
    p1, st, err = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        cur_gray,
        p0,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    )
    if st is None or st[0][0] == 0:
        return None, 0.0
    x, y = p1[0][0]
    conf = 1.0 / float(1.0 + (err[0][0] if err is not None else 20.0))
    return (float(x), float(y)), conf


# ---- basic helpers ----
def build_ball_mask(bgr, grass_h=(35, 95), min_v=170, max_s=120):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    non_grass = (H < grass_h[0]) | (H > grass_h[1])
    bright = V >= min_v
    low_sat = S <= max_s
    mask = ((non_grass & bright) | (bright & low_sat)).astype(np.uint8) * 255
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask


def field_mask_bgr(bgr):
    """Binary mask of playable grass (exclude benches/crowd/track).
    Tuned for daylight turf; adjust HSV ranges if needed."""
    H, W = bgr.shape[:2]
    pixels = bgr.reshape(-1, 3).astype(np.float32)
    if pixels.size == 0:
        return np.zeros((H, W), dtype=np.uint8)

    sample_size = min(20000, pixels.shape[0])
    if sample_size < 3:
        return np.zeros((H, W), dtype=np.uint8)

    if sample_size < pixels.shape[0]:
        rng = np.random.default_rng(12345)
        idx = rng.choice(pixels.shape[0], size=sample_size, replace=False)
        sample = pixels[idx]
    else:
        sample = pixels

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 25, 0.5)
    try:
        _, labels, centers = cv2.kmeans(sample, 3, None, criteria, 4, cv2.KMEANS_PP_CENTERS)
    except cv2.error:
        return np.zeros((H, W), dtype=np.uint8)

    centers = centers.astype(np.float32)
    counts = np.bincount(labels.flatten(), minlength=3).astype(np.float32)
    area_ratios = counts / max(counts.sum(), 1.0)

    def green_score(center, area_ratio):
        b, g, r = center
        hsv = cv2.cvtColor(center.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2HSV)[0, 0]
        hue = hsv[0] / 180.0
        sat = hsv[1] / 255.0
        hue_dist = min(abs(hue - 1 / 3), abs(hue - 1 / 3 + 1), abs(hue - 1 / 3 - 1))
        hue_score = max(0.0, 1.0 - hue_dist / 0.25)
        green_ratio = g / (r + b + 1.0)
        green_diff = (g - max(r, b)) / 255.0
        return (0.6 * green_ratio + 0.6 * green_diff + 0.4 * sat + 0.4 * hue_score + 0.2 * area_ratio)

    scores = [green_score(c, area_ratios[i]) for i, c in enumerate(centers)]
    green_idx = int(np.argmax(scores))

    dists = np.linalg.norm(pixels[:, None, :] - centers[None, :, :], axis=2)
    labels_full = np.argmin(dists, axis=1)
    mask = (labels_full.reshape(H, W) == green_idx).astype(np.uint8) * 255

    # clean & fill holes
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    # erode a bit to avoid lines/edge balls
    mask = cv2.erode(mask, np.ones((9, 9), np.uint8), iterations=1)
    return mask


def sideline_penalty(x, y, W, H, margin_xy=(32, 40), heavy=12.0):
    """Large penalty near image borders (where spare balls live)."""
    mx, my = margin_xy
    if x < mx or x > (W - mx) or y < my or y > (H - my):
        return heavy
    return 0.0


def circularity(cnt):
    a = cv2.contourArea(cnt)
    p = cv2.arcLength(cnt, True)
    if p <= 0 or a <= 0:
        return 0.0
    return float(4 * math.pi * a / (p * p))


def ncc_score(gray_win, tpl_gray):
    if gray_win.size == 0 or tpl_gray.size == 0:
        return -1.0
    if (
        gray_win.shape[0] < tpl_gray.shape[0]
        or gray_win.shape[1] < tpl_gray.shape[1]
    ):
        return -1.0
    g = cv2.equalizeHist(gray_win)
    t = cv2.equalizeHist(tpl_gray)
    r1 = cv2.matchTemplate(g, t, cv2.TM_CCOEFF_NORMED).max()
    h, w = t.shape[:2]
    tw, th = max(3, int(w * 0.75)), max(3, int(h * 0.75))
    t2 = cv2.resize(t, (tw, th), interpolation=cv2.INTER_AREA)
    r2 = cv2.matchTemplate(g, t2, cv2.TM_CCOEFF_NORMED).max()
    return float(max(r1, r2))


def extract_tpl(gray, x, y, side, W, H):
    x0 = int(max(0, x - side // 2))
    y0 = int(max(0, y - side // 2))
    x1 = int(min(W, x0 + side))
    y1 = int(min(H, y0 + side))
    return gray[y0:y1, x0:x1].copy()


# ---- candidate generator ----
Cand = namedtuple("Cand", "x y score ncc circ dist")


def gen_candidates(
    frame_bgr,
    pred_xy,
    tpl=None,
    search_r=280,
    min_r=6,
    max_r=22,
    min_circ=0.58,
    max_cands=12,
):
    H, W = frame_bgr.shape[:2]
    px, py = pred_xy
    x0 = int(max(0, px - search_r))
    y0 = int(max(0, py - search_r))
    x1 = int(min(W, px + search_r))
    y1 = int(min(H, py + search_r))
    roi = frame_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return []

    mask = build_ball_mask(roi)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []

    field = field_mask_bgr(roi)
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < (min_r * min_r * 0.6) or a > (max_r * max_r * 3.5):
            continue
        circ = circularity(c)
        if circ < min_circ:
            continue
        (cx, cy), rad = cv2.minEnclosingCircle(c)
        bx = x0 + cx
        by = y0 + cy
        # reject/penalize if NOT on field
        fx = int(np.clip(cx, 0, field.shape[1] - 1))
        fy = int(np.clip(cy, 0, field.shape[0] - 1))
        on_field = field[fy, fx] > 0
        if not on_field:
            continue

        dist = math.hypot(bx - px, by - py)
        # NCC window around candidate (as before)
        ncc = 0.0
        if tpl is not None:
            side = int(max(16, min(96, rad * 6)))
            sx0 = int(max(0, bx - side // 2))
            sy0 = int(max(0, by - side // 2))
            sx1 = int(min(W, sx0 + side))
            sy1 = int(min(H, sy0 + side))
            win = gray_roi[(sy0 - y0) : (sy1 - y0), (sx0 - x0) : (sx1 - x0)]
            ncc = ncc_score(win, tpl)

        base = (-0.02 * dist) + (3.0 * circ) + (1.8 * ncc)
        base -= sideline_penalty(bx, by, W, H, margin_xy=(36, 48), heavy=14.0)
        out.append(
            Cand(float(bx), float(by), float(base), float(ncc), float(circ), float(dist))
        )
    out.sort(key=lambda c: c.score, reverse=True)
    return out[:max_cands]


# ---- dynamic programming over time ----
def solve_path(cand_lists, start_xy, lam_vel=0.06, lam_acc=0.02, miss_penalty=1.2, max_jump=220.0):
    N = len(cand_lists)
    MISS = Cand(np.nan, np.nan, -999.0, 0.0, 0.0, 0.0)
    C = [c + [MISS] for c in cand_lists]
    K = [len(c) for c in C]
    INF = 1e15
    dp = [np.full(k, INF, np.float64) for k in K]
    prv = [np.full(k, -1, np.int32) for k in K]

    sx, sy = start_xy
    for j, c in enumerate(C[0]):
        dp[0][j] = miss_penalty if np.isnan(c.x) else (-c.score + lam_vel * math.hypot(c.x - sx, c.y - sy))

    for t in range(1, N):
        for j, cj in enumerate(C[t]):
            for i, ci in enumerate(C[t - 1]):
                xj, yj = (cj.x, cj.y) if not np.isnan(cj.x) else (ci.x, ci.y)
                xi, yi = (ci.x, ci.y) if not np.isnan(ci.x) else (xj, yj)
                if not (np.isnan(xi) or np.isnan(xj)):
                    jump = math.hypot(xj - xi, yj - yi)
                    if jump > max_jump:
                        continue
                unary = miss_penalty if np.isnan(cj.x) else (-cj.score)
                pair = 0.0 if (np.isnan(xi) or np.isnan(xj)) else lam_vel * math.hypot(xj - xi, yj - yi)
                cost = dp[t - 1][i] + unary + pair
                if cost < dp[t][j]:
                    dp[t][j] = cost
                    prv[t][j] = i

    j = int(np.argmin(dp[-1]))
    path = [None] * N
    for t in range(N - 1, -1, -1):
        c = C[t][j]
        path[t] = (c.x, c.y, ("miss" if np.isnan(c.x) else "cand"))
        j = int(prv[t][j]) if t > 0 and prv[t][j] >= 0 else j

    last = None
    for t in range(N):
        x, y, src = path[t]
        if np.isnan(x) or np.isnan(y):
            if last is None:
                last = start_xy
            path[t] = (last[0], last[1], "pred")
        else:
            last = (x, y)
    nxt = None
    for t in range(N - 1, -1, -1):
        x, y, src = path[t]
        if src == "pred" and nxt is not None:
            path[t] = (0.5 * x + 0.5 * nxt[0], 0.5 * y + 0.5 * nxt[1], "pred")
        else:
            nxt = (x, y)
    return path


def rts_smooth(path, k_alpha_pos=0.35, k_alpha_vel=0.25, iters=1):
    """Simple forward-backward EMA to reduce jitter."""

    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    vx = [0.0] * len(xs)
    vy = [0.0] * len(xs)

    for t in range(1, len(xs)):
        px, py = xs[t - 1] + vx[t - 1], ys[t - 1] + vy[t - 1]
        rx, ry = xs[t] - px, ys[t] - py
        vx[t] = vx[t - 1] + k_alpha_vel * rx
        vy[t] = vy[t - 1] + k_alpha_vel * ry
        xs[t] = (1 - k_alpha_pos) * px + k_alpha_pos * xs[t]
        ys[t] = (1 - k_alpha_pos) * py + k_alpha_pos * ys[t]

    for t in range(len(xs) - 2, -1, -1):
        px, py = xs[t + 1] - vx[t + 1], ys[t + 1] - vy[t + 1]
        rx, ry = xs[t] - px, ys[t] - py
        vx[t] = vx[t + 1] - k_alpha_vel * rx
        vy[t] = vy[t + 1] - k_alpha_vel * ry
        xs[t] = (1 - k_alpha_pos) * px + k_alpha_pos * xs[t]
        ys[t] = (1 - k_alpha_pos) * py + k_alpha_pos * ys[t]

    return list(zip(xs, ys))


def plan_zoom(
    xs,
    ys,
    fps,
    W,
    H,
    zmin=1.05,
    zmax=1.80,
    k_speed=0.0010,
    edge_m=0.14,
    edge_gain=0.14,
    z_alpha=0.22,
    z_rate=0.06,
):
    """Compute per-frame zoom from ball speed and edge proximity; 
       then forward-backward smooth with a rate limiter."""
    N = len(xs)
    # per-frame raw target
    zt = np.empty(N, dtype=np.float32)
    z = np.empty(N, dtype=np.float32)
    # seed
    z[0] = (zmin + zmax) / 2.0
    for t in range(N):
        bx, by = xs[t], ys[t]
        # speed in px/sec (use prev when possible)
        if t > 0:
            spf = math.hypot(xs[t] - xs[t - 1], ys[t] - ys[t - 1])
            speed = spf * fps
        else:
            speed = 0.0
        # base: faster → wider (smaller zoom factor → here we use wider = lower clamp of z? 
        # we define z as "zoom factor" where higher = tighter crop; so subtract k*speed)
        z_raw = zmax - k_speed * speed
        z_raw = max(zmin, min(zmax, z_raw))
        # edge pressure: nudge wider if near crop edges (approx with frame edges here; renderer refines)
        # Distance to image edges normalized:
        dl = bx / max(1, W)
        dr = (W - bx) / max(1, W)
        dt = by / max(1, H)
        db = (H - by) / max(1, H)
        prox = max(0.0, edge_m - min(dl, dr, dt, db)) / max(edge_m, 1e-6)
        z_raw = max(zmin, z_raw - edge_gain * prox)
        zt[t] = z_raw

    # forward-backward EMA + rate limit
    # forward
    for t in range(1, N):
        target = z[t - 1] + (zt[t] - z[t - 1]) * z_alpha
        # rate limit
        dz = max(-z_rate, min(z_rate, target - z[t - 1]))
        z[t] = z[t - 1] + dz
    # backward
    for t in range(N - 2, -1, -1):
        target = z[t + 1] + (z[t] - z[t + 1]) * z_alpha
        dz = max(-z_rate, min(z_rate, target - z[t + 1]))
        z[t] = z[t + 1] + dz
    # clamp final
    z = np.clip(z, zmin, zmax)
    return z.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--init-t", type=float, default=0.8)
    ap.add_argument("--init-manual", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-cands", type=int, default=12)
    ap.add_argument("--search-r", type=int, default=280)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.inp)
    if not cap.isOpened():
        raise SystemExit("Cannot open input")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    idx0 = max(0, int(round(args.init_t * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx0)
    ok, frame0 = cap.read()
    if not ok:
        raise SystemExit("Cannot read init frame")

    if args.init_manual and hasattr(cv2, "selectROI"):
        cv2.namedWindow("Select ball", cv2.WINDOW_NORMAL)
        r = cv2.selectROI(
            "Select ball", frame0, showCrosshair=True, fromCenter=False
        )
        cv2.destroyWindow("Select ball")
        if r is None or r[2] <= 0 or r[3] <= 0:
            raise SystemExit("No ROI selected")
        bx0 = r[0] + r[2] / 2.0
        by0 = r[1] + r[3] / 2.0
    else:
        raise SystemExit("Use --init-manual on this first run")

    tpl = extract_tpl(
        cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY), int(bx0), int(by0), 64, W, H
    )

    cand_lists = []
    positions = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    n = 0
    prev_gray = None
    pred = (bx0, by0)
    miss_streak = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None and positions:
            flow_xy, flow_conf = lk_refine(prev_gray, gray, positions[-1])
            if flow_xy is not None and flow_conf > 0.02:
                px = 0.7 * flow_xy[0] + 0.3 * positions[-1][0]
                py = 0.7 * flow_xy[1] + 0.3 * positions[-1][1]
                pred = (px, py)
            else:
                pred = positions[-1]

        sr = args.search_r + min(200, 40 * miss_streak)

        cands = gen_candidates(
            frame,
            pred,
            tpl,
            search_r=sr,
            max_cands=args.max_cands,
        )

        if miss_streak >= 12 and not cands:
            full_pred = (frame.shape[1] / 2.0, frame.shape[0] / 2.0)
            cands = gen_candidates(
                frame,
                full_pred,
                tpl,
                search_r=max(frame.shape[0], frame.shape[1]),
                max_cands=args.max_cands * 2,
            )

        if cands:
            positions.append((cands[0].x, cands[0].y))
            miss_streak = 0
            cur = extract_tpl(gray, int(cands[0].x), int(cands[0].y), 64, frame.shape[1], frame.shape[0])
            if cur.size and tpl.size and cur.shape == tpl.shape:
                tpl = cv2.addWeighted(tpl, 0.9, cur, 0.1, 0)
        else:
            positions.append(positions[-1] if positions else (bx0, by0))
            miss_streak += 1

        cand_lists.append(cands)
        prev_gray = gray
        n += 1
    cap.release()

    path = solve_path(
        cand_lists,
        (bx0, by0),
        lam_vel=0.08,
        lam_acc=0.02,
        miss_penalty=1.1,
        max_jump=280.0,
    )
    smoothed = rts_smooth(path, k_alpha_pos=0.42, k_alpha_vel=0.30, iters=1)

    xs = [sx for (sx, sy) in smoothed]
    ys = [sy for (sx, sy) in smoothed]
    z = plan_zoom(
        xs,
        ys,
        fps,
        W,
        H,
        zmin=1.05,
        zmax=1.80,
        k_speed=0.0010,
        edge_m=0.14,
        edge_gain=0.14,
        z_alpha=0.22,
        z_rate=0.055,
    )

    with open(args.out, "w", encoding="utf-8") as f:
        for i, ((x, y, src), (sx, sy)) in enumerate(zip(path, smoothed)):
            t = i / float(fps)
            f.write(
                json.dumps(
                    {
                        "t": t,
                        "bx": float(sx),
                        "by": float(sy),
                        "z": float(z[i]),
                        "src": src,
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    main()

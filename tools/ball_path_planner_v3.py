import argparse
import json
import math
from collections import namedtuple
from math import hypot
from typing import Dict, List, Tuple

import cv2
import numpy as np

Cand = namedtuple("Cand", "x y score ncc grad dist src")


def to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def eq(gray: np.ndarray) -> np.ndarray:
    return cv2.equalizeHist(gray)


def stabilize_step(prev_gray: np.ndarray, cur_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    g = eq(to_gray(cur_bgr))
    p0 = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=1200,
        qualityLevel=0.01,
        minDistance=8,
        blockSize=7,
    )
    if p0 is None:
        return g, np.eye(3)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, g, p0, None, winSize=(21, 21), maxLevel=3)
    if p1 is None or st is None:
        return g, np.eye(3)
    p0v = p0[st == 1].reshape(-1, 2)
    p1v = p1[st == 1].reshape(-1, 2)
    if len(p0v) < 12:
        return g, np.eye(3)
    M, _ = cv2.estimateAffinePartial2D(p1v, p0v, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    A = np.eye(3)
    if M is not None:
        A[:2, :] = M
    return g, A


def warp_affine(img: np.ndarray, A: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.warpAffine(
        img,
        A[:2, :],
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def field_mask_bgr(bgr: np.ndarray) -> np.ndarray:
    height, width = bgr.shape[:2]
    pixels = bgr.reshape(-1, 3).astype(np.float32)
    if pixels.size == 0:
        return np.zeros((height, width), dtype=np.uint8)

    sample_size = min(20000, pixels.shape[0])
    if sample_size < 3:
        return np.zeros((height, width), dtype=np.uint8)

    if sample_size < pixels.shape[0]:
        rng = np.random.default_rng(12345)
        sample_idx = rng.choice(pixels.shape[0], size=sample_size, replace=False)
        sample = pixels[sample_idx]
    else:
        sample = pixels

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.5)
    try:
        _, labels, centers = cv2.kmeans(sample, 3, None, criteria, 4, cv2.KMEANS_PP_CENTERS)
    except cv2.error:
        return np.zeros((height, width), dtype=np.uint8)

    centers = centers.astype(np.float32)
    label_counts = np.bincount(labels.flatten(), minlength=3).astype(np.float32)
    area_ratios = label_counts / max(label_counts.sum(), 1.0)

    def green_score(center: np.ndarray, area_ratio: float) -> float:
        b, g, r = center
        hsv = cv2.cvtColor(center.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2HSV)[0, 0]
        hue = hsv[0] / 180.0
        sat = hsv[1] / 255.0
        hue_dist = min(abs(hue - 1 / 3), abs(hue - 1 / 3 + 1), abs(hue - 1 / 3 - 1))
        hue_score = max(0.0, 1.0 - hue_dist / 0.25)
        green_ratio = g / (r + b + 1.0)
        green_diff = (g - max(r, b)) / 255.0
        return (
            0.6 * green_ratio
            + 0.6 * green_diff
            + 0.4 * sat
            + 0.4 * hue_score
            + 0.2 * area_ratio
        )

    scores = [green_score(center, area_ratios[i]) for i, center in enumerate(centers)]
    green_idx = int(np.argmax(scores))

    # Assign every pixel to the closest center and generate the mask for the green cluster.
    dists = np.linalg.norm(pixels[:, None, :] - centers[None, :, :], axis=2)
    full_labels = np.argmin(dists, axis=1)
    mask = (full_labels.reshape(height, width) == green_idx).astype(np.uint8) * 255

    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.erode(mask, np.ones((9, 9), np.uint8), 1)
    return mask


def sideline_penalty(x: float, y: float, width: int, height: int, margin_xy=(60, 70), heavy: float = 20.0) -> float:
    mx, my = margin_xy
    if x < mx or x > width - mx or y < my or y > height - my:
        return heavy
    return 0.0


def motion_strength(prev_stab: np.ndarray, cur_stab: np.ndarray) -> np.ndarray:
    diff = cv2.absdiff(prev_stab, cur_stab)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    return gray


def radial_grad(gray: np.ndarray, cx: int, cy: int, radius: int) -> float:
    r = int(max(6, min(24, radius)))
    x0 = int(max(0, cx - r))
    y0 = int(max(0, cy - r))
    x1 = int(min(gray.shape[1], cx + r))
    y1 = int(min(gray.shape[0], cy + r))
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.0
    gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return float(np.percentile(mag, 80))


def ncc_score(win: np.ndarray, tpl: np.ndarray) -> float:
    if win.size == 0 or tpl.size == 0:
        return -1.0
    if win.shape[0] < tpl.shape[0] or win.shape[1] < tpl.shape[1]:
        return -1.0
    gray = eq(win)
    tpl_eq = eq(tpl)
    r1 = cv2.matchTemplate(gray, tpl_eq, cv2.TM_CCOEFF_NORMED).max()
    h, w = tpl_eq.shape[:2]
    tpl_small = cv2.resize(
        tpl_eq,
        (max(3, int(w * 0.75)), max(3, int(h * 0.75))),
        cv2.INTER_AREA,
    )
    r2 = cv2.matchTemplate(gray, tpl_small, cv2.TM_CCOEFF_NORMED).max()
    return float(max(r1, r2))


def gen_candidates(
    stab_bgr: np.ndarray,
    raw_bgr: np.ndarray,
    field_mask: np.ndarray,
    motion_map: np.ndarray,
    pred_xy: Tuple[float, float],
    tpl: np.ndarray,
    search_r: int = 320,
    min_r: int = 6,
    max_r: int = 22,
    max_cands: int = 16,
) -> List[Cand]:
    height, width = stab_bgr.shape[:2]
    px, py = pred_xy
    x0 = int(max(0, px - search_r))
    y0 = int(max(0, py - search_r))
    x1 = int(min(width, px + search_r))
    y1 = int(min(height, py + search_r))
    if x1 <= x0 + 2 or y1 <= y0 + 2:
        return []

    roi = stab_bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    field_roi = field_mask[y0:y1, x0:x1]
    motion_roi = motion_map[y0:y1, x0:x1]

    out: List[Cand] = []

    def add_cand(cx: int, cy: int, rad: int, src: str) -> None:
        bx = x0 + cx
        by = y0 + cy
        if field_roi[int(np.clip(cy, 0, field_roi.shape[0] - 1)), int(np.clip(cx, 0, field_roi.shape[1] - 1))] == 0:
            return
        motion_val = float(
            motion_roi[int(np.clip(cy, 0, motion_roi.shape[0] - 1)), int(np.clip(cx, 0, motion_roi.shape[1] - 1))]
        )
        if motion_val < 3.0:
            return
        dist = hypot(bx - px, by - py)
        side = int(max(16, min(96, rad * 6)))
        sx0 = int(max(0, bx - side // 2))
        sy0 = int(max(0, by - side // 2))
        sx1 = int(min(width, sx0 + side))
        sy1 = int(min(height, sy0 + side))
        win = cv2.cvtColor(stab_bgr[sy0:sy1, sx0:sx1], cv2.COLOR_BGR2GRAY)
        ncc = ncc_score(win, tpl) if tpl is not None else 0.0
        grad = radial_grad(gray, int(cx), int(cy), int(rad))
        base = (-0.02 * dist) + (1.5 * ncc) + (0.02 * grad)
        base -= sideline_penalty(bx, by, width, height, margin_xy=(64, 76), heavy=22.0)
        out.append(Cand(float(bx), float(by), float(base), float(ncc), float(grad), float(dist), src))

    thr = cv2.adaptiveThreshold(eq(gray), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, -7)
    thr = cv2.medianBlur(thr, 5)
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < (min_r * min_r * 0.6) or area > (max_r * max_r * 4.0):
            continue
        (cx, cy), rad = cv2.minEnclosingCircle(contour)
        add_cand(int(cx), int(cy), int(rad), "contour")

    hough = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=14,
        param1=110,
        param2=18,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if hough is not None:
        for x, y, r in np.round(hough[0, :]).astype(int):
            add_cand(x, y, r, "hough")

    out.sort(key=lambda c: c.score, reverse=True)
    return out[:max_cands]


def solve_path(
    cand_lists: List[List[Cand]],
    start_xy: Tuple[float, float],
    anchors_by_frame: Dict[int, Tuple[float, float]],
    lam_vel: float = 0.07,
    miss_pen: float = 1.35,
    max_jump: float = 280.0,
    anchor_gain: float = 0.08,
    anchor_win: int = 6,
) -> List[Tuple[float, float]]:
    num_frames = len(cand_lists)
    miss_cand = Cand(np.nan, np.nan, -999.0, 0.0, 0.0, 0.0, "miss")
    candidates = [cand_list + [miss_cand] for cand_list in cand_lists]
    lengths = [len(c) for c in candidates]
    inf = 1e15
    dp = [np.full(k, inf, dtype=float) for k in lengths]
    prv = [np.full(k, -1, dtype=int) for k in lengths]
    start_x, start_y = start_xy

    def anchor_bonus(frame_idx: int, x_val: float, y_val: float) -> float:
        best = None
        best_d2 = None
        for dt in range(-anchor_win, anchor_win + 1):
            f = frame_idx + dt
            if f < 0 or f >= num_frames:
                continue
            if f in anchors_by_frame:
                ax, ay = anchors_by_frame[f]
                d2 = (x_val - ax) * (x_val - ax) + (y_val - ay) * (y_val - ay)
                if best is None or d2 < best_d2:
                    best = (ax, ay)
                    best_d2 = d2
        if best is None or best_d2 is None:
            return 0.0
        return -anchor_gain * min(4000.0, math.sqrt(best_d2))

    for j, cand in enumerate(candidates[0]):
        if np.isnan(cand.x):
            dp[0][j] = miss_pen
        else:
            dp[0][j] = -cand.score + lam_vel * hypot(cand.x - start_x, cand.y - start_y) + anchor_bonus(0, cand.x, cand.y)

    for t in range(1, num_frames):
        for j, cand_j in enumerate(candidates[t]):
            for i, cand_i in enumerate(candidates[t - 1]):
                xj, yj = (cand_j.x, cand_j.y) if not np.isnan(cand_j.x) else (cand_i.x, cand_i.y)
                xi, yi = (cand_i.x, cand_i.y) if not np.isnan(cand_i.x) else (xj, yj)
                if not (np.isnan(xi) or np.isnan(xj)):
                    if hypot(xj - xi, yj - yi) > max_jump:
                        continue
                unary = miss_pen if np.isnan(cand_j.x) else (-cand_j.score + anchor_bonus(t, xj, yj))
                pair = 0.0 if (np.isnan(xi) or np.isnan(xj)) else lam_vel * hypot(xj - xi, yj - yi)
                cost = dp[t - 1][i] + unary + pair
                if cost < dp[t][j]:
                    dp[t][j] = cost
                    prv[t][j] = i

    j = int(np.argmin(dp[-1]))
    path: List[Tuple[float, float, str]] = [None] * num_frames  # type: ignore
    for t in range(num_frames - 1, -1, -1):
        cand = candidates[t][j]
        path[t] = (cand.x, cand.y, "miss" if np.isnan(cand.x) else "cand")
        if t > 0 and prv[t][j] >= 0:
            j = int(prv[t][j])

    last = None
    for t in range(num_frames):
        x_val, y_val, src = path[t]
        if np.isnan(x_val) or np.isnan(y_val):
            if last is None:
                last = start_xy
            path[t] = (last[0], last[1], "pred")
        else:
            last = (x_val, y_val, src)  # type: ignore

    def ema(series: List[float], alpha: float = 0.28) -> List[float]:
        values = list(series)
        for idx in range(1, len(values)):
            values[idx] = (1 - alpha) * values[idx - 1] + alpha * values[idx]
        for idx in range(len(values) - 2, -1, -1):
            values[idx] = (1 - alpha) * values[idx + 1] + alpha * values[idx]
        return values

    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    xs = ema(xs, 0.26)
    ys = ema(ys, 0.26)
    return list(zip(xs, ys))


def plan_zoom(
    xs: List[float],
    ys: List[float],
    fps: float,
    width: int,
    height: int,
    zmin: float = 1.06,
    zmax: float = 1.80,
    k_speed: float = 0.0010,
    edge_m: float = 0.14,
    edge_gain: float = 0.16,
    rate: float = 0.038,
) -> List[float]:
    num_frames = len(xs)

    def edge_prox(bx: float, by: float) -> float:
        dl = bx / max(1, width)
        dr = (width - bx) / max(1, width)
        dt = by / max(1, height)
        db = (height - by) / max(1, height)
        return max(0.0, edge_m - min(dl, dr, dt, db)) / max(edge_m, 1e-6)

    zt = np.zeros(num_frames, np.float32)
    for t in range(num_frames):
        speed = 0.0 if t == 0 else hypot(xs[t] - xs[t - 1], ys[t] - ys[t - 1]) * fps
        zr = max(zmin, min(zmax, zmax - k_speed * speed))
        zr = max(zmin, zr - edge_gain * edge_prox(xs[t], ys[t]))
        zt[t] = zr

    try:
        from scipy.signal import savgol_filter

        win = max(9, (int(0.45 * fps) // 2) * 2 + 1)
        z = savgol_filter(zt, window_length=win, polyorder=3, mode="interp")
    except Exception:
        z = zt.copy()
        for alpha in (0.22, 0.22):
            for i in range(1, num_frames):
                z[i] = (1 - alpha) * z[i - 1] + alpha * z[i]
            for i in range(num_frames - 2, -1, -1):
                z[i] = (1 - alpha) * z[i + 1] + alpha * z[i]

    for i in range(1, num_frames):
        dz = np.clip(z[i] - z[i - 1], -rate, rate)
        z[i] = z[i - 1] + dz
    for i in range(num_frames - 2, -1, -1):
        dz = np.clip(z[i] - z[i + 1], -rate, rate)
        z[i] = z[i + 1] + dz

    for i in range(num_frames):
        crop_w = height / max(1e-6, z[i]) * (1080 / 1920)
        min_crop_w = width / 1.90
        if crop_w < min_crop_w:
            z[i] = height / (min_crop_w * (1080 / 1920))

    z = np.clip(z, zmin, zmax)
    return [float(val) for val in z]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--init-manual", action="store_true")
    ap.add_argument("--init-t", type=float, default=0.8)
    ap.add_argument("--anchors", help="JSONL from ball_anchors.py", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-cands", type=int, default=16)
    ap.add_argument("--search-r", type=int, default=320)
    ap.add_argument("--min-r", type=int, default=6)
    ap.add_argument("--max-r", type=int, default=22)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.inp)
    if not cap.isOpened():
        raise SystemExit("Cannot open input")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    anchors_by_frame: Dict[int, Tuple[float, float]] = {}
    with open(args.anchors, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            anchors_by_frame[int(round(data["t"] * fps))] = (float(data["bx"]), float(data["by"]))

    idx0 = max(0, int(round(args.init_t * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx0)
    ok, frame0 = cap.read()
    if not ok:
        raise SystemExit("Cannot read init frame")
    cv2.namedWindow("Select ball", cv2.WINDOW_NORMAL)
    roi = cv2.selectROI("Select ball", frame0, showCrosshair=True, fromCenter=False) if args.init_manual else None
    cv2.destroyWindow("Select ball")
    if roi is None or roi[2] <= 0 or roi[3] <= 0:
        raise SystemExit("Use --init-manual and select ball")
    bx0 = roi[0] + roi[2] / 2.0
    by0 = roi[1] + roi[3] / 2.0
    r_est = 0.25 * math.sqrt(roi[2] * roi[3])
    min_r = max(args.min_r, max(3, int(0.6 * r_est)))
    max_r = max(args.max_r, min(30, int(2.2 * r_est)))
    if max_r <= min_r:
        max_r = min(30, min_r + 2)
    template = eq(to_gray(frame0))[int(by0 - 32) : int(by0 + 32), int(bx0 - 32) : int(bx0 + 32)].copy()
    if template.size == 0:
        template = eq(to_gray(frame0))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, first_frame = cap.read()
    if not ok:
        raise SystemExit("Empty video")
    prev_gray = eq(to_gray(first_frame))
    prev_stab = first_frame.copy()
    field_mask = field_mask_bgr(first_frame)
    cand_lists: List[List[Cand]] = []
    pred = (bx0, by0)
    misses = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cur_gray, A = stabilize_step(prev_gray, frame)
        stab = warp_affine(frame, A, width, height)
        motion_map = motion_strength(prev_stab, stab)
        if frame_idx - 1 in anchors_by_frame:
            pred = anchors_by_frame[frame_idx - 1]
        elif frame_idx in anchors_by_frame:
            pred = anchors_by_frame[frame_idx]
        search_radius = args.search_r + min(240, 40 * misses)
        candidates = gen_candidates(
            stab,
            frame,
            field_mask,
            motion_map,
            pred,
            template,
            search_r=search_radius,
            min_r=min_r,
            max_r=max_r,
            max_cands=args.max_cands,
        )
        if not candidates and misses >= 10:
            candidates = gen_candidates(
                stab,
                frame,
                field_mask,
                motion_map,
                (width / 2.0, height / 2.0),
                template,
                search_r=max(width, height),
                min_r=min_r,
                max_r=max_r,
                max_cands=args.max_cands * 2,
            )
        if candidates:
            top = candidates[0]
            cur_tpl = eq(to_gray(stab))[int(top.y - 32) : int(top.y + 32), int(top.x - 32) : int(top.x + 32)]
            if cur_tpl.size and template.size and cur_tpl.shape == template.shape:
                template = cv2.addWeighted(template, 0.9, cur_tpl, 0.1, 0)
            misses = 0
        else:
            misses += 1
        cand_lists.append(candidates)
        prev_gray = cur_gray
        prev_stab = stab
        frame_idx += 1
    cap.release()

    path = solve_path(
        cand_lists,
        (bx0, by0),
        anchors_by_frame,
        lam_vel=0.07,
        miss_pen=1.35,
        max_jump=280.0,
        anchor_gain=0.10,
        anchor_win=8,
    )
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]

    z = plan_zoom(
        xs,
        ys,
        fps,
        width,
        height,
        zmin=1.08,
        zmax=1.78,
        k_speed=0.00105,
        edge_m=0.14,
        edge_gain=0.18,
        rate=0.035,
    )

    with open(args.out, "w", encoding="utf-8") as f:
        for i, (x, y) in enumerate(zip(xs, ys)):
            t = i / float(fps)
            f.write(json.dumps({"t": t, "bx": float(x), "by": float(y), "z": float(z[i])}) + "\n")


if __name__ == "__main__":
    main()

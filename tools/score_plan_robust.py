import json, argparse, math, os, sys
import numpy as np
import cv2


def load_plan(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            o = json.loads(ln)
            bx = o.get('bx_stab', o.get('bx', None))
            by = o.get('by_stab', o.get('by', None))
            if bx is None or by is None:
                rows.append(None)
                continue
            t = float(o.get('t')) if 't' in o else None
            rows.append({'t': t, 'bx': float(bx), 'by': float(by)})
    return rows


def clamp_int(v, lo, hi):
    return max(lo, min(int(v), hi))


def crop(img, cx, cy, hw, hh):
    h, w = img.shape[:2]
    x0 = clamp_int(cx - hw, 0, w - 1)
    y0 = clamp_int(cy - hh, 0, h - 1)
    x1 = clamp_int(cx + hw, 0, w - 1)
    y1 = clamp_int(cy + hh, 0, h - 1)
    # ensure odd sizes (centered)
    if (x1 - x0 + 1) % 2 == 0:
        if x1 < w - 1:
            x1 += 1
        elif x0 > 0:
            x0 -= 1
    if (y1 - y0 + 1) % 2 == 0:
        if y1 < h - 1:
            y1 += 1
        elif y0 > 0:
            y0 -= 1
    return img[y0 : y1 + 1, x0 : x1 + 1], x0, y0


def to_gray(img):
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def sobel_grad(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return mag


def zncc(a, b, eps=1e-6):
    am = a - a.mean()
    bm = b - b.mean()
    na = np.sqrt((am * am).sum()) + eps
    nb = np.sqrt((bm * bm).sum()) + eps
    return float((am * bm).sum() / (na * nb))


def ssd(a, b):
    d = a.astype(np.float32) - b.astype(np.float32)
    return float((d * d).mean())


def match_best(search_img, tpl_img, norm):
    H, W = search_img.shape[:2]
    h, w = tpl_img.shape[:2]
    if H < h or W < w:
        return None, None, None
    if norm == 'ncc':
        res = cv2.matchTemplate(search_img, tpl_img, cv2.TM_CCOEFF_NORMED)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res)
        return maxLoc[0], maxLoc[1], float(maxVal)
    elif norm == 'ssd':
        res = cv2.matchTemplate(search_img, tpl_img, cv2.TM_SQDIFF)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res)
        return minLoc[0], minLoc[1], float(-minVal)  # negate so "bigger is better"
    elif norm == 'gradncc':
        return match_best(sobel_grad(search_img), sobel_grad(tpl_img), 'ncc')
    else:
        raise ValueError("norm must be one of: ncc, ssd, gradncc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--plan', required=True)
    ap.add_argument('--step', type=int, default=1)
    ap.add_argument('--tpl', type=int, default=31, help='template half-size (pixels radius)')
    ap.add_argument('--search-r', type=int, default=96, help='search radius around plan (pixels)')
    ap.add_argument('--norm', default='gradncc', choices=['ncc', 'ssd', 'gradncc'])
    ap.add_argument('--dump', required=True, help='output residuals jsonl')
    args = ap.parse_args()

    plan = load_plan(args.plan)
    cap = cv2.VideoCapture(args.inp)
    if not cap.isOpened():
        print(json.dumps({"ok": False, "err": "cannot_open_video"}))
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    out = open(args.dump, 'w', encoding='utf-8')
    samples = 0
    good = 0
    for i in range(0, min(N, len(plan)), args.step):
        row = plan[i]
        if row is None or row['bx'] is None or row['by'] is None:
            continue
        # seek frame i
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            break
        gray = to_gray(frame)

        cx = int(round(row['bx']))
        cy = int(round(row['by']))
        tpl_img, tx0, ty0 = crop(gray, cx, cy, args.tpl, args.tpl)

        # reject degenerate templates
        if tpl_img.size < 9 or tpl_img.std() < 5.0:
            continue

        sr = args.search_r
        search_img, sx0, sy0 = crop(gray, cx, cy, args.tpl + sr, args.tpl + sr)

        # IMPORTANT: ensure search area actually larger than template
        if (
            search_img.shape[0] <= tpl_img.shape[0]
            or search_img.shape[1] <= tpl_img.shape[1]
        ):
            continue

        offx, offy, score = match_best(search_img, tpl_img, args.norm)
        if offx is None:
            continue

        # found template top-left in search; compute center offset
        cx_found = sx0 + offx + tpl_img.shape[1] // 2
        cy_found = sy0 + offy + tpl_img.shape[0] // 2
        dx = float(cx_found - cx)
        dy = float(cy_found - cy)

        t = row['t'] if row['t'] is not None else (i / fps)
        # keep as residual only if movement is non-trivial
        if abs(dx) + abs(dy) > 0.25:
            good += 1
            out.write(
                json.dumps({"i": i, "t": t, "dx": dx, "dy": dy, "score": score})
                + "\n"
            )
        samples += 1

    out.close()
    cap.release()
    # report
    print(json.dumps({"ok": True, "samples": samples, "valid": good}))


if __name__ == '__main__':
    main()

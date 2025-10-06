import sys, os, csv, cv2, numpy as np
try:
    from numpy import RankWarning
except Exception:
    class RankWarning(UserWarning):
        pass
import warnings
warnings.filterwarnings("ignore", category=RankWarning)

args = sys.argv[1:]
if len(args) < 2:
    raise SystemExit("usage: smart_track_one.py <in> <out_csv> [weights_or_NONE] [conf]")
in_path, out_csv = args[0], args[1]
weights = args[2] if len(args) >= 3 else "NONE"
try:
    conf_min = float(args[3]) if len(args) >= 4 else 0.35
except Exception:
    conf_min = 0.35

cap = cv2.VideoCapture(in_path)
if not cap.isOpened():
    raise SystemExit("Cannot open " + in_path)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = cap.get(cv2.CAP_PROP_FPS) or 24.0
dt = 1.0 / max(FPS, 1.0)

use_yolo = False
yolo = None
if weights and weights.upper() != "NONE" and os.path.exists(weights):
    try:
        from ultralytics import YOLO

        yolo = YOLO(weights)
        use_yolo = True
        print("YOLO loaded:", weights)
    except Exception as e:
        print("YOLO load failed:", e)


def clamp(v, a, b):
    return max(a, min(b, v))


# HSV bands (wider, more forgiving)
RED1 = ((0, 50, 50), (8, 255, 255))
RED2 = ((165, 50, 50), (179, 255, 255))
ORNG = ((8, 50, 50), (22, 255, 255))
WHITE = ((0, 0, 190), (179, 55, 255))  # white ball: low S, high V
GREEN = ((35, 30, 40), (95, 255, 255))  # field to veto


def hsv_mask(img, lohi):
    lo, hi = lohi
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array(lo), np.array(hi))


def red_or_white_mask(img):
    m = hsv_mask(img, RED1) | hsv_mask(img, RED2) | hsv_mask(img, ORNG) | hsv_mask(img, WHITE)
    g = hsv_mask(img, GREEN)
    m = cv2.bitwise_and(m, cv2.bitwise_not(g))
    m = cv2.medianBlur(m, 5)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    return m


def color_candidates(bgr, roi=None):
    if roi is None:
        x0 = y0 = 0
        x1 = bgr.shape[1]
        y1 = bgr.shape[0]
    else:
        x0, y0, x1, y1 = roi
    patch = bgr[y0:y1, x0:x1]
    m = red_or_white_mask(patch)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 20 or area > 6000:
            continue
        peri = cv2.arcLength(c, True)
        if peri < 1:
            continue
        circ = (4 * np.pi * area) / (peri * peri)  # 1~circle
        if circ < 0.55:
            continue  # tighter roundness to kill lines
        (x, y), r = cv2.minEnclosingCircle(c)
        cx = x0 + x
        cy = y0 + y
        score = float(area) * float(circ)
        out.append((cx, cy, score))
    return out


def hough_candidates(bgr, roi=None):
    if roi is None:
        x0 = y0 = 0
        x1 = bgr.shape[1]
        y1 = bgr.shape[0]
    else:
        x0, y0, x1, y1 = roi
    patch = bgr[y0:y1, x0:x1]
    g = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (7, 7), 1.6)
    cir = cv2.HoughCircles(
        g,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=18,
        param1=110,
        param2=18,
        minRadius=5,
        maxRadius=60,
    )
    out = []
    if cir is not None:
        for x, y, r in np.uint16(np.around(cir))[0, :]:
            out.append((x0 + float(x), y0 + float(y), float(r * r)))  # score ~ radius^2
    return out


def yolo_candidates(bgr, roi=None, pad=0):
    if not use_yolo:
        return []
    if roi is None:
        crop = bgr
        x0 = y0 = 0
    else:
        x0, y0, x1, y1 = roi
        crop = bgr[y0:y1, x0:x1]
    try:
        imgsz = max(512, ((max(W, H) + 31) // 32) * 32)
        rs = yolo.predict(source=crop, conf=conf_min, imgsz=imgsz, verbose=False)
        out = []
        if len(rs):
            r = rs[0]
            if getattr(r, "boxes", None) is not None:
                b = r.boxes
                for i in range(len(b)):
                    xyxy = b.xyxy[i].cpu().numpy()
                    conf = float(b.conf[i].item())
                    cx = 0.5 * (float(xyxy[0]) + float(xyxy[2]))
                    cy = 0.5 * (float(xyxy[1]) + float(xyxy[3]))
                    out.append((x0 + cx, y0 + cy, conf))
        return out
    except Exception:
        return []


def fuse(frame, last_xy, last_v):
    # propose locally then globally if needed
    props = []
    roi = None
    if last_xy is not None:
        cx, cy = last_xy
        x0 = int(clamp(cx - 180, 0, W - 1))
        x1 = int(clamp(cx + 180, 1, W))
        y0 = int(clamp(cy - 180, 0, H - 1))
        y1 = int(clamp(cy + 180, 1, H))
        roi = (x0, y0, x1, y1)
        props += yolo_candidates(frame, roi)
        props += color_candidates(frame, roi)
        props += hough_candidates(frame, roi)
    # if nothing strong, global
    if not props:
        props += yolo_candidates(frame, None)
        props += color_candidates(frame, None)
        props += hough_candidates(frame, None)

    best = None
    bestS = -1.0
    for px, py, raws in props:
        # decompose score
        yolo_c = raws if raws <= 1.0 else 0.0
        color = raws if raws > 1.0 else 0.0
        hough = raws if raws > 1e3 else 0.0
        mot = 0.0
        if (last_xy is not None) and (last_v is not None):
            ex = last_xy[0] + last_v[0] * dt
            ey = last_xy[1] + last_v[1] * dt
            d2 = (px - ex) ** 2 + (py - ey) ** 2
            mot = 1.0 / (1.0 + d2 / (180.0**2))
        # combine (cap each component)
        S = (
            2.2 * min(1.0, yolo_c)
            + 0.8 * np.log1p(color / 200.0)
            + 0.6 * min(1.0, hough / 2500.0)
            + 0.7 * mot
        )
        if S > bestS:
            bestS = S
            best = (
                px,
                py,
                float(max(min(1.0, yolo_c), 0.45 * (color > 0) + 0.35 * (hough > 0))),
            )
    return best if best is not None else (None, None, 0.0)


rows = []
n = 0
prev_gray = None
prev_pt = None
last_xy = None
last_v = None
lost = 999
lk_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03)
early = int(1.0 * FPS)
DEBUG = os.environ.get("DEBUG_TRACK", "0") == "1"
dbg_dir = os.path.join(os.path.dirname(out_csv), "_dbg_track")
if DEBUG and not os.path.exists(dbg_dir):
    os.makedirs(dbg_dir, exist_ok=True)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    det = None
    conf = 0.0
    force = (n < early) or (lost > 0)

    if force or last_xy is None:
        cx, cy, cf = fuse(frame, last_xy, last_v)
        if cx is not None:
            det = (cx, cy)
            conf = max(conf, cf)
            lost = 0

    if det is None and prev_gray is not None and prev_pt is not None:
        p1, st, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            gray,
            prev_pt,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=lk_crit,
        )
        if st is not None and int(st.ravel()[0]) == 1:
            p0r, st2, _ = cv2.calcOpticalFlowPyrLK(
                gray,
                prev_gray,
                p1,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=lk_crit,
            )
            st2_ok = st2 is not None and int(st2.ravel()[0]) == 1
            fb = float(np.linalg.norm(prev_pt - p0r)) if st2_ok else 9e9
            if fb < 2.0:
                x = float(p1[0, 0, 0])
                y = float(p1[0, 0, 1])
                if 0 <= x < W and 0 <= y < H:
                    det = (x, y)
                    conf = max(conf, 0.50)
                    lost = 0
            else:
                lost += 1

    # sanity: if hugging edge with low confidence, re-fuse globally
    edge = (
        (last_xy is not None)
        and (
            abs(last_xy[0] - 20) < 30
            or abs(last_xy[0] - (W - 20)) < 30
            or abs(last_xy[1] - 20) < 30
            or abs(last_xy[1] - (H - 20)) < 30
        )
    )
    if det is None or conf < 0.35 or edge:
        cx2, cy2, cf2 = fuse(frame, det if det is not None else last_xy, last_v)
        if cx2 is not None and (det is None or cf2 > conf):
            det = (cx2, cy2)
            conf = cf2
            lost = 0

    if det is None:
        if last_xy is not None:
            cx, cy = last_xy
        else:
            cx, cy = W / 2.0, H / 2.0
        conf = 0.0
        lost += 1
    else:
        cx, cy = det

    if last_xy is not None:
        last_v = ((cx - last_xy[0]) / dt, (cy - last_xy[1]) / dt)
    last_xy = (cx, cy)

    rows.append([n, f"{clamp(cx, 0, W - 1):.4f}", f"{clamp(cy, 0, H - 1):.4f}", f"{conf:.4f}", W, H, FPS])
    prev_gray = gray
    prev_pt = np.array([[[cx, cy]]], dtype=np.float32)
    n += 1

    if DEBUG and (n % 8 == 0):
        vis = frame.copy()
        cv2.circle(
            vis,
            (int(round(cx)), int(round(cy))),
            10,
            (0, 0, 255) if conf >= 0.35 else (0, 255, 255),
            2,
        )
        cv2.imwrite(os.path.join(dbg_dir, f"{n:06d}.png"), vis)

cap.release()
with open(out_csv, "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["n", "cx", "cy", "conf", "w", "h", "fps"])
    wr.writerows(rows)
print("wrote", out_csv)

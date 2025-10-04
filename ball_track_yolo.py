import sys, csv, os, cv2, math
from pathlib import Path

USE_YOLO = False
try:
    from ultralytics import YOLO
    USE_YOLO = True
except Exception:
    USE_YOLO = False

def clamp(v, lo, hi): return max(lo, min(hi, v))

in_path, out_csv, weights_arg, conf_min = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
cap = cv2.VideoCapture(in_path)
if not cap.isOpened(): raise SystemExit(f"Cannot open {in_path}")

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

rows = []
n = 0

def write_csv(rows):
    with open(out_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["n","cx","cy","conf","w","h","fps"])
        for r in rows:
            wr.writerow([r[0], f"{r[1]:.4f}", f"{r[2]:.4f}", f"{r[3]:.4f}", w, h, fps])

# ---------- Branch A: YOLO if weights file exists ----------
if USE_YOLO and weights_arg and os.path.exists(weights_arg):
    model = YOLO(weights_arg)
    last = None
    # imgsz: round to 32, at least 640
    imgsz = max(640, ((max(w, h) + 31) // 32) * 32)
    for result in model.predict(source=in_path, conf=conf_min, stream=True, imgsz=imgsz, verbose=False):
        det = None
        if result.boxes is not None and len(result.boxes) > 0:
            b = result.boxes
            i = int(b.conf.argmax().item())
            xyxy = b.xyxy[i].tolist()
            conf  = float(b.conf[i].item())
            cx = 0.5*(xyxy[0]+xyxy[2]); cy = 0.5*(xyxy[1]+xyxy[3])
            det = (cx, cy, conf)
        if det is None:
            if last is not None: rows.append((n, last[0], last[1], 0.0))
            else: rows.append((n, w/2.0, h/2.0, 0.0))
        else:
            cx, cy, conf = det
            last = (cx, cy)
            rows.append((n, clamp(cx,0,w-1), clamp(cy,0,h-1), conf))
        n += 1
    write_csv(rows)
    sys.exit(0)

# ---------- Branch B: Fallback (no weights): motion+blob with smoothing ----------
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
fg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
last = (w/2.0, h/2.0)
alpha_fast = 0.35
alpha_slow = 0.18
smoothed = None

while True:
    ok, frame = cap.read()
    if not ok: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    msk = fg.apply(gray)
    msk = cv2.morphologyEx(msk, cv2.MORPH_OPEN, kernel, iterations=1)
    msk = cv2.morphologyEx(msk, cv2.MORPH_CLOSE, kernel, iterations=1)
    cnts, _ = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if 8 <= area <= 3000:   # small-ish blobs
            (x,y), r = cv2.minEnclosingCircle(c)
            # prefer round-ish, small radius
            score = area / (r*r + 1e-6)
            if score > best_area:
                best_area = score
                best = (x, y)
    if best is None:
        cx, cy = last
        conf = 0.0
    else:
        cx, cy = best
        conf = 0.5  # heuristic
        last = (cx, cy)

    # 2-stage EMA smoothing
    if smoothed is None:
        s1x, s1y = cx, cy
        s2x, s2y = cx, cy
    else:
        s1x = alpha_fast*cx + (1-alpha_fast)*smoothed[0]
        s1y = alpha_fast*cy + (1-alpha_fast)*smoothed[1]
        s2x = alpha_slow*s1x + (1-alpha_slow)*smoothed[0]
        s2y = alpha_slow*s1y + (1-alpha_slow)*smoothed[1]
    smoothed = (s2x, s2y)
    rows.append((n, clamp(smoothed[0],0,w-1), clamp(smoothed[1],0,h-1), conf))
    n += 1

write_csv(rows)

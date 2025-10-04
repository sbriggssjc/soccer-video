import sys, csv, os, cv2
# Robust argv: in, out [, weights] [, conf]
args = sys.argv[1:]
if len(args) < 2:
    raise SystemExit("usage: ball_track_yolo.py <in> <out_csv> [weights_or_NONE] [conf]")
in_path = args[0]
out_csv = args[1]
weights_arg = args[2] if len(args) >= 3 else "NONE"
try:
    conf_min = float(args[3]) if len(args) >= 4 else 0.35
except Exception:
    conf_min = 0.35

USE_YOLO = False
try:
    if weights_arg and weights_arg.upper() != "NONE" and os.path.exists(weights_arg):
        from ultralytics import YOLO  # import only if we will use it
        USE_YOLO = True
except Exception:
    USE_YOLO = False

def clamp(v, lo, hi): return max(lo, min(hi, v))

cap = cv2.VideoCapture(in_path)
if not cap.isOpened(): raise SystemExit(f"Cannot open {in_path}")

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

rows = []
def write_csv():
    with open(out_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["n","cx","cy","conf","w","h","fps"])
        for r in rows:
            wr.writerow(r)

# ---------- A) YOLO path ----------
if USE_YOLO:
    model = YOLO(weights_arg)
    n = 0
    last = None
    imgsz = max(640, ((max(w, h) + 31) // 32) * 32)
    for result in model.predict(source=in_path, conf=conf_min, stream=True, imgsz=imgsz, verbose=False):
        cx = cy = None
        conf = 0.0
        if result.boxes is not None and len(result.boxes) > 0:
            b = result.boxes
            i = int(b.conf.argmax().item())
            xyxy = b.xyxy[i].tolist()
            conf  = float(b.conf[i].item())
            cx = 0.5*(xyxy[0]+xyxy[2]); cy = 0.5*(xyxy[1]+xyxy[3])
        if cx is None:
            if last is not None: cx, cy = last
            else: cx, cy = w/2.0, h/2.0
            conf = 0.0
        else:
            last = (cx, cy)
        rows.append([n, f"{clamp(cx,0,w-1):.4f}", f"{clamp(cy,0,h-1):.4f}", f"{conf:.4f}", w, h, fps])
        n += 1
    write_csv()
    sys.exit(0)

# ---------- B) Fallback (no weights): motion+blob with smoothing ----------
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
fg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
last = (w/2.0, h/2.0)
alpha_fast = 0.35
alpha_slow = 0.18
smoothed = None
n = 0
while True:
    ok, frame = cap.read()
    if not ok: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    msk = fg.apply(gray)
    msk = cv2.morphologyEx(msk, cv2.MORPH_OPEN, kernel, iterations=1)
    msk = cv2.morphologyEx(msk, cv2.MORPH_CLOSE, kernel, iterations=1)
    cnts, _ = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None; best_area = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if 8 <= area <= 3000:
            (x,y), r = cv2.minEnclosingCircle(c)
            score = area / (r*r + 1e-6)
            if score > best_area: best_area, best = score, (x, y)
    if best is None:
        cx, cy = last; conf = 0.0
    else:
        cx, cy = best; conf = 0.5; last = (cx, cy)

    if smoothed is None: s1x=s2x=cx; s1y=s2y=cy
    else:
        s1x = alpha_fast*cx + (1-alpha_fast)*smoothed[0]
        s1y = alpha_fast*cy + (1-alpha_fast)*smoothed[1]
        s2x = alpha_slow*s1x + (1-alpha_slow)*smoothed[0]
        s2y = alpha_slow*s1y + (1-alpha_slow)*smoothed[1]
    smoothed = (s2x, s2y)
    rows.append([n, f"{clamp(smoothed[0],0,w-1):.4f}", f"{clamp(smoothed[1],0,h-1):.4f}", f"{conf:.4f}", w, h, fps])
    n += 1
write_csv()

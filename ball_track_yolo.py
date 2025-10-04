import sys, csv, math, cv2
from pathlib import Path
from ultralytics import YOLO

# args: inVideo, outCsv, weights, conf_min
in_path, out_csv, weights, conf_min = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
cap = cv2.VideoCapture(in_path)
if not cap.isOpened(): raise SystemExit(f"Cannot open {in_path}")

# model
model = YOLO(weights)

# run
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

# helpers
def clamp(v, lo, hi): return max(lo, min(hi, v))

last = None  # (cx,cy)
rows = []
n = 0
# We process via model.predict for accurate indexing; stream=True uses generators
for result in model.predict(source=in_path, conf=conf_min, stream=True, imgsz=max(640, ((max(w,h)+31)//32)*32), verbose=False):
    # pick highest-conf 'ball' box; if your model uses class 0 for ball, keep it; otherwise just use max conf
    det = None
    if result.boxes is not None and len(result.boxes) > 0:
        # choose best confidence box
        b = result.boxes
        i = int(b.conf.argmax().item())
        xyxy = b.xyxy[i].tolist()
        conf  = float(b.conf[i].item())
        cx = 0.5*(xyxy[0]+xyxy[2]); cy = 0.5*(xyxy[1]+xyxy[3])
        det = (cx, cy, conf)
    if det is None:
        # simple hold if missing
        if last is not None:
            rows.append((n, last[0], last[1], 0.0))
        else:
            rows.append((n, w/2.0, h/2.0, 0.0))
    else:
        cx, cy, conf = det
        last = (cx, cy)
        rows.append((n, clamp(cx,0,w-1), clamp(cy,0,h-1), conf))
    n += 1

with open(out_csv, "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["n","cx","cy","conf","w","h","fps"])
    for r in rows:
        wr.writerow([r[0], f"{r[1]:.4f}", f"{r[2]:.4f}", f"{r[3]:.4f}", w, h, fps])

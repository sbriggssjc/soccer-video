import cv2, numpy as np, json, math, sys, os
from ultralytics import YOLO

inp = r'.\out\atomic_clips\004__GOAL__t266.50-t283.10.mp4'
out_csv = r'.\out\autoframe_work\ball_track.csv'

# 1) Try YOLO 'sports ball' first (works if visible enough)
model = YOLO('yolov8n.pt')  # small, fast; replace with a ball-specific model if you have one

cap = cv2.VideoCapture(inp)
fps = cap.get(cv2.CAP_PROP_FPS)
W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

rows = []
frame = 0

def hsv_orange_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Tunable orange range (two bands to cover lighting; tweak if needed)
    m1 = cv2.inRange(hsv, (5, 120, 90), (20, 255, 255))
    m2 = cv2.inRange(hsv, (0, 120, 90), (5, 255, 255))
    mask = cv2.bitwise_or(m1, m2)
    mask = cv2.medianBlur(mask, 5)
    return mask

def find_orange_centroid(bgr):
    mask = hsv_orange_mask(bgr)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    # prefer small-ish, roughly circular blobs
    best = None; best_score = 1e9
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 20 or area > 5000:  # tune
            continue
        (x,y), r = cv2.minEnclosingCircle(c)
        circ = cv2.contourArea(c) / (math.pi*r*r + 1e-6)
        score = abs(1-circ) + 0.0001*area
        if score < best_score:
            best_score = score; best = (x,y)
    return best

while True:
    ok, bgr = cap.read()
    if not ok: break
    t = frame / fps

    # YOLO detection
    cx, cy = None, None
    try:
        res = model.predict(bgr, verbose=False, conf=0.3, classes=[32])  # 32 = sports ball
        if res and len(res[0].boxes):
            # pick the smallest 'ball-like' box
            boxes = res[0].boxes.xyxy.cpu().numpy()
            areas = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
            i = int(np.argmin(areas))
            x1,y1,x2,y2 = boxes[i]
            cx, cy = float((x1+x2)/2), float((y1+y2)/2)
    except Exception:
        pass

    # Fallback to orange color if YOLO missed
    if cx is None:
        c = find_orange_centroid(bgr)
        if c is not None:
            cx, cy = c

    rows.append((frame, t, cx if cx is not None else '', cy if cy is not None else ''))
    frame += 1

cap.release()

# Save CSV
os.makedirs(os.path.dirname(out_csv), exist_ok=True)
with open(out_csv,'w', newline='') as f:
    f.write('frame,time,cx,cy\n')
    for r in rows:
        f.write(','.join(map(str,r))+'\n')
print(f'Saved {out_csv}')

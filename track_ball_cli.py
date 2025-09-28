import cv2, numpy as np, json, math, os, argparse
from ultralytics import YOLO

def hsv_orange_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (5, 120, 90), (20, 255, 255))
    m2 = cv2.inRange(hsv, (0, 120, 90), (5, 255, 255))
    mask = cv2.bitwise_or(m1, m2)
    mask = cv2.medianBlur(mask, 5)
    return mask

def find_orange_centroid(bgr):
    mask = hsv_orange_mask(bgr)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    best = None; best_score = 1e9
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 20 or area > 5000:
            continue
        (x,y), r = cv2.minEnclosingCircle(c)
        circ = cv2.contourArea(c) / (math.pi*r*r + 1e-6)
        score = abs(1-circ) + 0.0001*area
        if score < best_score:
            best_score = score; best = (x,y)
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True, help="input clip")
    ap.add_argument("--out_csv", required=True, help="output CSV path")
    args = ap.parse_args()

    model = YOLO("yolov8n.pt")  # sports ball class 32

    cap = cv2.VideoCapture(args.inp)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

    rows = []
    frame = 0
    while True:
        ok, bgr = cap.read()
        if not ok: break
        t = frame / fps

        cx = cy = None
        try:
            res = model.predict(bgr, verbose=False, conf=0.30, classes=[32])
            if res and len(res[0].boxes):
                boxes = res[0].boxes.xyxy.cpu().numpy()
                areas = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
                i = int(np.argmin(areas))
                x1,y1,x2,y2 = boxes[i]
                cx, cy = float((x1+x2)/2), float((y1+y2)/2)
        except Exception:
            pass

        if cx is None:
            c = find_orange_centroid(bgr)
            if c is not None:
                cx, cy = c

        rows.append((frame, t, cx if cx is not None else "", cy if cy is not None else ""))
        frame += 1

    cap.release()

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        f.write("frame,time,cx,cy\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")
    print("Saved", args.out_csv)

if __name__ == "__main__":
    main()

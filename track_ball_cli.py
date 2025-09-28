# -*- coding: utf-8 -*-
import argparse, os, math, csv
import numpy as np
import cv2
from ultralytics import YOLO

def hsv_orange_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (5, 120, 90), (20, 255, 255))
    m2 = cv2.inRange(hsv, (0, 120, 90), (5, 255, 255))
    mask = cv2.bitwise_or(m1, m2)
    mask = cv2.medianBlur(mask, 5)
    return mask

def find_orange_centroid(bgr):
    cnts, _ = cv2.findContours(hsv_orange_mask(bgr), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    best, best_score = None, 1e9
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 20 or area > 5000:
            continue
        (x,y), r = cv2.minEnclosingCircle(c)
        circ = cv2.contourArea(c) / (math.pi*r*r + 1e-6)
        score = abs(1-circ) + 0.0001*area
        if score < best_score:
            best_score, best = score, (float(x), float(y))
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp",      required=True, help="input clip (.mp4)")
    ap.add_argument("--out_csv",  required=True, help="output CSV path")
    ap.add_argument("--yolo_conf", type=float, default=0.15)
    ap.add_argument("--roi_pad",   type=int,   default=220)
    ap.add_argument("--roi_pad_max", type=int, default=560)
    ap.add_argument("--max_miss",  type=int,   default=45)
    args = ap.parse_args()

    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(args.inp)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    rows = []
    frame = 0
    miss = 0
    last_xy = None

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame","time","cx","cy","ppl"])  # NEW: ppl column

        while True:
            ok, bgr = cap.read()
            if not ok: break
            t = frame / fps

            # dynamic ROI to speed/steady detection
            if last_xy is not None:
                pad = min(args.roi_pad + miss*12, args.roi_pad_max)
                x0 = max(0, int(last_xy[0] - pad))
                y0 = max(0, int(last_xy[1] - pad))
                x1 = min(W, int(last_xy[0] + pad))
                y1 = min(H, int(last_xy[1] + pad))
                roi = bgr[y0:y1, x0:x1]
                roi_off = (x0, y0)
            else:
                roi = bgr
                roi_off = (0, 0)

            cx, cy = None, None
            ppl_centers = []

            try:
                # one pass; we’ll read both classes from results
                res = model.predict(roi, verbose=False, conf=args.yolo_conf, classes=[0, 32])  # 0=person, 32=sports ball
                if res and len(res[0].boxes):
                    boxes = res[0].boxes
                    xyxy = boxes.xyxy.cpu().numpy()
                    cls  = boxes.cls.cpu().numpy().astype(int)

                    # BALL: pick smallest "ball-like" box
                    ball_boxes = xyxy[cls==32]
                    if ball_boxes.size:
                        areas = (ball_boxes[:,2]-ball_boxes[:,0])*(ball_boxes[:,3]-ball_boxes[:,1])
                        i = int(np.argmin(areas))
                        x1,y1,x2,y2 = ball_boxes[i]
                        cx = float((x1+x2)/2) + roi_off[0]
                        cy = float((y1+y2)/2) + roi_off[1]

                    # PEOPLE: keep up to 10 centers
                    person_boxes = xyxy[cls==0]
                    for bx in person_boxes[:]:
                        x1,y1,x2,y2 = bx
                        ppl_centers.append((float((x1+x2)/2)+roi_off[0], float((y1+y2)/2)+roi_off[1]))
            except Exception:
                pass

            # color fallback for ball
            if cx is None:
                cen = find_orange_centroid(roi)
                if cen is not None:
                    cx, cy = cen[0] + roi_off[0], cen[1] + roi_off[1]

            if cx is not None:
                last_xy = (cx, cy if cy is not None else (last_xy[1] if last_xy else H/2))
                miss = 0
            else:
                miss += 1
                if miss > args.max_miss:
                    last_xy = None
                    miss = 0

            ppl_str = "|".join(f"{px:.1f}:{py:.1f}" for px,py in ppl_centers[:10])
            wr.writerow([frame, f"{t:.3f}",
                         f"{cx:.1f}" if cx is not None else "",
                         f"{cy:.1f}" if cy is not None else "",
                         ppl_str])
            frame += 1

    cap.release()
    print(f"Saved {args.out_csv}")

if __name__ == "__main__":
    main()

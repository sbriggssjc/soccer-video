import argparse
import math
import os

import cv2
import numpy as np
from ultralytics import YOLO


class KalmanFilter:
    """Constant-velocity Kalman filter for (x, y, vx, vy)."""

    def __init__(self, process_var_pos=6.0, process_var_vel=3.0, measurement_var=36.0):
        self.state = None  # [x, y, vx, vy]
        self.P = None
        self.process_var_pos = process_var_pos
        self.process_var_vel = process_var_vel
        self.measurement_var = measurement_var

    def is_initialized(self):
        return self.state is not None

    def initialize(self, x, y):
        self.state = np.array([x, y, 0.0, 0.0], dtype=np.float32)
        self.P = np.diag([400.0, 400.0, 900.0, 900.0]).astype(np.float32)

    def predict(self, dt):
        if not self.is_initialized():
            return None

        F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        q_pos = self.process_var_pos * (dt ** 2)
        q_vel = self.process_var_vel * dt
        Q = np.diag([q_pos, q_pos, q_vel, q_vel]).astype(np.float32)

        self.state = F @ self.state
        self.P = F @ self.P @ F.T + Q
        return self.state.copy()

    def update(self, x, y):
        if not self.is_initialized():
            self.initialize(x, y)
            return self.state.copy()

        H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        R = np.diag([self.measurement_var, self.measurement_var]).astype(np.float32)
        z = np.array([x, y], dtype=np.float32)

        residual = z - (H @ self.state)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ residual
        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ H) @ self.P
        return self.state.copy()


def hsv_orange_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (5, 120, 90), (20, 255, 255))
    m2 = cv2.inRange(hsv, (0, 120, 90), (5, 255, 255))
    mask = cv2.bitwise_or(m1, m2)
    mask = cv2.medianBlur(mask, 5)
    return mask


def clamp_roi(x1, y1, x2, y2, width, height):
    x1 = max(0, min(width - 1, int(round(x1))))
    y1 = max(0, min(height - 1, int(round(y1))))
    x2 = max(x1 + 1, min(width, int(round(x2))))
    y2 = max(y1 + 1, min(height, int(round(y2))))
    return x1, y1, x2, y2


def add_candidate(candidates, x, y, p, source, radius=None):
    candidates.append({
        "x": float(x),
        "y": float(y),
        "p": float(p),
        "source": source,
        "radius": float(radius) if radius is not None else None,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp", required=True, help="input clip")
    parser.add_argument("--out_csv", required=True, help="output CSV path")
    args = parser.parse_args()

    model = YOLO("yolov8n.pt")  # sports ball class 32

    cap = cv2.VideoCapture(args.inp)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080

    rows = []
    frame_idx = 0

    kalman = KalmanFilter()
    roi_half_min = 80.0
    roi_half_max = max(frame_w, frame_h) / 2.0
    roi_half = roi_half_max
    miss_count = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        t = frame_idx / fps
        dt = 1.0 / fps
        prediction = kalman.predict(dt)
        pred_x, pred_y = (prediction[:2] if prediction is not None else (None, None))

        if pred_x is not None and pred_y is not None:
            roi_half = max(roi_half_min, min(roi_half, roi_half_max))
            x1 = pred_x - roi_half
            y1 = pred_y - roi_half
            x2 = pred_x + roi_half
            y2 = pred_y + roi_half
        else:
            roi_half = roi_half_max
            x1, y1, x2, y2 = 0, 0, frame_w, frame_h

        x1, y1, x2, y2 = clamp_roi(x1, y1, x2, y2, frame_w, frame_h)
        roi_frame = frame_bgr[y1:y2, x1:x2]
        roi_size = max(x2 - x1, y2 - y1)

        candidates = []

        # 1) YOLO detections
        try:
            res = model.predict(roi_frame, verbose=False, conf=0.25, classes=[32])
            if res and len(res[0].boxes):
                boxes = res[0].boxes.xyxy.cpu().numpy()
                confs = res[0].boxes.conf.cpu().numpy()
                for (bx1, by1, bx2, by2), conf in zip(boxes, confs):
                    cx = x1 + (bx1 + bx2) / 2.0
                    cy = y1 + (by1 + by2) / 2.0
                    radius = max(bx2 - bx1, by2 - by1) / 2.0
                    add_candidate(candidates, cx, cy, conf, "yolo", radius)
        except Exception:
            pass

        # 2) HSV contour centroids
        mask = hsv_orange_mask(roi_frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 20 or area > 8000:
                continue
            (cx_local, cy_local), radius = cv2.minEnclosingCircle(contour)
            if radius < 3 or radius > 60:
                continue
            perimeter = cv2.arcLength(contour, True)
            circularity = 0.0 if perimeter == 0 else 4.0 * math.pi * area / (perimeter * perimeter)
            if circularity < 0.55:
                continue
            confidence = 0.45 + 0.4 * min(1.0, circularity)
            global_x = x1 + cx_local
            global_y = y1 + cy_local
            add_candidate(candidates, global_x, global_y, confidence, "hsv", radius)

        # 3) Hough circles
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 1.5)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=15,
            param1=120,
            param2=18,
            minRadius=3,
            maxRadius=60,
        )
        if circles is not None:
            for cx_local, cy_local, radius in circles[0, :]:
                global_x = x1 + cx_local
                global_y = y1 + cy_local
                add_candidate(candidates, global_x, global_y, 0.55, "hough", radius)

        chosen = None
        best_score = -1e9

        if candidates:
            if pred_x is None or pred_y is None:
                chosen = max(candidates, key=lambda c: c["p"])
            else:
                gate_distance = max(90.0, 0.25 * roi_size)
                lam = 1.0 / max(gate_distance, 1.0)
                for cand in candidates:
                    dist = math.hypot(cand["x"] - pred_x, cand["y"] - pred_y)
                    if dist > gate_distance:
                        continue
                    score = cand["p"] - lam * dist
                    if score > best_score:
                        best_score = score
                        chosen = cand

        if chosen is not None:
            kalman.update(chosen["x"], chosen["y"])
            miss_count = 0
            roi_half = max(roi_half_min, roi_half - 8.0)
        else:
            miss_count += 1
            roi_half = min(roi_half + 12.0, roi_half_max)

        state = kalman.state
        if state is not None:
            rows.append((frame_idx, t, float(state[0]), float(state[1])))
        else:
            rows.append((frame_idx, t, "", ""))

        frame_idx += 1

    cap.release()

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        f.write("frame,time,cx,cy\n")
        for row in rows:
            f.write(",".join(map(str, row)) + "\n")

    print("Saved", args.out_csv)


if __name__ == "__main__":
    main()

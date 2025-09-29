# track_ball_cli.py
import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


def hsv_orange_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (5, 110, 80), (22, 255, 255))
    m2 = cv2.inRange(hsv, (0, 110, 80), (5, 255, 255))
    mask = cv2.bitwise_or(m1, m2)
    mask = cv2.medianBlur(mask, 5)
    return mask


def find_orange_centroid(bgr: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None) -> Optional[Tuple[float, float, float]]:
    if roi is not None:
        H, W = bgr.shape[:2]
        x0 = max(0, min(int(roi[0]), W - 1))
        y0 = max(0, min(int(roi[1]), H - 1))
        x1 = max(0, min(int(roi[2]), W))
        y1 = max(0, min(int(roi[3]), H))

        if x1 <= x0 or y1 <= y0:
            return None

        bgr_roi = bgr[y0:y1, x0:x1]
        if bgr_roi.size == 0:
            return None
        mask = hsv_orange_mask(bgr_roi)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_s = -1.0
        best_xy = None
        for c in cnts:
            area = float(cv2.contourArea(c))
            if area < 20 or area > 6000:
                continue
            (cx, cy), r = cv2.minEnclosingCircle(c)
            circ = cv2.contourArea(c) / (math.pi * r * r + 1e-6)
            score = max(0.0, 1.0 - abs(1.0 - circ)) * area
            if score > best_s:
                best_s = score
                best_xy = (x0 + float(cx), y0 + float(cy), area)
        return best_xy
    else:
        mask = hsv_orange_mask(bgr)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_s = -1.0
        best_xy = None
        for c in cnts:
            area = float(cv2.contourArea(c))
            if area < 20 or area > 6000:
                continue
            (cx, cy), r = cv2.minEnclosingCircle(c)
            circ = cv2.contourArea(c) / (math.pi * r * r + 1e-6)
            score = max(0.0, 1.0 - abs(1.0 - circ)) * area
            if score > best_s:
                best_s = score
                best_xy = (float(cx), float(cy), area)
        return best_xy


def clip_roi(x0: int, y0: int, x1: int, y1: int, W: int, H: int) -> Tuple[int, int, int, int]:
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(W, x1)
    y1 = min(H, y1)
    if x1 <= x0:
        x1 = min(W, x0 + 1)
    if y1 <= y0:
        y1 = min(H, y0 + 1)
    return (int(x0), int(y0), int(x1), int(y1))


@dataclass
class KFState:
    x: np.ndarray
    P: np.ndarray


class ConstantAccelerationKF:
    def __init__(self, dt: float, accel_var: float = 3000.0) -> None:
        self.dt = float(dt)
        dt2 = self.dt * self.dt
        dt3 = dt2 * self.dt
        self.A = np.array(
            [
                [1.0, 0.0, self.dt, 0.0, 0.5 * dt2, 0.0],
                [0.0, 1.0, 0.0, self.dt, 0.0, 0.5 * dt2],
                [0.0, 0.0, 1.0, 0.0, self.dt, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, self.dt],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        q = float(accel_var)
        # Process noise for jerk on acceleration (discrete white noise on jerk)
        G = np.array(
            [
                [1 / 6 * dt3, 0.0],
                [0.0, 1 / 6 * dt3],
                [0.5 * dt2, 0.0],
                [0.0, 0.5 * dt2],
                [self.dt, 0.0],
                [0.0, self.dt],
            ],
            dtype=np.float64,
        )
        Qc = np.eye(2, dtype=np.float64) * q
        self.Q = G @ Qc @ G.T
        self.R = np.eye(2, dtype=np.float64) * 120.0
        self.x = np.zeros((6, 1), dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64) * 1e4

    def predict(self) -> KFState:
        x_pred = self.A @ self.x
        P_pred = self.A @ self.P @ self.A.T + self.Q
        self.x = x_pred
        self.P = P_pred
        return KFState(x=x_pred.copy(), P=P_pred.copy())

    def update(self, z: np.ndarray) -> KFState:
        z = np.asarray(z, dtype=np.float64).reshape(2, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(6, dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P
        return KFState(x=self.x.copy(), P=self.P.copy())


def ncc_similarity(patch_a: np.ndarray, patch_b: np.ndarray) -> float:
    if patch_a.size == 0 or patch_b.size == 0:
        return -1.0
    a = patch_a.astype(np.float32).flatten()
    b = patch_b.astype(np.float32).flatten()
    if a.size != b.size:
        return -1.0
    a_mean = float(np.mean(a))
    b_mean = float(np.mean(b))
    num = float(np.sum((a - a_mean) * (b - b_mean)))
    den = float(np.sqrt(np.sum((a - a_mean) ** 2)) * np.sqrt(np.sum((b - b_mean) ** 2)))
    if den <= 1e-6:
        return -1.0
    return num / den


def extract_patch(img: np.ndarray, cx: float, cy: float, size: int = 21) -> Optional[np.ndarray]:
    if img is None:
        return None
    r = size // 2
    x0 = int(round(cx)) - r
    y0 = int(round(cy)) - r
    x1 = x0 + size
    y1 = y0 + size
    H, W = img.shape[:2]
    if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
        return None
    patch = img[y0:y1, x0:x1].copy()
    if patch.shape[0] != size or patch.shape[1] != size:
        return None
    return patch


def lightweight_motion_gate(gray: np.ndarray, prev_gray: Optional[np.ndarray], center: Tuple[float, float], radius: float) -> Optional[Tuple[float, float, float]]:
    if prev_gray is None:
        return None
    px, py = center
    r = int(max(8, radius))
    H, W = gray.shape[:2]
    x0 = max(0, int(px - r))
    y0 = max(0, int(py - r))
    x1 = min(W, int(px + r))
    y1 = min(H, int(py + r))
    if x1 <= x0 or y1 <= y0:
        return None
    cur = gray[y0:y1, x0:x1]
    prev = prev_gray[y0:y1, x0:x1]
    if cur.size == 0 or prev.size == 0:
        return None
    diff = cv2.absdiff(cur, prev)
    _, th = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
    th = cv2.medianBlur(th, 5)
    m = cv2.moments(th)
    if m["m00"] < 10:
        return None
    cx = float(m["m10"] / m["m00"]) + x0
    cy = float(m["m01"] / m["m00"]) + y0
    score = min(1.0, m["m00"] / (radius * radius + 1e-6))
    return (cx, cy, float(score))


def ensure_float(v: float) -> float:
    if isinstance(v, (np.floating,)):
        return float(v)
    return float(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True, help="input clip (.mp4)")
    ap.add_argument("--out_csv", required=True, help="output CSV path")
    ap.add_argument("--yolo_conf", type=float, default=0.12)
    ap.add_argument("--roi_pad", type=int, default=220)
    ap.add_argument("--roi_pad_max", type=int, default=560)
    ap.add_argument("--max_miss", type=int, default=45)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.inp)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    model = YOLO("yolov8n.pt")

    dt = 1.0 / max(fps, 1e-6)
    kf = ConstantAccelerationKF(dt)
    kf.x[0, 0] = W * 0.5
    kf.x[1, 0] = H * 0.5

    records: List[Dict[str, float]] = []
    predicted_states: List[np.ndarray] = []
    predicted_covs: List[np.ndarray] = []
    filtered_states: List[np.ndarray] = []
    filtered_covs: List[np.ndarray] = []
    visible_flags: List[int] = []
    conf_scores: List[float] = []

    prev_gray: Optional[np.ndarray] = None
    last_good_xy: Optional[Tuple[float, float]] = None
    last_template: Optional[np.ndarray] = None
    miss = 0
    pad = float(args.roi_pad)
    last_conf = 0.0

    frame = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        time_s = frame * dt

        state_pred = kf.predict()
        px = float(state_pred.x[0, 0])
        py = float(state_pred.x[1, 0])
        predicted_states.append(state_pred.x.copy())
        predicted_covs.append(state_pred.P.copy())

        candidates: List[Tuple[str, Tuple[float, float], float]] = []

        roi = None
        if last_good_xy is not None:
            cx_last, cy_last = last_good_xy
            x0 = int(cx_last - pad)
            y0 = int(cy_last - pad)
            x1 = int(cx_last + pad)
            y1 = int(cy_last + pad)
            roi = clip_roi(x0, y0, x1, y1, W, H)

        gate_radius = max(40.0, min(float(pad) * 0.8, 260.0))
        if miss > 15:
            gate_radius = max(gate_radius, float(pad))

        # YOLO detection
        try:
            y_inp = bgr if roi is None else bgr[roi[1] : roi[3], roi[0] : roi[2]]
            res = model.predict(y_inp, verbose=False, conf=args.yolo_conf, classes=[32])
            if res and len(res[0].boxes):
                best_score = -1.0
                best_det = None
                for b in res[0].boxes:
                    xyxy = b.xyxy[0].cpu().tolist()
                    x1d, y1d, x2d, y2d = [float(v) for v in xyxy]
                    if roi is not None:
                        x1d += roi[0]
                        x2d += roi[0]
                        y1d += roi[1]
                        y2d += roi[1]
                    cx_det = (x1d + x2d) * 0.5
                    cy_det = (y1d + y2d) * 0.5
                    area = max(1.0, (x2d - x1d) * (y2d - y1d))
                    conf = float(b.conf.item())
                    score = conf / math.sqrt(area)
                    if score > best_score:
                        best_score = score
                        best_det = (cx_det, cy_det, conf)
                if best_det is not None:
                    candidates.append(("yolo", (best_det[0], best_det[1]), float(best_det[2])))
        except Exception:
            pass

        # HSV detection
        hsv_roi = roi if roi is not None else None
        hsv_candidate = find_orange_centroid(bgr, hsv_roi)
        if hsv_candidate is None and roi is not None and (miss >= 5 or frame % 10 == 0):
            hsv_candidate = find_orange_centroid(bgr, None)
        if hsv_candidate is not None:
            candidates.append(("hsv", (float(hsv_candidate[0]), float(hsv_candidate[1])), min(1.0, 0.35 + 0.0002 * float(hsv_candidate[2]))))

        # Motion gate candidate
        motion_candidate = lightweight_motion_gate(gray, prev_gray, (px, py), gate_radius)
        if motion_candidate is not None:
            candidates.append(("motion", (motion_candidate[0], motion_candidate[1]), float(0.4 + 0.4 * motion_candidate[2])))

        meas = None
        meas_src = ""
        best_score = -1e9
        for label, (mx, my), base_score in candidates:
            dist = math.hypot(mx - px, my - py) if np.isfinite(px) and np.isfinite(py) else 0.0
            allowed = dist <= gate_radius or miss >= 8 or last_good_xy is None
            if not allowed:
                continue
            score = float(base_score) - 0.005 * dist
            if score > best_score:
                best_score = score
                meas = np.array([[mx], [my]], dtype=np.float64)
                meas_src = label

        # Optical flow fallback
        flow_candidate = None
        if meas is None and miss >= 5 and prev_gray is not None and last_good_xy is not None:
            p0 = np.array([[last_good_xy]], dtype=np.float32)
            p1, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, winSize=(21, 21), maxLevel=3,
                                                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
            if status is not None and int(status[0, 0]) == 1:
                fx = float(p1[0, 0, 0])
                fy = float(p1[0, 0, 1])
                disp = math.hypot(fx - last_good_xy[0], fy - last_good_xy[1])
                if disp <= 40.0:
                    if last_template is not None:
                        patch = extract_patch(gray, fx, fy, 21)
                        if patch is not None:
                            ncc = ncc_similarity(last_template, patch)
                            if ncc >= 0.65:
                                flow_candidate = (fx, fy, max(0.3, min(0.75, ncc)))
                    else:
                        flow_candidate = (fx, fy, 0.4)
        if meas is None and flow_candidate is not None:
            meas = np.array([[flow_candidate[0]], [flow_candidate[1]]], dtype=np.float64)
            meas_src = "flow"
            best_score = float(flow_candidate[2])

        # Reacquire global color scan every 10th frame when missing
        if meas is None and (frame % 10 == 0):
            global_candidate = find_orange_centroid(bgr, None)
            if global_candidate is not None:
                mx, my, area = global_candidate
                dist = math.hypot(mx - px, my - py)
                if dist <= gate_radius * 1.8 or last_good_xy is None:
                    meas = np.array([[mx], [my]], dtype=np.float64)
                    meas_src = "hsv_global"
                    best_score = 0.25 + 0.0001 * area

        if meas is not None:
            state_filt = kf.update(meas)
            cx = float(state_filt.x[0, 0])
            cy = float(state_filt.x[1, 0])
            vx = float(state_filt.x[2, 0])
            vy = float(state_filt.x[3, 0])
            last_good_xy = (cx, cy)
            last_template = extract_patch(gray, cx, cy, 21)
            miss = 0
            pad = float(args.roi_pad)
            visible = 1
            conf = float(max(0.05, min(1.0, best_score)))
            last_conf = conf
        else:
            state_filt = KFState(x=state_pred.x.copy(), P=state_pred.P.copy())
            cx = float(state_filt.x[0, 0])
            cy = float(state_filt.x[1, 0])
            vx = float(state_filt.x[2, 0])
            vy = float(state_filt.x[3, 0])
            miss += 1
            visible = 0
            last_conf *= 0.8
            conf = float(max(0.01, min(1.0, last_conf)))
            if miss == 1:
                pad = min(float(args.roi_pad_max), float(args.roi_pad) * 1.5)
            elif miss == 2:
                pad = min(float(args.roi_pad_max), float(args.roi_pad) * 2.0)
            else:
                pad = min(float(args.roi_pad_max), pad * 1.1)
            if miss > args.max_miss:
                pad = float(args.roi_pad_max)

        filtered_states.append(state_filt.x.copy())
        filtered_covs.append(state_filt.P.copy())
        visible_flags.append(int(visible))
        conf_scores.append(conf)

        records.append(
            {
                "frame": frame,
                "time": time_s,
                "cx": cx,
                "cy": cy,
                "visible": visible,
                "conf": conf,
                "vx": vx,
                "vy": vy,
            }
        )

        prev_gray = gray.copy()
        frame += 1

    cap.release()

    if not records:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True) if os.path.dirname(args.out_csv) else None
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame", "time", "cx", "cy", "visible", "conf", "vx", "vy"])
        print(f"Saved {args.out_csv}")
        return

    # RTS smoother
    A = kf.A
    N = len(filtered_states)
    xs = np.stack(filtered_states, axis=0)
    Ps = np.stack(filtered_covs, axis=0)
    xp = np.stack(predicted_states, axis=0)
    Pp = np.stack(predicted_covs, axis=0)

    xs_smooth = xs.copy()
    Ps_smooth = Ps.copy()

    for k in range(N - 2, -1, -1):
        Pk = Ps[k]
        Pk1_pred = Pp[k + 1]
        if np.linalg.cond(Pk1_pred) > 1e12:
            continue
        Ck = Pk @ A.T @ np.linalg.inv(Pk1_pred)
        xs_smooth[k] = xs[k] + Ck @ (xs_smooth[k + 1] - xp[k + 1])
        Ps_smooth[k] = Pk + Ck @ (Ps_smooth[k + 1] - Pk1_pred) @ Ck.T

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True) if os.path.dirname(args.out_csv) else None
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "time", "cx", "cy", "visible", "conf", "vx", "vy"])
        for i, rec in enumerate(records):
            xs_i = xs_smooth[i]
            cx = ensure_float(xs_i[0, 0])
            cy = ensure_float(xs_i[1, 0])
            vx = ensure_float(xs_i[2, 0])
            vy = ensure_float(xs_i[3, 0])
            writer.writerow(
                [
                    int(rec["frame"]),
                    f"{ensure_float(rec['time']):.3f}",
                    f"{cx:.3f}",
                    f"{cy:.3f}",
                    int(visible_flags[i]),
                    f"{float(conf_scores[i]):.3f}",
                    f"{vx:.3f}",
                    f"{vy:.3f}",
                ]
            )

    print(f"Saved {args.out_csv}")


if __name__ == "__main__":
    main()


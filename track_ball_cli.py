# track_ball_cli.py
import argparse
import csv
import io
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


# === ADD: robust helpers ===
def to_float(o):
    # Convert numpy scalars/arrays to plain Python floats for JSON/logging
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray) and o.size == 1:
        return float(o.item())
    return o


def safe_json(obj):
    def default(x):
        return to_float(x)

    return json.dumps(obj, default=default)


def nonempty_roi(bgr, x0, y0, x1, y1):
    h, w = bgr.shape[:2]
    x0 = max(0, min(w - 1, int(x0)))
    y0 = max(0, min(h - 1, int(y0)))
    x1 = max(0, min(w, int(x1)))
    y1 = max(0, min(h, int(y1)))
    if x1 <= x0 or y1 <= y0:
        return None
    return bgr[y0:y1, x0:x1]


# === ADD: fused candidate scoring + Mahalanobis gating ===
def score_candidate(px, py, yolo_conf, color_prob, pred, S_inv):
    # Larger is better. Weight prediction proximity and detectors.
    dx, dy = px - pred[0], py - pred[1]
    maha = dx * dx * S_inv[0, 0] + 2 * dx * dy * S_inv[0, 1] + dy * dy * S_inv[1, 1]
    prox = 1.0 / (1.0 + maha)
    return 2.0 * prox + 1.5 * yolo_conf + 1.0 * color_prob


def hsv_orange_mask(bgr: np.ndarray) -> np.ndarray:
    if bgr is None or bgr.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (5, 110, 80), (22, 255, 255))
    m2 = cv2.inRange(hsv, (0, 110, 80), (5, 255, 255))
    mask = cv2.bitwise_or(m1, m2)
    mask = cv2.medianBlur(mask, 5)
    return mask


def find_orange_centroid(
    bgr: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None
) -> Optional[Tuple[float, float, float]]:
    if roi is not None:
        H, W = bgr.shape[:2]
        x0 = max(0, min(int(roi[0]), W - 1))
        y0 = max(0, min(int(roi[1]), H - 1))
        x1 = max(0, min(int(roi[2]), W))
        y1 = max(0, min(int(roi[3]), H))

        if x1 <= x0 or y1 <= y0:
            return None

        bgr_roi = nonempty_roi(bgr, x0, y0, x1, y1)
        if bgr_roi is None or bgr_roi.size == 0:
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
    def __init__(self, dt: float, accel_var: float = 3800.0) -> None:
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
        self.R = np.eye(2, dtype=np.float64) * 150.0
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


def color_likelihood_at(
    bgr: np.ndarray, x: float, y: float, radius: float = 14.0
) -> float:
    if bgr is None or bgr.size == 0 or not np.isfinite(x) or not np.isfinite(y):
        return 0.0
    H, W = bgr.shape[:2]
    r = max(4, int(radius))
    x0 = max(0, int(round(x) - r))
    y0 = max(0, int(round(y) - r))
    x1 = min(W, int(round(x) + r))
    y1 = min(H, int(round(y) + r))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    patch = bgr[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    mask = hsv_orange_mask(patch)
    if mask.size == 0:
        return 0.0
    return float(np.mean(mask) / 255.0)


def savitzky_golay(values: List[float]) -> List[float]:
    if not values:
        return []
    if len(values) < 9:
        # Fallback to simple exponential smoothing for short sequences
        smoothed: List[float] = []
        alpha = 0.4
        prev = values[0]
        for v in values:
            prev = alpha * v + (1.0 - alpha) * prev
            smoothed.append(prev)
        return smoothed
    coeffs = np.array([-21, 14, 39, 54, 59, 54, 39, 14, -21], dtype=np.float64) / 231.0
    padded = [values[0]] * 4 + list(values) + [values[-1]] * 4
    smoothed = []
    for i in range(4, len(padded) - 4):
        window = padded[i - 4 : i + 5]
        smoothed.append(float(np.dot(coeffs, window)))
    return smoothed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True, help="input clip (.mp4)")
    ap.add_argument("--out_csv", required=True, help="output CSV path")
    ap.add_argument("--yolo_conf", type=float, default=0.05)
    ap.add_argument("--roi_pad", type=int, default=220)
    ap.add_argument("--roi_pad_max", type=int, default=1080)
    ap.add_argument("--max_miss", type=int, default=120)
    ap.add_argument("--imgsz", type=int, default=1280, help="YOLO input size")
    ap.add_argument("--model", default="yolov8n.pt", help="YOLO model path/name")
    ap.add_argument(
        "--classes",
        default="32",
        help="Comma-separated class ids for detection (use 'all' for no filter)",
    )
    ap.add_argument(
        "--agnostic_nms",
        type=int,
        default=0,
        choices=[0, 1],
        help="Use class-agnostic NMS",
    )
    ap.add_argument(
        "--min_track_len",
        type=int,
        default=6,
        help="Minimum consecutive visible frames before trusting detections",
    )
    ap.add_argument(
        "--velocity_jump_gate",
        type=float,
        default=140.0,
        help="Reject detections jumping more than this many pixels from prediction",
    )
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.inp)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    model = YOLO(args.model)

    classes = None
    if args.classes and str(args.classes).lower() != "all":
        classes = []
        for c in str(args.classes).split(","):
            c = c.strip()
            if not c:
                continue
            try:
                classes.append(int(c))
            except ValueError:
                raise ValueError(f"Invalid class id '{c}' for --classes") from None
        if not classes:
            classes = None

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
    miss_history: List[int] = []

    prev_gray: Optional[np.ndarray] = None
    last_good_xy: Optional[Tuple[float, float]] = None
    last_template: Optional[np.ndarray] = None
    miss = 0
    track_streak = 0
    roi_pad = float(args.roi_pad)
    last_conf = 0.0
    yolo_conf_dyn = float(args.yolo_conf)
    maha_gate_base = 9.21  # ~99% for 2 DoF

    frame = 0
    trace_file: Optional[io.TextIOBase] = None
    trace_writer: Optional[csv.writer] = None
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

        candidates: List[Dict[str, float]] = []

        roi = None
        if last_good_xy is not None:
            cx_last, cy_last = last_good_xy
            x0 = int(cx_last - roi_pad)
            y0 = int(cy_last - roi_pad)
            x1 = int(cx_last + roi_pad)
            y1 = int(cy_last + roi_pad)
            roi = clip_roi(x0, y0, x1, y1, W, H)

        gate_radius = max(40.0, min(float(roi_pad) * 0.8, 260.0))
        if miss > 15:
            gate_radius = max(gate_radius, float(roi_pad))

        S = state_pred.P[:2, :2].astype(np.float64)
        S += np.eye(2, dtype=np.float64) * 1e-6
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        maha_gate = maha_gate_base * (1.0 + min(miss, 6) * 0.35)

        # YOLO detection
        try:
            y_inp = bgr if roi is None else nonempty_roi(bgr, roi[0], roi[1], roi[2], roi[3])
            if y_inp is not None and y_inp.size != 0:
                predict_kwargs = dict(
                    verbose=False,
                    conf=yolo_conf_dyn,
                    imgsz=int(args.imgsz),
                    agnostic_nms=bool(args.agnostic_nms),
                )
                if classes is not None:
                    predict_kwargs["classes"] = classes
                res = model.predict(y_inp, **predict_kwargs)
            else:
                res = None
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
                        best_det = (cx_det, cy_det, conf, area)
                if best_det is not None:
                    color_lk = color_likelihood_at(bgr, best_det[0], best_det[1], radius=math.sqrt(best_det[3]) * 0.5)
                    candidates.append(
                        {
                            "src": "yolo",
                            "x": float(best_det[0]),
                            "y": float(best_det[1]),
                            "yolo_conf": float(best_det[2]),
                            "color": float(color_lk),
                        }
                    )
        except Exception:
            pass

        # HSV detection
        hsv_roi = roi if roi is not None else None
        hsv_candidate = find_orange_centroid(bgr, hsv_roi)
        if hsv_candidate is None and roi is not None and (miss >= 5 or frame % 10 == 0):
            hsv_candidate = find_orange_centroid(bgr, None)
        if hsv_candidate is not None:
            hx, hy, area = hsv_candidate
            color_lk = color_likelihood_at(bgr, hx, hy, radius=math.sqrt(max(area, 1.0)))
            hsv_score = min(1.0, 0.35 + 0.0002 * float(area))
            candidates.append(
                {
                    "src": "hsv",
                    "x": float(hx),
                    "y": float(hy),
                    "yolo_conf": 0.2,
                    "color": max(color_lk, hsv_score),
                }
            )

        # Motion gate candidate
        motion_candidate = lightweight_motion_gate(gray, prev_gray, (px, py), gate_radius)
        if motion_candidate is not None:
            mx, my, quality = motion_candidate
            color_lk = color_likelihood_at(bgr, mx, my, radius=gate_radius * 0.3)
            candidates.append(
                {
                    "src": "motion",
                    "x": float(mx),
                    "y": float(my),
                    "yolo_conf": float(0.25 + 0.35 * quality),
                    "color": float(color_lk * 0.8),
                }
            )

        # Optical flow fallback
        flow_candidate = None
        if miss >= 5 and prev_gray is not None and last_good_xy is not None:
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
        if flow_candidate is not None:
            mx, my, quality = flow_candidate
            color_lk = color_likelihood_at(bgr, mx, my, radius=gate_radius * 0.3)
            candidates.append(
                {
                    "src": "flow",
                    "x": float(mx),
                    "y": float(my),
                    "yolo_conf": float(max(0.2, min(0.8, quality))),
                    "color": float(color_lk),
                }
            )

        # Reacquire global color scan every 10th frame when missing
        if frame % 10 == 0:
            global_candidate = find_orange_centroid(bgr, None)
            if global_candidate is not None:
                mx, my, area = global_candidate
                dist = math.hypot(mx - px, my - py)
                if dist <= gate_radius * 1.8 or last_good_xy is None:
                    candidates.append(
                        {
                            "src": "hsv_global",
                            "x": float(mx),
                            "y": float(my),
                            "yolo_conf": 0.15,
                            "color": float(min(1.0, 0.25 + 0.0001 * area)),
                        }
                    )

        meas = None
        fused_conf = 0.0

        pred_xy = (float(px), float(py))
        best = None
        best_score = -1e9
        jump_gate = float(args.velocity_jump_gate) * (1.0 + min(miss, 6) * 0.25)
        for cand in candidates:
            mx = cand.get("x", np.nan)
            my = cand.get("y", np.nan)
            if not np.isfinite(mx) or not np.isfinite(my):
                continue
            diff = np.array([[mx - pred_xy[0]], [my - pred_xy[1]]], dtype=np.float64)
            maha_sq = float(diff.T @ S_inv @ diff)
            if not np.isfinite(maha_sq):
                continue
            if maha_sq > maha_gate and miss < 6:
                continue
            y_conf = float(max(0.0, min(1.0, cand.get("yolo_conf", 0.0))))
            color_prob = float(max(0.0, min(1.0, cand.get("color", 0.0))))
            score = score_candidate(mx, my, y_conf, color_prob, pred_xy, S_inv)
            if score > best_score:
                best_score = score
                best = (mx, my, str(cand.get("src", "")))

        if trace_writer is None:
            os.makedirs("out\\diag_trace", exist_ok=True)
            trace_file = open("out\\diag_trace\\cands.csv", "w", encoding="utf-8", newline="")
            trace_writer = csv.writer(trace_file)
            trace_writer.writerow(["frame", "num_cands", "top_score", "pred_x", "pred_y", "miss_streak"])
        if trace_writer is not None:
            top_score = "" if best is None else best_score
            trace_writer.writerow([
                frame,
                len(candidates),
                top_score,
                pred_xy[0],
                pred_xy[1],
                miss,
            ])

        if best is not None:
            jump = math.hypot(best[0] - pred_xy[0], best[1] - pred_xy[1])
            if jump > jump_gate:
                best = None

        was_missing = miss > 0

        if best is not None:
            meas = np.array([[best[0]], [best[1]]], dtype=np.float64)
            fused_conf = float(max(0.0, min(1.0, best_score / 4.5)))

        if meas is not None:
            state_filt = kf.update(meas)
            cx = float(state_filt.x[0, 0])
            cy = float(state_filt.x[1, 0])
            vx = float(state_filt.x[2, 0])
            vy = float(state_filt.x[3, 0])
            last_good_xy = (cx, cy)
            last_template = extract_patch(gray, cx, cy, 21)
            miss = 0
            track_streak += 1
            if was_missing or roi_pad > float(args.roi_pad):
                roi_pad = float(max(args.roi_pad, roi_pad * 0.9))
            else:
                roi_pad = float(max(args.roi_pad, roi_pad))
            roi_pad = float(min(roi_pad, float(args.roi_pad_max)))
            visible = 1 if track_streak >= args.min_track_len else 0
            conf = float(max(0.05, min(1.0, fused_conf)))
            last_conf = conf
            yolo_conf_dyn = float(args.yolo_conf)
        else:
            state_filt = KFState(x=state_pred.x.copy(), P=state_pred.P.copy())
            cx = float(state_filt.x[0, 0])
            cy = float(state_filt.x[1, 0])
            vx = float(state_filt.x[2, 0])
            vy = float(state_filt.x[3, 0])
            miss += 1
            track_streak = 0
            visible = 0
            last_conf *= 0.8
            conf = float(max(0.01, min(1.0, last_conf)))
            roi_pad = float(min(float(args.roi_pad_max), roi_pad * 1.2))
            yolo_conf_dyn = max(0.03, args.yolo_conf - 0.02)
            if miss > args.max_miss:
                roi_pad = float(args.roi_pad_max)

        filtered_states.append(state_filt.x.copy())
        filtered_covs.append(state_filt.P.copy())
        visible_flags.append(int(visible))
        conf_scores.append(conf)
        miss_history.append(int(miss))

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
                "miss_streak": miss,
            }
        )

        prev_gray = gray.copy()
        frame += 1

    cap.release()

    if not records:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True) if os.path.dirname(args.out_csv) else None
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame", "time", "cx", "cy", "visible", "conf", "miss_streak"])
        print(f"Saved {args.out_csv}")
        if trace_file is not None:
            trace_file.close()
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

    cx_vals = [ensure_float(xs_smooth[i][0, 0]) for i in range(N)]
    cy_vals = [ensure_float(xs_smooth[i][1, 0]) for i in range(N)]
    cx_smooth = savitzky_golay(cx_vals)
    cy_smooth = savitzky_golay(cy_vals)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True) if os.path.dirname(args.out_csv) else None
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "time", "cx", "cy", "visible", "conf", "miss_streak"])
        for i, rec in enumerate(records):
            cx = cx_smooth[i] if i < len(cx_smooth) else cx_vals[i]
            cy = cy_smooth[i] if i < len(cy_smooth) else cy_vals[i]
            writer.writerow(
                [
                    int(rec["frame"]),
                    f"{ensure_float(rec['time']):.3f}",
                    f"{cx:.3f}",
                    f"{cy:.3f}",
                    int(visible_flags[i]),
                    f"{float(conf_scores[i]):.3f}",
                    int(miss_history[i]),
                ]
            )

    print(f"Saved {args.out_csv}")

    if trace_file is not None:
        trace_file.close()


if __name__ == "__main__":
    main()


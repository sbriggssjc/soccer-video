import sys, json, math, os
import cv2 as cv
import numpy as np

# Usage: python tools/track_orange_ball.py path/to/video.mp4
# Outputs JSON with keyframes: [{"t":seconds,"x":pixels,"y":pixels,"conf":0..1}, ...]

path = sys.argv[1]
cap = cv.VideoCapture(path)
if not cap.isOpened():
    print(json.dumps({"err": f"Cannot open {path}"}))
    sys.exit(1)

fps   = cap.get(cv.CAP_PROP_FPS) or 24.0
w     = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
h     = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# --- HSV range for a “typical” orange match ball ---
# Tweak if needed (indoor/outdoor lighting shifts hue):
LOW1  = np.array([5,  110, 120], np.uint8)
HIGH1 = np.array([20, 255, 255], np.uint8)
# Optional second band for red-orange tails (wraparound not used here):
LOW2  = np.array([0,  130, 120], np.uint8)
HIGH2 = np.array([5,  255, 255], np.uint8)

# Kalman-ish smoothing
alpha_pos = 0.35
alpha_vel = 0.25

px, py, vx, vy = None, None, 0.0, 0.0
radius_px_min, radius_px_max = 4, 50   # widened for motion-blurred ball on windy days
keyframes = []

frame_idx = 0
while True:
    ok, frame_bgr = cap.read()
    if not ok: break
    t = frame_idx / fps
    frame_idx += 1

    hsv = cv.cvtColor(frame_bgr, cv.COLOR_BGR2HSV)
    mask1 = cv.inRange(hsv, LOW1, HIGH1)
    mask2 = cv.inRange(hsv, LOW2, HIGH2)
    mask = cv.bitwise_or(mask1, mask2)

    # Clean: small remove + close holes
    mask = cv.medianBlur(mask, 5)
    kernel = np.ones((5,5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    # Contours -> pick most ball-like (area + circularity)
    contours,_ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cand = None
    best_score = 0.0
    for c in contours:
        area = cv.contourArea(c)
        if area < 30: continue
        (cx, cy), rad = cv.minEnclosingCircle(c)
        if rad < radius_px_min or rad > radius_px_max: continue
        perim = cv.arcLength(c, True)
        circ = 0.0 if perim == 0 else 4*math.pi*area/(perim*perim)
        # Reject flat/long blobs — lowered from 0.6 to 0.4 so motion-blurred
        # oval shapes (wind/camera shake) can still be detected.
        if circ < 0.4: continue
        # Score: circularity * size (favor stable balls)
        score = circ * min(1.0, rad/20.0)
        if score > best_score:
            best_score = score
            cand = (cx, cy, rad)

    conf = 0.0
    if cand is not None:
        cx, cy, rad = cand
        conf = float(min(1.0, best_score))
        if px is None:
            px, py = cx, cy
        # crude velocity estimate
        vx = (1-alpha_vel)*vx + alpha_vel*(cx - px)
        vy = (1-alpha_vel)*vy + alpha_vel*(cy - py)
        # position smoothing
        px = (1-alpha_pos)*px + alpha_pos*cx
        py = (1-alpha_pos)*py + alpha_pos*cy
    else:
        # no detection: gentle dead-reckon/decay — 0.95 coasts longer through
        # wind-induced detection gaps
        vx *= 0.95; vy *= 0.95
        if px is not None:
            px += vx; py += vy

    if px is not None:
        keyframes.append({"t": round(t,3), "x": float(px), "y": float(py), "conf": round(conf,3)})

cap.release()

# Reduce to ~12–18 knots with highest confidence and time coverage
if keyframes:
    # keep every Nth plus local maxima by conf
    N = max(1, int(len(keyframes)/300))  # ~ <=300 raw points
    subsampled = keyframes[::N]
    # also take top K by conf to anchor critical points
    topK = sorted(keyframes, key=lambda k: k["conf"], reverse=True)[:18]
    # merge by time uniqueness
    used = set()
    knots = []
    for k in sorted(subsampled + topK, key=lambda kk: kk["t"]):
        tt = k["t"]
        if all(abs(tt - z["t"]) > 0.12 for z in knots):  # 120 ms spacing
            knots.append(k)
    # clamp to field width
    for k in knots:
        k["x"] = float(max(0, min(w-1, k["x"])))
        k["y"] = float(max(0, min(h-1, k["y"])))
    out = {"w": w, "h": h, "fps": fps, "knots": knots}
else:
    out = {"w": w, "h": h, "fps": fps, "knots": [], "warn": "no_ball_detected"}

print(json.dumps(out))

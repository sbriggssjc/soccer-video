import json
import numpy as np
from scipy.signal import savgol_filter


SPEED_THRESHOLD = 3500
ZOOM_OUT_FACTOR = 1.12


def hybrid_elite_process(
    in_path: str,
    out_path: str,
    smooth_window: int = 11,
    smooth_poly: int = 3,
    zoom_smooth: float = 0.85,
    predict_lookahead: float = 0.08,   # seconds to predict ahead
    fps: int = 24,
):
    """
    Hybrid Elite Follow Mode:
    1. Smooth jitter from exact-follow
    2. Predict forward for fast motion
    3. Preserve zoom but smooth slightly
    """

    # Load frames
    rows = []
    with open(in_path, "r") as f:
        for line in f:
            rows.append(json.loads(line))

    cx = np.array([r["cx"] for r in rows])
    cy = np.array([r["cy"] for r in rows])
    zoom = np.array([r.get("zoom", 1.0) for r in rows])

    # ─────────────────────────────────────────────
    # 1. Savitzky–Golay smoothing (low jitter, no lag)
    # ─────────────────────────────────────────────
    cx_s = savgol_filter(cx, smooth_window, smooth_poly)
    cy_s = savgol_filter(cy, smooth_window, smooth_poly)
    zoom_s = (
        zoom_smooth * savgol_filter(zoom, smooth_window, smooth_poly)
        + (1 - zoom_smooth) * zoom
    )

    # ─────────────────────────────────────────────
    # 2. Predictive lookahead based on velocity
    # ─────────────────────────────────────────────
    dt = 1.0 / fps
    vx = np.gradient(cx_s) / dt
    vy = np.gradient(cy_s) / dt

    # Look ahead by predict_lookahead seconds
    cx_pred = cx_s + vx * predict_lookahead
    cy_pred = cy_s + vy * predict_lookahead

    # Clip positions for sanity
    cx_pred = np.clip(cx_pred, 0, 3840)
    cy_pred = np.clip(cy_pred, 0, 2160)

    # Auto zoom-out when motion spikes
    speed = np.sqrt(vx ** 2 + vy ** 2)
    zoom_s = np.where(speed > SPEED_THRESHOLD, zoom_s * ZOOM_OUT_FACTOR, zoom_s)

    # ─────────────────────────────────────────────
    # 3. Write output
    # ─────────────────────────────────────────────
    with open(out_path, "w") as f:
        for i, r in enumerate(rows):
            r["cx"] = float(cx_pred[i])
            r["cy"] = float(cy_pred[i])
            r["zoom"] = float(zoom_s[i])
            f.write(json.dumps(r) + "\n")

    return out_path

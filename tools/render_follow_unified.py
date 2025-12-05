"""Unified ball-lock renderer.

This script consolidates the historical ``render_follow_*`` variants into a single
implementation that reproduces the behaviour of the "good tester clip" while
remaining configurable through presets and CLI overrides.

The module is intentionally self-contained so that the calibration and debug
helpers can import and reuse the building blocks (label loading, camera
planning, etc.).  The implementation is optimised for clarity and predictable
Windows behaviour rather than raw performance.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, TextIO, Tuple, Union

from math import hypot
from statistics import median

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def load_any_telemetry(path):
    """
    Unified loader for ball telemetry (JSONL) and follow telemetry
    (JSON or JSONL). Returns a list of dicts with keys:
        t, cx, cy, zoom, valid
    No other fields required.
    """

    import json, os

    rows = []

    # JSON follow telemetry (single JSON containing "keyframes")
    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            root = json.load(f)
        kfs = root.get("keyframes", [])
        for kf in kfs:
            rows.append({
                "t": kf.get("t"),
                "cx": kf.get("cx"),
                "cy": kf.get("cy"),
                "zoom": kf.get("zoom", 1.0),
                "valid": True,
            })
        return rows

    # JSONL (either ball telemetry or flattened follow telemetry)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Modern follow telemetry (already has cx/cy/zoom)
            if "cx" in row and "cy" in row:
                rows.append({
                    "t": row.get("t"),
                    "cx": row.get("cx"),
                    "cy": row.get("cy"),
                    "zoom": row.get("zoom", 1.0),
                    "valid": True,
                })
                continue

            # Legacy ball telemetry (ball→camera logic will fill zoom later)
            if "ball" in row or "ball_src" in row:
                cx = row.get("cx")
                cy = row.get("cy")
                if cx is None or cy is None:
                    continue
                rows.append({
                    "t": row.get("t"),
                    "cx": cx,
                    "cy": cy,
                    "zoom": 1.0,
                    "valid": True,
                })
                continue

    return rows


def _safe_float(v):
    if not math.isfinite(f):
        return None
    return f


def safe_float(val, default):
    try:
        if val is None:
            return default
        x = float(val)
        if not math.isfinite(x):
            return default
        return x
    except Exception:
        return default


def _load_ball_telemetry(path):
    from render_follow_unified import load_any_telemetry

    telemetry_rows = load_any_telemetry(path)
    if not telemetry_rows:
        raise ValueError(f"No usable telemetry in {path}")

    out = []
    for row in telemetry_rows:
        cx = _safe_float(row.get("cx"))
        cy = _safe_float(row.get("cy"))
        if cx is None or cy is None:
            continue
        rec = dict(row)
        rec["cx"] = cx
        rec["cy"] = cy
        out.append(rec)

    return out

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Ball-follow tuning constants. These are intentionally easy to tweak so we can
# dial in a responsive-but-human feel without digging through the rendering
# code. Increase BALL_FOLLOW_ALPHA for snappier response (more jitter), or
# decrease it for smoother, lazier motion. Tighten BALL_MAX_DX/BALL_MAX_DY to
# further clamp per-frame motion when the ball makes sharp cuts.
BALL_FOLLOW_ALPHA = 0.25
BALL_MAX_DX = 60.0
BALL_MAX_DY = 40.0

# Ball-follow camera path tuning (portrait reels)
LEAD_WINDOW_FACTOR = 0.35  # ~0.35s lookahead
BALL_PATH_SMOOTHING_RADIUS = 5
CAM_SMOOTH_ALPHA_X = 0.22
CAM_SMOOTH_ALPHA_Y = 0.08
CAM_SMOOTH_ALPHA_ZOOM = 0.08
MAX_CAM_DX_PER_FRAME_FRAC = 0.03  # relative to frame width
MAX_CAM_DY_PER_FRAME_FRAC = 0.012
TELEPORT_THRESHOLD_FRAC = 0.14  # treat sudden jumps as long passes
EASE_FRAMES_FOR_TELEPORT = 10
BALL_CENTER_TOLERANCE_PCT = 0.20
ZOOM_MIN = 1.0
ZOOM_MAX = 1.4

BALL_CAM_CONFIG: dict[str, object] = {
    "min_coverage": 0.4,
    "min_confidence": 0.25,
    "lead_frames": 3,
    "base_alpha": 0.20,
    "fast_alpha": 0.60,
    "catchup_thresh_px": 80.0,
    "ball_margin_px": 80.0,
    "final_smooth_alpha": 0.1,
    "final_smooth_passes": 2,
    "max_pan_per_frame": 40.0,
    "max_accel_per_frame": 20.0,
    "zoom": {
        "base_crop_width": 1080,
        "min_crop_width": 800,
        "max_crop_width": 1080,
        "zoom_alpha": 0.2,
        "max_zoom_delta": 30.0,
    },
}

from tools.ball_telemetry import (
    BallSample,
    load_and_interpolate_telemetry,
    load_ball_telemetry,
    load_ball_telemetry_for_clip,
    set_telemetry_frame_bounds,
    telemetry_path_for_video,
)
from tools.offline_portrait_planner import (
    OfflinePortraitPlanner,
    PlannerConfig,
    keyframes_to_arrays,
    load_plan,
    plan_ball_portrait_crop,
)
from tools.upscale import upscale_video

# Confidence thresholds for telemetry-driven selection
BALL_CONF_THRESH = 0.5
PLAYER_CONF_THRESH = 0.5


def load_ball_path_from_jsonl(path: str, logger=None):
    """
    Return (ball_x, ball_y, stats) from a telemetry jsonl file.

    Prefers explicit ``ball_x/ball_y`` pairs when available, then falls back to
    legacy ``ball_src`` or ``ball`` tuples.
    """

    from render_follow_unified import load_any_telemetry

    telemetry_rows = load_any_telemetry(path)
    if not telemetry_rows:
        raise ValueError(f"No usable telemetry in {path}")

    xs: list[float] = []
    ys: list[float] = []
    confs: list[float] = []
    total_rows = len(telemetry_rows)
    kept_rows = 0
    high_conf_valid_frames = 0

    for row in telemetry_rows:
        cx = _safe_float(row.get("cx"))
        cy = _safe_float(row.get("cy"))
        if cx is None or cy is None:
            continue
        xs.append(float(cx))
        ys.append(float(cy))
        kept_rows += 1

        conf_val = _safe_float(row.get("conf"))
        if conf_val is not None:
            confs.append(conf_val)
        if bool(row.get("valid", True)) and (conf_val is None or conf_val >= BALL_CONF_THRESH):
            high_conf_valid_frames += 1

    xs_arr = np.asarray(xs, dtype=np.float32)
    ys_arr = np.asarray(ys, dtype=np.float32)
    meta = {
        "total_rows": total_rows,
        "kept_rows": kept_rows,
        "avg_conf": float(np.mean(confs)) if confs else 1.0,
        "telemetry_quality": float(high_conf_valid_frames) / float(total_rows) if total_rows else 0.0,
    }

    if logger:
        logger.info(
            "[BALL-TELEMETRY] ball_src_x=[%.1f, %.1f], ball_src_y=[%.1f, %.1f], kept=%d/%d, conf=%.2f",
            float(xs_arr.min()),
            float(xs_arr.max()),
            float(ys_arr.min()),
            float(ys_arr.max()),
            kept_rows,
            total_rows,
            meta["avg_conf"],
        )

    return xs_arr, ys_arr, meta


def emit_follow_telemetry(
    path: str | os.PathLike[str] | None,
    cx: Sequence[float],
    cy: Sequence[float],
    zoom: Sequence[float],
    *,
    workdir: str | os.PathLike[str] | None = None,
    basename: str | None = None,
) -> str:
    """Write a follow-telemetry JSONL file from camera centers.

    The output filename is derived from ``basename`` (or the telemetry stem) and
    is always placed under ``workdir`` (or the telemetry directory).
    """

    stem = basename or (Path(path).stem if path else "follow")
    root = Path(workdir) if workdir is not None else (Path(path).parent if path else Path.cwd())
    root.mkdir(parents=True, exist_ok=True)

    cx = [0.0 if v is None else float(v) for v in cx]
    cy = [0.0 if v is None else float(v) for v in cy]
    zoom = [1.0 if v is None else float(v) for v in zoom]

    out_path = root / f"{stem}.follow.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for i in range(len(cx)):
            cx_i = float(cx[i] or 0)
            cy_i = float(cy[i] or 0)
            zoom_i = float(zoom[i] or 1.0)

            f.write(
                json.dumps(
                    {
                        "f": i,
                        "cx": cx_i,
                        "cy": cy_i,
                        "zoom": zoom_i,
                    }
                )
                + "\n"
            )
    return os.fspath(out_path)


def smooth_follow_telemetry(path: str | os.PathLike[str]) -> str:
    """Apply smoothing to a follow telemetry file and return the smoothed path."""

    cx_vals: list[float] = []
    cy_vals: list[float] = []
    zoom_vals: list[float] = []

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

    smoothed_cx, smoothed_cy = smooth_and_limit_camera_path(cx_vals, cy_vals)
    zoom_arr = np.asarray(zoom_vals, dtype=float)
    if zoom_arr.size > 0:
        kernel = np.array([0.25, 0.5, 0.25], dtype=float)
        zoom_smoothed = np.convolve(zoom_arr, kernel, mode="same")
    else:
        zoom_smoothed = zoom_arr

    out_path = Path(path).with_name(f"{Path(path).stem}__smooth.jsonl")
    with out_path.open("w", encoding="utf-8") as handle:
        for idx, (x, y) in enumerate(zip(smoothed_cx, smoothed_cy)):
            handle.write(
                json.dumps(
                    {
                        "f": idx,
                        "cx": float(x),
                        "cy": float(y),
                        "zoom": float(zoom_smoothed[idx]) if idx < len(zoom_smoothed) else 1.0,
                    }
                )
                + "\n"
            )

    return os.fspath(out_path)


def edge_zoom_out(
    cx,
    cy,
    bx,
    by,
    crop_w,
    crop_h,
    W,
    H,
    margin_px,
    s_cap=1.30,
    *,
    edge_frac: float = 1.0,
):
    """Return a zoom-out multiplier that keeps the ball off the crop edge.

    ``s_out`` scales the crop dimensions (``eff_w = crop_w * s_out``) and is
    constrained by ``s_cap`` and the available border headroom.  ``edge_frac``
    softens the onset of the zoom so that we start nudging the crop outward
    before the ball fully breaches the requested ``margin_px``.
    """

    hx, hy = 0.5 * crop_w, 0.5 * crop_h
    dx, dy = abs(bx - cx), abs(by - cy)

    margin_px = max(0.0, float(margin_px))
    s_cap = max(1.0, float(s_cap))
    if not math.isfinite(edge_frac) or edge_frac <= 0.0:
        edge_frac = 1.0

    def _axis_scale(delta: float, half: float) -> float:
        if half <= 0.0:
            return 1.0

        need = max(1.0, (delta + margin_px) / max(half, 1e-6))
        if margin_px <= 0.0:
            return need

        actual_margin = max(0.0, half - delta)
        trigger_margin = margin_px / edge_frac if edge_frac > 0.0 else half
        trigger_margin = max(margin_px, min(trigger_margin, half))

        if actual_margin >= trigger_margin:
            return 1.0
        if actual_margin <= margin_px or trigger_margin <= margin_px:
            return need

        blend = (trigger_margin - actual_margin) / max(
            trigger_margin - margin_px, 1e-6
        )
        return 1.0 + blend * (need - 1.0)

    s_soft_x = _axis_scale(dx, hx)
    s_soft_y = _axis_scale(dy, hy)
    s_ball = max(1.0, s_soft_x, s_soft_y)

    # Border headroom: max zoom-out we can afford without leaving the image
    s_max_l = cx / max(hx, 1e-6)
    s_max_r = (W - 1 - cx) / max(hx, 1e-6)
    s_max_t = cy / max(hy, 1e-6)
    s_max_b = (H - 1 - cy) / max(hy, 1e-6)
    s_border = max(1.0, min(s_max_l, s_max_r, s_max_t, s_max_b))

    return max(1.0, min(s_ball, min(s_border, s_cap)))


def _inside_crop(bx, by, cx, cy, crop_w, crop_h, margin):
    x0 = cx - crop_w / 2 + margin
    x1 = cx + crop_w / 2 - margin
    y0 = cy - crop_h / 2 + margin
    y1 = cy + crop_h / 2 - margin
    return (bx >= x0) and (bx <= x1) and (by >= y0) and (by <= y1)


def _clamp_cam(cx, cy, W, H, crop_w, crop_h):
    cx = max(crop_w / 2, min(W - crop_w / 2, cx))
    cy = max(crop_h / 2, min(H - crop_h / 2, cy))
    return cx, cy


def _motion_centroid(
    prev_gray: Optional[np.ndarray],
    cur_gray: Optional[np.ndarray],
    field_mask: Optional[np.ndarray],
    flow_thresh_px: float = 1.6,
) -> Optional[Tuple[float, float]]:
    if prev_gray is None or cur_gray is None:
        return None
    mag = cv2.magnitude(flow[..., 0], flow[..., 1])
    mot = (mag >= float(flow_thresh_px)).astype(np.uint8) * 255
    if field_mask is not None and field_mask.size == mag.size:
        mot = cv2.bitwise_and(mot, field_mask)
    mot = cv2.morphologyEx(mot, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"] <= 1e-3:
        return None
    cx = float(M["m10"] / M["m00"])
    cy = float(M["m01"] / M["m00"])
    return cx, cy


def _get_ball_xy_src(rec, src_w, src_h):
    """
    Return ball center in *source pixel space* (x,y), regardless of which fields exist in the record.
    Accepts bx/by, bx_stab/by_stab, bx_raw/by_raw, or normalized u/v.
    """
    # priority: stabilized, then plain, then raw
    for kx, ky in (("bx_stab", "by_stab"), ("bx", "by"), ("bx_raw", "by_raw")):
        if kx in rec and ky in rec:
            return float(rec[kx]), float(rec[ky])

    # normalized fallback (0..1); tolerate slight overshoot
    if "u" in rec and "v" in rec:
        u = float(rec["u"])
        v = float(rec["v"])
        return max(0.0, min(1.0, u)) * (src_w - 1), max(0.0, min(1.0, v)) * (src_h - 1)

    # last resort: not found
    return None, None


def edge_aware_zoom(
    cx: float,
    cy: float,
    bx: Optional[float],
    by: Optional[float],
    cw: float,
    ch: float,
    width: float,
    height: float,
    margin_px: float,
    *,
    s_min: float = 0.75,
) -> float:
    """Return a zoom scale (<= 1.0) that avoids edge clamps while keeping the ball inside a margin."""

    if cw <= 0.0 or ch <= 0.0 or width <= 0.0 or height <= 0.0:
        return 1.0

    cx = float(cx)
    cy = float(cy)
    cw = float(cw)
    ch = float(ch)
    width = float(width)
    height = float(height)
    margin_px = max(0.0, float(margin_px))

    s_needed = 1.0

    half_w = cw / 2.0
    half_h = ch / 2.0

    if half_w <= 0.0 or half_h <= 0.0:
        return 1.0

    # compute minimum scale needed to avoid clamping against source edges
    s_clamp_x = 1.0
    s_clamp_y = 1.0
    if cx - half_w < 0.0:
        s_clamp_x = min(s_clamp_x, (max(cx, 0.0) * 2.0) / max(cw, 1e-6))
    if cx + half_w > width:
        s_clamp_x = min(s_clamp_x, (max(width - cx, 0.0) * 2.0) / max(cw, 1e-6))
    if cy - half_h < 0.0:
        s_clamp_y = min(s_clamp_y, (max(cy, 0.0) * 2.0) / max(ch, 1e-6))
    if cy + half_h > height:
        s_clamp_y = min(s_clamp_y, (max(height - cy, 0.0) * 2.0) / max(ch, 1e-6))

    s_needed = min(s_needed, s_clamp_x, s_clamp_y)

    if bx is not None and by is not None and math.isfinite(bx) and math.isfinite(by):
        bx = float(bx)
        by = float(by)
        dx = abs(bx - cx)
        dy = abs(by - cy)
        if margin_px > 0.0:
            hx_margin = max(half_w - margin_px, 0.0)
            hy_margin = max(half_h - margin_px, 0.0)
            if hx_margin > 0.0 and dx > hx_margin:
                s_needed = min(s_needed, hx_margin / max(dx, 1e-6))
            if hy_margin > 0.0 and dy > hy_margin:
                s_needed = min(s_needed, hy_margin / max(dy, 1e-6))

    s_min = max(0.0, min(1.0, float(s_min)))
    s_needed = max(s_min, min(1.0, float(s_needed)))
    return float(s_needed)


def smooth_and_limit_camera_path(
    cx_path: Sequence[float],
    cy_path: Sequence[float],
    *,
    max_dx: float = BALL_MAX_DX,
    max_dy: float = BALL_MAX_DY,
    alpha: float = BALL_FOLLOW_ALPHA,
) -> tuple[list[float], list[float]]:
    """
    Given per-frame crop centers (cx_path, cy_path), apply exponential
    smoothing followed by a per-frame slew-rate clamp to keep the virtual
    camera feeling human-operated instead of snapping.

    - ``alpha`` controls responsiveness. Higher values react faster but can
      jitter; lower values are smoother.
    - ``max_dx`` / ``max_dy`` cap the per-frame delta so whip-pans are
      softened.
    """

    cx_s = list(cx_path)
    cy_s = list(cy_path)

    for i in range(1, len(cx_s)):
        cx_s[i] = alpha * cx_s[i] + (1.0 - alpha) * cx_s[i - 1]
        cy_s[i] = alpha * cy_s[i] + (1.0 - alpha) * cy_s[i - 1]

    for i in range(1, len(cx_s)):
        dx = cx_s[i] - cx_s[i - 1]
        dy = cy_s[i] - cy_s[i - 1]

        if abs(dx) > max_dx:
            cx_s[i] = cx_s[i - 1] + math.copysign(max_dx, dx)
        if abs(dy) > max_dy:
            cy_s[i] = cy_s[i - 1] + math.copysign(max_dy, dy)

    return cx_s, cy_s


def smooth_center_path(cx_path, cy_path, window=9, max_step_px=40.0):
    """
    Heavy smoothing so the camera feels human-operated.
    - window: moving average window
    - max_step_px: max allowed jump per frame for the center
    """

    n = len(cx_path)
    if n == 0:
        return list(cx_path), list(cy_path)

    def smooth_1d(values):
        half = window // 2
        out = [0.0] * n
        for i in range(n):
            j0 = max(0, i - half)
            j1 = min(n, i + half + 1)
            out[i] = sum(values[j0:j1]) / (j1 - j0)
        return out

    sx = smooth_1d(cx_path)
    sy = smooth_1d(cy_path)

    # Limit per-frame speed (jerk guard)
    for i in range(1, n):
        dx = sx[i] - sx[i - 1]
        dy = sy[i] - sy[i - 1]
        dist = (dx * dx + dy * dy) ** 0.5
        if dist > max_step_px and dist > 0:
            scale = max_step_px / dist
            sx[i] = sx[i - 1] + dx * scale
            sy[i] = sy[i - 1] + dy * scale

    return sx, sy


def build_raw_ball_center_path(
    telemetry: Sequence[Mapping[str, object]],
    frame_width: int,
    frame_height: int,
    crop_width: int,
    crop_height: int,
    *,
    default_y_frac: float = 0.45,
    vertical_bias_frac: float = 0.08,
) -> tuple[list[float], list[float]]:
    """
    Build raw crop centers from telemetry, clamping to keep the crop window in
    bounds and biasing the view slightly above the ball so there's forward
    field context.

    Missing or invalid telemetry frames are filled by carrying forward the last
    valid position, falling back to a neutral center when needed.
    """

    n_frames = len(telemetry)
    if n_frames <= 0:
        return [], []

    half_w = float(crop_width) / 2.0
    half_h = float(crop_height) / 2.0
    default_cx = float(frame_width) / 2.0
    default_cy = float(frame_height) * float(default_y_frac)
    bias_px = float(frame_height) * float(vertical_bias_frac)

    def clamp_center(cx: float, cy: float) -> tuple[float, float]:
        if crop_width >= frame_width:
            cx_clamped = float(frame_width) / 2.0
        else:
            max_x0 = max(0.0, float(frame_width) - float(crop_width))
            x0 = min(max(cx - half_w, 0.0), max_x0)
            cx_clamped = x0 + half_w

        if crop_height >= frame_height:
            cy_clamped = float(frame_height) / 2.0
        else:
            max_y0 = max(0.0, float(frame_height) - float(crop_height))
            y0 = min(max(cy - half_h, 0.0), max_y0)
            cy_clamped = y0 + half_h

        return cx_clamped, cy_clamped

    cx_vals: list[float] = []
    cy_vals: list[float] = []
    last_valid: tuple[float, float] | None = None

    for rec in telemetry:
        bx_raw = rec.get("x") if isinstance(rec, Mapping) else None
        by_raw = rec.get("y") if isinstance(rec, Mapping) else None
        vis = bool(rec.get("visible")) if isinstance(rec, Mapping) else False


        if vis and math.isfinite(bx_val) and math.isfinite(by_val):
            last_valid = (bx_val, by_val)

        if last_valid is not None:
            bx_use, by_use = last_valid
        else:
            bx_use, by_use = default_cx, default_cy

        cx_val = float(bx_use)
        cy_val = float(by_use - bias_px)
        cx_val, cy_val = clamp_center(cx_val, cy_val)

        cx_vals.append(cx_val)
        cy_vals.append(cy_val)

    return cx_vals, cy_vals


def clamp_center_path_to_bounds(
    cx_path: Sequence[float],
    cy_path: Sequence[float],
    frame_width: int,
    frame_height: int,
    crop_width: int,
    crop_height: int,
) -> tuple[list[float], list[float]]:
    """Clamp center paths so the implied crop stays inside the source frame."""

    if len(cx_path) != len(cy_path):
        return list(cx_path), list(cy_path)

    clamped_cx: list[float] = []
    clamped_cy: list[float] = []

    max_x0 = max(0.0, float(frame_width) - float(crop_width))
    max_y0 = max(0.0, float(frame_height) - float(crop_height))

    for cx, cy in zip(cx_path, cy_path):
        x0 = int(round(float(cx) - float(crop_width) / 2.0))
        y0 = int(round(float(cy) - float(crop_height) / 2.0))
        x0 = max(0, min(x0, int(max_x0)))
        y0 = max(0, min(y0, int(max_y0)))
        clamped_cx.append(float(x0 + float(crop_width) / 2.0))
        clamped_cy.append(float(y0 + float(crop_height) / 2.0))

    return clamped_cx, clamped_cy


def _clamp_ball_cam_center(cx: float, *, crop_width: float, frame_width: float) -> float:
    half = crop_width / 2.0
    return max(half, min(frame_width - half, cx))


def _jerk95(px: np.ndarray, *, fps: float) -> float:
    if len(px) < 4 or fps <= 0:
        return 0.0
    v = np.diff(px)
    a = np.diff(v)
    j = np.diff(a) * (fps**3)
    j_abs = np.abs(j)
    return float(np.percentile(j_abs, 95)) if j_abs.size else 0.0


def _interp_nan(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    out = values.copy()
    n = len(out)
    idx = np.arange(n)
    mask = np.isfinite(out)
    if not mask.any():
        return out
    out[~mask] = np.interp(idx[~mask], idx[mask], out[mask])
    return out


def smooth_path(values: Sequence[float], alpha: float = 0.2) -> list[float]:
    """Simple exponential smoother to tame frame-to-frame jitter."""

    smoothed: list[float] = []
    prev: float | None = None
    for v in values:
        if prev is None:
            prev = float(v)
        else:
            prev = alpha * float(v) + (1.0 - alpha) * prev
        smoothed.append(prev)
    return smoothed


def smooth_series(values, alpha: float = 0.1, passes: int = 3):
    """
    Smooth a 1D sequence using an exponential moving average, applied
    multiple passes to strongly low-pass the motion.

    Args:
        values: iterable of floats (e.g., ball cx samples).
        alpha: EMA smoothing factor in (0, 1]; lower = smoother.
        passes: how many forward/backward EMA passes to apply.

    Returns:
        List of smoothed values, same length as input.
    """
    vals = list(values)
    n = len(vals)
    if n <= 1:
        return vals

    for _ in range(passes):
        # forward pass
        prev = vals[0]
        for i in range(1, n):
            v = vals[i]
            prev = alpha * v + (1.0 - alpha) * prev
            vals[i] = prev

        # backward pass (for more “camera operator” feel)
        prev = vals[-1]
        for i in range(n - 2, -1, -1):
            v = vals[i]
            prev = alpha * v + (1.0 - alpha) * prev
            vals[i] = prev

    return vals


def smooth_centered(seq, window):
    """
    Zero-lag, centered moving average with edge handling by reflection.
    window must be an odd integer >= 1.
    """
    n = len(seq)
    if n == 0 or window <= 1:
        return list(seq)

    if window % 2 == 0:
        window += 1  # ensure odd

    half = window // 2
    out = [0.0] * n

    for i in range(n):
        acc = 0.0
        cnt = 0
        for k in range(-half, half + 1):
            j = i + k
            # reflect at edges
            if j < 0:
                j = -j
            elif j >= n:
                j = 2 * n - j - 2
            acc += seq[j]
            cnt += 1
        out[i] = acc / cnt
    return out


def _load_ball_cam_array(path: Path, num_frames: int) -> np.ndarray:
    samples = load_ball_telemetry(path)
    arr = np.full((num_frames, 3), np.nan, dtype=float)
    for sample in samples:
        if frame_idx < 0 or frame_idx >= num_frames:
            continue
        conf = _safe_float(getattr(sample, "conf", None))
        x = _safe_float(getattr(sample, "x", None))
        y = _safe_float(getattr(sample, "y", None))
        if x is None or y is None:
            continue
        arr[frame_idx, 0] = x
        arr[frame_idx, 1] = y
        arr[frame_idx, 2] = conf if conf is not None else 0.0
    return arr


def load_ball_telemetry_jsonl(path: str, src_w: int, src_h: int, logger=None):
    if logger:
        logger.info("[BALL-TELEMETRY] loading %s", path)

    ball_samples = _load_ball_telemetry(path)
    xs = [s["cx"] for s in ball_samples]
    ys = [s["cy"] for s in ball_samples]

    if logger:
        logger.info(
            "[BALL-TELEMETRY] tele_range_x=[%.1f, %.1f], tele_range_y=[%.1f, %.1f]",
            min(xs),
            max(xs),
            min(ys),
            max(ys),
        )

    return xs, ys


def build_raw_ball_path(telemetry: np.ndarray, fps: float) -> np.ndarray:
    """Return Nx2 array of raw ball positions with finite interpolation."""

    if telemetry.size == 0:
        return np.zeros((0, 2), dtype=float)

    valid_mask = np.isfinite(telemetry[:, 0]) & np.isfinite(telemetry[:, 1]) & (
        telemetry[:, 2] >= 0.5
    )
    raw = np.full((telemetry.shape[0], 2), np.nan, dtype=float)
    raw[valid_mask, 0] = telemetry[valid_mask, 0]
    raw[valid_mask, 1] = telemetry[valid_mask, 1]

    raw[:, 0] = _interp_nan(raw[:, 0])
    raw[:, 1] = _interp_nan(raw[:, 1])
    return raw


def build_target_ball_path(raw_path: np.ndarray, fps: float) -> np.ndarray:
    """Pre-smooth raw detections and add predictive lead."""

    if raw_path.size == 0:
        return raw_path

    radius = max(1, int(BALL_PATH_SMOOTHING_RADIUS))
    kernel_size = 2 * radius + 1
    kernel = np.ones(kernel_size, dtype=float) / kernel_size

    padded_x = np.pad(raw_path[:, 0], (radius, radius), mode="edge")
    padded_y = np.pad(raw_path[:, 1], (radius, radius), mode="edge")
    smooth_x = np.convolve(padded_x, kernel, mode="valid")
    smooth_y = np.convolve(padded_y, kernel, mode="valid")
    smooth = np.stack([smooth_x, smooth_y], axis=1)

    lead_frames = max(1, int(round(max(fps, 1.0) * LEAD_WINDOW_FACTOR)))
    target = smooth.copy()
    n = len(smooth)
    for i in range(n):
        j1 = min(n, i + lead_frames)
        window = smooth[i:j1]
        if window.size == 0:
            continue
        target[i, 0] = float(np.mean(window[:, 0]))
        target[i, 1] = float(np.mean(window[:, 1]))
    return target


def build_camera_path(
    target_ball_path: np.ndarray,
    fps: float,
    base_width: float,
    base_height: float,
    portrait_width: float,
    portrait_height: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return smooth camera center and zoom arrays."""

    if target_ball_path.size == 0:
        return (
            np.zeros(0, dtype=float),
            np.zeros(0, dtype=float),
            np.zeros(0, dtype=float),
        )

    n = len(target_ball_path)
    cam_x = np.zeros(n, dtype=float)
    cam_y = np.zeros(n, dtype=float)
    cam_zoom = np.zeros(n, dtype=float)

    half_w = portrait_width / 2.0
    half_h = portrait_height / 2.0
    max_dy = max(3.0, base_height * MAX_CAM_DY_PER_FRAME_FRAC)

    ball_x = target_ball_path[:, 0]
    ball_y = target_ball_path[:, 1]

    # Smooth + predictive target for X
    smoothed_x = smooth_path(ball_x, alpha=0.25)
    vx = np.diff(ball_x, prepend=ball_x[0])
    lookahead_frames = 4
    predicted_x = [sx + float(v) * lookahead_frames for sx, v in zip(smoothed_x, vx)]
    blend = 0.5
    target_cx = [((1.0 - blend) * sx) + (blend * px) for sx, px in zip(smoothed_x, predicted_x)]

    deadzone_px = 30.0
    max_pan_per_frame = 25.0
    min_dwell_frames = 3
    off_center_count = 0

    current_cx = float(np.clip(target_cx[0], half_w, base_width - half_w))
    cam_x[0] = current_cx
    cam_y[0] = float(np.clip(ball_y[0], half_h, base_height - half_h))
    cam_zoom[0] = ZOOM_MIN

    for i in range(1, n):
        desired_x = float(np.clip(target_cx[i], half_w, base_width - half_w))
        delta = desired_x - current_cx

        if abs(delta) < deadzone_px:
            off_center_count = 0
        else:
            off_center_count += 1
            if off_center_count >= min_dwell_frames:
                delta = max(-max_pan_per_frame, min(max_pan_per_frame, delta))
                current_cx = float(np.clip(current_cx + delta, half_w, base_width - half_w))
                off_center_count = 0

        cam_x[i] = current_cx

        desired_y = float(np.clip(ball_y[i], half_h, base_height - half_h))
        cam_y[i] = cam_y[i - 1] + CAM_SMOOTH_ALPHA_Y * (desired_y - cam_y[i - 1])
        dy = cam_y[i] - cam_y[i - 1]
        if abs(dy) > max_dy:
            cam_y[i] = cam_y[i - 1] + math.copysign(max_dy, dy)
        cam_y[i] = float(np.clip(cam_y[i], half_h, base_height - half_h))

    # Zoom driven by ball speed
    speed = np.abs(np.diff(target_ball_path[:, 0], prepend=target_ball_path[0, 0]))
    speed_norm = np.clip(speed / (base_width * 0.10), 0.0, 1.0)
    desired_zoom = ZOOM_MAX - speed_norm * (ZOOM_MAX - ZOOM_MIN)
    for i in range(n):
        if i == 0:
            cam_zoom[i] = desired_zoom[i]
        else:
            cam_zoom[i] = cam_zoom[i - 1] + CAM_SMOOTH_ALPHA_ZOOM * (desired_zoom[i] - cam_zoom[i - 1])
        cam_zoom[i] = float(np.clip(cam_zoom[i], ZOOM_MIN, ZOOM_MAX))

        crop_w = portrait_width / cam_zoom[i]
        crop_h = portrait_height / cam_zoom[i]
        cam_x[i] = float(np.clip(cam_x[i], crop_w / 2.0, base_width - crop_w / 2.0))
        cam_y[i] = float(np.clip(cam_y[i], crop_h / 2.0, base_height - crop_h / 2.0))

    return cam_x, cam_y, cam_zoom


def compute_locked_ball_cam_path(ball_cx_raw, src_w, crop_w, cfg, logger=None):
    """
    Compute a camera center path that stays locked to a smoothed ball trajectory.
    - ball_cx_raw: list/sequence of ball x positions per frame in source coords
    - src_w: source frame width in pixels
    - crop_w: width of the virtual crop in pixels
    """
    N = len(ball_cx_raw)
    if N == 0:
        return []

    lead_frames = int(cfg.get("ball_cam_lead_frames", 3))
    smooth_window = int(cfg.get("ball_cam_smooth_window", 5))
    max_speed = float(cfg.get("ball_cam_max_speed_px", 60.0))
    margin = float(cfg.get("ball_cam_margin_px", 100.0))

    # Safety clamps
    if smooth_window < 1:
        smooth_window = 1
    if max_speed <= 0:
        max_speed = 1.0

    half_crop_w = crop_w / 2.0

    cam_cx = [0.0] * N

    # Initialize at the ball position (with lead) on frame 0
    j0 = min(lead_frames, N - 1)
    cx0 = ball_cx_raw[j0]
    cx0 = max(half_crop_w - margin, min(src_w - half_crop_w + margin, cx0))
    cam_cx[0] = cx0

    for i in range(1, N):
        # Look ahead a little to avoid visual lag
        j = min(i + lead_frames, N - 1)

        # Local smoothing window around the (possibly leaded) index
        w = smooth_window
        start = max(0, j - w // 2)
        end = min(N, j + w // 2 + 1)
        count = end - start
        if count <= 0:
            desired = ball_cx_raw[j]
        else:
            desired = sum(ball_cx_raw[start:end]) / float(count)

        # Clamp movement per frame to avoid crazy jumps
        prev = cam_cx[i - 1]
        delta = desired - prev

        if delta > max_speed:
            delta = max_speed
        elif delta < -max_speed:
            delta = -max_speed

        cx = prev + delta

        # Keep crop inside frame, honoring margin
        cx = max(half_crop_w - margin, min(src_w - half_crop_w + margin, cx))

        cam_cx[i] = cx

    if logger:
        logger.info(
            "[BALL-CAM LOCK] N=%d, cx_range=[%.1f, %.1f], lead_frames=%d, "
            "smooth_window=%d, max_speed_px=%.1f, margin=%.1f",
            N,
            min(cam_cx),
            max(cam_cx),
            lead_frames,
            smooth_window,
            max_speed,
            margin,
        )

    return cam_cx


def compute_ball_lock_strict(
    ball_cx_raw,
    ball_cy_raw,
    src_w,
    src_h,
    crop_w,
    crop_h,
    cfg,
    logger=None,
):
    """
    Compute a strict ball-locked camera path:
    - For each frame, choose a crop center (cam_cx, cam_cy) that keeps the ball
      inside the 9:16 crop with a margin.
    - Uses a small temporal smoothing window + a lead in time to avoid visual lag.
    - Clamps camera so the resulting crop never goes outside the source frame.
    """
    N = len(ball_cx_raw)
    if N == 0:
        return [], []

    # If we don't have vertical telemetry, just fake a flat line so we at least
    # get perfect horizontal behavior.
    if not ball_cy_raw or len(ball_cy_raw) != N:
        ball_cy_raw = [src_h * 0.5] * N

    lead_frames = int(cfg.get("ball_cam_lead_frames", 3))
    smooth_window = int(cfg.get("ball_cam_smooth_window", 5))
    max_speed = float(cfg.get("ball_cam_max_speed_px", 120.0))
    margin = float(cfg.get("ball_cam_margin_px", 80.0))
    vpos = float(cfg.get("ball_cam_vertical_pos", 0.60))

    if smooth_window < 1:
        smooth_window = 1
    if max_speed <= 0:
        max_speed = 1.0
    if vpos < 0.0:
        vpos = 0.0
    if vpos > 1.0:
        vpos = 1.0

    half_w = crop_w / 2.0
    half_h = crop_h / 2.0

    cam_cx = [0.0] * N
    cam_cy = [0.0] * N

    # Build leaded index sequence (so we look a few frames ahead)
    idx_seq = [min(i + lead_frames, N - 1) for i in range(N)]

    # Smooth horizontal, use raw vertical (or you can smooth vertical similarly)
    smoothed_cx = [0.0] * N
    for i in range(N):
        j = idx_seq[i]
        # local average in a window around the leaded index
        start = max(0, j - smooth_window // 2)
        end = min(N, j + smooth_window // 2 + 1)
        count = max(1, end - start)
        smoothed_cx[i] = sum(ball_cx_raw[start:end]) / float(count)

    # Initialize camera at frame 0
    bx0 = smoothed_cx[0]
    by0 = ball_cy_raw[0]

    # Horizontal: center on ball, but clamp to valid crop range
    cx0 = bx0
    cx0 = max(half_w, min(src_w - half_w, cx0))

    # Vertical: place ball at 'vpos' fraction of the crop height (0=top,1=bottom)
    # We want: ball_y = cam_cy - half_h + vpos*crop_h
    cy0 = by0 - (vpos * crop_h - half_h)
    cy0 = max(half_h, min(src_h - half_h, cy0))

    cam_cx[0] = cx0
    cam_cy[0] = cy0

    for i in range(1, N):
        # Desired ball position for this frame
        j = idx_seq[i]
        bx = smoothed_cx[i]
        by = ball_cy_raw[j]

        # Ideal camera center before speed clamp
        desired_cx = bx
        desired_cy = by - (vpos * crop_h - half_h)

        # Clamp to legal center range so crop stays inside frame
        desired_cx = max(half_w, min(src_w - half_w, desired_cx))
        desired_cy = max(half_h, min(src_h - half_h, desired_cy))

        # Speed limit (per-frame) to avoid insane jumps, but keep it generous
        prev_cx = cam_cx[i - 1]
        prev_cy = cam_cy[i - 1]

        dx = desired_cx - prev_cx
        dy = desired_cy - prev_cy

        if dx > max_speed:
            dx = max_speed
        elif dx < -max_speed:
            dx = -max_speed

        if dy > max_speed:
            dy = max_speed
        elif dy < -max_speed:
            dy = -max_speed

        cx = prev_cx + dx
        cy = prev_cy + dy

        # Final clamp
        cx = max(half_w, min(src_w - half_w, cx))
        cy = max(half_h, min(src_h - half_h, cy))

        cam_cx[i] = cx
        cam_cy[i] = cy

    if logger:
        logger.info(
            "[BALL-CAM STRICT] N=%d, cx_range=[%.1f, %.1f], cy_range=[%.1f, %.1f], "
            "lead_frames=%d, smooth_window=%d, max_speed_px=%.1f, margin=%.1f, vpos=%.2f",
            N,
            min(cam_cx),
            max(cam_cx),
            min(cam_cy),
            max(cam_cy),
            lead_frames,
            smooth_window,
            max_speed,
            margin,
            vpos,
        )

    return cam_cx, cam_cy


def compute_ball_lock_raw(
    ball_cx_raw,
    ball_cy_raw,
    src_w,
    src_h,
    crop_w,
    crop_h,
    cfg,
    logger=None,
):
    """
    Hard lock: center the crop on the ball every frame (no smoothing, no speed limit).
    This is mainly for debugging and 'never lose the ball' behavior.
    """
    N = len(ball_cx_raw)
    if N == 0:
        return [], []

    if not ball_cy_raw or len(ball_cy_raw) != N:
        ball_cy_raw = [src_h * 0.5] * N

    vpos = float(cfg.get("ball_cam_vertical_pos", 0.60))
    if vpos < 0.0:
        vpos = 0.0
    if vpos > 1.0:
        vpos = 1.0

    half_w = crop_w / 2.0
    half_h = crop_h / 2.0

    cam_cx = [0.0] * N
    cam_cy = [0.0] * N

    for i in range(N):
        bx = ball_cx_raw[i]
        by = ball_cy_raw[i]

        # Ideal center: ball at vpos inside the crop
        cx = bx
        cy = by - (vpos * crop_h - half_h)

        # Clamp to valid center so crop stays within frame
        cx = max(half_w, min(src_w - half_w, cx))
        cy = max(half_h, min(src_h - half_h, cy))

        cam_cx[i] = cx
        cam_cy[i] = cy

    if logger:
        logger.info(
            "[BALL-CAM RAW] N=%d, cx_range=[%.1f, %.1f], cy_range=[%.1f, %.1f], vpos=%.2f",
            N,
            min(cam_cx),
            max(cam_cx),
            min(cam_cy),
            max(cam_cy),
            vpos,
        )

    return cam_cx, cam_cy


def compute_ema_ball_cam_path(ball_cx_raw, ball_cy_raw, src_w, src_h, crop_w, crop_h, cfg, logger=None):
    N = len(ball_cx_raw)
    if N == 0:
        return [], []

    lead_frames = int(cfg.get("ball_cam_lead_frames", 5))
    base_alpha = float(cfg.get("ball_cam_base_alpha", 0.30))
    fast_alpha = float(cfg.get("ball_cam_fast_alpha", 0.80))
    catchup_px = float(cfg.get("ball_cam_catchup_thresh_px", 40.0))
    margin = float(cfg.get("ball_cam_margin_px", 100.0))

    speed_fast_px = float(cfg.get("ball_cam_speed_fast_px", 40.0))
    max_pan_slow = float(cfg.get("ball_cam_max_pan_per_frame_slow", 24.0))
    max_pan_fast = float(cfg.get("ball_cam_max_pan_per_frame_fast", 44.0))

    half_crop_w = crop_w / 2.0
    cam_cx = [0.0] * N
    cam_cy = [float(src_h) / 2.0] * N
    cx_prev = float(ball_cx_raw[0]) if N > 0 else 0.0

    for i in range(N):
        j = min(i + lead_frames, N - 1)
        desired = float(ball_cx_raw[j])

        if i == 0:
            ball_vel = 0.0
        else:
            ball_vel = float(ball_cx_raw[i] - ball_cx_raw[i - 1])
        speed = abs(ball_vel)

        if speed_fast_px > 0:
            speed_scale = min(1.0, speed / speed_fast_px)
        else:
            speed_scale = 0.0

        alpha = base_alpha + speed_scale * (fast_alpha - base_alpha)

        delta = desired - cx_prev
        if abs(delta) > catchup_px:
            alpha = max(alpha, fast_alpha)

        step = alpha * delta

        max_pan = max_pan_slow + speed_scale * (max_pan_fast - max_pan_slow)
        if step > max_pan:
            step = max_pan
        elif step < -max_pan:
            step = -max_pan

        cx = cx_prev + step

        cx = max(half_crop_w - margin, min(src_w - half_crop_w + margin, cx))

        cam_cx[i] = cx
        cam_cy[i] = float(src_h) / 2.0 if ball_cy_raw is None else cam_cy[i]
        cx_prev = cx

    return cam_cx, cam_cy


def write_ball_crop_debug_clip(
    in_path: str,
    out_path: str,
    ball_cx: list[float],
    ball_cy: list[float],
    crop_x: list[float],
    crop_y: list[float],
    src_w: int,
    src_h: int,
    crop_w: int,
    crop_h: int,
    fps: float,
    logger=None,
):
    """
    Generate a 1920x1080 debug clip showing:
      - the ball telemetry position as a dot
      - the portrait crop window as a rectangle
    on top of the original wide frame.
    """
    if logger:
        logger.info(
            "[BALL-DEBUG] Writing ball/crop overlay debug clip: %s", out_path
        )

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        if logger:
            logger.error("[BALL-DEBUG] Failed to open %s", in_path)
        return

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (src_w, src_h))

    N = min(len(ball_cx), len(crop_x), len(crop_y))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx >= N:
            break

        bx = float(ball_cx[frame_idx])
        by = float(ball_cy[frame_idx])

        x0 = float(crop_x[frame_idx])
        y0 = float(crop_y[frame_idx])

        # Draw ball position (cyan-ish)
        cv2.circle(
            frame,
            (int(round(bx)), int(round(by))),
            8,
            (255, 255, 0),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

        # Draw crop rect (green)
        x0 = int(round(x0))
        y0 = int(round(y0))
        x1 = int(round(x0 + crop_w))
        y1 = int(round(y0 + crop_h))
        cv2.rectangle(
            frame,
            (x0, y0),
            (x1, y1),
            (0, 255, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    if logger:
        logger.info("[BALL-DEBUG] Finished debug clip: %s", out_path)


def build_ball_cam_plan(
    telemetry_path: Path,
    *,
    num_frames: int,
    fps: float,
    frame_width: int,
    frame_height: int,
    portrait_width: int,
    config: Mapping[str, object] | None = None,
    preset_name: str | None = None,
    in_path: Path | None = None,
    out_path: Path | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float]] | tuple[None, dict[str, float]]:
    cfg = dict(BALL_CAM_CONFIG)
    if config:
        cfg.update(config)

    if preset_name == "wide_follow":
        cfg["ball_cam_mode"] = "raw_lock"
        cfg["ball_cam_vertical_pos"] = 0.55
        cfg["ball_cam_margin_px"] = 0.0
        cfg["ball_debug_overlay"] = True

    min_coverage = float(cfg.get("min_coverage", 0.4))

    src_w = float(frame_width)
    src_h = float(frame_height)

    ball_cx_values, ball_cy_values, telemetry_meta = load_ball_path_from_jsonl(telemetry_path, logger)

    vpos = float(cfg.get("ball_cam_vertical_pos", 0.55))

    ball_cx_raw = np.full(num_frames, np.nan, dtype=float)
    ball_cy_raw = np.full(num_frames, np.nan, dtype=float)
    max_fill = min(num_frames, len(ball_cx_values), len(ball_cy_values))
    if max_fill > 0:
        ball_cx_raw[:max_fill] = ball_cx_values[:max_fill]
        ball_cy_raw[:max_fill] = ball_cy_values[:max_fill]

    N = max_fill
    logger.info(
        "[BALL-CAM RAW] N=%d, ball_cx_range=[%.1f, %.1f], ball_cy_range=[%.1f, %.1f], vpos=%.2f",
        N,
        float(np.nanmin(ball_cx_raw[:N])) if N else float("nan"),
        float(np.nanmax(ball_cx_raw[:N])) if N else float("nan"),
        float(np.nanmin(ball_cy_raw[:N])) if N else float("nan"),
        float(np.nanmax(ball_cy_raw[:N])) if N else float("nan"),
        vpos,
    )

    valid_mask = np.isfinite(ball_cx_raw) & np.isfinite(ball_cy_raw)
    coverage = float(np.mean(valid_mask)) if num_frames > 0 else 0.0

    stats = {
        "coverage": coverage,
        "telemetry_rows": int(telemetry_meta.get("total_rows", 0)),
        "telemetry_kept": int(telemetry_meta.get("kept_rows", 0)),
        "telemetry_conf": float(telemetry_meta.get("avg_conf", 1.0)),
        "telemetry_quality": float(telemetry_meta.get("telemetry_quality", coverage)),
    }
    min_conf = float(cfg.get("min_confidence", 0.0))
    if coverage < min_coverage or stats["telemetry_conf"] < min_conf:
        logger.info(
            "[BALL-CAM] coverage/conf too weak (coverage=%.1f%%, conf=%.2f < %.2f); falling back to legacy follow",
            coverage * 100.0,
            stats["telemetry_conf"],
            min_conf,
        )
        return None, stats

    # Build a ball-centric camera path with a small predictive lead and adaptive smoothing.
    ball_cx_raw = _interp_nan(ball_cx_raw)
    ball_cy_raw = _interp_nan(ball_cy_raw)

    portrait_mode = portrait_width is not None and portrait_width > 0
    if portrait_mode:
        crop_h = float(src_h)
        crop_w = float(int(round(src_h * 9.0 / 16.0)))
        crop_w = max(1.0, min(crop_w, float(src_w)))
    else:
        crop_w = float(portrait_width)
        desired_crop_h = (
            float(portrait_width) * 16.0 / 9.0 if portrait_width > 0 else float(frame_height)
        )
        crop_h = min(desired_crop_h, float(frame_height))
    half_crop_w = crop_w / 2.0

    finite_ball_x = ball_cx_raw[np.isfinite(ball_cx_raw)]
    finite_ball_y = ball_cy_raw[np.isfinite(ball_cy_raw)]
    if finite_ball_x.size > 0 and float(np.nanmax(finite_ball_x)) <= 1.0:
        ball_cx_raw = ball_cx_raw * src_w
    if finite_ball_y.size > 0 and float(np.nanmax(finite_ball_y)) <= 1.0:
        ball_cy_raw = ball_cy_raw * src_h

    ball_cx_list = [float(v) for v in ball_cx_raw.tolist()]
    ball_cy_list = [float(v) for v in ball_cy_raw.tolist()]

    log_params: dict[str, float | int] | None = None
    cam_cx: List[float] = []
    cam_cy: List[float] = []
    mode = cfg.get("ball_cam_mode", "strict_lock")
    override_crop_x: np.ndarray | None = None
    override_crop_y: np.ndarray | None = None

    if mode == "perfect_follow":
        # Optional smoothing to remove tiny jitter
        window = int(cfg.get("ball_cam_smooth_window", 5))
        if window > 1:
            k = np.ones(window, dtype=np.float32) / float(window)
            ball_cx = np.convolve(ball_cx_raw, k, mode="same")
        else:
            ball_cx = ball_cx_raw.copy()

        ball_cy = ball_cy_raw.copy()

        # Vertical location: use a single fixed target line (e.g. 55% of frame height)
        vpos = float(cfg.get("ball_cam_vertical_pos", 0.55))
        target_cy = vpos * src_h

        # For this test: follow only horizontally; keep vertical fixed
        cam_cx_arr = ball_cx.copy()
        cam_cy_arr = np.full_like(cam_cx_arr, target_cy)

        # Compute crop_x, crop_y for each frame
        half_w = crop_w / 2.0
        half_h = crop_h / 2.0

        crop_x = cam_cx_arr - half_w
        crop_y = cam_cy_arr - half_h

        # Clamp to source bounds
        crop_x = np.clip(crop_x, 0, src_w - crop_w)
        crop_y = np.clip(crop_y, 0, src_h - crop_h)

        override_crop_x = crop_x.astype(float)
        override_crop_y = crop_y.astype(float)

        cam_cx = (crop_x + half_w).tolist()
        cam_cy = (crop_y + half_h).tolist()

    elif mode == "raw_lock":
        cam_cx, cam_cy = compute_ball_lock_raw(
            ball_cx_raw=ball_cx_list,
            ball_cy_raw=ball_cy_list,
            src_w=src_w,
            src_h=src_h,
            crop_w=crop_w,
            crop_h=crop_h,
            cfg=cfg,
            logger=logger,
        )
    elif mode == "strict_lock":
        cam_cx, cam_cy = compute_ball_lock_strict(
            ball_cx_raw=ball_cx_list,
            ball_cy_raw=ball_cy_list,
            src_w=src_w,
            src_h=src_h,
            crop_w=crop_w,
            crop_h=crop_h,
            cfg=cfg,
            logger=logger,
        )
    else:
        lead_frames = int(cfg.get("ball_cam_lead_frames", 5))
        base_alpha = float(cfg.get("ball_cam_base_alpha", 0.30))
        fast_alpha = float(cfg.get("ball_cam_fast_alpha", 0.80))
        catchup_px = float(cfg.get("ball_cam_catchup_thresh_px", 40.0))
        margin = float(cfg.get("ball_cam_margin_px", 100.0))

        speed_fast_px = float(cfg.get("ball_cam_speed_fast_px", 40.0))
        max_pan_slow = float(cfg.get("ball_cam_max_pan_per_frame_slow", 24.0))
        max_pan_fast = float(cfg.get("ball_cam_max_pan_per_frame_fast", 44.0))

        log_params = {
            "lead_frames": lead_frames,
            "base_alpha": base_alpha,
            "fast_alpha": fast_alpha,
            "catchup_px": catchup_px,
            "margin": margin,
            "speed_fast_px": speed_fast_px,
            "max_pan_slow": max_pan_slow,
            "max_pan_fast": max_pan_fast,
        }

        cam_cx, cam_cy = compute_ema_ball_cam_path(
            ball_cx_raw=ball_cx_list,
            ball_cy_raw=ball_cy_list,
            src_w=src_w,
            src_h=src_h,
            crop_w=crop_w,
            crop_h=crop_h,
            cfg=cfg,
            logger=logger,
        )

    cam_cx_arr = np.asarray(cam_cx, dtype=float)
    cam_cy_arr = np.asarray(cam_cy, dtype=float)
    cam_zoom = np.ones(num_frames, dtype=float)

    crop_w_arr = np.full(num_frames, float(crop_w), dtype=float)
    crop_h_arr = np.full(num_frames, float(crop_h), dtype=float)

    if override_crop_x is not None and override_crop_y is not None:
        x0 = override_crop_x
        y0 = override_crop_y
    else:
        x0 = np.clip(cam_cx_arr - (crop_w_arr / 2.0), 0.0, float(frame_width) - crop_w_arr)
        y0 = np.clip(cam_cy_arr - (crop_h_arr / 2.0), 0.0, float(frame_height) - crop_h_arr)

    cam_cx_arr = x0 + (crop_w_arr / 2.0)
    cam_cy_arr = y0 + (crop_h_arr / 2.0)

    inside_count = 0
    inside_strict = 0
    margin = float(cfg.get("ball_cam_margin_px", 80.0))

    for i in range(num_frames):
        bx = ball_cx_raw[i]
        by = ball_cy_raw[i]

        cx = cam_cx[i]
        cy = cam_cy[i]

        half_w = crop_w / 2.0
        half_h = crop_h / 2.0

        # EXACT same clamp as render path
        crop_x = max(0.0, min(src_w - crop_w, cx - half_w))
        crop_y = max(0.0, min(src_h - crop_h, cy - half_h))
        eff_cx = crop_x + half_w
        eff_cy = crop_y + half_h

        if abs(bx - eff_cx) <= half_w and abs(by - eff_cy) <= half_h:
            inside_count += 1

        if abs(bx - eff_cx) <= (half_w - margin) and abs(by - eff_cy) <= (half_h - margin):
            inside_strict += 1

    coverage_pct = 100.0 * inside_count / max(1, num_frames)

    jerk95_raw = _jerk95(np.asarray(ball_cx_raw, dtype=float), fps=fps)
    jerk95_cam = _jerk95(cam_cx_arr, fps=fps)
    stats["jerk95_raw"] = jerk95_raw
    stats["jerk95_cam"] = jerk95_cam
    stats["jerk95"] = jerk95_cam  # backwards compatibility
    stats["ball_in_crop_pct"] = coverage_pct
    stats["ball_in_crop_frames"] = inside_count

    plan_data = {
        "x0": x0.astype(float),
        "y0": y0.astype(float),
        "w": crop_w_arr.astype(float),
        "h": crop_h_arr.astype(float),
        "spd": np.full(
            num_frames,
            float(max(1.0, np.max(np.abs(np.diff(cam_cx_arr, prepend=cam_cx_arr[0]))))),
        ),
        "z": cam_zoom.astype(float),
        "cx": cam_cx_arr.astype(float),
        "cy": cam_cy_arr.astype(float),
    }

    if log_params is not None:
        logger.info(
            "[BALL-CAM] N=%d, cx_range=[%.1f, %.1f], lead_frames=%d, "
            "base_alpha=%.2f, fast_alpha=%.2f, catchup_thresh_px=%.1f, margin=%.1f, "
            "speed_fast_px=%.1f, max_pan_slow=%.1f, max_pan_fast=%.1f, hard_lock=%s",
            len(cam_cx_arr),
            float(np.nanmin(cam_cx_arr)) if cam_cx_arr.size else 0.0,
            float(np.nanmax(cam_cx_arr)) if cam_cx_arr.size else 0.0,
            int(log_params.get("lead_frames", 0)),
            float(log_params.get("base_alpha", 0.0)),
            float(log_params.get("fast_alpha", 0.0)),
            float(log_params.get("catchup_px", 0.0)),
            float(log_params.get("margin", 0.0)),
            float(log_params.get("speed_fast_px", 0.0)),
            float(log_params.get("max_pan_slow", 0.0)),
            float(log_params.get("max_pan_fast", 0.0)),
            False,
        )
    logger.info(
        "[BALL-CAM COVERAGE] ball_in_crop: %.1f%% (%d/%d), strict_with_margin: %.1f%% (%d/%d)",
        coverage_pct,
        inside_count,
        num_frames,
        100.0 * inside_strict / max(1, num_frames),
        inside_strict,
        num_frames,
    )

    debug_overlay = bool(cfg.get("ball_debug_overlay", False))
    if debug_overlay and in_path is not None and out_path is not None:
        debug_out = str(out_path).replace(".mp4", "__BALL_DEBUG_WIDE.mp4")
        write_ball_crop_debug_clip(
            in_path=str(in_path),
            out_path=debug_out,
            ball_cx=ball_cx_list,
            ball_cy=ball_cy_list,
            crop_x=x0.tolist(),
            crop_y=y0.tolist(),
            src_w=int(src_w),
            src_h=int(src_h),
            crop_w=int(crop_w),
            crop_h=int(crop_h),
            fps=fps,
            logger=logger,
        )

    return plan_data, stats


class CamFollow2O:
    def __init__(
        self,
        zeta: float = 0.95,
        wn: float = 6.0,
        dt: float = 1 / 30,
        max_vel: Optional[float] = None,
        max_acc: Optional[float] = None,
        deadzone: float = 0.0,
    ) -> None:
        self.z = float(zeta)
        self.w = float(wn)
        self.dt = float(dt)
        self.cx = 0.0
        self.cy = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.max_vel = max_vel
        self.max_acc = max_acc
        self.dead = max(0.0, float(deadzone))

    def _clamp(self, vec: tuple[float, float], limit: Optional[float]) -> tuple[float, float]:
        if limit is None or limit <= 0:
            return vec
        vx, vy = vec
        mag = math.hypot(vx, vy)
        if mag <= limit:
            return vec
        scale = limit / max(mag, 1e-6)
        return vx * scale, vy * scale

    def step(self, target_x: float, target_y: float) -> tuple[float, float]:
        ex = target_x - self.cx
        ey = target_y - self.cy
        if math.hypot(ex, ey) < self.dead:
            ex = ey = 0.0

        dt = self.dt
        w = self.w
        z = self.z
        ax = w * w * ex - 2 * z * w * self.vx
        ay = w * w * ey - 2 * z * w * self.vy
        ax, ay = self._clamp((ax, ay), self.max_acc)
        self.vx += ax * dt
        self.vy += ay * dt
        self.vx, self.vy = self._clamp((self.vx, self.vy), self.max_vel)
        self.cx += self.vx * dt
        self.cy += self.vy * dt
        return self.cx, self.cy

    def damp_velocity(self, factor: float) -> None:
        factor = float(max(0.0, min(1.0, factor)))
        self.vx *= factor
        self.vy *= factor


class FollowHoldController:
    """Stateful helper that gates follow targets during dropouts."""

    def __init__(
        self,
        *,
        dt: float,
        release_frames: int = 3,
        decay_time: float = 0.4,
        initial_target: Optional[Tuple[float, float]] = None,
    ) -> None:
        self.release_frames = max(1, int(release_frames))
        self.valid_streak = self.release_frames
        self.decay_time = max(0.05, float(decay_time))
        self.decay_factor = float(
            math.exp(-dt / self.decay_time) if dt > 0 else 0.0
        )
        self._target = [0.0, 0.0]
        if initial_target is not None:
            self._target[0] = float(initial_target[0])
            self._target[1] = float(initial_target[1])
            self.valid_streak = self.release_frames
        self._initialised = initial_target is not None

    @property
    def target(self) -> Tuple[float, float]:
        return float(self._target[0]), float(self._target[1])

    def apply(
        self, target_x: float, target_y: float, valid: bool
    ) -> Tuple[float, float, bool]:
        if not math.isfinite(target_x) or not math.isfinite(target_y):
            valid = False
        if not self._initialised:
            self._target[0] = float(target_x)
            self._target[1] = float(target_y)
            self._initialised = True
        if valid:
            if self.valid_streak < self.release_frames:
                self.valid_streak += 1
            else:
                self.valid_streak = self.release_frames
            if self.valid_streak >= self.release_frames:
                self._target[0] = float(target_x)
                self._target[1] = float(target_y)
                return float(target_x), float(target_y), False
            return self.target[0], self.target[1], True
        self.valid_streak = 0
        return self.target[0], self.target[1], True

    def reset_target(self, cx: float, cy: float) -> None:
        self._target[0] = float(cx)
        self._target[1] = float(cy)
        self.valid_streak = self.release_frames
        self._initialised = True


def ema_path(xs: Sequence[float], ys: Sequence[float], alpha: float) -> tuple[List[float], List[float]]:
    if alpha <= 0:
        return list(xs), list(ys)
    sx: List[float] = []
    sy: List[float] = []
    for idx in range(len(xs)):
        x_val = float(xs[idx])
        y_val = float(ys[idx])
        if idx == 0:
            sx.append(x_val)
            sy.append(y_val)
            continue
        sx.append(alpha * x_val + (1 - alpha) * sx[-1])
        sy.append(alpha * y_val + (1 - alpha) * sy[-1])
    return sx, sy


# --- Jerk helpers --------------------------------------------------------


def compute_camera_jerk95(xs: Sequence[float], ys: Sequence[float], fps: float) -> float:
    """Return the 95th percentile jerk magnitude for a camera path."""

    if fps <= 0.0 or len(xs) < 4 or len(xs) != len(ys):
        return 0.0

    dt = 1.0 / float(fps)
    arr_x = np.asarray(xs, dtype=np.float64)
    arr_y = np.asarray(ys, dtype=np.float64)

    vx = np.gradient(arr_x, dt, edge_order=2)
    vy = np.gradient(arr_y, dt, edge_order=2)
    ax = np.gradient(vx, dt, edge_order=2)
    ay = np.gradient(vy, dt, edge_order=2)
    jx = np.gradient(ax, dt, edge_order=2)
    jy = np.gradient(ay, dt, edge_order=2)

    jerk_mag = np.hypot(jx, jy)
    jerk_mag = np.nan_to_num(jerk_mag, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.percentile(np.abs(jerk_mag), 95.0))


# --- Simple constant-velocity tracker (EMA-based Kalman-lite) ---

BallPathEntry = Union[Tuple[float, float, float], Tuple[float, float], Mapping[str, float], None]


class CV2DKalman:
    def __init__(self, bx, by):
        self.bx = float(bx)
        self.by = float(by)
        self.vx = 0.0
        self.vy = 0.0
        self.alpha_pos = 0.35
        self.alpha_vel = 0.25

    def predict(self):
        return self.bx + self.vx, self.by + self.vy

    def correct(self, mx, my):
        px, py = self.predict()
        rx, ry = (mx - px), (my - py)
        self.vx += self.alpha_vel * rx
        self.vy += self.alpha_vel * ry
        self.bx = (1 - self.alpha_pos) * px + self.alpha_pos * mx
        self.by = (1 - self.alpha_pos) * py + self.alpha_pos * my
        return self.bx, self.by


# --- Color/shape gating to remove grass and favor white-ish round blobs ---


def pick_ball(d, src_w, src_h):
    bx, by = _get_ball_xy_src(d, src_w, src_h)
    if bx is not None and by is not None:
        return bx, by
    if "ball" in d:
        ball_val = d["ball"]
    return None, None


def _get_ball_xy_src(
    rec: Optional[Union[Mapping[str, object], Sequence[object]]],
    src_w: float,
    src_h: float,
) -> tuple[Optional[float], Optional[float]]:
    """Return best-effort (x, y) ball coordinates in source pixels."""

    if rec is None:
        return None, None

    def _pair_from_mapping(
        mapping: Mapping[str, object],
        key_x: str,
        key_y: str,
    ) -> Optional[tuple[float, float]]:
        if key_x not in mapping or key_y not in mapping:
            return None
        val_x = mapping.get(key_x)
        val_y = mapping.get(key_y)

    def _pair_from_sequence(seq: Sequence[object]) -> Optional[tuple[float, float]]:
        if len(seq) < 2:
            return None

    def _to_src(pair: tuple[float, float]) -> tuple[float, float]:
        x, y = pair
        if src_w > 1 and src_h > 1 and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            return x * float(src_w), y * float(src_h)
        return x, y

    if isinstance(rec, Mapping):
        for key_x, key_y in ("bx_src", "by_src"), ("bx_raw", "by_raw"):
            pair = _pair_from_mapping(rec, key_x, key_y)
            if pair is not None:
                return _to_src(pair)

        ball_seq = rec.get("ball") if isinstance(rec, Mapping) else None
        if isinstance(ball_seq, Sequence):
            pair = _pair_from_sequence(ball_seq)
            if pair is not None:
                return _to_src(pair)

        for key_x, key_y in ("bx_stab", "by_stab"), ("bx", "by"):
            pair = _pair_from_mapping(rec, key_x, key_y)
            if pair is not None:
                return _to_src(pair)

    if isinstance(rec, Sequence) and not isinstance(rec, (str, bytes, bytearray)):
        pair = _pair_from_sequence(rec)
        if pair is not None:
            return _to_src(pair)

    return None, None


def build_ball_mask(bgr, grass_h=(35, 95), min_v=170, max_s=120):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    non_grass = (H < grass_h[0]) | (H > grass_h[1])
    bright = V >= min_v
    low_sat = S <= max_s
    mask = ((non_grass & bright) | (bright & low_sat)).astype(np.uint8) * 255
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask


def _circularity(cnt):
    a = cv2.contourArea(cnt)
    p = cv2.arcLength(cnt, True)
    if p <= 0 or a <= 0:
        return 0.0
    return float(4 * math.pi * a / (p * p))


def ncc_score(gray_win, tpl_gray):
    if (
        gray_win.shape[0] < tpl_gray.shape[0]
        or gray_win.shape[1] < tpl_gray.shape[1]
    ):
        return -1.0
    g = cv2.equalizeHist(gray_win)
    t = cv2.equalizeHist(tpl_gray)
    r1 = cv2.matchTemplate(g, t, cv2.TM_CCOEFF_NORMED).max()
    h, w = t.shape[:2]
    tw, th = max(3, int(w * 0.75)), max(3, int(h * 0.75))
    t2 = cv2.resize(t, (tw, th), interpolation=cv2.INTER_AREA)
    r2 = cv2.matchTemplate(g, t2, cv2.TM_CCOEFF_NORMED).max()
    return float(max(r1, r2))


def find_ball_candidate(
    frame_bgr,
    pred_xy,
    tpl=None,
    search_r=260,
    min_r=6,
    max_r=22,
    min_circ=0.58,
):
    H, W = frame_bgr.shape[:2]
    px, py = pred_xy
    x0 = int(max(0, px - search_r))
    y0 = int(max(0, py - search_r))
    x1 = int(min(W, px + search_r))
    y1 = int(min(H, py + search_r))
    if x1 <= x0 + 2 or y1 <= y0 + 2:
        return None
    roi = frame_bgr[y0:y1, x0:x1]
    mask = build_ball_mask(roi)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    best = None
    best_score = -1e9

    for c in cnts:
        a = cv2.contourArea(c)
        if a < (min_r * min_r * 0.6) or a > (max_r * max_r * 3.5):
            continue
        circ = _circularity(c)
        if circ < min_circ:
            continue
        (cx, cy), rad = cv2.minEnclosingCircle(c)
        bx = x0 + cx
        by = y0 + cy
        dist = math.hypot(bx - px, by - py)
        ncc = 0.0
        if tpl is not None:
            side = int(max(16, min(96, rad * 6)))
            sx0 = int(max(0, bx - side // 2))
            sy0 = int(max(0, by - side // 2))
            sx1 = int(min(W, sx0 + side))
            sy1 = int(min(H, sy0 + side))
            win = gray_roi[(sy0 - y0) : (sy1 - y0), (sx0 - x0) : (sx1 - x0)]
            if win.size:
                ncc = ncc_score(win, tpl)
        score = (-0.02 * dist) + (3.0 * circ) + (1.8 * ncc)
        if score > best_score:
            best_score = score
            best = (bx, by, float(circ), float(ncc), float(dist))

    return best


def grab_frame_at_time(path, t_sec, fps_hint=30.0):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, None
    fps = cap.get(cv2.CAP_PROP_FPS) or fps_hint
    idx = max(0, int(round(t_sec * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    return (frame if ok else None), fps


def manual_select_ball(frame_bgr, window="Select ball"):
    # Fallback if selectROI is missing: simple click-to-center
    if not hasattr(cv2, "selectROI"):
        clicked = {"pt": None}

        def _cb(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                clicked["pt"] = (x, y)

        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.imshow(window, frame_bgr)
        cv2.setMouseCallback(window, _cb)
        print("Click the ball; press ENTER to confirm.")
        while True:
            k = cv2.waitKey(20) & 0xFF
            if k in (13, 32):
                break
            if k == 27:
                clicked["pt"] = None
                break
        cv2.destroyWindow(window)
        if clicked["pt"] is None:
            return None
        x, y = clicked["pt"]
        side = 56
        return (int(x - side // 2), int(y - side // 2), side, side)

    # Preferred: drag a rectangle
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    r = cv2.selectROI(window, frame_bgr, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window)
    if r is None or len(r) != 4 or r[2] <= 0 or r[3] <= 0:
        return None
    return tuple(int(v) for v in r)  # (x, y, w, h)


def _clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def _clamp_roi(x, y, w, h, W, H):
    x = _round_i(x)
    y = _round_i(y)
    w = _round_i(w)
    h = _round_i(h)
    x = _clamp(x, 0, max(0, W - 1))
    y = _clamp(y, 0, max(0, H - 1))
    w = _clamp(w, 2, max(2, W - x))
    h = _clamp(h, 2, max(2, H - y))
    return x, y, w, h


def _roi_around_point(bx, by, W, H, side):
    # side may be float â†' force int and keep odd size for better centering
    side_i = max(3, _round_i(side) | 1)
    x = _round_i(bx) - side_i // 2
    y = _round_i(by) - side_i // 2
    return _clamp_roi(x, y, side_i, side_i, W, H)


class ZoomPlanner:
    """
    Speed â†' zoom with hysteresis and slew-rate limiting.
    zoom=1.0 means the crop is exactly target height; >1 zooms in.
    """

    def __init__(
        self,
        z_min=1.0,
        z_max=1.8,
        s_lo=3.0,
        s_hi=18.0,
        hysteresis=0.20,
        max_step=0.03,
    ):
        self.z_min, self.z_max = z_min, z_max
        self.s_lo, self.s_hi = s_lo, s_hi
        self.hysteresis = hysteresis
        self.max_step = max_step

    def plan(self, spd):
        # map speed to raw zoom target
        t = np.clip((spd - self.s_lo) / (self.s_hi - self.s_lo), 0, 1)
        raw = self.z_min + t * (self.z_max - self.z_min)

        # apply hysteresis around last zoom to avoid flicker
        out = np.empty_like(raw)
        z = raw[0]
        out[0] = z
        for i in range(1, len(raw)):
            rz = raw[i]
            band = self.hysteresis * (self.z_max - self.z_min)
            if abs(rz - z) <= band:
                rz = z  # stick
            # limit zoom change per frame
            z += np.clip(rz - z, -self.max_step, self.max_step)
            out[i] = z
        # final mild smoothing
        return smooth_series(out, alpha=0.25)


FPS = 30.0


def plan_camera_from_ball(
    bx,
    by,
    frame_w,
    frame_h,
    target_aspect,
    pan_alpha=0.15,
    lead=0.10,
    bounds_pad=16,
    center_frac=0.5,
    *,
    margin_px_override: Optional[float] = None,
    headroom_frac_override: Optional[float] = None,
    lead_px_override: Optional[float] = None,
):
    """Plan a portrait crop using the offline cinematic planner."""

    if frame_w <= 0 or frame_h <= 0:
        raise ValueError("Frame dimensions must be positive")

    fps = FPS if FPS > 0 else 30.0
    smooth_window = max(3, int(round((1.0 - float(np.clip(pan_alpha, 0.01, 0.95))) * 12.0)) | 1)
    headroom_frac = 0.5 - float(np.clip(center_frac, 0.0, 1.0))
    default_headroom = max(0.08, min(0.20, headroom_frac))
    lead_px = max(frame_w * 0.05, float(lead) * fps * 40.0)
    max_step_x = max(12.0, frame_w * 0.012)
    max_step_y = max(8.0, frame_h * 0.008)
    passes = 3 if pan_alpha < 0.3 else 2

    margin_value = max(bounds_pad, 90.0)

    cfg = PlannerConfig(
        frame_size=(float(frame_w), float(frame_h)),
        crop_aspect=float(target_aspect) if target_aspect > 0 else (9.0 / 16.0),
        fps=float(fps),
        margin_px=float(margin_value),
        headroom_frac=float(default_headroom),
        lead_px=float(lead_px),
        smooth_window=int(smooth_window),
        max_step_x=float(max_step_x),
        max_step_y=float(max_step_y),
        accel_limit_x=float(max_step_x * 0.35),
        accel_limit_y=float(max_step_y * 0.35),
        smoothing_passes=passes,
        portrait_pad=float(bounds_pad),
    )

    planner = OfflinePortraitPlanner(cfg)
    plan = planner.plan(bx, by)

    x0 = plan["x0"].round().astype(int)
    y0 = plan["y0"].round().astype(int)
    w = plan["w"].round().astype(int)
    h = plan["h"].round().astype(int)
    spd = plan["spd"].astype(float)
    z = plan["z"].astype(float)
    return x0, y0, w, h, spd, z


def guarantee_ball_in_crop(
    x0,
    y0,
    cw,
    ch,
    bx,
    by,
    src_w,
    src_h,
    zoom,
    zoom_min,
    zoom_max,
    margin=0.12,
    step_zoom=0.90,
):
    """Adjust the crop so the ball remains inside with a configurable margin."""

    inner_l = x0 + margin * cw
    inner_r = x0 + cw - margin * cw
    inner_t = y0 + margin * ch
    inner_b = y0 + ch - margin * ch

    dx = 0.0
    dy = 0.0
    if bx < inner_l:
        dx = bx - inner_l
    elif bx > inner_r:
        dx = bx - inner_r
    if by < inner_t:
        dy = by - inner_t
    elif by > inner_b:
        dy = by - inner_b

    if dx or dy:
        x0 += dx
        y0 += dy
        x0 = max(0.0, min(x0, src_w - cw))
        y0 = max(0.0, min(y0, src_h - ch))
        inner_l = x0 + margin * cw
        inner_r = x0 + cw - margin * cw
        inner_t = y0 + margin * ch
        inner_b = y0 + ch - margin * ch

    tries = 0
    while (
        (bx < inner_l or bx > inner_r or by < inner_t or by > inner_b)
        and tries < 12
    ):
        new_zoom = max(zoom_min, zoom * step_zoom)
        if abs(new_zoom - zoom) < 1e-6 and tries > 0:
            break
        zoom = new_zoom
        cx = x0 + cw / 2.0
        cy = y0 + ch / 2.0
        cx = 0.7 * bx + 0.3 * cx
        cy = 0.7 * by + 0.3 * cy

        aspect = cw / float(ch) if ch > 0 else 1080.0 / 1920.0
        ch = src_h / float(zoom) if zoom else src_h
        cw = ch * aspect
        if cw > src_w:
            cw = float(src_w)
            ch = cw / aspect if aspect else src_h

        x0 = max(0.0, min(cx - cw / 2.0, src_w - cw))
        y0 = max(0.0, min(cy - ch / 2.0, src_h - ch))

        inner_l = x0 + margin * cw
        inner_r = x0 + cw - margin * cw
        inner_t = y0 + margin * ch
        inner_b = y0 + ch - margin * ch
        tries += 1

    return x0, y0, cw, ch, zoom


import yaml


def to_jsonable(obj):
    """Recursively convert numpy/Path/datetime objects into JSON-serialisable types."""

    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(np, "generic") and isinstance(obj, np.generic):
        return obj.item()
    if hasattr(np, "ndarray") and isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (float, int, bool, str)) or obj is None:
        return obj


def plan_crop_from_ball(
    bx,
    by,
    src_w,
    src_h,
    out_w=1080,
    out_h=1920,
    zoom_min=0.55,
    zoom_max=0.95,
    pad=24,
    state=None,
    center_frac: float = 0.5,
):
    """Return integer (x0,y0,w,h) portrait crop centered on (bx,by) with damped smoothing."""

    if state is None:
        state = {}

    if src_w <= 0 or src_h <= 0:
        return 0, 0, src_w, src_h, state

    if out_w > 0 and out_h > 0:
        portrait_w = float(out_w)
        portrait_h = float(out_h)
    else:
        portrait_w = float(src_w)
        portrait_h = float(src_h)
    if portrait_w <= 0:
        portrait_w = float(src_w)
    if portrait_h <= 0:
        portrait_h = float(src_h)

    if portrait_h <= 0:
        portrait_h = 1.0
    aspect = portrait_w / float(portrait_h) if portrait_h else 1.0
    if aspect <= 0:
        aspect = float(src_w) / float(src_h) if src_h else 1.0


    candidates = [zoom_value]
    zoom_value = next((c for c in reversed(candidates) if c and abs(c) > 1e-6), 1.0)

    if zoom_min > 0 and zoom_max > 0 and zoom_max >= zoom_min:
        zoom_value = max(zoom_min, min(zoom_max, zoom_value))
    elif zoom_min > 0 and (zoom_max <= 0 or zoom_max < zoom_min):
        zoom_value = max(zoom_min, zoom_value)
    elif zoom_max > 0:
        zoom_value = min(zoom_max, zoom_value)
    if abs(zoom_value) <= 1e-6:
        zoom_value = 1.0

    pad_px = 0.0
    pad_frac = 0.0

    shrink = 1.0
    if pad_frac > 0.0:
        shrink = max(0.0, 1.0 - 2.0 * pad_frac)

    max_crop_w = max(1.0, float(src_w) - 2.0 * pad_px)
    max_crop_h = max(1.0, float(src_h) - 2.0 * pad_px)

    max_zoom_candidates = [zoom_value]
    max_zoom = max(max_zoom_candidates) if max_zoom_candidates else 1.0
    if max_zoom <= 1e-6:
        max_zoom = 1.0

    min_crop_h = portrait_h / max_zoom if max_zoom else portrait_h
    if pad_frac > 0.0:
        min_crop_h *= shrink
    min_crop_h = max(1.0, min(min_crop_h, max_crop_h))
    min_crop_w = max(1.0, min(min_crop_h * aspect, max_crop_w))
    if min_crop_w < min_crop_h * aspect:
        min_crop_h = min_crop_w / max(aspect, 1e-6)

    target_crop_h = portrait_h / zoom_value
    if pad_frac > 0.0:
        target_crop_h *= shrink
    target_crop_h = max(min_crop_h, min(target_crop_h, max_crop_h))
    target_crop_w = target_crop_h * aspect
    if target_crop_w > max_crop_w:
        scale = max_crop_w / target_crop_w if target_crop_w > 0 else 1.0
        target_crop_w = max_crop_w
        target_crop_h *= scale
    if target_crop_h < min_crop_h:
        target_crop_h = min_crop_h
        target_crop_w = min(target_crop_h * aspect, max_crop_w)
        if target_crop_w <= 0:
            target_crop_w = min_crop_w
        if target_crop_w > max_crop_w:
            target_crop_w = max_crop_w
            target_crop_h = target_crop_w / max(aspect, 1e-6)

    if prev_w <= 0:
        prev_w = float(target_crop_w)
    if prev_h <= 0:
        prev_h = float(target_crop_h)


    prev_cx = prev_x + prev_w / 2.0
    prev_cy = prev_y + prev_h / 2.0

    bx = float(bx)
    by = float(by)

    alpha_center = 0.20
    alpha_zoom = 0.15

    cx = (1 - alpha_center) * prev_cx + alpha_center * bx
    cy = (1 - alpha_center) * prev_cy + alpha_center * by
    cy = cy + (0.5 - float(np.clip(center_frac, 0.0, 1.0))) * src_h

    cw = (1 - alpha_zoom) * prev_w + alpha_zoom * target_crop_w
    ch = (1 - alpha_zoom) * prev_h + alpha_zoom * target_crop_h

    ch = max(min_crop_h, min(ch, max_crop_h))
    cw = ch * aspect
    if cw > max_crop_w:
        scale = max_crop_w / cw if cw > 0 else 1.0
        cw = max_crop_w
        ch *= scale
    min_crop_w = min_crop_h * aspect
    if cw < min_crop_w:
        cw = min_crop_w
        ch = cw / max(aspect, 1e-6)

    half_w = cw / 2.0
    half_h = ch / 2.0
    min_x = float(pad_px)
    max_x = float(src_w) - float(pad_px) - cw
    min_y = float(pad_px)
    max_y = float(src_h) - float(pad_px) - ch

    x0 = cx - half_w
    y0 = cy - half_h

    if max_x < min_x:
        x0 = float(src_w) / 2.0 - half_w
    else:
        x0 = max(min_x, min(max_x, x0))
    if max_y < min_y:
        y0 = float(src_h) / 2.0 - half_h
    else:
        y0 = max(min_y, min(max_y, y0))

    cx = x0 + half_w
    cy = y0 + half_h

    zoom_actual = float(src_h) / ch if ch else zoom_value
    if zoom_min > 0 and zoom_max > 0 and zoom_max >= zoom_min:
        zoom_actual = max(zoom_min, min(zoom_max, zoom_actual))

    state["x"] = float(x0)
    state["y"] = float(y0)
    state["w"] = float(cw)
    state["h"] = float(ch)
    state["cx"] = float(cx)
    state["cy"] = float(cy)
    state["zoom"] = float(zoom_actual)

    x0_int = int(round(x0))
    y0_int = int(round(y0))
    w_int = max(1, int(round(cw)))
    h_int = max(1, int(round(ch)))

    return x0_int, y0_int, w_int, h_int, state


def compute_portrait_crop(cx, cy, zoom, src_w, src_h, target_aspect, pad):
    # target aspect (w/h)
    if target_aspect and target_aspect > 0:
        t_aspect = float(target_aspect)
    else:
        t_aspect = src_w / float(src_h)

    # derive crop size from zoom while honoring aspect
    crop_h = src_h / float(zoom)
    crop_w = crop_h * t_aspect
    if crop_w > src_w:  # bound if too wide
        crop_w = float(src_w)
        crop_h = crop_w / t_aspect if t_aspect else crop_h

    # pad shrinks the box a bit to keep safety margins around ball
    if pad and pad > 0:
        crop_w *= (1.0 - 2 * pad)
        crop_h *= (1.0 - 2 * pad)

    # clamp center so the crop stays inside the source
    x0 = max(0.0, min(cx - crop_w / 2.0, src_w - crop_w))
    y0 = max(0.0, min(cy - crop_h / 2.0, src_h - crop_h))

    return x0, y0, crop_w, crop_h


def dynamic_zoom(
    prev_zoom,
    bx,
    by,
    x0,
    y0,
    cw,
    ch,
    src_w,
    src_h,
    speed_px,
    target_zoom_min,
    target_zoom_max,
    k_speed_out=0.0006,
    edge_margin=0.14,
    edge_gain=0.08,
    z_rate=0.06,
):
    """Return a smoothed zoom that reacts to ball speed and proximity to edges."""

    z_target = max(
        target_zoom_min,
        min(target_zoom_max, target_zoom_max - k_speed_out * speed_px),
    )

    if cw > 1 and ch > 1:
        dl = (bx - x0) / cw
        dr = (x0 + cw - bx) / cw
        dt = (by - y0) / ch
        db = (y0 + ch - by) / ch
        proximity = max(0.0, edge_margin - min(dl, dr, dt, db)) / max(edge_margin, 1e-6)
        z_target = max(target_zoom_min, z_target - edge_gain * proximity)

    z_next = prev_zoom + (z_target - prev_zoom) * 0.20
    z_next = max(prev_zoom - z_rate, min(prev_zoom + z_rate, z_next))
    return z_next


PRESETS_PATH = Path(__file__).resolve().parent / "render_presets.yaml"
FOLLOW_DEFAULTS = {
    "zeta": 1.10,
    "wn": 3.5,
    "deadzone": 8.0,
    "max_vel": 250.0,
    "max_acc": 1200.0,
    "pre_smooth": 0.35,
    "lookahead": 2,
}

DEFAULT_PRESETS = {
    "cinematic": {
        "fps": 30,
        "portrait": "1080x1920",
        "lookahead": 20,
        "smoothing": 0.30,
        "pad": 0.12,
        "speed_limit": 1400,
        "zoom_min": 1.0,
        "zoom_max": 1.8,
        "crf": 19,
        "keyint_factor": 4,
    },
    "wide_follow": {
        "fps": 24,
        "portrait": "1080x1920",
        "lookahead": 20,
        "smoothing": 0.30,
        "pad": 0.02,
        "speed_limit": 1400,
        "zoom_min": 1.0,
        "zoom_max": 1.25,
        "crf": 19,
        "keyint_factor": 4,
        "follow": {
            "smoothing": 0.30,
            "lead_time": 0.10,
            "margin_px": 140,
            "zoom_out_max": 1.25,
            "zoom_edge_frac": 0.9,
            "speed_zoom": {
                "enabled": True,
                "v_lo": 2.0,
                "v_hi": 10.0,
                "zoom_lo": 1.0,
                "zoom_hi": 0.90,
            },
            "controller": {
                "zeta": 1.10,
                "wn": 2.20,
                "deadzone": 40,
                "max_vel": 220,
                "max_acc": 2200,
                "pre_smooth": 0.45,
                "lookahead": 4,
            },
        },
    },
    "gentle": {
        "fps": 30,
        "portrait": "1080x1920",
        "lookahead": 12,
        "smoothing": 0.55,
        "pad": 0.20,
        "speed_limit": 360,
        "zoom_min": 1.0,
        "zoom_max": 1.8,
        "crf": 20,
        "keyint_factor": 4,
    },
    "realzoom": {
        "fps": 30,
        "portrait": "1080x1920",
        "lookahead": 10,
        "smoothing": 0.50,
        "pad": 0.18,
        "speed_limit": 520,
        "zoom_min": 1.0,
        "zoom_max": 2.4,
        "crf": 19,
        "keyint_factor": 4,
    },
}


def ensure_presets_file() -> None:
    """Create the presets file with defaults when missing."""

    if PRESETS_PATH.exists():
        return
    PRESETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PRESETS_PATH.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(DEFAULT_PRESETS, handle, sort_keys=True)


def load_presets() -> dict:
    """Load the preset configuration, creating defaults if required."""

    ensure_presets_file()
    with PRESETS_PATH.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def ffprobe_fps(path):
    """
    Return the FPS of the input video using ffprobe.
    Always returns a float. Errors bubble upward for visibility.
    """
    import subprocess
    import json

    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "json", path
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # If ffprobe fails, raise an informative error
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    data = json.loads(result.stdout)

    if "streams" not in data or not data["streams"]:
        raise ValueError(f"No video stream found in {path}")

    rate = data["streams"][0].get("r_frame_rate", "0/0")

    # Convert "30000/1001" → float
    num, den = rate.split("/")
    num = float(num)
    den = float(den)
    if den == 0:
        raise ValueError(f"Invalid r_frame_rate in ffprobe: {rate}")

    return num / den


def ffprobe_duration(path: Path) -> float:
    """Return the media duration in seconds using ffprobe."""

    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]

    value = result.stdout.strip()
    if not value:
        raise RuntimeError("ffprobe did not return a duration value.")



def parse_portrait(value: Optional[str]) -> Optional[Tuple[int, int]]:
    """Convert a WxH string into integers."""

    if not value:
        return None
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Invalid portrait specification: {value}")
    width = int(parts[0])
    height = int(parts[1])
    if width <= 0 or height <= 0:
        raise ValueError("Portrait dimensions must be positive integers.")
    return width, height


def portrait_config_from_preset(
    value: Optional[Union[str, Mapping[str, object], Sequence[object]]]
) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[float, float]], float]:
    """Extract portrait size, minimum crop, and horizon lock from a preset entry."""

    portrait: Optional[Tuple[int, int]] = None
    min_box: Optional[Tuple[float, float]] = None
    horizon_lock = 0.0

    def _parse_size(size_value: object) -> Optional[Tuple[int, int]]:
        if size_value is None:
            return None
        if isinstance(size_value, str):
            return parse_portrait(size_value)
        if isinstance(size_value, Mapping):
            width = size_value.get("width")
            height = size_value.get("height")
            if width is None or height is None:
                return None
            if w > 0 and h > 0:
                return w, h
            return None
        if isinstance(size_value, Sequence):
            seq = list(size_value)
            if len(seq) < 2:
                return None
            if w > 0 and h > 0:
                return w, h
        return None

    if value is None:
        return None, None, 0.0

    if isinstance(value, Mapping):
        size_value = value.get("size")
        if size_value is None:
            size_value = value.get("canvas") or value.get("dimensions")
        portrait = _parse_size(size_value)
        if portrait is None and "width" in value and "height" in value:
            portrait = _parse_size({"width": value.get("width"), "height": value.get("height")})

        min_box_value = value.get("min_box_px") or value.get("min_box")
        if isinstance(min_box_value, Mapping):
            mbw = min_box_value.get("width")
            mbh = min_box_value.get("height")
            if mbw_f is not None and mbh_f is not None and mbw_f > 0 and mbh_f > 0:
                min_box = (mbw_f, mbh_f)
        elif isinstance(min_box_value, Sequence):
            seq = list(min_box_value)
    
        horizon_value = value.get("horizon_lock")

        if portrait is None:
            inline = value.get("size")
            if isinstance(inline, str):
                portrait = parse_portrait(inline)
    else:
        portrait = _parse_size(value)

    return portrait, min_box, horizon_lock


def find_label_files(stem: str, labels_root: str) -> List[Path]:
    root = Path(labels_root or "out/yolo").expanduser()
    # Match ANY depth .../labels/<stem>_*.txt
    return sorted(Path(p) for p in glob.glob(str(root / "**" / "labels" / f"{stem}_*.txt"), recursive=True))



def _detect_normalized(x: float, y: float, width: int, height: int) -> bool:
    """Return ``True`` when coordinates appear to be normalised."""

    return (0.0 <= x <= 1.0) and (0.0 <= y <= 1.0) and (width > 2 and height > 2)


def load_labels(
    paths: Sequence[Path],
    frame_width: int,
    frame_height: int,
    input_fps: float,
) -> List[Tuple[float, float, float]]:
    """Load label shards and return time-stamped positions in pixel space."""

    pts: List[Tuple[float, float, float]] = []
    fps = float(input_fps) if input_fps else 30.0

    for file_path in paths:
        with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = stripped.replace(",", " ").split()
                if len(parts) < 3:
                    continue

                if _detect_normalized(x_value, y_value, frame_width, frame_height):
                    x_value *= float(frame_width)
                    y_value *= float(frame_height)

                t_value = frame_idx / fps if fps else 0.0
                pts.append((t_value, x_value, y_value))

    pts.sort(key=lambda record: record[0])
    if not pts:
        return []

    import statistics

    dx = [pts[i + 1][1] - pts[i][1] for i in range(len(pts) - 1)]
    dy = [pts[i + 1][2] - pts[i][2] for i in range(len(pts) - 1)]

    def _trim(values: List[float]) -> set[int]:
        if len(values) < 8:
            return set()
        mean_value = statistics.mean(values)
        stdev_value = statistics.pstdev(values) or 1.0
        bad_indices: set[int] = set()
        for idx, value in enumerate(values):
            if abs((value - mean_value) / stdev_value) > 3.0:
                bad_indices.add(idx)
                bad_indices.add(idx + 1)
        return bad_indices

    bad_idx = _trim(dx) | _trim(dy)
    filtered = [record for idx, record in enumerate(pts) if idx not in bad_idx]
    return filtered


def resample_labels_by_time(
    label_pts: Sequence[Tuple[float, float, float]],
    render_fps: float,
    duration_s: float,
) -> List[Tuple[float, float, float]]:
    """Return per-frame (t, x, y) aligned to render frames by time."""

    if not label_pts:
        return []

    import bisect

    ts = [point[0] for point in label_pts]
    xs = [point[1] for point in label_pts]
    ys = [point[2] for point in label_pts]

    out: List[Tuple[float, float, float]] = []
    total_frames = int(round(max(duration_s, 0.0) * float(render_fps)))
    for frame_idx in range(total_frames):
        t_value = frame_idx / float(render_fps) if render_fps else 0.0
        pos = bisect.bisect_left(ts, t_value)
        if pos <= 0:
            x_value, y_value = xs[0], ys[0]
        elif pos >= len(ts):
            x_value, y_value = xs[-1], ys[-1]
        else:
            t0, t1 = ts[pos - 1], ts[pos]
            weight = 0.0 if t1 == t0 else (t_value - t0) / (t1 - t0)
            x_value = xs[pos - 1] * (1.0 - weight) + xs[pos] * weight
            y_value = ys[pos - 1] * (1.0 - weight) + ys[pos] * weight
        out.append((t_value, x_value, y_value))
    return out


def labels_to_positions(
    label_pts: Sequence[Tuple[float, float, float]],
    render_fps: float,
    duration_s: float,
    source_pts: Optional[Sequence[Tuple[float, float, float]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert per-frame label points into arrays for planning."""

    total_frames = int(round(max(duration_s, 0.0) * float(render_fps)))
    if total_frames <= 0:
        empty_positions = np.empty((0, 2), dtype=np.float32)
        empty_used = np.zeros(0, dtype=bool)
        return empty_positions, empty_used

    if not label_pts:
        positions = np.full((total_frames, 2), np.nan, dtype=np.float32)
        used = np.zeros(total_frames, dtype=bool)
        return positions, used

    resampled = list(label_pts)
    if len(resampled) > total_frames:
        resampled = resampled[:total_frames]
    while len(resampled) < total_frames and resampled:
        t_value = len(resampled) / float(render_fps) if render_fps else 0.0
        resampled.append((t_value, resampled[-1][1], resampled[-1][2]))

    positions = np.array([[x, y] for _, x, y in resampled], dtype=np.float32)

    reference = source_pts if source_pts is not None else resampled
    times = [point[0] for point in reference]
    import bisect

    used = np.zeros(len(resampled), dtype=bool)
    if times:
        threshold = 1.5 / float(render_fps) if render_fps else 0.0
        for idx, (t_value, _, _) in enumerate(resampled):
            insert_pos = bisect.bisect_left(times, t_value)
            best = float("inf")
            if insert_pos < len(times):
                best = min(best, abs(times[insert_pos] - t_value))
            if insert_pos > 0:
                best = min(best, abs(times[insert_pos - 1] - t_value))
            if best <= threshold:
                used[idx] = True

    return positions, used


def _positions_range(positions: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    """Return ``(min_x, max_x, min_y, max_y)`` for valid position samples."""

    if positions.size == 0:
        return None
    valid_mask = ~np.isnan(positions).any(axis=1)
    if not np.any(valid_mask):
        return None
    xs = positions[valid_mask, 0]
    ys = positions[valid_mask, 1]
    if xs.size == 0 or ys.size == 0:
        return None
    return (
        float(np.min(xs)),
        float(np.max(xs)),
        float(np.min(ys)),
        float(np.max(ys)),
    )


def rough_motion_path(
    video_path: str, fps: float, duration_s: float, sample_every: int = 2
) -> List[Tuple[float, float, float]]:
    """Estimate a coarse (t, x, y) path from optical flow as a labels fallback."""

    if fps <= 0.0 or duration_s <= 0.0:
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total = int(round(duration_s * fps))
    ok, prev = cap.read()
    if not ok:
        cap.release()
        return []

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    height, width = prev_gray.shape[:2]
    centers: List[Tuple[float, float, float]] = []
    cx, cy = width / 2.0, height / 2.0
    frame_idx = 1

    while len(centers) < total:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % max(1, sample_every) != 0:
            frame_idx += 1
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, 0.5, 2, 15, 3, 5, 1.2, 0
        )
        fx = float(np.median(flow[..., 0]))
        fy = float(np.median(flow[..., 1]))
        cx = float(np.clip(cx + fx, 0, width - 1))
        cy = float(np.clip(cy + fy, 0, height - 1))
        centers.append((len(centers) / float(fps), cx, cy))
        prev_gray = gray
        frame_idx += 1

    cap.release()

    if not centers:
        return []

    times = [t for t, _, _ in centers]
    xs = [x for _, x, _ in centers]
    ys = [y for _, _, y in centers]

    out: List[Tuple[float, float, float]] = []
    for frame in range(total):
        t_value = frame / float(fps)
        pos = np.searchsorted(times, t_value)
        if pos <= 0:
            x_value, y_value = xs[0], ys[0]
        elif pos >= len(times):
            x_value, y_value = xs[-1], ys[-1]
        else:
            t0, t1 = times[pos - 1], times[pos]
            weight = 0.0 if t1 == t0 else (t_value - t0) / (t1 - t0)
            x_value = xs[pos - 1] * (1.0 - weight) + xs[pos] * weight
            y_value = ys[pos - 1] * (1.0 - weight) + ys[pos] * weight
        out.append((t_value, float(x_value), float(y_value)))

    return out


def build_ball_keepinview_path(
    telemetry: list[dict],
    frame_width: int,
    frame_height: int,
    crop_width: int,
    crop_height: int,
    *,
    default_y_frac: float = 0.45,
    margin_frac: float = 0.15,
    smooth_radius: int = 5,
    max_speed_px: float = 80.0,
) -> tuple[list[float], list[float]]:
    """
    Given per-frame ball telemetry, return (center_x, center_y) for each frame
    such that the ball stays inside a crop window of size crop_width x crop_height.

    - `telemetry` is a list of dicts with keys: t, x, y, visible.
    - center_x/center_y must have length == len(telemetry).
    - All returned centers must be finite floats.

    Strategy:
    - For each frame i:
      - If visible and x/y are finite, use that ball point.
      - Else, carry forward the last valid ball center if we have one.
      - If we still don't have any valid ball yet (start of clip),
        use a neutral default center:
          cx = frame_width / 2
          cy = frame_height * default_y_frac
    - After we build raw center_x[], center_y[]:
      - Clamp centers so that the crop window stays fully inside the frame:
        left = cx - crop_width / 2, right = cx + crop_width / 2
        top  = cy - crop_height / 2, bottom = cy + crop_height / 2
        Adjust cx, cy if those bounds go outside [0, frame_width/height].
      - Apply a simple moving-average smoothing (radius = smooth_radius) to
        center_x and center_y to remove jitter.
      - Optionally clamp per-frame motion to max_speed_px in each direction.
    - Return the smoothed center_x, center_y.
    """

    import math

    n_frames = len(telemetry)
    if n_frames <= 0 or crop_width <= 0 or crop_height <= 0 or frame_width <= 0 or frame_height <= 0:
        return [], []

    half_w = float(crop_width) / 2.0
    half_h = float(crop_height) / 2.0
    margin_x = float(crop_width) * float(margin_frac)
    margin_y = float(crop_height) * float(margin_frac)
    default_cx = float(frame_width) / 2.0
    default_cy = float(frame_height) * float(default_y_frac)

    center_x: list[float] = []
    center_y: list[float] = []
    last_valid: tuple[float, float] | None = None

    for rec in telemetry:
        bx = rec.get("x") if isinstance(rec, Mapping) else None
        by = rec.get("y") if isinstance(rec, Mapping) else None
        vis = bool(rec.get("visible")) if isinstance(rec, Mapping) else False

        bx_val = float(bx) if bx is not None else float("nan")
        by_val = float(by) if by is not None else float("nan")

        if vis and math.isfinite(bx_val) and math.isfinite(by_val):
            last_valid = (float(bx_val), float(by_val))

        if last_valid is not None:
            bx_use, by_use = last_valid
        else:
            bx_use, by_use = default_cx, default_cy

        cx_val = float(bx_use)
        cy_val = float(by_use)

        if half_w > margin_x:
            cx_val = max(bx_use - (half_w - margin_x), min(cx_val, bx_use + (half_w - margin_x)))
        if half_h > margin_y:
            cy_val = max(by_use - (half_h - margin_y), min(cy_val, by_use + (half_h - margin_y)))

        cx_val = max(half_w, min(float(frame_width) - half_w, cx_val))
        cy_val = max(half_h, min(float(frame_height) - half_h, cy_val))

        center_x.append(cx_val)
        center_y.append(cy_val)

    def _smooth(values: list[float]) -> list[float]:
        radius = max(0, int(smooth_radius))
        if radius <= 0 or len(values) <= 1:
            return [float(v) for v in values]
        smoothed: list[float] = []
        for idx, _ in enumerate(values):
            lo = max(0, idx - radius)
            hi = min(len(values), idx + radius + 1)
            window = values[lo:hi]
            smoothed.append(float(sum(window) / max(len(window), 1)))
        return smoothed

    center_x = _smooth(center_x)
    center_y = _smooth(center_y)

    max_speed = float(max_speed_px)
    if math.isfinite(max_speed) and max_speed > 0.0:
        for idx in range(1, n_frames):
            dx = center_x[idx] - center_x[idx - 1]
            dy = center_y[idx] - center_y[idx - 1]
            if abs(dx) > max_speed:
                center_x[idx] = center_x[idx - 1] + math.copysign(max_speed, dx)
            if abs(dy) > max_speed:
                center_y[idx] = center_y[idx - 1] + math.copysign(max_speed, dy)
            center_x[idx] = max(half_w, min(float(frame_width) - half_w, center_x[idx]))
            center_y[idx] = max(half_h, min(float(frame_height) - half_h, center_y[idx]))

    return center_x, center_y


@dataclass
class CamState:
    frame: int
    cx: float
    cy: float
    zoom: float
    crop_w: float
    crop_h: float
    x0: float
    y0: float
    used_label: bool
    clamp_flags: List[str]
    ball: Optional[Tuple[float, float]] = None
    zoom_scale: float = 1.0


class CameraPlanner:
    """Planner that tracks the ball and produces smoothed camera states."""

    def __init__(
        self,
        width: int,
        height: int,
        fps: float,
        lookahead: int,
        smoothing: float,
        pad: float,
        speed_limit: float,
        zoom_min: float,
        zoom_max: float,
        portrait: Optional[Tuple[int, int]] = None,
        *,
        margin_px: float = 0.0,
        lead_frames: int = 0,
        speed_zoom: Optional[Mapping[str, object]] = None,
        min_box: Optional[Tuple[float, float]] = None,
        horizon_lock: float = 0.0,
        emergency_gain: float = 0.6,
        emergency_zoom_max: float = 1.45,
        keepinview_margin_px: Optional[float] = None,
        keepinview_nudge_gain: float = 0.5,
        keepinview_zoom_gain: float = 0.4,
        keepinview_zoom_out_max: float = 1.6,
        center_frac: float = 0.5,
    ) -> None:
        self.width = float(width)
        self.height = float(height)
        self.fps = float(fps)
        self.lookahead = max(0, int(lookahead))
        self.smoothing = float(np.clip(smoothing, 0.0, 1.0))
        self.pad = max(0.0, min(0.45, float(pad)))
        self.speed_limit = max(0.0, float(speed_limit))
        self.zoom_min = max(0.1, float(zoom_min))
        self.zoom_max = max(self.zoom_min, float(zoom_max))
        self.portrait = portrait

        self.margin_px = max(0.0, float(margin_px))
        self.lead_frames = max(0, int(lead_frames))
        self.emergency_gain = float(np.clip(emergency_gain, 0.0, 1.0))
        self.emergency_zoom_max = max(1.0, float(emergency_zoom_max))

        if keepinview_margin_px is None:
            keepinview_margin_px = self.margin_px
        self.keepinview_margin_px = max(0.0, float(keepinview_margin_px))
        self.keepinview_nudge_gain = max(0.0, float(keepinview_nudge_gain))
        self.keepinview_zoom_gain = max(0.0, float(keepinview_zoom_gain))
        self.keepinview_zoom_out_max = max(1.0, float(keepinview_zoom_out_max))
        # Explicit keep-in-view band edges for any external planner code
        # that wants named attributes instead of the implicit band.
        # These are vertical fractions of the portrait height where we
        # consider the ball "comfortably framed".
        self.keepinview_min_band_frac = 0.25
        self.keepinview_max_band_frac = 0.75

        if not math.isfinite(center_frac):
            center_frac = 0.5
        self.center_frac = float(np.clip(center_frac, 0.0, 1.0))
        self.center_bias_px = (0.5 - self.center_frac) * self.height

        render_fps = self.fps if self.fps > 0 else 30.0
        if render_fps <= 0:
            render_fps = 30.0
        self.render_fps = render_fps
        self.speed_norm_px = 24.0 * (render_fps / 24.0)
        self.zoom_slew = 0.02 * (render_fps / 24.0)

        self.min_box_w = 0.0
        self.min_box_h = 0.0

        self.horizon_lock = float(np.clip(horizon_lock, 0.0, 1.0))

        self.speed_zoom_config: Optional[dict[str, float]] = None

        base_side = min(self.width, self.height)
        base_side = max(1.0, base_side)
        target_final_side = max(base_side * (1.0 - 2.0 * self.pad), base_side * 0.35)
        shrink_factor = 1.0
        if self.pad > 0.0:
            shrink_factor = max(0.05, 1.0 - 2.0 * self.pad)
        pre_pad_target = target_final_side / shrink_factor
        desired_zoom = base_side / max(pre_pad_target, 1.0)
        self.base_zoom = float(np.clip(desired_zoom, self.zoom_min, self.zoom_max))
        self.edge_zoom_min_scale = 0.75

    def plan(self, positions: np.ndarray, used_mask: np.ndarray) -> List[CamState]:
        frame_count = len(positions)
        states: List[CamState] = []
        prev_cx = self.width / 2.0
        prev_cy = self.height * self.center_frac
        prev_zoom = self.base_zoom
        fallback_center = np.array([prev_cx, self.height * self.center_frac], dtype=np.float32)
        fallback_alpha = 0.05
        render_fps = self.render_fps
        px_per_sec_x = self.speed_limit * 1.35
        px_per_sec_y = self.speed_limit * 0.90
        pxpf_x = px_per_sec_x / render_fps if render_fps > 0 else 0.0
        pxpf_y = px_per_sec_y / render_fps if render_fps > 0 else 0.0
        center_alpha = float(np.clip(self.smoothing, 0.0, 1.0))
        if math.isclose(center_alpha, 0.0, abs_tol=1e-6):
            center_alpha = 0.28
        zoom_slew = self.zoom_slew
        prev_target_x = prev_cx
        prev_target_y = prev_cy

        aspect_target = None
        aspect_ratio = self.width / max(self.height, 1e-6)
        if self.portrait:
            aspect_target = float(self.portrait[0]) / float(self.portrait[1])
            aspect_ratio = aspect_target

        def _clamp_axis(prev_value: float, current_value: float, limit: float) -> Tuple[float, bool]:
            if limit <= 0.0:
                if not math.isclose(current_value, prev_value, rel_tol=1e-9, abs_tol=1e-3):
                    return prev_value, True
                return current_value, False
            delta = current_value - prev_value
            if abs(delta) > limit:
                return prev_value + (limit if delta > 0 else -limit), True
            return current_value, False

        def _compute_crop_dimensions(
            zoom_value: float,
        ) -> Tuple[float, float, float]:
            zoom_clamped = float(np.clip(zoom_value, self.zoom_min, self.zoom_max))
            crop_h = self.height / max(zoom_clamped, 1e-6)
            crop_w = crop_h * aspect_ratio
            if crop_w > self.width:
                crop_w = self.width
                crop_h = crop_w / max(aspect_ratio, 1e-6)

            if self.pad > 0.0:
                pad_scale = max(0.0, 1.0 - 2.0 * self.pad)
                crop_w *= pad_scale
                crop_h *= pad_scale

            min_box_w = self.min_box_w
            min_box_h = self.min_box_h
            if min_box_w > 0.0 or min_box_h > 0.0:
                if min_box_w <= 0.0 and min_box_h > 0.0 and aspect_ratio > 0.0:
                    min_box_w = min_box_h * aspect_ratio
                elif min_box_h <= 0.0 and min_box_w > 0.0 and aspect_ratio > 0.0:
                    min_box_h = min_box_w / max(aspect_ratio, 1e-6)
                crop_w = max(crop_w, min_box_w)
                crop_h = max(crop_h, min_box_h)

            crop_w = float(np.clip(crop_w, 1.0, self.width))
            crop_h = float(np.clip(crop_h, 1.0, self.height))
            return zoom_clamped, crop_w, crop_h

        def _compute_crop(
            center_x: float,
            center_y: float,
            zoom_value: float,
        ) -> Tuple[float, float, float, float, float, float, float, bool]:
            zoom_clamped, crop_w, crop_h = _compute_crop_dimensions(zoom_value)
            adjusted_cy = center_y
            if aspect_target:
                adjusted_cy = adjusted_cy + 0.10 * crop_h
                if self.horizon_lock > 0.0:
                    anchor = self.height * self.horizon_lock
                    adjusted_cy = float(
                        (1.0 - self.horizon_lock) * adjusted_cy + self.horizon_lock * anchor
                    )

            desired_x0 = center_x - crop_w / 2.0
            desired_y0 = adjusted_cy - crop_h / 2.0
            max_x0 = max(0.0, self.width - crop_w)
            max_y0 = max(0.0, self.height - crop_h)
            x0 = float(np.clip(desired_x0, 0.0, max_x0))
            y0 = float(np.clip(desired_y0, 0.0, max_y0))
            bounds_clamped = not (
                math.isclose(x0, desired_x0, rel_tol=1e-6, abs_tol=1e-3)
                and math.isclose(y0, desired_y0, rel_tol=1e-6, abs_tol=1e-3)
            )

            actual_cx = x0 + crop_w / 2.0
            actual_cy = y0 + crop_h / 2.0

            return crop_w, crop_h, x0, y0, actual_cx, actual_cy, zoom_clamped, bounds_clamped

        for frame_idx in range(frame_count):
            pos = positions[frame_idx]
            has_position = bool(used_mask[frame_idx]) and not np.isnan(pos).any()

            if has_position:
                target = pos.copy()
                if self.lead_frames > 0:
                    lead_idx = min(frame_count - 1, frame_idx + self.lead_frames)
                    if lead_idx != frame_idx:
                        future_valid = bool(used_mask[lead_idx]) and not np.isnan(
                            positions[lead_idx]
                        ).any()
                        if future_valid:
                            lead_pos = positions[lead_idx]
                            target = 0.6 * target + 0.4 * lead_pos
            else:
                fallback_target = np.array(
                    [self.width / 2.0, self.height * self.center_frac], dtype=np.float32
                )
                fallback_center = (
                    fallback_alpha * fallback_target + (1.0 - fallback_alpha) * fallback_center
                )
                target = fallback_center

            # Lookahead bias.
            if self.lookahead > 0 and frame_idx < frame_count - 1:
                max_future = min(frame_count - 1, frame_idx + self.lookahead)
                future_positions = positions[frame_idx + 1 : max_future + 1]
                future_mask = used_mask[frame_idx + 1 : max_future + 1]
                valid_future = future_positions[future_mask]
                if valid_future.size:
                    future_mean = valid_future.mean(axis=0)
                    target = 0.65 * target + 0.35 * future_mean

            bx_used = float(target[0])
            by_used = float(target[1])

            if has_position:
                target_center_y = float(
                    np.clip(by_used + self.center_bias_px, 0.0, self.height)
                )
            else:
                target_center_y = float(np.clip(by_used, 0.0, self.height))

            speed_pf = math.hypot(bx_used - prev_target_x, target_center_y - prev_target_y)
            if self.speed_zoom_config:
                config = self.speed_zoom_config
                v_lo = config["v_lo"]
                v_hi = config["v_hi"]
                if v_hi <= v_lo:
                    norm = 1.0 if speed_pf >= v_hi else 0.0
                else:
                    norm = float(np.clip((speed_pf - v_lo) / max(v_hi - v_lo, 1e-6), 0.0, 1.0))
                zoom_factor = config["zoom_lo"] + (config["zoom_hi"] - config["zoom_lo"]) * norm
                zoom_target = float(config["base_zoom"] * zoom_factor)
                zoom_target = float(np.clip(zoom_target, self.zoom_min, self.zoom_max))
            else:
                speed_norm_px = self.speed_norm_px
                norm = 0.0
                if speed_norm_px > 1e-6:
                    norm = min(1.0, speed_pf / speed_norm_px)
                zoom_target = self.zoom_min + (self.zoom_max - self.zoom_min) * (1.0 - norm)
            zoom_step = float(np.clip(zoom_target - prev_zoom, -zoom_slew, zoom_slew))
            zoom = float(np.clip(prev_zoom + zoom_step, self.zoom_min, self.zoom_max))

            cx = center_alpha * bx_used + (1.0 - center_alpha) * prev_cx
            cy = center_alpha * target_center_y + (1.0 - center_alpha) * prev_cy

            ball_point: Optional[Tuple[float, float]] = None
            if has_position:
                ball_point = (float(pos[0]), float(pos[1]))

            clamp_flags: List[str] = []
            edge_zoom_scale = 1.0

            keepinview_zoom_out = 1.0
            keepinview_margin = self.keepinview_margin_px
            if keepinview_margin > 0.0:
                bx_guard: Optional[float]
                by_guard: Optional[float]
                if ball_point:
                    bx_guard, by_guard = ball_point
                else:
                    bx_guard, by_guard = float(target[0]), float(target[1])

                if math.isfinite(bx_guard) and math.isfinite(by_guard):
                    _, kv_crop_w, kv_crop_h = _compute_crop_dimensions(zoom)
                    if kv_crop_w > 0.0 and kv_crop_h > 0.0:
                        guard_frac = max(0.0, float(self.keepinview_min_band_frac))
                        if guard_frac > 0.0:
                            keepinview_margin = max(
                                keepinview_margin,
                                kv_crop_w * guard_frac,
                                kv_crop_h * guard_frac,
                            )
                        half_w = kv_crop_w * 0.5
                        half_h = kv_crop_h * 0.5

                        left_gap = bx_guard - (cx - half_w + keepinview_margin)
                        right_gap = (cx + half_w - keepinview_margin) - bx_guard
                        top_gap = by_guard - (cy - half_h + keepinview_margin)
                        bot_gap = (cy + half_h - keepinview_margin) - by_guard
                        tight = min(left_gap, right_gap, top_gap, bot_gap)

                        threshold_nudge = keepinview_margin * 0.15
                        if self.keepinview_nudge_gain > 0.0 and tight < threshold_nudge:
                            ex = 0.0
                            ey = 0.0
                            if left_gap < threshold_nudge:
                                ex -= threshold_nudge - left_gap
                            if right_gap < threshold_nudge:
                                ex += threshold_nudge - right_gap
                            if top_gap < threshold_nudge:
                                ey -= threshold_nudge - top_gap
                            if bot_gap < threshold_nudge:
                                ey += threshold_nudge - bot_gap
                            if ex or ey:
                                cx += self.keepinview_nudge_gain * ex
                                cy += self.keepinview_nudge_gain * ey
                                if "keepin_nudge" not in clamp_flags:
                                    clamp_flags.append("keepin_nudge")
                                left_gap = bx_guard - (cx - half_w + keepinview_margin)
                                right_gap = (cx + half_w - keepinview_margin) - bx_guard
                                top_gap = by_guard - (cy - half_h + keepinview_margin)
                                bot_gap = (cy + half_h - keepinview_margin) - by_guard

                        threshold_zoom = keepinview_margin * 0.25
                        min_gap = min(left_gap, right_gap, top_gap, bot_gap)
                        if (
                            self.keepinview_zoom_gain > 0.0
                            and threshold_zoom > 0.0
                            and min_gap < threshold_zoom
                        ):
                            deficit = threshold_zoom - min_gap
                            keepinview_zoom_out = 1.0 + self.keepinview_zoom_gain * (
                                deficit / threshold_zoom
                            )
                            keepinview_zoom_out = min(
                                keepinview_zoom_out,
                                self.keepinview_zoom_out_max,
                            )

            if keepinview_zoom_out > 1.0:
                keepin_scale = 1.0 / keepinview_zoom_out
                zoom = max(self.zoom_min, zoom * keepin_scale)
                edge_zoom_scale *= keepin_scale
                if not any(flag.startswith("keepin_zoom=") for flag in clamp_flags):
                    clamp_flags.append(f"keepin_zoom={keepinview_zoom_out:.3f}")

            cx, x_clamped = _clamp_axis(prev_cx, cx, pxpf_x)
            cy, y_clamped = _clamp_axis(prev_cy, cy, pxpf_y)
            speed_limited = x_clamped or y_clamped
            if speed_limited:
                clamp_flags.append("speed")

            _, est_crop_w, est_crop_h = _compute_crop_dimensions(zoom)
            bx_margin: Optional[float]
            by_margin: Optional[float]
            if ball_point:
                bx_margin, by_margin = ball_point
            else:
                bx_margin, by_margin = float(target[0]), float(target[1])
            zoom_scale = edge_aware_zoom(
                cx,
                cy,
                bx_margin,
                by_margin,
                est_crop_w,
                est_crop_h,
                self.width,
                self.height,
                self.margin_px,
                s_min=self.edge_zoom_min_scale,
            )
            if zoom_scale < 0.999:
                edge_zoom_scale *= zoom_scale
                zoom = max(self.zoom_min, zoom * zoom_scale)
                clamp_flags.append(f"edge_zoom={zoom_scale:.3f}")

            # --- emergency keep-in-view ---
            if math.isfinite(bx_margin) and math.isfinite(by_margin):
                margin = float(self.margin_px)
                _, crop_w_est, crop_h_est = _compute_crop_dimensions(zoom)
                if crop_w_est > 0.0 and crop_h_est > 0.0:
                    bx = float(bx_margin)
                    by = float(by_margin)
                    crop_w = float(crop_w_est)
                    crop_h = float(crop_h_est)

                    em_gain = self.emergency_gain if hasattr(self, "emergency_gain") else 0.6
                    em_zoom = self.emergency_zoom_max if hasattr(self, "emergency_zoom_max") else 1.45
                    em_gain = float(np.clip(em_gain, 0.0, 1.0))

                    halfW = crop_w * 0.5
                    halfH = crop_h * 0.5

                    dxL = bx - (cx - halfW + margin)
                    dxR = (cx + halfW - margin) - bx
                    dyT = by - (cy - halfH + margin)
                    dyB = (cy + halfH - margin) - by

                    need_dx = 0.0
                    if dxL < 0.0:
                        need_dx = -(margin - (bx - (cx - halfW)))
                    elif dxR < 0.0:
                        need_dx = margin - ((cx + halfW) - bx)

                    need_dy = 0.0
                    if dyT < 0.0:
                        need_dy = -(margin - (by - (cy - halfH)))
                    elif dyB < 0.0:
                        need_dy = margin - ((cy + halfH) - by)

                    if need_dx or need_dy:
                        cx += em_gain * need_dx
                        cy += em_gain * need_dy

                    left_d = bx - (cx - halfW + margin)
                    right_d = (cx + halfW - margin) - bx
                    top_d = by - (cy - halfH + margin)
                    bot_d = (cy + halfH - margin) - by
                    tight = min(left_d, right_d, top_d, bot_d)

                    if tight < 2.0:
                        req_halfW = max(bx - (cx - margin), (cx + margin) - bx)
                        req_halfH = max(by - (cy - margin), (cy + margin) - by)
                        needW = max(crop_w, 2.0 * req_halfW)
                        needH = max(crop_h, 2.0 * req_halfH)
                        zoom_out = max(1.0, min(em_zoom, max(needW / max(crop_w, 1e-6), needH / max(crop_h, 1e-6))))
                    else:
                        zoom_out = 1.0

                    if zoom_out > 1.0:
                        zoom = float(np.clip(zoom / zoom_out, self.zoom_min, self.zoom_max))
                        edge_zoom_scale *= 1.0 / zoom_out

            crop_w, crop_h, x0, y0, actual_cx, actual_cy, zoom, bounds_clamped = _compute_crop(
                cx, cy, zoom
            )

            if ball_point and crop_w > 1.0 and crop_h > 1.0:
                bx, by = ball_point
                dist_left = (bx - x0) / crop_w
                dist_right = (x0 + crop_w - bx) / crop_w
                dist_top = (by - y0) / crop_h
                dist_bot = (y0 + crop_h - by) / crop_h

                edge_thr = 0.12
                zoomout_gain = 0.10

                edge_risk = min(dist_left, dist_right, dist_top, dist_bot)
                if edge_risk < edge_thr:
                    zoom = max(self.zoom_min, zoom * (1.0 - zoomout_gain))
                    crop_w, crop_h, x0, y0, actual_cx, actual_cy, zoom, bounds_again = _compute_crop(
                        cx, cy, zoom
                    )
                    bounds_clamped = bounds_clamped or bounds_again

                max_x0 = max(0.0, self.width - crop_w)
                max_y0 = max(0.0, self.height - crop_h)

                margin_px = self.margin_px
                if margin_px > 0.0:
                    desired_x0 = x0
                    desired_y0 = y0
                    if bx < x0 + margin_px:
                        desired_x0 = max(0.0, min(max_x0, bx - margin_px))
                    elif bx > x0 + crop_w - margin_px:
                        desired_x0 = max(0.0, min(max_x0, bx + margin_px - crop_w))
                    if by < y0 + margin_px:
                        desired_y0 = max(0.0, min(max_y0, by - margin_px))
                    elif by > y0 + crop_h - margin_px:
                        desired_y0 = max(0.0, min(max_y0, by + margin_px - crop_h))
                    if not (
                        math.isclose(desired_x0, x0, rel_tol=1e-6, abs_tol=1e-3)
                        and math.isclose(desired_y0, y0, rel_tol=1e-6, abs_tol=1e-3)
                    ):
                        x0 = desired_x0
                        y0 = desired_y0
                        bounds_clamped = True

                def _rounded_bounds(x_start: float, y_start: float) -> Tuple[int, int, int, int]:
                    x1_i = int(round(x_start))
                    y1_i = int(round(y_start))
                    x2_i = int(round(min(x_start + crop_w, self.width)))
                    y2_i = int(round(min(y_start + crop_h, self.height)))
                    return x1_i, y1_i, x2_i, y2_i

                for _ in range(3):
                    x1_i, y1_i, x2_i, y2_i = _rounded_bounds(x0, y0)
                    moved = False
                    if bx < x1_i:
                        shift = x1_i - bx
                        new_x0 = max(0.0, min(max_x0, x0 - shift))
                        moved = moved or not math.isclose(new_x0, x0, rel_tol=1e-6, abs_tol=1e-3)
                        x0 = new_x0
                    elif bx > x2_i - 1:
                        shift = bx - (x2_i - 1)
                        new_x0 = max(0.0, min(max_x0, x0 + shift))
                        moved = moved or not math.isclose(new_x0, x0, rel_tol=1e-6, abs_tol=1e-3)
                        x0 = new_x0

                    if by < y1_i:
                        shift = y1_i - by
                        new_y0 = max(0.0, min(max_y0, y0 - shift))
                        moved = moved or not math.isclose(new_y0, y0, rel_tol=1e-6, abs_tol=1e-3)
                        y0 = new_y0
                    elif by > y2_i - 1:
                        shift = by - (y2_i - 1)
                        new_y0 = max(0.0, min(max_y0, y0 + shift))
                        moved = moved or not math.isclose(new_y0, y0, rel_tol=1e-6, abs_tol=1e-3)
                        y0 = new_y0

                    if not moved:
                        break

                actual_cx = x0 + crop_w / 2.0
                actual_cy = y0 + crop_h / 2.0
                if (
                    math.isclose(x0, 0.0, rel_tol=1e-6, abs_tol=1e-3)
                    or math.isclose(y0, 0.0, rel_tol=1e-6, abs_tol=1e-3)
                    or math.isclose(x0, max_x0, rel_tol=1e-6, abs_tol=1e-3)
                    or math.isclose(y0, max_y0, rel_tol=1e-6, abs_tol=1e-3)
                ):
                    bounds_clamped = True

            if bounds_clamped:
                clamp_flags.append("bounds")

            prev_cx = actual_cx
            prev_cy = actual_cy
            prev_zoom = zoom
            prev_target_x = bx_used
            prev_target_y = target_center_y

            states.append(
                CamState(
                    frame=frame_idx,
                    cx=actual_cx,
                    cy=actual_cy,
                    zoom=zoom,
                    crop_w=crop_w,
                    crop_h=crop_h,
                    x0=x0,
                    y0=y0,
                    used_label=bool(has_position),
                    clamp_flags=clamp_flags,
                    ball=ball_point,
                    zoom_scale=edge_zoom_scale,
                )
            )

        return states


def _load_overlay(path: Optional[Path], output_size: Tuple[int, int]) -> Optional[np.ndarray]:
    if not path:
        return None
    if not path.exists():
        logging.warning("Brand overlay %s not found; skipping.", path)
        return None
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        logging.warning("Failed to read brand overlay at %s; skipping.", path)
        return None
    resized = cv2.resize(image, output_size, interpolation=cv2.INTER_LINEAR)
    return resized


def _apply_overlay(frame: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    if overlay.shape[2] < 4:
        return cv2.addWeighted(frame, 0.7, overlay[:, :, :3], 0.3, 0.0)
    alpha = overlay[:, :, 3:] / 255.0
    base = frame.astype(np.float32)
    overlay_rgb = overlay[:, :, :3].astype(np.float32)
    blended = overlay_rgb * alpha + base * (1.0 - alpha)
    return blended.astype(np.uint8)


class Renderer:
    def __init__(
        self,
        input_path: Path,
        output_path: Path,
        temp_dir: Path,
        fps_in: float,
        fps_out: float,
        flip180: bool,
        portrait: Optional[Tuple[int, int]],
        brand_overlay: Optional[Path],
        endcard: Optional[Path],
        pad: float,
        zoom_min: float,
        zoom_max: float,
        speed_limit: float,
        telemetry: Optional[TextIO],
        telemetry_simple: Optional[TextIO] = None,
        init_manual: bool = False,
        init_t: float = 0.8,
        ball_path: Optional[Sequence[BallPathEntry]] = None,
        follow_lead_time: float = 0.0,
        follow_margin_px: float = 0.0,
        follow_smoothing: float = 0.3,
        *,
        follow_zeta: float = 0.95,
        follow_wn: float = 6.0,
        follow_deadzone: float = 0.0,
        follow_max_vel: Optional[float] = None,
        follow_max_acc: Optional[float] = None,
        follow_lookahead: int = 0,
        follow_pre_smooth: float = 0.0,
        follow_zoom_out_max: float = 1.35,
        follow_zoom_edge_frac: float = 1.0,
        follow_center_frac: float = 0.5,
        lost_hold_ms: int = 500,
        lost_pan_ms: int = 1200,
        lost_lookahead_s: float = 6.0,
        lost_chase_motion_ms: int = 900,
        lost_motion_thresh: float = 1.6,
        lost_use_motion: bool = False,
        portrait_plan_margin_px: Optional[float] = None,
        portrait_plan_headroom: Optional[float] = None,
        portrait_plan_lead_px: Optional[float] = None,
        plan_override_data: Optional[dict[str, np.ndarray]] = None,
        plan_override_len: int = 0,
        ball_samples: Optional[List[BallSample]] = None,
        keep_path_lookup_data: Optional[dict[int, Tuple[float, float]]] = None,
        debug_ball_overlay: bool = False,
        follow_override: Optional[Mapping[int, Mapping[str, float]]] = None,
        follow_exact: bool = False,
        disable_controller: bool = False,
        follow_trajectory: Optional[List[Mapping[str, float]]] = None,
    ) -> None:
        # Fallback initialization for variables that used to be set in try/except blocks
        motion_thresh_value = None
        # Ensure defaults for variables that may have been assigned inside removed try/except blocks
        zoom_edge_frac = None
        motion_thresh_value = None  # fallback for removed try/except

        self.input_path = input_path
        self.output_path = output_path
        self.temp_dir = temp_dir
        self.fps_in = fps_in
        self.fps_out = fps_out
        self.flip180 = flip180
        self.portrait = portrait
        self.brand_overlay_path = brand_overlay
        self.endcard_path = endcard
        self.pad = float(pad)
        self.zoom_min = float(zoom_min)
        self.zoom_max = float(zoom_max)
        self.speed_limit = float(speed_limit)
        self.telemetry = telemetry
        self.telemetry_simple = telemetry_simple
        self.last_ffmpeg_command: Optional[List[str]] = None
        self.init_manual = bool(init_manual)
        self.init_t = float(init_t)
        self.follow_lead_time = max(0.0, float(follow_lead_time))
        self.follow_margin_px = max(0.0, float(follow_margin_px))
        self.follow_smoothing = float(np.clip(follow_smoothing, 0.0, 1.0))
        self.follow_zeta = float(follow_zeta)
        self.follow_wn = float(follow_wn)
        self.follow_deadzone = max(0.0, float(follow_deadzone))
        self.follow_max_vel = None if follow_max_vel is None else float(follow_max_vel)
        self.follow_max_acc = None if follow_max_acc is None else float(follow_max_acc)
        self.follow_lookahead = int(follow_lookahead)
        self.follow_pre_smooth = float(np.clip(follow_pre_smooth, 0.0, 1.0))
        self.follow_zoom_out_max = max(1.0, float(follow_zoom_out_max))
        zoom_edge_frac = follow_zoom_edge_frac
        if zoom_edge_frac is None:
            zoom_edge_frac = 0.15  # safe default; avoids crash
        if not math.isfinite(zoom_edge_frac) or zoom_edge_frac <= 0.0:
            zoom_edge_frac = 1.0
        self.follow_zoom_edge_frac = zoom_edge_frac
        if not math.isfinite(follow_center_frac):
            follow_center_frac = 0.5
        self.follow_center_frac = float(np.clip(follow_center_frac, 0.0, 1.0))
        self.lost_hold_ms = max(0, int(lost_hold_ms))
        self.lost_pan_ms = max(0, int(lost_pan_ms))
        self.lost_chase_motion_ms = max(0, int(lost_chase_motion_ms))
        if not math.isfinite(lost_lookahead_s) or lost_lookahead_s < 0.0:
            lost_lookahead_s = 0.0
        self.lost_lookahead_s = lost_lookahead_s
        if motion_thresh_value is None:
            motion_thresh_value = 0.02  # safe small threshold
        # Ensure variable is initialized
        if motion_thresh_value is None:
            motion_thresh_value = 0.02  # safe fallback threshold
        if not math.isfinite(motion_thresh_value):
            motion_thresh_value = 1.6
        self.lost_motion_thresh = max(0.0, motion_thresh_value)
        self.lost_use_motion = bool(lost_use_motion)
        self.keepinview_min_band_frac = 0.12
        self.follow_exact = bool(follow_exact)
        self.disable_controller = bool(disable_controller)
        self.follow_trajectory = list(follow_trajectory) if follow_trajectory else None

        def _coerce_float(value: Optional[float]) -> Optional[float]:
            if value is None:
                return None

        self.portrait_plan_margin_px = _coerce_float(portrait_plan_margin_px)
        self.portrait_plan_headroom_frac = _coerce_float(portrait_plan_headroom)
        self.portrait_plan_lead_px = _coerce_float(portrait_plan_lead_px)
        self.plan_override_data = plan_override_data
        self.plan_override_len = int(plan_override_len or 0)
        self.ball_samples = ball_samples or []
        self.keep_path_lookup_data = keep_path_lookup_data or {}
        self.debug_ball_overlay = bool(debug_ball_overlay)
        self.follow_override = follow_override

        normalized_ball_path: Optional[List[Optional[dict[str, float]]]] = None
        if ball_path:
            normalized_ball_path = []
            for entry in ball_path:
                if entry is None:
                    normalized_ball_path.append(None)
                    continue
                if isinstance(entry, Mapping):
                    sanitized: dict[str, float] = {}
                    for key, value in entry.items():
                        if isinstance(value, (int, float)):
                            sanitized[key] = float(value)
                    if "z" not in sanitized:
                        z_value = entry.get("z", 1.30) if hasattr(entry, "get") else 1.30
                    bx_norm = sanitized.get("bx")
                    by_norm = sanitized.get("by")
                    if bx_norm is None or by_norm is None:
                        bx_norm = sanitized.get("bx_stab", sanitized.get("bx_raw", bx_norm))
                        by_norm = sanitized.get("by_stab", sanitized.get("by_raw", by_norm))
                    if bx_norm is None or by_norm is None:
                        normalized_ball_path.append(None)
                        continue
                    sanitized["bx"] = float(bx_norm)
                    sanitized["by"] = float(by_norm)
                    normalized_ball_path.append(sanitized)
                else:
                    entry_seq = tuple(entry)
                    if len(entry_seq) < 2:
                        normalized_ball_path.append(None)
                        continue
                    bx_val = entry_seq[0]
                    by_val = entry_seq[1]
                    if bx_val is None or by_val is None:
                        normalized_ball_path.append(None)
                        continue
                    z_val = entry_seq[2] if len(entry_seq) >= 3 else 1.30
        self.offline_ball_path = normalized_ball_path

    def _simulate_follow_centers(
        self,
        follow_targets: Optional[Tuple[List[float], List[float]]],
        follow_lookahead_frames: int,
        render_fps: float,
        start_cx: float,
        start_cy: float,
        frame_count: int,
        follow_valid_mask: Optional[Sequence[bool]] = None,
    ) -> List[Tuple[float, float]]:
        if (
            follow_targets is None
            or render_fps <= 0.0
            or frame_count <= 0
            or not follow_targets[0]
            or not follow_targets[1]
        ):
            return []

        follower = CamFollow2O(
            zeta=self.follow_zeta,
            wn=self.follow_wn,
            dt=1.0 / render_fps,
            max_vel=self.follow_max_vel,
            max_acc=self.follow_max_acc,
            deadzone=self.follow_deadzone,
        )
        follower.cx = float(start_cx)
        follower.cy = float(start_cy)
        follower.vx = 0.0
        follower.vy = 0.0

        xs, ys = follow_targets
        last_idx = min(len(xs), len(ys)) - 1
        hold = FollowHoldController(
            dt=1.0 / render_fps if render_fps > 1e-6 else 1.0 / 30.0,
            release_frames=3,
            decay_time=0.4,
            initial_target=(start_cx, start_cy),
        )
        centers: List[Tuple[float, float]] = []
        for frame_idx in range(frame_count):
            target_idx = min(frame_idx + follow_lookahead_frames, last_idx)
            target_x = float(xs[target_idx])
            target_y = float(ys[target_idx])
            valid = True
            if follow_valid_mask is not None and target_idx < len(follow_valid_mask):
                valid = bool(follow_valid_mask[target_idx])
            elif not (math.isfinite(target_x) and math.isfinite(target_y)):
                valid = False
            eff_x, eff_y, holding = hold.apply(target_x, target_y, valid)
            if holding and 0.0 < hold.decay_factor < 1.0:
                follower.damp_velocity(hold.decay_factor)
            cx, cy = follower.step(eff_x, eff_y)
            if not holding:
                hold.reset_target(cx, cy)
            centers.append((cx, cy))
        return centers

    @staticmethod
    def _center_bias_px_for_height(frame_h: float, center_frac: float) -> float:
        if not math.isfinite(center_frac):
            center_frac = 0.5
        return (0.5 - center_frac) * frame_h

    def _compose_frame(
        self,
        frame: np.ndarray,
        state: CamState,
        output_size: Tuple[int, int],
        overlay_image: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        height, width = frame.shape[:2]
        target_ar = 0.0
        if output_size[0] > 0 and output_size[1] > 0:
            target_ar = float(output_size[0]) / float(output_size[1])

        crop_w = float(np.clip(state.crop_w, 1.0, float(width)))
        crop_h = float(np.clip(state.crop_h, 1.0, float(height)))

        if target_ar > 0.0 and crop_h > 0.0:
            desired_w = crop_h * target_ar
            desired_h = crop_w / target_ar if target_ar > 0.0 else crop_h
            if desired_w <= width and not math.isclose(desired_w, crop_w, rel_tol=1e-4, abs_tol=1e-3):
                crop_w = float(desired_w)
            elif desired_h <= height and not math.isclose(desired_h, crop_h, rel_tol=1e-4, abs_tol=1e-3):
                crop_h = float(desired_h)

        max_x0 = max(0.0, float(width) - crop_w)
        max_y0 = max(0.0, float(height) - crop_h)
        clamped_x0 = float(np.clip(state.x0, 0.0, max_x0))
        clamped_y0 = float(np.clip(state.y0, 0.0, max_y0))

        x2_f = clamped_x0 + crop_w
        y2_f = clamped_y0 + crop_h
        if x2_f > width:
            clamped_x0 = max(0.0, float(width) - crop_w)
            x2_f = clamped_x0 + crop_w
        if y2_f > height:
            clamped_y0 = max(0.0, float(height) - crop_h)
            y2_f = clamped_y0 + crop_h

        x1 = int(round(clamped_x0))
        y1 = int(round(clamped_y0))
        x2 = int(round(min(x2_f, float(width))))
        y2 = int(round(min(y2_f, float(height))))
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))

        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            cropped = frame
            x1, y1 = 0, 0
            x2, y2 = width, height

        resized = cv2.resize(cropped, output_size, interpolation=cv2.INTER_CUBIC)

        if overlay_image is not None:
            resized = _apply_overlay(resized, overlay_image)

        actual_crop = (float(x1), float(y1), float(x2 - x1), float(y2 - y1))
        return resized, actual_crop

    def _append_endcard(self, output_size: Tuple[int, int]) -> List[np.ndarray]:
        if not self.endcard_path:
            return []
        if not self.endcard_path.exists():
            logging.warning("Endcard %s not found; skipping.", self.endcard_path)
            return []
        image = cv2.imread(str(self.endcard_path))
        if image is None:
            logging.warning("Failed to read endcard at %s; skipping.", self.endcard_path)
            return []
        resized = cv2.resize(image, output_size, interpolation=cv2.INTER_LINEAR)
        frame_count = int(round(self.fps_out * 2.0))
        return [resized for _ in range(frame_count)]

    def write_frames(self, states: Sequence[CamState], *, probe_only: bool = False) -> float:
        input_mp4 = str(self.input_path)
        cap = cv2.VideoCapture(input_mp4)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_mp4}")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or float(self.fps_in) or 30.0
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if src_w <= 0 or src_h <= 0:
            ok, _first_frame = cap.read()
            if not ok or _first_frame is None:
                cap.release()
                raise RuntimeError("No frames decoded from the input video.")
            src_h, src_w = _first_frame.shape[:2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        width = int(src_w)
        height = int(src_h)
        if self.portrait:
            output_size = self.portrait
        else:
            output_size = (width, height)

        target_w = int(output_size[0]) if output_size[0] else width
        target_h = int(output_size[1]) if output_size[1] else height
        if target_h <= 0:
            target_h = height
        if target_w <= 0:
            target_w = width
        target_aspect = float(target_w) / float(target_h) if target_h else (width / float(height))
        output_size = (target_w, target_h)

        overlay_image = _load_overlay(self.brand_overlay_path, output_size)
        tf = self.telemetry
        simple_tf = self.telemetry_simple

        is_portrait = target_h > target_w
        portrait_plan_state: dict[str, float] = {}
        out_w = target_w
        out_h = target_h
        portrait_w = out_w if is_portrait else None
        portrait_h = out_h if is_portrait else None
        portrait_crop_w = None
        portrait_crop_h = None
        if is_portrait:
            portrait_crop_h = float(src_h)
            portrait_crop_w = float(int(round(src_h * 9.0 / 16.0)))
            portrait_crop_w = max(1.0, min(portrait_crop_w, float(src_w)))

        offline_ball_path = self.offline_ball_path

        frame_count = len(states)
        keep_path_lookup: dict[int, tuple[float, float]] = dict(self.keep_path_lookup_data)
        keepinview_enabled = bool(
            is_portrait and portrait_w and portrait_h and portrait_w > 0 and portrait_h > 0
        )
        if keepinview_enabled:
            crop_w = float(portrait_crop_w if portrait_crop_w else (portrait_w or out_w))
            crop_h = float(portrait_crop_h if portrait_crop_h else (portrait_h or out_h))
            if not keep_path_lookup:
                samples = self.ball_samples or load_ball_telemetry_for_clip(str(self.input_path))
                if samples:
                    total_frames = frame_count
                    if total_frames <= 0:
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if total_frames <= 0:
                        total_frames = max((int(getattr(s, "frame", 0)) for s in samples), default=0) + 1
                    telemetry_frames = []
                    sample_by_frame: dict[int, BallSample] = {}
                    for s in samples:
                        if idx < 0:
                            continue
                        sample_by_frame[idx] = s
                    fps_hint = float(self.fps_out or src_fps or 30.0)
                    for idx in range(total_frames):
                        sample = sample_by_frame.get(idx)
                        if sample is None:
                            telemetry_frames.append(
                                {
                                    "t": idx / fps_hint,
                                    "x": None,
                                    "y": None,
                                    "visible": False,
                                }
                            )
                            continue
                        bx = _safe_float(getattr(sample, "x", None))
                        by = _safe_float(getattr(sample, "y", None))
                        telemetry_frames.append(
                            {
                                "t": getattr(sample, "t", idx / fps_hint),
                                "x": bx,
                                "y": by,
                                "visible": bx is not None and by is not None,
                            }
                        )
                    raw_cx, raw_cy = build_raw_ball_center_path(
                        telemetry_frames,
                        frame_width=int(width),
                        frame_height=int(height),
                        crop_width=int(crop_w),
                        crop_height=int(crop_h),
                    )
                    if raw_cx and raw_cy and len(raw_cx) == len(raw_cy):
                        cx_vals, cy_vals = smooth_center_path(
                            raw_cx,
                            raw_cy,
                            window=9,
                            max_step_px=40.0,
                        )
                        cx_vals, cy_vals = clamp_center_path_to_bounds(
                            cx_vals,
                            cy_vals,
                            frame_width=int(width),
                            frame_height=int(height),
                            crop_width=int(crop_w),
                            crop_height=int(crop_h),
                        )
                        keep_path_lookup = {
                            idx: (float(cx), float(cy)) for idx, (cx, cy) in enumerate(zip(cx_vals, cy_vals))
                        }

        cam = [(state.cx, state.cy, state.zoom) for state in states]
        if cam:
            cx_values = [value[0] for value in cam]
            cy_values = [value[1] for value in cam]
        else:
            cx_values = []
            cy_values = []

        duration_s = frame_count / float(self.fps_out) if self.fps_out else 0.0
        if not cam or (
            (max(cx_values) - min(cx_values) if cx_values else 0.0) < 1.0
            and (max(cy_values) - min(cy_values) if cy_values else 0.0) < 1.0
        ):
            fallback_path = rough_motion_path(str(self.input_path), float(self.fps_out), duration_s)
            if fallback_path:
                default_zoom = cam[0][2] if cam else 1.2
                cam = [(x, y, default_zoom) for _, x, y in fallback_path]
            else:
                default_zoom = cam[0][2] if cam else 1.2
                cam = [(width / 2.0, height / 2.0, default_zoom) for _ in range(frame_count)]

        if frame_count and len(cam) < frame_count:
            last = cam[-1]
            cam.extend([last] * (frame_count - len(cam)))
        elif frame_count and len(cam) > frame_count:
            cam = cam[:frame_count]

        render_fps = float(self.fps_out)
        zoom_min = float(self.zoom_min)
        zoom_max = float(self.zoom_max)
        src_w_f = float(width)
        src_h_f = float(height)
        center_bias_px = self._center_bias_px_for_height(src_h_f, self.follow_center_frac)
        speed_px_sec = float(self.speed_limit or 3000.0)

        offline_plan_data: Optional[dict[str, np.ndarray]] = self.plan_override_data
        follow_targets: Optional[Tuple[List[float], List[float]]] = None
        follow_valid_mask: Optional[List[bool]] = None
        offline_plan_len = int(self.plan_override_len or 0)
        if offline_plan_data is None and offline_ball_path:
            bx_vals: List[float] = []
            by_vals: List[float] = []
            for entry in offline_ball_path:
                if entry is None:
                    bx_vals.append(float("nan"))
                    by_vals.append(float("nan"))
                    continue
                bx_val: Optional[float] = None
                by_val: Optional[float] = None
                if isinstance(entry, Mapping):
                    if "bx_stab" in entry and "by_stab" in entry:
                        bx_val = entry.get("bx_stab")
                        by_val = entry.get("by_stab")
                    elif "bx" in entry and "by" in entry:
                        bx_val = entry.get("bx")
                        by_val = entry.get("by")
                    elif "bx_raw" in entry and "by_raw" in entry:
                        bx_val = entry.get("bx_raw")
                        by_val = entry.get("by_raw")
                else:
                    entry_seq = tuple(entry)
                    if len(entry_seq) >= 2:
                        bx_candidate = entry_seq[0]
                        by_candidate = entry_seq[1]
                        if bx_candidate is not None and by_candidate is not None:
                            bx_val = bx_candidate
                            by_val = by_candidate

                if bx_val is None or by_val is None:
                    bx_vals.append(float("nan"))
                    by_vals.append(float("nan"))
                else:
                    bx_vals.append(float(bx_val))
                    by_vals.append(float(by_val))

            if bx_vals:
                bx_arr = np.asarray(bx_vals, dtype=float)
                by_arr = np.asarray(by_vals, dtype=float)
                valid_mask_arr = np.isfinite(bx_arr) & np.isfinite(by_arr)

                def _ffill_nan(arr: np.ndarray, default: float) -> np.ndarray:
                    out = arr.copy()
                    last = default
                    for idx in range(len(out)):
                        if np.isfinite(out[idx]):
                            last = out[idx]
                        else:
                            out[idx] = last
                    return out

                default_cx = float(width) / 2.0
                default_cy = float(height) * self.follow_center_frac
                if not np.isfinite(bx_arr[0]):
                    bx_arr[0] = default_cx
                if not np.isfinite(by_arr[0]):
                    by_arr[0] = default_cy

                bx_arr = _ffill_nan(bx_arr, default_cx)
                by_arr = _ffill_nan(by_arr, default_cy)

                bx_list = bx_arr.astype(float).tolist()
                by_list = by_arr.astype(float).tolist()
                if self.follow_pre_smooth > 0:
                    bx_list, by_list = ema_path(bx_list, by_list, self.follow_pre_smooth)
                bias_px = self._center_bias_px_for_height(src_h_f, self.follow_center_frac)
                by_list = [float(np.clip(y + bias_px, 0.0, src_h_f)) for y in by_list]
                follow_targets = (bx_list, by_list)
                follow_valid_mask = valid_mask_arr.astype(bool).tolist()

                fps_for_plan = render_fps if render_fps > 0 else (src_fps if src_fps > 0 else 30.0)
                if fps_for_plan <= 0:
                    fps_for_plan = 30.0
                global FPS
                FPS = float(fps_for_plan)

                if portrait_w and portrait_h and portrait_w > 0 and portrait_h > 0:
                    target_aspect = float(portrait_w) / float(portrait_h)
                else:
                    target_aspect = (float(width) / float(height)) if height > 0 else 1.0

                pan_alpha = float(np.clip(self.follow_smoothing, 0.05, 0.95))
                lead_seconds = max(0.0, float(self.follow_lead_time))
                bounds_pad = int(round(self.follow_margin_px)) if self.follow_margin_px > 0 else 16
                bounds_pad = max(8, bounds_pad)

                planner_enabled = bool(
                    is_portrait
                    and portrait_w
                    and portrait_h
                    and portrait_w > 0
                    and portrait_h > 0
                )
                if planner_enabled:
                    plan_x0, plan_y0, plan_w, plan_h, plan_spd, plan_zoom = plan_camera_from_ball(
                        bx_arr,
                        by_arr,
                        float(width),
                        float(height),
                        float(target_aspect),
                        pan_alpha=pan_alpha,
                        lead=lead_seconds,
                        bounds_pad=bounds_pad,
                        center_frac=self.follow_center_frac,
                        margin_px_override=self.portrait_plan_margin_px,
                        headroom_frac_override=self.portrait_plan_headroom_frac,
                        lead_px_override=self.portrait_plan_lead_px,
                    )

                    offline_plan_len = len(plan_x0)
                    offline_plan_data = {
                        "x0": plan_x0.astype(float),
                        "y0": plan_y0.astype(float),
                        "w": plan_w.astype(float),
                        "h": plan_h.astype(float),
                        "spd": plan_spd.astype(float),
                        "z": plan_zoom.astype(float),
                    }

                else:
                    offline_plan_len = 0
                    offline_plan_data = None

        kal: Optional[CV2DKalman] = None
        template: Optional[np.ndarray] = None
        tpl_side = 64
        prev_cx = src_w_f / 2.0
        prev_cy = src_h_f / 2.0
        initial_zoom = cam[0][2] if cam else 1.2
        zoom = float(np.clip(float(initial_zoom), zoom_min, zoom_max))
        prev_zoom = float(zoom)
        prev_ball_x: Optional[float] = None
        prev_ball_y: Optional[float] = None
        prev_ball_source: Optional[str] = None
        prev_ball_src_x: Optional[float] = None
        prev_ball_src_y: Optional[float] = None
        prev_bx = float(prev_cx)
        prev_by = float(prev_cy)
        if self.disable_controller:
            follow_targets = None
            follow_valid_mask = None
        follow_targets_len = len(follow_targets[0]) if follow_targets else 0
        plan_pts: List[Tuple[float, float, float]] = []
        fps_plan = render_fps if render_fps and render_fps > 0 else (src_fps if src_fps and src_fps > 0 else 30.0)
        if fps_plan <= 0:
            fps_plan = 30.0
        if follow_targets_len > 0 and follow_targets:
            for idx in range(follow_targets_len):
                plan_pts.append((idx / fps_plan if fps_plan > 0 else 0.0, bx_plan, by_plan))
        elif states:
            for state in states:
                bx_plan: float
                by_plan: float
                if state.ball is not None:
                    bx_plan = float(state.ball[0])
                    by_plan = float(state.ball[1])
                else:
                    bx_plan = float(state.cx)
                    by_plan = float(state.cy)
                frame_time = state.frame / fps_plan if fps_plan > 0 else 0.0
                plan_pts.append((frame_time, bx_plan, by_plan))
        follow_lookahead_frames = max(0, int(self.follow_lookahead))
        follower: Optional[CamFollow2O]
        if self.disable_controller:
            follower = None
        elif render_fps > 0:
            follower = CamFollow2O(
                zeta=self.follow_zeta,
                wn=self.follow_wn,
                dt=1.0 / render_fps,
                max_vel=self.follow_max_vel,
                max_acc=self.follow_max_acc,
                deadzone=self.follow_deadzone,
            )
            follower.cx = float(prev_cx)
            follower.cy = float(prev_cy)
            follower.vx = 0.0
            follower.vy = 0.0
        else:
            follower = None

        follow_hold = FollowHoldController(
            dt=1.0 / render_fps if render_fps > 1e-6 else 1.0 / 30.0,
            release_frames=3,
            decay_time=0.4,
            initial_target=(prev_cx, prev_cy),
        )

        # --- PATCH: initialize tracking flags so they always exist ---
        have_ball = False
        bx = by = None

        jerk95 = 0.0
        if follow_targets_len and render_fps > 0:
            centers = self._simulate_follow_centers(
                follow_targets,
                follow_lookahead_frames,
                render_fps,
                prev_cx,
                prev_cy,
                len(states),
                follow_valid_mask=follow_valid_mask,
            )
            if centers:
                xs = [c[0] for c in centers]
                ys = [c[1] for c in centers]
                jerk95 = compute_camera_jerk95(xs, ys, render_fps)
        self.last_jerk95 = float(jerk95)

        tf = self.telemetry

        endcard_frames = self._append_endcard(output_size)
        if endcard_frames:
            start_index = len(states)
            for offset, endcard_frame in enumerate(endcard_frames):
                out_path = self.temp_dir / f"f_{start_index + offset:06d}.jpg"
                cv2.imwrite(str(out_path), endcard_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        return float(jerk95)

def ffmpeg_stitch(
        self,
        crf: int,
        keyint: int,
        log_path: Optional[Path] = None,
    ) -> None:
        print("[DEBUG] building ffmpeg command")
        pattern = str(self.temp_dir / "f_%06d.jpg")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(self.fps_out),
            "-i",
            pattern,
            "-i",
            str(self.input_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
        ]

        if self.portrait and self.portrait[0] > 0 and self.portrait[1] > 0:
            out_w, out_h = self.portrait
            command.extend(["-vf", f"scale={int(out_w)}:{int(out_h)}"])

        command.extend(
            [
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                "-profile:v",
                "high",
                "-level",
                "4.0",
                "-x264-params",
                f"keyint={keyint}:min-keyint={keyint}:scenecut=0",
                "-movflags",
                "+faststart",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                str(self.output_path),
            ]
        )


        self.last_ffmpeg_command = list(command)

        print("[DEBUG] launching ffmpeg...")
        subprocess.run(command, check=True)
        print("[INFO] render complete")
        print("[DEBUG] ffmpeg finished successfully")



def _prepare_temp_dir(temp_dir: Path, clean: bool) -> None:
    if clean and temp_dir.exists():
        shutil.rmtree(temp_dir)
    if not temp_dir.exists():
        temp_dir.mkdir(parents=True, exist_ok=True)
        return

def _default_output_path(input_path: Path, preset: str) -> Path:
    suffix = f".__{preset.upper()}.mp4"
    return input_path.with_name(input_path.stem + suffix)


def load_ball_path(
    path: Union[str, os.PathLike[str]],
    ball_key_x: str = "bx_stab",
    ball_key_y: str = "by_stab",
) -> List[Optional[dict[str, float]]]:
    """Load a planned ball path JSONL file with stabilized coordinates."""

    seq: List[Optional[dict[str, float]]] = []
    default_zoom = 1.30
    with open(path, "r", encoding="utf-8") as f:
        for _, line in enumerate(f):
            line = line.strip()
            if not line:
                seq.append(None)
                continue
            if not isinstance(data, Mapping):
                seq.append(None)
                continue

            bx_norm: Optional[float] = None
            by_norm: Optional[float] = None
            for key_x, key_y in (
                (ball_key_x, ball_key_y),
                ("bx", "by"),
                ("bx_raw", "by_raw"),
            ):
                val_x = data.get(key_x)
                val_y = data.get(key_y)
                if val_x is None or val_y is None:
                    continue
                else:
                    break

            if bx_norm is None or by_norm is None:
                seq.append(None)
                continue

            rec: dict[str, float] = {}

            seen_pairs: set[tuple[str, str]] = set()
            for key_x, key_y in (
                (ball_key_x, ball_key_y),
                ("bx_stab", "by_stab"),
                ("bx_raw", "by_raw"),
                ("bx", "by"),
            ):
                if (key_x, key_y) in seen_pairs:
                    continue
                seen_pairs.add((key_x, key_y))
                val_x = data.get(key_x)
                val_y = data.get(key_y)
                if val_x is None or val_y is None:
                    continue

            rec["bx"] = bx_norm
            rec["by"] = by_norm

            z_value = data.get("z", default_zoom)

            t_value = data.get("t")
            if isinstance(t_value, (int, float)):
                rec["t"] = float(t_value)

            frame_value = data.get("f")
            if isinstance(frame_value, (int, float)):
                rec["f"] = float(frame_value)

            seq.append(rec)

    return seq


def run(
    args: argparse.Namespace,
    telemetry_path: Optional[Path] = None,
    telemetry_simple_path: Optional[Path] = None,
) -> None:
    renderer = None  # initialize renderer safely
    render_telemetry_path = telemetry_path
    render_telemetry_simple_path = telemetry_simple_path
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    ### PATCH: sanitize controller parameters
    def _safe(val, default):
        try:
            if val is None:
                return default
            v = float(val)
            if not math.isfinite(v):
                return default
            return v
        except Exception:
            return default

    original_source_path = Path(args.in_path).expanduser().resolve()
    if not original_source_path.exists():
        raise FileNotFoundError(f"Input file not found: {original_source_path}")

    src_path = original_source_path
    if getattr(args, "upscale", False):
        scale_value = args.upscale_scale if args.upscale_scale and args.upscale_scale > 0 else 2
        upscaled_str = upscale_video(str(src_path), scale=scale_value)
        src_path = Path(upscaled_str).expanduser().resolve()
        logging.info(
            "Upscaled source with Real-ESRGAN (scale=%s): %s -> %s",
            scale_value,
            original_source_path,
            src_path,
        )

    input_path = src_path

    presets = load_presets()
    preset_key = (args.preset or "cinematic").lower()
    if preset_key not in presets:
        raise ValueError(f"Preset '{preset_key}' not found in {PRESETS_PATH}")

    preset_config = presets[preset_key]

    fps_in = float(ffprobe_fps(input_path))
    fps_out = float(args.fps) if args.fps is not None else float(preset_config.get("fps", fps_in))
    if fps_out <= 0:
        fps_out = fps_in if fps_in > 0 else 30.0

    # --- Ensure duration_s always exists ---
    duration_s = None
    frame_count = None

    follow_config_raw = preset_config.get("follow")
    follow_config: Mapping[str, object] = {}
    if isinstance(follow_config_raw, Mapping):
        follow_config = follow_config_raw

    ball_cam_config_raw = preset_config.get("ball_cam")
    ball_cam_config: dict[str, object] = {}
    if isinstance(ball_cam_config_raw, Mapping):
        ball_cam_config = dict(ball_cam_config_raw)

    if preset_key == "wide_follow":
        ball_cam_config.setdefault("ball_cam_enabled", True)
        ball_cam_config.setdefault("ball_cam_mode", "strict_lock")

        # Basic lock & smoothing parameters
        ball_cam_config.setdefault("ball_cam_lead_frames", 3)
        ball_cam_config.setdefault("ball_cam_smooth_window", 5)  # moving average window
        ball_cam_config.setdefault("ball_cam_max_speed_px", 120.0)  # generous; avoid visible lag

        # Margin: how much space between ball and crop edge (in pixels)
        ball_cam_config.setdefault("ball_cam_margin_px", 80.0)

        # Vertical bias: where to put the ball inside the portrait frame
        # 0.5 = dead center; 0.6 = slightly lower than center (more space above)
        ball_cam_config.setdefault("ball_cam_vertical_pos", 0.60)

    portrait_w = getattr(args, "portrait_w", None)
    portrait_h = getattr(args, "portrait_h", None)
    preset_portrait, portrait_min_box, portrait_horizon_lock = portrait_config_from_preset(
        preset_config.get("portrait")
    )
    portrait: Optional[Tuple[int, int]] = None
    if portrait_w is not None and portrait_h is not None:
        portrait = (portrait_w, portrait_h)
    elif args.portrait:
        portrait = parse_portrait(args.portrait)
        if portrait:
            portrait_w, portrait_h = portrait
    elif preset_portrait:
        portrait = preset_portrait
        portrait_w, portrait_h = portrait
    if portrait_w is not None:
        setattr(args, "portrait_w", portrait_w)
    if portrait_h is not None:
        setattr(args, "portrait_h", portrait_h)

    plan_lookahead_arg = getattr(args, "plan_lookahead", None)
    if plan_lookahead_arg is not None:
        lookahead = int(plan_lookahead_arg)
    else:
        lookahead = int(preset_config.get("lookahead", 18))
        if getattr(args, "_follow_lookahead_cli", False):
            lookahead = int(args.follow_lookahead)
    lookahead = max(0, lookahead)
    smoothing_default = preset_config.get("smoothing", 0.65)
    follow_smoothing = follow_config.get("smoothing") if follow_config else None
    smoothing = float(args.smoothing) if args.smoothing is not None else float(smoothing_default)
    pad = float(args.pad) if args.pad is not None else float(preset_config.get("pad", 0.22))
    speed_limit = float(args.speed_limit) if args.speed_limit is not None else float(preset_config.get("speed_limit", 480))
    zoom_min = float(args.zoom_min) if args.zoom_min is not None else float(preset_config.get("zoom_min", 1.0))
    zoom_max = float(args.zoom_max) if args.zoom_max is not None else float(preset_config.get("zoom_max", 2.2))

    ### PATCH START: Safe cy_frac initialization

    # Initialize cy_frac safely
    if hasattr(args, "cy_frac") and args.cy_frac is not None:
        cy_frac = float(args.cy_frac)
    else:
        # Default: keep subject ~55% down the frame (good for portrait soccer tracking)
        cy_frac = 0.55

    # Validate
    try:
        if not math.isfinite(cy_frac) or cy_frac <= 0 or cy_frac >= 1:
            cy_frac = 0.55
    except Exception:
        cy_frac = 0.55

    ### PATCH END

    if not math.isfinite(cy_frac):
        cy_frac = 0.46
    cy_frac = float(np.clip(cy_frac, 0.0, 1.0))
    crf = int(args.crf) if args.crf is not None else int(preset_config.get("crf", 19))
    keyint_factor = int(args.keyint_factor) if args.keyint_factor is not None else int(preset_config.get("keyint_factor", 4))

    controller_config_raw = follow_config.get("controller") if follow_config else None
    controller_config: Mapping[str, object] = {}
    if isinstance(controller_config_raw, Mapping):
        controller_config = controller_config_raw

    def _controller_value(key: str) -> Optional[object]:
        if key in controller_config:
            return controller_config[key]
        preset_key_name = f"follow_{key}"
        if preset_key_name in preset_config:
            return preset_config[preset_key_name]
        return None

    def _controller_float(name, default):
        """
        Safe float reader for controller parameters.
        Returns sanitized defaults when args or values are missing or invalid.
        """
        val = getattr(args, name, None)
        if val is None:
            return default

        try:
            x = float(val)
            if not math.isfinite(x):
                return default
            return x
        except Exception:
            return default

    def _controller_optional_float(key: str, fallback: Optional[float]) -> Optional[float]:
        value = _controller_value(key)
        if value is None:
            return fallback
        if isinstance(value, str) and value.strip().lower() in {"none", "", "null"}:
            return None

    def _controller_int(key: str, fallback: int) -> int:
        value = _controller_value(key)
        if value is None:
            return fallback

    follow_zeta = (
        float(args.follow_zeta)
        if args.follow_zeta is not None
        else _controller_float("zeta", FOLLOW_DEFAULTS["zeta"])
    )
    follow_wn = (
        float(args.follow_wn)
        if args.follow_wn is not None
        else _controller_float("wn", FOLLOW_DEFAULTS["wn"])
    )
    follow_deadzone = max(
        0.0,
        _safe(
            _controller_float("deadzone", FOLLOW_DEFAULTS["deadzone"]),
            FOLLOW_DEFAULTS["deadzone"],
        ),
    )
    follow_max_vel = (
        float(args.max_vel)
        if args.max_vel is not None
        else _controller_optional_float("max_vel", FOLLOW_DEFAULTS["max_vel"])
    )
    follow_max_acc = (
        float(args.max_acc)
        if args.max_acc is not None
        else _controller_optional_float("max_acc", FOLLOW_DEFAULTS["max_acc"])
    )
    # ---- SAFE LOOKAHEAD ----
    raw_lookahead = getattr(args, "follow_lookahead", None)

    try:
        follow_lookahead_value = float(raw_lookahead)
        if not math.isfinite(follow_lookahead_value):
            raise ValueError()
    except Exception:
        follow_lookahead_value = FOLLOW_DEFAULTS.get("lookahead", 3.0)

    follow_lookahead_frames = max(0, int(follow_lookahead_value))
    follow_pre_smooth = (
        float(np.clip(float(args.pre_smooth), 0.0, 1.0))
        if args.pre_smooth is not None
        else float(
            np.clip(
                _controller_float("pre_smooth", FOLLOW_DEFAULTS["pre_smooth"]),
                0.0,
                1.0,
            )
        )
    )

    margin_px = 0.0
    margin_val = follow_config.get("margin_px") if follow_config else None

    keepinview_margin = max(96.0, margin_px)
    keepinview_nudge = 0.6
    keepinview_zoom_gain = 0.55
    keepinview_zoom_cap = 1.8
    keepinview_cfg = follow_config.get("keepinview") if follow_config else None

    keepinview_margin_arg = getattr(args, "keepinview_margin", None)

    plan_override_data: Optional[dict[str, np.ndarray]] = None
    plan_override_len = 0
    plan_arg = getattr(args, "plan", None)
    if plan_arg:
        plan_path = Path(plan_arg).expanduser()
        if not plan_path.exists():
            raise FileNotFoundError(f"Plan file not found: {plan_path}")
        keyframes, _ = load_plan(plan_path)
        plan_override_data = keyframes_to_arrays(keyframes)
        plan_override_len = len(keyframes)
        logging.info(
            "Loaded camera plan %s (%s keyframes)",
            plan_path,
            plan_override_len,
        )
    keepinview_nudge_arg = getattr(args, "keepinview_nudge", None)
    keepinview_zoom_arg = getattr(args, "keepinview_zoom", None)
    keepinview_zoom_cap_arg = getattr(args, "keepinview_zoom_cap", None)
    keepinview_zoom_cap_override = False

    zoom_out_max_default = follow_config.get("zoom_out_max") if follow_config else None
    follow_zoom_out_max = 1.35
    if getattr(args, "zoom_out_max", None) is not None:
        follow_zoom_out_max = max(1.0, float(args.zoom_out_max))

    if not keepinview_zoom_cap_override:
        keepinview_zoom_cap = min(keepinview_zoom_cap, follow_zoom_out_max)
    keepinview_zoom_cap = max(1.0, float(keepinview_zoom_cap))

    zoom_edge_frac_default = follow_config.get("zoom_edge_frac") if follow_config else None
    follow_zoom_edge_frac = 0.80
    if getattr(args, "zoom_edge_frac", None) is not None:
        follow_zoom_edge_frac = float(args.zoom_edge_frac)
    if not math.isfinite(follow_zoom_edge_frac) or follow_zoom_edge_frac <= 0.0:
        follow_zoom_edge_frac = 1.0

    lost_hold_ms = getattr(args, "lost_hold_ms", 500)
    lost_pan_ms = getattr(args, "lost_pan_ms", 1200)
    lost_lookahead_s = getattr(args, "lost_lookahead_s", 6.0)
    lost_chase_motion_ms = getattr(args, "lost_chase_motion_ms", 900)
    lost_motion_thresh = getattr(args, "lost_motion_thresh", 1.6)
    lost_use_motion = bool(getattr(args, "lost_use_motion", False))

    lead_time_s = 0.0
    lead_val = follow_config.get("lead_time") if follow_config else None
    lead_frames = int(round(lead_time_s * fps_out)) if fps_out > 0 else 0

    speed_zoom_value = follow_config.get("speed_zoom") if follow_config else None
    speed_zoom_config = speed_zoom_value if isinstance(speed_zoom_value, Mapping) else None

    default_ball_key_x = "bx_stab"
    default_ball_key_y = "by_stab"
    keys_value = follow_config.get("keys") if follow_config else None

    if not getattr(args, "ball_key_x", None):
        setattr(args, "ball_key_x", default_ball_key_x)
    if not getattr(args, "ball_key_y", None):
        setattr(args, "ball_key_y", default_ball_key_y)

    output_path = Path(args.out) if args.out else _default_output_path(original_source_path, preset_key)
    output_path = output_path.expanduser().resolve()

    labels_root = args.labels_root or "out/yolo"
    label_files = find_label_files(original_source_path.stem, labels_root)

    log_dict: dict[str, object] = {}

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError("Unable to open input video for metadata extraction.")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()

    if duration_s is None:
        duration_s = (frame_count / fps_in) if (frame_count and fps_in and fps_in > 0) else 0.0

    if duration_s <= 0 and frame_count and frame_count > 0 and fps_in and fps_in > 0:
        duration_s = frame_count / float(fps_in)
    if duration_s <= 0 and frame_count > 0:
        fallback_fps = fps_in if fps_in > 0 else 30.0
        duration_s = frame_count / float(fallback_fps)

    override_samples = getattr(args, "follow_override_samples", None)
    total_frames = frame_count
    if total_frames <= 0:
        fps_hint = fps_in if fps_in > 0 else fps_out
        total_frames = int(round((duration_s or 0.0) * fps_hint)) if fps_hint > 0 else 0
    total_frames = max(total_frames, 1)

    renderer: Optional[Renderer]
    ball_samples: List[BallSample] = []
    keep_path_lookup_data: dict[int, Tuple[float, float]] = {}
    keepinview_path: list[tuple[float, float]] | None = None
    follow_telemetry_path: str | None = None
    use_ball_telemetry = bool(getattr(args, "use_ball_telemetry", False))
    telemetry_path: Path | None = None
    offline_ball_path: Optional[List[Optional[dict[str, float]]]] = None

    follow_exact_flag = bool(getattr(args, "follow_exact", False))
    if follow_exact_flag:
        print("[DEBUG] follow-exact mode active (controller disabled; using follow_override only)")
    disable_controller = follow_exact_flag or bool(getattr(args, "disable_controller", False))
    follow_override_map: Optional[Mapping[int, Mapping[str, float]]] = None
    if args.follow_override:
        follow_override_map = {}
        with open(args.follow_override, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                frame = int(row["frame"])
                follow_override_map[frame] = {
                    "t": float(row.get("t", frame / fps_in)),
                    "cx": float(row["cx"]),
                    "cy": float(row["cy"]),
                    "zoom": float(row.get("zoom", 1.0)),
                }
    elif override_samples:
        follow_override_map = {}
        for row in override_samples:
            frame = int(row["frame"])
            follow_override_map[frame] = {
                "t": float(row.get("t", frame / fps_in)),
                "cx": float(row["cx"]),
                "cy": float(row["cy"]),
                "zoom": float(row.get("zoom", 1.0)),
            }
    elif "follow_frames" in locals() and len(follow_frames) > 0:
        follow_override_map = {
            int(f["frame"]): {
                "t": float(f.get("t", int(f["frame"]) / fps_in)),
                "cx": float(f["cx"]),
                "cy": float(f["cy"]),
                "zoom": float(f.get("zoom", 1.0)),
            }
            for f in follow_frames
            if f.get("frame") is not None
        }

    if follow_override_map is not None:
        keep_path_lookup_data = {}
        use_ball_telemetry = False

    if override_samples:
        use_ball_telemetry = False

    if getattr(args, "telemetry", None):
        telemetry_rows = load_any_telemetry(args.telemetry)

        # === FOLLOW TELEMETRY NORMALIZATION (NEW) ===
        follow_frames = []
        for row in telemetry_rows:
            if not row.get("valid", True):
                continue
            cx = row.get("cx")
            cy = row.get("cy")
            zoom = row.get("zoom", 1.0)
            t = row.get("t")
            frame = row.get("frame")

            # Reject unusable rows
            if cx is None or cy is None or t is None:
                continue

            follow_frames.append({
                "t": float(t),
                "frame": int(frame) if frame is not None else None,
                "cx": float(cx),
                "cy": float(cy),
                "zoom": float(zoom),
            })

        if not follow_frames:
            raise ValueError(f"No valid follow telemetry in {args.telemetry}")

    telemetry_coverage = 0
    telemetry_coverage_ratio = 0.0
    ball_cam_stats: dict[str, float] = {}
    if use_ball_telemetry:
        telemetry_in = getattr(args, "ball_telemetry", None)
        telemetry_path = Path(telemetry_in).expanduser() if telemetry_in else Path(telemetry_path_for_video(input_path))
        if telemetry_path.is_file():
            set_telemetry_frame_bounds(width, height)
            total_frames = frame_count
            if total_frames <= 0:
                fps_hint = fps_in if fps_in > 0 else fps_out
                total_frames = int(round((duration_s or 0.0) * fps_hint)) if fps_hint > 0 else 0
            total_frames = max(total_frames, 1)

            plan_data, ball_cam_stats = build_ball_cam_plan(
                telemetry_path,
                num_frames=total_frames,
                fps=fps_in if fps_in > 0 else fps_out,
                frame_width=width,
                frame_height=height,
                portrait_width=int(portrait[0]) if portrait else 1080,
                config=ball_cam_config,
                preset_name=preset_key,
                in_path=input_path,
                out_path=output_path,
            )
            telemetry_coverage_ratio = float(ball_cam_stats.get("coverage", 0.0))
            telemetry_coverage = int(round(telemetry_coverage_ratio * total_frames))
            if plan_data is None:
                use_ball_telemetry = False
                keep_path_lookup_data = {}
            else:
                cx_vals = plan_data.get("cx")
                cy_vals = plan_data.get("cy")
                if cx_vals is not None and cy_vals is not None:
                    keep_path_lookup_data = {
                        idx: (float(cx_vals[idx]), float(cy_vals[idx])) for idx in range(len(cx_vals))
                    }
                if plan_override_data is None:
                    plan_override_data = {k: v for k, v in plan_data.items() if k in {"x0", "y0", "w", "h", "spd", "z"}}
                    plan_override_len = len(next(iter(plan_data.values()))) if plan_data else 0
                if telemetry_path is not None:
                    cx_vals = plan_data.get("cx")
                    cy_vals = plan_data.get("cy")
                    zoom_vals = plan_data.get("z")
                    if cx_vals is not None and cy_vals is not None and zoom_vals is not None:
                        follow_telemetry_path = emit_follow_telemetry(
                            telemetry_path,
                            cx_vals,
                            cy_vals,
                            zoom_vals,
                            basename=telemetry_path.stem,
                        )
                        follow_telemetry_path = smooth_follow_telemetry(follow_telemetry_path)
                        log_dict["follow_telemetry"] = follow_telemetry_path
                ball_samples = load_ball_telemetry(telemetry_path)
                jerk_stat = ball_cam_stats.get("jerk95_cam", ball_cam_stats.get("jerk95", 0.0))
                jerk_raw = ball_cam_stats.get("jerk95_raw", 0.0)
                ball_in_crop_pct = ball_cam_stats.get("ball_in_crop_pct")
                ball_in_crop_frames = int(ball_cam_stats.get("ball_in_crop_frames", 0))
                telemetry_conf = float(ball_cam_stats.get("telemetry_conf", 1.0))
                telemetry_quality = float(ball_cam_stats.get("telemetry_quality", 1.0))
                cx_range = (float(np.nanmin(plan_data.get("cx", np.array([0.0])))), float(np.nanmax(plan_data.get("cx", np.array([0.0])))))
                logger.info(
                    "[BALL-CAM] coverage: %.1f%% (conf=%.2f, quality=%.2f), path: %d frames, cx range=[%.1f, %.1f], jerk95_raw=%.1f, jerk95_cam=%.1f px/s^3",
                    telemetry_coverage_ratio * 100.0,
                    telemetry_conf,
                    telemetry_quality,
                    len(plan_data.get("cx", [])),
                    cx_range[0],
                    cx_range[1],
                    jerk_raw,
                    jerk_stat,
                )
                if telemetry_quality < 0.6:
                    logger.info("[FORCE-BALL-FOLLOW] Using ball telemetry even if weak")
                if ball_in_crop_pct is not None and plan_data is not None:
                    logger.info(
                        "[BALL-CAM] ball_in_crop_horiz: %.1f%% (%d/%d)",
                        ball_in_crop_pct,
                        ball_in_crop_frames,
                        len(plan_data.get("cx", [])),
                    )
        else:
            logger.info("[BALL-CAM] No ball telemetry found for %s; reactive follow", input_path)
            use_ball_telemetry = False

    raw_points = load_labels(label_files, width, height, fps_in)
    log_dict["labels_raw_count"] = len(raw_points)
    if raw_points:
        max_label_time = max(point[0] for point in raw_points)
        if duration_s <= max_label_time:
            frame_step = 1.0 / float(fps_in) if fps_in > 0 else 0.0
            duration_s = max_label_time + frame_step

    label_pts = resample_labels_by_time(raw_points, fps_out, duration_s)

    def _rng(arr):
        xs = [a[1] for a in arr]
        ys = [a[2] for a in arr]
        return (min(xs), max(xs), min(ys), max(ys)) if arr else None

    log_dict["labels_resampled_count"] = len(label_pts)
    log_dict["labels_resampled_range"] = _rng(label_pts)

    positions, used_mask = labels_to_positions(label_pts, fps_out, duration_s, raw_points)

    if len(positions) == 0 and frame_count > 0 and fps_out > 0:
        target_frames = int(round(frame_count * (fps_out / float(fps_in if fps_in > 0 else fps_out))))
        target_frames = max(target_frames, frame_count)
        positions = np.full((target_frames, 2), np.nan, dtype=np.float32)
        used_mask = np.zeros(target_frames, dtype=bool)

    if args.flip180 and len(positions) > 0:
        flipped_positions = positions.copy()
        valid_mask = ~np.isnan(flipped_positions).any(axis=1)
        if valid_mask.any():
            flipped_positions[valid_mask, 0] = float(width) - flipped_positions[valid_mask, 0]
            flipped_positions[valid_mask, 1] = float(height) - flipped_positions[valid_mask, 1]
        positions = flipped_positions

    planner = CameraPlanner(
        width=width,
        height=height,
        fps=fps_out,
        lookahead=lookahead,
        smoothing=smoothing,
        pad=pad,
        speed_limit=speed_limit,
        zoom_min=zoom_min,
        zoom_max=zoom_max,
        portrait=portrait,
        margin_px=margin_px,
        lead_frames=lead_frames,
        speed_zoom=speed_zoom_config,
        min_box=portrait_min_box,
        horizon_lock=portrait_horizon_lock,
        emergency_gain=getattr(args, "emergency_gain", 0.6),
        emergency_zoom_max=getattr(args, "emergency_zoom_max", 1.45),
        keepinview_margin_px=keepinview_margin,
        keepinview_nudge_gain=keepinview_nudge,
        keepinview_zoom_gain=keepinview_zoom_gain,
        keepinview_zoom_out_max=keepinview_zoom_cap,
        center_frac=cy_frac,
    )
    states = planner.plan(positions, used_mask)

    temp_root = Path("out/autoframe_work")
    temp_dir = temp_root / preset_key / original_source_path.stem
    _prepare_temp_dir(temp_dir, args.clean_temp)

    brand_overlay_path = Path(args.brand_overlay).expanduser() if args.brand_overlay else None
    endcard_path = Path(args.endcard).expanduser() if args.endcard else None
    offline_ball_path: Optional[List[Optional[dict[str, float]]]] = None
    if getattr(args, "ball_path", None):
        ball_path_file = Path(args.ball_path).expanduser()
        if not ball_path_file.exists():
            raise FileNotFoundError(f"Ball path file not found: {ball_path_file}")
        offline_ball_path = load_ball_path(
            ball_path_file,
            ball_key_x=str(getattr(args, "ball_key_x", "bx_stab")),
            ball_key_y=str(getattr(args, "ball_key_y", "by_stab")),
        )
    elif use_ball_telemetry and ball_samples:
        max_frame_idx = max(
            (
                int(frame_idx)
                for frame_idx in (getattr(s, "frame", None) for s in ball_samples)
                if frame_idx is not None and isinstance(frame_idx, (int, float))
            ),
            default=0,
        )
        total_frames = max(len(states), frame_count, max_frame_idx + 1)
        path: list[Optional[dict[str, float]]] = [None] * total_frames
        for sample in ball_samples:
            bx_val = _safe_float(getattr(sample, "x", None))
            by_val = _safe_float(getattr(sample, "y", None))
            if bx_val is None or by_val is None:
                continue
            if idx < 0:
                continue
            if idx >= len(path):
                path.extend([None] * (idx - len(path) + 1))
            path[idx] = {"bx": float(bx_val), "by": float(by_val)}
        offline_ball_path = path

        if portrait:
            portrait_w, portrait_h = portrait
            plan_config = PlannerConfig(
                frame_size=(float(width), float(height)),
                crop_aspect=float(portrait_w) / float(portrait_h),
                fps=float(fps_out) if fps_out else float(fps_in) if fps_in else 30.0,
                keep_in_frame_frac_x=(0.4, 0.6),
                keep_in_frame_frac_y=(0.4, 0.6),
                min_zoom=float(zoom_min),
                max_zoom=float(zoom_max),
            )
            planned = plan_ball_portrait_crop(
                ball_samples,
                src_w=width,
                src_h=height,
                portrait_w=int(portrait_w),
                portrait_h=int(portrait_h),
                config=plan_config,
            )
            if planned and not keep_path_lookup_data:
                keep_path_lookup_data = {frame: (cx, cy) for frame, (cx, cy, _z) in planned.items()}
    if not use_ball_telemetry:
        keep_path_lookup_data = {}

    debug_ball_overlay = bool(getattr(args, "debug_ball_overlay", False) and use_ball_telemetry)

    telemetry_file: Optional[TextIO] = None
    telemetry_simple_file: Optional[TextIO] = None
    if render_telemetry_path:
        telemetry_file = render_telemetry_path.open("w", encoding="utf-8")
    if render_telemetry_simple_path:
        telemetry_simple_file = render_telemetry_simple_path.open("w", encoding="utf-8")

    jerk95 = 0.0
    renderer = Renderer(
        input_path=input_path,
        output_path=output_path,
        temp_dir=temp_dir,
        fps_in=fps_in,
        fps_out=fps_out,
        flip180=args.flip180,
        portrait=portrait,
        brand_overlay=brand_overlay_path,
        endcard=endcard_path,
        pad=pad,
        zoom_min=zoom_min,
        zoom_max=zoom_max,
        speed_limit=speed_limit,
        telemetry=telemetry_file,
        telemetry_simple=telemetry_simple_file,
        init_manual=getattr(args, "init_manual", False),
        init_t=getattr(args, "init_t", 0.8),
        ball_path=offline_ball_path,
        follow_lead_time=lead_time_s,
        follow_margin_px=keepinview_margin,
        follow_smoothing=smoothing,
        follow_zeta=follow_zeta,
        follow_wn=follow_wn,
        follow_deadzone=follow_deadzone,
        follow_max_vel=follow_max_vel,
        follow_max_acc=follow_max_acc,
        follow_lookahead=follow_lookahead_frames,
        follow_pre_smooth=follow_pre_smooth,
        follow_zoom_out_max=follow_zoom_out_max,
        follow_zoom_edge_frac=follow_zoom_edge_frac,
        follow_center_frac=cy_frac,
        lost_hold_ms=lost_hold_ms,
        lost_pan_ms=lost_pan_ms,
        lost_lookahead_s=lost_lookahead_s,
        lost_chase_motion_ms=lost_chase_motion_ms,
        lost_motion_thresh=lost_motion_thresh,
        lost_use_motion=lost_use_motion,
        portrait_plan_margin_px=getattr(args, "portrait_plan_margin", None),
        portrait_plan_headroom=getattr(args, "portrait_plan_headroom", None),
        portrait_plan_lead_px=getattr(args, "portrait_plan_lead", None),
        plan_override_data=plan_override_data,
        plan_override_len=plan_override_len,
        ball_samples=ball_samples,
        keep_path_lookup_data=keep_path_lookup_data,
        debug_ball_overlay=debug_ball_overlay,
        follow_override=follow_override_map,
        follow_exact=follow_exact_flag,
        disable_controller=disable_controller,
        follow_trajectory=None,
    )
    jerk95 = renderer.write_frames(states)

    assert renderer is not None

    keyint = max(1, int(round(float(keyint_factor) * float(fps_out))))
    log_path = Path(args.log).expanduser() if args.log else None
    renderer.ffmpeg_stitch(crf=crf, keyint=keyint, log_path=log_path)

    log_dict["jerk95"] = float(jerk95)

    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "input": os.fspath(input_path),
            "output": os.fspath(output_path),
            "fps_in": float(fps_in),
            "fps_out": float(fps_out),
            "labels_found": int(len(raw_points)),
            "preset": preset_key,
            "ffmpeg_command": renderer.last_ffmpeg_command,
        }
        summary["jerk95"] = float(jerk95)
        summary.update(log_dict)
        with log_path.open("w", encoding="utf-8") as handle:
            json.dump(to_jsonable(log_dict), handle, ensure_ascii=False, indent=2)
            handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified cinematic ball-follow renderer")
    parser.add_argument("--in", dest="in_path", required=True, help="Input MP4 path")
    parser.add_argument("--src", dest="src", help="Legacy compatibility input path (ignored)")
    parser.add_argument("--out", dest="out", help="Output MP4 path")
    parser.add_argument("--preset", dest="preset", default="cinematic", help="Preset name to load from render_presets.yaml")
    parser.add_argument("--portrait", dest="portrait", help="Portrait canvas WxH")
    parser.add_argument(
        "--portrait-plan-margin",
        dest="portrait_plan_margin",
        type=float,
        help="Margin in px for offline portrait planner keep-in-frame band",
    )
    parser.add_argument(
        "--portrait-plan-headroom",
        dest="portrait_plan_headroom",
        type=float,
        help="Headroom fraction override for offline portrait planner",
    )
    parser.add_argument(
        "--portrait-plan-lead",
        dest="portrait_plan_lead",
        type=float,
        help="Lead distance (px) for offline portrait planner",
    )
    parser.add_argument(
        "--upscale",
        action="store_true",
        help="Upscale source with Real-ESRGAN (or ffmpeg fallback) before processing.",
    )
    parser.add_argument(
        "--upscale-scale",
        type=int,
        default=2,
        help="Scale factor for upscaling (default: 2).",
    )
    parser.add_argument("--fps", dest="fps", type=float, help="Output FPS")
    parser.add_argument("--flip180", dest="flip180", action="store_true", help="Rotate frames by 180 degrees before processing")
    parser.add_argument("--labels-root", dest="labels_root", help="Root directory containing YOLO label shards")
    parser.add_argument("--clean-temp", dest="clean_temp", action="store_true", help="Remove temporary frame folder before rendering")
    parser.add_argument(
        "--lookahead",
        "--follow-lookahead",
        dest="follow_lookahead",
        type=int,
        default=None,
        help="frames to look ahead when following (defaults to preset)",
    )
    parser.add_argument(
        "--follow-zeta",
        dest="follow_zeta",
        type=float,
        default=None,
        help="2nd-order damping ratio (>=1 is overdamped; defaults to preset)",
    )
    parser.add_argument(
        "--follow-wn",
        dest="follow_wn",
        type=float,
        default=None,
        help="2nd-order natural freq (rad/s; defaults to preset)",
    )
    parser.add_argument(
        "--deadzone",
        dest="deadzone",
        type=float,
        default=None,
        help="pixels; ignore target error inside this radius (defaults to preset)",
    )
    parser.add_argument(
        "--max-vel",
        dest="max_vel",
        type=float,
        default=None,
        help="px/s clamp on camera velocity (defaults to preset)",
    )
    parser.add_argument(
        "--max-acc",
        dest="max_acc",
        type=float,
        default=None,
        help="px/s^2 clamp on camera acceleration (defaults to preset)",
    )
    parser.add_argument(
        "--pre-smooth",
        dest="pre_smooth",
        type=float,
        default=None,
        help="EMA alpha to pre-smooth bx/by (0..1; defaults to preset)",
    )
    parser.add_argument(
        "--jerk-threshold",
        dest="jerk_threshold",
        type=float,
        default=0.0,
        help="Max allowed camera jerk95 in px/s^3 (0 disables the gate)",
    )
    parser.add_argument(
        "--jerk-wn-scale",
        dest="jerk_wn_scale",
        type=float,
        default=0.9,
        help="Multiplier applied to --follow-wn when jerk exceeds the threshold",
    )
    parser.add_argument(
        "--jerk-deadzone-step",
        dest="jerk_deadzone_step",
        type=float,
        default=2.0,
        help="Deadzone increment (px) when jerk exceeds the threshold",
    )
    parser.add_argument(
        "--jerk-max-attempts",
        dest="jerk_max_attempts",
        type=int,
        default=3,
        help="Maximum number of retune attempts when enforcing jerk threshold",
    )
    parser.add_argument(
        "--plan-lookahead",
        dest="plan_lookahead",
        type=int,
        help="Frames of lookahead for planning",
    )
    parser.add_argument("--smoothing", dest="smoothing", type=float, help="EMA smoothing factor")
    parser.add_argument("--pad", dest="pad", type=float, help="Edge padding ratio used to derive zoom")
    parser.add_argument("--speed-limit", dest="speed_limit", type=float, help="Maximum pan speed in px/sec")
    parser.add_argument("--zoom-min", dest="zoom_min", type=float, help="Minimum zoom multiplier")
    parser.add_argument("--zoom-max", dest="zoom_max", type=float, help="Maximum zoom multiplier")
    parser.add_argument(
        "--zoom-out-max",
        dest="zoom_out_max",
        type=float,
        default=None,
        help="Maximum automatic zoom-out multiplier",
    )
    parser.add_argument(
        "--zoom-edge-frac",
        dest="zoom_edge_frac",
        type=float,
        default=0.80,
        help="Fraction of safe margin where zoom-out begins to ease out",
    )
    parser.add_argument(
        "--cy-frac",
        dest="cy_frac",
        type=float,
        default=0.46,
        help=(
            "Desired ball vertical position as a fraction of frame height (0=top, 1=bottom). "
            "Default 0.46 puts ball slightly above center to avoid bottom-edge saturation."
        ),
    )
    parser.add_argument(
        "--emergency-gain",
        dest="emergency_gain",
        type=float,
        default=0.6,
        help="Emergency recenter gain when the ball breaches the safety margin",
    )
    parser.add_argument(
        "--emergency-zoom-max",
        dest="emergency_zoom_max",
        type=float,
        default=1.45,
        help="Maximum emergency zoom-out multiplier to keep the ball in view",
    )
    parser.add_argument(
        "--lost-hold-ms",
        type=int,
        default=500,
        help="when ball leaves frame, hold the last camera center this long (ms)",
    )
    parser.add_argument(
        "--lost-pan-ms",
        type=int,
        default=1200,
        help="duration of the slow pan toward re-entry (ms)",
    )
    parser.add_argument(
        "--lost-chase-motion-ms",
        type=int,
        default=900,
        help="during LOST, time to pan toward motion centroid before re-entry (ms)",
    )
    parser.add_argument(
        "--lost-motion-thresh",
        type=float,
        default=1.6,
        help="optical flow magnitude threshold (px/frame) to pick 'action' blobs",
    )
    parser.add_argument(
        "--lost-use-motion",
        action="store_true",
        help="enable motion centroid chase while ball is out",
    )
    parser.add_argument(
        "--lost-lookahead-s",
        type=float,
        default=6.0,
        help="search window ahead to find when ball returns inside",
    )
    parser.add_argument(
        "--keepinview-margin",
        dest="keepinview_margin",
        type=float,
        help="Safety band in pixels for the keep-in-view guard (defaults to follow margin)",
    )
    parser.add_argument(
        "--keepinview-nudge",
        dest="keepinview_nudge",
        type=float,
        help="Proportional gain for keep-in-view recenter nudges",
    )
    parser.add_argument(
        "--keepinview-zoom",
        dest="keepinview_zoom",
        type=float,
        help="Gain controlling keep-in-view adaptive zoom-out strength",
    )
    parser.add_argument(
        "--keepinview-zoom-cap",
        dest="keepinview_zoom_cap",
        type=float,
        help="Maximum keep-in-view zoom-out multiplier (defaults to follow zoom limit)",
    )
    parser.add_argument(
        "--telemetry",
        "--ball-telemetry",
        dest="telemetry",
        help="Input ball telemetry JSONL (default: out/telemetry/<clip>.ball.jsonl)",
    )
    parser.add_argument(
        "--use-ball-telemetry",
        action="store_true",
        help="Enable ball-aware portrait planning when telemetry is available",
    )
    parser.add_argument(
        "--debug-ball-overlay",
        action="store_true",
        help="Draw detected ball markers before cropping (requires --use-ball-telemetry)",
    )
    parser.add_argument("--render-telemetry", dest="render_telemetry", help="Output JSONL telemetry file")
    parser.add_argument(
        "--plan",
        dest="plan",
        help="Optional camera plan JSON (skips the internal planner)",
    )
    parser.add_argument(
        "--telemetry-out",
        dest="telemetry_out",
        default=None,
        help="Write per-frame JSONL with t,cx,cy,bx,by",
    )
    parser.add_argument("--brand-overlay", dest="brand_overlay", help="PNG overlay composited on every frame")
    parser.add_argument("--endcard", dest="endcard", help="Optional endcard image displayed for ~2 seconds")
    parser.add_argument("--log", dest="log", help="Optional render log path")
    parser.add_argument("--crf", dest="crf", type=int, help="Override CRF value")
    parser.add_argument("--keyint-factor", dest="keyint_factor", type=int, help="Override keyint factor")
    parser.add_argument(
        "--ball-key-x",
        dest="ball_key_x",
        default=None,
        help="Preferred X key when reading planned ball path JSONL",
    )
    parser.add_argument(
        "--ball-key-y",
        dest="ball_key_y",
        default=None,
        help="Preferred Y key when reading planned ball path JSONL",
    )
    parser.add_argument(
        "--init-manual",
        dest="init_manual",
        action="store_true",
        help="Manually select the ball ROI at the start of the clip.",
    )
    parser.add_argument(
        "--init-t",
        dest="init_t",
        type=float,
        default=0.8,
        help="Time in seconds to show for manual init (default 0.8).",
    )
    parser.add_argument(
        "--ball-path",
        dest="ball_path",
        help="JSONL from ball_path_planner_v2.py",
    )
    parser.add_argument(
        "--follow-override",
        dest="follow_override",
        type=str,
        default=None,
        help="Path to a JSONL file with explicit per-frame cx/cy/zoom overrides.",
    )
    parser.add_argument(
        "--follow-exact",
        action="store_true",
        help="Use override follow telemetry exactly, bypassing controller and smoothing.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    print("[DEBUG] main() entered")
    parser = build_parser()
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(argv)
    print("[DEBUG] args parsed OK")
    override_samples = None
    if args.follow_override:
        print("[DEBUG] follow_override =", args.follow_override)
        override_samples = []
        with open(args.follow_override, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                # Must contain: frame, cx, cy, zoom
                if "frame" in row and "cx" in row and "cy" in row:
                    override_samples.append(row)
    follow_lookahead_cli = any(
        arg == "--lookahead" or arg.startswith("--lookahead=") for arg in raw_argv
    )
    setattr(args, "_follow_lookahead_cli", follow_lookahead_cli)
    # --- portrait helpers ---
    portrait_w, portrait_h = (None, None)
    setattr(args, "portrait_w", portrait_w)
    setattr(args, "portrait_h", portrait_h)
    render_telemetry_path: Optional[Path] = None
    telemetry_simple_path: Optional[Path] = None
    if getattr(args, "telemetry", None):
        setattr(args, "ball_telemetry", args.telemetry)
    if getattr(args, "render_telemetry", None):
        render_telemetry_path = Path(args.render_telemetry).expanduser()
        render_telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        args.render_telemetry = os.fspath(render_telemetry_path)
    if getattr(args, "telemetry_out", None):
        telemetry_simple_path = Path(args.telemetry_out).expanduser()
        telemetry_simple_path.parent.mkdir(parents=True, exist_ok=True)
        args.telemetry_out = os.fspath(telemetry_simple_path)

    if getattr(args, "use_ball_telemetry", False) and not getattr(args, "ball_telemetry", None):
        args.ball_telemetry = telemetry_path_for_video(Path(args.input))
    setattr(args, "follow_override_samples", override_samples)
    run(args, telemetry_path=render_telemetry_path, telemetry_simple_path=telemetry_simple_path)
    print("[DEBUG] main() reached end")


if __name__ == "__main__":
    import traceback

    try:
        main()
    except SystemExit:
        # allow argparse / sys.exit() to behave normally
        raise
    except Exception as exc:
        print("\n[FATAL] Unhandled exception in render_follow_unified.py:")
        print(f"  {exc!r}")
        traceback.print_exc()
        # re-raise so the process has a non-zero exit code
        raise



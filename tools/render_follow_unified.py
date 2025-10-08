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
from typing import List, Optional, Sequence, TextIO, Tuple

import cv2
import numpy as np


class CamFollow2O:
    def __init__(self, zeta=0.95, wn=6.0, dt=1 / 30):
        self.z = zeta
        self.w = wn
        self.dt = dt
        self.cx = 0.0
        self.cy = 0.0
        self.vx = 0.0
        self.vy = 0.0

    def step(self, target_x: float, target_y: float) -> tuple[float, float]:
        dt = self.dt
        w = self.w
        z = self.z
        ax = w * w * (target_x - self.cx) - 2 * z * w * self.vx
        ay = w * w * (target_y - self.cy) - 2 * z * w * self.vy
        self.vx += ax * dt
        self.vy += ay * dt
        self.cx += self.vx * dt
        self.cy += self.vy * dt
        return self.cx, self.cy


# --- Simple constant-velocity tracker (EMA-based Kalman-lite) ---
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


def _round_i(x):  # robust int rounding
    try:
        return int(round(float(x)))
    except Exception:
        return int(x)


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
    # side may be float → force int and keep odd size for better centering
    side_i = max(3, _round_i(side) | 1)
    x = _round_i(bx) - side_i // 2
    y = _round_i(by) - side_i // 2
    return _clamp_roi(x, y, side_i, side_i, W, H)


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
    try:
        return float(obj)
    except Exception:
        return str(obj)


def compute_portrait_crop(cx, cy, zoom, src_w, src_h, portrait_w, portrait_h, pad):
    # target aspect (w/h)
    t_aspect = float(portrait_w) / float(portrait_h) if portrait_w and portrait_h else (src_w / float(src_h))

    # derive crop size from zoom while honoring aspect
    crop_h = src_h / float(zoom)
    crop_w = crop_h * t_aspect
    if crop_w > src_w:  # bound if too wide
        crop_w = float(src_w)
        crop_h = crop_w / t_aspect

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


def ffprobe_fps(path: Path) -> float:
    """Return the floating-point FPS using ffprobe."""

    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:  # pragma: no cover - execution context dependant
        raise RuntimeError(
            "Failed to read FPS using ffprobe. Ensure ffmpeg is installed and on PATH."
        ) from exc

    value = result.stdout.strip()
    if not value:
        raise RuntimeError("ffprobe did not return a frame rate value.")

    if "/" in value:
        num, den = value.split("/", 1)
        den_value = float(den)
        if den_value == 0:
            return float(num)
        return float(num) / den_value
    return float(value)


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
    try:
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "Failed to read duration using ffprobe. Ensure ffmpeg is installed and on PATH."
        ) from exc

    value = result.stdout.strip()
    if not value:
        raise RuntimeError("ffprobe did not return a duration value.")

    try:
        return float(value)
    except ValueError as exc:
        raise RuntimeError(f"Unable to parse ffprobe duration output: {value}") from exc


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
                try:
                    frame_idx = int(float(parts[0]))
                    x_value = float(parts[1])
                    y_value = float(parts[2])
                except Exception:
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

        base_side = min(self.width, self.height)
        base_side = max(1.0, base_side)
        target_final_side = max(base_side * (1.0 - 2.0 * self.pad), base_side * 0.35)
        shrink_factor = 1.0
        if self.pad > 0.0:
            shrink_factor = max(0.05, 1.0 - 2.0 * self.pad)
        pre_pad_target = target_final_side / shrink_factor
        desired_zoom = base_side / max(pre_pad_target, 1.0)
        self.base_zoom = float(np.clip(desired_zoom, self.zoom_min, self.zoom_max))

    def plan(self, positions: np.ndarray, used_mask: np.ndarray) -> List[CamState]:
        frame_count = len(positions)
        states: List[CamState] = []
        prev_cx = self.width / 2.0
        prev_cy = self.height / 2.0
        prev_zoom = self.base_zoom
        fallback_center = np.array([prev_cx, self.height * 0.45], dtype=np.float32)
        fallback_alpha = 0.05
        render_fps = self.fps if self.fps > 0 else 30.0
        px_per_sec_x = self.speed_limit * 1.35
        px_per_sec_y = self.speed_limit * 0.90
        pxpf_x = px_per_sec_x / render_fps if render_fps > 0 else 0.0
        pxpf_y = px_per_sec_y / render_fps if render_fps > 0 else 0.0

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

        def _compute_crop(
            center_x: float,
            center_y: float,
            zoom_value: float,
        ) -> Tuple[float, float, float, float, float, float, float, bool]:
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

            crop_w = float(np.clip(crop_w, 1.0, self.width))
            crop_h = float(np.clip(crop_h, 1.0, self.height))

            adjusted_cy = center_y
            if aspect_target:
                adjusted_cy = adjusted_cy + 0.10 * crop_h

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
            else:
                fallback_target = np.array([self.width / 2.0, self.height * 0.40], dtype=np.float32)
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

            target_zoom = self.base_zoom

            cx = self.smoothing * float(target[0]) + (1.0 - self.smoothing) * prev_cx
            cy = self.smoothing * float(target[1]) + (1.0 - self.smoothing) * prev_cy
            zoom = self.smoothing * target_zoom + (1.0 - self.smoothing) * prev_zoom

            ball_point: Optional[Tuple[float, float]] = None
            if has_position:
                bx = float(pos[0])
                by = float(pos[1])
                ball_point = (bx, by)
                follow_gain_x = 0.55
                follow_gain_y = 0.35
                cx = cx * (1.0 - follow_gain_x) + bx * follow_gain_x
                cy = cy * (1.0 - follow_gain_y) + by * follow_gain_y

            clamp_flags: List[str] = []

            cx, x_clamped = _clamp_axis(prev_cx, cx, pxpf_x)
            cy, y_clamped = _clamp_axis(prev_cy, cy, pxpf_y)
            speed_limited = x_clamped or y_clamped
            if speed_limited:
                clamp_flags.append("speed")

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

            if bounds_clamped:
                clamp_flags.append("bounds")

            prev_cx = actual_cx
            prev_cy = actual_cy
            prev_zoom = zoom

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
        init_manual: bool,
        init_t: float,
        ball_path: Optional[Sequence[Tuple[float, float, float]]],
    ) -> None:
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
        self.last_ffmpeg_command: Optional[List[str]] = None
        self.init_manual = bool(init_manual)
        self.init_t = float(init_t)
        self.offline_ball_path = list(ball_path) if ball_path else None

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

    def write_frames(self, states: Sequence[CamState]) -> None:
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

        portrait_w = output_size[0]
        portrait_h = output_size[1]

        overlay_image = _load_overlay(self.brand_overlay_path, output_size)
        tf = self.telemetry

        offline_ball_path = self.offline_ball_path

        cam = [(state.cx, state.cy, state.zoom) for state in states]
        if cam:
            cx_values = [value[0] for value in cam]
            cy_values = [value[1] for value in cam]
        else:
            cx_values = []
            cy_values = []

        frame_count = len(states)
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
        speed_px_sec = float(self.speed_limit or 3000.0)

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
        follower = CamFollow2O(zeta=0.95, wn=7.0, dt=1.0 / render_fps) if render_fps else None
        if follower is not None:
            follower.cx = float(prev_cx)
            follower.cy = float(prev_cy)
            follower.vx = 0.0
            follower.vy = 0.0

        if self.init_manual:
            frame0, _fps0 = grab_frame_at_time(
                input_mp4, max(0.0, self.init_t), fps_hint=src_fps or 30.0
            )
            if frame0 is not None:
                roi = manual_select_ball(frame0, window="Drag around the BALL, press Enter")
                if roi:
                    x, y, w, h = roi
                    bx0 = x + w / 2.0
                    by0 = y + h / 2.0
                    kal = CV2DKalman(bx0, by0)
                    frame0_gray = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
                    tx0 = int(max(0, bx0 - tpl_side // 2))
                    ty0 = int(max(0, by0 - tpl_side // 2))
                    tx1 = int(min(frame0.shape[1], tx0 + tpl_side))
                    ty1 = int(min(frame0.shape[0], ty0 + tpl_side))
                    template = frame0_gray[ty0:ty1, tx0:tx1].copy()
                    prev_cx, prev_cy = float(bx0), float(by0)
                    if tf:
                        tf.write(
                            json.dumps(
                                to_jsonable(
                                    {
                                        "t": float(self.init_t),
                                        "used": "manual_bootstrap",
                                        "cx": float(prev_cx),
                                        "cy": float(prev_cy),
                                        "zoom": 1.2,
                                        "crop": [
                                            float(max(0, bx0 - 240)),
                                            float(max(0, by0 - 432)),
                                            480.0,
                                            864.0,
                                        ],
                                        "ball": [float(bx0), float(by0)],
                                    }
                                )
                            )
                            + "\n"
                        )
                else:
                    print("[WARN] Manual init skipped (no ROI selected).")
            else:
                print(f"[WARN] Could not grab frame for manual init at t={self.init_t:.2f} s")

        try:
            for state in states:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                if self.flip180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)

                n = state.frame
                t = n / float(render_fps) if render_fps else 0.0

                bx = by = None
                ball_available = False
                label_available = False
                used_tag = "planner"
                planned_zoom: Optional[float] = None

                def _refresh_template(cx_val: float, cy_val: float) -> None:
                    nonlocal template
                    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    sx0 = int(max(0, cx_val - tpl_side // 2))
                    sy0 = int(max(0, cy_val - tpl_side // 2))
                    sx1 = int(min(frame.shape[1], sx0 + tpl_side))
                    sy1 = int(min(frame.shape[0], sy0 + tpl_side))
                    cur_tpl = g[sy0:sy1, sx0:sx1]
                    if cur_tpl.size < 9:
                        return
                    if template is None or template.shape != cur_tpl.shape:
                        template = cur_tpl.copy()
                    else:
                        template = cv2.addWeighted(template, 0.85, cur_tpl, 0.15, 0)

                cam_center_override: Optional[Tuple[float, float]] = None

                if offline_ball_path and n < len(offline_ball_path):
                    path_xyz = offline_ball_path[n]
                    if path_xyz is not None:
                        bx = float(path_xyz[0])
                        by = float(path_xyz[1])
                        z_planned = float(path_xyz[2])
                        ball_available = True
                        used_tag = "offline_path"
                        planned_zoom = float(np.clip(z_planned, zoom_min, zoom_max))
                        zoom = planned_zoom
                        if follower is not None:
                            cam_center_override = follower.step(bx, by)
                        else:
                            cam_center_override = (bx, by)
                        if kal is None:
                            kal = CV2DKalman(bx, by)
                        else:
                            kal.bx, kal.by = bx, by
                        _refresh_template(bx, by)
                else:
                    if state.ball:
                        label_bx, label_by = state.ball
                        label_bx = float(label_bx)
                        label_by = float(label_by)
                        bx, by = label_bx, label_by
                        ball_available = True
                        label_available = True
                        used_tag = "label"

                    pred_x = pred_y = None
                    if kal is not None:
                        pred_x, pred_y = kal.predict()

                    if label_available and bx is not None and by is not None:
                        if kal is None:
                            kal = CV2DKalman(bx, by)
                        else:
                            kal.correct(bx, by)
                        _refresh_template(bx, by)
                    elif kal is not None and pred_x is not None and pred_y is not None:
                        cand = find_ball_candidate(
                            frame,
                            (pred_x, pred_y),
                            tpl=template,
                            search_r=280,
                            min_r=7,
                            max_r=22,
                            min_circ=0.58,
                        )
                        if cand is not None:
                            cbx, cby, _circ, ncc, dist = cand
                            if dist < 140 or ncc >= 0.36:
                                bx, by = float(cbx), float(cby)
                                kal.correct(bx, by)
                                _refresh_template(bx, by)
                                ball_available = True
                                used_tag = "model_cand"
                            else:
                                bx, by = float(pred_x), float(pred_y)
                                kal.bx, kal.by = bx, by
                                ball_available = True
                                used_tag = "model_pred"
                        else:
                            bx, by = float(pred_x), float(pred_y)
                            kal.bx, kal.by = bx, by
                            ball_available = True
                            used_tag = "model_pred"

                if label_available and kal is not None:
                    ball_available = True

                pcx, pcy, pzoom = cam[n] if n < len(cam) else (prev_cx, prev_cy, 1.2)
                if cam_center_override is not None:
                    cx, cy = cam_center_override
                elif ball_available and bx is not None and by is not None:
                    cx = 0.90 * bx + 0.10 * prev_cx
                    cy = 0.90 * by + 0.10 * prev_cy
                else:
                    cx, cy = pcx, pcy

                if render_fps > 0:
                    max_dx = 9999.0
                    max_dy = 9999.0
                    dx = cx - prev_cx
                    dy = cy - prev_cy
                    if abs(dx) > max_dx:
                        cx = prev_cx + (max_dx if dx > 0 else -max_dx)
                    if abs(dy) > max_dy:
                        cy = prev_cy + (max_dy if dy > 0 else -max_dy)

                if planned_zoom is not None:
                    plan_zoom = planned_zoom
                else:
                    plan_zoom = float(np.clip(float(pzoom), zoom_min, zoom_max))
                zoom = plan_zoom
                x0, y0, crop_w, crop_h = compute_portrait_crop(
                    float(cx),
                    float(cy),
                    zoom,
                    width,
                    height,
                    portrait_w,
                    portrait_h,
                    self.pad,
                )

                cur_bx: Optional[float] = None
                cur_by: Optional[float] = None
                speed_px = 0.0
                if ball_available and bx is not None and by is not None and crop_w > 1 and crop_h > 1:
                    x0, y0, crop_w, crop_h, zoom = guarantee_ball_in_crop(
                        x0,
                        y0,
                        crop_w,
                        crop_h,
                        float(bx),
                        float(by),
                        float(width),
                        float(height),
                        float(zoom),
                        zoom_min,
                        zoom_max,
                        margin=0.10,
                        step_zoom=0.96,
                    )
                    cur_bx = float(bx)
                    cur_by = float(by)
                    if prev_ball_x is None or prev_ball_y is None:
                        prev_ball_x, prev_ball_y = cur_bx, cur_by
                    speed_px = math.hypot(cur_bx - prev_ball_x, cur_by - prev_ball_y)
                    prev_ball_x, prev_ball_y = cur_bx, cur_by
                else:
                    prev_ball_x = None
                    prev_ball_y = None

                if (
                    used_tag != "offline_path"
                    and cur_bx is not None
                    and cur_by is not None
                    and crop_w > 1
                    and crop_h > 1
                ):
                    zoom = dynamic_zoom(
                        prev_zoom=prev_zoom,
                        bx=cur_bx,
                        by=cur_by,
                        x0=x0,
                        y0=y0,
                        cw=crop_w,
                        ch=crop_h,
                        src_w=float(width),
                        src_h=float(height),
                        speed_px=speed_px,
                        target_zoom_min=zoom_min,
                        target_zoom_max=zoom_max,
                        k_speed_out=0.0007,
                        edge_margin=0.14,
                        edge_gain=0.10,
                        z_rate=0.07,
                    )

                    x0, y0, crop_w, crop_h = compute_portrait_crop(
                        float(cx),
                        float(cy),
                        float(zoom),
                        width,
                        height,
                        portrait_w,
                        portrait_h,
                        self.pad,
                    )

                    x0, y0, crop_w, crop_h, zoom = guarantee_ball_in_crop(
                        x0,
                        y0,
                        crop_w,
                        crop_h,
                        cur_bx,
                        cur_by,
                        float(width),
                        float(height),
                        float(zoom),
                        zoom_min,
                        zoom_max,
                        margin=0.10,
                        step_zoom=0.96,
                    )

                prev_cx, prev_cy = float(cx), float(cy)
                prev_zoom = float(zoom)
                if tf:
                    ball_log_x = float(bx) if bx is not None else float("nan")
                    ball_log_y = float(by) if by is not None else float("nan")
                    tf.write(
                        json.dumps(
                            to_jsonable(
                                {
                                    "t": float(t),
                                    "used": used_tag,
                                    "cx": float(cx),
                                    "cy": float(cy),
                                    "zoom": float(zoom),
                                    "crop": [float(x0), float(y0), float(crop_w), float(crop_h)],
                                    "ball": [ball_log_x, ball_log_y],
                                }
                            )
                        )
                        + "\n"
                    )

                clamp_flags = list(state.clamp_flags) if state.clamp_flags is not None else []
                frame_state = CamState(
                    frame=state.frame,
                    cx=float(cx),
                    cy=float(cy),
                    zoom=float(zoom),
                    crop_w=float(crop_w),
                    crop_h=float(crop_h),
                    x0=float(x0),
                    y0=float(y0),
                    used_label=state.used_label,
                    clamp_flags=clamp_flags,
                    ball=state.ball,
                )

                composed, _ = self._compose_frame(frame, frame_state, output_size, overlay_image)

                out_path = self.temp_dir / f"f_{state.frame:06d}.jpg"
                success = cv2.imwrite(str(out_path), composed, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                if not success:
                    raise RuntimeError(f"Failed to write frame to {out_path}")
        finally:
            cap.release()
            if tf:
                tf.close()
                self.telemetry = None

        endcard_frames = self._append_endcard(output_size)
        if endcard_frames:
            start_index = len(states)
            for offset, endcard_frame in enumerate(endcard_frames):
                out_path = self.temp_dir / f"f_{start_index + offset:06d}.jpg"
                cv2.imwrite(str(out_path), endcard_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    def ffmpeg_stitch(
        self,
        crf: int,
        keyint: int,
        log_path: Optional[Path] = None,
    ) -> None:
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

        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError("ffmpeg failed during stitching.") from exc

        self.last_ffmpeg_command = list(command)



def _prepare_temp_dir(temp_dir: Path, clean: bool) -> None:
    if clean and temp_dir.exists():
        shutil.rmtree(temp_dir)
    if not temp_dir.exists():
        temp_dir.mkdir(parents=True, exist_ok=True)
        return
    for file in temp_dir.glob("*.jpg"):
        try:
            file.unlink()
        except OSError:
            logging.warning("Failed to remove temp frame %s", file)


def _default_output_path(input_path: Path, preset: str) -> Path:
    suffix = f".__{preset.upper()}.mp4"
    return input_path.with_name(input_path.stem + suffix)


def run(args: argparse.Namespace, telemetry: Optional[TextIO] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    input_path = Path(args.in_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    presets = load_presets()
    preset_key = (args.preset or "cinematic").lower()
    if preset_key not in presets:
        raise ValueError(f"Preset '{preset_key}' not found in {PRESETS_PATH}")

    preset_config = presets[preset_key]

    fps_in = float(ffprobe_fps(input_path))
    try:
        duration_s = float(ffprobe_duration(input_path))
    except RuntimeError:
        duration_s = 0.0
    fps_out = float(args.fps) if args.fps is not None else float(preset_config.get("fps", fps_in))
    if fps_out <= 0:
        fps_out = fps_in if fps_in > 0 else 30.0

    portrait_str = args.portrait or preset_config.get("portrait")
    portrait = parse_portrait(portrait_str) if portrait_str else None

    lookahead = int(args.lookahead) if args.lookahead is not None else int(preset_config.get("lookahead", 18))
    smoothing = float(args.smoothing) if args.smoothing is not None else float(preset_config.get("smoothing", 0.65))
    pad = float(args.pad) if args.pad is not None else float(preset_config.get("pad", 0.22))
    speed_limit = float(args.speed_limit) if args.speed_limit is not None else float(preset_config.get("speed_limit", 480))
    zoom_min = float(args.zoom_min) if args.zoom_min is not None else float(preset_config.get("zoom_min", 1.0))
    zoom_max = float(args.zoom_max) if args.zoom_max is not None else float(preset_config.get("zoom_max", 2.2))
    crf = int(args.crf) if args.crf is not None else int(preset_config.get("crf", 19))
    keyint_factor = int(args.keyint_factor) if args.keyint_factor is not None else int(preset_config.get("keyint_factor", 4))

    output_path = Path(args.out) if args.out else _default_output_path(input_path, preset_key)
    output_path = output_path.expanduser().resolve()

    labels_root = args.labels_root or "out/yolo"
    label_files = find_label_files(input_path.stem, labels_root)

    log_dict: dict[str, object] = {}

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError("Unable to open input video for metadata extraction.")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()

    if duration_s <= 0 and frame_count > 0 and fps_in > 0:
        duration_s = frame_count / float(fps_in)
    if duration_s <= 0 and frame_count > 0:
        fallback_fps = fps_in if fps_in > 0 else 30.0
        duration_s = frame_count / float(fallback_fps)

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
    )
    states = planner.plan(positions, used_mask)

    temp_root = Path("out/autoframe_work")
    temp_dir = temp_root / preset_key / input_path.stem
    _prepare_temp_dir(temp_dir, args.clean_temp)

    brand_overlay_path = Path(args.brand_overlay).expanduser() if args.brand_overlay else None
    endcard_path = Path(args.endcard).expanduser() if args.endcard else None
    offline_ball_path: Optional[List[Tuple[float, float, float]]] = None
    if getattr(args, "ball_path", None):
        ball_path_file = Path(args.ball_path).expanduser()
        if not ball_path_file.exists():
            raise FileNotFoundError(f"Ball path file not found: {ball_path_file}")
        with open(ball_path_file, "r", encoding="utf-8") as f:
            offline_ball_path = []
            default_zoom = 1.30
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                bx = float(data["bx"])
                by = float(data["by"])
                bz = float(data.get("z", default_zoom))
                offline_ball_path.append((bx, by, bz))
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
        telemetry=telemetry,
        init_manual=getattr(args, "init_manual", False),
        init_t=getattr(args, "init_t", 0.8),
        ball_path=offline_ball_path,
    )

    renderer.write_frames(states)

    keyint = max(1, int(round(float(keyint_factor) * float(fps_out))))
    log_path = Path(args.log).expanduser() if args.log else None
    renderer.ffmpeg_stitch(crf=crf, keyint=keyint, log_path=log_path)

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
        summary.update(log_dict)
        with log_path.open("w", encoding="utf-8") as handle:
            json.dump(to_jsonable(log_dict), handle, ensure_ascii=False, indent=2)
            handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified cinematic ball-follow renderer")
    parser.add_argument("--in", dest="in_path", required=True, help="Input MP4 path")
    parser.add_argument("--src", dest="src", help="Legacy compatibility input path (ignored)")
    parser.add_argument("--out", dest="out", help="Output MP4 path")
    parser.add_argument("--preset", dest="preset", choices=["cinematic", "gentle", "realzoom"], default="cinematic")
    parser.add_argument("--portrait", dest="portrait", help="Portrait canvas WxH")
    parser.add_argument("--fps", dest="fps", type=float, help="Output FPS")
    parser.add_argument("--flip180", dest="flip180", action="store_true", help="Rotate frames by 180 degrees before processing")
    parser.add_argument("--labels-root", dest="labels_root", help="Root directory containing YOLO label shards")
    parser.add_argument("--clean-temp", dest="clean_temp", action="store_true", help="Remove temporary frame folder before rendering")
    parser.add_argument("--lookahead", dest="lookahead", type=int, help="Frames of lookahead for planning")
    parser.add_argument("--smoothing", dest="smoothing", type=float, help="EMA smoothing factor")
    parser.add_argument("--pad", dest="pad", type=float, help="Edge padding ratio used to derive zoom")
    parser.add_argument("--speed-limit", dest="speed_limit", type=float, help="Maximum pan speed in px/sec")
    parser.add_argument("--zoom-min", dest="zoom_min", type=float, help="Minimum zoom multiplier")
    parser.add_argument("--zoom-max", dest="zoom_max", type=float, help="Maximum zoom multiplier")
    parser.add_argument("--telemetry", dest="telemetry", help="Output JSONL telemetry file")
    parser.add_argument("--brand-overlay", dest="brand_overlay", help="PNG overlay composited on every frame")
    parser.add_argument("--endcard", dest="endcard", help="Optional endcard image displayed for ~2 seconds")
    parser.add_argument("--log", dest="log", help="Optional render log path")
    parser.add_argument("--crf", dest="crf", type=int, help="Override CRF value")
    parser.add_argument("--keyint-factor", dest="keyint_factor", type=int, help="Override keyint factor")
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
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    tf = None
    if getattr(args, "telemetry", None):
        telemetry_path = Path(args.telemetry).expanduser()
        telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        args.telemetry = os.fspath(telemetry_path)
        tf = open(args.telemetry, "w", encoding="utf-8")
    try:
        run(args, telemetry=tf)
    finally:
        if tf:
            tf.close()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    try:
        main()
    except Exception as exc:  # pylint: disable=broad-except
        logging.error(str(exc))
        sys.exit(1)

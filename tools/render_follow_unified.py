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

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.upscale import upscale_video


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
    try:
        edge_frac = float(edge_frac)
    except (TypeError, ValueError):
        edge_frac = 1.0
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
        if isinstance(ball_val, (list, tuple)) and len(ball_val) >= 2:
            try:
                return float(ball_val[0]), float(ball_val[1])
            except (TypeError, ValueError):
                return None, None
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
        try:
            return float(val_x), float(val_y)
        except (TypeError, ValueError):
            return None

    def _pair_from_sequence(seq: Sequence[object]) -> Optional[tuple[float, float]]:
        if len(seq) < 2:
            return None
        try:
            return float(seq[0]), float(seq[1])
        except (TypeError, ValueError):
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


def smooth_series(x, alpha=0.15):
    # EWMA – robust, low-latency
    y = np.empty_like(x, dtype=float)
    acc = x[0]
    for i, v in enumerate(x):
        acc = alpha * v + (1 - alpha) * acc
        y[i] = acc
    return y


def speed(px):
    # |Δ| per frame (px/frame)
    d = np.sqrt(np.sum(np.diff(px, axis=0) ** 2, axis=1))
    return np.concatenate([[0.0], d])


class ZoomPlanner:
    """
    Speed → zoom with hysteresis and slew-rate limiting.
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
):
    """
    bx/by are STABILIZED ball coords per-frame (not raw detector jitter).
    - pan_alpha: EWMA for camera center (higher = tighter follow)
    - lead: amount of lookahead in seconds, implemented as future-index peek
    """

    n = len(bx)
    pts = np.stack([bx, by], axis=1)
    spd = speed(pts)

    # lookahead index in frames (use fps already known in your script)
    LA = int(round(lead * FPS)) if lead > 0 else 0
    bx_look = np.concatenate([bx[LA:], np.repeat(bx[-1], LA)])
    by_look = np.concatenate([by[LA:], np.repeat(by[-1], LA)])

    cx = smooth_series(bx_look, alpha=pan_alpha)
    cy = smooth_series(by_look, alpha=pan_alpha)

    # compute zoom per-frame
    zp = ZoomPlanner(z_min=1.0, z_max=1.9, s_lo=2.5, s_hi=22.0, hysteresis=0.18, max_step=0.025)
    z = zp.plan(spd)

    # convert zoom to crop size (height); ensure target aspect; keep in bounds
    crop_h = np.clip(frame_h / z, 240, frame_h - 2 * bounds_pad)
    crop_w = np.minimum(crop_h * target_aspect, frame_w - 2 * bounds_pad)
    crop_h = crop_w / target_aspect  # enforce exact aspect

    # clamp camera center so crop box stays inside frame
    half_w = crop_w / 2.0
    half_h = crop_h / 2.0
    cx = np.clip(cx, half_w + bounds_pad, frame_w - half_w - bounds_pad)
    cy = np.clip(cy, half_h + bounds_pad, frame_h - half_h - bounds_pad)

    # produce integer crops
    x0 = (cx - half_w).round().astype(int)
    y0 = (cy - half_h).round().astype(int)
    w = crop_w.round().astype(int)
    h = crop_h.round().astype(int)

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
    try:
        return float(obj)
    except Exception:
        return str(obj)


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

    try:
        zoom_value = float(state.get("zoom", zoom_max))
    except (TypeError, ValueError):
        zoom_value = float(zoom_max)

    candidates = [zoom_value]
    for candidate in (zoom_min, zoom_max):
        try:
            candidates.append(float(candidate))
        except (TypeError, ValueError):
            continue
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
    if pad <= 0.49:
        try:
            pad_frac = max(0.0, min(float(pad), 0.49))
        except (TypeError, ValueError):
            pad_frac = 0.0
    else:
        try:
            pad_px = float(max(0, int(round(pad))))
        except (TypeError, ValueError):
            pad_px = 0.0

    shrink = 1.0
    if pad_frac > 0.0:
        shrink = max(0.0, 1.0 - 2.0 * pad_frac)

    max_crop_w = max(1.0, float(src_w) - 2.0 * pad_px)
    max_crop_h = max(1.0, float(src_h) - 2.0 * pad_px)

    max_zoom_candidates = [zoom_value]
    for candidate in (zoom_min, zoom_max):
        try:
            value = float(candidate)
        except (TypeError, ValueError):
            continue
        if value > 0:
            max_zoom_candidates.append(value)
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

    try:
        prev_w = float(state.get("w", target_crop_w))
    except (TypeError, ValueError):
        prev_w = float(target_crop_w)
    try:
        prev_h = float(state.get("h", target_crop_h))
    except (TypeError, ValueError):
        prev_h = float(target_crop_h)
    if prev_w <= 0:
        prev_w = float(target_crop_w)
    if prev_h <= 0:
        prev_h = float(target_crop_h)

    try:
        prev_x = float(state.get("x", float(bx) - prev_w / 2.0))
    except (TypeError, ValueError):
        prev_x = float(bx) - prev_w / 2.0
    try:
        prev_y = float(state.get("y", float(by) - prev_h / 2.0))
    except (TypeError, ValueError):
        prev_y = float(by) - prev_h / 2.0

    prev_cx = prev_x + prev_w / 2.0
    prev_cy = prev_y + prev_h / 2.0

    bx = float(bx)
    by = float(by)

    alpha_center = 0.20
    alpha_zoom = 0.15

    cx = (1 - alpha_center) * prev_cx + alpha_center * bx
    cy = (1 - alpha_center) * prev_cy + alpha_center * by

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
            try:
                w = int(width)
                h = int(height)
            except (TypeError, ValueError):
                return None
            if w > 0 and h > 0:
                return w, h
            return None
        if isinstance(size_value, Sequence):
            seq = list(size_value)
            if len(seq) < 2:
                return None
            try:
                w = int(seq[0])
                h = int(seq[1])
            except (TypeError, ValueError):
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
            try:
                mbw_f = float(mbw) if mbw is not None else None
                mbh_f = float(mbh) if mbh is not None else None
            except (TypeError, ValueError):
                mbw_f = mbh_f = None
            if mbw_f is not None and mbh_f is not None and mbw_f > 0 and mbh_f > 0:
                min_box = (mbw_f, mbh_f)
        elif isinstance(min_box_value, Sequence):
            seq = list(min_box_value)
            if len(seq) >= 2:
                try:
                    mbw_f = float(seq[0])
                    mbh_f = float(seq[1])
                except (TypeError, ValueError):
                    mbw_f = mbh_f = None
                else:
                    if mbw_f > 0 and mbh_f > 0:
                        min_box = (mbw_f, mbh_f)

        horizon_value = value.get("horizon_lock")
        if horizon_value is not None:
            try:
                horizon_lock = float(horizon_value)
            except (TypeError, ValueError):
                horizon_lock = 0.0
            else:
                horizon_lock = float(np.clip(horizon_lock, 0.0, 1.0))

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

        render_fps = self.fps if self.fps > 0 else 30.0
        if render_fps <= 0:
            render_fps = 30.0
        self.render_fps = render_fps
        self.speed_norm_px = 24.0 * (render_fps / 24.0)
        self.zoom_slew = 0.02 * (render_fps / 24.0)

        self.min_box_w = 0.0
        self.min_box_h = 0.0
        if min_box:
            try:
                mbw = float(min_box[0])
                mbh = float(min_box[1])
            except (TypeError, ValueError, IndexError):
                mbw = mbh = 0.0
            else:
                if mbw > 0:
                    self.min_box_w = min(self.width, mbw)
                if mbh > 0:
                    self.min_box_h = min(self.height, mbh)

        self.horizon_lock = float(np.clip(horizon_lock, 0.0, 1.0))

        self.speed_zoom_config: Optional[dict[str, float]] = None
        if speed_zoom and bool(speed_zoom.get("enabled", True)):
            try:
                v_lo = float(speed_zoom.get("v_lo", 0.0))
                v_hi = float(speed_zoom.get("v_hi", 0.0))
                zoom_lo = float(speed_zoom.get("zoom_lo", 1.0))
                zoom_hi = float(speed_zoom.get("zoom_hi", 1.0))
            except (TypeError, ValueError):
                self.speed_zoom_config = None
            else:
                if v_lo < 0:
                    v_lo = 0.0
                if v_hi < 0:
                    v_hi = 0.0
                if v_hi < v_lo:
                    v_hi, v_lo = v_lo, v_hi
                base_zoom = float(self.zoom_max)
                self.speed_zoom_config = {
                    "v_lo": v_lo,
                    "v_hi": v_hi,
                    "zoom_lo": zoom_lo,
                    "zoom_hi": zoom_hi,
                    "base_zoom": base_zoom,
                }

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
        prev_cy = self.height / 2.0
        prev_zoom = self.base_zoom
        fallback_center = np.array([prev_cx, self.height * 0.45], dtype=np.float32)
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

            bx_used = float(target[0])
            by_used = float(target[1])

            speed_pf = math.hypot(bx_used - prev_target_x, by_used - prev_target_y)
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
            cy = center_alpha * by_used + (1.0 - center_alpha) * prev_cy

            ball_point: Optional[Tuple[float, float]] = None
            if has_position:
                ball_point = (float(pos[0]), float(pos[1]))

            clamp_flags: List[str] = []

            cx, x_clamped = _clamp_axis(prev_cx, cx, pxpf_x)
            cy, y_clamped = _clamp_axis(prev_cy, cy, pxpf_y)
            speed_limited = x_clamped or y_clamped
            if speed_limited:
                clamp_flags.append("speed")

            _, est_crop_w, est_crop_h = _compute_crop_dimensions(zoom)
            edge_zoom_scale = 1.0
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
                edge_zoom_scale = zoom_scale
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
            prev_target_y = by_used

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
        try:
            zoom_edge_frac = float(follow_zoom_edge_frac)
        except (TypeError, ValueError):
            zoom_edge_frac = 1.0
        if not math.isfinite(zoom_edge_frac) or zoom_edge_frac <= 0.0:
            zoom_edge_frac = 1.0
        self.follow_zoom_edge_frac = zoom_edge_frac

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
                        try:
                            sanitized["z"] = float(z_value)
                        except (TypeError, ValueError):
                            sanitized["z"] = 1.30
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
                    try:
                        normalized_ball_path.append(
                            {"bx": float(bx_val), "by": float(by_val), "z": float(z_val)}
                        )
                    except (TypeError, ValueError):
                        normalized_ball_path.append(None)
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

        offline_plan_data: Optional[dict[str, np.ndarray]] = None
        follow_targets: Optional[Tuple[List[float], List[float]]] = None
        follow_valid_mask: Optional[List[bool]] = None
        offline_plan_len = 0
        if offline_ball_path:
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
                default_cy = float(height) * 0.45
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

                plan_x0, plan_y0, plan_w, plan_h, plan_spd, plan_zoom = plan_camera_from_ball(
                    bx_arr,
                    by_arr,
                    float(width),
                    float(height),
                    float(target_aspect),
                    pan_alpha=pan_alpha,
                    lead=lead_seconds,
                    bounds_pad=bounds_pad,
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
        follow_targets_len = len(follow_targets[0]) if follow_targets else 0
        follow_lookahead_frames = max(0, int(self.follow_lookahead))
        follower: Optional[CamFollow2O]
        if render_fps > 0:
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
        try:
            if probe_only:
                return float(jerk95)

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
                ball_source_tag: Optional[str] = prev_ball_source
                planned_zoom: Optional[float] = None
                telemetry_ball_src: Optional[Tuple[float, float]] = None
                telemetry_ball_out: Optional[Tuple[Optional[float], Optional[float]]] = None
                telemetry_crop: Optional[Tuple[float, float, float, float]] = None
                planner_spd: Optional[float] = None
                planner_zoom: Optional[float] = None
                edge_zoom_scale_follow = 1.0

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

                ball_path_entry: Optional[Tuple[float, float]] = None
                ball_path_space: Optional[str] = None
                ball_path_rec_for_frame: Optional[Union[Mapping[str, object], Sequence[object]]] = None

                if offline_ball_path and n < len(offline_ball_path):
                    path_rec = offline_ball_path[n]
                    if path_rec is not None:
                        ball_path_rec_for_frame = path_rec
                        entry_bx: Optional[float] = None
                        entry_by: Optional[float] = None
                        entry_space: Optional[str] = None
                        z_planned_val: float = zoom

                        if isinstance(path_rec, Mapping):
                            ball_x, ball_y = pick_ball(path_rec, frame.shape[1], frame.shape[0])
                            if ball_x is not None and ball_y is not None:
                                entry_bx = float(ball_x)
                                entry_by = float(ball_y)
                                if "bx_stab" in path_rec:
                                    entry_space = "stab"
                                elif "bx_raw" in path_rec:
                                    entry_space = "raw"
                                elif "ball" in path_rec:
                                    entry_space = "ball"
                                elif "bx" in path_rec:
                                    entry_space = "generic"

                            z_candidate = path_rec.get("z")
                            if z_candidate is not None:
                                try:
                                    z_planned_val = float(z_candidate)
                                except (TypeError, ValueError):
                                    z_planned_val = zoom
                        else:
                            entry_vals = tuple(path_rec)
                            if len(entry_vals) >= 2:
                                entry_bx_val = entry_vals[0]
                                entry_by_val = entry_vals[1]
                                if entry_bx_val is not None and entry_by_val is not None:
                                    try:
                                        entry_bx = float(entry_bx_val)
                                        entry_by = float(entry_by_val)
                                        entry_space = "generic"
                                    except (TypeError, ValueError):
                                        entry_bx = None
                                        entry_by = None
                                if len(entry_vals) >= 3:
                                    z_candidate = entry_vals[2]
                                    if z_candidate is not None:
                                        try:
                                            z_planned_val = float(z_candidate)
                                        except (TypeError, ValueError):
                                            pass

                        if entry_bx is not None and entry_by is not None:
                            bx = entry_bx
                            by = entry_by
                            ball_path_entry = (bx, by)
                            ball_path_space = entry_space
                            ball_available = True
                            used_tag = "offline_path"
                            ball_source_tag = "ball_path"
                            planned_zoom = None
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
                        ball_source_tag = "label"

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
                                ball_source_tag = "model_cand"
                            else:
                                bx, by = float(pred_x), float(pred_y)
                                kal.bx, kal.by = bx, by
                                ball_available = True
                                used_tag = "model_pred"
                                ball_source_tag = "model_pred"
                        else:
                            bx, by = float(pred_x), float(pred_y)
                            kal.bx, kal.by = bx, by
                            ball_available = True
                            used_tag = "model_pred"
                            ball_source_tag = "model_pred"

                if label_available and kal is not None:
                    ball_available = True
                    if ball_source_tag is None:
                        ball_source_tag = "label"

                bx_src: Optional[float] = None
                by_src: Optional[float] = None
                if ball_available and bx is not None and by is not None:
                    bx_src = float(bx)
                    by_src = float(by)
                    alpha = 0.25
                    if (
                        prev_ball_src_x is not None
                        and prev_ball_src_y is not None
                    ):
                        bx_src = prev_ball_src_x + alpha * (bx_src - prev_ball_src_x)
                        by_src = prev_ball_src_y + alpha * (by_src - prev_ball_src_y)
                prev_ball_src_x, prev_ball_src_y = bx_src, by_src
                if bx_src is not None:
                    bx_src = max(0.0, min(src_w_f - 1.0, bx_src))
                if by_src is not None:
                    by_src = max(0.0, min(src_h_f - 1.0, by_src))
                if bx_src is not None:
                    bx = bx_src
                if by_src is not None:
                    by = by_src

                if used_tag == "offline_path":
                    W = float(width)
                    H = float(height)
                    prev_ball_bx = float(prev_bx)
                    prev_ball_by = float(prev_by)
                    have_ball = bool(ball_available and bx is not None and by is not None)

                    if have_ball:
                        eff_bx = float(bx)
                        eff_by = float(by)
                    else:
                        eff_bx = prev_ball_bx
                        eff_by = prev_ball_by

                    planner_handled = False
                    holding_follow = False
                    if offline_plan_data is not None and n < offline_plan_len:
                        plan_x0 = float(offline_plan_data["x0"][n])
                        plan_y0 = float(offline_plan_data["y0"][n])
                        plan_w = float(offline_plan_data["w"][n])
                        plan_h = float(offline_plan_data["h"][n])
                        planner_zoom = float(offline_plan_data["z"][n])
                        planner_spd = float(offline_plan_data["spd"][n])
                        if planner_zoom <= 0:
                            planner_zoom = float(H / plan_h) if plan_h > 0 else prev_zoom

                        x0 = plan_x0
                        y0 = plan_y0
                        crop_w = plan_w
                        crop_h = plan_h
                        cx = x0 + 0.5 * crop_w
                        cy = y0 + 0.5 * crop_h
                        zoom = planner_zoom
                        telemetry_crop = (x0, y0, crop_w, crop_h)
                        telemetry_ball_src = (eff_bx, eff_by)
                        used_tag = "planner"
                        planner_handled = True
                        prev_ball_x = eff_bx
                        prev_ball_y = eff_by
                        prev_cx = float(cx)
                        prev_cy = float(cy)
                        prev_zoom = float(zoom)
                        prev_bx = eff_bx
                        prev_by = eff_by
                        follow_hold.reset_target(cx, cy)
                        if have_ball:
                            ball_path_entry = (eff_bx, eff_by)
                    if not planner_handled:
                        base_target_x = float(eff_bx)
                        base_target_y = float(eff_by)
                        if have_ball and follow_targets and follow_targets_len > 0:
                            idx = min(n + follow_lookahead_frames, follow_targets_len - 1)
                            base_target_x = float(follow_targets[0][idx])
                            base_target_y = float(follow_targets[1][idx])

                        fps_for_vel = (
                            float(render_fps)
                            if render_fps and render_fps > 0
                            else (float(src_fps) if src_fps and src_fps > 0 else 30.0)
                        )
                        plan_len = follow_targets_len if follow_targets else len(states)
                        plan_idx = min(max(n, 0), plan_len - 1) if plan_len > 0 else 0
                        vx = 0.0
                        vy = 0.0
                        if (
                            follow_targets
                            and follow_targets_len > 1
                            and plan_idx > 0
                            and plan_idx < follow_targets_len - 1
                            and fps_for_vel > 0.0
                        ):
                            bx_prev_plan = float(follow_targets[0][plan_idx - 1])
                            by_prev_plan = float(follow_targets[1][plan_idx - 1])
                            bx_next_plan = float(follow_targets[0][plan_idx + 1])
                            by_next_plan = float(follow_targets[1][plan_idx + 1])
                            vx = 0.5 * (bx_next_plan - bx_prev_plan) * fps_for_vel
                            vy = 0.5 * (by_next_plan - by_prev_plan) * fps_for_vel
                        elif n > 0:
                            vx = float(base_target_x - prev_bx)
                            vy = float(base_target_y - prev_by)

                        lead_k = 0.02
                        target_x = float(base_target_x + lead_k * vx)
                        target_y = float(base_target_y + lead_k * vy)

                        valid_for_hold = bool(have_ball)
                        if follow_valid_mask and n < len(follow_valid_mask):
                            valid_for_hold = valid_for_hold and bool(follow_valid_mask[n])
                        eff_target_x, eff_target_y, holding_follow = follow_hold.apply(
                            target_x,
                            target_y,
                            valid_for_hold,
                        )
                        if follower is not None:
                            if holding_follow and 0.0 < follow_hold.decay_factor < 1.0:
                                follower.damp_velocity(follow_hold.decay_factor)
                            cam_x, cam_y = follower.step(
                                float(eff_target_x), float(eff_target_y)
                            )
                        else:
                            alpha_pan = 0.30 if not holding_follow else 0.0
                            cam_x = alpha_pan * eff_target_x + (1 - alpha_pan) * prev_cx
                            cam_y = alpha_pan * eff_target_y + (1 - alpha_pan) * prev_cy
                        if not holding_follow:
                            follow_hold.reset_target(cam_x, cam_y)

                        speed = hypot(eff_bx - prev_ball_bx, eff_by - prev_ball_by)
                        norm = min(1.0, speed / 24.0)
                        z_min = float(zoom_min)
                        z_max = float(zoom_max)
                        zoom_target = z_min + (z_max - z_min) * (1.0 - norm)
                        slew = 0.02
                        zoom = prev_zoom + max(-slew, min(slew, zoom_target - prev_zoom))
                        zoom = float(np.clip(zoom, z_min if z_min > 0 else 1.0, z_max))
                    else:
                        cam_x = cx
                        cam_y = cy

                    crop_h = H / float(zoom) if zoom > 0 else H
                    crop_w = crop_h * target_aspect
                    if crop_w > W:
                        crop_w = float(W)
                        crop_h = crop_w / target_aspect if target_aspect else crop_h

                    margin_follow = (
                        float(self.follow_margin_px)
                        if self.follow_margin_px > 0
                        else 16.0
                    )
                    margin_follow = max(0.0, float(margin_follow))
                    half_w = 0.5 * float(crop_w)
                    half_h = 0.5 * float(crop_h)
                    max_margin = max(0.0, min(half_w, half_h) - 1.0)
                    margin_follow = min(margin_follow, max_margin)

                    s_out = edge_zoom_out(
                        float(cam_x),
                        float(cam_y),
                        float(eff_bx),
                        float(eff_by),
                        float(crop_w),
                        float(crop_h),
                        float(W),
                        float(H),
                        margin_px=margin_follow,
                        s_cap=self.follow_zoom_out_max,
                        edge_frac=self.follow_zoom_edge_frac,
                    )
                    eff_w = float(crop_w * s_out)
                    eff_h = float(crop_h * s_out)
                    hx = 0.5 * eff_w
                    hy = 0.5 * eff_h
                    cam_x = max(hx, min((W - 1.0) - hx, float(cam_x)))
                    cam_y = max(hy, min((H - 1.0) - hy, float(cam_y)))
                    x0 = max(0.0, min(W - eff_w, cam_x - hx))
                    y0 = max(0.0, min(H - eff_h, cam_y - hy))

                    crop_w = eff_w
                    crop_h = eff_h
                    cx = float(cam_x)
                    cy = float(cam_y)
                    if eff_h > 1e-6:
                        zoom = float(H / eff_h)

                    edge_zoom_scale_follow = float(s_out)

                    telemetry_ball_src = (eff_bx, eff_by)
                    telemetry_crop = (x0, y0, crop_w, crop_h)

                    if is_portrait and have_ball:
                        portrait_plan_state["zoom"] = float(
                            max(zoom_min, min(zoom_max, float(zoom)))
                        ) if zoom_max >= zoom_min else float(zoom)
                        x0, y0, crop_w, crop_h, portrait_plan_state = plan_crop_from_ball(
                            eff_bx,
                            eff_by,
                            width,
                            height,
                            out_w=portrait_w,
                            out_h=portrait_h,
                            zoom_min=zoom_min,
                            zoom_max=zoom_max,
                            pad=self.pad,
                            state=portrait_plan_state,
                        )
                        zoom = float(height) / float(max(crop_h, 1.0))
                        if zoom_max >= zoom_min:
                            zoom = max(zoom_min, min(zoom_max, zoom))
                        cx = float(x0 + 0.5 * crop_w)
                        cy = float(y0 + 0.5 * crop_h)
                    else:
                        x0, y0, crop_w, crop_h = compute_portrait_crop(
                            float(cx),
                            float(cy),
                            float(zoom),
                            width,
                            height,
                            target_aspect,
                            self.pad,
                        )

                    if have_ball and crop_w > 1 and crop_h > 1:
                        x0, y0, crop_w, crop_h, zoom = guarantee_ball_in_crop(
                            x0,
                            y0,
                            crop_w,
                            crop_h,
                            eff_bx,
                            eff_by,
                            float(width),
                            float(height),
                            float(zoom),
                            zoom_min,
                            zoom_max,
                            margin=self.pad,
                        )

                        if have_ball and crop_w > 1 and crop_h > 1:
                            x0, y0, crop_w, crop_h, zoom = guarantee_ball_in_crop(
                                x0,
                                y0,
                                crop_w,
                                crop_h,
                                eff_bx,
                                eff_by,
                                float(width),
                                float(height),
                                float(zoom),
                                zoom_min,
                                zoom_max,
                                margin=0.10,
                                step_zoom=0.96,
                            )
                            prev_ball_x = eff_bx
                            prev_ball_y = eff_by
                        else:
                            prev_ball_x = None
                            prev_ball_y = None

                        cx = float(x0 + 0.5 * crop_w)
                        cy = float(y0 + 0.5 * crop_h)
                        telemetry_crop = (x0, y0, crop_w, crop_h)
                        zoom = float(zoom)
                        prev_cx = float(cx)
                        prev_cy = float(cy)
                        prev_zoom = float(zoom)
                        if have_ball:
                            prev_bx = eff_bx
                            prev_by = eff_by
                        if not holding_follow:
                            follow_hold.reset_target(cx, cy)
                else:
                    zoom_speed: Optional[float] = None
                    speed_px = 0.0
                    bx_val: Optional[float] = None
                    by_val: Optional[float] = None

                    if ball_available and bx is not None and by is not None:
                        bx_val = float(bx)
                        by_val = float(by)
                        bx = bx_val
                        by = by_val
                        prev_bx_for_speed = prev_ball_x if prev_ball_x is not None else bx_val
                        prev_by_for_speed = prev_ball_y if prev_ball_y is not None else by_val

                        speed_px = hypot(bx_val - prev_bx_for_speed, by_val - prev_by_for_speed)
                        norm = min(1.0, speed_px / 24.0)
                        z_min = zoom_min
                        z_max = zoom_max
                        zoom_target = z_min + (z_max - z_min) * (1.0 - norm)
                        dmax = 0.02
                        zoom_step = max(-dmax, min(dmax, zoom_target - prev_zoom))
                        zoom_speed = float(np.clip(prev_zoom + zoom_step, z_min, z_max))

                    pcx, pcy, pzoom = cam[n] if n < len(cam) else (prev_cx, prev_cy, 1.2)
                    target_x: Optional[float] = None
                    target_y: Optional[float] = None
                    if follow_targets and follow_targets_len > 0:
                        idx = min(n + follow_lookahead_frames, follow_targets_len - 1)
                        target_x = float(follow_targets[0][idx])
                        target_y = float(follow_targets[1][idx])
                    if target_x is None or target_y is None:
                        if bx_val is not None and by_val is not None:
                            target_x = bx_val
                            target_y = by_val
                        else:
                            target_x = float(pcx)
                            target_y = float(pcy)

                    candidate_x = float(target_x if target_x is not None else pcx)
                    candidate_y = float(target_y if target_y is not None else pcy)
                    ball_valid_for_hold = bool(
                        bx_val is not None
                        and by_val is not None
                        and math.isfinite(bx_val)
                        and math.isfinite(by_val)
                    )
                    eff_target_x, eff_target_y, holding_follow = follow_hold.apply(
                        candidate_x,
                        candidate_y,
                        ball_valid_for_hold,
                    )

                    if follower is not None:
                        if holding_follow and 0.0 < follow_hold.decay_factor < 1.0:
                            follower.damp_velocity(follow_hold.decay_factor)
                        cx, cy = follower.step(eff_target_x, eff_target_y)
                    else:
                        if cam_center_override is not None:
                            cx, cy = cam_center_override
                        elif not holding_follow and bx_val is not None and by_val is not None:
                            cx = 0.90 * bx_val + 0.10 * prev_cx
                            cy = 0.90 * by_val + 0.10 * prev_cy
                        else:
                            cx = eff_target_x if not holding_follow else prev_cx
                            cy = eff_target_y if not holding_follow else prev_cy

                    if cam_center_override is not None:
                        cx, cy = cam_center_override
                    if not holding_follow:
                        follow_hold.reset_target(cx, cy)

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
                        zoom = planned_zoom
                    else:
                        default_zoom = float(np.clip(float(pzoom), zoom_min, zoom_max))
                        if zoom_speed is not None:
                            zoom = zoom_speed
                        else:
                            zoom = default_zoom

                    if ball_available and bx is not None and by is not None and zoom > 0:
                        view_h = height / float(zoom)
                        view_w = view_h * target_aspect
                        if view_w > width:
                            view_w = float(width)
                            view_h = view_w / target_aspect if target_aspect else view_h
                        x0 = min(max(cx - 0.5 * view_w, 0.0), width - view_w)
                        y0 = min(max(cy - 0.5 * view_h, 0.0), height - view_h)
                        cx = x0 + 0.5 * view_w
                        cy = y0 + 0.5 * view_h

                    if is_portrait and ball_available and bx is not None and by is not None:
                        portrait_plan_state["zoom"] = float(
                            max(zoom_min, min(zoom_max, float(zoom)))
                        ) if zoom_max >= zoom_min else float(zoom)
                        x0, y0, crop_w, crop_h, portrait_plan_state = plan_crop_from_ball(
                            float(bx),
                            float(by),
                            width,
                            height,
                            out_w=portrait_w,
                            out_h=portrait_h,
                            zoom_min=zoom_min,
                            zoom_max=zoom_max,
                            pad=self.pad,
                            state=portrait_plan_state,
                        )
                        zoom = float(height) / float(max(crop_h, 1.0))
                        if zoom_max >= zoom_min:
                            zoom = max(zoom_min, min(zoom_max, zoom))
                        cx = float(x0 + 0.5 * crop_w)
                        cy = float(y0 + 0.5 * crop_h)
                    else:
                        x0, y0, crop_w, crop_h = compute_portrait_crop(
                            float(cx),
                            float(cy),
                            zoom,
                            width,
                            height,
                            target_aspect,
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
                        telemetry_ball_src = (cur_bx, cur_by)
                    else:
                        prev_ball_x = None
                        prev_ball_y = None

                    if cur_bx is not None and cur_by is not None and crop_w > 1 and crop_h > 1:
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

                        if is_portrait:
                            portrait_plan_state["zoom"] = float(
                                max(zoom_min, min(zoom_max, float(zoom)))
                            ) if zoom_max >= zoom_min else float(zoom)
                            x0, y0, crop_w, crop_h, portrait_plan_state = plan_crop_from_ball(
                                cur_bx,
                                cur_by,
                                width,
                                height,
                                out_w=portrait_w,
                                out_h=portrait_h,
                                zoom_min=zoom_min,
                                zoom_max=zoom_max,
                                pad=self.pad,
                                state=portrait_plan_state,
                            )
                            zoom = float(height) / float(max(crop_h, 1.0))
                            if zoom_max >= zoom_min:
                                zoom = max(zoom_min, min(zoom_max, zoom))
                            cx = float(x0 + 0.5 * crop_w)
                            cy = float(y0 + 0.5 * crop_h)
                        else:
                            x0, y0, crop_w, crop_h = compute_portrait_crop(
                                float(cx),
                                float(cy),
                                float(zoom),
                                width,
                                height,
                                target_aspect,
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

                    telemetry_crop = (x0, y0, crop_w, crop_h)
                    if not holding_follow:
                        follow_hold.reset_target(cx, cy)
                    prev_cx, prev_cy = float(cx), float(cy)
                    prev_zoom = float(zoom)
                    if ball_available and bx is not None and by is not None:
                        prev_bx = float(bx)
                        prev_by = float(by)
                zoom_ball_x: Optional[float] = None
                zoom_ball_y: Optional[float] = None
                if have_ball:
                    try:
                        zoom_ball_x = float(eff_bx)
                        zoom_ball_y = float(eff_by)
                    except NameError:
                        zoom_ball_x = zoom_ball_y = None
                    except (TypeError, ValueError):
                        zoom_ball_x = zoom_ball_y = None
                if (
                    zoom_ball_x is None
                    or zoom_ball_y is None
                ) and ball_available and bx is not None and by is not None:
                    try:
                        zoom_ball_x = float(bx)
                        zoom_ball_y = float(by)
                    except (TypeError, ValueError):
                        zoom_ball_x = zoom_ball_y = None

                if (
                    is_portrait
                    and zoom_ball_x is not None
                    and zoom_ball_y is not None
                    and crop_w is not None
                    and crop_h is not None
                    and crop_w > 1.0
                    and crop_h > 1.0
                ):
                    margin_ratio = self.pad if self.pad > 0.0 else 0.10
                    margin_px = float(margin_ratio) * min(float(crop_w), float(crop_h))
                    if margin_px > 0.0:
                        zoom_out_max = float(self.follow_zoom_out_max)
                        edge_frac = float(self.follow_zoom_edge_frac)
                        headroom_px = 4.0
                        halfW = float(crop_w) * 0.5
                        halfH = float(crop_h) * 0.5
                        cx_val = float(cx)
                        cy_val = float(cy)
                        bx_val = float(zoom_ball_x)
                        by_val = float(zoom_ball_y)

                        left_d = bx_val - (cx_val - halfW + margin_px)
                        right_d = (cx_val + halfW - margin_px) - bx_val
                        top_d = by_val - (cy_val - halfH + margin_px)
                        bot_d = (cy_val + halfH - margin_px) - by_val

                        need_w = 0.0
                        need_h = 0.0
                        if min(left_d, right_d) < edge_frac * margin_px:
                            req_halfW = max(
                                bx_val - (cx_val - margin_px - headroom_px),
                                (cx_val + margin_px + headroom_px) - bx_val,
                            )
                            need_w = max(0.0, req_halfW - halfW) * 2.0

                        if min(top_d, bot_d) < edge_frac * margin_px:
                            req_halfH = max(
                                by_val - (cy_val - margin_px - headroom_px),
                                (cy_val + margin_px + headroom_px) - by_val,
                            )
                            need_h = max(0.0, req_halfH - halfH) * 2.0

                        effW = max(float(crop_w), need_w)
                        effH = max(float(crop_h), need_h)

                        if effW > float(width):
                            effW = float(width)
                        if effH > float(height):
                            effH = float(height)

                        base_zoom_out = max(
                            1.0,
                            min(
                                zoom_out_max,
                                max(
                                    effW / max(float(crop_w), 1e-6),
                                    effH / max(float(crop_h), 1e-6),
                                ),
                            ),
                        )

                        if base_zoom_out > 1.0 and effW > 1.0 and effH > 1.0:
                            hx = 0.5 * effW
                            hy = 0.5 * effH
                            width_f = float(width)
                            height_f = float(height)
                            cx_val = max(hx, min((width_f - 1.0) - hx, cx_val))
                            cy_val = max(hy, min((height_f - 1.0) - hy, cy_val))
                            x0 = max(0.0, min(width_f - effW, cx_val - hx))
                            y0 = max(0.0, min(height_f - effH, cy_val - hy))
                            crop_w = effW
                            crop_h = effH
                            cx = float(cx_val)
                            cy = float(cy_val)
                            telemetry_crop = (x0, y0, crop_w, crop_h)
                            zoom = float(height) / max(crop_h, 1e-6)
                            if zoom_max >= zoom_min:
                                zoom = max(zoom_min, min(zoom_max, zoom))
                            edge_zoom_scale_follow = float(
                                edge_zoom_scale_follow * base_zoom_out
                            )
                            prev_cx = float(cx)
                            prev_cy = float(cy)
                            prev_zoom = float(zoom)
                half_w_final = float(crop_w) * 0.5 if crop_w is not None else 0.0
                half_h_final = float(crop_h) * 0.5 if crop_h is not None else 0.0
                clamped_flag = 0
                if half_w_final > 0.0 and half_h_final > 0.0:
                    clamped_flag = int(
                        cy <= half_h_final + 0.5
                        or cy >= (height - 1.0) - half_h_final - 0.5
                        or cx <= half_w_final + 0.5
                        or cx >= (width - 1.0) - half_w_final - 0.5
                    )

                    if simple_tf:
                        bx_val = float(bx) if bx is not None else None
                        by_val = float(by) if by is not None else None
                        if bx_val is not None and not math.isfinite(bx_val):
                            bx_val = None
                    if by_val is not None and not math.isfinite(by_val):
                        by_val = None
                    simple_record = {
                        "t": float(t),
                        "cx": float(cx),
                        "cy": float(cy),
                        "bx": bx_val,
                        "by": by_val,
                        "zoom": float(zoom),
                        "zoom_out": float(edge_zoom_scale_follow),
                        "clamped": int(clamped_flag),
                    }
                    simple_tf.write(json.dumps(simple_record) + "\n")

                    if tf:
                        telemetry_rec = {
                            "t": float(t),
                            "used": used_tag,
                            "cx": float(cx),
                            "cy": float(cy),
                            "zoom": float(zoom),
                            "zoom_out": float(edge_zoom_scale_follow),
                        }
                    if used_tag == "offline_path":
                        telemetry_rec["zoom_edge_scale"] = float(edge_zoom_scale_follow)
                    else:
                        telemetry_rec["zoom_edge_scale"] = float(state.zoom_scale)
                    telemetry_rec["f"] = int(state.frame)
                    if used_tag == "planner":
                        if planner_spd is not None:
                            telemetry_rec["plan_spd"] = float(planner_spd)
                        if planner_zoom is not None:
                            telemetry_rec["plan_zoom"] = float(planner_zoom)
                    crop_vals = telemetry_crop if telemetry_crop is not None else (x0, y0, crop_w, crop_h)
                    crop_list = [
                        float(crop_vals[0]),
                        float(crop_vals[1]),
                        float(crop_vals[2]),
                        float(crop_vals[3]),
                    ]
                    telemetry_rec["crop"] = crop_list
                    telemetry_rec["crop_src"] = list(crop_list)

                    bx_src_val: Optional[float]
                    by_src_val: Optional[float]
                    if telemetry_ball_src is not None:
                        bx_src_val, by_src_val = telemetry_ball_src
                    else:
                        bx_src_val = None
                        by_src_val = None

                    if (
                        (bx_src_val is None or by_src_val is None)
                        and ball_path_rec_for_frame is not None
                    ):
                        bx_candidate, by_candidate = _get_ball_xy_src(
                            ball_path_rec_for_frame,
                            float(width),
                            float(height),
                        )
                        if bx_candidate is not None and by_candidate is not None:
                            bx_src_val, by_src_val = bx_candidate, by_candidate

                    if (
                        (bx_src_val is None or by_src_val is None)
                        and ball_path_entry is not None
                        and ball_path_space in (None, "raw", "ball", "generic")
                    ):
                        bx_src_val = float(ball_path_entry[0])
                        by_src_val = float(ball_path_entry[1])

                    if (
                        (bx_src_val is None or by_src_val is None)
                        and bx is not None
                        and by is not None
                    ):
                        bx_src_val = float(bx)
                        by_src_val = float(by)

                    if bx_src_val is not None and by_src_val is not None:
                        telemetry_ball_src = (bx_src_val, by_src_val)

                    out_w = float(output_size[0]) if output_size[0] else float(width)
                    out_h = float(output_size[1]) if output_size[1] else float(height)
                    if (
                        bx_src_val is not None
                        and by_src_val is not None
                        and crop_vals[2] > 0
                        and crop_vals[3] > 0
                        and out_w > 0
                        and out_h > 0
                    ):
                        crop_cx = float(cx)
                        crop_cy = float(cy)
                        crop_w = float(crop_vals[2])
                        crop_h = float(crop_vals[3])
                        crop_l = crop_cx - 0.5 * crop_w
                        crop_t = crop_cy - 0.5 * crop_h
                        sx = (bx_src_val - crop_l) / max(1e-6, crop_w)
                        sy = (by_src_val - crop_t) / max(1e-6, crop_h)
                        x_out = sx * out_w
                        y_out = sy * out_h
                        telemetry_ball_out = (x_out, y_out)
                    else:
                        telemetry_ball_out = (None, None)

                    if telemetry_ball_src is not None:
                        telemetry_rec["ball_src"] = [
                            float(telemetry_ball_src[0]),
                            float(telemetry_ball_src[1]),
                        ]
                        telemetry_rec["ball"] = [
                            float(telemetry_ball_src[0]),
                            float(telemetry_ball_src[1]),
                        ]
                    else:
                        telemetry_rec["ball_src"] = [None, None]
                        telemetry_rec["ball"] = [
                            float(bx) if bx is not None else float("nan"),
                            float(by) if by is not None else float("nan"),
                        ]

                    if telemetry_ball_out is not None:
                        telemetry_rec["ball_out"] = [
                            float(telemetry_ball_out[0]) if telemetry_ball_out[0] is not None else None,
                            float(telemetry_ball_out[1]) if telemetry_ball_out[1] is not None else None,
                        ]
                    else:
                        telemetry_rec["ball_out"] = [None, None]

                    telemetry_rec["ball_space"] = "source"
                    if ball_source_tag:
                        telemetry_rec["ball_src_tag"] = ball_source_tag
                        telemetry_rec["ball_src_name"] = ball_source_tag
                    if ball_path_entry is not None:
                        ball_path_x = float(ball_path_entry[0])
                        ball_path_y = float(ball_path_entry[1])
                        telemetry_rec["ball_path"] = [ball_path_x, ball_path_y]
                        if ball_path_space == "stab":
                            telemetry_rec["bx_stab"] = ball_path_x
                            telemetry_rec["by_stab"] = ball_path_y
                            telemetry_rec["ball_path_space"] = "stab"
                        elif ball_path_space == "raw":
                            telemetry_rec["bx_raw"] = ball_path_x
                            telemetry_rec["by_raw"] = ball_path_y
                            telemetry_rec["ball_path_space"] = "raw"
                        elif ball_path_space:
                            telemetry_rec["ball_path_space"] = ball_path_space
                    tf.write(json.dumps(to_jsonable(telemetry_rec)) + "\n")
                prev_ball_source = ball_source_tag

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
                    zoom_scale=state.zoom_scale,
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
            if simple_tf:
                simple_tf.close()
                self.telemetry_simple = None

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
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
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
                try:
                    bx_norm = float(val_x)
                    by_norm = float(val_y)
                except (TypeError, ValueError):
                    bx_norm = by_norm = None
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
                try:
                    rec[key_x] = float(val_x)
                    rec[key_y] = float(val_y)
                except (TypeError, ValueError):
                    continue

            rec["bx"] = bx_norm
            rec["by"] = by_norm

            z_value = data.get("z", default_zoom)
            try:
                rec["z"] = float(z_value)
            except (TypeError, ValueError):
                rec["z"] = float(default_zoom)

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
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

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
    try:
        duration_s = float(ffprobe_duration(input_path))
    except RuntimeError:
        duration_s = 0.0
    fps_out = float(args.fps) if args.fps is not None else float(preset_config.get("fps", fps_in))
    if fps_out <= 0:
        fps_out = fps_in if fps_in > 0 else 30.0

    follow_config_raw = preset_config.get("follow")
    follow_config: Mapping[str, object] = {}
    if isinstance(follow_config_raw, Mapping):
        follow_config = follow_config_raw

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
    if follow_smoothing is not None:
        try:
            smoothing_default = float(follow_smoothing)
        except (TypeError, ValueError):
            smoothing_default = preset_config.get("smoothing", 0.65)
    smoothing = float(args.smoothing) if args.smoothing is not None else float(smoothing_default)
    pad = float(args.pad) if args.pad is not None else float(preset_config.get("pad", 0.22))
    speed_limit = float(args.speed_limit) if args.speed_limit is not None else float(preset_config.get("speed_limit", 480))
    zoom_min = float(args.zoom_min) if args.zoom_min is not None else float(preset_config.get("zoom_min", 1.0))
    zoom_max = float(args.zoom_max) if args.zoom_max is not None else float(preset_config.get("zoom_max", 2.2))
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

    def _controller_float(key: str, fallback: float) -> float:
        value = _controller_value(key)
        if value is None:
            return fallback
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback

    def _controller_optional_float(key: str, fallback: Optional[float]) -> Optional[float]:
        value = _controller_value(key)
        if value is None:
            return fallback
        if isinstance(value, str) and value.strip().lower() in {"none", "", "null"}:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback

    def _controller_int(key: str, fallback: int) -> int:
        value = _controller_value(key)
        if value is None:
            return fallback
        try:
            return int(round(float(value)))
        except (TypeError, ValueError):
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
        float(args.deadzone)
        if args.deadzone is not None
        else _controller_float("deadzone", FOLLOW_DEFAULTS["deadzone"]),
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
    follow_lookahead_value = (
        int(args.follow_lookahead)
        if args.follow_lookahead is not None
        else _controller_int("lookahead", int(FOLLOW_DEFAULTS["lookahead"]))
    )
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
    if margin_val is not None:
        try:
            margin_px = max(0.0, float(margin_val))
        except (TypeError, ValueError):
            margin_px = 0.0

    zoom_out_max_default = follow_config.get("zoom_out_max") if follow_config else None
    follow_zoom_out_max = 1.35
    if zoom_out_max_default is not None:
        try:
            follow_zoom_out_max = max(1.0, float(zoom_out_max_default))
        except (TypeError, ValueError):
            follow_zoom_out_max = 1.35
    if getattr(args, "zoom_out_max", None) is not None:
        follow_zoom_out_max = max(1.0, float(args.zoom_out_max))

    zoom_edge_frac_default = follow_config.get("zoom_edge_frac") if follow_config else None
    follow_zoom_edge_frac = 0.80
    if zoom_edge_frac_default is not None:
        try:
            follow_zoom_edge_frac = float(zoom_edge_frac_default)
        except (TypeError, ValueError):
            follow_zoom_edge_frac = 0.80
    if getattr(args, "zoom_edge_frac", None) is not None:
        follow_zoom_edge_frac = float(args.zoom_edge_frac)
    if not math.isfinite(follow_zoom_edge_frac) or follow_zoom_edge_frac <= 0.0:
        follow_zoom_edge_frac = 1.0

    lead_time_s = 0.0
    lead_val = follow_config.get("lead_time") if follow_config else None
    if lead_val is not None:
        try:
            lead_time_s = max(0.0, float(lead_val))
        except (TypeError, ValueError):
            lead_time_s = 0.0
    lead_frames = int(round(lead_time_s * fps_out)) if fps_out > 0 else 0

    speed_zoom_value = follow_config.get("speed_zoom") if follow_config else None
    speed_zoom_config = speed_zoom_value if isinstance(speed_zoom_value, Mapping) else None

    default_ball_key_x = "bx_stab"
    default_ball_key_y = "by_stab"
    keys_value = follow_config.get("keys") if follow_config else None
    if isinstance(keys_value, Sequence) and len(keys_value) >= 2:
        try:
            default_ball_key_x = str(keys_value[0])
            default_ball_key_y = str(keys_value[1])
        except Exception:
            default_ball_key_x = "bx_stab"
            default_ball_key_y = "by_stab"

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
        margin_px=margin_px,
        lead_frames=lead_frames,
        speed_zoom=speed_zoom_config,
        min_box=portrait_min_box,
        horizon_lock=portrait_horizon_lock,
        emergency_gain=getattr(args, "emergency_gain", 0.6),
        emergency_zoom_max=getattr(args, "emergency_zoom_max", 1.45),
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
    jerk_threshold = float(getattr(args, "jerk_threshold", 0.0) or 0.0)
    jerk_enabled = jerk_threshold > 0.0
    jerk_wn_scale = float(getattr(args, "jerk_wn_scale", 0.9) or 0.9)
    jerk_deadzone_step = float(getattr(args, "jerk_deadzone_step", 2.0) or 2.0)
    jerk_max_attempts = max(1, int(getattr(args, "jerk_max_attempts", 3) or 3))

    current_follow_wn = float(follow_wn)
    current_deadzone = float(follow_deadzone)
    jerk95 = 0.0
    renderer: Optional[Renderer] = None

    attempt = 0
    while True:
        attempt += 1
        probe_renderer = Renderer(
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
            telemetry=None,
            telemetry_simple=None,
            init_manual=getattr(args, "init_manual", False),
            init_t=getattr(args, "init_t", 0.8),
            ball_path=offline_ball_path,
            follow_lead_time=lead_time_s,
            follow_margin_px=margin_px,
            follow_smoothing=smoothing,
            follow_zeta=follow_zeta,
            follow_wn=current_follow_wn,
            follow_deadzone=current_deadzone,
            follow_max_vel=follow_max_vel,
            follow_max_acc=follow_max_acc,
            follow_lookahead=follow_lookahead_frames,
            follow_pre_smooth=follow_pre_smooth,
            follow_zoom_out_max=follow_zoom_out_max,
            follow_zoom_edge_frac=follow_zoom_edge_frac,
        )
        jerk95 = probe_renderer.write_frames(states, probe_only=True)
        logging.info(
            "jerk95=%.1f px/s^3 (attempt %d/%d, follow_wn=%.2f, deadzone=%.1f)",
            jerk95,
            attempt,
            jerk_max_attempts,
            current_follow_wn,
            current_deadzone,
        )

        if (not jerk_enabled) or jerk95 <= jerk_threshold or attempt >= jerk_max_attempts:
            if jerk_enabled and jerk95 > jerk_threshold and attempt >= jerk_max_attempts:
                logging.warning(
                    "jerk95 %.1f exceeds threshold %.1f; reached max attempts, proceeding with last settings.",
                    jerk95,
                    jerk_threshold,
                )
            telemetry_handle: Optional[TextIO] = None
            telemetry_simple_handle: Optional[TextIO] = None
            try:
                if telemetry_path is not None:
                    telemetry_handle = open(telemetry_path, "w", encoding="utf-8")
                if telemetry_simple_path is not None:
                    telemetry_simple_handle = open(
                        telemetry_simple_path, "w", encoding="utf-8"
                    )
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
                    telemetry=telemetry_handle,
                    telemetry_simple=telemetry_simple_handle,
                    init_manual=getattr(args, "init_manual", False),
                    init_t=getattr(args, "init_t", 0.8),
                    ball_path=offline_ball_path,
                    follow_lead_time=lead_time_s,
                    follow_margin_px=margin_px,
                    follow_smoothing=smoothing,
                    follow_zeta=follow_zeta,
                    follow_wn=current_follow_wn,
                    follow_deadzone=current_deadzone,
                    follow_max_vel=follow_max_vel,
                    follow_max_acc=follow_max_acc,
                    follow_lookahead=follow_lookahead_frames,
                    follow_pre_smooth=follow_pre_smooth,
                    follow_zoom_out_max=follow_zoom_out_max,
                    follow_zoom_edge_frac=follow_zoom_edge_frac,
                )
                jerk95 = renderer.write_frames(states)
            finally:
                if telemetry_handle:
                    telemetry_handle.close()
                if telemetry_simple_handle:
                    telemetry_simple_handle.close()
            break

        logging.info(
            "jerk95 %.1f exceeds threshold %.1f; retuning follow_wn/deadzone", jerk95, jerk_threshold
        )
        current_follow_wn = max(0.1, current_follow_wn * max(jerk_wn_scale, 0.1))
        current_deadzone = max(0.0, current_deadzone + jerk_deadzone_step)

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
        default=1.35,
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
    parser.add_argument("--telemetry", dest="telemetry", help="Output JSONL telemetry file")
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
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(argv)
    follow_lookahead_cli = any(
        arg == "--lookahead" or arg.startswith("--lookahead=") for arg in raw_argv
    )
    setattr(args, "_follow_lookahead_cli", follow_lookahead_cli)
    # --- portrait helpers ---
    portrait_w, portrait_h = (None, None)
    if args.portrait:
        try:
            portrait_w, portrait_h = map(int, str(args.portrait).lower().split("x"))
        except Exception:
            raise SystemExit(
                f"Bad --portrait '{args.portrait}'. Use e.g. 1080x1920"
            )
    setattr(args, "portrait_w", portrait_w)
    setattr(args, "portrait_h", portrait_h)
    telemetry_path: Optional[Path] = None
    telemetry_simple_path: Optional[Path] = None
    if getattr(args, "telemetry", None):
        telemetry_path = Path(args.telemetry).expanduser()
        telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        args.telemetry = os.fspath(telemetry_path)
    if getattr(args, "telemetry_out", None):
        telemetry_simple_path = Path(args.telemetry_out).expanduser()
        telemetry_simple_path.parent.mkdir(parents=True, exist_ok=True)
        args.telemetry_out = os.fspath(telemetry_simple_path)
    run(args, telemetry_path=telemetry_path, telemetry_simple_path=telemetry_simple_path)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    try:
        main()
    except Exception as exc:  # pylint: disable=broad-except
        logging.error(str(exc))
        sys.exit(1)

"""Fit lightweight polynomial expressions for FFmpeg crop automation."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml


def _finite_mask(*arrs: np.ndarray) -> np.ndarray:
    m = np.ones_like(arrs[0], dtype=bool)
    for a in arrs:
        m &= np.isfinite(a)
    return m


def _nan_to(val, fallback):
    v = np.asarray(val, dtype=np.float64)
    bad = ~np.isfinite(v)
    if np.isscalar(fallback):
        v[bad] = float(fallback)
    else:
        fb = np.asarray(fallback, dtype=np.float64)
        v[bad] = fb[bad]
    return v


def safe_norm(v, eps: float = 1e-9) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1)
    return np.maximum(n, eps)


def safe_smoothstep(e0: float, e1: float, x) -> np.ndarray:
    x = np.clip((x - e0) / max(e1 - e0, 1e-9), 0.0, 1.0)
    return x * x * (3 - 2 * x)


def safe_polyfit(x, y, deg: int, min_pts: Optional[int] = None):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = _finite_mask(x, y)
    x2, y2 = x[m], y[m]
    need = (deg + 1) if min_pts is None else max(min_pts, deg + 1)
    if x2.size < need:
        for d in range(min(deg, 5), -1, -1):
            if x2.size >= d + 1:
                return np.polyfit(x2, y2, d).tolist(), d
        return [float(np.nanmean(y2)) if y2.size else 0.0], 0
    return np.polyfit(x2, y2, deg).tolist(), deg


def savgol_smooth_series(series: Sequence[float], window: int, order: int = 3) -> np.ndarray:
    arr = np.asarray(series, dtype=np.float64)
    n = len(arr)
    if n == 0 or window <= 2:
        return arr.copy()
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    window = min(window, n if n % 2 == 1 else n - 1 if n > 1 else n)
    if window <= 2:
        return arr.copy()
    order = int(max(1, min(order, window - 1)))
    half = window // 2
    smoothed = np.empty_like(arr)
    for idx in range(n):
        start = max(0, idx - half)
        end = min(n, start + window)
        if end - start < window:
            start = max(0, end - window)
        segment = arr[start:end]
        if segment.size <= order:
            smoothed[idx] = arr[idx]
            continue
        x = np.arange(segment.size, dtype=np.float64)
        try:
            coeffs = np.polyfit(x, segment, order)
        except np.linalg.LinAlgError:
            smoothed[idx] = arr[idx]
            continue
        rel = idx - start
        smoothed[idx] = float(np.polyval(coeffs, rel))
    return smoothed


def extract_cli_int(metadata: Dict[str, str], key: str) -> Optional[int]:
    cli = metadata.get("cli")
    if not cli:
        return None
    for token in cli.split():
        if token.startswith(f"{key}="):
            _, value = token.split("=", 1)
            try:
                return int(float(value))
            except ValueError:
                return None
    return None


def deep_update(base: Dict, override: Dict) -> Dict:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_profile_config(config_path: Path, profile: str, roi: str) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    defaults = data.get("defaults", {})
    profiles = data.get("profiles", {})
    rois = data.get("roi", {})

    cfg: Dict[str, Any] = dict(defaults)
    if profile in profiles:
        cfg = deep_update(cfg, profiles[profile])
    if roi in rois:
        cfg = deep_update(cfg, rois[roi])
    return cfg


def load_zoom_bounds(config_path: Path, profile: str, roi: str) -> Tuple[float, float]:
    cfg = load_profile_config(config_path, profile, roi)
    zoom_cfg = cfg.get("zoom", {})
    z_min = float(zoom_cfg.get("min", 1.05))
    z_max = float(zoom_cfg.get("max", 2.4))
    if z_min > z_max:
        z_min, z_max = z_max, z_min
    return z_min, z_max


def read_track(
    csv_path: Path,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Dict[str, str],
    Dict[str, np.ndarray],
]:
    metadata: Dict[str, str] = {}
    frames: List[int] = []
    cx: List[float] = []
    cy: List[float] = []
    zoom: List[float] = []
    extras_lists: Dict[str, List[float]] = {}

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        header_row: Optional[List[str]] = None
        while True:
            line = handle.readline()
            if not line:
                break
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                payload = stripped[1:].strip()
                if payload:
                    for part in payload.split(","):
                        if "=" in part:
                            key, value = part.split("=", 1)
                            metadata[key.strip()] = value.strip()
                continue
            header_row = next(csv.reader([line]))
            break

        if header_row is None:
            raise SystemExit(f"No header row found in {csv_path}")

        reader = csv.DictReader(handle, fieldnames=header_row)
        base_fields = {"frame", "cx", "cy", "z"}
        extras_fields = [key for key in header_row if key not in base_fields]
        extras_lists = {key: [] for key in extras_fields}
        for row in reader:
            if row is None:
                continue
            frame_str = row.get("frame")
            cx_str = row.get("cx")
            cy_str = row.get("cy")
            z_str = row.get("z")
            if not frame_str or cx_str is None or cy_str is None or z_str is None:
                continue
            try:
                frame_val = int(float(frame_str))
                cx_val = float(cx_str)
                cy_val = float(cy_str)
                z_val = float(z_str)
            except ValueError:
                continue
            frames.append(frame_val)
            cx.append(cx_val)
            cy.append(cy_val)
            zoom.append(z_val)
            for key in extras_fields:
                value = row.get(key)
                if value is None or value == "":
                    extras_lists[key].append(float("nan"))
                    continue
                try:
                    extras_lists[key].append(float(value))
                except ValueError:
                    extras_lists[key].append(float("nan"))

    if not frames:
        raise SystemExit(f"No rows found in {csv_path}")

    order = np.argsort(frames)
    frames_arr = np.asarray(frames, dtype=np.int64)[order]
    cx_arr = np.asarray(cx, dtype=np.float64)[order]
    cy_arr = np.asarray(cy, dtype=np.float64)[order]
    zoom_arr = np.asarray(zoom, dtype=np.float64)[order]
    extras = {
        key: np.asarray(values, dtype=np.float64)[order]
        for key, values in extras_lists.items()
        if values
    }
    return frames_arr, cx_arr, cy_arr, zoom_arr, metadata, extras
def ema_adaptive(x: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        a = float(np.clip(alpha[i], 0.0, 1.0))
        y[i] = a * x[i] + (1.0 - a) * y[i - 1]
    return y


def lead_signal(x: np.ndarray, frames: int) -> np.ndarray:
    if frames <= 0:
        return x.copy()
    lead = np.empty_like(x)
    if frames >= len(x):
        lead.fill(x[-1])
        return lead
    lead[:-frames] = x[frames:]
    lead[-frames:] = x[-1]
    return lead


def smoothstep(edge0: float, edge1: float, value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if edge0 == edge1:
        return np.clip((arr >= edge1).astype(float), 0.0, 1.0)
    t = np.clip((arr - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def rolling_min_streak(mask: np.ndarray, streak: int) -> np.ndarray:
    if streak <= 1:
        return mask.astype(bool)
    out = np.zeros(mask.shape, dtype=bool)
    counter = 0
    for idx, flag in enumerate(mask.astype(bool)):
        if flag:
            counter += 1
        else:
            counter = 0
        out[idx] = counter >= streak
    return out


def apply_deadzone(x: np.ndarray, dead: float) -> np.ndarray:
    if dead <= 0:
        return x.copy()
    y = x.copy()
    ref = float(y[0])
    for i in range(1, len(y)):
        delta = float(y[i] - ref)
        if abs(delta) < dead:
            y[i] = ref
        else:
            ref = float(y[i])
    return y


def limit_speed(x: np.ndarray, max_delta: float) -> np.ndarray:
    if max_delta <= 0:
        return x.copy()
    y = x.copy()
    for i in range(1, len(y)):
        delta = float(y[i] - y[i - 1])
        delta = np.clip(delta, -max_delta, max_delta)
        y[i] = y[i - 1] + delta
    return y


def apply_zoom_hysteresis_asym(
    z: np.ndarray, tighten_rate: float, widen_rate: float
) -> np.ndarray:
    if tighten_rate <= 0 and widen_rate <= 0:
        return z.copy()
    out = z.copy()
    for i in range(1, len(out)):
        delta = float(out[i] - out[i - 1])
        limit = widen_rate if delta > 0 else tighten_rate
        if limit > 0:
            delta = np.clip(delta, -limit, limit)
        out[i] = out[i - 1] + delta
    return out
def compute_ball_velocity(
    ball_x: np.ndarray, ball_y: np.ndarray, time_s: np.ndarray
) -> np.ndarray:
    if ball_x.size == 0 or time_s.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    bx = _nan_to(ball_x, 0.0)
    by = _nan_to(ball_y, 0.0)
    vx = _nan_to(np.gradient(bx, time_s), 0.0)
    vy = _nan_to(np.gradient(by, time_s), 0.0)
    vel = np.stack([vx, vy], axis=1)
    return vel


def clamp_scalar(value: float, maximum: float) -> float:
    if maximum <= 0:
        return 0.0
    return float(np.clip(value, -maximum, maximum))


def mix_scalar(a: float, b: float, weight: float) -> float:
    w = float(np.clip(weight, 0.0, 1.0))
    return (1.0 - w) * float(a) + w * float(b)


def euclidean_distance(ax: float, ay: float, bx: float, by: float) -> float:
    return float(np.hypot(ax - bx, ay - by))


def toward_goal(ball_vel: np.ndarray, goal_vec: np.ndarray) -> np.ndarray:
    speed = safe_norm(ball_vel)
    goal_mag = safe_norm(goal_vec)
    denom = speed * goal_mag
    dots = np.sum(ball_vel * goal_vec, axis=1)
    cos = np.divide(dots, denom, out=np.zeros_like(dots), where=denom > 0)
    return np.clip(cos, -1.0, 1.0)


def apply_boundary_cushion(
    cx: np.ndarray,
    cy: np.ndarray,
    z: np.ndarray,
    frame_size: Optional[Tuple[int, int]],
    margin_ratio: float = 0.08,
) -> Tuple[np.ndarray, np.ndarray]:
    if frame_size is None or margin_ratio <= 0:
        return cx.copy(), cy.copy()
    width, height = frame_size
    cx_out = cx.copy()
    cy_out = cy.copy()
    for i in range(len(cx_out)):
        zoom = float(max(z[i], 1e-6))
        crop_w = width / zoom
        crop_h = height / zoom
        half_w = crop_w * 0.5
        half_h = crop_h * 0.5
        margin_x = crop_w * margin_ratio
        margin_y = crop_h * margin_ratio
        min_cx = half_w + margin_x
        max_cx = width - half_w - margin_x
        min_cy = half_h + margin_y
        max_cy = height - half_h - margin_y
        if min_cx > max_cx:
            min_cx = max_cx = width * 0.5
        if min_cy > max_cy:
            min_cy = max_cy = height * 0.5
        cx_out[i] = float(np.clip(cx_out[i], min_cx, max_cx))
        cy_out[i] = float(np.clip(cy_out[i], min_cy, max_cy))
    return cx_out, cy_out


def _fit_len(a, N: int, fill: float = 0.0) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).ravel()
    if a.size == N:
        return a
    if a.size > N:
        return a[:N]
    out = np.full((N,), fill, dtype=np.float64)
    out[: a.size] = a
    return out


def _all_finite(*coefs) -> bool:
    return all(np.all(np.isfinite(np.asarray(c))) for c in coefs)


def format_coeff(value: float) -> str:
    if abs(value) < 1e-9:
        value = 0.0
    return f"{value:.8g}"


def build_expr(coeffs: Sequence[float]) -> str:
    terms: List[str] = []
    for power, coeff in enumerate(coeffs):
        coeff_str = format_coeff(float(coeff))
        if power == 0:
            terms.append(f"({coeff_str})")
        else:
            n_term = "*".join(["n"] * power)
            terms.append(f"({coeff_str})*{n_term}")
    joined = "+".join(terms)
    return f"({joined})"


def format_run_flags(args: argparse.Namespace) -> str:
    parts: List[str] = []
    for key in sorted(vars(args)):
        value = getattr(args, key)
        if isinstance(value, Path):
            value = str(value)
        parts.append(f"{key}={value}")
    return " ".join(parts)


def write_ps1vars(
    out_path: Path,
    cx_expr: str,
    cy_expr: str,
    z_expr: str,
    flag_lines: Sequence[str],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write("# Auto-generated polynomial expressions\n")
        for line in flag_lines:
            handle.write(f"# {line}\n")
        handle.write(f"$cxExpr = \"{cx_expr}\"\n")
        handle.write(f"$cyExpr = \"{cy_expr}\"\n")
        handle.write(f"$zExpr = \"{z_expr}\"\n")


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=Path("configs/zoom.yaml"))
    config_parser.add_argument("--profile", choices=["portrait", "landscape"], default="portrait")
    config_parser.add_argument("--roi", choices=["generic", "goal"], default="generic")

    config_args, _ = config_parser.parse_known_args()
    profile_cfg = load_profile_config(config_args.config, config_args.profile, config_args.roi)

    parser = argparse.ArgumentParser(
        description="Fit FFmpeg-ready expressions from motion tracks",
        parents=[config_parser],
    )
    parser.add_argument("--csv", type=Path, required=True, help="CSV generated by autoframe.py")
    parser.add_argument("--out", type=Path, required=True, help="Destination .ps1vars file")
    parser.add_argument("--degree", type=int, default=5, help="Base polynomial degree for cx/cy")
    parser.add_argument("--no-clip", action="store_true", help="Skip wrapping z polynomial in clip()")
    parser.add_argument("--lead-ms", type=int, default=120, help="Frame lead in milliseconds")
    parser.add_argument("--alpha-slow", type=float, default=0.08, help="EMA alpha when play is slow")
    parser.add_argument("--alpha-fast", type=float, default=0.40, help="EMA alpha when play is fast")
    parser.add_argument("--z-tight", type=float, default=2.2, help="Tight zoom factor")
    parser.add_argument("--z-wide", type=float, default=1.18, help="Wide zoom factor")
    parser.add_argument(
        "--celebration-ms",
        type=int,
        default=1400,
        help="Window after a detected shot to tighten on scorer",
    )
    parser.add_argument(
        "--celebration-tight",
        type=float,
        default=2.35,
        help="Zoom level during celebration lock-in",
    )
    parser.add_argument(
        "--zoom-tighten-rate",
        type=float,
        default=0.015,
        help="Max per-frame tighten step",
    )
    parser.add_argument(
        "--zoom-widen-rate",
        type=float,
        default=0.035,
        help="Max per-frame widen step (faster than tighten)",
    )
    parser.add_argument(
        "--boot-wide-ms",
        type=int,
        default=1600,
        help="Keep wide at start until lock or time passes",
    )
    parser.add_argument(
        "--snap-accel-th",
        type=float,
        default=0.75,
        help="Normalized acceleration threshold for zoom snaps",
    )
    parser.add_argument(
        "--snap-widen",
        type=float,
        default=0.18,
        help="Additional zoom-out applied during acceleration spikes",
    )
    parser.add_argument(
        "--snap-decay-ms",
        type=int,
        default=200,
        help="Decay window for acceleration-triggered zoom snaps",
    )
    parser.add_argument("--deadzone", type=float, default=6.0, help="Deadzone in pixels for pan stability")
    parser.add_argument(
        "--boot-anchor",
        choices=["first", "mean", "median"],
        default="median",
        help="Statistic used to anchor boot centering",
    )
    parser.add_argument(
        "--boot-anchor-frames",
        type=int,
        default=24,
        help="Frames considered when establishing boot anchor",
    )
    parser.add_argument(
        "--conf-th",
        type=float,
        default=0.35,
        help="Confidence threshold for trusting ball detections",
    )
    parser.add_argument("--v-enabled", action="store_true", help="Enable vertical steering adjustments")
    parser.add_argument(
        "--v-gain",
        type=float,
        default=0.85,
        help="Gain applied to vertical corrections when steering",
    )
    parser.add_argument(
        "--v-deadzone",
        type=float,
        default=6.0,
        help="Deadzone for vertical steering to avoid jitter",
    )
    parser.add_argument(
        "--v-top-margin",
        type=int,
        default=40,
        help="Minimum top margin maintained during vertical steering",
    )
    parser.add_argument(
        "--v-bottom-margin",
        type=int,
        default=40,
        help="Minimum bottom margin maintained during vertical steering",
    )
    parser.add_argument(
        "--snap-hold-ms",
        type=int,
        default=120,
        help="Duration to hold snap zoom and reduced lead after trigger",
    )
    parser.add_argument(
        "--goal-bias",
        type=float,
        default=0.0,
        help="Additional framing bias toward goal mouth on shot snaps",
    )
    config_defaults: Dict[str, Any] = {}
    for key in ("v_enabled", "v_gain", "v_deadzone", "v_top_margin", "v_bottom_margin"):
        if key not in profile_cfg:
            continue
        value = profile_cfg[key]
        if key == "v_enabled":
            config_defaults[key] = bool(value)
        else:
            config_defaults[key] = value
    if config_defaults:
        parser.set_defaults(**config_defaults)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    (
        frames,
        cx,
        cy,
        zoom,
        metadata,
        extras,
    ) = read_track(args.csv)
    z_min, z_max = load_zoom_bounds(args.config, args.profile, args.roi)

    if "zoom_min" in metadata:
        try:
            z_min = float(metadata["zoom_min"])
        except ValueError:
            pass
    if "zoom_max" in metadata:
        try:
            z_max = float(metadata["zoom_max"])
        except ValueError:
            pass
    if z_min > z_max:
        z_min, z_max = z_max, z_min

    zoom = np.clip(zoom, z_min, z_max)

    smooth_win = extract_cli_int(metadata, "smooth_win")
    if smooth_win is not None:
        if smooth_win < 0:
            smooth_win = 0
        if smooth_win and smooth_win % 2 == 0:
            smooth_win += 1
        if smooth_win and smooth_win > 2:
            cx = savgol_smooth_series(cx, smooth_win)
            cy = savgol_smooth_series(cy, smooth_win)

    try:
        fps = float(metadata.get("fps", 0.0))
    except (TypeError, ValueError):
        fps = 0.0
    if fps <= 0.0:
        fps = 30.0
    dt = 1.0 / max(fps, 1e-6)

    vx = np.gradient(cx, dt)
    vy = np.gradient(cy, dt)
    speed = np.hypot(vx, vy)
    accel = np.gradient(speed, dt)

    s_hi = np.percentile(speed, 85)
    a_hi = np.percentile(np.abs(accel), 85)
    s_n = np.clip(speed / (s_hi + 1e-6), 0.0, 1.0)
    a_n = np.clip(np.abs(accel) / (a_hi + 1e-6), 0.0, 1.0)

    alpha = args.alpha_slow + (args.alpha_fast - args.alpha_slow) * np.maximum(s_n, a_n)
    alpha = np.clip(alpha, 0.0, 1.0)

    cx_s = ema_adaptive(cx, alpha)
    cy_s = ema_adaptive(cy, alpha)

    lead_frames = int(round(fps * args.lead_ms / 1000.0))
    snap_hold_frames = int(round(fps * max(args.snap_hold_ms, 0) / 1000.0))
    snap_lead_ms = max(args.lead_ms - 20, 0)
    snap_lead_frames = int(round(fps * snap_lead_ms / 1000.0))

    BOOT_WIDE_MS = float(args.boot_wide_ms)
    LOCK_CONF = 0.60
    LOCK_STREAK = 6
    PLAYER_RADIUS_PX = 95.0
    PAN_CAP = 55.0
    TOWARD_GOAL_CO = 0.55
    GOAL_BIAS_MAX = 0.35
    FINAL_MS = 1600.0
    BASE_LEAD_ALPHA = 0.35
    SHOT_TOWARD_CO = 0.65
    SHOT_SPEED_Q = 88.0
    SHOT_WIDEN = 0.20

    time_ms = (frames - frames[0]).astype(np.float64) * (1000.0 / fps)
    time_s = time_ms / 1000.0
    remaining_ms = (frames[-1] - frames).astype(np.float64) * (1000.0 / fps)
    N = int(frames.shape[0])

    field_center_x: Optional[float] = None
    field_center_y: Optional[float] = None
    frame_size: Optional[Tuple[int, int]] = None
    try:
        width = int(float(metadata.get("width", 0)))
        height = int(float(metadata.get("height", 0)))
        if width > 0 and height > 0:
            frame_size = (width, height)
            field_center_x = width * 0.5
            field_center_y = height * 0.5
    except (TypeError, ValueError):
        frame_size = None
    if field_center_x is None:
        field_center_x = float(np.nanmean(cx))
    if field_center_y is None:
        field_center_y = float(np.nanmean(cy))
    if np.isnan(field_center_x):
        field_center_x = 0.0
    if np.isnan(field_center_y):
        field_center_y = 0.0

    def extract_series(names: Sequence[str]) -> Optional[np.ndarray]:
        for name in names:
            series = extras.get(name)
            if series is not None and series.shape[0] == frames.shape[0]:
                return series.astype(np.float64, copy=True)
        return None

    ball_conf = extract_series(["ball_conf", "ball_confidence"])
    if ball_conf is None or np.isnan(ball_conf).all():
        ball_conf = np.ones_like(cx_s)
    else:
        np.nan_to_num(ball_conf, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
        np.clip(ball_conf, 0.0, 1.0, out=ball_conf)
    ball_conf = _fit_len(ball_conf, N, 0.0)

    ball_x = extract_series(["ball_x", "ball_cx", "ball_px"])
    ball_y = extract_series(["ball_y", "ball_cy", "ball_py"])
    if ball_x is None or ball_y is None:
        ball_x = np.full((N,), np.nan, dtype=np.float64)
        ball_y = np.full((N,), np.nan, dtype=np.float64)
    ball_x = _fit_len(ball_x, N, field_center_x)
    ball_y = _fit_len(ball_y, N, field_center_y)

    def compute_boot_anchor(series: np.ndarray, fallback: float) -> float:
        frames_limit = max(int(args.boot_anchor_frames), 1)
        if series.size == 0:
            return fallback
        count = min(series.size, frames_limit)
        conf_slice = ball_conf[:count]
        values = series[:count]
        finite = np.isfinite(values)
        mask = finite & (conf_slice >= args.conf_th)
        if mask.any():
            candidates = values[mask]
        else:
            candidates = values[finite]
        if candidates.size == 0:
            return fallback
        if args.boot_anchor == "median":
            return float(np.median(candidates))
        if args.boot_anchor == "mean":
            return float(np.mean(candidates))
        idxs = np.where(mask if mask.any() else finite)[0]
        if idxs.size:
            return float(values[idxs[0]])
        return fallback

    boot_seed_x = compute_boot_anchor(ball_x, float(field_center_x))
    boot_seed_y = compute_boot_anchor(ball_y, float(field_center_y))

    goal_x = extract_series(["goal_x", "goal_cx"])
    goal_y = extract_series(["goal_y", "goal_cy"])
    goal_w = extract_series(["goal_w", "goal_width"])
    goal_h = extract_series(["goal_h", "goal_height"])

    goal_center_x = None
    goal_center_y = None
    if goal_x is not None and goal_w is not None:
        goal_center_x = goal_x + goal_w * 0.5
    elif goal_x is not None:
        goal_center_x = goal_x
    if goal_y is not None and goal_h is not None:
        goal_center_y = goal_y + goal_h * 0.5
    elif goal_y is not None:
        goal_center_y = goal_y

    def collect_player_tracks(prefixes: Sequence[str]) -> List[np.ndarray]:
        tracks: List[np.ndarray] = []
        for name, values in extras.items():
            if not name.endswith("_x"):
                continue
            base = name[:-2]
            if not any(base.startswith(prefix) for prefix in prefixes):
                continue
            y_key = f"{base}_y"
            y_values = extras.get(y_key)
            if y_values is None or y_values.shape[0] != values.shape[0]:
                continue
            tracks.append(
                np.stack([values.astype(np.float64), y_values.astype(np.float64)], axis=1)
            )
        return tracks

    player_tracks = collect_player_tracks(["player", "runner", "attacker", "defender"])

    boot_phase = time_ms < BOOT_WIDE_MS
    locked = rolling_min_streak(ball_conf > LOCK_CONF, LOCK_STREAK)

    ball_vel = compute_ball_velocity(ball_x, ball_y, time_s)
    ball_speed = safe_norm(ball_vel)
    shot_speed_q = (
        float(np.nanpercentile(ball_speed[ball_speed > 0], SHOT_SPEED_Q))
        if np.any(ball_speed > 0)
        else 0.0
    )
    shot_speed_threshold = max(shot_speed_q, 1e-6)

    if goal_center_x is not None:
        goal_center_x = _fit_len(goal_center_x, N, field_center_x)
    if goal_center_y is not None:
        goal_center_y = _fit_len(goal_center_y, N, field_center_y)

    cx_s = _nan_to(cx_s, field_center_x)
    cy_s = _nan_to(cy_s, field_center_y)
    cx_lead = lead_signal(cx_s, lead_frames)
    cy_lead = lead_signal(cy_s, lead_frames)
    cx_lead_snap = lead_signal(cx_s, snap_lead_frames)
    cy_lead_snap = lead_signal(cy_s, snap_lead_frames)
    cy_base = 0.85 * cy_s + 0.15 * cy_lead
    cy_base_snap = 0.85 * cy_s + 0.15 * cy_lead_snap

    gx = (
        _nan_to(goal_center_x, field_center_x)
        if goal_center_x is not None
        else np.full((N,), field_center_x, dtype=np.float64)
    )
    gy = (
        _nan_to(goal_center_y, field_center_y)
        if goal_center_y is not None
        else np.full((N,), field_center_y, dtype=np.float64)
    )
    goal_vec = np.stack(
        [
            gx - _nan_to(ball_x, field_center_x),
            gy - _nan_to(ball_y, field_center_y),
        ],
        axis=1,
    )
    cos_toward = np.sum(ball_vel * goal_vec, axis=1) / (
        safe_norm(ball_vel) * safe_norm(goal_vec)
    )
    cos_toward = np.clip(cos_toward, -1.0, 1.0)

    aligned = safe_smoothstep(0.65, 1.0, cos_toward)
    fast_enough = (ball_speed > shot_speed_threshold).astype(np.float64)
    shot_like = np.clip(aligned * fast_enough, 0.0, 1.0)
    shot_like[~np.isfinite(shot_like)] = 0.0
    shot_like = _fit_len(shot_like, N, 0.0)

    spike = (a_n > args.snap_accel_th).astype(float)
    tau = int(round(args.snap_decay_ms * fps / 1000.0))
    if tau <= 0:
        tau = 1
    kernel = np.exp(-np.arange(0, 3 * tau) / max(tau, 1))
    snap = np.convolve(spike, kernel, mode="same")
    if snap.max() > 0:
        snap = np.clip(snap / snap.max(), 0.0, 1.0)

    snap_arr = _fit_len(snap, N, 0.0)
    spike_arr = _fit_len(spike, N, 0.0)
    snap_hold_mask = np.zeros(N, dtype=bool)
    snap_envelope = snap_arr.copy()
    if snap_hold_frames > 0:
        for trigger_idx in np.where(spike_arr > 0.0)[0]:
            end = min(N, trigger_idx + snap_hold_frames + 1)
            snap_hold_mask[trigger_idx:end] = True
            base_level = float(snap_arr[trigger_idx]) if trigger_idx < snap_arr.size else 0.0
            level = max(base_level, 1.0)
            snap_envelope[trigger_idx:end] = np.maximum(snap_envelope[trigger_idx:end], level)
    else:
        snap_hold_mask = spike_arr > 0.0
        snap_envelope = np.maximum(snap_arr, snap_hold_mask.astype(np.float64))
    np.clip(snap_envelope, 0.0, 1.0, out=snap_envelope)

    combo = 0.6 * shot_like + 0.4 * (
        snap if snap.size == shot_like.size else np.zeros_like(shot_like)
    )
    shot_idx = int(np.argmax(combo)) if combo.size else None

    celebration_mask = np.zeros_like(cx_s, dtype=bool)
    if shot_idx is not None and combo[shot_idx] > 0.35:
        shot_t = time_ms[shot_idx]
        celebration_mask = (time_ms >= shot_t) & (
            time_ms <= shot_t + float(args.celebration_ms)
        )

    def goal_target_for_frame(idx: int) -> Tuple[float, float]:
        gx = field_center_x
        gy = field_center_y
        if goal_center_x is not None and not np.isnan(goal_center_x[idx]):
            gx = float(goal_center_x[idx])
        else:
            if not np.isnan(ball_x[idx]) and frame_size is not None:
                gx = frame_size[0] * (0.12 if ball_x[idx] <= field_center_x else 0.88)
        if goal_center_y is not None and not np.isnan(goal_center_y[idx]):
            gy = float(goal_center_y[idx])
        return gx, gy

    cx_out = np.empty_like(cx_s)
    cy_out = np.empty_like(cy_s)
    initial_pan = boot_seed_x if np.isfinite(boot_seed_x) else float(cx_s[0])
    pan_prev = float(initial_pan)
    pan_prev_target = float(initial_pan)
    tilt_prev = float(
        boot_seed_y
        if args.v_enabled and np.isfinite(boot_seed_y)
        else (cy_base_snap[0] if snap_hold_mask[0] else cy_base[0])
    )
    if args.v_enabled and not np.isfinite(tilt_prev):
        tilt_prev = float(field_center_y)
    keep_wide_mask = np.zeros(cx_s.shape, dtype=bool)
    conf_primary = max(args.conf_th, 0.5)
    for idx in range(len(cx_s)):
        pan_target = float(cx_s[idx])
        cy_base_val = float(cy_base_snap[idx] if snap_hold_mask[idx] else cy_base[idx])
        if not np.isfinite(cy_base_val):
            cy_base_val = float(field_center_y)
        tilt_target = cy_base_val
        z_guard_wide = False
        boot_active = bool(boot_phase[idx] and not locked[idx])
        if boot_active:
            pan_target = float(boot_seed_x)
            if not np.isfinite(pan_target):
                pan_target = float(field_center_x)
            if frame_size is not None:
                min_x = frame_size[0] * 0.18
                max_x = frame_size[0] * 0.82
                pan_target = float(np.clip(pan_target, min_x, max_x))
            if args.v_enabled:
                tilt_target = float(boot_seed_y)
                if not np.isfinite(tilt_target):
                    tilt_target = float(field_center_y)
            z_guard_wide = True

        bx = float(ball_x[idx]) if not np.isnan(ball_x[idx]) else None
        by = float(ball_y[idx]) if not np.isnan(ball_y[idx]) else None
        conf = float(ball_conf[idx])
        if bx is not None and by is not None:
            if conf >= conf_primary:
                pan_target = bx
            else:
                nearest_pos: Optional[Tuple[float, float]] = None
                if player_tracks:
                    best_dist = float('inf')
                    for track in player_tracks:
                        px, py = track[idx]
                        if np.isnan(px) or np.isnan(py):
                            continue
                        d = euclidean_distance(px, py, bx, by)
                        if d < best_dist:
                            best_dist = d
                            nearest_pos = (float(px), float(py))
                if (
                    nearest_pos is not None
                    and best_dist < PLAYER_RADIUS_PX
                    and conf >= max(args.conf_th, 0.3)
                ):
                    pan_target = mix_scalar(pan_prev_target, nearest_pos[0], 0.15)
                else:
                    pan_target = pan_prev_target
            if args.v_enabled and conf >= args.conf_th:
                tilt_target = by

        if celebration_mask[idx]:
            ref_bx = (
                float(ball_x[shot_idx])
                if (
                    shot_idx is not None
                    and 0 <= shot_idx < ball_x.shape[0]
                    and not np.isnan(ball_x[shot_idx])
                )
                else pan_target
            )
            ref_by = (
                float(ball_y[shot_idx])
                if (
                    shot_idx is not None
                    and 0 <= shot_idx < ball_y.shape[0]
                    and not np.isnan(ball_y[shot_idx])
                )
                else field_center_y
            )
            best_px = None
            best_d = float('inf')
            for track in player_tracks:
                px, py = track[idx]
                if np.isnan(px) or np.isnan(py):
                    continue
                d = euclidean_distance(px, py, ref_bx, ref_by)
                if d < best_d:
                    best_d = d
                    best_px = float(px)
            if best_px is not None:
                pan_target = 0.8 * pan_target + 0.2 * best_px

        snap_triggered = bool(spike_arr[idx] > 0.0)
        if snap_triggered and args.goal_bias > 0.0 and frame_size is not None:
            iw, ih = frame_size
            left = max(0, idx - 12)
            recent = ball_x[left : idx + 1]
            finite_recent = recent[np.isfinite(recent)] if recent.size else np.array([], dtype=np.float64)
            if finite_recent.size >= 2:
                dir_right = np.median(np.diff(finite_recent[-12:])) > 0.0
            else:
                dir_right = pan_target >= field_center_x
            goal_bias_x = iw * (0.90 if dir_right else 0.10)
            goal_bias_y = ih * 0.50
            pan_target = (1.0 - args.goal_bias) * pan_target + args.goal_bias * goal_bias_x
            tilt_target = (1.0 - args.goal_bias) * tilt_target + args.goal_bias * goal_bias_y

        goal_xy = goal_target_for_frame(idx)
        goal_vec = np.array(
            [[goal_xy[0] - (bx if bx is not None else pan_target), goal_xy[1] - (by if by is not None else field_center_y)]]
        )
        shot_indicator = remaining_ms[idx] < FINAL_MS
        if (
            bx is not None
            and by is not None
            and conf >= 0.2
            and not np.isnan(ball_speed[idx])
        ):
            toward = toward_goal(ball_vel[[idx]], goal_vec)
            speed_ok = (
                ball_speed[idx] > shot_speed_threshold
                if shot_speed_threshold > 0.0
                else ball_speed[idx] > 0.0
            )
            shot_indicator = shot_indicator or (
                speed_ok and toward[0] > TOWARD_GOAL_CO
            )
        gamma = float(smoothstep(0.0, 1.0, 1.0 if shot_indicator else 0.0)) * GOAL_BIAS_MAX
        pan_target = (1.0 - gamma) * pan_target + gamma * goal_xy[0]

        delta = clamp_scalar(pan_target - pan_prev, PAN_CAP)
        pan_target = pan_prev + delta

        lead_alpha = BASE_LEAD_ALPHA if locked[idx] else 0.25
        lead_ref = float(cx_lead_snap[idx] if snap_hold_mask[idx] else cx_lead[idx])
        pan_target = (1.0 - lead_alpha) * pan_target + lead_alpha * lead_ref

        cx_out[idx] = pan_target
        if args.v_enabled:
            if not np.isfinite(tilt_target):
                tilt_target = float(field_center_y)
            dy = tilt_target - tilt_prev
            if abs(dy) < args.v_deadzone:
                tilt_target = tilt_prev
            else:
                tilt_target = tilt_prev + args.v_gain * dy
            tilt_prev = tilt_target
            cy_out[idx] = tilt_target
        else:
            cy_out[idx] = cy_base_val

        pan_prev = pan_target
        pan_prev_target = pan_target
        if z_guard_wide:
            keep_wide_mask[idx] = True

    widen_guard_mask = (time_ms < max(float(args.boot_wide_ms), 2500.0)) & (
        ball_conf < 0.35
    )
    if widen_guard_mask.any():
        np.logical_or(keep_wide_mask, widen_guard_mask, out=keep_wide_mask)

    pan_speed_cap = PAN_CAP
    cx_out = limit_speed(cx_out, pan_speed_cap)
    cy_out = limit_speed(cy_out, pan_speed_cap)

    cx_out = apply_deadzone(cx_out, args.deadzone)
    cy_out = apply_deadzone(cy_out, args.deadzone)

    k = np.clip(1.0 - s_n, 0.0, 1.0)
    z_base = args.z_wide + (args.z_tight - args.z_wide) * k

    z_bonus_widen = args.snap_widen * snap_envelope
    z_bonus_widen += SHOT_WIDEN * shot_like
    z = np.clip(z_base - z_bonus_widen, 1.10, 2.60)

    if keep_wide_mask.any():
        z[keep_wide_mask] = np.minimum(z[keep_wide_mask], args.z_wide)

    if celebration_mask.any():
        z[celebration_mask] = np.maximum(z[celebration_mask], args.celebration_tight)

    z = apply_zoom_hysteresis_asym(z, args.zoom_tighten_rate, args.zoom_widen_rate)

    z = _nan_to(z, 1.60)
    z = np.clip(z, 1.08, 2.80)

    if frame_size is not None:
        iw, ih = frame_size
        top_margin = max(args.v_top_margin, 0)
        bottom_margin = max(args.v_bottom_margin, 0)
        cx_safe = np.empty_like(cx_out)
        cy_safe = np.empty_like(cy_out)
        for idx in range(N):
            z_current = float(max(z[idx], 1e-6))
            if args.profile == "portrait":
                crop_h = ih / z_current
                crop_w = crop_h * 9.0 / 16.0
            else:
                crop_w = iw / z_current
                crop_h = crop_w * 9.0 / 16.0
            crop_w = min(crop_w, iw)
            crop_h = min(crop_h, ih)
            center_x = cx_out[idx]
            if not np.isfinite(center_x):
                center_x = float(field_center_x)
            x = float(np.clip(center_x - crop_w * 0.5, 0.0, iw - crop_w))
            if args.v_enabled:
                y_min = float(top_margin)
                y_max = float(ih - crop_h - bottom_margin)
                if y_min > y_max:
                    mid = max(0.0, (ih - crop_h) * 0.5)
                    y_min = y_max = mid
            else:
                y_min = 0.0
                y_max = float(ih - crop_h)
            center_y = cy_out[idx]
            if not np.isfinite(center_y):
                center_y = float(field_center_y)
            y = float(np.clip(center_y - crop_h * 0.5, y_min, y_max))
            cx_safe[idx] = x + crop_w * 0.5
            cy_safe[idx] = y + crop_h * 0.5
        cx_out = cx_safe
        cy_out = cy_safe

    cx_out, cy_out = apply_boundary_cushion(cx_out, cy_out, z, frame_size)

    z = np.clip(z, z_min, z_max)

    if frame_size is not None:
        iw, ih = frame_size
        safe_z = np.maximum(z, 1e-6)
        crop_h = ih / safe_z
        crop_w = (ih * 9.0 / 16.0) / safe_z
        half_w = 0.5 * crop_w
        half_h = 0.5 * crop_h
        max_x = np.maximum(iw - crop_w, 0.0)
        max_y = np.maximum(ih - crop_h, 0.0)
        cx_out = np.clip(cx_out - half_w, 0.0, max_x) + half_w
        cy_out = np.clip(cy_out - half_h, 0.0, max_y) + half_h

    cx_out = _nan_to(cx_out, field_center_x)
    cy_out = _nan_to(cy_out, field_center_y)

    n = np.arange(N, dtype=np.float64)
    conf = np.asarray(ball_conf, dtype=np.float64)
    x_raw = np.asarray(cx_out, dtype=np.float64)
    y_raw = np.asarray(cy_out, dtype=np.float64)
    deg = int(args.degree)

    mask = conf >= 0.35
    min_pts = max(6, deg + 2)
    if mask.sum() < min_pts:
        mask = conf >= 0.20
    if mask.sum() < max(2, deg + 1):
        mask = np.ones_like(conf, dtype=bool)

    nx = n[mask]
    x = x_raw[mask]
    y = y_raw[mask]

    nmu = float(nx.mean()) if nx.size else 0.0
    ns = nx - nmu

    def guarded_polyfit(t: np.ndarray, v: np.ndarray, d: int) -> Tuple[np.ndarray, int]:
        if t.size == 0:
            avg = float(np.nanmean(v)) if v.size else 0.0
            return np.array([avg], dtype=np.float64), 0
        d_eff = int(min(d, max(1, t.size - 1)))
        if t.size <= d_eff:
            d_eff = max(0, t.size - 1)
        if d_eff <= 0:
            avg = float(np.nanmean(v)) if v.size else 0.0
            return np.array([avg], dtype=np.float64), 0
        coef = np.polyfit(t, v, d_eff)
        return coef, d_eff

    cx_coef, cx_deg = guarded_polyfit(ns, x, deg)
    cy_coef, cy_deg = guarded_polyfit(ns, y, deg)

    def shift_poly(coef: np.ndarray, center: float) -> np.ndarray:
        q = np.poly1d([1.0, -center])
        composed = np.poly1d([0.0])
        for power, coeff in enumerate(coef[::-1]):
            composed += float(coeff) * q**power
        return composed.c

    cx_shifted = shift_poly(cx_coef, nmu)
    cy_shifted = shift_poly(cy_coef, nmu)

    cx_coeffs = list(reversed(cx_shifted))
    cy_coeffs = list(reversed(cy_shifted))

    z_coef, z_deg = safe_polyfit(n, z, args.degree)

    if cx_deg < args.degree or cy_deg < args.degree or z_deg < args.degree:
        print(f"[warn] lowered poly degree: cx={cx_deg}, cy={cy_deg}, z={z_deg}")

    z_coeffs = list(reversed(z_coef))

    if not _all_finite(cx_coeffs, cy_coeffs, z_coeffs):
        print("[warn] non-finite coefficients after fit; writing conservative default.")
        cx_coeffs = [960.0, 0.0, 0.0, 0.0]
        cy_coeffs = [540.0, 0.0, 0.0, 0.0]
        z_coeffs = [1.60, 0.0, 0.0, 0.0]

    cx_expr = build_expr(cx_coeffs)
    cy_expr = build_expr(cy_coeffs)
    z_expr_core = build_expr(z_coeffs)

    if args.no_clip:
        z_expr = z_expr_core
    else:
        z_expr = (
            f"(clip({z_expr_core},{format_coeff(z_min)},{format_coeff(z_max)}))"
        )

    flag_lines: List[str] = [f"fit: {format_run_flags(args)}"]
    cli_flags = metadata.get("cli")
    if cli_flags:
        flag_lines.append(f"track: {cli_flags}")

    write_ps1vars(args.out, cx_expr, cy_expr, z_expr, flag_lines)

    print(f"Frames: {len(frames)}  degree={args.degree}")
    print(f"cx: {cx_expr}")
    print(f"cy: {cy_expr}")
    print(f"z:  {z_expr}")


if __name__ == "__main__":
    main()

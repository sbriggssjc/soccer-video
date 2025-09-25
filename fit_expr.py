"""Fit lightweight polynomial expressions for FFmpeg crop automation."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml


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


def load_zoom_bounds(config_path: Path, profile: str, roi: str) -> Tuple[float, float]:
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    defaults = data.get("defaults", {})
    profiles = data.get("profiles", {})
    rois = data.get("roi", {})

    cfg = dict(defaults)
    if profile in profiles:
        cfg = deep_update(cfg, profiles[profile])
    if roi in rois:
        cfg = deep_update(cfg, rois[roi])

    zoom_cfg = cfg.get("zoom", {})
    z_min = float(zoom_cfg.get("min", 1.05))
    z_max = float(zoom_cfg.get("max", 2.4))
    if z_min > z_max:
        z_min, z_max = z_max, z_min
    return z_min, z_max


def read_track(
    csv_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, str]]:
    metadata: Dict[str, str] = {}
    frames: List[int] = []
    cx: List[float] = []
    cy: List[float] = []
    zoom: List[float] = []

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
                frames.append(int(float(frame_str)))
                cx.append(float(cx_str))
                cy.append(float(cy_str))
                zoom.append(float(z_str))
            except ValueError:
                continue

    if not frames:
        raise SystemExit(f"No rows found in {csv_path}")

    order = np.argsort(frames)
    frames_arr = np.asarray(frames, dtype=np.int64)[order]
    cx_arr = np.asarray(cx, dtype=np.float64)[order]
    cy_arr = np.asarray(cy, dtype=np.float64)[order]
    zoom_arr = np.asarray(zoom, dtype=np.float64)[order]
    return frames_arr, cx_arr, cy_arr, zoom_arr, metadata


def fit_poly(values: Sequence[float], degree: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("values must be 1-D")
    if arr.size < 2:
        raise ValueError("need at least two samples to fit polynomial")
    n = np.arange(len(arr), dtype=np.float64)
    max_degree = min(max(int(degree), 1), arr.size - 1)
    for deg in range(max_degree, 0, -1):
        try:
            coeffs = np.polyfit(n, arr, deg)
        except np.linalg.LinAlgError:
            continue
        else:
            return coeffs[::-1]  # convert to [a0, a1, ...]
    return np.array([float(arr.mean())])


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


def apply_zoom_hysteresis(z: np.ndarray, max_delta: float) -> np.ndarray:
    if max_delta <= 0:
        return z.copy()
    out = z.copy()
    for i in range(1, len(out)):
        delta = float(out[i] - out[i - 1])
        delta = np.clip(delta, -max_delta, max_delta)
        out[i] = out[i - 1] + delta
    return out


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
    parser = argparse.ArgumentParser(description="Fit FFmpeg-ready expressions from motion tracks")
    parser.add_argument("--csv", type=Path, required=True, help="CSV generated by autoframe.py")
    parser.add_argument("--out", type=Path, required=True, help="Destination .ps1vars file")
    parser.add_argument("--degree", type=int, default=5, help="Base polynomial degree for cx/cy")
    parser.add_argument("--profile", choices=["portrait", "landscape"], default="portrait")
    parser.add_argument("--roi", choices=["generic", "goal"], default="generic")
    parser.add_argument("--config", type=Path, default=Path("configs/zoom.yaml"))
    parser.add_argument("--no-clip", action="store_true", help="Skip wrapping z polynomial in clip()")
    parser.add_argument("--lead-ms", type=int, default=120, help="Frame lead in milliseconds")
    parser.add_argument("--alpha-slow", type=float, default=0.08, help="EMA alpha when play is slow")
    parser.add_argument("--alpha-fast", type=float, default=0.40, help="EMA alpha when play is fast")
    parser.add_argument("--z-tight", type=float, default=2.2, help="Tight zoom factor")
    parser.add_argument("--z-wide", type=float, default=1.18, help="Wide zoom factor")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames, cx, cy, zoom, metadata = read_track(args.csv)
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
    cx_lead = lead_signal(cx_s, lead_frames)
    cy_lead = lead_signal(cy_s, lead_frames)

    cx_out = 0.8 * cx_s + 0.2 * cx_lead
    cy_out = 0.85 * cy_s + 0.15 * cy_lead

    pan_speed_cap = 42.0
    cx_out = limit_speed(cx_out, pan_speed_cap)
    cy_out = limit_speed(cy_out, pan_speed_cap)

    cx_out = apply_deadzone(cx_out, args.deadzone)
    cy_out = apply_deadzone(cy_out, args.deadzone)

    k = np.clip(1.0 - s_n, 0.0, 1.0)
    z_base = args.z_wide + (args.z_tight - args.z_wide) * k

    spike = (a_n > args.snap_accel_th).astype(float)
    tau = int(round(args.snap_decay_ms * fps / 1000.0))
    if tau <= 0:
        tau = 1
    kernel = np.exp(-np.arange(0, 3 * tau) / max(tau, 1))
    snap = np.convolve(spike, kernel, mode="same")
    if snap.max() > 0:
        snap = np.clip(snap / snap.max(), 0.0, 1.0)
    z_bonus_widen = args.snap_widen * snap
    z = np.clip(z_base - z_bonus_widen, 1.10, 2.60)

    z = apply_zoom_hysteresis(z, 0.02)

    frame_size: Optional[Tuple[int, int]] = None
    try:
        width = int(float(metadata.get("width", 0)))
        height = int(float(metadata.get("height", 0)))
        if width > 0 and height > 0:
            frame_size = (width, height)
    except (TypeError, ValueError):
        frame_size = None

    cx_out, cy_out = apply_boundary_cushion(cx_out, cy_out, z, frame_size)

    z = np.clip(z, z_min, z_max)

    deg_xy = max(args.degree, 5)
    deg_z = max(4, min(args.degree, 6))
    cx_coeffs = fit_poly(cx_out, deg_xy)
    cy_coeffs = fit_poly(cy_out, deg_xy)
    z_coeffs = fit_poly(z, deg_z)

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

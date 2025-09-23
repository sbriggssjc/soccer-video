"""Utility helpers for autoframe coefficient fitting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class MotionBounds:
    """Bounding box for valid motion centers."""

    width: float
    height: float

    def clamp(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Clamp positions to the image bounds."""
        x_clamped = np.clip(x, 0.0, max(0.0, self.width - 1.0))
        y_clamped = np.clip(y, 0.0, max(0.0, self.height - 1.0))
        return x_clamped, y_clamped


def velocity_clamped_ema(values: Sequence[float], vmax: float, alpha: float = 0.2) -> np.ndarray:
    """Smooth a 1-D path with velocity-clamped exponential moving average."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("values must be a 1-D sequence")
    if len(arr) == 0:
        return arr.copy()

    out = np.empty_like(arr)
    current = arr[0]
    out[0] = current
    for i in range(1, len(arr)):
        delta = arr[i] - current
        if vmax > 0:
            delta = np.clip(delta, -vmax, vmax)
        current = current + delta
        current = current * (1.0 - alpha) + arr[i] * alpha
        out[i] = current
    return out


def _format_coeff(value: float) -> str:
    if abs(value) < 1e-10:
        return "0"
    return f"{value:.8g}"


def ffmpeg_polyfit(values: Sequence[float], degree: int = 2, var: str = "n") -> str:
    """Fit a polynomial and format it for FFmpeg expressions."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("values must be a 1-D sequence")
    if len(arr) <= degree:
        raise ValueError("need more samples than polynomial degree")

    n = np.arange(len(arr), dtype=np.float64)
    coeffs = np.polyfit(n, arr, degree)
    coeffs = coeffs.tolist()
    # np.polyfit returns highest degree first.
    expr = []
    if degree >= 2:
        expr.append(f"({_format_coeff(coeffs[0])})*{var}*{var}")
        expr.append(f"({_format_coeff(coeffs[1])})*{var}")
        expr.append(f"({_format_coeff(coeffs[2])})")
    else:
        for power, coeff in zip(range(degree, -1, -1), coeffs):
            if power == 0:
                expr.append(f"({_format_coeff(coeff)})")
            elif power == 1:
                expr.append(f"({_format_coeff(coeff)})*{var}")
            else:
                expr.append(f"({_format_coeff(coeff)})*{var}^{power}")
    joined = "+".join(expr)
    return f"({joined})"


def normalized_speed(x: Sequence[float], y: Sequence[float]) -> np.ndarray:
    """Compute normalized per-frame speed from path coordinates."""
    arr_x = np.asarray(x, dtype=np.float64)
    arr_y = np.asarray(y, dtype=np.float64)
    if arr_x.shape != arr_y.shape:
        raise ValueError("x and y must have the same shape")
    if arr_x.ndim != 1:
        raise ValueError("inputs must be 1-D sequences")

    dx = np.diff(arr_x, prepend=arr_x[0])
    dy = np.diff(arr_y, prepend=arr_y[0])
    speed = np.hypot(dx, dy)
    min_speed = speed.min()
    spread = speed.max() - min_speed
    if spread <= 1e-9:
        return np.zeros_like(speed)
    return (speed - min_speed) / spread

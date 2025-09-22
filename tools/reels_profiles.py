"""Output profile definitions and helpers for reel exports."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class Profile:
    """Definition of a social output profile."""

    name: str
    w: int
    h: int
    pad_color: str = "black"

    @property
    def aspect(self) -> float:
        return self.w / self.h


PROFILES = {
    "tiktok": Profile("tiktok", 1080, 1920),
    "instagram": Profile("instagram", 1080, 1350),
    "square": Profile("square", 1080, 1080),
    "landscape": Profile("landscape", 1920, 1080),
}


def _clamp_int(value: float, minimum: int, maximum: int) -> int:
    value_int = int(round(value))
    if value_int < minimum:
        return minimum
    if value_int > maximum:
        return maximum
    return value_int


def compute_focus_crop(
    src_w: int,
    src_h: int,
    profile: Profile,
    focus_x: Optional[float] = None,
    focus_y: Optional[float] = None,
    zoom: Optional[float] = None,
) -> Tuple[int, int, int, int]:
    """Return a crop box matching the profile aspect around a focal point."""

    if src_w <= 0 or src_h <= 0:
        raise ValueError("Source dimensions must be positive")

    zoom = float(zoom) if zoom not in (None, "") else 1.0
    if zoom <= 0:
        zoom = 1.0

    target_aspect = profile.aspect
    src_aspect = src_w / src_h

    if src_aspect > target_aspect:
        base_height = src_h
        base_width = base_height * target_aspect
    else:
        base_width = src_w
        base_height = base_width / target_aspect

    width = min(src_w, base_width / zoom)
    height = min(src_h, base_height / zoom)

    if focus_x in (None, ""):
        focus_x = src_w / 2
    if focus_y in (None, ""):
        focus_y = src_h / 2

    x = float(focus_x) - width / 2
    y = float(focus_y) - height / 2

    x = max(0.0, min(x, src_w - width))
    y = max(0.0, min(y, src_h - height))

    width_i = _clamp_int(width, 1, src_w)
    height_i = _clamp_int(height, 1, src_h)
    x_i = _clamp_int(x, 0, src_w - width_i)
    y_i = _clamp_int(y, 0, src_h - height_i)

    return width_i, height_i, x_i, y_i


def resolve_crop_box(
    src_w: int,
    src_h: int,
    profile: Profile,
    explicit: Optional[Tuple[float, float, float, float]] = None,
    focus_x: Optional[float] = None,
    focus_y: Optional[float] = None,
    zoom: Optional[float] = None,
) -> Optional[Tuple[int, int, int, int]]:
    """Resolve an explicit or focus/zoom-based crop box."""

    if explicit is not None:
        crop_w, crop_h, crop_x, crop_y = explicit
        crop_w_i = _clamp_int(crop_w, 1, src_w)
        crop_h_i = _clamp_int(crop_h, 1, src_h)
        crop_x_i = _clamp_int(crop_x, 0, src_w - crop_w_i)
        crop_y_i = _clamp_int(crop_y, 0, src_h - crop_h_i)
        return crop_w_i, crop_h_i, crop_x_i, crop_y_i

    if any(value not in (None, "") for value in (focus_x, focus_y, zoom)):
        return compute_focus_crop(src_w, src_h, profile, focus_x, focus_y, zoom)

    return None


def profile_filters(
    profile: Profile,
    crop_box: Optional[Tuple[int, int, int, int]] = None,
    pad_color: Optional[str] = None,
) -> List[str]:
    """Build the list of filters to crop, scale, and pad for a profile."""

    filters: List[str] = []
    if crop_box is not None:
        crop_w, crop_h, crop_x, crop_y = crop_box
        filters.append(
            f"crop=w={crop_w}:h={crop_h}:x={crop_x}:y={crop_y}"
        )

    filters.append(
        f"scale=w={profile.w}:h={profile.h}:force_original_aspect_ratio=decrease:flags=bicubic"
    )

    color = pad_color or profile.pad_color
    filters.append(
        f"pad=w={profile.w}:h={profile.h}:x=(ow-iw)/2:y=(oh-ih)/2:color={color}"
    )

    return filters


__all__ = ["Profile", "PROFILES", "compute_focus_crop", "resolve_crop_box", "profile_filters"]

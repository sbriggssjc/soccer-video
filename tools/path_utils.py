"""Helpers for deterministic output naming across render pipelines."""

from __future__ import annotations

import re
from typing import Iterable, List, Tuple

_DOT_TAG_RE = re.compile(r"\.__([A-Z0-9_]+)", flags=re.IGNORECASE)
_TRAILING_FINAL_RE = re.compile(r"(?:_portrait_FINAL)+$", flags=re.IGNORECASE)
_DOTS_RE = re.compile(r"\.{2,}")
_DOUBLE_UNDERSCORE_RE = re.compile(r"__{2,}")


def _extract_dot_tags(stem: str) -> Tuple[str, List[str]]:
    tags: List[str] = []
    seen: set[str] = set()

    def _collect(match: re.Match[str]) -> str:
        tag = match.group(1).upper()
        if tag not in seen:
            tags.append(tag)
            seen.add(tag)
        return ""

    base = _DOT_TAG_RE.sub(_collect, stem)
    return base, tags


def _clean_tag(tag: str) -> str:
    cleaned = tag.strip()
    if cleaned.startswith(".__"):
        cleaned = cleaned[3:]
    elif cleaned.startswith("__"):
        cleaned = cleaned[2:]
    elif cleaned.startswith("."):
        cleaned = cleaned[1:]
    return cleaned.strip().upper()


def normalize_stem(stem: str) -> str:
    """Normalize cosmetic tags and separators in a clip stem.

    - Collapse repeated cosmetic dot-tags (e.g., .__CINEMATIC) to a single instance.
    - Collapse repeated separators: multiple dots -> single dot, multiple "__" -> "__".
    - Remove any trailing "_portrait_FINAL" (so callers can re-append deterministically).
    """
    stem = _TRAILING_FINAL_RE.sub("", stem)
    base, tags = _extract_dot_tags(stem)
    base = _DOTS_RE.sub(".", base)
    base = _DOUBLE_UNDERSCORE_RE.sub("__", base)
    base = base.strip(".")
    if tags:
        base = f"{base}{''.join(f'.__{tag}' for tag in tags)}"
    return base


def build_output_stem(
    base_stem: str,
    preset: str,
    portrait: bool,
    is_final: bool,
    extra_tags: Iterable[str] | None,
) -> str:
    """Build a deterministic output stem from a base stem and render settings."""
    preset_label = preset.strip().upper() if preset else ""
    stem = _TRAILING_FINAL_RE.sub("", base_stem)
    stem, existing_tags = _extract_dot_tags(stem)
    if preset_label:
        stem = re.sub(fr"(?:__{re.escape(preset_label)})+", "", stem)

    seen_tags = set(existing_tags)
    extra_unique: List[str] = []
    for tag in extra_tags or []:
        cleaned = _clean_tag(tag)
        if not cleaned or cleaned in seen_tags:
            continue
        extra_unique.append(cleaned)
        seen_tags.add(cleaned)

    stem = _DOTS_RE.sub(".", stem)
    stem = _DOUBLE_UNDERSCORE_RE.sub("__", stem)
    stem = stem.strip(".")
    if preset_label:
        stem = f"{stem}__{preset_label}"
    if existing_tags or extra_unique:
        stem = f"{stem}{''.join(f'.__{tag}' for tag in existing_tags + extra_unique)}"

    stem = normalize_stem(stem)
    if is_final:
        final_suffix = "_portrait_FINAL" if portrait else "_FINAL"
        stem = f"{stem}{final_suffix}"
    return stem

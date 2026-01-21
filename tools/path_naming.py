"""Deterministic naming helpers for render outputs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List

_DOT_TAG_RE = re.compile(
    r"\.__[A-Z0-9]+(?:_[A-Z0-9]+)*(?:-[A-Za-z0-9]+)*(?=(?:__|\.|$|_portrait_FINAL))",
    flags=re.IGNORECASE,
)
_DOTS_RE = re.compile(r"\.{2,}")
_DOUBLE_UNDERSCORE_RE = re.compile(r"__{2,}")
_TRAILING_PORTRAIT_FINAL_RE = re.compile(r"(?:_portrait_FINAL)+$", flags=re.IGNORECASE)


def _clean_tag(tag: str) -> str:
    cleaned = tag.strip()
    if cleaned.startswith(".__"):
        cleaned = cleaned[3:]
    elif cleaned.startswith("__"):
        cleaned = cleaned[2:]
    elif cleaned.startswith("."):
        cleaned = cleaned[1:]
    return cleaned.strip().upper()


def normalize_tags_in_stem(stem: str) -> str:
    """Normalize cosmetic dot-tags and separators in a filename stem."""
    normalized = _DOTS_RE.sub(".", stem)
    normalized = _DOUBLE_UNDERSCORE_RE.sub("__", normalized)
    portrait_suffix = ""
    if _TRAILING_PORTRAIT_FINAL_RE.search(normalized):
        normalized = _TRAILING_PORTRAIT_FINAL_RE.sub("", normalized)
        portrait_suffix = "_portrait_FINAL"

    tags: List[str] = []
    seen: set[str] = set()
    for match in _DOT_TAG_RE.findall(normalized):
        cleaned = _clean_tag(match)
        if cleaned and cleaned not in seen:
            tags.append(cleaned)
            seen.add(cleaned)

    first_tag = _DOT_TAG_RE.search(normalized)
    first_tag_index = first_tag.start() if first_tag else None
    first_double_index = normalized.find("__")

    base_end = len(normalized)
    if first_tag_index is not None:
        base_end = min(base_end, first_tag_index)
    if first_double_index != -1:
        base_end = min(base_end, first_double_index)

    base = normalized[:base_end]
    suffix = normalized[base_end:]
    suffix = _DOT_TAG_RE.sub("", suffix)

    rebuilt = f"{base}{''.join(f'.__{tag}' for tag in tags)}{suffix}{portrait_suffix}"
    rebuilt = _DOTS_RE.sub(".", rebuilt)
    rebuilt = _DOUBLE_UNDERSCORE_RE.sub("__", rebuilt)
    return rebuilt


def build_output_name(
    input_path: str,
    preset: str,
    portrait: str | None,
    follow: str | None,
    is_final: bool,
    extra_tags: Iterable[str] | None,
) -> str:
    """Build a deterministic output filename (no directory)."""
    path = Path(input_path)
    suffix = path.suffix or ".mp4"
    preset_label = preset.strip().upper() if preset else ""

    stem = _TRAILING_PORTRAIT_FINAL_RE.sub("", path.stem)

    if preset_label:
        stem = re.sub(fr"(?:__{re.escape(preset_label)})+", "", stem, flags=re.IGNORECASE)

    tags: List[str] = []
    seen: set[str] = set()

    def _collect(match: re.Match[str]) -> str:
        cleaned = _clean_tag(match.group(0))
        if cleaned and cleaned not in seen:
            tags.append(cleaned)
            seen.add(cleaned)
        return ""

    stem = _DOT_TAG_RE.sub(_collect, stem)

    for tag in extra_tags or []:
        cleaned = _clean_tag(tag)
        if not cleaned or cleaned in seen:
            continue
        tags.append(cleaned)
        seen.add(cleaned)

    stem = _DOTS_RE.sub(".", stem)
    stem = _DOUBLE_UNDERSCORE_RE.sub("__", stem)
    stem = stem.strip(".")

    if preset_label:
        stem = f"{stem}__{preset_label}"
    if tags:
        stem = f"{stem}{''.join(f'.__{tag}' for tag in tags)}"
    if is_final:
        stem = f"{stem}_portrait_FINAL"

    stem = normalize_tags_in_stem(stem)
    return f"{stem}{suffix}"

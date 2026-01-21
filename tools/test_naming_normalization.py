"""Self-test for naming normalization helpers."""

from __future__ import annotations

import os
import sys

# Ensure repo root is importable even when running `python tools/...`
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from tools.path_naming import build_output_name, normalize_tags_in_stem


def main() -> None:
    stem = "001__2025-01-01__SHOT__t000-001.__CINEMATIC.__CINEMATIC__SEGMENT_SMOOTH"
    normalized = normalize_tags_in_stem(stem)
    assert normalized.count(".__CINEMATIC") == 1, normalized

    name_a = build_output_name(
        input_path="/clips/001__SHOT.mp4",
        preset="SEGMENT_SMOOTH",
        portrait="1080x1920",
        follow=None,
        is_final=True,
        extra_tags=[".__CINEMATIC", ".__CINEMATIC"],
    )
    name_b = build_output_name(
        input_path="/clips/001__SHOT.mp4",
        preset="SEGMENT_SMOOTH",
        portrait="1080x1920",
        follow=None,
        is_final=True,
        extra_tags=[".__CINEMATIC"],
    )
    assert name_a == name_b, (name_a, name_b)
    assert name_a.count("__SEGMENT_SMOOTH") == 1, name_a
    assert name_a.count("_portrait_FINAL") == 1, name_a

    print("normalize_tags_in_stem:", normalized)
    print("build_output_name:", name_a)


if __name__ == "__main__":
    main()

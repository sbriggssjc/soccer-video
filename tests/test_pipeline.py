from __future__ import annotations

import json
from pathlib import Path

import shutil

import pytest

if shutil.which('ffmpeg') is None:
    pytest.skip('ffmpeg required for pipeline test', allow_module_level=True)

try:
    import cv2  # type: ignore
    import numpy  # type: ignore # noqa: F401
except Exception:
    pytest.skip('OpenCV and numpy are required for the pipeline test', allow_module_level=True)

from soccer_highlights.cli import main
from examples.generate_sample import main as generate_sample


@pytest.mark.slow
def test_end_to_end(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    generate_sample(video_path)
    output_dir = tmp_path / "out"
    config_path = tmp_path / "config.yaml"
    config = {
        "paths": {"video": str(video_path), "output_dir": str(output_dir)},
        "detect": {"pre": 2.0, "post": 3.0, "min_gap": 1.0, "max_count": 10, "audio_weight": 0.0, "threshold_std": 0.3, "hysteresis": 0.25, "sustain": 0.5},
        "shrink": {"mode": "smart", "pre": 2.0, "post": 3.0, "aspect": "horizontal", "zoom": 1.0, "bias_blue": False},
        "clips": {"min_duration": 2.0, "workers": 1},
        "rank": {"k": 3, "max_len": 12.0, "min_tail": 4.0},
        "reels": {"profile": "broadcast", "topk_title": "Sample Top Plays"},
    }
    config_path.write_text(json.dumps(config))

    main(["--config", str(config_path), "detect"])
    highlights_csv = output_dir / "highlights.csv"
    assert highlights_csv.exists()

    main(["--config", str(config_path), "shrink", "--csv", str(highlights_csv), "--out", str(output_dir / "highlights_smart.csv")])
    refined_csv = output_dir / "highlights_smart.csv"
    assert refined_csv.exists()

    main(["--config", str(config_path), "clips", "--csv", str(refined_csv), "--outdir", str(output_dir / "clips"), "--workers", "1"])
    clip_dir = output_dir / "clips"
    clips = list(clip_dir.glob("clip_*.mp4"))
    assert clips, "expected at least one clip"

    main([
        "--config",
        str(config_path),
        "topk",
        "--candirs",
        str(clip_dir),
        "--csv",
        str(output_dir / "smart_topk.csv"),
        "--list",
        str(output_dir / "smart_topk_concat.txt"),
        "--k",
        "2",
    ])
    concat_list = output_dir / "smart_topk_concat.txt"
    assert concat_list.exists()

    reel_out = output_dir / "reels" / "top10.mp4"
    main([
        "--config",
        str(config_path),
        "reel",
        "--list",
        str(concat_list),
        "--out",
        str(reel_out),
        "--title",
        "Test Reel",
    ])
    assert reel_out.exists() and reel_out.stat().st_size > 0

    report_json = output_dir / "report.json"
    assert report_json.exists()

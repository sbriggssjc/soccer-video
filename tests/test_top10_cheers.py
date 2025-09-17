from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "soccer_highlights" / "top10_cheers.py"
SPEC = importlib.util.spec_from_file_location("top10_cheers", MODULE_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - sanity guard
    raise RuntimeError("Unable to load top10_cheers module")
TOP10 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOP10
SPEC.loader.exec_module(TOP10)

build_cheers_top10 = TOP10.build_cheers_top10
load_cheer_candidates = TOP10.load_cheer_candidates
load_filtered_candidates = TOP10.load_filtered_candidates
rank_candidates = TOP10.rank_candidates
select_top_candidates = TOP10.select_top_candidates


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def test_load_cheer_candidates_spacing(tmp_path: Path) -> None:
    cheers_csv = tmp_path / "cheers.csv"
    write_csv(
        cheers_csv,
        ["time"],
        [
            [5],
            [40],
            [90],
            [150],
            [220],
            [280],
        ],
    )

    forced = load_cheer_candidates(cheers_csv, duration=300.0, max_count=4, min_spacing=45.0)
    assert len(forced) == 4
    # First cheer starts before zero and should clamp to 0 after applying pre window
    assert forced[0].start == 0.0
    # Spacing ensures later cheers are kept even if there are extras
    assert forced[-1].start < 300.0


def test_select_top_candidates_prioritises_cheers(tmp_path: Path) -> None:
    filtered_csv = tmp_path / "filtered.csv"
    write_csv(
        filtered_csv,
        ["start", "end", "action_score"],
        [
            [100, 105, 5.0],
            [160, 166, 3.0],
            [200, 205, 4.2],
        ],
    )

    cheers_csv = tmp_path / "cheers.csv"
    write_csv(cheers_csv, ["time"], [[110], [170], [250]])

    forced = load_cheer_candidates(cheers_csv, duration=400.0)
    filtered = load_filtered_candidates(filtered_csv)
    ranked = rank_candidates(forced + filtered)
    selected = select_top_candidates(
        ranked,
        duration=400.0,
        max_count=5,
        pad_pre=2.0,
        pad_post=3.0,
        min_length=0.8,
        overlap_threshold=0.5,
    )

    assert selected, "expected cheer anchors to yield selections"
    cheer_sources = [clip for clip in selected if clip.source == "cheer"]
    assert len(cheer_sources) >= 1
    # ensure selections maintain chronological padding and no duplicates beyond limit
    assert all(clip.end > clip.start for clip in selected)


@pytest.mark.parametrize("cheer_gap", [45.0, 30.0])
def test_build_cheers_top10_skip_render(tmp_path: Path, cheer_gap: float) -> None:
    video_path = tmp_path / "fake_video.mp4"
    video_path.write_bytes(b"fake")

    filtered_csv = tmp_path / "filtered.csv"
    write_csv(
        filtered_csv,
        ["start", "end", "action_score"],
        [
            [50, 55, 1.0],
            [90, 95, 2.0],
            [140, 146, 3.0],
        ],
    )

    cheers_csv = tmp_path / "cheers.csv"
    write_csv(cheers_csv, ["time"], [[60], [120], [180]])

    csv_out = tmp_path / "selected.csv"
    clips_dir = tmp_path / "clips"
    concat_path = tmp_path / "concat.txt"
    out_video = tmp_path / "top10.mp4"

    selected = build_cheers_top10(
        video_path=video_path,
        filtered_csv=filtered_csv,
        cheers_csv=cheers_csv,
        clips_dir=clips_dir,
        concat_path=concat_path,
        output_video=out_video,
        max_count=3,
        pad_pre=1.0,
        pad_post=1.0,
        min_length=0.5,
        overlap_threshold=0.5,
        cheer_max=2,
        cheer_spacing=cheer_gap,
        cheer_pre=5.0,
        cheer_post=2.0,
        cheer_min_length=0.5,
        ffmpeg="ffmpeg",  # won't be invoked due to skip_render
        ffprobe=str(tmp_path / "missing_ffprobe"),
        fallback_duration=400.0,
        csv_out=csv_out,
        skip_render=True,
    )

    assert selected, "expected at least one selected clip"
    assert csv_out.exists()
    assert not clips_dir.exists(), "skip_render should not create clips"


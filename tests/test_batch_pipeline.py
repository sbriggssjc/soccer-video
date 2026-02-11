"""Tests for the unified batch pipeline orchestrator."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.batch_pipeline import (
    _clips_from_catalog,
    _load_duplicate_set,
    _output_path_for_clip,
    main,
    parse_args,
    run_report,
)


# --- Helpers ------------------------------------------------------------------

_clip_counter = 0


def _make_clip(path: Path, width: int = 640, height: int = 360,
               fps: float = 30.0, duration: float = 1.0) -> Path:
    global _clip_counter
    _clip_counter += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        pytest.skip("OpenCV cannot open MP4 writer")
    n_frames = int(fps * duration)
    for i in range(n_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (34, 139, 34)
        cx = int(50 + (width - 100) * (i / max(n_frames - 1, 1)))
        cy = height // 2 + _clip_counter * 10
        cv2.circle(frame, (cx, cy), 8, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def pipeline_env(tmp_path, monkeypatch):
    """Set up isolated directories for pipeline testing."""
    import tools.catalog as cat

    atomic_dir = tmp_path / "out" / "atomic_clips"
    catalog_dir = tmp_path / "out" / "catalog"
    sidecar_dir = catalog_dir / "sidecar"
    portrait_dir = tmp_path / "out" / "portrait_reels" / "clean"
    upscale_dir = tmp_path / "out" / "upscaled"

    monkeypatch.setattr(cat, "ROOT", tmp_path)
    monkeypatch.setattr(cat, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(cat, "ATOMIC_DIR", atomic_dir)
    monkeypatch.setattr(cat, "GAMES_DIR", tmp_path / "out" / "games")
    monkeypatch.setattr(cat, "CATALOG_DIR", catalog_dir)
    monkeypatch.setattr(cat, "SIDE_CAR_DIR", sidecar_dir)
    monkeypatch.setattr(cat, "ATOMIC_INDEX_PATH", catalog_dir / "atomic_index.csv")
    monkeypatch.setattr(cat, "PIPELINE_STATUS_PATH", catalog_dir / "pipeline_status.csv")
    monkeypatch.setattr(cat, "DUPLICATES_PATH", catalog_dir / "duplicates.csv")
    monkeypatch.setattr(cat, "MASTERS_INDEX_PATH", catalog_dir / "masters_index.csv")
    monkeypatch.setattr(cat, "TRASH_ROOT", tmp_path / "out" / "_trash" / "atomic_dupes")
    monkeypatch.setattr(cat, "CLEANUP_LOG_PATH", catalog_dir / "cleanup_log.txt")

    import tools.batch_pipeline as bp
    monkeypatch.setattr(bp, "REPO_ROOT", tmp_path)

    return {
        "root": tmp_path,
        "atomic_dir": atomic_dir,
        "catalog_dir": catalog_dir,
        "portrait_dir": portrait_dir,
        "upscale_dir": upscale_dir,
    }


# --- parse_args ---------------------------------------------------------------

class TestParseArgs:
    def test_defaults(self):
        args = parse_args([])
        assert args.preset == "cinematic"
        assert args.portrait == "1080x1920"
        assert args.limit == 0
        assert not args.upscale
        assert not args.dry_run
        assert not args.report

    def test_preset_override(self):
        args = parse_args(["--preset", "wide_follow"])
        assert args.preset == "wide_follow"

    def test_dry_run(self):
        args = parse_args(["--dry-run"])
        assert args.dry_run

    def test_report_flag(self):
        args = parse_args(["--report"])
        assert args.report

    def test_upscale_options(self):
        args = parse_args(["--upscale", "--upscale-scale", "4", "--upscale-method", "realesrgan"])
        assert args.upscale
        assert args.upscale_scale == 4
        assert args.upscale_method == "realesrgan"

    def test_limit(self):
        args = parse_args(["--limit", "10"])
        assert args.limit == 10

    def test_clip_args(self):
        args = parse_args(["--clip", "a.mp4", "--clip", "b.mp4"])
        assert args.clip == ["a.mp4", "b.mp4"]


# --- _output_path_for_clip ----------------------------------------------------

class TestOutputPath:
    def test_deterministic(self):
        out_dir = Path("/tmp/out")
        p1 = _output_path_for_clip("001__GOAL__t10-t20.mp4", "CINEMATIC", "1080x1920", out_dir)
        p2 = _output_path_for_clip("001__GOAL__t10-t20.mp4", "CINEMATIC", "1080x1920", out_dir)
        assert p1 == p2
        assert p1.parent == out_dir
        assert p1.suffix == ".mp4"

    def test_preset_in_name(self):
        out_dir = Path("/tmp/out")
        p = _output_path_for_clip("001__GOAL__t10-t20.mp4", "cinematic", "1080x1920", out_dir)
        assert "CINEMATIC" in p.stem


# --- _load_duplicate_set ------------------------------------------------------

class TestLoadDuplicateSet:
    def test_no_duplicates_csv(self, pipeline_env):
        dupes = _load_duplicate_set()
        assert dupes == set()

    def test_with_duplicates(self, pipeline_env):
        import shutil
        from tools.catalog import rebuild_atomic_index

        game_dir = pipeline_env["atomic_dir"] / "2025-01-01__Test_Game"
        clip1 = _make_clip(game_dir / "001__GOAL__t10-t20.mp4")
        dup_dir = pipeline_env["atomic_dir"] / "2025-01-01__Test_Game_copy"
        dup_dir.mkdir(parents=True)
        shutil.copy2(clip1, dup_dir / "001__GOAL__t10-t20.mp4")

        rebuild_atomic_index()

        dupes = _load_duplicate_set()
        assert len(dupes) >= 1


# --- _clips_from_catalog -----------------------------------------------------

class TestClipsFromCatalog:
    def test_empty_catalog(self, pipeline_env):
        clips = _clips_from_catalog("out/atomic_clips", "*.mp4", None)
        assert clips == []

    def test_from_directory_scan(self, pipeline_env):
        game_dir = pipeline_env["atomic_dir"] / "2025-01-01__Test_Game"
        _make_clip(game_dir / "001__GOAL__t10-t20.mp4")
        _make_clip(game_dir / "002__SHOT__t30-t40.mp4")

        clips = _clips_from_catalog("out/atomic_clips", "*.mp4", None)
        assert len(clips) == 2

    def test_game_filter(self, pipeline_env):
        g1 = pipeline_env["atomic_dir"] / "2025-01-01__Game_A"
        g2 = pipeline_env["atomic_dir"] / "2025-02-01__Game_B"
        _make_clip(g1 / "001__GOAL__t10-t20.mp4")
        _make_clip(g2 / "002__SHOT__t30-t40.mp4")

        clips = _clips_from_catalog("out/atomic_clips", "*.mp4", "Game_A")
        assert len(clips) == 1


# --- main (dry run) ----------------------------------------------------------

class TestMainDryRun:
    def test_dry_run_no_clips(self, pipeline_env):
        ret = main(["--dry-run"])
        assert ret == 0

    def test_dry_run_with_clips(self, pipeline_env):
        game_dir = pipeline_env["atomic_dir"] / "2025-01-01__Test_Game"
        _make_clip(game_dir / "001__GOAL__t10-t20.mp4")
        _make_clip(game_dir / "002__SHOT__t30-t40.mp4")

        ret = main(["--dry-run", "--src-dir", "out/atomic_clips"])
        assert ret == 0

    def test_dry_run_with_limit(self, pipeline_env):
        game_dir = pipeline_env["atomic_dir"] / "2025-01-01__Test_Game"
        _make_clip(game_dir / "001__GOAL__t10-t20.mp4")
        _make_clip(game_dir / "002__SHOT__t30-t40.mp4")
        _make_clip(game_dir / "003__SAVE__t50-t60.mp4")

        ret = main(["--dry-run", "--limit", "2"])
        assert ret == 0


# --- Report -------------------------------------------------------------------

class TestReport:
    def test_report_empty(self, pipeline_env):
        ret = main(["--report"])
        assert ret == 0

    def test_report_with_data(self, pipeline_env):
        from tools.catalog import rebuild_atomic_index
        game_dir = pipeline_env["atomic_dir"] / "2025-01-01__Test_Game"
        _make_clip(game_dir / "001__GOAL__t10-t20.mp4")
        rebuild_atomic_index()

        ret = main(["--report"])
        assert ret == 0


# --- Rebuild catalog ----------------------------------------------------------

class TestRebuildCatalog:
    def test_rebuild_flag(self, pipeline_env):
        game_dir = pipeline_env["atomic_dir"] / "2025-01-01__Test_Game"
        _make_clip(game_dir / "001__GOAL__t10-t20.mp4")

        ret = main(["--rebuild-catalog", "--dry-run"])
        assert ret == 0
        # atomic_index.csv should now exist
        assert (pipeline_env["catalog_dir"] / "atomic_index.csv").exists()

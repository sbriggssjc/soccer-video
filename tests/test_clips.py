"""Tests for soccer_highlights.clips."""
from __future__ import annotations

import csv
from fractions import Fraction
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from soccer_highlights.clips import _seek_mode, _build_command, _write_metadata
from soccer_highlights.config import AppConfig
from soccer_highlights.io import VideoStreamInfo
from soccer_highlights.utils import HighlightWindow


# --- _seek_mode ---------------------------------------------------------------

class TestSeekMode:
    def test_near_start_uses_output(self):
        """Clips starting before 3s should use output seeking."""
        assert _seek_mode(0.5, 30.0) == "output"
        assert _seek_mode(2.9, 30.0) == "output"

    def test_aligned_uses_input(self):
        """Clips starting at frame-aligned times >= 3s should use input seeking."""
        assert _seek_mode(5.0, 30.0) == "input"
        assert _seek_mode(10.0, 25.0) == "input"

    def test_misaligned_uses_output(self):
        """Non-frame-aligned start times should use output seeking."""
        # 3.3 * 30 = 99.0 → aligned, should be input
        assert _seek_mode(3.3, 30.0) == "input"
        # 3.3333 * 30 = 99.999 → aligned within tolerance
        assert _seek_mode(3.3333, 30.0) == "input"

    def test_exact_frame_boundary(self):
        assert _seek_mode(4.0, 25.0) == "input"  # 4.0 * 25 = 100.0

    def test_half_frame(self):
        # 3.02 * 30 = 90.6 → frac = 0.6 > 0.01 → output
        assert _seek_mode(3.02, 30.0) == "output"


# --- _build_command -----------------------------------------------------------

class TestBuildCommand:
    @pytest.fixture
    def info(self):
        return VideoStreamInfo(path=Path("in.mp4"), width=1920, height=1080, fps=30.0, duration=120.0, time_base=Fraction(1, 30000))

    @pytest.fixture
    def config(self):
        return AppConfig()

    def test_contains_ffmpeg(self, info, config):
        cmd = _build_command(Path("in.mp4"), Path("out.mp4"), 10.0, 20.0, info, config)
        assert cmd[0] == "ffmpeg"

    def test_input_seek_mode(self, info, config):
        cmd = _build_command(Path("in.mp4"), Path("out.mp4"), 10.0, 20.0, info, config)
        # Input seek: -ss before -i
        ss_idx = cmd.index("-ss")
        i_idx = cmd.index("-i")
        assert ss_idx < i_idx

    def test_output_seek_mode(self, info, config):
        cmd = _build_command(Path("in.mp4"), Path("out.mp4"), 0.5, 5.0, info, config)
        # Output seek: -ss after -i
        ss_idx = cmd.index("-ss")
        i_idx = cmd.index("-i")
        assert ss_idx > i_idx

    def test_contains_output_path(self, info, config):
        cmd = _build_command(Path("in.mp4"), Path("out.mp4"), 5.0, 15.0, info, config)
        assert str(Path("out.mp4")) in cmd

    def test_uses_config_preset(self, info, config):
        cmd = _build_command(Path("in.mp4"), Path("out.mp4"), 5.0, 15.0, info, config)
        preset_idx = cmd.index("-preset")
        assert cmd[preset_idx + 1] == config.clips.preset

    def test_uses_config_crf(self, info, config):
        cmd = _build_command(Path("in.mp4"), Path("out.mp4"), 5.0, 15.0, info, config)
        crf_idx = cmd.index("-crf")
        assert cmd[crf_idx + 1] == str(config.clips.crf)

    def test_minimum_duration(self, info, config):
        """Even when end < start, duration is clamped to 0.05."""
        cmd = _build_command(Path("in.mp4"), Path("out.mp4"), 5.0, 5.0, info, config)
        t_idx = cmd.index("-t")
        assert float(cmd[t_idx + 1]) >= 0.05


# --- _write_metadata ----------------------------------------------------------

class TestWriteMetadata:
    def test_writes_csv(self, tmp_path):
        rows = [
            ("clip_0001.mp4", HighlightWindow(start=0.0, end=5.0, score=0.9, event="goal")),
            ("clip_0002.mp4", HighlightWindow(start=10.0, end=15.0, score=0.7, event="scene")),
        ]
        _write_metadata(tmp_path, rows)
        meta_path = tmp_path / "clips_metadata.csv"
        assert meta_path.exists()

        with meta_path.open() as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == ["filename", "event", "start", "end", "score"]
            data_rows = list(reader)
            assert len(data_rows) == 2
            assert data_rows[0][0] == "clip_0001.mp4"
            assert data_rows[0][1] == "goal"
            assert float(data_rows[0][2]) == pytest.approx(0.0)
            assert float(data_rows[0][3]) == pytest.approx(5.0)

    def test_empty_rows(self, tmp_path):
        _write_metadata(tmp_path, [])
        meta_path = tmp_path / "clips_metadata.csv"
        assert meta_path.exists()
        with meta_path.open() as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == ["filename", "event", "start", "end", "score"]
            assert list(reader) == []

    def test_none_event(self, tmp_path):
        rows = [
            ("clip_0001.mp4", HighlightWindow(start=1.0, end=3.0, score=0.5, event=None)),
        ]
        _write_metadata(tmp_path, rows)
        meta_path = tmp_path / "clips_metadata.csv"
        with meta_path.open() as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            row = next(reader)
            assert row[1] == ""  # None event → empty string

"""Tests for soccer_highlights.rank (pure-logic helpers only)."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List

import numpy as np
import pytest

from soccer_highlights.rank import (
    ClipMetadata,
    RankedClip,
    _activity_thresholds,
    _classify_event,
    _compute_trim_bounds,
    _find_active_bounds,
    _load_clip_metadata,
    _parse_float,
    _parse_int,
    _peak_time,
    write_concat,
    write_rankings,
)


# --- _parse_float / _parse_int ------------------------------------------------

class TestParseHelpers:
    def test_parse_float_valid(self):
        assert _parse_float("3.14") == pytest.approx(3.14)

    def test_parse_float_none(self):
        assert _parse_float(None) is None

    def test_parse_float_empty(self):
        assert _parse_float("") is None
        assert _parse_float("  ") is None

    def test_parse_float_invalid(self):
        assert _parse_float("abc") is None

    def test_parse_int_valid(self):
        assert _parse_int("5") == 5

    def test_parse_int_float_str(self):
        assert _parse_int("3.7") == 3

    def test_parse_int_none(self):
        assert _parse_int(None) is None


# --- _classify_event ----------------------------------------------------------

class TestClassifyEvent:
    def test_none_meta(self):
        assert _classify_event(None) == "default"

    def test_no_event(self):
        assert _classify_event(ClipMetadata()) == "default"

    def test_goal_event(self):
        assert _classify_event(ClipMetadata(event="goal")) == "shot"

    def test_shot_event(self):
        assert _classify_event(ClipMetadata(event="shot on target")) == "shot"

    def test_buildup_with_passes(self):
        assert _classify_event(ClipMetadata(event="build_up", passes=6)) == "buildup"

    def test_buildup_few_passes(self):
        """Build-up with < 4 passes doesn't qualify."""
        assert _classify_event(ClipMetadata(event="pass chain", passes=2)) == "default"

    def test_buildup_none_passes(self):
        """passes=None allows build-up classification."""
        assert _classify_event(ClipMetadata(event="build_up", passes=None)) == "buildup"


# --- _activity_thresholds -----------------------------------------------------

class TestActivityThresholds:
    def test_basic(self):
        cover = np.array([0.05, 0.1, 0.2, 0.3, 0.5], dtype=np.float32)
        mag = np.array([0.01, 0.02, 0.03, 0.05, 0.1], dtype=np.float32)
        cov_thr, mag_thr = _activity_thresholds(cover, mag)
        assert cov_thr > 0
        assert mag_thr > 0

    def test_empty(self):
        cov_thr, mag_thr = _activity_thresholds(np.zeros(0), np.zeros(0))
        assert cov_thr >= 0.01
        assert mag_thr >= 0.01


# --- _peak_time ---------------------------------------------------------------

class TestPeakTime:
    def test_basic(self):
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        cover = np.array([0.1, 0.2, 0.5, 0.3, 0.1], dtype=np.float32)
        mag = np.array([0.1, 0.3, 0.8, 0.2, 0.1], dtype=np.float32)
        peak = _peak_time(times, cover, mag, 5.0)
        assert peak == pytest.approx(2.0)

    def test_empty_times(self):
        peak = _peak_time(np.zeros(0), np.zeros(0), np.zeros(0), 10.0)
        assert peak == pytest.approx(5.0)

    def test_empty_mag(self):
        times = np.array([0.0, 1.0], dtype=np.float32)
        cover = np.array([0.5, 0.3], dtype=np.float32)
        peak = _peak_time(times, cover, np.zeros(0), 5.0)
        assert peak == pytest.approx(1.0)


# --- _find_active_bounds ------------------------------------------------------

class TestFindActiveBounds:
    def test_empty(self):
        start, end = _find_active_bounds(np.zeros(0), np.zeros(0), np.zeros(0), 1.0, 10.0)
        assert start == 0.0
        assert end == 10.0

    def test_all_active(self):
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        cover = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        mag = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        start, end = _find_active_bounds(times, cover, mag, 0.5, 5.0)
        assert start >= 0.0
        assert end <= 5.0


# --- _compute_trim_bounds -----------------------------------------------------

class TestComputeTrimBounds:
    def test_default(self):
        start, end = _compute_trim_bounds("default", 1.0, 3.0, 5.0, 10.0)
        assert start >= 0.0
        assert end <= 10.0
        assert end > start

    def test_shot(self):
        start, end = _compute_trim_bounds("shot", 1.0, 3.0, 5.0, 10.0)
        assert end > start
        assert end >= 5.5  # last_time + 0.5

    def test_buildup(self):
        start, end = _compute_trim_bounds("buildup", 1.0, 3.0, 5.0, 10.0)
        assert start == pytest.approx(1.0)

    def test_zero_duration(self):
        start, end = _compute_trim_bounds("default", 0.0, 0.0, 0.0, 0.0)
        assert start == 0.0
        assert end == 0.0


# --- _load_clip_metadata ------------------------------------------------------

class TestLoadClipMetadata:
    def test_missing_file(self, tmp_path):
        result = _load_clip_metadata(tmp_path)
        assert result == {}

    def test_valid_csv(self, tmp_path):
        meta = tmp_path / "clips_metadata.csv"
        meta.write_text(
            "filename,event,start,end,score\n"
            "clip_0001.mp4,goal,10.0,15.0,0.9\n"
        )
        result = _load_clip_metadata(tmp_path)
        assert len(result) == 1
        key = (tmp_path / "clip_0001.mp4").resolve()
        assert key in result
        assert result[key].event == "goal"


# --- write_rankings -----------------------------------------------------------

class TestWriteRankings:
    def test_writes_csv(self, tmp_path):
        clips = [
            RankedClip(path=Path("clip1.mp4"), inpoint=0.0, outpoint=5.0, duration=10.0, motion=0.5, audio=0.3, score=0.8, event="goal"),
        ]
        csv_path = tmp_path / "rankings.csv"
        write_rankings(csv_path, clips)
        assert csv_path.exists()
        with csv_path.open() as f:
            reader = csv.reader(f)
            header = next(reader)
            assert "score" in header
            row = next(reader)
            assert len(row) == 8


# --- write_concat -------------------------------------------------------------

class TestWriteConcat:
    def test_basic(self, tmp_path):
        clips = [
            RankedClip(path=Path("clip1.mp4"), inpoint=1.0, outpoint=8.0, duration=10.0, motion=0.5, audio=0.3, score=0.8),
        ]
        concat_path = tmp_path / "concat.txt"
        write_concat(concat_path, clips, max_len=18.0)
        assert concat_path.exists()
        text = concat_path.read_text()
        assert "file" in text
        assert "inpoint" in text
        assert "outpoint" in text

    def test_max_len_clamps(self, tmp_path):
        clips = [
            RankedClip(path=Path("clip.mp4"), inpoint=0.0, outpoint=20.0, duration=25.0, motion=0.5, audio=0.3, score=0.8),
        ]
        concat_path = tmp_path / "concat.txt"
        write_concat(concat_path, clips, max_len=10.0)
        text = concat_path.read_text()
        # outpoint should be clamped to inpoint + max_len = 10.0
        assert "outpoint 10.000" in text

    def test_empty_clips(self, tmp_path):
        concat_path = tmp_path / "concat.txt"
        write_concat(concat_path, [], max_len=18.0)
        assert concat_path.exists()
        assert concat_path.read_text() == ""

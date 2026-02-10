"""Tests for soccer_highlights.utils."""
from __future__ import annotations

import json
import pytest
from pathlib import Path

from soccer_highlights.utils import (
    HighlightWindow,
    PipelineReport,
    clamp,
    load_report,
    merge_overlaps,
    read_highlights,
    safe_remove,
    summary_stats,
    trim_to_duration,
    write_highlights,
    write_report_md,
)


# --- HighlightWindow -----------------------------------------------------

class TestHighlightWindow:
    def test_duration_normal(self):
        hw = HighlightWindow(start=1.0, end=5.0, score=0.9)
        assert hw.duration() == pytest.approx(4.0)

    def test_duration_zero(self):
        hw = HighlightWindow(start=3.0, end=3.0, score=0.5)
        assert hw.duration() == 0.0

    def test_duration_negative_clamped(self):
        hw = HighlightWindow(start=5.0, end=3.0, score=0.5)
        assert hw.duration() == 0.0

    def test_default_event(self):
        hw = HighlightWindow(start=0, end=1, score=0)
        assert hw.event == "scene"


# --- clamp ----------------------------------------------------------------

class TestClamp:
    def test_in_range(self):
        assert clamp(5.0, 0.0, 10.0) == 5.0

    def test_below(self):
        assert clamp(-1.0, 0.0, 10.0) == 0.0

    def test_above(self):
        assert clamp(15.0, 0.0, 10.0) == 10.0

    def test_exact_bounds(self):
        assert clamp(0.0, 0.0, 10.0) == 0.0
        assert clamp(10.0, 0.0, 10.0) == 10.0


# --- merge_overlaps -------------------------------------------------------

class TestMergeOverlaps:
    def test_empty(self):
        assert merge_overlaps([], 1.0) == []

    def test_single_window(self):
        windows = [HighlightWindow(start=0, end=5, score=1.0)]
        result = merge_overlaps(windows, 1.0)
        assert len(result) == 1

    def test_non_overlapping(self):
        windows = [
            HighlightWindow(start=0, end=3, score=0.5),
            HighlightWindow(start=10, end=15, score=0.8),
        ]
        result = merge_overlaps(windows, 1.0)
        assert len(result) == 2

    def test_overlapping_merged(self):
        windows = [
            HighlightWindow(start=0, end=5, score=0.5),
            HighlightWindow(start=4, end=10, score=0.9),
        ]
        result = merge_overlaps(windows, 2.0)
        assert len(result) == 1
        assert result[0].start == 0
        assert result[0].end == 10
        assert result[0].score == 0.9

    def test_goal_event_priority(self):
        windows = [
            HighlightWindow(start=0, end=5, score=0.5, event="scene"),
            HighlightWindow(start=4, end=10, score=0.4, event="goal"),
        ]
        result = merge_overlaps(windows, 2.0)
        assert len(result) == 1
        assert result[0].event == "goal"

    def test_unsorted_input(self):
        windows = [
            HighlightWindow(start=10, end=15, score=0.8),
            HighlightWindow(start=0, end=5, score=0.5),
        ]
        result = merge_overlaps(windows, 1.0)
        assert result[0].start == 0
        assert result[1].start == 10


# --- summary_stats --------------------------------------------------------

class TestSummaryStats:
    def test_empty(self):
        stats = summary_stats([])
        assert stats["count"] == 0
        assert stats["mean_duration"] == 0.0

    def test_single(self):
        stats = summary_stats([HighlightWindow(start=0, end=4, score=1)])
        assert stats["count"] == 1
        assert stats["mean_duration"] == 4.0
        assert stats["median_duration"] == 4.0

    def test_multiple(self):
        windows = [
            HighlightWindow(start=0, end=2, score=1),
            HighlightWindow(start=0, end=6, score=1),
            HighlightWindow(start=0, end=10, score=1),
        ]
        stats = summary_stats(windows)
        assert stats["count"] == 3
        assert stats["mean_duration"] == 6.0
        assert stats["median_duration"] == 6.0


# --- trim_to_duration -----------------------------------------------------

class TestTrimToDuration:
    def test_normal(self):
        s, e = trim_to_duration(5.0, 15.0, 3.0, 5.0, 60.0)
        assert s == pytest.approx(7.0)  # center=10, 10-3=7
        assert e == pytest.approx(15.0)  # center=10, 10+5=15

    def test_clamped_start(self):
        s, e = trim_to_duration(0.0, 2.0, 5.0, 5.0, 60.0)
        # center = 1.0, start = max(0, 1-5)=0
        assert s == 0.0

    def test_clamped_end(self):
        s, e = trim_to_duration(55.0, 60.0, 3.0, 5.0, 60.0)
        assert e == 60.0

    def test_degenerate_produces_tiny_window(self):
        s, e = trim_to_duration(5.0, 5.0, 0.0, 0.0, 10.0)
        # center=5, start=5, end=5, end<=start → end = start + 0.1
        assert e > s


# --- CSV round-trip -------------------------------------------------------

class TestCSVRoundTrip:
    def test_read_write(self, tmp_path):
        windows = [
            HighlightWindow(start=1.0, end=5.5, score=0.8, event="goal"),
            HighlightWindow(start=10.0, end=15.0, score=0.6, event="scene"),
        ]
        csv_path = tmp_path / "highlights.csv"
        write_highlights(csv_path, windows)
        assert csv_path.exists()

        loaded = read_highlights(csv_path)
        assert len(loaded) == 2
        assert loaded[0].start == pytest.approx(1.0)
        assert loaded[0].end == pytest.approx(5.5)
        assert loaded[0].event == "goal"
        assert loaded[1].score == pytest.approx(0.6)


# --- PipelineReport ------------------------------------------------------

class TestPipelineReport:
    def test_load_create_update(self, tmp_path):
        report = load_report(tmp_path)
        assert report.data == {}

        report.update("detect", {"count": 5, "threshold": 0.3})
        json_path = tmp_path / "report.json"
        assert json_path.exists()

        data = json.loads(json_path.read_text())
        assert data["detect"]["count"] == 5

    def test_write_report_md(self, tmp_path):
        md_path = tmp_path / "report.md"
        write_report_md(md_path, {"detect": {"count": 5}})
        assert md_path.exists()
        content = md_path.read_text()
        assert "Detect" in content
        assert "Count" in content


# --- safe_remove ----------------------------------------------------------

class TestSafeRemove:
    def test_existing_file(self, tmp_path):
        f = tmp_path / "temp.txt"
        f.write_text("data")
        safe_remove(f)
        assert not f.exists()

    def test_missing_file(self, tmp_path):
        f = tmp_path / "nonexistent.txt"
        safe_remove(f)  # should not raise

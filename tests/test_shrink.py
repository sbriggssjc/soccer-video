"""Tests for soccer_highlights.shrink (pure-logic helpers)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from soccer_highlights.config import AppConfig
from soccer_highlights.shrink import (
    _ball_in_play_gate,
    _find_peak,
    simple_shrink,
)
from soccer_highlights.utils import HighlightWindow


# --- _ball_in_play_gate -------------------------------------------------------

class TestBallInPlayGate:
    def test_empty_motion(self):
        start, end = _ball_in_play_gate(
            np.zeros(0), np.zeros(0), 5.0, 3.0, 8.0, 60.0
        )
        assert start == 3.0
        assert end == 8.0

    def test_empty_times(self):
        start, end = _ball_in_play_gate(
            np.zeros(0), np.array([0.5], dtype=np.float32), 5.0, 3.0, 8.0, 60.0
        )
        assert start == 3.0
        assert end == 8.0

    def test_zero_peak(self):
        """All-zero motion should return defaults."""
        times = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        motion = np.zeros(5, dtype=np.float32)
        start, end = _ball_in_play_gate(times, motion, 3.0, 1.0, 6.0, 10.0)
        assert start == 1.0
        assert end == 6.0

    def test_active_region(self):
        """When motion exists, gate should narrow bounds around peak."""
        times = np.arange(0, 10, 1.0, dtype=np.float32)
        motion = np.array([0.0, 0.0, 0.1, 0.5, 0.9, 0.8, 0.4, 0.1, 0.0, 0.0], dtype=np.float32)
        start, end = _ball_in_play_gate(times, motion, 4.0, 0.0, 10.0, 10.0)
        # Gate should narrow around the active region
        assert start >= 0.0
        assert end <= 10.0
        assert end > start


# --- simple_shrink ------------------------------------------------------------

class TestSimpleShrink:
    @patch("soccer_highlights.shrink.video_stream_info")
    def test_basic(self, mock_info):
        mock_info.return_value = MagicMock(duration=60.0)
        config = AppConfig()
        windows = [
            HighlightWindow(start=10.0, end=20.0, score=0.8, event="scene"),
            HighlightWindow(start=30.0, end=40.0, score=0.6, event="goal"),
        ]
        refined = simple_shrink(config, Path("fake.mp4"), windows)
        assert len(refined) == 2
        for win in refined:
            assert win.start >= 0.0
            assert win.end <= 60.0
            assert win.end > win.start

    @patch("soccer_highlights.shrink.video_stream_info")
    def test_preserves_events(self, mock_info):
        mock_info.return_value = MagicMock(duration=60.0)
        config = AppConfig()
        windows = [
            HighlightWindow(start=10.0, end=20.0, score=0.9, event="goal"),
        ]
        refined = simple_shrink(config, Path("fake.mp4"), windows)
        assert refined[0].event == "goal"

    @patch("soccer_highlights.shrink.video_stream_info")
    def test_empty_windows(self, mock_info):
        mock_info.return_value = MagicMock(duration=60.0)
        config = AppConfig()
        refined = simple_shrink(config, Path("fake.mp4"), [])
        assert refined == []

    @patch("soccer_highlights.shrink.video_stream_info")
    def test_clamps_to_duration(self, mock_info):
        mock_info.return_value = MagicMock(duration=15.0)
        config = AppConfig()
        windows = [
            HighlightWindow(start=12.0, end=20.0, score=0.5, event="scene"),
        ]
        refined = simple_shrink(config, Path("fake.mp4"), windows)
        assert refined[0].end <= 15.0

"""Tests for soccer_highlights.io."""
from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from soccer_highlights.io import (
    AudioStreamInfo,
    FFmpegError,
    VideoStreamInfo,
    _parse_fraction,
    ensure_dir,
    list_from_paths,
    run_command,
    seconds_to_timestamp,
    video_stream_info,
    audio_stream_info,
)


# --- _parse_fraction ------------------------------------------------------

class TestParseFraction:
    def test_integer_fraction(self):
        assert _parse_fraction("30/1") == Fraction(30, 1)

    def test_ntsc_fraction(self):
        assert _parse_fraction("30000/1001") == Fraction(30000, 1001)

    def test_float_string(self):
        result = _parse_fraction("29.97")
        assert float(result) == pytest.approx(29.97, rel=1e-3)

    def test_integer_string(self):
        assert _parse_fraction("24") == Fraction(24)


# --- seconds_to_timestamp ------------------------------------------------

class TestSecondsToTimestamp:
    def test_basic(self):
        assert seconds_to_timestamp(12.345) == "12.345"

    def test_zero(self):
        assert seconds_to_timestamp(0.0) == "0.000"

    def test_large(self):
        assert seconds_to_timestamp(3600.5) == "3600.500"


# --- list_from_paths ------------------------------------------------------

class TestListFromPaths:
    def test_basic(self):
        result = list_from_paths([Path("a.mp4"), Path("b.mp4")])
        assert result == ["a.mp4", "b.mp4"]

    def test_empty(self):
        assert list_from_paths([]) == []


# --- ensure_dir -----------------------------------------------------------

class TestEnsureDir:
    def test_creates_dir(self, tmp_path):
        d = tmp_path / "a" / "b" / "c"
        ensure_dir(d)
        assert d.is_dir()

    def test_existing_dir(self, tmp_path):
        ensure_dir(tmp_path)  # should not raise


# --- run_command ----------------------------------------------------------

class TestRunCommand:
    def test_success(self):
        with patch("soccer_highlights.io.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=["echo", "hi"], returncode=0, stdout="hi\n", stderr=""
            )
            result = run_command(["echo", "hi"])
            assert result.returncode == 0

    def test_failure_raises(self):
        with patch("soccer_highlights.io.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=["bad"], returncode=1, stdout="", stderr="error"
            )
            with pytest.raises(FFmpegError, match="Command failed"):
                run_command(["bad"])

    def test_check_false_no_raise(self):
        with patch("soccer_highlights.io.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=["bad"], returncode=1, stdout="", stderr=""
            )
            result = run_command(["bad"], check=False)
            assert result.returncode == 1


# --- video_stream_info ----------------------------------------------------

_FAKE_FFPROBE = {
    "streams": [
        {
            "codec_type": "video",
            "width": 1920,
            "height": 1080,
            "duration": "60.0",
            "r_frame_rate": "30/1",
            "time_base": "1/30000",
        }
    ],
    "format": {"duration": "60.0"},
}


class TestVideoStreamInfo:
    def test_parses_video(self, tmp_path):
        with patch("soccer_highlights.io.ffprobe_json", return_value=_FAKE_FFPROBE):
            info = video_stream_info(tmp_path / "test.mp4")
            assert info.width == 1920
            assert info.height == 1080
            assert info.fps == pytest.approx(30.0)
            assert info.duration == pytest.approx(60.0)

    def test_no_video_stream_raises(self, tmp_path):
        data = {"streams": [{"codec_type": "audio"}], "format": {}}
        with patch("soccer_highlights.io.ffprobe_json", return_value=data):
            with pytest.raises(FFmpegError, match="No video streams"):
                video_stream_info(tmp_path / "audio.mp3")

    def test_missing_dimensions_default_zero(self, tmp_path):
        data = {
            "streams": [{"codec_type": "video", "r_frame_rate": "30/1"}],
            "format": {"duration": "10.0"},
        }
        with patch("soccer_highlights.io.ffprobe_json", return_value=data):
            info = video_stream_info(tmp_path / "test.mp4")
            assert info.width == 0
            assert info.height == 0

    def test_fraction_avg_frame_rate_fallback(self, tmp_path):
        data = {
            "streams": [
                {
                    "codec_type": "video",
                    "width": 640,
                    "height": 480,
                    "r_frame_rate": "0/0",
                    "avg_frame_rate": "25/1",
                }
            ],
            "format": {"duration": "5.0"},
        }
        with patch("soccer_highlights.io.ffprobe_json", return_value=data):
            info = video_stream_info(tmp_path / "test.mp4")
            assert info.fps == pytest.approx(25.0)


# --- audio_stream_info ----------------------------------------------------

class TestAudioStreamInfo:
    def test_parses_audio(self, tmp_path):
        data = {
            "streams": [
                {
                    "codec_type": "audio",
                    "duration": "30.0",
                    "sample_rate": "44100",
                    "channels": "2",
                }
            ],
            "format": {"duration": "30.0"},
        }
        with patch("soccer_highlights.io.ffprobe_json", return_value=data):
            info = audio_stream_info(tmp_path / "test.mp4")
            assert info.sample_rate == 44100
            assert info.channels == 2
            assert info.duration == pytest.approx(30.0)

    def test_no_audio_stream_raises(self, tmp_path):
        data = {"streams": [{"codec_type": "video"}], "format": {}}
        with patch("soccer_highlights.io.ffprobe_json", return_value=data):
            with pytest.raises(FFmpegError, match="No audio streams"):
                audio_stream_info(tmp_path / "test.mp4")

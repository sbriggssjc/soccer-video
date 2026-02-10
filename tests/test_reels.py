"""Tests for soccer_highlights.reels."""
from __future__ import annotations

from pathlib import Path

import pytest

from soccer_highlights.reels import ReelEntry, _escape_drawtext, _parse_concat


# --- ReelEntry ----------------------------------------------------------------

class TestReelEntry:
    def test_duration_positive(self):
        entry = ReelEntry(path=Path("clip.mp4"), inpoint=2.0, outpoint=7.0)
        assert entry.duration == pytest.approx(5.0)

    def test_duration_zero(self):
        entry = ReelEntry(path=Path("clip.mp4"), inpoint=3.0, outpoint=3.0)
        assert entry.duration == 0.0

    def test_duration_negative_clamped(self):
        """Outpoint before inpoint should clamp to 0."""
        entry = ReelEntry(path=Path("clip.mp4"), inpoint=5.0, outpoint=2.0)
        assert entry.duration == 0.0


# --- _escape_drawtext ---------------------------------------------------------

class TestEscapeDrawtext:
    def test_plain_text(self):
        assert _escape_drawtext("hello") == "hello"

    def test_escapes_colon(self):
        assert _escape_drawtext("Top:10") == "Top\\:10"

    def test_escapes_backslash(self):
        assert _escape_drawtext("a\\b") == "a\\\\b"

    def test_escapes_single_quote(self):
        result = _escape_drawtext("it's")
        assert "\\\\'" in result

    def test_combined_special_chars(self):
        result = _escape_drawtext("a\\b:c'd")
        assert "\\\\" in result
        assert "\\:" in result
        assert "\\\\'" in result

    def test_empty_string(self):
        assert _escape_drawtext("") == ""

    def test_numbered_label(self):
        result = _escape_drawtext("#1")
        assert result == "#1"


# --- _parse_concat ------------------------------------------------------------

class TestParseConcat:
    def test_basic_concat(self, tmp_path):
        concat_file = tmp_path / "list.txt"
        concat_file.write_text(
            "file 'clip1.mp4'\n"
            "inpoint 1.0\n"
            "outpoint 5.0\n"
            "file 'clip2.mp4'\n"
            "inpoint 0.0\n"
            "outpoint 10.0\n"
        )
        entries = _parse_concat(concat_file)
        assert len(entries) == 2
        assert entries[0].path == Path("clip1.mp4")
        assert entries[0].inpoint == pytest.approx(1.0)
        assert entries[0].outpoint == pytest.approx(5.0)
        assert entries[1].path == Path("clip2.mp4")
        assert entries[1].inpoint == pytest.approx(0.0)
        assert entries[1].outpoint == pytest.approx(10.0)

    def test_empty_file(self, tmp_path):
        concat_file = tmp_path / "empty.txt"
        concat_file.write_text("")
        assert _parse_concat(concat_file) == []

    def test_missing_inpoint_defaults_zero(self, tmp_path):
        concat_file = tmp_path / "list.txt"
        concat_file.write_text(
            "file 'clip.mp4'\n"
            "outpoint 8.0\n"
        )
        entries = _parse_concat(concat_file)
        assert len(entries) == 1
        assert entries[0].inpoint == pytest.approx(0.0)
        assert entries[0].outpoint == pytest.approx(8.0)

    def test_path_without_quotes(self, tmp_path):
        concat_file = tmp_path / "list.txt"
        concat_file.write_text(
            "file clip.mp4\n"
            "inpoint 0.0\n"
            "outpoint 3.0\n"
        )
        entries = _parse_concat(concat_file)
        assert len(entries) == 1
        assert entries[0].path == Path("clip.mp4")

    def test_multiple_entries_order(self, tmp_path):
        concat_file = tmp_path / "list.txt"
        lines = []
        for i in range(5):
            lines.append(f"file 'clip{i}.mp4'\n")
            lines.append(f"inpoint {i}.0\n")
            lines.append(f"outpoint {i + 3}.0\n")
        concat_file.write_text("".join(lines))
        entries = _parse_concat(concat_file)
        assert len(entries) == 5
        for i, entry in enumerate(entries):
            assert entry.path == Path(f"clip{i}.mp4")
            assert entry.duration == pytest.approx(3.0)

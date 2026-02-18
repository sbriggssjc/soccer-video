"""Tests for tools/timestamp_parse.py — shared timestamp parsing."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from timestamp_parse import parse_timestamp, parse_timestamp_pair


# ---------------------------------------------------------------------------
# parse_timestamp — single value
# ---------------------------------------------------------------------------

class TestParseTimestamp:
    """Format 1 (MM:SS.f), Format 2 (plain seconds), Format 3 (MM:SS:cs)."""

    # MM:SS and MM:SS.f
    def test_mmss_integer(self):
        assert parse_timestamp("6:27") == 6 * 60 + 27  # 387

    def test_mmss_fractional(self):
        assert parse_timestamp("6:27.5") == pytest.approx(387.5)

    def test_mmss_zero_start(self):
        assert parse_timestamp("0:30") == 30.0

    # Plain seconds
    def test_plain_integer(self):
        assert parse_timestamp("421") == 421.0

    def test_plain_decimal(self):
        assert parse_timestamp("1247.80") == pytest.approx(1247.8)

    def test_plain_with_comma(self):
        assert parse_timestamp("1,247.80") == pytest.approx(1247.8)

    # Three-part defaults to MM:SS:cs when parsed individually
    def test_three_part_mmss_cs(self):
        # "5:15:00" as MM:SS:cs → 5*60 + 15 + 0/100 = 315.0
        assert parse_timestamp("5:15:00") == pytest.approx(315.0)

    def test_three_part_with_centis(self):
        # "2:02:50" as MM:SS:cs → 2*60 + 2 + 50/100 = 122.5
        assert parse_timestamp("2:02:50") == pytest.approx(122.5)

    # Edge cases
    def test_empty_string(self):
        assert parse_timestamp("") is None

    def test_whitespace(self):
        assert parse_timestamp("  ") is None

    def test_quoted(self):
        assert parse_timestamp('"6:27"') == 6 * 60 + 27

    def test_garbage(self):
        assert parse_timestamp("abc") is None


# ---------------------------------------------------------------------------
# parse_timestamp_pair — disambiguation
# ---------------------------------------------------------------------------

class TestParseTimestampPair:
    """Formats 3 vs 4: MM:SS:cs vs H:MM:SS, plus mixed format pairs."""

    def test_mmss_cs_normal_duration(self):
        """Typical MM:SS:cs pair with comfortable duration → stays MM:SS:cs."""
        # "5:15:00" to "5:30:00" → 315s to 330s → 15s duration
        t0, t1 = parse_timestamp_pair("5:15:00", "5:30:00")
        assert t0 == pytest.approx(315.0)
        assert t1 == pytest.approx(330.0)
        assert t1 - t0 == pytest.approx(15.0)

    def test_hmmss_disambiguation(self):
        """Navy-style "0:06:37" to "0:07:10" — too short as MM:SS:cs, falls back to H:MM:SS."""
        # As MM:SS:cs: 6.37s to 7.10s → 0.73s duration (< 2s)
        # As H:MM:SS:  397s to 430s → 33s duration
        t0, t1 = parse_timestamp_pair("0:06:37", "0:07:10")
        assert t0 == pytest.approx(397.0)
        assert t1 == pytest.approx(430.0)
        assert t1 - t0 == pytest.approx(33.0)

    def test_hmmss_longer_game(self):
        """H:MM:SS pair deeper into the game."""
        # "1:15:20" to "1:16:00"
        # As MM:SS:cs: 75.20s to 76.00s → 0.80s (< 2s) → fallback
        # As H:MM:SS:  4520s to 4560s → 40s
        t0, t1 = parse_timestamp_pair("1:15:20", "1:16:00")
        assert t0 == pytest.approx(4520.0)
        assert t1 == pytest.approx(4560.0)

    def test_mmss_format_pair(self):
        """Two-part MM:SS timestamps pass through normally."""
        t0, t1 = parse_timestamp_pair("6:27", "6:50")
        assert t0 == pytest.approx(387.0)
        assert t1 == pytest.approx(410.0)

    def test_plain_seconds_pair(self):
        """Plain seconds pass through."""
        t0, t1 = parse_timestamp_pair("421", "441")
        assert t0 == pytest.approx(421.0)
        assert t1 == pytest.approx(441.0)

    def test_mixed_formats(self):
        """One three-part and one two-part fall back to individual parsing."""
        t0, t1 = parse_timestamp_pair("5:15:00", "6:27")
        assert t0 == pytest.approx(315.0)  # MM:SS:cs default
        assert t1 == pytest.approx(387.0)  # MM:SS

    def test_empty_start(self):
        t0, t1 = parse_timestamp_pair("", "6:27")
        assert t0 is None
        assert t1 is None

    def test_empty_end(self):
        t0, t1 = parse_timestamp_pair("6:27", "")
        assert t0 is None
        assert t1 is None

    def test_none_handling(self):
        t0, t1 = parse_timestamp_pair(None, None)
        assert t0 is None
        assert t1 is None

    def test_borderline_duration(self):
        """Exactly at the 2-second boundary — MM:SS:cs is accepted."""
        # "5:00:00" to "5:02:00" as MM:SS:cs → 300s to 302s → 2s duration (>=2, accepted)
        t0, t1 = parse_timestamp_pair("5:00:00", "5:02:00")
        assert t0 == pytest.approx(300.0)
        assert t1 == pytest.approx(302.0)


# ---------------------------------------------------------------------------
# Real-world timestamp pairs from the game indexes
# ---------------------------------------------------------------------------

class TestRealWorldExamples:
    """Verify parsing of actual timestamps from the 11-game catalog."""

    def test_rvfc_clip_001(self):
        """RVFC uses H:MM:SS format (e.g., "0:02:07" to "0:02:25")."""
        t0, t1 = parse_timestamp_pair("0:02:07", "0:02:25")
        assert t0 == pytest.approx(127.0)
        assert t1 == pytest.approx(145.0)
        assert t1 - t0 == pytest.approx(18.0)

    def test_rvfc_clip_002(self):
        """RVFC clip #002."""
        t0, t1 = parse_timestamp_pair("0:03:12", "0:03:30")
        assert t0 == pytest.approx(192.0)
        assert t1 == pytest.approx(210.0)

    def test_standard_mmss_cs(self):
        """Standard MM:SS:cs format (e.g., BASC games)."""
        # "2:02:00" to "2:20:00" → 122s to 140s → 18s
        t0, t1 = parse_timestamp_pair("2:02:00", "2:20:00")
        assert t0 == pytest.approx(122.0)
        assert t1 == pytest.approx(140.0)

    def test_mmss_fractional_pair(self):
        """MM:SS.f format (e.g., "6:27.5")."""
        t0, t1 = parse_timestamp_pair("6:27.5", "6:50.0")
        assert t0 == pytest.approx(387.5)
        assert t1 == pytest.approx(410.0)

    def test_plain_seconds_from_csv(self):
        """Plain seconds as sometimes seen in clip_index.csv."""
        t0, t1 = parse_timestamp_pair("1247.80", "1267.80")
        assert t0 == pytest.approx(1247.8)
        assert t1 == pytest.approx(1267.8)

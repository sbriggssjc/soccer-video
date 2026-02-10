"""Tests for tools/prepare_masters.py — game segment scanning and concatenation."""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import prepare_masters as pm


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------

class TestParseTimestamp:
    def test_standard_xbotgo(self):
        assert pm._parse_xbotgo_timestamp("20250920_174654000_iOS.MP4") == "20250920_174654"

    def test_lowercase_ext(self):
        assert pm._parse_xbotgo_timestamp("20250920_174654000_iOS.mp4") == "20250920_174654"

    def test_mov_extension(self):
        assert pm._parse_xbotgo_timestamp("20250412_143045000_iOS.MOV") == "20250412_143045"

    def test_duplicate_variant(self):
        assert pm._parse_xbotgo_timestamp("20250412_150018000_iOS 1.MOV") == "20250412_150018"

    def test_duplicate_variant_2(self):
        assert pm._parse_xbotgo_timestamp("20250412_150117000_iOS 2.MOV") == "20250412_150117"

    def test_non_xbotgo_filename(self):
        assert pm._parse_xbotgo_timestamp("MASTER.mp4") is None

    def test_named_file(self):
        assert pm._parse_xbotgo_timestamp("November 22, 2025.mp4") is None

    def test_processed_file(self):
        assert pm._parse_xbotgo_timestamp("smart10_clean_zoom.mp4") is None


class TestIsDuplicateVariant:
    def test_normal(self):
        assert not pm._is_duplicate_variant("20250412_150018000_iOS.MOV")

    def test_variant_1(self):
        assert pm._is_duplicate_variant("20250412_150018000_iOS 1.MOV")

    def test_variant_2(self):
        assert pm._is_duplicate_variant("20250412_150117000_iOS 2.MOV")

    def test_mp4(self):
        assert pm._is_duplicate_variant("20250920_174654000_iOS 1.MP4")


# ---------------------------------------------------------------------------
# Halftime detection
# ---------------------------------------------------------------------------

class TestDetectHalftime:
    def _seg(self, ts: str) -> pm.Segment:
        return pm.Segment(path=Path(f"{ts}_iOS.MP4"), sort_key=ts, timestamp_str="")

    def test_no_halftime_close_segments(self):
        # 17:46 to 17:49 = 3 min gap, below 5-min threshold
        segs = [self._seg("20250920_174654"), self._seg("20250920_174954")]
        assert pm._detect_halftime(segs, 300) is None

    def test_halftime_detected(self):
        # 17:46 to 18:16 = 30 min gap (1800 sec)
        segs = [self._seg("20250920_174654"), self._seg("20250920_181631")]
        result = pm._detect_halftime(segs, 300)
        assert result == 1

    def test_three_segments_gap_in_middle(self):
        segs = [
            self._seg("20251105_011404"),
            self._seg("20251105_014501"),  # gap: ~31 min
            self._seg("20251105_021501"),
        ]
        # Between seg 0 and 1: 01:14 to 01:45 = 31 min
        result = pm._detect_halftime(segs, 300)
        assert result == 1

    def test_single_segment(self):
        segs = [self._seg("20250920_174654")]
        assert pm._detect_halftime(segs, 300) is None

    def test_empty(self):
        assert pm._detect_halftime([], 300) is None


# ---------------------------------------------------------------------------
# Game scanning
# ---------------------------------------------------------------------------

class TestScanGame:
    def test_empty_dir(self, tmp_path):
        game_dir = tmp_path / "2025-01-01__Test_vs_Test"
        game_dir.mkdir()
        info = pm.scan_game(game_dir)
        assert info.game_name == "2025-01-01__Test_vs_Test"
        assert info.segment_count == 0
        assert not info.has_master

    def test_existing_master(self, tmp_path):
        game_dir = tmp_path / "2025-01-01__Test"
        game_dir.mkdir()
        (game_dir / "MASTER.mp4").write_bytes(b"\x00" * 100)
        info = pm.scan_game(game_dir)
        assert info.has_master
        assert info.master_path == game_dir / "MASTER.mp4"

    def test_two_xbotgo_segments(self, tmp_path):
        game_dir = tmp_path / "2025-09-20__TSC_vs_Greenwood"
        game_dir.mkdir()
        (game_dir / "20250920_174654000_iOS.MP4").write_bytes(b"\x00" * 100)
        (game_dir / "20250920_181631000_iOS.MP4").write_bytes(b"\x00" * 100)
        info = pm.scan_game(game_dir)
        assert info.segment_count == 2
        assert not info.has_master
        # Segments should be sorted chronologically
        assert info.segments[0].sort_key == "20250920_174654"
        assert info.segments[1].sort_key == "20250920_181631"

    def test_single_named_file(self, tmp_path):
        game_dir = tmp_path / "2025-11-23__TSC_vs_BASC"
        game_dir.mkdir()
        (game_dir / "November 23, 2025.mp4").write_bytes(b"\x00" * 100)
        info = pm.scan_game(game_dir)
        assert info.single_file
        assert info.segment_count == 1

    def test_duplicates_detected(self, tmp_path):
        game_dir = tmp_path / "2025-04-12__Route_66"
        game_dir.mkdir()
        (game_dir / "20250412_150018000_iOS.MOV").write_bytes(b"\x00" * 100)
        (game_dir / "20250412_150018000_iOS 1.MOV").write_bytes(b"\x00" * 100)
        info = pm.scan_game(game_dir)
        dups = [s for s in info.segments if s.is_duplicate]
        assert len(dups) >= 1

    def test_segments_sorted_chronologically(self, tmp_path):
        game_dir = tmp_path / "game"
        game_dir.mkdir()
        # Create out of order
        (game_dir / "20250517_182003000_iOS.MOV").write_bytes(b"\x00" * 100)
        (game_dir / "20250517_141920000_iOS.MOV").write_bytes(b"\x00" * 100)
        (game_dir / "20250517_150643000_iOS.MOV").write_bytes(b"\x00" * 100)
        info = pm.scan_game(game_dir)
        keys = [s.sort_key for s in info.segments]
        assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# Prepare master (dry-run only — no ffmpeg in test env)
# ---------------------------------------------------------------------------

class TestPrepareMaster:
    def test_already_has_master(self, tmp_path, capsys):
        game_dir = tmp_path / "game"
        game_dir.mkdir()
        (game_dir / "MASTER.mp4").write_bytes(b"\x00" * 100)
        info = pm.scan_game(game_dir)
        result = pm.prepare_master(info)
        assert result is True
        assert "SKIP" in capsys.readouterr().out

    def test_single_file_dry_run(self, tmp_path, capsys):
        game_dir = tmp_path / "game"
        game_dir.mkdir()
        (game_dir / "November 23, 2025.mp4").write_bytes(b"\x00" * 100)
        info = pm.scan_game(game_dir)
        result = pm.prepare_master(info, dry_run=True)
        assert result is True
        assert "DRY-RUN" in capsys.readouterr().out

    def test_multi_segment_dry_run(self, tmp_path, capsys):
        game_dir = tmp_path / "game"
        game_dir.mkdir()
        (game_dir / "20250920_174654000_iOS.MP4").write_bytes(b"\x00" * 100)
        (game_dir / "20250920_181631000_iOS.MP4").write_bytes(b"\x00" * 100)
        info = pm.scan_game(game_dir)
        result = pm.prepare_master(info, dry_run=True)
        assert result is True
        out = capsys.readouterr().out
        assert "DRY-RUN" in out
        assert "2 segments" in out

    def test_no_segments(self, tmp_path, capsys):
        game_dir = tmp_path / "game"
        game_dir.mkdir()
        info = pm.scan_game(game_dir)
        result = pm.prepare_master(info)
        assert result is False


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

class TestCLI:
    def test_scan_flag(self):
        parser = pm.build_parser()
        args = parser.parse_args(["--scan"])
        assert args.scan

    def test_all_flag(self):
        parser = pm.build_parser()
        args = parser.parse_args(["--all"])
        assert args.all

    def test_game_flag(self):
        parser = pm.build_parser()
        args = parser.parse_args(["--game", "2025-09-20__TSC_vs_Greenwood"])
        assert args.game == "2025-09-20__TSC_vs_Greenwood"

    def test_halftime_gap(self):
        parser = pm.build_parser()
        args = parser.parse_args(["--scan", "--halftime-gap", "600"])
        assert args.halftime_gap == 600.0

    def test_dry_run(self):
        parser = pm.build_parser()
        args = parser.parse_args(["--all", "--dry-run"])
        assert args.dry_run

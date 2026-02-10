"""Tests for tools/ingest_clip_index.py — clip_index.txt parsing and ingestion."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import ingest_clip_index as ici


# ---------------------------------------------------------------------------
# Line parsing
# ---------------------------------------------------------------------------

class TestParseClipLine:
    def test_t_seconds_format(self):
        result = ici.parse_clip_line("t155.50-t166.40  GOAL  Left-foot finish")
        assert result is not None
        assert result["t_start_s"] == 155.50
        assert result["t_end_s"] == 166.40
        assert result["event_type"] == "GOAL"
        assert "Left-foot finish" in result["description"]

    def test_t_seconds_no_second_t(self):
        result = ici.parse_clip_line("t320.00-330.50  SHOT")
        assert result is not None
        assert result["t_start_s"] == 320.00
        assert result["t_end_s"] == 330.50
        assert result["event_type"] == "SHOT"

    def test_colon_mm_ss(self):
        result = ici.parse_clip_line("12:30-13:10 SAVE Diving save")
        assert result is not None
        assert result["t_start_s"] == 750.0
        assert result["t_end_s"] == 790.0
        assert result["event_type"] == "SAVE"

    def test_colon_h_mm_ss(self):
        result = ici.parse_clip_line("0:45:20-0:46:00 BUILD_UP")
        assert result is not None
        assert result["t_start_s"] == 2720.0
        assert result["t_end_s"] == 2760.0
        assert result["event_type"] == "BUILD_UP"

    def test_empty_line(self):
        assert ici.parse_clip_line("") is None

    def test_comment_line(self):
        assert ici.parse_clip_line("# this is a comment") is None

    def test_no_event_type(self):
        result = ici.parse_clip_line("t100.00-t110.00 some description")
        assert result is not None
        assert result["event_type"] == "HIGHLIGHT"
        assert "some description" in result["description"]

    def test_dribbling_event(self):
        result = ici.parse_clip_line("t500-t520 DRIBBLING quick feet")
        assert result is not None
        assert result["event_type"] == "DRIBBLING"

    def test_integer_timestamps(self):
        result = ici.parse_clip_line("t100-t200 GOAL")
        assert result is not None
        assert result["t_start_s"] == 100.0
        assert result["t_end_s"] == 200.0

    def test_whitespace_tolerance(self):
        result = ici.parse_clip_line("  t100.00 - t200.00   SHOT  ")
        assert result is not None
        assert result["t_start_s"] == 100.0
        assert result["t_end_s"] == 200.0

    def test_en_dash_separator(self):
        result = ici.parse_clip_line("t100.00\u2013t200.00 GOAL")
        assert result is not None
        assert result["t_start_s"] == 100.0

    def test_garbage_line(self):
        assert ici.parse_clip_line("no timestamps here at all") is None

    def test_underscore_h_mm_ss(self):
        result = ici.parse_clip_line("001__Pressure & Cross__0_06_37-0_07_10")
        assert result is not None
        assert result["t_start_s"] == 397.0
        assert result["t_end_s"] == 430.0
        assert result["event_type"] == "CROSS"
        assert "Pressure & Cross" in result["description"]

    def test_underscore_over_one_hour(self):
        result = ici.parse_clip_line("013__Dribbling & Shot__1_02_55-1_03_10")
        assert result is not None
        assert result["t_start_s"] == 3775.0
        assert result["t_end_s"] == 3790.0
        assert result["event_type"] == "SHOT"

    def test_event_priority_goal_over_cross(self):
        result = ici.parse_clip_line("003__Pressure, Cross and Goal__0_16_09-0_16_33")
        assert result is not None
        assert result["event_type"] == "GOAL"  # GOAL outranks CROSS

    def test_clip_number_stripped_from_description(self):
        result = ici.parse_clip_line("001__Build Up__0_28_28-0_28_42")
        assert result is not None
        assert "001__" not in result["description"]
        assert "Build Up" in result["description"]


class TestColonToSeconds:
    def test_mm_ss(self):
        assert ici._colon_to_seconds("12:30") == 750.0

    def test_h_mm_ss(self):
        assert ici._colon_to_seconds("1:30:00") == 5400.0

    def test_ss_only(self):
        assert ici._colon_to_seconds("45") == 45.0

    def test_fractional(self):
        assert ici._colon_to_seconds("1:30.5") == 90.5


# ---------------------------------------------------------------------------
# File parsing
# ---------------------------------------------------------------------------

class TestParseClipIndex:
    def test_parse_multi_line(self, tmp_path):
        ci = tmp_path / "clip_index.txt"
        ci.write_text(
            "# Game highlights\n"
            "t155.50-t166.40 GOAL Left-foot finish\n"
            "t320.00-t330.50 SHOT\n"
            "\n"
            "12:30-13:10 SAVE Diving save\n"
        )
        clips = ici.parse_clip_index(ci)
        assert len(clips) == 3
        assert clips[0]["event_type"] == "GOAL"
        assert clips[1]["event_type"] == "SHOT"
        assert clips[2]["event_type"] == "SAVE"

    def test_empty_file(self, tmp_path):
        ci = tmp_path / "clip_index.txt"
        ci.write_text("")
        assert ici.parse_clip_index(ci) == []

    def test_all_comments(self, tmp_path):
        ci = tmp_path / "clip_index.txt"
        ci.write_text("# comment 1\n# comment 2\n")
        assert ici.parse_clip_index(ci) == []


# ---------------------------------------------------------------------------
# Ingestion into catalog
# ---------------------------------------------------------------------------

class TestIngestGame:
    def _setup_game(self, tmp_path, game_name="2025-11-05__TSC_vs_Navy"):
        """Set up game folder with clip_index.txt and empty catalog."""
        games_dir = tmp_path / "out" / "games"
        game_dir = games_dir / game_name
        game_dir.mkdir(parents=True)

        catalog_dir = tmp_path / "out" / "catalog"
        catalog_dir.mkdir(parents=True)

        ci = game_dir / "clip_index.txt"
        ci.write_text(
            "t155.50-t166.40 GOAL\n"
            "t320.00-t330.50 SHOT\n"
            "t500.00-t515.00 DRIBBLING nice move\n"
        )

        return game_dir, catalog_dir, ci

    def test_ingest_new_clips(self, tmp_path, monkeypatch):
        game_name = "2025-11-05__TSC_vs_Navy"
        game_dir, catalog_dir, ci = self._setup_game(tmp_path, game_name)

        # Patch module paths
        monkeypatch.setattr(ici, "GAMES_DIR", tmp_path / "out" / "games")
        monkeypatch.setattr(ici, "CATALOG_DIR", catalog_dir)
        monkeypatch.setattr(ici, "ATOMIC_INDEX", catalog_dir / "atomic_index.csv")

        result = ici.ingest_game(game_name)
        assert result["added"] == 3
        assert result["skipped"] == 0

        # Verify CSV was written
        idx = catalog_dir / "atomic_index.csv"
        assert idx.exists()
        with idx.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3
        assert rows[0]["t_start_s"] == "155.5"
        assert rows[0]["t_end_s"] == "166.4"
        assert "GOAL" in rows[0]["clip_name"]
        assert game_name in rows[0]["clip_rel"]

    def test_skip_duplicates(self, tmp_path, monkeypatch):
        game_name = "2025-11-05__TSC_vs_Navy"
        game_dir, catalog_dir, ci = self._setup_game(tmp_path, game_name)

        monkeypatch.setattr(ici, "GAMES_DIR", tmp_path / "out" / "games")
        monkeypatch.setattr(ici, "CATALOG_DIR", catalog_dir)
        monkeypatch.setattr(ici, "ATOMIC_INDEX", catalog_dir / "atomic_index.csv")

        # First ingest
        ici.ingest_game(game_name)
        # Second ingest — should skip all
        result = ici.ingest_game(game_name)
        assert result["added"] == 0
        assert result["skipped"] == 3

    def test_dry_run(self, tmp_path, monkeypatch):
        game_name = "2025-11-05__TSC_vs_Navy"
        game_dir, catalog_dir, ci = self._setup_game(tmp_path, game_name)

        monkeypatch.setattr(ici, "GAMES_DIR", tmp_path / "out" / "games")
        monkeypatch.setattr(ici, "CATALOG_DIR", catalog_dir)
        monkeypatch.setattr(ici, "ATOMIC_INDEX", catalog_dir / "atomic_index.csv")

        result = ici.ingest_game(game_name, dry_run=True)
        assert result["added"] == 3
        # No file written in dry-run
        assert not (catalog_dir / "atomic_index.csv").exists()

    def test_clip_numbering(self, tmp_path, monkeypatch):
        game_name = "2025-11-05__TSC_vs_Navy"
        game_dir, catalog_dir, ci = self._setup_game(tmp_path, game_name)

        monkeypatch.setattr(ici, "GAMES_DIR", tmp_path / "out" / "games")
        monkeypatch.setattr(ici, "CATALOG_DIR", catalog_dir)
        monkeypatch.setattr(ici, "ATOMIC_INDEX", catalog_dir / "atomic_index.csv")

        ici.ingest_game(game_name)
        idx = catalog_dir / "atomic_index.csv"
        with idx.open() as f:
            rows = list(csv.DictReader(f))

        # Clip numbers should start at 001 and increment
        assert rows[0]["clip_name"].startswith("001__")
        assert rows[1]["clip_name"].startswith("002__")
        assert rows[2]["clip_name"].startswith("003__")

    def test_no_clip_index(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(ici, "GAMES_DIR", tmp_path / "out" / "games")
        result = ici.ingest_game("nonexistent_game")
        assert result["added"] == 0


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------

class TestScan:
    def test_find_clip_indexes(self, tmp_path, monkeypatch):
        games_dir = tmp_path / "out" / "games"
        g1 = games_dir / "2025-01-01__A_vs_B"
        g1.mkdir(parents=True)
        (g1 / "clip_index.txt").write_text("t100-t200 GOAL\n")

        g2 = games_dir / "2025-02-02__C_vs_D"
        g2.mkdir(parents=True)
        # No clip_index.txt

        monkeypatch.setattr(ici, "GAMES_DIR", games_dir)
        found = ici.scan_for_clip_indexes()
        assert len(found) == 1
        assert found[0][0] == "2025-01-01__A_vs_B"


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

class TestCLI:
    def test_scan_flag(self):
        parser = ici.build_parser()
        args = parser.parse_args(["--scan"])
        assert args.scan

    def test_all_flag(self):
        parser = ici.build_parser()
        args = parser.parse_args(["--all"])
        assert args.all

    def test_game_flag(self):
        parser = ici.build_parser()
        args = parser.parse_args(["--game", "2025-01-01__A"])
        assert args.game == "2025-01-01__A"

    def test_dry_run(self):
        parser = ici.build_parser()
        args = parser.parse_args(["--all", "--dry-run"])
        assert args.dry_run

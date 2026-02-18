"""Tests for tools/bootstrap_clips.py — clip bootstrapping from game indexes."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import bootstrap_clips as bc


# ---------------------------------------------------------------------------
# ClipDef
# ---------------------------------------------------------------------------

class TestClipDef:
    def test_duration(self):
        clip = bc.ClipDef(clip_num=1, label="GOAL", t_start_s=100.0, t_end_s=115.0)
        assert clip.duration_s == 15.0

    def test_canonical_name(self):
        clip = bc.ClipDef(clip_num=3, label="Goal", t_start_s=48.5, t_end_s=58.0)
        name = clip.canonical_name("2025-04-12__TSC_vs_Route_66")
        assert name == "003__2025-04-12__TSC_vs_Route_66__GOAL__t48.50-t58.00.mp4"

    def test_canonical_name_ampersand(self):
        clip = bc.ClipDef(clip_num=7, label="Pressure & Cross", t_start_s=100.0, t_end_s=120.0)
        name = clip.canonical_name("2025-09-20__TSC_vs_GW")
        assert "PRESSURE_AND_CROSS" in name
        assert "&" not in name

    def test_canonical_name_double_underscore(self):
        clip = bc.ClipDef(clip_num=1, label="Build  Up  Play", t_start_s=0.0, t_end_s=10.0)
        name = clip.canonical_name("GAME")
        # Double underscores from spaces should be collapsed
        assert "___" not in name


# ---------------------------------------------------------------------------
# CSV loading — plays_manual.csv
# ---------------------------------------------------------------------------

class TestLoadPlaysManual:
    def test_basic_csv(self, tmp_path):
        csv_path = tmp_path / "plays_manual.csv"
        csv_path.write_text(
            "clip_id,label,master_start,master_end,notes\n"
            "1,Goal,6:27,6:50,nice finish\n"
            "2,Shot,7:10,7:25,\n"
        )
        clips = bc.load_plays_manual(csv_path)
        assert len(clips) == 2
        assert clips[0].clip_num == 1
        assert clips[0].label == "Goal"
        assert clips[0].t_start_s == pytest.approx(387.0)
        assert clips[0].t_end_s == pytest.approx(410.0)

    def test_skips_empty_clip_id(self, tmp_path):
        csv_path = tmp_path / "plays_manual.csv"
        csv_path.write_text(
            "clip_id,label,master_start,master_end,notes\n"
            ",Goal,6:27,6:50,\n"
            "1,Shot,7:10,7:25,\n"
        )
        clips = bc.load_plays_manual(csv_path)
        assert len(clips) == 1

    def test_skips_invalid_timestamps(self, tmp_path):
        csv_path = tmp_path / "plays_manual.csv"
        csv_path.write_text(
            "clip_id,label,master_start,master_end,notes\n"
            "1,Goal,abc,def,\n"
        )
        clips = bc.load_plays_manual(csv_path)
        assert len(clips) == 0

    def test_skips_reversed_timestamps(self, tmp_path):
        csv_path = tmp_path / "plays_manual.csv"
        csv_path.write_text(
            "clip_id,label,master_start,master_end,notes\n"
            "1,Goal,10:00,5:00,reversed\n"
        )
        clips = bc.load_plays_manual(csv_path)
        assert len(clips) == 0

    def test_deduplicates_clip_ids(self, tmp_path):
        csv_path = tmp_path / "plays_manual.csv"
        csv_path.write_text(
            "clip_id,label,master_start,master_end,notes\n"
            "1,Goal,6:27,6:50,\n"
            "1,Shot,7:10,7:25,duplicate\n"
        )
        clips = bc.load_plays_manual(csv_path)
        assert len(clips) == 1
        assert clips[0].label == "Goal"

    def test_three_part_disambiguation(self, tmp_path):
        """Navy-style H:MM:SS timestamps disambiguated via parse_timestamp_pair."""
        csv_path = tmp_path / "plays_manual.csv"
        csv_path.write_text(
            "clip_id,label,master_start,master_end,notes\n"
            "1,Pressure,0:06:37,0:07:10,\n"
        )
        clips = bc.load_plays_manual(csv_path)
        assert len(clips) == 1
        assert clips[0].t_start_s == pytest.approx(397.0)
        assert clips[0].t_end_s == pytest.approx(430.0)


# ---------------------------------------------------------------------------
# CSV loading — clip_index.csv
# ---------------------------------------------------------------------------

class TestLoadClipIndex:
    def test_basic_csv(self, tmp_path):
        csv_path = tmp_path / "clip_index.csv"
        csv_path.write_text(
            "clip_num,description,start,end\n"
            "1,Goal,421,441\n"
            "2,Shot,500.5,520.5\n"
        )
        clips = bc.load_clip_index(csv_path)
        assert len(clips) == 2
        assert clips[0].t_start_s == pytest.approx(421.0)
        assert clips[1].t_start_s == pytest.approx(500.5)


# ---------------------------------------------------------------------------
# Game clip loading
# ---------------------------------------------------------------------------

class TestLoadGameClips:
    def test_prefers_plays_manual(self, tmp_path):
        game_dir = tmp_path / "game"
        game_dir.mkdir()
        (game_dir / "plays_manual.csv").write_text(
            "clip_id,label,master_start,master_end,notes\n"
            "1,Goal,6:27,6:50,\n"
        )
        (game_dir / "clip_index.csv").write_text(
            "clip_num,description,start,end\n"
            "1,Shot,100,200\n"
        )
        clips, source = bc.load_game_clips(game_dir)
        assert source == "plays_manual.csv"
        assert len(clips) == 1
        assert clips[0].label == "Goal"

    def test_falls_back_to_clip_index(self, tmp_path):
        game_dir = tmp_path / "game"
        game_dir.mkdir()
        (game_dir / "clip_index.csv").write_text(
            "clip_num,description,start,end\n"
            "1,Shot,100,200\n"
        )
        clips, source = bc.load_game_clips(game_dir)
        assert source == "clip_index.csv"

    def test_no_index_files(self, tmp_path):
        game_dir = tmp_path / "game"
        game_dir.mkdir()
        clips, source = bc.load_game_clips(game_dir)
        assert clips == []
        assert source == ""


# ---------------------------------------------------------------------------
# Master resolution
# ---------------------------------------------------------------------------

class TestFindMaster:
    def test_finds_master_mp4(self, tmp_path):
        (tmp_path / "MASTER.mp4").touch()
        assert bc.find_master(tmp_path) == tmp_path / "MASTER.mp4"

    def test_finds_full_game(self, tmp_path):
        (tmp_path / "full_game_720p.mp4").touch()
        result = bc.find_master(tmp_path)
        assert result is not None
        assert "full_game" in result.name

    def test_prefers_master_over_full_game(self, tmp_path):
        (tmp_path / "MASTER.mp4").touch()
        (tmp_path / "full_game.mp4").touch()
        assert bc.find_master(tmp_path).name == "MASTER.mp4"

    def test_none_when_missing(self, tmp_path):
        assert bc.find_master(tmp_path) is None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCLI:
    def test_report_flag(self):
        args = bc.build_parser().parse_args(["--report"])
        assert args.report

    def test_all_flag(self):
        args = bc.build_parser().parse_args(["--all"])
        assert args.all

    def test_game_flag(self):
        args = bc.build_parser().parse_args(["--game", "SLSG"])
        assert args.game == "SLSG"

    def test_dry_run(self):
        args = bc.build_parser().parse_args(["--all", "--dry-run"])
        assert args.dry_run

    def test_mutually_exclusive(self):
        with pytest.raises(SystemExit):
            bc.build_parser().parse_args(["--report", "--all"])

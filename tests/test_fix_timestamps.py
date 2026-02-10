"""Tests for tools/fix_timestamps.py — timestamp detection and repair."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import fix_timestamps as ft


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row(clip_name, t_start, t_end, duration=""):
    return {
        "clip_name": clip_name,
        "clip_rel": f"out/atomic_clips/game/{clip_name}",
        "t_start_s": str(t_start),
        "t_end_s": str(t_end),
        "duration_s": str(duration),
    }


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

class TestParseGameAndNum:
    def test_standard(self):
        game, num = ft._parse_game_and_num(
            "003__2025-09-20__TSC_vs_Greenwood__GOAL__t421-t441.mp4"
        )
        assert game == "2025-09-20__TSC_vs_Greenwood"
        assert num == "003"

    def test_no_match(self):
        game, num = ft._parse_game_and_num("random_file.mp4")
        assert game is None
        assert num is None


class TestFilenameTimestamps:
    def test_standard(self):
        s, e = ft._filename_timestamps("003__game__GOAL__t25260.00-t26460.00.mp4")
        assert s == 25260.0
        assert e == 26460.0

    def test_with_second_t(self):
        s, e = ft._filename_timestamps("001__game__SHOT__t7320.00-t7860.00.mp4")
        assert s == 7320.0
        assert e == 7860.0

    def test_no_timestamps(self):
        s, e = ft._filename_timestamps("random_file.mp4")
        assert s is None
        assert e is None


# ---------------------------------------------------------------------------
# Portrait lookup
# ---------------------------------------------------------------------------

class TestPortraitLookup:
    def test_builds_lookup(self):
        rows = [
            _row("003__2025-09-20__TSC_vs_GW__GOAL__t421-t441_portrait_POST.mp4", 421, 441, 20),
            _row("003__2025-09-20__TSC_vs_GW__GOAL__t25260-t26460.mp4", 25260, 26460, 1.2),
        ]
        lookup = ft.build_portrait_lookup(rows)
        assert ("2025-09-20__TSC_vs_GW", "003") in lookup
        assert lookup[("2025-09-20__TSC_vs_GW", "003")] == (421.0, 441.0)

    def test_ignores_non_portrait(self):
        rows = [
            _row("003__2025-09-20__TSC_vs_GW__GOAL__t25260-t26460.mp4", 25260, 26460, 1.2),
        ]
        lookup = ft.build_portrait_lookup(rows)
        assert len(lookup) == 0


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

class TestFindBroken:
    def test_detects_negative_duration(self):
        rows = [
            _row("001__2025-09-20__TSC_vs_GW__SHOT__t7320-t7860.mp4", 7320, 3137),
        ]
        broken = ft.find_broken_clips(rows)
        assert len(broken) == 1

    def test_skips_portrait(self):
        rows = [
            _row("001__game__SHOT__t100-t200_portrait_POST.mp4", 200, 100),
        ]
        broken = ft.find_broken_clips(rows)
        assert len(broken) == 0

    def test_skips_valid(self):
        rows = [
            _row("001__2025-09-20__TSC_vs_GW__SHOT__t100-t200.mp4", 100, 200, 100),
        ]
        broken = ft.find_broken_clips(rows)
        assert len(broken) == 0


# ---------------------------------------------------------------------------
# Fix computation
# ---------------------------------------------------------------------------

class TestComputeFix:
    def test_portrait_match(self):
        portrait = {("2025-09-20__TSC_vs_GW", "027"): (2040.0, 2058.0)}
        row = _row("027__2025-09-20__TSC_vs_GW__SHOT__t2937600-t3002400.mp4", 2937600, 2996)
        result = ft.compute_fix(row, portrait)
        assert result is not None
        assert result == (2040.0, 2058.0, "portrait")

    def test_div60_fallback(self):
        row = _row("001__2025-09-20__TSC_vs_RVFC__SHOT__t7320-t7860.mp4", 7320, 3137)
        result = ft.compute_fix(row, {})
        assert result is not None
        assert result == (122.0, 131.0, "/60")

    def test_div1440_fallback(self):
        row = _row("014__2025-10-04__TSC_vs_Tulsa__SHOT__t2354400-t2394000.mp4", 2354400, 1950)
        result = ft.compute_fix(row, {})
        assert result is not None
        assert result[0] == pytest.approx(1635.0)
        assert result[1] == pytest.approx(1662.5)
        assert result[2] == "/1440"

    def test_no_fix_available(self):
        row = _row("bad__file.mp4", 99999999, 1)
        result = ft.compute_fix(row, {})
        assert result is None


# ---------------------------------------------------------------------------
# Apply fixes
# ---------------------------------------------------------------------------

class TestApplyFixes:
    def test_fixes_broken_with_portrait(self):
        rows = [
            _row("027__2025-09-20__TSC_vs_GW__SHOT__t2937600-t3002400.mp4", 2937600, 2996, 1.2),
            _row("027__2025-09-20__TSC_vs_GW__SHOT__t2040-t2058_portrait_POST.mp4", 2040, 2058, 18),
        ]
        fixed = ft.apply_fixes(rows)
        assert fixed == 1
        assert rows[0]["t_start_s"] == "2040.0"
        assert rows[0]["t_end_s"] == "2058.0"
        assert rows[0]["duration_s"] == "18.000"

    def test_dry_run_does_not_modify(self):
        rows = [
            _row("001__2025-09-20__TSC_vs_RVFC__SHOT__t7320-t7860.mp4", 7320, 3137, 0.667),
        ]
        fixed = ft.apply_fixes(rows, dry_run=True)
        assert fixed == 1
        assert rows[0]["t_start_s"] == "7320"  # unchanged

    def test_skips_valid_clips(self):
        rows = [
            _row("003__2025-09-20__TSC_vs_GW__GOAL__t421-t441.mp4", 421, 441, 20),
        ]
        fixed = ft.apply_fixes(rows)
        assert fixed == 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCLI:
    def test_scan_flag(self):
        args = ft.build_parser().parse_args(["--scan"])
        assert args.scan

    def test_fix_flag(self):
        args = ft.build_parser().parse_args(["--fix"])
        assert args.fix

    def test_dry_run(self):
        args = ft.build_parser().parse_args(["--fix", "--dry-run"])
        assert args.fix and args.dry_run

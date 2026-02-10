"""Tests for soccer_highlights.clip_gating."""
from __future__ import annotations

import math
from collections import namedtuple
from dataclasses import dataclass

import pytest

from soccer_highlights.clip_gating import (
    ball_on_pitch,
    ball_speed,
    first_live_frame,
    has_touch,
    last_action_index,
    moving_players_count,
    trim_to_live,
    _ensure_sequence,
    _extract_bool,
    _extract_float,
    _normalise_scalar,
    _to_mapping,
)


# --- _to_mapping ----------------------------------------------------------

class TestToMapping:
    def test_none(self):
        assert _to_mapping(None) == {}

    def test_dict(self):
        d = {"a": 1}
        assert _to_mapping(d) is d  # returns same mutable mapping

    def test_immutable_mapping(self):
        from types import MappingProxyType
        m = MappingProxyType({"x": 10})
        result = _to_mapping(m)
        assert result == {"x": 10}
        assert isinstance(result, dict)

    def test_dataclass(self):
        @dataclass
        class Pos:
            x: float
            y: float
        result = _to_mapping(Pos(1.0, 2.0))
        assert result == {"x": 1.0, "y": 2.0}

    def test_namedtuple(self):
        Point = namedtuple("Point", ["x", "y"])
        result = _to_mapping(Point(3, 4))
        assert result == {"x": 3, "y": 4}

    def test_object_with_dict(self):
        class Obj:
            def __init__(self):
                self.val = 42
        result = _to_mapping(Obj())
        assert result == {"val": 42}


# --- _extract_bool --------------------------------------------------------

class TestExtractBool:
    def test_true_strings(self):
        for s in ("true", "yes", "on", "live", "in", "1"):
            assert _extract_bool({"flag": s}, "flag") is True

    def test_false_strings(self):
        for s in ("false", "no", "off", "out", "dead", "0"):
            assert _extract_bool({"flag": s}, "flag") is False

    def test_bool_value(self):
        assert _extract_bool({"a": True}, "a") is True
        assert _extract_bool({"a": False}, "a") is False

    def test_int_value(self):
        assert _extract_bool({"a": 1}, "a") is True
        assert _extract_bool({"a": 0}, "a") is False

    def test_nan_skipped(self):
        assert _extract_bool({"a": float("nan")}, "a") is None

    def test_missing_key(self):
        assert _extract_bool({"a": 1}, "b") is None

    def test_none_value(self):
        assert _extract_bool({"a": None}, "a") is None

    def test_fallback_to_second_key(self):
        assert _extract_bool({"b": True}, "a", "b") is True


# --- _extract_float -------------------------------------------------------

class TestExtractFloat:
    def test_float_value(self):
        assert _extract_float({"v": 3.14}, "v") == pytest.approx(3.14)

    def test_int_value(self):
        assert _extract_float({"v": 7}, "v") == 7.0

    def test_string_float(self):
        assert _extract_float({"v": "3.5"}, "v") == pytest.approx(3.5)

    def test_nan_skipped(self):
        assert _extract_float({"v": float("nan")}, "v") is None
        assert _extract_float({"v": "nan"}, "v") is None

    def test_empty_string(self):
        assert _extract_float({"v": ""}, "v") is None

    def test_missing_key(self):
        assert _extract_float({"a": 1}, "b") is None

    def test_invalid_string(self):
        assert _extract_float({"v": "abc"}, "v") is None

    def test_fallback_keys(self):
        assert _extract_float({"b": 5.0}, "a", "b") == 5.0


# --- _normalise_scalar ----------------------------------------------------

class TestNormaliseScalar:
    def test_none(self):
        assert _normalise_scalar(None) is None

    def test_float(self):
        assert _normalise_scalar(3.14) == pytest.approx(3.14)

    def test_int(self):
        assert _normalise_scalar(5) == 5.0

    def test_nan(self):
        assert _normalise_scalar(float("nan")) is None


# --- _ensure_sequence -----------------------------------------------------

class TestEnsureSequence:
    def test_none(self):
        assert _ensure_sequence(None) == []

    def test_list(self):
        assert _ensure_sequence([1, 2, 3]) == [1, 2, 3]

    def test_tuple(self):
        assert _ensure_sequence((1, 2)) == [1, 2]

    def test_string_returns_empty(self):
        assert _ensure_sequence("hello") == []

    def test_bytes_returns_empty(self):
        assert _ensure_sequence(b"data") == []


# --- ball_on_pitch --------------------------------------------------------

class TestBallOnPitch:
    def test_explicit_bool(self):
        assert ball_on_pitch({"ball_on_pitch": True}) is True
        assert ball_on_pitch({"ball_on_pitch": False}) is False

    def test_live_ball_key(self):
        assert ball_on_pitch({"live_ball": "yes"}) is True

    def test_pitch_ratio_above_threshold(self):
        assert ball_on_pitch({"field_ratio": 0.5}) is True

    def test_pitch_ratio_below_threshold(self):
        assert ball_on_pitch({"field_ratio": 0.05}) is False

    def test_pitch_ratio_percentage(self):
        # >1.0 is treated as percentage, divided by 100
        assert ball_on_pitch({"pitch_ratio": 50.0}) is True
        assert ball_on_pitch({"pitch_ratio": 5.0}) is False

    def test_region_string(self):
        assert ball_on_pitch({"ball_region": "pitch"}) is True
        assert ball_on_pitch({"ball_region": "out"}) is False

    def test_empty_frame(self):
        assert ball_on_pitch({}) is False


# --- ball_speed -----------------------------------------------------------

class TestBallSpeed:
    def test_direct_speed(self):
        traj = [{"speed": 15.0}]
        assert ball_speed(traj, 0) == pytest.approx(15.0)

    def test_speed_auto_scale(self):
        # speed <= 1.0 gets multiplied by scale or default 10
        traj = [{"speed": 0.5}]
        assert ball_speed(traj, 0) == pytest.approx(5.0)

    def test_speed_with_custom_scale(self):
        traj = [{"speed": 0.3, "speed_scale": 20.0}]
        assert ball_speed(traj, 0) == pytest.approx(6.0)

    def test_velocity_components(self):
        traj = [{"vx": 3.0, "vy": 4.0}]
        assert ball_speed(traj, 0) == pytest.approx(5.0)

    def test_positional_delta(self):
        traj = [{"x": 0.0, "y": 0.0}, {"x": 3.0, "y": 4.0}]
        assert ball_speed(traj, 1) == pytest.approx(5.0)

    def test_missing_returns_zero(self):
        assert ball_speed([], 0) == 0.0
        assert ball_speed(None, 0) == 0.0


# --- has_touch ------------------------------------------------------------

class TestHasTouch:
    def test_direct_flag(self):
        traj = [{"touch": True}]
        assert has_touch(None, traj, 0) is True

    def test_direct_flag_false(self):
        traj = [{"touch": False}]
        assert has_touch(None, traj, 0) is False

    def test_touch_prob_high(self):
        traj = [{"touch_prob": 0.8}]
        assert has_touch(None, traj, 0) is True

    def test_player_near_ball(self):
        traj = [{"x": 100.0, "y": 100.0}]
        players = [[{"x": 110.0, "y": 100.0, "speed": 5.0}]]
        assert has_touch(players, traj, 0) is True

    def test_player_far_from_ball(self):
        traj = [{"x": 100.0, "y": 100.0}]
        players = [[{"x": 500.0, "y": 500.0, "speed": 5.0}]]
        assert has_touch(players, traj, 0) is False


# --- moving_players_count -------------------------------------------------

class TestMovingPlayersCount:
    def test_direct_count(self):
        assert moving_players_count({"moving_players": 8}) == 8

    def test_normalized_count(self):
        # <= 1.0 gets multiplied by 10
        assert moving_players_count({"moving_players": 0.6}) == 6

    def test_player_speeds(self):
        frame = {"player_speeds": [0.5, 0.3, 0.1, 0.8]}
        # 3 values >= 0.25 flow_min
        assert moving_players_count(frame) == 3

    def test_motion_fallback(self):
        frame = {"flow": 0.7}
        assert moving_players_count(frame) == 7

    def test_empty_frame(self):
        assert moving_players_count({}) == 0


# --- first_live_frame -----------------------------------------------------

class TestFirstLiveFrame:
    def test_found(self):
        frames = [
            {"ball_on_pitch": False},
            {"ball_on_pitch": True, "moving_players": 6},
            {"ball_on_pitch": True, "moving_players": 8},
        ]
        traj = [
            {"speed": 0.0},
            {"speed": 2.0, "touch": True},
            {"speed": 3.0, "touch": True},
        ]
        result = first_live_frame(frames, traj, None)
        assert result == 1

    def test_not_found(self):
        frames = [{"ball_on_pitch": False}]
        traj = [{"speed": 0.0}]
        assert first_live_frame(frames, traj, None) is None

    def test_empty(self):
        assert first_live_frame([], [], None) is None


# --- last_action_index ----------------------------------------------------

class TestLastActionIndex:
    def test_action_continues(self):
        frames = [
            {"ball_on_pitch": True, "moving_players": 6},
            {"ball_on_pitch": True, "moving_players": 6},
            {"ball_on_pitch": True, "moving_players": 6},
        ]
        traj = [
            {"speed": 3.0, "touch": True},
            {"speed": 3.0, "touch": True},
            {"speed": 3.0, "touch": True},
        ]
        result = last_action_index(frames, traj, None, 0, idle_tolerance=5)
        assert result == 2

    def test_idle_breaks(self):
        frames = [
            {"ball_on_pitch": True, "moving_players": 6},
            {"ball_on_pitch": False, "moving_players": 0},
            {"ball_on_pitch": False, "moving_players": 0},
        ]
        traj = [
            {"speed": 3.0, "touch": True},
            {"speed": 0.0},
            {"speed": 0.0},
        ]
        result = last_action_index(frames, traj, None, 0, idle_tolerance=1)
        assert result == 0


# --- trim_to_live ---------------------------------------------------------

class TestTrimToLive:
    def test_empty(self):
        assert trim_to_live([], [], None) is None

    def test_no_live_frame(self):
        frames = [{"ball_on_pitch": False}]
        traj = [{"speed": 0.0}]
        assert trim_to_live(frames, traj, None) is None

    def test_valid_trim(self):
        frames = [
            {"ball_on_pitch": True, "moving_players": 6},
            {"ball_on_pitch": True, "moving_players": 6},
        ]
        traj = [
            {"speed": 3.0, "touch": True},
            {"speed": 3.0, "touch": True},
        ]
        result = trim_to_live(frames, traj, None, pre=0.0, post=0.0, fps=30.0)
        assert result is not None
        start, end = result
        assert start == 0
        assert end == 1

"""Tests for soccer_highlights.config."""
from __future__ import annotations

import pytest
from pathlib import Path

from soccer_highlights.config import (
    AppConfig,
    BallConfig,
    DetectConfig,
    PathsConfig,
    ShrinkConfig,
    _parse_scalar,
    _simple_yaml_load,
    load_config,
)


# --- _parse_scalar -------------------------------------------------------

class TestParseScalar:
    def test_empty(self):
        assert _parse_scalar("") is None
        assert _parse_scalar("   ") is None

    def test_null_variants(self):
        for s in ("null", "None", "~", "NULL"):
            assert _parse_scalar(s) is None

    def test_true_variants(self):
        for s in ("true", "True", "yes", "YES", "on"):
            assert _parse_scalar(s) is True

    def test_false_variants(self):
        for s in ("false", "False", "no", "NO", "off"):
            assert _parse_scalar(s) is False

    def test_int(self):
        assert _parse_scalar("42") == 42
        assert _parse_scalar("-3") == -3

    def test_float(self):
        assert _parse_scalar("3.14") == pytest.approx(3.14)
        assert _parse_scalar("1e3") == pytest.approx(1000.0)

    def test_quoted_string(self):
        assert _parse_scalar('"hello"') == "hello"
        assert _parse_scalar("'world'") == "world"

    def test_bare_string(self):
        assert _parse_scalar("some_value") == "some_value"

    def test_inline_list(self):
        result = _parse_scalar("[1, 2, 3]")
        assert result == [1, 2, 3]

    def test_empty_inline_list(self):
        result = _parse_scalar("[]")
        assert result == []


# --- _simple_yaml_load ----------------------------------------------------

class TestSimpleYamlLoad:
    def test_empty(self):
        assert _simple_yaml_load("") == {}
        assert _simple_yaml_load("   \n\n  ") == {}

    def test_flat_mapping(self):
        result = _simple_yaml_load("a: 1\nb: hello\nc: true\n")
        assert result == {"a": 1, "b": "hello", "c": True}

    def test_nested_mapping(self):
        result = _simple_yaml_load("outer:\n  inner: 42\n")
        assert result == {"outer": {"inner": 42}}

    def test_comments_stripped(self):
        result = _simple_yaml_load("key: value # this is a comment\n")
        assert result == {"key": "value"}

    def test_hash_inside_quotes_preserved(self):
        result = _simple_yaml_load('title: "Game #5"\n')
        assert result == {"title": "Game #5"}

    def test_sequence(self):
        result = _simple_yaml_load("items:\n  - one\n  - two\n  - three\n")
        assert result == {"items": ["one", "two", "three"]}

    def test_sequence_of_mappings(self):
        text = "items:\n  - name: a\n  - name: b\n"
        result = _simple_yaml_load(text)
        assert result == {"items": [{"name": "a"}, {"name": "b"}]}

    def test_null_value(self):
        result = _simple_yaml_load("key: null\n")
        assert result == {"key": None}


# --- load_config ----------------------------------------------------------

class TestLoadConfig:
    def test_missing_file_returns_defaults(self, tmp_path):
        cfg = load_config(tmp_path / "nonexistent.yaml")
        assert isinstance(cfg, AppConfig)
        assert cfg.detect.min_gap == 2.0

    def test_load_from_file(self, sample_yaml):
        cfg = load_config(sample_yaml)
        assert cfg.paths.video == Path("game.mp4")
        assert cfg.paths.output_dir == Path("results")
        assert cfg.detect.min_gap == 3.0
        assert cfg.detect.pre == 1.5
        assert cfg.rank.k == 5

    def test_defaults_preserved(self, sample_yaml):
        cfg = load_config(sample_yaml)
        # Fields not in the yaml get defaults
        assert cfg.detect.post == 2.0
        assert cfg.clips.crf == 20


# --- AppConfig defaults ---------------------------------------------------

class TestAppConfigDefaults:
    def test_default_construction(self):
        cfg = AppConfig()
        assert cfg.paths.video == Path("full_game_stabilized.mp4")
        assert cfg.detect.audio_weight == 0.5
        assert cfg.rank.k == 10
        assert cfg.clips.preset == "veryfast"

    def test_resolve_profile_known(self):
        cfg = AppConfig()
        p = cfg.resolve_profile("broadcast")
        assert p.name == "broadcast"
        assert p.width == 1920

    def test_resolve_profile_unknown_raises(self):
        cfg = AppConfig()
        with pytest.raises(KeyError, match="Unknown reel profile"):
            cfg.resolve_profile("nonexistent")


# --- Validators -----------------------------------------------------------

class TestValidators:
    def test_shrink_mode_valid(self):
        sc = ShrinkConfig(mode="smart")
        assert sc.mode == "smart"
        sc2 = ShrinkConfig(mode="simple")
        assert sc2.mode == "simple"

    def test_shrink_mode_invalid(self):
        with pytest.raises(ValueError, match="mode"):
            ShrinkConfig(mode="invalid")

    def test_shrink_aspect_valid(self):
        sc = ShrinkConfig(aspect="vertical")
        assert sc.aspect == "vertical"

    def test_shrink_aspect_invalid(self):
        with pytest.raises(ValueError, match="aspect"):
            ShrinkConfig(aspect="diagonal")

    def test_ball_detector_normalizes(self):
        bc = BallConfig(detector="YOLO")
        assert bc.detector == "yolo"

    def test_ball_detector_invalid(self):
        with pytest.raises(ValueError, match="detector"):
            BallConfig(detector="tensorflow")

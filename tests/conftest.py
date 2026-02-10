"""Shared fixtures for the soccer highlights test suite."""
from __future__ import annotations

import pytest
from pathlib import Path

from soccer_highlights.config import AppConfig, load_config
from soccer_highlights.utils import HighlightWindow


@pytest.fixture
def default_config() -> AppConfig:
    """Return a default AppConfig with no file."""
    return AppConfig()


@pytest.fixture
def sample_windows() -> list[HighlightWindow]:
    """A small collection of highlight windows for testing."""
    return [
        HighlightWindow(start=0.0, end=5.0, score=0.8, event="scene"),
        HighlightWindow(start=10.0, end=18.0, score=1.0, event="goal"),
        HighlightWindow(start=25.0, end=30.0, score=0.6, event="scene"),
    ]


@pytest.fixture
def sample_yaml(tmp_path: Path) -> Path:
    """Write a minimal config.yaml and return its path."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "paths:\n"
        "  video: game.mp4\n"
        "  output_dir: results\n"
        "detect:\n"
        "  min_gap: 3.0\n"
        "  pre: 1.5\n"
        "rank:\n"
        "  k: 5\n"
    )
    return cfg

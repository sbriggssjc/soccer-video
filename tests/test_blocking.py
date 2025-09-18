from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).resolve().parents[1]
BLOCKING = _load_module(ROOT / "soccer_highlights" / "blocking.py", "blocking_mod")
ClipBlockState = BLOCKING.ClipBlockState
first_live_frame = BLOCKING.first_live_frame


@dataclass
class HighlightWindow:
    start: float
    end: float
    score: float
    event: str = "scene"


def test_first_live_frame_uses_next_touch() -> None:
    touches = [12.0, 18.5, 24.0]
    assert first_live_frame(17.0, touches) == pytest.approx(18.5)


def test_first_live_frame_fallback_when_missing() -> None:
    assert first_live_frame(5.0, [], fallback=8.2, max_wait=10.0) == pytest.approx(8.2)


def test_clip_block_state_blocks_until_touch() -> None:
    state = ClipBlockState()
    resume = state.record_restart(30.0, touches=[31.0, 33.5])
    assert resume == pytest.approx(31.0)
    assert state.block_until == pytest.approx(31.0)
    assert state.is_blocked(29.0, 30.5)
    assert not state.is_blocked(32.0, 34.0)


def test_out_of_play_merges_overlaps() -> None:
    state = ClipBlockState()
    state.add_out_of_play(10.0, touches=[11.0], linger=0.0, max_wait=5.0)
    state.add_out_of_play(10.8, touches=[12.0], linger=0.0, max_wait=5.0)
    zones = state.no_clip_windows
    assert len(zones) == 1
    start, end = zones[0]
    assert start == pytest.approx(10.0)
    assert end == pytest.approx(12.0)


def test_stoppage_zone_extends_beyond_cluster() -> None:
    state = ClipBlockState()
    zone = state.add_stoppage(40.0, 42.0, touches=[43.5], linger=1.0)
    assert zone[0] == pytest.approx(40.0)
    assert zone[1] == pytest.approx(43.5)


def test_filter_windows_excludes_categories_and_zones() -> None:
    state = ClipBlockState()
    state.add_out_of_play(95.0, touches=[98.0], linger=0.0)
    windows = [
        HighlightWindow(start=90.0, end=92.0, score=0.5, event="scene"),
        HighlightWindow(start=94.0, end=96.0, score=0.6, event="restart"),
        HighlightWindow(start=96.5, end=97.2, score=0.4, event="scene"),
    ]
    filtered = state.filter_windows(windows)
    assert len(filtered) == 1
    assert filtered[0].start == pytest.approx(90.0)

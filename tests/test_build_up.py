from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

# Ensure the package root is importable when tests run without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from soccer_highlights.build_up import (
    detect_build_up,
    field_progress,
    pass_chains,
    span_to_clip,
    time_span,
)


@dataclass
class MockTrack:
    time: float
    x: float
    y: float
    team: str
    player: str


def _build_tracks() -> list[MockTrack]:
    # Build a sequence with a sustained navy possession followed by a turnover.
    return [
        MockTrack(time=5.0, x=12.0, y=30.0, team="navy", player="A"),
        MockTrack(time=6.0, x=15.5, y=33.0, team="navy", player="B"),
        MockTrack(time=7.0, x=18.0, y=35.5, team="navy", player="C"),
        MockTrack(time=8.0, x=22.5, y=38.0, team="navy", player="D"),
        MockTrack(time=8.9, x=26.0, y=42.0, team="navy", player="E"),
        # turnover (should not join chain)
        MockTrack(time=16.0, x=40.0, y=25.0, team="gold", player="F"),
        MockTrack(time=17.0, x=44.0, y=28.0, team="gold", player="G"),
        MockTrack(time=18.2, x=46.0, y=31.0, team="navy", player="H"),
        MockTrack(time=19.6, x=47.5, y=33.0, team="navy", player="I"),
        MockTrack(time=21.2, x=49.0, y=35.0, team="navy", player="J"),
        MockTrack(time=22.8, x=50.0, y=37.0, team="navy", player="K"),
    ]


BALL_EVENTS = [
    {"time": 4.7, "x": 11.0, "y": 29.0, "team": "navy"},
    {"time": 9.2, "x": 27.5, "y": 44.0, "team": "navy"},
    {"time": 15.4, "x": 38.0, "y": 23.5, "team": "gold"},
    {"time": 18.0, "x": 45.0, "y": 30.0, "team": "navy"},
    {"time": 24.5, "x": 50.5, "y": 38.0, "team": "navy"},
]


def test_pass_chains_groups_sequences() -> None:
    tracks = _build_tracks()
    chains = pass_chains(tracks, BALL_EVENTS, team="navy", max_gap_s=2.0, min_len=4)
    assert len(chains) == 2
    first, second = chains
    assert pytest.approx(first.start_time, rel=1e-6) == 4.7
    assert pytest.approx(first.end_time, rel=1e-6) == 9.2
    assert len(first.touches) == 5
    assert len(second.touches) == 4
    # ensure second possession starts around 18s (after turnover gap)
    assert second.start_time >= 18.0


def test_chain_metrics_and_clipping() -> None:
    tracks = _build_tracks()
    chains = pass_chains(tracks, BALL_EVENTS, team="navy", max_gap_s=2.0, min_len=4)
    first = chains[0]
    progress = field_progress(first)
    assert progress > 12.0
    assert pytest.approx(time_span(first), rel=1e-6) == pytest.approx(first.end_time - first.start_time)
    clip = span_to_clip(first, extra_pre=0.8, extra_post=1.2)
    assert clip.event == "build_up"
    assert clip.start <= first.start_time
    assert clip.end >= first.end_time
    assert clip.score > 0.0


def test_detect_build_up_filters_sequences() -> None:
    tracks = _build_tracks()
    clips = detect_build_up(frames=[], tracks=tracks, ball=BALL_EVENTS)
    assert len(clips) == 1  # only the first chain has enough forward progress
    clip = clips[0]
    assert clip.event == "build_up"
    # clip should include padding while clamping to zero when needed
    assert clip.start >= 0.0
    assert clip.end > clip.start
    assert clip.score > 0.0

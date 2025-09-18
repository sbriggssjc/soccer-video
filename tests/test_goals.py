from __future__ import annotations

import pytest

try:  # pragma: no cover - optional dependency for unit test
    import numpy as np
except Exception:  # pragma: no cover
    pytest.skip("numpy is required for goal helper tests", allow_module_level=True)

from soccer_highlights.goals import _merge_signals
from soccer_highlights.shrink import _ball_in_play_gate


def test_merge_signals_prefers_non_scoreboard_anchor() -> None:
    signals = [(100.0, "scoreboard"), (97.8, "net"), (200.0, "scoreboard"), (205.0, "crowd")]
    groups = _merge_signals(signals, tolerance=4.0)
    assert len(groups) == 2
    first, second = groups
    assert first.sources == {"scoreboard", "net"}
    assert first.anchor_time() == pytest.approx(97.8, abs=0.2)
    assert second.sources == {"scoreboard", "crowd"}
    anchor2 = second.anchor_time()
    assert 0.0 <= anchor2 < 205.0


def test_ball_in_play_gate_trims_dead_time() -> None:
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    motion = np.array([0.05, 0.1, 0.45, 0.9, 0.58, 0.2, 0.05], dtype=np.float32)
    start, end = _ball_in_play_gate(times, motion, peak_time=3.0, default_start=0.0, default_end=6.0, total_dur=7.0)
    assert 0.2 < start < 2.0
    assert start < 3.0
    assert 4.0 <= end <= 5.5
    assert end < 6.0
    assert end - start >= 1.2
    assert start <= 3.0 <= end

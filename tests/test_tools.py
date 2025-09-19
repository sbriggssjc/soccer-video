import pytest

pd = pytest.importorskip("pandas")

from tools.utils import merge_nearby_events, parse_time_to_seconds, snap_to_rising_edge


def test_parse_time_to_seconds_handles_various_formats():
    assert parse_time_to_seconds(12.5) == 12.5
    assert parse_time_to_seconds("01:02:03.500") == 3723.5
    assert parse_time_to_seconds("2:15") == 135.0
    assert parse_time_to_seconds("1,234.25") == 1234.25
    assert parse_time_to_seconds("00:00:00") == 0.0
    assert parse_time_to_seconds("invalid") is None


def test_merge_nearby_events_merges_close_events():
    df = pd.DataFrame(
        [
            {"t0": 10.0, "t1": 12.0, "label": "GOAL", "score": 0.5, "src": "core"},
            {"t0": 13.0, "t1": 15.0, "label": "GOAL", "score": 0.8, "src": "shots"},
            {"t0": 50.0, "t1": 55.0, "label": "SHOT", "score": 0.4, "src": "shots"},
        ]
    )
    merged = merge_nearby_events(df, merge_window=2.5)
    assert len(merged) == 2
    first = merged.iloc[0]
    assert first.t0 == 10.0 and first.t1 == 15.0
    assert first.score == 0.8
    assert first.label == "GOAL"
    assert first.src == "core+shots"


def test_snap_to_rising_edge_uses_latest_edge():
    times = pd.Index([0.0, 1.0, 2.0, 3.0, 4.0])
    in_play = pd.Series([False, False, True, False, True], index=times)
    adjusted = snap_to_rising_edge(in_play, nominal_start=3.5, lookback=2.5)
    # Rising edges at 2.0 and 4.0; 2.0 is within lookback window and before start.
    assert adjusted == 2.0

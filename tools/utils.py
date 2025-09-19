"""Utility helpers for soccer highlight tooling."""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import pandas as pd

def parse_time_to_seconds(value: object) -> Optional[float]:
    """Parse a time representation into seconds.

    Accepts numeric values (seconds) or strings in ``hh:mm:ss(.ms)`` or
    ``mm:ss(.ms)`` formats. Thousands separators are tolerated. Returns ``None``
    if the value cannot be interpreted.
    """

    if value is None:
        return None

    if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            return None
        return numeric

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.lower() in {"nan", "none", "null"}:
            return None
        if ":" in text:
            parts = text.split(":")
            try:
                values = [float(part.replace(",", ".")) for part in parts]
            except ValueError:
                return None
            total = 0.0
            for part in values:
                total = total * 60.0 + part
            return total
        if text.count(",") > 0 and text.count(".") == 0:
            text = text.replace(",", ".", 1).replace(",", "")
        else:
            text = text.replace(",", "")
        try:
            return float(text)
        except ValueError:
            return None

    return None


def merge_nearby_events(df: pd.DataFrame, merge_window: float = 2.0) -> pd.DataFrame:
    """Merge events with the same label that are temporally close.

    Parameters
    ----------
    df:
        DataFrame containing at least ``t0``, ``t1``, ``label``, ``score`` and
        ``src`` columns.
    merge_window:
        Maximum gap between events with the same label for them to be merged.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with merged rows sorted by start time.
    """

    if df.empty:
        return df.copy()

    records: List[dict] = []
    for label, group in df.sort_values("t0").groupby("label", sort=False):
        current = None
        for row in group.itertuples(index=False):
            data = {
                "t0": float(row.t0),
                "t1": float(row.t1),
                "label": label,
                "score": float(row.score) if not pd.isna(row.score) else 0.0,
                "src": {row.src} if isinstance(row.src, str) else set(),
            }
            if current is None:
                current = data
                continue

            if data["t0"] - current["t1"] <= merge_window:
                current["t1"] = max(current["t1"], data["t1"])
                current["t0"] = min(current["t0"], data["t0"])
                current["score"] = max(current["score"], data["score"])
                current["src"].update(data["src"])
            else:
                current["src"] = "+".join(sorted(current["src"])) if current["src"] else ""
                records.append(current)
                current = data
        if current is not None:
            current["src"] = "+".join(sorted(current["src"])) if current["src"] else ""
            records.append(current)

    merged = pd.DataFrame.from_records(records)
    return merged.sort_values("t0").reset_index(drop=True)


def snap_to_rising_edge(
    in_play: pd.Series,
    nominal_start: float,
    lookback: float = 4.0,
) -> float:
    """Snap a start time to the latest rising edge of an in-play signal.

    Parameters
    ----------
    in_play:
        Boolean series indexed by time (seconds). Values should be ``True`` when
        the ball is considered in play.
    nominal_start:
        The proposed clip start time.
    lookback:
        Maximum time (seconds) to look back when searching for a rising edge.

    Returns
    -------
    float
        Adjusted start time.
    """

    if in_play.empty:
        return nominal_start

    bool_series = in_play.sort_index().astype(bool)
    times = bool_series.index.to_numpy(dtype=float)
    values = bool_series.to_numpy()
    diffs = np.diff(values.astype(int), prepend=0)
    rising_edges = times[diffs == 1]
    if rising_edges.size == 0:
        return nominal_start

    window_start = max(0.0, nominal_start - lookback)
    eligible = rising_edges[(rising_edges >= window_start) & (rising_edges <= nominal_start)]
    if eligible.size == 0:
        return nominal_start
    return float(eligible[-1])

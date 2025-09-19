"""Aggregate and normalise soccer event CSV files."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

if __package__ is None or __package__ == "":  # pragma: no cover - CLI execution
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.append(str(_Path(__file__).resolve().parent.parent))

from tools.utils import merge_nearby_events, parse_time_to_seconds

START_CANDIDATES = [
    "t0",
    "start",
    "start_time",
    "startsec",
    "start_sec",
    "begin",
    "time",
    "timestamp",
]
END_CANDIDATES = [
    "t1",
    "end",
    "end_time",
    "stop",
    "finish",
    "endsec",
    "end_sec",
]
LABEL_CANDIDATES = [
    "label",
    "event",
    "type",
    "tag",
    "category",
    "play",
    "title",
]
SCORE_CANDIDATES = [
    "score",
    "rank",
    "rating",
    "priority",
    "confidence",
    "value",
]
DURATION_CANDIDATES = ["duration", "len", "length", "clip_length"]


def _find_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    lookup = {col.lower(): col for col in columns}
    for name in candidates:
        if name in lookup:
            return lookup[name]
    return None


def _parse_score(value: object) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return 0.0
    for needle in ["home", "away", "null", "nan"]:
        if needle in text.lower():
            return 0.0
    text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        return 0.0


def load_events(path: Path, source_name: str) -> List[dict]:
    if not path.exists():
        return []
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=["", "NA", "NaN"])  # type: ignore[arg-type]
    if df.empty:
        return []

    start_col = _find_column(df.columns, START_CANDIDATES)
    end_col = _find_column(df.columns, END_CANDIDATES)
    label_col = _find_column(df.columns, LABEL_CANDIDATES)
    score_col = _find_column(df.columns, SCORE_CANDIDATES)
    duration_col = _find_column(df.columns, DURATION_CANDIDATES)

    records: List[dict] = []
    for row in df.to_dict(orient="records"):
        start_val = parse_time_to_seconds(row.get(start_col)) if start_col else None
        end_val = parse_time_to_seconds(row.get(end_col)) if end_col else None
        label_val = row.get(label_col, source_name) if label_col else source_name
        label_text = str(label_val).strip() or source_name
        score_val = _parse_score(row.get(score_col)) if score_col else 0.0
        if end_val is None and duration_col:
            duration = parse_time_to_seconds(row.get(duration_col))
            if duration is not None and start_val is not None:
                end_val = start_val + duration
        if end_val is None and start_val is not None:
            end_val = start_val + 1.0
        if start_val is None or end_val is None:
            continue
        records.append(
            {
                "t0": float(start_val),
                "t1": float(end_val),
                "label": label_text.upper(),
                "score": float(score_val),
                "src": source_name,
            }
        )
    return records


def index_events(paths: List[Path]) -> pd.DataFrame:
    records: List[dict] = []
    for path in paths:
        records.extend(load_events(path, path.stem))
    if not records:
        return pd.DataFrame(columns=["t0", "t1", "label", "score", "src"])
    df = pd.DataFrame.from_records(records)
    df = df[(df["t1"] > df["t0"]) & df["t0"].notna() & df["t1"].notna()]
    df = df.sort_values("t0").reset_index(drop=True)
    merged = merge_nearby_events(df, merge_window=2.0)
    return merged


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index soccer event CSV files")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cores", type=Path, default=None)
    parser.add_argument("--goals", type=Path, default=None)
    parser.add_argument("--forced-goals", type=Path, default=None)
    parser.add_argument("--shots", type=Path, default=None)
    parser.add_argument("--filtered", type=Path, default=None)
    parser.add_argument("--build", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    paths = [p for p in [
        args.cores,
        args.goals,
        args.forced_goals,
        args.shots,
        args.filtered,
        args.build,
    ] if p is not None]
    df = index_events(paths)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, quoting=csv.QUOTE_MINIMAL)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())

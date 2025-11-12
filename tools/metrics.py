"""Metrics utilities for Smart Soccer Highlight Selector."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


@dataclass
class Metrics:
    num_spans: int
    num_goals_found: int
    num_shots_found: int
    num_saves_found: int
    num_woodwork_found: int
    total_duration: float
    avg_in_play_ratio: float
    coverage_of_resets: float
    uncovered_resets: List[float]

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2)


def _duration(row: pd.Series) -> float:
    return float(max(0.0, row["end"] - row["start"]))


def compute_metrics(
    events_df: pd.DataFrame,
    reset_times: Sequence[float],
    in_play_ratio: Dict[int, float],
) -> Metrics:
    """Compute summary metrics from the final event dataframe."""

    label_counts = events_df["label"].value_counts() if not events_df.empty else pd.Series(dtype=int)
    num_goals = int(label_counts.get("GOAL", 0))
    num_shots = int(label_counts.get("SHOT", 0) + label_counts.get("WOODWORK", 0))
    num_saves = int(label_counts.get("SAVE", 0))
    num_woodwork = int(label_counts.get("WOODWORK", 0))
    total_duration = float(events_df.apply(_duration, axis=1).sum()) if not events_df.empty else 0.0

    ratios = list(in_play_ratio.values())
    avg_ratio = float(np.mean(ratios)) if ratios else 0.0

    uncovered: List[float] = []
    if reset_times and not events_df.empty:
        for reset in reset_times:
            window_end = reset
            window_start = reset - 25.0
            window_anchor = reset - 12.0
            mask = (
                (events_df["label"] == "GOAL")
                & (events_df["end"] >= window_anchor)
                & (events_df["end"] <= window_end)
                & (events_df["start"] <= window_start)
            )
            if mask.sum() == 0:
                uncovered.append(float(reset))

    coverage = 0.0
    if reset_times:
        coverage = 1.0 - len(uncovered) / float(len(reset_times))

    return Metrics(
        num_spans=int(len(events_df)),
        num_goals_found=num_goals,
        num_shots_found=num_shots,
        num_saves_found=num_saves,
        num_woodwork_found=num_woodwork,
        total_duration=total_duration,
        avg_in_play_ratio=avg_ratio,
        coverage_of_resets=coverage,
        uncovered_resets=uncovered,
    )


def write_reports(metrics: Metrics, out_dir: Path) -> None:
    """Persist metrics to JSON and a human readable text file."""

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics_report.json").write_text(metrics.to_json() + "\n", encoding="utf-8")

    lines = [
        "Smart Select metrics report\n",
        "==========================\n",
        f"Spans exported : {metrics.num_spans}\n",
        f"Goals detected : {metrics.num_goals_found}\n",
        f"Shots detected : {metrics.num_shots_found}\n",
        f"Woodwork hits  : {metrics.num_woodwork_found}\n",
        f"Saves detected : {metrics.num_saves_found}\n",
        f"Total duration : {metrics.total_duration:.1f} s\n",
        f"Avg in-play    : {metrics.avg_in_play_ratio:.3f}\n",
        f"Goal coverage  : {metrics.coverage_of_resets:.3f}\n",
    ]
    if metrics.uncovered_resets:
        lines.append("\nUncovered goal resets:\n")
        for reset in metrics.uncovered_resets:
            lines.append(f"  - Reset @ {reset:.2f}s needs review\n")
    else:
        lines.append("\nAll goal resets covered.\n")

    (out_dir / "metrics_readable.txt").write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover - smoke test
    dummy = pd.DataFrame(
        [
            {"label": "GOAL", "start": 10.0, "end": 22.0},
            {"label": "SHOT", "start": 100.0, "end": 108.0},
        ]
    )
    metrics = compute_metrics(dummy, [40.0], {0: 0.92, 1: 0.96})
    write_reports(metrics, Path("/tmp/smart_metrics"))
    print(metrics)


"""Utility helpers used across pipeline stages."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Iterable, List, Sequence

from ._loguru import logger


@dataclass
class HighlightWindow:
    start: float
    end: float
    score: float
    event: str = "scene"

    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def read_highlights(csv_path: Path) -> List[HighlightWindow]:
    rows: List[HighlightWindow] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                HighlightWindow(
                    start=float(row.get("start", 0.0)),
                    end=float(row.get("end", 0.0)),
                    score=float(row.get("score", 0.0)),
                    event=row.get("event", "scene"),
                )
            )
    return rows


def write_highlights(csv_path: Path, windows: Sequence[HighlightWindow]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["start", "end", "score", "event"])
        for w in windows:
            writer.writerow([f"{w.start:.3f}", f"{w.end:.3f}", f"{w.score:.4f}", w.event])


def merge_overlaps(windows: Sequence[HighlightWindow], min_gap: float) -> List[HighlightWindow]:
    if not windows:
        return []
    ordered = sorted(windows, key=lambda w: w.start)
    merged: List[HighlightWindow] = [ordered[0]]
    for win in ordered[1:]:
        last = merged[-1]
        if win.start - last.end <= min_gap:
            new_end = max(last.end, win.end)
            new_score = max(last.score, win.score)
            merged[-1] = HighlightWindow(start=last.start, end=new_end, score=new_score, event=last.event)
        else:
            merged.append(win)
    return merged


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


@dataclass
class PipelineReport:
    path: Path
    data: dict

    def update(self, section: str, payload: dict) -> None:
        self.data[section] = payload
        self.write()

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2))
        write_report_md(self.path.with_suffix(".md"), self.data)


def load_report(base_dir: Path) -> PipelineReport:
    json_path = base_dir / "report.json"
    if json_path.exists():
        data = json.loads(json_path.read_text())
    else:
        data = {}
    return PipelineReport(path=json_path, data=data)


def write_report_md(path: Path, data: dict) -> None:
    lines = ["# Soccer Highlights Report", ""]
    if not data:
        lines.append("No pipeline runs recorded yet.")
    else:
        for section, payload in data.items():
            lines.append(f"## {section.title()}")
            for key, value in payload.items():
                lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")
            lines.append("")
    path.write_text("\n".join(lines))


def summary_stats(windows: Sequence[HighlightWindow]) -> dict:
    if not windows:
        return {"count": 0, "mean_duration": 0.0, "median_duration": 0.0}
    durations = [w.duration() for w in windows]
    return {
        "count": len(windows),
        "mean_duration": round(mean(durations), 3),
        "median_duration": round(median(durations), 3),
    }


def trim_to_duration(start: float, end: float, pre: float, post: float, bounds: float) -> tuple[float, float]:
    center = (start + end) / 2.0
    new_start = clamp(center - pre, 0.0, bounds)
    new_end = clamp(center + post, 0.0, bounds)
    if new_end <= new_start:
        new_end = clamp(new_start + 0.1, 0.0, bounds)
    return new_start, new_end


def safe_remove(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError as exc:
        logger.warning("Failed to remove %s: %s", path, exc)

"""Refine event windows to produce concise highlight clips."""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

if __package__ is None or __package__ == "":  # pragma: no cover - CLI execution
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.append(str(_Path(__file__).resolve().parent.parent))

from tools.ball_in_play import estimate_in_play
from tools.utils import snap_to_rising_edge


@dataclass
class ClipRule:
    pre: float
    post: float
    anchor: str  # 'start', 'mid', 'end'


DEFAULT_RULES: Dict[str, ClipRule] = {
    "GOAL": ClipRule(10.0, 7.0, "end"),
    "SHOT": ClipRule(6.0, 5.0, "end"),
    "CROSS": ClipRule(6.0, 5.0, "end"),
    "SAVE": ClipRule(5.0, 4.0, "end"),
    "GK": ClipRule(5.0, 4.0, "end"),
    "BUILD": ClipRule(4.0, 4.0, "mid"),
    "OFFENSE": ClipRule(4.0, 4.0, "mid"),
    "ATTACK": ClipRule(4.0, 4.0, "mid"),
    "PASS": ClipRule(4.0, 4.0, "mid"),
    "COMBINE": ClipRule(4.0, 4.0, "mid"),
    "DEFENSE": ClipRule(4.0, 4.0, "mid"),
    "TACKLE": ClipRule(4.0, 4.0, "mid"),
    "INTERCEPT": ClipRule(4.0, 4.0, "mid"),
    "BLOCK": ClipRule(4.0, 4.0, "mid"),
    "CLEAR": ClipRule(4.0, 4.0, "mid"),
}


CATEGORY_MAP = {
    "GOAL": "GOAL",
    "FORCED_GOALS": "GOAL",
    "SHOT": "SHOT",
    "SHOTS": "SHOT",
    "CROSS": "SHOT",
    "SAVE": "SAVE",
    "KEEPER": "SAVE",
    "BUILD": "BUILD",
    "OFFENSE": "OFFENSE",
    "ATTACK": "OFFENSE",
    "PASS": "OFFENSE",
    "COMBINE": "OFFENSE",
    "DEFENSE": "DEFENSE",
    "TACKLE": "DEFENSE",
    "INTERCEPT": "DEFENSE",
    "BLOCK": "DEFENSE",
    "CLEAR": "DEFENSE",
}


def _classify(label: str) -> ClipRule:
    upper = label.upper()
    for key, category in CATEGORY_MAP.items():
        if key in upper:
            rule = DEFAULT_RULES.get(category, ClipRule(4.0, 4.0, "mid"))
            return rule
    return ClipRule(4.0, 4.0, "mid")


def _probe_duration(video: Path) -> float:
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    if frame_count <= 0:
        return 0.0
    return float(frame_count / fps)


def _merge_overlaps(df: pd.DataFrame, merge_gap: float) -> pd.DataFrame:
    if df.empty:
        return df
    sorted_df = df.sort_values("t0").reset_index(drop=True)
    merged_rows: List[dict] = []
    current = sorted_df.iloc[0].to_dict()
    current_sources = set(filter(None, str(current.get("src", "")).split("+")))
    current_labels = set(filter(None, str(current.get("label", "")).split("+")))

    for row in sorted_df.iloc[1:].itertuples(index=False):
        gap = float(getattr(row, "t0")) - float(current["t1"])
        if gap <= merge_gap:
            current["t1"] = max(current["t1"], float(getattr(row, "t1")))
            current["t0"] = min(current["t0"], float(getattr(row, "t0")))
            current["score"] = max(current.get("score", 0.0), float(getattr(row, "score", 0.0)))
            current_labels.update(filter(None, str(getattr(row, "label", "")).split("+")))
            current_sources.update(filter(None, str(getattr(row, "src", "")).split("+")))
        else:
            current["label"] = "+".join(sorted(current_labels))
            current["src"] = "+".join(sorted(current_sources))
            merged_rows.append(current)
            current = row._asdict()
            current_sources = set(filter(None, str(current.get("src", "")).split("+")))
            current_labels = set(filter(None, str(current.get("label", "")).split("+")))

    current["label"] = "+".join(sorted(current_labels))
    current["src"] = "+".join(sorted(current_sources))
    merged_rows.append(current)
    return pd.DataFrame(merged_rows)


def _ensure_length(t0: float, t1: float, min_len: float, max_len: float, anchor: float, duration: float) -> tuple[float, float]:
    length = t1 - t0
    if length <= 0:
        length = min_len
        t0 = max(0.0, anchor - length / 2)
        t1 = t0 + length
    if length < min_len:
        needed = min_len - length
        extend_before = min(needed / 2, t0)
        t0 -= extend_before
        t1 += needed - extend_before
    elif length > max_len:
        t0 = max(anchor - max_len / 2, 0.0)
        t1 = t0 + max_len
    if t1 > duration:
        overshoot = t1 - duration
        t1 = duration
        t0 = max(0.0, t0 - overshoot)
    if t1 - t0 < min_len:
        t0 = max(0.0, t1 - min_len)
    return t0, t1


def tighten_events(
    events_path: Path,
    video_path: Path,
    min_len: float,
    max_len: float,
    merge_gap: float,
    hop: float = 0.10,
) -> pd.DataFrame:
    events = pd.read_csv(events_path)
    if events.empty:
        return events
    in_play_df = estimate_in_play(video_path, hop=hop)
    duration = _probe_duration(video_path)

    in_play_series = in_play_df["in_play"].astype(bool)
    rms_series = in_play_df["audio_rms"]

    refined: List[dict] = []
    for row in events.itertuples(index=False):
        rule = _classify(getattr(row, "label", ""))
        anchor_source = rule.anchor
        anchor_time = getattr(row, "t1") if anchor_source == "end" else getattr(row, "t0")
        if anchor_source == "mid":
            anchor_time = (getattr(row, "t0") + getattr(row, "t1")) / 2.0
        start = anchor_time - rule.pre
        end = anchor_time + rule.post
        start = min(start, getattr(row, "t0"))
        end = max(end, getattr(row, "t1"))
        start = max(0.0, start)
        end = max(end, start + 0.01)

        snapped = snap_to_rising_edge(in_play_series, start, lookback=4.0)
        if snapped < start:
            end += start - snapped
            start = snapped

        window = rms_series.loc[(rms_series.index >= start) & (rms_series.index <= end)]
        if not window.empty:
            peak_idx = window.idxmax()
            if end - peak_idx < 1.0:
                shift = min(rule.pre, 1.0 - (end - peak_idx), start)
                if shift > 0:
                    start -= shift
                    end -= shift

        start, end = _ensure_length(start, end, min_len, max_len, anchor_time, duration)
        refined.append(
            {
                "t0": max(0.0, start),
                "t1": min(duration, end),
                "label": getattr(row, "label"),
                "score": getattr(row, "score", 0.0),
                "src": getattr(row, "src", ""),
            }
        )

    df = pd.DataFrame(refined).sort_values("t0").reset_index(drop=True)
    merged = _merge_overlaps(df, merge_gap)
    return merged.sort_values("t0").reset_index(drop=True)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tighten highlight windows")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pre-goal", type=float, default=10.0)
    parser.add_argument("--post-goal", type=float, default=7.0)
    parser.add_argument("--pre-shot", type=float, default=6.0)
    parser.add_argument("--post-shot", type=float, default=5.0)
    parser.add_argument("--pre-save", type=float, default=5.0)
    parser.add_argument("--post-save", type=float, default=4.0)
    parser.add_argument("--pre-build", type=float, default=4.0)
    parser.add_argument("--post-build", type=float, default=4.0)
    parser.add_argument("--pre-defense", type=float, default=4.0)
    parser.add_argument("--post-defense", type=float, default=4.0)
    parser.add_argument("--merge-gap", type=float, default=0.30)
    parser.add_argument("--min-len", type=float, default=3.0)
    parser.add_argument("--max-len", type=float, default=14.0)
    parser.add_argument("--hop", type=float, default=0.10)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    DEFAULT_RULES["GOAL"] = ClipRule(args.pre_goal, args.post_goal, "end")
    DEFAULT_RULES["SHOT"] = ClipRule(args.pre_shot, args.post_shot, "end")
    DEFAULT_RULES["CROSS"] = ClipRule(args.pre_shot, args.post_shot, "end")
    DEFAULT_RULES["SAVE"] = ClipRule(args.pre_save, args.post_save, "end")
    DEFAULT_RULES["GK"] = ClipRule(args.pre_save, args.post_save, "end")
    for key in ["BUILD", "OFFENSE", "ATTACK", "PASS", "COMBINE"]:
        DEFAULT_RULES[key] = ClipRule(args.pre_build, args.post_build, "mid")
    for key in ["DEFENSE", "TACKLE", "INTERCEPT", "BLOCK", "CLEAR"]:
        DEFAULT_RULES[key] = ClipRule(args.pre_defense, args.post_defense, "mid")

    tightened = tighten_events(
        args.events,
        args.video,
        min_len=args.min_len,
        max_len=args.max_len,
        merge_gap=args.merge_gap,
        hop=args.hop,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tightened.to_csv(args.out, index=False, quoting=csv.QUOTE_MINIMAL)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

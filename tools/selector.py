"""Event selection logic for Smart Soccer Highlight Selector."""
from __future__ import annotations

import dataclasses
import logging
logger = logging.getLogger(__name__)
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .ffconcat import Clip


LABEL_PRIORITY = {"GOAL": 5, "WOODWORK": 4, "SHOT": 3, "SAVE": 2, "BUILDUP": 1}
TYPE_WEIGHT = {"GOAL": 100.0, "WOODWORK": 90.0, "SHOT": 80.0, "SAVE": 70.0, "BUILDUP": 40.0}


@dataclass
class SelectorConfig:
    audio_peak_threshold: float = 0.60
    motion_peak_threshold: float = 0.60
    min_clip: float = 3.0
    max_clip: float = 16.0
    merge_gap: float = 0.8
    in_play_min_ratio: float = 0.9
    goal_pre_range: Tuple[float, float] = (8.0, 14.0)
    goal_post_range: Tuple[float, float] = (6.0, 9.0)
    goal_lookback: float = 80.0
    goal_ahead: float = 1.0
    shot_pre_range: Tuple[float, float] = (4.0, 8.0)
    shot_post_range: Tuple[float, float] = (4.0, 6.0)
    buildup_pre_range: Tuple[float, float] = (3.0, 5.0)
    buildup_post_range: Tuple[float, float] = (3.0, 5.0)
    save_pre_range: Tuple[float, float] = (4.0, 6.0)
    save_post_range: Tuple[float, float] = (4.0, 6.0)
    0.60
    0.60
    combined_peak_percentile: float = 85.0


@dataclasses.dataclass
class Candidate:
    time: float
    audio_z: float
    motion_z: float
    centroid_z: float
    crowd_z: float
    combined: float
    src: str = "detected"
    label_hint: Optional[str] = None


@dataclasses.dataclass
class EventSpan:
    label: str
    start: float
    end: float
    score: float
    src: str
    notes: str
    anchor: float

    def to_clip(self) -> Clip:
        return Clip(self.start, self.end, self.label, self.score)

    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def _robust_z(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros_like(values)
    median = np.median(values)
    mad = np.median(np.abs(values - median)) + 1e-6
    return (values - median) / (1.4826 * mad)


def _interp(base_time: np.ndarray, source: pd.DataFrame, column: str) -> np.ndarray:
    """Interpolate source[column] onto base_time (both in seconds)."""
    # If a list/tuple slipped in, take the first name
    if isinstance(column, (list, tuple)):
        column = column[0]

    s = source[column]
    # If duplicate names yielded a DataFrame, take the first column
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]

    x = pd.to_numeric(source["time"], errors="coerce").to_numpy()
    y = pd.to_numeric(s, errors="coerce").to_numpy()

    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        # Not enough points to interpolate – return NaNs
        return np.full_like(base_time, np.nan, dtype=float)

    # Let np.interp handle endpoints (no custom left/right scalars needed)
    return np.interp(base_time, x[m], y[m])


def _combine_features(audio_df: pd.DataFrame, motion_df: pd.DataFrame) -> pd.DataFrame:
    def _time_column(df: pd.DataFrame) -> Optional[str]:
        for name in ("time", "t"):
            if name in df.columns:
                return name
        return None

    audio_time_col = _time_column(audio_df) if not audio_df.empty else None
    if audio_time_col is not None:
        time_values = pd.to_numeric(audio_df[audio_time_col], errors="coerce").to_numpy()
    else:
        motion_time_col = _time_column(motion_df) if not motion_df.empty else None
        if motion_time_col is None:
            return pd.DataFrame(columns=["time"])
        time_values = pd.to_numeric(motion_df[motion_time_col], errors="coerce").to_numpy()

    base = pd.DataFrame({"time": time_values})
    base_time = base["time"].to_numpy()

    if audio_time_col is not None:
        for col in audio_df.columns:
            if col == audio_time_col:
                continue
            base[col] = pd.to_numeric(audio_df[col], errors="coerce")

    for column in ["rms", "rms_smooth", "centroid", "centroid_smooth", "zcr", "crowd_score", "whistle_score"]:
        if column not in base.columns:
            base[column] = 0.0

    if not motion_df.empty:
        motion_time_col = _time_column(motion_df)
        if motion_time_col is not None:
            if motion_time_col != "time":
                motion_source = motion_df.rename(columns={motion_time_col: "time"})
            else:
                motion_source = motion_df
            for col in motion_source.columns:
                if col == "time":
                    continue
                base[col] = _interp(base_time, motion_source, col)

    for column in ["motion", "motion_smooth", "pan_score"]:
        if column not in base.columns:
            base[column] = 0.0

    base["audio_z"] = _robust_z(base.get("rms_smooth", base.get("rms", pd.Series(0))).to_numpy())
    base["motion_z"] = _robust_z(base.get("motion_smooth", pd.Series(0)).to_numpy())
    base["centroid_z"] = _robust_z(base.get("centroid_smooth", base.get("centroid", pd.Series(0))).to_numpy())
    base["crowd_z"] = _robust_z(base.get("crowd_score", pd.Series(0)).to_numpy())
    combined = 0.6 * base["audio_z"] + 0.4 * base["motion_z"] + 0.2 * np.clip(base["crowd_z"], 0, None)
    base["combined"] = combined
    return base


def _find_local_maxima(series: np.ndarray, times: np.ndarray, threshold: float, min_spacing: float) -> List[int]:
    if series.size < 3:
        return []
    indices: List[int] = []
    last_time = -np.inf
    for idx in range(1, len(series) - 1):
        if series[idx] < threshold:
            continue
        if times[idx] - last_time < min_spacing:
            continue
        if series[idx] > series[idx - 1] and series[idx] >= series[idx + 1]:
            indices.append(idx)
            last_time = times[idx]
    return indices


def _range_from_strength(strength: float, base_range: Tuple[float, float]) -> float:
    low, high = base_range
    strength = np.clip(strength, 0.0, 5.0)
    return float(low + (high - low) * (strength / 5.0))


class SmartSelector:
    def __post_init__(self):
        logger.info(
            "Selector (instance) thresholds: audio_z>=%.2f motion_z>=%.2f combined>=%.3f",
            getattr(self, 'audio_z_thr', self.config.audio_z_thr),
            getattr(self, 'motion_z_thr', self.config.motion_z_thr),
            getattr(self, 'combined_thr', float('nan'))
        )
    def __init__(
        self,
        audio_df: pd.DataFrame,
        motion_df: pd.DataFrame,
        in_play_spans: Sequence[Tuple[float, float]],
        config: SelectorConfig,
        prior_events: Optional[pd.DataFrame] = None,
        goal_resets: Optional[Sequence[float]] = None,
        forced_goal_marks: Optional[pd.DataFrame] = None,
    ) -> None:
        self.timeline = _combine_features(audio_df, motion_df)
        self.config = config
        self.in_play_spans = list(sorted(in_play_spans))
        self.prior_events = prior_events if prior_events is not None else pd.DataFrame()
        self.goal_resets = list(goal_resets or [])
        self.forced_goal_marks = forced_goal_marks if forced_goal_marks is not None else pd.DataFrame()
        self.logger = logging.getLogger("smart_selector")
        self._last_combined_threshold: Optional[float] = None

    # ------------------------------------------------------------------
    def _candidate_peaks(self) -> List[Candidate]:
        if self.timeline.empty:
            return []

        threshold = np.percentile(self.timeline["combined"], self.config.combined_peak_percentile)
        times = self.timeline["time"].to_numpy()
        indices = _find_local_maxima(
            self.timeline["combined"].to_numpy(),
            times,
            max(threshold, self.config.audio_peak_threshold),
            min_spacing=4.0,
        )
        self._last_combined_threshold = threshold
        candidates: List[Candidate] = []
        for idx in indices:
            candidates.append(
                Candidate(
                    time=float(times[idx]),
                    audio_z=float(self.timeline.at[idx, "audio_z"]),
                    motion_z=float(self.timeline.at[idx, "motion_z"]),
                    centroid_z=float(self.timeline.at[idx, "centroid_z"]),
                    crowd_z=float(self.timeline.at[idx, "crowd_z"]),
                    combined=float(self.timeline.at[idx, "combined"]),
                    src="detected",
                    label_hint=self._hint_from_priors(times[idx]),
                )
            )
        return candidates

    def _hint_from_priors(self, time_point: float) -> Optional[str]:
        if self.prior_events.empty:
            return None
        window = 3.0
        close = self.prior_events[
            (self.prior_events["t0"] <= time_point + window)
            & (self.prior_events["t1"] >= time_point - window)
        ]
        if close.empty:
            return None
        labels = close["label"].unique()
        for label in ["GOAL", "WOODWORK", "SHOT", "SAVE"]:
            if label in labels:
                return label
        return close["label"].iloc[0]

    # ------------------------------------------------------------------
    def _forced_goal_windows(self) -> List[Tuple[float, float, str]]:
        windows: List[Tuple[float, float, str]] = []
        for reset in self.goal_resets:
            start = max(0.0, float(reset) - self.config.goal_lookback)
            end = max(0.0, float(reset) - self.config.goal_ahead)
            windows.append((start, end, f"reset@{reset:.1f}"))
        if not self.forced_goal_marks.empty:
            for row in self.forced_goal_marks.itertuples(index=False):
                if hasattr(row, "t"):
                    center = float(row.t)
                elif hasattr(row, "time"):
                    center = float(row.time)
                elif hasattr(row, "start"):
                    center = float(getattr(row, "start"))
                else:
                    continue
                windows.append((max(0.0, center - 20.0), center + 2.0, "forced"))
        return windows

    def _pick_candidate_in_window(self, candidates: List[Candidate], window: Tuple[float, float, str]) -> Optional[Candidate]:
        start, end, _ = window
        pool = [c for c in candidates if start <= c.time <= end]
        if not pool:
            return None
        pool.sort(key=lambda c: (c.combined, c.audio_z, c.motion_z), reverse=True)
        return pool[0]

    def _score(self, label: str, cand: Candidate, bonus: float = 0.0) -> float:
        weight = TYPE_WEIGHT.get(label, 10.0)
        return weight + 5.0 * max(0.0, cand.audio_z) + 5.0 * max(0.0, cand.motion_z) + 2.0 * max(0.0, cand.crowd_z) + bonus

    def _shape_window(self, label: str, cand: Candidate) -> Tuple[float, float]:
        if label == "GOAL":
            pre = _range_from_strength(max(cand.audio_z, cand.motion_z), self.config.goal_pre_range)
            post = _range_from_strength(max(cand.audio_z, cand.motion_z), self.config.goal_post_range)
        elif label == "WOODWORK":
            pre = _range_from_strength(cand.audio_z, self.config.shot_pre_range)
            post = _range_from_strength(cand.audio_z, self.config.shot_post_range)
        elif label == "SHOT":
            pre = _range_from_strength(cand.motion_z, self.config.shot_pre_range)
            post = _range_from_strength(cand.motion_z, self.config.shot_post_range)
        elif label == "SAVE":
            pre = _range_from_strength(cand.motion_z, self.config.save_pre_range)
            post = _range_from_strength(cand.motion_z, self.config.save_post_range)
        else:
            pre = _range_from_strength(cand.combined, self.config.buildup_pre_range)
            post = _range_from_strength(cand.combined, self.config.buildup_post_range)
        pre = np.clip(pre, 0.5, self.config.max_clip)
        post = np.clip(post, 0.5, self.config.max_clip)
        return cand.time - pre, cand.time + post

    def _classify_candidate(self, cand: Candidate) -> str:
        if cand.label_hint:
            return cand.label_hint
        if cand.audio_z > 2.5 and cand.centroid_z > 1.8:
            return "WOODWORK"
        if cand.audio_z > 2.3 and cand.motion_z > 1.7:
            return "SHOT"
        if cand.motion_z > 2.5 and cand.audio_z > 1.0:
            return "SAVE"
        if cand.audio_z > 1.8 or cand.motion_z > 1.5:
            return "BUILDUP"
        return "BUILDUP"

    def _deduplicate(self, spans: List[EventSpan]) -> List[EventSpan]:
        spans = sorted(spans, key=lambda s: s.start)
        pruned: List[EventSpan] = []
        for span in spans:
            keep = True
            for other in pruned:
                inter = max(0.0, min(span.end, other.end) - max(span.start, other.start))
                if inter <= 0:
                    continue
                ratio = inter / min(span.duration(), other.duration(), 1e9)
                if ratio >= 0.9:
                    if span.score <= other.score:
                        keep = False
                        break
                    else:
                        pruned.remove(other)
                        break
            if keep:
                pruned.append(span)
        return pruned

    def _merge_spans(self, spans: List[EventSpan]) -> List[EventSpan]:
        if not spans:
            return []
        spans.sort(key=lambda s: s.start)
        merged: List[EventSpan] = [spans[0]]
        for span in spans[1:]:
            current = merged[-1]
            if span.start - current.end <= self.config.merge_gap:
                new_label = current.label if LABEL_PRIORITY[current.label] >= LABEL_PRIORITY[span.label] else span.label
                new_score = max(current.score, span.score)
                merged[-1] = EventSpan(
                    label=new_label,
                    start=min(current.start, span.start),
                    end=max(current.end, span.end),
                    score=new_score,
                    src=f"{current.src}+{span.src}",
                    notes=f"{current.notes};{span.notes}".strip(";"),
                    anchor=current.anchor,
                )
            else:
                merged.append(span)
        return merged

    def _clamp_span(self, span: EventSpan) -> Tuple[Optional[EventSpan], float]:
        if not self.in_play_spans:
            return span, 1.0
        overlaps: List[Tuple[float, float]] = []
        for s, e in self.in_play_spans:
            t0 = max(span.start, s)
            t1 = min(span.end, e)
            if t1 - t0 > 0.1:
                overlaps.append((t0, t1))
        if not overlaps:
            ratio = 0.0
            if span.label == "GOAL":
                return span, ratio
            return None, ratio

        total_overlap = sum(e - s for s, e in overlaps)
        ratio = total_overlap / max(span.duration(), 1e-6)

        if span.label != "GOAL" and ratio < self.config.in_play_min_ratio:
            return None, ratio

        if span.label == "GOAL":
            best = None
            for s, e in overlaps:
                if s <= span.anchor <= e:
                    best = (s, e)
                    break
            if best is None:
                best = (overlaps[0][0], overlaps[-1][1])
            span.start = max(span.start, best[0])
            span.end = min(span.end, best[1])
            if span.duration() < self.config.min_clip:
                span.end = min(span.end + (self.config.min_clip - span.duration()), best[1])
            return span, ratio

        # Non-goal span: clamp to bounds of largest overlap
        best = max(overlaps, key=lambda x: x[1] - x[0])
        span.start = max(span.start, best[0])
        span.end = min(span.end, best[1])
        if span.duration() < self.config.min_clip:
            # Try to expand within original limits
            needed = self.config.min_clip - span.duration()
            span.start = max(span.start - needed / 2, span.anchor - self.config.min_clip / 2)
            span.end = span.start + self.config.min_clip
        return span, ratio

    def _in_play_ratio(self, span: EventSpan) -> float:
        if not self.in_play_spans:
            return 1.0
        covered = 0.0
        for s, e in self.in_play_spans:
            covered += max(0.0, min(span.end, e) - max(span.start, s))
        return covered / max(span.duration(), 1e-6)

    # ------------------------------------------------------------------
    def run(self) -> Tuple[pd.DataFrame, List[EventSpan], Dict[int, float]]:
        candidates = self._candidate_peaks()
        spans: List[EventSpan] = []
        used_candidates: set = set()

        for window in self._forced_goal_windows():
            cand = self._pick_candidate_in_window(candidates, window)
            notes = window[2]
            if cand is None:
                cand = Candidate(
                    time=float((window[0] + window[1]) / 2.0),
                    audio_z=0.0,
                    motion_z=0.0,
                    centroid_z=0.0,
                    crowd_z=0.0,
                    combined=0.0,
                    src="forced",
                    label_hint="GOAL",
                )
                notes += ":synthetic"
            else:
                used_candidates.add(cand.time)
            start, end = self._shape_window("GOAL", cand)
            start = max(0.0, start)
            end = min(start + self.config.max_clip, end)
            score = self._score("GOAL", cand, bonus=8.0)
            span = EventSpan("GOAL", start, end, score, src="forced", notes=notes, anchor=cand.time)
            clamped, ratio = self._clamp_span(span)
            if clamped is not None:
                spans.append(clamped)

        for cand in candidates:
            if cand.time in used_candidates:
                continue
            label = self._classify_candidate(cand)
            start, end = self._shape_window(label, cand)
            start = max(0.0, start)
            if end - start > self.config.max_clip:
                end = start + self.config.max_clip
            score = self._score(label, cand)
            span = EventSpan(label, start, end, score, src=cand.src, notes="", anchor=cand.time)
            clamped, ratio = self._clamp_span(span)
            if clamped is not None:
                spans.append(clamped)

        spans = self._merge_spans(spans)
        spans = self._deduplicate(spans)

        filtered_spans: List[EventSpan] = []
        ratios_reindexed: Dict[int, float] = {}
        for idx, span in enumerate(sorted(spans, key=lambda s: s.start)):
            # enforce min duration
            if span.duration() < self.config.min_clip:
                span.end = span.start + self.config.min_clip
            if span.duration() > self.config.max_clip:
                span.end = span.start + self.config.max_clip
            filtered_spans.append(span)
            ratios_reindexed[idx] = self._in_play_ratio(span)

        rows = [
            {
                "label": span.label,
                "start": round(span.start, 3),
                "end": round(span.end, 3),
                "score": round(span.score, 3),
                "src": span.src,
                "notes": span.notes,
            }
            for span in filtered_spans
        ]
        df = pd.DataFrame(rows, columns=["label", "start", "end", "score", "src", "notes"])
        return df.sort_values("start").reset_index(drop=True), filtered_spans, ratios_reindexed

    @property
    def last_combined_threshold(self) -> Optional[float]:
        return self._last_combined_threshold


if __name__ == "__main__":  # pragma: no cover - smoke test
    times = np.linspace(0, 120, 240)
    audio = pd.DataFrame({
        "time": times,
        "rms": np.sin(times / 5.0) ** 2 + 0.2,
        "rms_smooth": np.sin(times / 5.0) ** 2 + 0.2,
        "centroid": np.cos(times / 7.0) ** 2,
        "centroid_smooth": np.cos(times / 7.0) ** 2,
        "zcr": np.zeros_like(times),
        "crowd_score": np.sin(times / 5.0) ** 2,
        "whistle_score": np.zeros_like(times),
    })
    motion = pd.DataFrame({
        "time": times,
        "motion": np.cos(times / 6.0) ** 2,
        "motion_smooth": np.cos(times / 6.0) ** 2,
        "pan_score": np.zeros_like(times),
    })
    selector = SmartSelector(audio, motion, [(0.0, 120.0)], SelectorConfig(), goal_resets=[60.0])
    result, spans, ratios = selector.run()
    print(result.head())









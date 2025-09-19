#!/usr/bin/env python3
"""Fuse detector CSVs and heuristics into a highlight event stream.

This tool reads the existing detector outputs (goal resets, forced goals,
shot/action CSVs) and optionally inspects the source video/audio to produce a
consistent highlight event timeline.  The output is a pair of CSVs plus a JSON
summary that downstream tooling can rely on:

* ``events_raw.csv`` contains every candidate event, including stoppages.
* ``events_selected.csv`` is the pruned/merged stream that should feed the reel.
* ``review_summary.json`` captures recall/coverage metrics for quick QA.

The script aims to be deterministic and debuggable.  All important heuristics
can be tuned via CLI flags and the console summary makes it easy to track
recall.  We also expose ``--never-drop`` for high-priority classes so that caps
never remove goals or shots.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

try:  # NumPy is a hard requirement
    import numpy as np  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"NumPy is required: {exc}")

try:  # OpenCV is optional; motion heuristics degrade gracefully without it
    import cv2  # type: ignore
    CV2_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    cv2 = None  # type: ignore
    CV2_AVAILABLE = False
    print(f"[select_events] Warning: OpenCV unavailable ({exc}); motion heuristics disabled.")


###############################################################################
# Utility helpers
###############################################################################

def parse_float(val: object) -> Optional[float]:
    """Parse a loose CSV field into ``float``.

    We accept comma decimal separators and strip whitespace.  Returns ``None``
    when parsing fails instead of raising.
    """

    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if not s:
        return None
    s = s.replace("\ufeff", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        rows = [dict(row) for row in reader]
    return rows


def infer_team(team_presence: Optional[float]) -> str:
    """Infer team ownership from a heuristic ``team_presence`` metric.

    The detector CSVs expose a normalized presence score (roughly ``0-0.2`` in
    the provided data).  Values near zero likely indicate the opponent controlling
    the ball, while higher numbers lean towards our team.  We use a conservative
    split so that ambiguous segments stay ``unknown``.
    """

    if team_presence is None:
        return "unknown"
    if team_presence >= 0.075:
        return "us"
    if team_presence <= 0.025:
        return "opp"
    return "unknown"


###############################################################################
# Video/audio analysis
###############################################################################

@dataclass
class MotionSeries:
    times: List[float]
    global_motion: List[float]
    pan_speed: List[float]
    left_motion: List[float]
    right_motion: List[float]
    center_motion: List[float]
    goal_left: List[float]
    goal_right: List[float]

    def percentile(self, values: Sequence[float], pct: float) -> float:
        if not values:
            return 0.0
        arr = np.asarray(values, dtype=np.float32)
        return float(np.percentile(arr, pct))


def compute_motion_series(
    video: Path,
    sample_hz: float = 6.0,
    resize_width: int = 640,
    max_frames: Optional[int] = None,
) -> MotionSeries:
    """Sample the video and compute lightweight motion descriptors.

    We downsample the frame-rate aggressively (default ~6 fps) so that a full
    match remains tractable even on modest hardware.  Each sampled frame
    contributes mean absolute difference ("global motion"), camera pan speed,
    and mean motion energy for each third of the pitch and the goal corridors.
    """

    if not CV2_AVAILABLE:
        return MotionSeries(
            times=[],
            global_motion=[],
            pan_speed=[],
            left_motion=[],
            right_motion=[],
            center_motion=[],
            goal_left=[],
            goal_right=[],
        )

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps / max(sample_hz, 0.1))))

    times: List[float] = []
    global_motion: List[float] = []
    pan_speed: List[float] = []
    left_motion: List[float] = []
    right_motion: List[float] = []
    center_motion: List[float] = []
    goal_left: List[float] = []
    goal_right: List[float] = []

    prev_gray: Optional[np.ndarray] = None
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step != 0:
            frame_idx += 1
            continue
        h, w = frame.shape[:2]
        if resize_width and w > resize_width:
            scale = resize_width / float(w)
            frame = cv2.resize(
                frame,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
            h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            prev_gray = gray
            frame_idx += 1
            continue

        diff = cv2.absdiff(gray, prev_gray)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        prev_gray = gray

        t = (frame_idx) / fps
        times.append(t)
        global_motion.append(float(np.mean(diff)))
        pan_speed.append(float(np.median(np.abs(flow[..., 0]))))

        thirds = np.array_split(diff, 3, axis=1)
        left_motion.append(float(np.mean(thirds[0])))
        center_motion.append(float(np.mean(thirds[1])))
        right_motion.append(float(np.mean(thirds[2])))

        goal_band = int(round(0.15 * w))
        goal_left.append(float(np.mean(diff[:, :goal_band])))
        goal_right.append(float(np.mean(diff[:, -goal_band:])))

        frame_idx += 1
        if max_frames and len(times) >= max_frames:
            break

    cap.release()
    return MotionSeries(
        times=times,
        global_motion=global_motion,
        pan_speed=pan_speed,
        left_motion=left_motion,
        right_motion=right_motion,
        center_motion=center_motion,
        goal_left=goal_left,
        goal_right=goal_right,
    )


def load_audio_envelope(
    video: Path,
    hop_seconds: float = 0.25,
    target_sr: int = 8000,
) -> Tuple[List[float], List[float]]:
    """Return a coarse RMS envelope for the video's mono audio track."""

    try:
        import librosa  # type: ignore

        y, sr = librosa.load(str(video), sr=target_sr, mono=True)
        hop_len = max(1, int(sr * hop_seconds))
        if hop_len <= 1:
            hop_len = int(sr * 0.25)
        env = librosa.feature.rms(y=y, frame_length=hop_len * 2, hop_length=hop_len)[0]
        times = librosa.frames_to_time(
            np.arange(len(env)), sr=sr, hop_length=hop_len
        )
        env = env.astype(np.float32)
        if env.size:
            env = (env - env.min()) / (env.max() - env.min() + 1e-8)
        return times.tolist(), env.tolist()
    except Exception:
        pass

    # Fallback: decode via ffmpeg pipe to mono float32
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(target_sr),
        "-f",
        "f32le",
        "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=False)
    except FileNotFoundError:
        return [], []
    if proc.returncode != 0 or not proc.stdout:
        return [], []
    data = np.frombuffer(proc.stdout, dtype="<f4")
    if data.size == 0:
        return [], []
    hop = max(1, int(target_sr * hop_seconds))
    window = hop * 2
    env = []
    times = []
    for idx in range(0, len(data) - window, hop):
        segment = data[idx : idx + window]
        env.append(float(np.sqrt(np.mean(segment ** 2))))
        times.append(idx / target_sr)
    if env:
        arr = np.asarray(env, dtype=np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        env = arr.tolist()
    return times, env


###############################################################################
# Event modelling
###############################################################################


def probe_video_duration(video: Path) -> float:
    """Use ffprobe to fetch video duration when OpenCV is unavailable."""

    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=nk=1:nw=1",
                str(video),
            ],
            stderr=subprocess.STDOUT,
        )
        return float(out.decode("utf-8").strip())
    except Exception:
        return 0.0


def estimate_max_timestamp(paths: Sequence[Path]) -> float:
    """Best-effort upper bound on event timestamps from CSV sources."""

    max_t = 0.0
    for path in paths:
        for row in load_csv(path):
            for key in ("start", "end", "t0", "t1"):
                val = parse_float(row.get(key))
                if val is not None:
                    max_t = max(max_t, float(val))
    return max_t

TYPE_PRIORITY: Dict[str, int] = {
    "GOAL": 100,
    "SHOT": 80,
    "SAVE_OUR_GK": 70,
    "OFF_BUILDUP": 55,
    "DEF_ACTION": 55,
    "STOPPAGE": -100,
}


@dataclass
class Event:
    event_id: str
    t0: float
    t1: float
    type: str
    source: str
    score: float
    team: str
    reason: str
    must_keep: bool = False

    def as_row(self) -> Dict[str, object]:
        return {
            "start": round(self.t0, 3),
            "end": round(self.t1, 3),
            "type": self.type,
            "source": self.source,
            "score": round(self.score, 3),
            "team": self.team,
            "reason": self.reason,
            "event_id": self.event_id,
        }


@dataclass
class Span:
    t0: float
    t1: float
    score: float
    reasons: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    teams: List[str] = field(default_factory=list)
    types: Counter = field(default_factory=Counter)
    event_ids: Set[str] = field(default_factory=set)
    must_keep: bool = False

    def add_event(self, event: Event) -> None:
        self.score = max(self.score, event.score)
        if event.reason:
            self.reasons.append(event.reason)
        if event.source:
            self.sources.append(event.source)
        if event.team:
            self.teams.append(event.team)
        self.types[event.type] += 1
        self.event_ids.add(event.event_id)
        self.must_keep = self.must_keep or event.must_keep

    @property
    def primary_type(self) -> str:
        if not self.types:
            return "UNKNOWN"
        # prefer higher priority; break ties by frequency then alphabetical
        best_type = None
        best_key = None
        for t, count in self.types.items():
            pri = TYPE_PRIORITY.get(t, 0)
            key = (pri, count, -abs(TYPE_PRIORITY.get(t, 0)))
            if best_key is None or key > best_key:
                best_type = t
                best_key = key
        return best_type or next(iter(self.types))

    @property
    def all_types(self) -> str:
        return ";".join(sorted(self.types))

    @property
    def team(self) -> str:
        if not self.teams:
            return "unknown"
        uniq = {t for t in self.teams if t and t != "unknown"}
        if len(uniq) == 1:
            return uniq.pop()
        return "mixed" if uniq else "unknown"

    @property
    def reason_text(self) -> str:
        seen = []
        out: List[str] = []
        for r in self.reasons:
            r = r.strip()
            if not r:
                continue
            if r in seen:
                continue
            seen.append(r)
            out.append(r)
        return "; ".join(out)

    @property
    def source_text(self) -> str:
        seen = []
        out: List[str] = []
        for s in self.sources:
            s = s.strip()
            if not s:
                continue
            if s in seen:
                continue
            seen.append(s)
            out.append(s)
        return ";".join(out)

    def to_row(self) -> Dict[str, object]:
        primary = self.primary_type
        base = TYPE_PRIORITY.get(primary, 0)
        score = max(self.score, float(base))
        return {
            "start": round(self.t0, 3),
            "end": round(self.t1, 3),
            "type": primary,
            "score": round(score, 3),
            "team": self.team,
            "source": self.source_text,
            "reason": self.reason_text,
            "event_ids": ",".join(sorted(self.event_ids)),
            "all_types": self.all_types,
        }

    @property
    def length(self) -> float:
        return max(0.0, self.t1 - self.t0)


###############################################################################
# Candidate extraction helpers
###############################################################################

@dataclass
class CollectorContext:
    duration: float
    pre_goal: float
    post_goal: float
    pre_shot: float
    post_shot: float
    pre_save: float = 5.0
    post_save: float = 4.0
    pre_def: float = 3.5
    post_def: float = 3.5
    pre_build: float = 7.0
    post_build: float = 3.0


class EventCollector:
    def __init__(self, never_drop: Set[str]) -> None:
        self._never_drop = {t.upper() for t in never_drop}
        self._counter = count(1)
        self.lookup: Dict[str, Event] = {}
        self.by_type: Dict[str, Set[str]] = defaultdict(set)

    def new_event(
        self,
        t0: float,
        t1: float,
        etype: str,
        source: str,
        score: float,
        team: str,
        reason: str,
    ) -> Event:
        etype = etype.upper()
        event_id = f"{etype.lower()}_{next(self._counter):04d}"
        evt = Event(
            event_id=event_id,
            t0=t0,
            t1=t1,
            type=etype,
            source=source,
            score=score,
            team=team,
            reason=reason,
            must_keep=(etype in self._never_drop),
        )
        self.lookup[event_id] = evt
        self.by_type[etype].add(event_id)
        return evt


def collect_goals(
    ctx: CollectorContext,
    collector: EventCollector,
    goal_csv: Path,
    forced_csv: Path,
) -> List[Event]:
    rows = []
    for src, path in (("goal_resets", goal_csv), ("forced_goals", forced_csv)):
        for row in load_csv(path):
            start = parse_float(row.get("start"))
            end = parse_float(row.get("end"))
            if start is None:
                continue
            score = parse_float(row.get("score")) or 1.0
            rows.append(
                {
                    "anchor": float(start),
                    "start": start,
                    "end": end if end is not None else start,
                    "score": float(score),
                    "source": src,
                    "reason": f"{src} score={score:.3f}",
                }
            )
    if not rows:
        return []
    rows.sort(key=lambda r: r["anchor"])
    merged: List[List[Dict[str, float]]] = []
    for row in rows:
        if not merged or row["anchor"] - merged[-1][-1]["anchor"] > 6.0:
            merged.append([row])
        else:
            merged[-1].append(row)

    events: List[Event] = []
    for cluster in merged:
        best = max(cluster, key=lambda r: r["score"])
        anchor = best["anchor"]
        reason = "; ".join(c["reason"] for c in cluster)
        t0 = clamp(anchor - ctx.pre_goal, 0.0, ctx.duration)
        t1 = clamp(anchor + ctx.post_goal, 0.0, ctx.duration)
        evt = collector.new_event(
            t0,
            t1,
            "GOAL",
            source=best["source"],
            score=TYPE_PRIORITY["GOAL"] + best["score"] * 10.0,
            team="us",
            reason=reason,
        )
        events.append(evt)
    return events


def collect_shots(
    ctx: CollectorContext,
    collector: EventCollector,
    shots_csv: Path,
) -> List[Event]:
    rows = load_csv(shots_csv)
    events: List[Event] = []
    for row in rows:
        start = parse_float(row.get("start"))
        end = parse_float(row.get("end"))
        if start is None or end is None or end <= start:
            continue
        action_score = parse_float(row.get("action_score")) or 0.0
        ball_speed = parse_float(row.get("ball_speed")) or 0.0
        team_presence = parse_float(row.get("team_presence"))
        team = infer_team(team_presence)
        reason_bits = [
            f"action={action_score:.3f}",
            f"ball_speed={ball_speed:.1f}",
        ]
        if team_presence is not None:
            reason_bits.append(f"team_presence={team_presence:.3f}")
        reason = f"shots_csv {' '.join(reason_bits)}"
        t0 = clamp(start - ctx.pre_shot, 0.0, ctx.duration)
        t1 = clamp(end + ctx.post_shot, 0.0, ctx.duration)
        score = TYPE_PRIORITY["SHOT"] + action_score * 8.0 + ball_speed * 0.05
        evt = collector.new_event(
            t0,
            t1,
            "SHOT",
            source="highlights_shots",
            score=score,
            team=team if team != "unknown" else "us",
            reason=reason,
        )
        events.append(evt)
    return events


def collect_offense(
    ctx: CollectorContext,
    collector: EventCollector,
    build_csv: Path,
    limit: int = 30,
) -> List[Event]:
    rows = load_csv(build_csv)
    if not rows:
        return []
    scored: List[Tuple[float, Dict[str, str]]] = []
    for row in rows:
        action_score = parse_float(row.get("action_score"))
        if action_score is None:
            continue
        scored.append((float(action_score), row))
    scored.sort(key=lambda x: x[0], reverse=True)
    events: List[Event] = []
    for action_score, row in scored[:limit]:
        start = parse_float(row.get("start"))
        end = parse_float(row.get("end"))
        if start is None or end is None or end <= start:
            continue
        team_presence = parse_float(row.get("team_presence"))
        team = infer_team(team_presence)
        ball_speed = parse_float(row.get("ball_speed")) or 0.0
        reason = (
            f"build_csv action={action_score:.3f} ball_speed={ball_speed:.1f}"
        )
        t0 = clamp(start - ctx.pre_build, 0.0, ctx.duration)
        t1 = clamp(end + ctx.post_build, 0.0, ctx.duration)
        score = TYPE_PRIORITY["OFF_BUILDUP"] + action_score * 6.0 + ball_speed * 0.03
        events.append(
            collector.new_event(
                t0,
                t1,
                "OFF_BUILDUP",
                source="highlights_build",
                score=score,
                team=team if team != "opp" else "unknown",
                reason=reason,
            )
        )
    return events


def collect_defense_and_saves(
    ctx: CollectorContext,
    collector: EventCollector,
    filtered_csv: Path,
) -> Tuple[List[Event], List[Event]]:
    rows = load_csv(filtered_csv)
    saves: List[Event] = []
    defenses: List[Event] = []
    if not rows:
        return saves, defenses

    # compute distribution stats for adaptive thresholds
    action_scores: List[float] = []
    team_presence_vals: List[float] = []
    ball_speeds: List[float] = []
    for row in rows:
        action = parse_float(row.get("action_score"))
        presence = parse_float(row.get("team_presence"))
        speed = parse_float(row.get("ball_speed"))
        if action is not None:
            action_scores.append(action)
        if presence is not None:
            team_presence_vals.append(presence)
        if speed is not None:
            ball_speeds.append(speed)

    action_thresh = float(np.percentile(action_scores, 75)) if action_scores else 1.5
    presence_low = float(np.percentile(team_presence_vals, 20)) if team_presence_vals else 0.03
    presence_high = float(np.percentile(team_presence_vals, 80)) if team_presence_vals else 0.08
    speed_hi = float(np.percentile(ball_speeds, 75)) if ball_speeds else 120.0

    for row in rows:
        start = parse_float(row.get("start"))
        end = parse_float(row.get("end"))
        if start is None or end is None or end <= start:
            continue
        action_score = parse_float(row.get("action_score")) or 0.0
        team_presence = parse_float(row.get("team_presence"))
        ball_speed = parse_float(row.get("ball_speed")) or 0.0
        mean_flow = parse_float(row.get("mean_flow")) or 0.0
        team = infer_team(team_presence)

        is_save = (
            team_presence is not None
            and team_presence <= presence_low
            and ball_speed >= speed_hi
            and action_score >= action_thresh * 0.92
        )
        is_def = (
            team_presence is not None
            and team_presence <= presence_low * 1.35
            and action_score >= action_thresh * 0.85
        )

        if is_save:
            t0 = clamp(start - ctx.pre_save, 0.0, ctx.duration)
            t1 = clamp(end + ctx.post_save, 0.0, ctx.duration)
            reason = (
                f"filtered save action={action_score:.3f} speed={ball_speed:.1f}"
            )
            saves.append(
                collector.new_event(
                    t0,
                    t1,
                    "SAVE_OUR_GK",
                    source="highlights_filtered",
                    score=TYPE_PRIORITY["SAVE_OUR_GK"]
                    + action_score * 6.5
                    + ball_speed * 0.04
                    + mean_flow * 2.0,
                    team="us",
                    reason=reason,
                )
            )
            continue

        if is_def:
            t0 = clamp(start - ctx.pre_def, 0.0, ctx.duration)
            t1 = clamp(end + ctx.post_def, 0.0, ctx.duration)
            reason = (
                f"filtered def action={action_score:.3f} speed={ball_speed:.1f}"
            )
            defenses.append(
                collector.new_event(
                    t0,
                    t1,
                    "DEF_ACTION",
                    source="highlights_filtered",
                    score=TYPE_PRIORITY["DEF_ACTION"]
                    + action_score * 5.5
                    + mean_flow * 2.5,
                    team="us" if team == "us" else "unknown",
                    reason=reason,
                )
            )
    return saves, defenses


def detect_stoppages(
    collector: EventCollector,
    motion: MotionSeries,
    audio_env: Tuple[List[float], List[float]],
    duration: float,
    min_len: float = 1.2,
) -> List[Event]:
    if not motion.times:
        return []

    # Determine thresholds adaptively
    low_motion = np.percentile(np.asarray(motion.global_motion), 12) if motion.global_motion else 0.0
    quiet_audio = 0.35
    audio_times, audio_vals = audio_env

    def audio_level_at(t: float) -> float:
        if not audio_times or not audio_vals:
            return 0.0
        if len(audio_times) < 2:
            return float(audio_vals[0])
        step = max(audio_times[1] - audio_times[0], 1e-6)
        idx = int(round(t / step))
        idx = max(0, min(len(audio_vals) - 1, idx))
        return float(audio_vals[idx])

    events: List[Event] = []
    start_idx: Optional[int] = None
    for idx, (t, gm) in enumerate(zip(motion.times, motion.global_motion)):
        if gm <= low_motion:
            if start_idx is None:
                start_idx = idx
            continue
        if start_idx is not None:
            s = motion.times[start_idx]
            e = motion.times[idx]
            if e - s >= min_len:
                aud = audio_level_at((s + e) / 2.0)
                if aud <= quiet_audio:
                    reason = f"low motion={gm:.2f} audio={aud:.2f}"
                    events.append(
                        collector.new_event(
                            clamp(s, 0.0, duration),
                            clamp(e, 0.0, duration),
                            "STOPPAGE",
                            source="motion",
                            score=-100.0,
                            team="unknown",
                            reason=reason,
                        )
                    )
            start_idx = None
    if start_idx is not None:
        s = motion.times[start_idx]
        e = motion.times[-1]
        if e - s >= min_len:
            aud = audio_level_at((s + e) / 2.0)
            if aud <= quiet_audio:
                reason = f"low motion audio={aud:.2f}"
                events.append(
                    collector.new_event(
                        clamp(s, 0.0, duration),
                        clamp(e, 0.0, duration),
                        "STOPPAGE",
                        source="motion",
                        score=-100.0,
                        team="unknown",
                        reason=reason,
                    )
                )
    return events


###############################################################################
# Selection pipeline
###############################################################################

def subtract_intervals(interval: Tuple[float, float], blocks: Sequence[Tuple[float, float]]):
    """Subtract ``blocks`` from ``interval`` and yield remaining spans."""

    start, end = interval
    if end <= start:
        return []
    spans = [(start, end)]
    for b0, b1 in blocks:
        next_spans = []
        for s, e in spans:
            if e <= b0 or s >= b1:
                next_spans.append((s, e))
                continue
            if s < b0:
                next_spans.append((s, max(s, b0)))
            if e > b1:
                next_spans.append((min(e, b1), e))
        spans = [(max(start, s), min(end, e)) for s, e in next_spans if e - s > 1e-3]
    return spans


def merge_spans(spans: List[Span], merge_gap: float) -> List[Span]:
    if not spans:
        return []
    ordered = sorted(spans, key=lambda s: (s.t0, s.t1))
    merged: List[Span] = [ordered[0]]
    for span in ordered[1:]:
        prev = merged[-1]
        if span.t0 <= prev.t1 + merge_gap:
            prev.t1 = max(prev.t1, span.t1)
            prev.score = max(prev.score, span.score)
            prev.reasons.extend(span.reasons)
            prev.sources.extend(span.sources)
            prev.teams.extend(span.teams)
            prev.types.update(span.types)
            prev.event_ids.update(span.event_ids)
            prev.must_keep = prev.must_keep or span.must_keep
        else:
            merged.append(span)
    return merged


def apply_cap(spans: List[Span], cap_duration: float) -> Tuple[List[Span], List[Span]]:
    if cap_duration <= 0:
        return spans, []
    total = sum(s.length for s in spans)
    if total <= cap_duration:
        return spans, []
    # Determine droppable spans (lowest priority/score first)
    droppable = [
        s
        for s in spans
        if not s.must_keep and TYPE_PRIORITY.get(s.primary_type, 0) < TYPE_PRIORITY["SHOT"]
    ]
    droppable.sort(
        key=lambda s: (
            TYPE_PRIORITY.get(s.primary_type, 0),
            s.score,
            s.length,
        )
    )
    dropped: List[Span] = []
    for span in droppable:
        if total <= cap_duration:
            break
        total -= span.length
        span.length  # no-op to appease linters
        dropped.append(span)
    kept = [s for s in spans if s not in dropped]
    return kept, dropped


###############################################################################
# Output helpers
###############################################################################

def write_events_csv(path: Path, events: Iterable[Dict[str, object]]) -> None:
    rows = list(events)
    ensure_dir(path)
    if not rows:
        with path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "start",
                "end",
                "type",
                "source",
                "score",
                "team",
                "reason",
                "event_id",
            ])
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_spans_csv(path: Path, spans: List[Span]) -> None:
    ensure_dir(path)
    if not spans:
        with path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "start",
                    "end",
                    "type",
                    "score",
                    "team",
                    "source",
                    "reason",
                    "event_ids",
                    "all_types",
                ]
            )
        return
    fieldnames = [
        "start",
        "end",
        "type",
        "score",
        "team",
        "source",
        "reason",
        "event_ids",
        "all_types",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for span in spans:
            writer.writerow(span.to_row())


###############################################################################
# Main CLI
###############################################################################

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True, type=Path, help="Full-game video path")
    p.add_argument("--outdir", default=Path("out"), type=Path)
    p.add_argument("--goal-resets", default=Path("out/goal_resets.csv"), type=Path)
    p.add_argument("--forced-goals", default=Path("out/forced_goals.csv"), type=Path)
    p.add_argument("--shots", default=Path("out/highlights_shots.csv"), type=Path)
    p.add_argument("--filtered", default=Path("out/highlights_filtered.csv"), type=Path)
    p.add_argument("--build", default=Path("out/highlights_build.csv"), type=Path)
    p.add_argument("--plays", default=Path("out/plays.csv"), type=Path)
    p.add_argument("--audio-levels", default=Path("out/audio_levels.csv"), type=Path)
    p.add_argument("--pre-goal", type=float, default=8.0)
    p.add_argument("--post-goal", type=float, default=4.0)
    p.add_argument("--pre-shot", type=float, default=6.0)
    p.add_argument("--post-shot", type=float, default=3.0)
    p.add_argument("--min-span", type=float, default=2.0)
    p.add_argument("--merge-gap", type=float, default=0.35)
    p.add_argument("--cap-frac", type=float, default=0.25)
    p.add_argument(
        "--never-drop",
        default="goals,shots",
        help="Comma separated list of types that should never be removed by the cap",
    )
    p.add_argument(
        "--write-srt",
        action="store_true",
        help="Emit events_overlay.srt alongside the CSV outputs",
    )
    p.add_argument(
        "--team-colors",
        nargs="*",
        help="Optional jersey colours (unused, kept for compatibility)",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.video.exists():
        raise SystemExit(f"Missing video: {args.video}")
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    if CV2_AVAILABLE:
        cap = cv2.VideoCapture(str(args.video))
        if not cap.isOpened():
            raise SystemExit(f"Could not open video: {args.video}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / fps if frame_count else 0.0
        cap.release()
    else:
        fps = 24.0
        duration = probe_video_duration(args.video)

    csv_sources = [
        args.goal_resets,
        args.forced_goals,
        args.shots,
        args.filtered,
        args.build,
        args.plays,
    ]
    duration_hint = estimate_max_timestamp(csv_sources)
    if duration_hint:
        duration = max(
            duration,
            duration_hint + max(args.post_goal, args.post_shot, 5.0),
        )

    never_drop = {t.strip().upper() for t in str(args.never_drop).split(",") if t.strip()}
    ctx = CollectorContext(
        duration=duration,
        pre_goal=args.pre_goal,
        post_goal=args.post_goal,
        pre_shot=args.pre_shot,
        post_shot=args.post_shot,
    )
    collector = EventCollector(never_drop)

    goals = collect_goals(ctx, collector, args.goal_resets, args.forced_goals)
    shots = collect_shots(ctx, collector, args.shots)
    saves, defenses = collect_defense_and_saves(ctx, collector, args.filtered)
    offense = collect_offense(ctx, collector, args.build)

    audio_env = load_audio_envelope(args.video)
    motion = compute_motion_series(args.video)
    stoppages = detect_stoppages(collector, motion, audio_env, duration)

    raw_events = goals + shots + saves + defenses + offense + stoppages
    raw_events_sorted = sorted(raw_events, key=lambda e: (e.t0, e.t1))
    events_raw_path = outdir / "events_raw.csv"
    write_events_csv(events_raw_path, (evt.as_row() for evt in raw_events_sorted))

    # Build stoppage intervals for subtraction
    stoppage_intervals = [(evt.t0, evt.t1) for evt in stoppages]

    spans: List[Span] = []
    for event in raw_events_sorted:
        if event.type == "STOPPAGE":
            continue
        intervals = subtract_intervals((event.t0, event.t1), stoppage_intervals)
        for s, e in intervals:
            if e - s < args.min_span:
                continue
            span = Span(t0=s, t1=e, score=event.score)
            span.add_event(event)
            spans.append(span)

    spans = merge_spans(spans, args.merge_gap)
    spans = [s for s in spans if s.length >= args.min_span]

    cap_duration = duration * float(args.cap_frac)
    kept_spans, dropped_spans = apply_cap(spans, cap_duration)

    selected_ids: Set[str] = set()
    for span in kept_spans:
        selected_ids.update(span.event_ids)

    events_selected_path = outdir / "events_selected.csv"
    write_spans_csv(events_selected_path, kept_spans)

    summary = {
        "video_duration": duration,
        "counts": {
            "raw_events": len(raw_events_sorted),
            "selected_spans": len(kept_spans),
            "dropped_spans": len(dropped_spans),
        },
        "coverage": {
            "goals_total": len(collector.by_type.get("GOAL", set())),
            "goals_included": len(
                collector.by_type.get("GOAL", set()).intersection(selected_ids)
            ),
            "shots_total": len(collector.by_type.get("SHOT", set())),
            "shots_included": len(
                collector.by_type.get("SHOT", set()).intersection(selected_ids)
            ),
            "saves_total": len(collector.by_type.get("SAVE_OUR_GK", set())),
            "saves_included": len(
                collector.by_type.get("SAVE_OUR_GK", set()).intersection(selected_ids)
            ),
            "def_total": len(collector.by_type.get("DEF_ACTION", set())),
            "def_included": len(
                collector.by_type.get("DEF_ACTION", set()).intersection(selected_ids)
            ),
            "off_total": len(collector.by_type.get("OFF_BUILDUP", set())),
            "off_included": len(
                collector.by_type.get("OFF_BUILDUP", set()).intersection(selected_ids)
            ),
            "stoppage_segments": len(stoppages),
        },
        "missing": {
            "goals": sorted(
                collector.by_type.get("GOAL", set()) - selected_ids
            ),
            "shots": sorted(
                collector.by_type.get("SHOT", set()) - selected_ids
            ),
        },
    }
    summary_path = outdir / "review_summary.json"
    ensure_dir(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2))

    print(
        "[select_events] goals {}/{} | shots {}/{} | saves {} | offense {} | defense {}".format(
            summary["coverage"]["goals_included"],
            summary["coverage"]["goals_total"],
            summary["coverage"]["shots_included"],
            summary["coverage"]["shots_total"],
            summary["coverage"]["saves_included"],
            summary["coverage"]["off_included"],
            summary["coverage"]["def_included"],
        )
    )
    total_duration = sum(span.length for span in kept_spans)
    print(
        "[select_events] final spans={} total={:.1f}s cap={:.1f}s".format(
            len(kept_spans), total_duration, cap_duration
        )
    )

    if args.write_srt:
        try:
            from make_overlay import build_overlay_entries, write_srt  # type: ignore

            entries = build_overlay_entries(kept_spans)
            write_srt(outdir / "events_overlay.srt", entries)
        except Exception as exc:  # pragma: no cover - optional feature
            print(f"[select_events] failed to write SRT overlay: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

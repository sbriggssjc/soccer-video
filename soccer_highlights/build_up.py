"""Detection helpers for sustained build-up possessions."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, List, Optional, Sequence

from .utils import HighlightWindow


def _to_float(value: Any) -> Optional[float]:
    """Best-effort conversion to float returning ``None`` on failure."""

    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _get(data: Any, name: str, default: Any = None) -> Any:
    """Return ``name`` from ``data`` supporting dicts and objects."""

    if isinstance(data, dict):
        return data.get(name, default)
    return getattr(data, name, default)


def _normalize_team(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text.lower() if text else None


def _extract_time(data: Any) -> Optional[float]:
    """Return a timestamp in seconds from ``data`` if available."""

    for key in ("time", "timestamp", "ts"):
        value = _get(data, key)
        if value is not None:
            time = _to_float(value)
            if time is not None:
                return time
    frame = _get(data, "frame")
    fps = _get(data, "fps")
    if frame is not None and fps:
        frame_v = _to_float(frame)
        fps_v = _to_float(fps)
        if frame_v is not None and fps_v not in (None, 0):
            try:
                return frame_v / fps_v
            except ZeroDivisionError:  # pragma: no cover - defensive
                return None
    return None


def _extract_coord(data: Any, key: str) -> Optional[float]:
    value = _get(data, key)
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and value:
        return _to_float(value[0 if key == "x" else 1])
    return _to_float(value)


@dataclass(frozen=True)
class TouchEvent:
    time: float
    x: Optional[float]
    y: Optional[float]
    player: Optional[Any]
    team: str


@dataclass(frozen=True)
class BallEvent:
    time: float
    x: Optional[float]
    y: Optional[float]
    team: Optional[str]


@dataclass
class PassChain:
    touches: List[TouchEvent]
    team: str
    start_time: float
    end_time: float
    ball_events: List[BallEvent]

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)


def _prepare_touches(tracks: Sequence[Any], team: str) -> List[TouchEvent]:
    touches: List[TouchEvent] = []
    for track in tracks:
        if _normalize_team(_get(track, "team")) != team:
            continue
        time = _extract_time(track)
        if time is None:
            continue
        x = _extract_coord(track, "x")
        y = _extract_coord(track, "y")
        player = _get(track, "player", _get(track, "id", None))
        touches.append(TouchEvent(time=time, x=x, y=y, player=player, team=team))
    touches.sort(key=lambda t: t.time)
    return touches


def _prepare_ball(ball: Sequence[Any]) -> List[BallEvent]:
    events: List[BallEvent] = []
    for sample in ball:
        time = _extract_time(sample)
        if time is None:
            continue
        x = _extract_coord(sample, "x")
        y = _extract_coord(sample, "y")
        team = _normalize_team(_get(sample, "team"))
        events.append(BallEvent(time=time, x=x, y=y, team=team))
    events.sort(key=lambda e: e.time)
    return events


def _slice_ball(events: Sequence[BallEvent], start: float, end: float, pad: float) -> List[BallEvent]:
    if not events:
        return []
    lo = start - pad
    hi = end + pad
    return [event for event in events if lo <= event.time <= hi]


def _extend_with_ball(start: float, end: float, events: Sequence[BallEvent], team: str, max_gap: float) -> tuple[float, float]:
    if not events:
        return start, end
    # last event before start
    prev: Optional[BallEvent] = None
    for event in events:
        if event.time < start:
            prev = event
        else:
            break
    if prev and prev.team == team and start - prev.time <= max_gap:
        start = prev.time
    # first event after end
    next_event: Optional[BallEvent] = None
    for event in events:
        if event.time > end:
            next_event = event
            break
    if next_event and next_event.team == team and next_event.time - end <= max_gap:
        end = next_event.time
    return start, end


def pass_chains(
    tracks: Sequence[Any],
    ball: Sequence[Any],
    *,
    team: str = "navy",
    max_gap_s: float = 2.0,
    min_len: int = 4,
) -> List[PassChain]:
    """Group touches into sustained same-team pass chains."""

    target_team = _normalize_team(team)
    if not target_team:
        return []
    touches = _prepare_touches(tracks, target_team)
    if not touches:
        return []
    ball_events = _prepare_ball(ball)

    chains: List[PassChain] = []
    current: List[TouchEvent] = []
    for touch in touches:
        if not current:
            current.append(touch)
            continue
        gap = touch.time - current[-1].time
        if gap <= max_gap_s:
            current.append(touch)
        else:
            if len(current) >= min_len:
                start = current[0].time
                end = current[-1].time
                subset = _slice_ball(ball_events, start, end, max_gap_s)
                start, end = _extend_with_ball(start, end, subset, target_team, max_gap_s)
                chains.append(PassChain(touches=list(current), team=target_team, start_time=start, end_time=end, ball_events=subset))
            current = [touch]
    if len(current) >= min_len:
        start = current[0].time
        end = current[-1].time
        subset = _slice_ball(ball_events, start, end, max_gap_s)
        start, end = _extend_with_ball(start, end, subset, target_team, max_gap_s)
        chains.append(PassChain(touches=list(current), team=target_team, start_time=start, end_time=end, ball_events=subset))
    return chains


def _first_position(events: Sequence[BallEvent]) -> Optional[tuple[float, float]]:
    for event in events:
        if event.x is not None and event.y is not None:
            return float(event.x), float(event.y)
    return None


def _last_position(events: Sequence[BallEvent]) -> Optional[tuple[float, float]]:
    for event in reversed(events):
        if event.x is not None and event.y is not None:
            return float(event.x), float(event.y)
    return None


def _touch_position(touch: TouchEvent) -> Optional[tuple[float, float]]:
    if touch.x is None or touch.y is None:
        return None
    return float(touch.x), float(touch.y)


def field_progress(chain: PassChain) -> float:
    """Return field progress in arbitrary units for a pass chain."""

    start = _first_position(chain.ball_events) or _touch_position(chain.touches[0])
    end = _last_position(chain.ball_events) or _touch_position(chain.touches[-1])
    if not start or not end:
        return 0.0
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    advance = math.hypot(dx, dy)
    return float(max(0.0, advance))


def time_span(chain: PassChain) -> float:
    return chain.duration


def _score_chain(chain: PassChain) -> float:
    progress = field_progress(chain)
    passes = len(chain.touches)
    span = time_span(chain)
    score = 0.0
    if progress:
        score += min(progress / 30.0, 0.6)
    if passes:
        score += min(passes / 10.0, 0.3)
    if span:
        score += min(span / 30.0, 0.1)
    return max(0.0, min(score, 1.0))


def span_to_clip(chain: PassChain, *, extra_pre: float, extra_post: float) -> HighlightWindow:
    start = max(0.0, chain.start_time - extra_pre)
    end = max(start + 0.1, chain.end_time + extra_post)
    return HighlightWindow(start=start, end=end, score=_score_chain(chain), event="build_up")


def detect_build_up(frames: Sequence[Any], tracks: Sequence[Any], ball: Sequence[Any]) -> List[HighlightWindow]:
    del frames  # build-up relies on tracks + ball metadata, not raw frames
    chains = pass_chains(tracks, ball, team="navy", max_gap_s=2.0, min_len=4)
    good = [c for c in chains if field_progress(c) > 12.0 and time_span(c) <= 18.0]
    return [span_to_clip(c, extra_pre=0.8, extra_post=1.2) for c in good]


__all__ = [
    "BallEvent",
    "PassChain",
    "TouchEvent",
    "detect_build_up",
    "field_progress",
    "pass_chains",
    "span_to_clip",
    "time_span",
]

"""Helpers for suppressing highlight clips around stoppages and restarts.

The detection scripts emit coarse event tags such as restarts, ball-out,
and prolonged stoppages.  To avoid producing highlight clips from those
moments we maintain a :class:`ClipBlockState` that tracks two concepts:

``block_until``
    Absolute time before which new clips should be discarded.  This is
    primarily driven by restart tags – we wait until the first live touch
    after the restart before considering new highlights.

``no_clip_windows``
    A merged list of intervals that should not produce clips.  These grow
    as we observe the ball exiting play or a stoppage cluster (medical
    staff, substitutions, referee conference, ...).  The windows "slide"
    forward as new tags arrive, ensuring we never output footage of the
    dead-ball sequences themselves.

The functions below remain agnostic of the underlying video analytics –
tests provide synthetic touch timestamps to validate the behaviour.  Real
detectors simply need to supply the times at which a first touch (or other
"live" signal) was observed after a given anchor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, List, Sequence, Tuple

__all__ = ["ClipBlockState", "first_live_frame"]

if TYPE_CHECKING:  # pragma: no cover - typing aid
    from .utils import HighlightWindow


def _merge_intervals(intervals: List[Tuple[float, float]], *, epsilon: float = 1e-6) -> List[Tuple[float, float]]:
    """Merge overlapping or touching intervals.

    ``epsilon`` avoids tiny floating point gaps from splitting otherwise
    continuous regions.
    """

    if not intervals:
        return []
    ordered = sorted((max(0.0, start), max(0.0, end)) for start, end in intervals if end > start)
    if not ordered:
        return []
    merged: List[Tuple[float, float]] = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + epsilon:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def first_live_frame(
    anchor_time: float,
    touch_times: Sequence[float] | None,
    *,
    max_wait: float = 8.0,
    fallback: float | None = None,
) -> float:
    """Return the first time at or after ``anchor_time`` where play resumes.

    ``touch_times`` should contain timestamps (in seconds) where the ball
    was touched or otherwise deemed "live".  The function returns the first
    such timestamp greater than or equal to ``anchor_time``.  When no touch
    is observed within ``max_wait`` seconds we fall back to either the
    supplied ``fallback`` value or ``anchor_time + max_wait``.
    """

    limit = anchor_time + max_wait
    tolerance = 1e-3
    if touch_times:
        for ts in sorted(touch_times):
            if ts + tolerance >= anchor_time:
                return min(max(ts, anchor_time), limit)
    if fallback is not None:
        return min(max(fallback, anchor_time), limit)
    return limit


@dataclass
class ClipBlockState:
    """Track clip suppression windows derived from event tags."""

    block_until: float = 0.0
    _no_clip: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def no_clip_windows(self) -> List[Tuple[float, float]]:
        """Return merged no-clip windows accumulated so far."""

        return list(self._no_clip)

    def _add_zone(self, start: float, end: float) -> Tuple[float, float]:
        if end <= start:
            return start, start
        self._no_clip.append((start, end))
        self._no_clip = _merge_intervals(self._no_clip)
        return start, end

    def record_restart(
        self,
        restart_time: float,
        touches: Sequence[float] | None,
        *,
        cooldown: float = 3.0,
        max_wait: float = 8.0,
    ) -> float:
        """Register a restart and extend ``block_until`` until play resumes."""

        resume = first_live_frame(
            restart_time,
            touches,
            max_wait=max_wait,
            fallback=restart_time + cooldown,
        )
        self.block_until = max(self.block_until, resume)
        self._add_zone(restart_time, resume)
        return resume

    def add_out_of_play(
        self,
        exit_time: float,
        *,
        return_time: float | None = None,
        touches: Sequence[float] | None = None,
        linger: float = 0.5,
        max_wait: float = 8.0,
    ) -> Tuple[float, float]:
        """Suppress clips while the ball is out of play.

        ``return_time`` can be supplied when the detector explicitly
        observes the re-entry frame; otherwise we fall back to the next
        touch timestamp (or ``exit_time + max_wait``).
        """

        anchor = return_time if return_time is not None else exit_time
        resume = first_live_frame(
            anchor,
            touches,
            max_wait=max_wait,
            fallback=anchor + max_wait,
        )
        resume = max(resume + linger, exit_time)
        return self._add_zone(exit_time, resume)

    def add_stoppage(
        self,
        start: float,
        end: float,
        *,
        touches: Sequence[float] | None = None,
        linger: float = 1.0,
        max_wait: float = 8.0,
    ) -> Tuple[float, float]:
        """Add a no-clip window for a stoppage cluster."""

        resume = first_live_frame(
            end,
            touches,
            max_wait=max_wait,
            fallback=end + linger,
        )
        resume = max(resume, end + linger)
        return self._add_zone(start, resume)

    def is_blocked(self, start: float, end: float) -> bool:
        """Return ``True`` when the given span overlaps a suppression zone."""

        if end <= start:
            return False
        if start < self.block_until:
            return True
        for zone_start, zone_end in self._no_clip:
            if start < zone_end and end > zone_start:
                return True
        return False

    def filter_windows(
        self,
        windows: Sequence["HighlightWindow"],
        *,
        banned_events: Iterable[str] | None = None,
    ) -> List["HighlightWindow"]:
        """Drop windows that fall inside suppression regions or categories."""

        if banned_events is None:
            banned = {"restart", "setup"}
        else:
            banned = set(banned_events)

        kept: List["HighlightWindow"] = []
        for win in windows:
            if win.event in banned:
                continue
            if self.is_blocked(win.start, win.end):
                continue
            kept.append(win)
        return kept


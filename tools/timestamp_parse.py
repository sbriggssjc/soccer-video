"""Shared timestamp parsing for soccer-video pipeline tools.

Handles five timestamp formats found across game indexes:

  1. MM:SS.f     — "6:27.5"  → 387.5s
  2. Plain secs  — "1,247.80" → 1247.8s
  3. MM:SS:cs    — "5:15:00"  → 315.0s   (minutes, seconds, centiseconds)
  4. H:MM:SS     — "0:06:37"  → 397.0s   (hours, minutes, seconds)
  5. Mixed pairs — one format per column

Formats 3 and 4 are ambiguous when parsed individually (both are A:B:C).
``parse_timestamp_pair`` resolves the ambiguity by trying MM:SS:cs first; if
that yields a clip shorter than 2 seconds it falls back to H:MM:SS.
"""

from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

_THREE_PART_RE = re.compile(r"^(\d+):(\d+):(\d+(?:\.\d+)?)$")  # A:B:C
_MMSS_RE = re.compile(r"^(\d+):(\d+(?:\.\d+)?)$")              # MM:SS or MM:SS.f
_SEC_RE = re.compile(r"^[\d,]+(?:\.\d+)?$")                      # plain seconds

# Minimum duration (seconds) for MM:SS:cs to be accepted over H:MM:SS
MIN_MMSS_CS_DURATION = 2.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_three_part_as_mmss_cs(a: float, b: float, c: float) -> float:
    """Interpret A:B:C as MM:SS:centiseconds (e.g., "5:15:00" = 5m 15.00s)."""
    return a * 60 + b + c / 100


def _parse_three_part_as_hmmss(a: float, b: float, c: float) -> float:
    """Interpret A:B:C as H:MM:SS (e.g., "0:06:37" = 0h 6m 37s)."""
    return a * 3600 + b * 60 + c


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_timestamp(raw: str) -> Optional[float]:
    """Parse a single timestamp string to seconds.

    For three-part timestamps (A:B:C), defaults to MM:SS:cs.
    Use ``parse_timestamp_pair`` when you have both start and end to
    disambiguate MM:SS:cs vs H:MM:SS.
    """
    raw = raw.strip().strip('"')
    if not raw:
        return None

    # Three-part: default to MM:SS:cs; pair-level disambiguation overrides
    m = _THREE_PART_RE.match(raw)
    if m:
        a, b, c = float(m.group(1)), float(m.group(2)), float(m.group(3))
        return _parse_three_part_as_mmss_cs(a, b, c)

    # Two-part: MM:SS or MM:SS.f
    m = _MMSS_RE.match(raw)
    if m:
        return float(m.group(1)) * 60 + float(m.group(2))

    # Plain seconds with optional comma separators
    m = _SEC_RE.match(raw)
    if m:
        return float(raw.replace(",", ""))

    return None


def parse_timestamp_pair(
    raw_start: str,
    raw_end: str,
    *,
    master_duration: Optional[float] = None,
) -> tuple[Optional[float], Optional[float]]:
    """Parse a start/end timestamp pair, disambiguating three-part formats.

    When both timestamps match A:B:C, tries MM:SS:cs first.  If that gives
    a clip shorter than ``MIN_MMSS_CS_DURATION`` seconds, retries as H:MM:SS
    (needed for TSC Navy timestamps like "0:06:37" = 6 min 37 sec, not 6.37 sec).

    When *master_duration* is provided (seconds), an additional guard rejects
    any interpretation whose end timestamp exceeds the master length — this
    catches the common "5:40:00" mis-parse where MM:SS:00 is read as HH:MM:SS
    producing 20400 s against a 3000 s master.
    """
    raw_start = (raw_start or "").strip().strip('"')
    raw_end = (raw_end or "").strip().strip('"')
    if not raw_start or not raw_end:
        return None, None

    ms = _THREE_PART_RE.match(raw_start)
    me = _THREE_PART_RE.match(raw_end)

    if ms and me:
        sa, sb, sc = float(ms.group(1)), float(ms.group(2)), float(ms.group(3))
        ea, eb, ec = float(me.group(1)), float(me.group(2)), float(me.group(3))

        t0_mmss = _parse_three_part_as_mmss_cs(sa, sb, sc)
        t1_mmss = _parse_three_part_as_mmss_cs(ea, eb, ec)
        t0_hms = _parse_three_part_as_hmmss(sa, sb, sc)
        t1_hms = _parse_three_part_as_hmmss(ea, eb, ec)

        dur_mmss = t1_mmss - t0_mmss
        dur_hms = t1_hms - t0_hms

        # If MM:SS:cs gives a clip < 2 seconds but H:MM:SS gives a reasonable
        # duration, use H:MM:SS  (handles Navy-style "0:06:37-0:07:10")
        if dur_mmss < MIN_MMSS_CS_DURATION and dur_hms >= MIN_MMSS_CS_DURATION:
            return t0_hms, t1_hms

        # Master-duration guard: if MM:SS:cs end exceeds the master length,
        # the H:MM:SS interpretation was likely wrong (e.g. "5:40:00" should
        # be 340 s not 20400 s).  Already-selected MM:SS:cs is fine — but if
        # *that* also exceeds the master, something is very wrong; return it
        # anyway and let downstream handle it.
        if master_duration is not None:
            if t1_mmss <= master_duration:
                return t0_mmss, t1_mmss
            if t1_hms <= master_duration:
                return t0_hms, t1_hms
            # Both exceed — fall through with the default (mmss) interpretation

        return t0_mmss, t1_mmss

    # Fall back to individual parsing
    return parse_timestamp(raw_start), parse_timestamp(raw_end)

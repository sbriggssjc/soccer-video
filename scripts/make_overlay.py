#!/usr/bin/env python3
"""Generate an SRT overlay describing highlight events.

The script is intentionally lightweight: it accepts a CSV describing the final
spans (``events_selected.csv`` from :mod:`select_events`) and emits a matching
``events_overlay.srt`` file that can be muxed with the highlight reel.  The SRT
labels list the event type, score, and condensed reason/source text which makes
frame-by-frame QA much easier.

The module also exposes ``build_overlay_entries``/``write_srt`` helpers so that
other tooling (e.g. :mod:`select_events`) can opportunistically reuse the same
format.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


@dataclass
class OverlayEntry:
    idx: int
    start: float
    end: float
    text: str


def _to_float(val) -> float:
    if isinstance(val, (int, float)):
        return float(val)
    if val is None:
        return 0.0
    s = str(val).strip().replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return 0.0


def _format_ts(seconds: float) -> str:
    seconds = max(0.0, seconds)
    ms = int(round((seconds - int(seconds)) * 1000))
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, sec = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d},{ms:03d}"


def _extract_text(obj) -> str:
    if hasattr(obj, "reason_text"):
        return getattr(obj, "reason_text")
    if isinstance(obj, dict):
        return str(obj.get("reason", "")).strip()
    return ""


def _extract_source(obj) -> str:
    if hasattr(obj, "source_text"):
        return getattr(obj, "source_text")
    if isinstance(obj, dict):
        return str(obj.get("source", "")).strip()
    return ""


def _extract_type(obj) -> str:
    if hasattr(obj, "primary_type"):
        return getattr(obj, "primary_type")
    if isinstance(obj, dict):
        return str(obj.get("type", "EVENT")).strip() or "EVENT"
    return "EVENT"


def _extract_score(obj) -> float:
    if isinstance(obj, dict):
        return _to_float(obj.get("score", 0.0))
    return float(getattr(obj, "score", 0.0))


def _extract_length(obj) -> float:
    if hasattr(obj, "length"):
        return getattr(obj, "length")
    if isinstance(obj, dict):
        s = _to_float(obj.get("start", 0.0))
        e = _to_float(obj.get("end", 0.0))
        return max(0.0, e - s)
    start = float(getattr(obj, "t0", 0.0))
    end = float(getattr(obj, "t1", 0.0))
    return max(0.0, end - start)


def build_overlay_entries(
    spans: Sequence[object],
    mode: str = "reel",
) -> List[OverlayEntry]:
    """Convert spans to ``OverlayEntry`` objects.

    ``mode`` controls the timeline reference:

    * ``reel`` (default): each entry is placed sequentially so that the SRT
      aligns with the rendered highlight reel.
    * ``absolute``: times are kept relative to the raw match video (useful when
      burning overlays onto the source footage).
    """

    entries: List[OverlayEntry] = []
    offset = 0.0
    for idx, span in enumerate(spans, start=1):
        span_type = _extract_type(span)
        if span_type.upper() == "STOPPAGE":
            continue
        start = float(getattr(span, "t0", _to_float(getattr(span, "start", 0.0))))
        end = float(getattr(span, "t1", _to_float(getattr(span, "end", 0.0))))
        length = max(0.1, _extract_length(span))
        score = _extract_score(span)
        reason = _extract_text(span)
        source = _extract_source(span)
        label_bits = [f"{span_type.upper()} ({score:.1f})"]
        if source:
            label_bits.append(source)
        if reason:
            label_bits.append(reason)
        text = " | ".join(label_bits)
        if mode == "reel":
            s = offset
            e = offset + length
            offset = e
        else:
            s = start
            e = end
        entries.append(OverlayEntry(idx=idx, start=s, end=e, text=text))
    return entries


def write_srt(path: Path, entries: Iterable[OverlayEntry]) -> None:
    entries = list(entries)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(f"{entry.idx}\n")
            fh.write(f"{_format_ts(entry.start)} --> {_format_ts(entry.end)}\n")
            fh.write(f"{entry.text}\n\n")


def _read_events_csv(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if (row.get("type", "").strip() or "").upper() == "STOPPAGE":
                continue
            rows.append(row)
    return rows


def _cli(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", required=True, type=Path, help="events_selected.csv")
    ap.add_argument("--out", required=True, type=Path, help="Output SRT path")
    ap.add_argument(
        "--mode",
        choices=("reel", "absolute"),
        default="reel",
        help="Timeline reference for the overlay",
    )
    args = ap.parse_args(argv)

    if not args.events.exists():
        raise SystemExit(f"Missing events CSV: {args.events}")

    rows = _read_events_csv(args.events)
    if not rows:
        raise SystemExit("No events to annotate; aborting overlay generation")

    entries = build_overlay_entries(rows, mode=args.mode)
    write_srt(args.out, entries)
    print(f"[make_overlay] wrote {len(entries)} entries -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())

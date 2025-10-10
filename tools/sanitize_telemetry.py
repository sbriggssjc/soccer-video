#!/usr/bin/env python
"""Clean telemetry JSONL files by replacing non-finite numeric values."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable


def _sanitize_value(value: Any) -> Any:
    """Recursively replace non-finite floats with zero equivalents.

    Overlay tooling downstream expects numeric values to be finite so that they
    can be rounded safely.  We coerce NaN/inf to ``0.0`` while leaving other
    values untouched.  Containers (lists/tuples/dicts) are sanitized
    recursively.
    """

    if isinstance(value, float):
        return value if math.isfinite(value) else 0.0
    if isinstance(value, (int, bool)):
        # ints/bools are already safe; bool is subclass of int but we preserve
        # the original value for readability.
        return value
    if isinstance(value, dict):
        return {k: _sanitize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(v) for v in value]
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in {"nan", "+nan", "-nan", "inf", "+inf", "-inf"}:
            return 0.0
    return value


def _sanitize_record(record: dict[str, Any]) -> dict[str, Any]:
    return {k: _sanitize_value(v) for k, v in record.items()}


def sanitize_jsonl(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as inp, dst.open("w", encoding="utf-8") as out:
        for line in inp:
            stripped = line.strip()
            if not stripped:
                out.write("\n")
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                # Preserve the original line if it wasn't valid JSON to begin
                # with—matching the lenient behaviour of overlay_debug.
                out.write(line)
                continue
            clean = _sanitize_record(record)
            out.write(json.dumps(clean, ensure_ascii=False) + "\n")



def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="src", required=True, help="Telemetry JSONL to sanitize")
    parser.add_argument(
        "--out",
        dest="dst",
        help="Optional path for sanitized output (defaults to in-place update)",
    )
    args = parser.parse_args(argv)

    src = Path(args.src).expanduser().resolve()
    if args.dst:
        dst = Path(args.dst).expanduser().resolve()
    else:
        dst = src

    if not src.exists():
        parser.error(f"Telemetry file not found: {src}")

    if dst == src:
        tmp = src.with_suffix(src.suffix + ".sanitized")
        sanitize_jsonl(src, tmp)
        tmp.replace(src)
    else:
        sanitize_jsonl(src, dst)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

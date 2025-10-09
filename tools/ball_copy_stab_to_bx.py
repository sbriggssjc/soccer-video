#!/usr/bin/env python3
"""Rewrite ball path JSONL files to expose bx/by in source pixel units."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def _select_coord(record: dict, keys: Iterable[str]) -> float | None:
    for key in keys:
        if key in record and record[key] is not None:
            try:
                return float(record[key])
            except (TypeError, ValueError):
                return None
    return None


def process_ball_jsonl(src_path: Path, dst_path: Path) -> int:
    """Copy *stab/raw* coordinates into ``bx``/``by`` for each record."""
    count = 0
    with src_path.open("r", encoding="utf-8") as fsrc, dst_path.open(
        "w", encoding="utf-8"
    ) as fdst:
        for line in fsrc:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                continue

            bx = _select_coord(record, ("bx_stab", "bx_raw", "bx"))
            by = _select_coord(record, ("by_stab", "by_raw", "by"))
            if bx is None or by is None:
                continue

            record["bx"] = float(bx)
            record["by"] = float(by)
            fdst.write(json.dumps(record) + "\n")
            count += 1
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite a ball-path JSONL file so that bx/by contain source pixel coordinates "
            "copied from bx_stab/by_stab (or raw fallbacks)."
        )
    )
    parser.add_argument("input", type=Path, help="Path to the source JSONL file")
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        help="Optional output path; defaults to <input>.fixed.jsonl",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    src_path: Path = args.input.expanduser().resolve()
    if not src_path.exists():
        parser.error(f"Input file not found: {src_path}")

    if args.output is None:
        name = src_path.name
        if name.endswith(".jsonl"):
            dst_name = name[: -len(".jsonl")] + ".fixed.jsonl"
        else:
            dst_name = name + ".fixed"
        dst_path = src_path.with_name(dst_name)
    else:
        dst_path = args.output.expanduser().resolve()
        dst_path.parent.mkdir(parents=True, exist_ok=True)

    count = process_ball_jsonl(src_path, dst_path)
    print(f"Wrote {count} records to {dst_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

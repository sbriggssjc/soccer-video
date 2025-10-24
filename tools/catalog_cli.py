"""Thin command-line shim for the catalog module."""

from __future__ import annotations

import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import catalog  # type: ignore  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    return catalog.main(argv)


if __name__ == "__main__":
    sys.exit(main())


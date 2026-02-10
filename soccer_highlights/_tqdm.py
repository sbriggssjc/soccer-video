
"""Fallback wrapper for tqdm progress bars."""
from __future__ import annotations

try:  # pragma: no cover
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    class _Dummy:
        def __init__(self, iterable=None):
            self.iterable = iterable or []

        def __iter__(self):
            return iter(self.iterable)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def update(self, *args, **kwargs):
            return None

    def tqdm(iterable=None, **kwargs):  # type: ignore
        if iterable is None:
            return _Dummy()
        return _Dummy(iterable)

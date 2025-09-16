"""Fallback logger mirroring the loguru API when loguru is unavailable."""
from __future__ import annotations

import logging
from typing import Any, Callable

try:  # pragma: no cover
    from loguru import logger  # type: ignore
except Exception:  # pragma: no cover
    _logger = logging.getLogger("soccer_highlights")
    if not _logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        _logger.addHandler(handler)
    _logger.setLevel(logging.INFO)

    class _ProxyLogger:
        def add(self, sink: Any, level: str = "INFO") -> None:
            if isinstance(sink, int):
                return
            _logger.setLevel(level)

        def remove(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __getattr__(self, name: str) -> Callable[..., None]:
            return getattr(_logger, name)

    logger = _ProxyLogger()

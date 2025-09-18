"""Soccer highlights extraction toolkit."""


from typing import TYPE_CHECKING, Any

__all__ = ["AppConfig", "load_config"]

if TYPE_CHECKING:  # pragma: no cover - for static type checkers only
    from .config import AppConfig, load_config


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .config import AppConfig as _AppConfig, load_config as _load_config

        globals().update({"AppConfig": _AppConfig, "load_config": _load_config})
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

from .blocking import ClipBlockState, first_live_frame
from .config import AppConfig, load_config

__all__ = ["AppConfig", "ClipBlockState", "first_live_frame", "load_config"]


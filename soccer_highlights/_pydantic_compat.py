"""Lightweight compatibility layer for pydantic."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Type

try:  # pragma: no cover - use real pydantic when available
    from pydantic import BaseModel, Field, field_validator  # type: ignore
except Exception:  # pragma: no cover
    class FieldInfo:
        def __init__(self, default: Any = ..., default_factory: Any | None = None, **kwargs: Any) -> None:
            self.default = default
            self.default_factory = default_factory
            self.extra = kwargs
            self.ge = kwargs.get('ge')
            self.le = kwargs.get('le')

    def Field(default: Any = ..., default_factory: Any | None = None, **kwargs: Any) -> FieldInfo:
        return FieldInfo(default=default, default_factory=default_factory, **kwargs)

    def field_validator(field_name: str, *, mode: str = "after"):
        def decorator(func):
            func._validator_field = field_name
            return func
        return decorator

    class BaseModelMeta(type):
        def __new__(mcls, name: str, bases: tuple[type, ...], namespace: Dict[str, Any]):
            annotations: Dict[str, Any] = {}
            for base in reversed(bases):
                annotations.update(getattr(base, "__annotations__", {}))
            annotations.update(namespace.get("__annotations__", {}))
            field_defs: Dict[str, FieldInfo] = {}
            field_types: Dict[str, Any] = {}
            validators: Dict[str, Any] = {}
            for key, value in list(namespace.items()):
                raw = value
                if isinstance(raw, classmethod):
                    raw = raw.__func__
                if callable(raw) and hasattr(raw, "_validator_field"):
                    validators[raw._validator_field] = raw
                elif hasattr(value, "_validator_field"):
                    # classmethod descriptors aren't callable in Python 3.11+
                    validators[value._validator_field] = raw
            for key, anno in annotations.items():
                if key.startswith("__") and key.endswith("__"):
                    continue
                field_types[key] = anno
                default = namespace.get(key, ...)
                if isinstance(default, FieldInfo):
                    field_defs[key] = default
                    namespace.pop(key, None)
                elif default is ...:
                    field_defs[key] = FieldInfo(default=...)
                else:
                    field_defs[key] = FieldInfo(default=default)
            namespace["__field_definitions__"] = field_defs
            namespace["__field_types__"] = field_types
            namespace["__validators__"] = validators
            cls = super().__new__(mcls, name, bases, namespace)
            # Resolve string annotations (from __future__ annotations)
            try:
                import typing as _typing
                hints = _typing.get_type_hints(cls)
                resolved: Dict[str, Any] = {}
                for k in field_types:
                    resolved[k] = hints.get(k, field_types[k])
                cls.__field_types__ = resolved
            except Exception:
                pass
            return cls

    class BaseModel(metaclass=BaseModelMeta):
        __field_definitions__: Dict[str, FieldInfo]
        __field_types__: Dict[str, Any]
        __validators__: Dict[str, Any]

        def __init__(self, **data: Any) -> None:
            for name, info in self.__field_definitions__.items():
                value = data.get(name, ...)
                if value is ...:
                    if info.default is not ...:
                        value = info.default
                    elif info.default_factory is not None:
                        value = info.default_factory()
                    else:
                        raise ValueError(f"Field '{name}' is required")
                value = self._coerce(name, value)
                info = self.__field_definitions__[name]
                if info.ge is not None and value is not None and value < info.ge:
                    raise ValueError(f'{name} must be >= {info.ge}')
                if info.le is not None and value is not None and value > info.le:
                    raise ValueError(f'{name} must be <= {info.le}')
                validator_fn = self.__validators__.get(name)
                if validator_fn is not None:
                    value = validator_fn(self.__class__, value)
                setattr(self, name, value)

        def _coerce(self, name: str, value: Any) -> Any:
            anno = self.__field_types__.get(name)
            origin = getattr(anno, "__origin__", None)
            if origin is list and isinstance(value, list):
                subtype = anno.__args__[0]
                return [self._coerce_value(subtype, item) for item in value]
            if origin is dict and isinstance(value, dict):
                key_type, val_type = anno.__args__
                return {
                    self._coerce_value(key_type, k): self._coerce_value(val_type, v)
                    for k, v in value.items()
                }
            return self._coerce_value(anno, value)

        def _coerce_value(self, anno: Any, value: Any) -> Any:
            if anno is None:
                return value
            if isinstance(anno, type) and issubclass(anno, BaseModel) and isinstance(value, dict):
                return anno.parse_obj(value)
            if anno is Path and isinstance(value, str):
                return Path(value)
            return value

        @classmethod
        def parse_obj(cls: Type["BaseModel"], obj: Dict[str, Any]) -> "BaseModel":
            return cls(**obj)

        @classmethod
        def model_validate(cls: Type["BaseModel"], obj: Dict[str, Any]) -> "BaseModel":
            return cls(**obj)

        def dict(self) -> Dict[str, Any]:
            return self.__dict__.copy()

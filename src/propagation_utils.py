"""Utilities for propagating entity identifiers across row collections.

This module provides helpers that normalize database cursor rows and then
propagate known entity identifiers to other rows that share the same names.
"""
from __future__ import annotations

from typing import Any, Iterable, MutableMapping, Sequence


def _normalize_row(row: Any, columns: Sequence[str] | None) -> MutableMapping[str, Any]:
    """Return a mapping view of *row*.

    The function tolerates dictionary input, database cursor rows that support
    the mapping protocol, and positional tuples that align with *columns*.
    """

    if isinstance(row, dict):
        return row

    try:
        return dict(row)
    except Exception:
        pass

    if columns is None:
        raise TypeError("Column names are required to normalize non-mapping rows")

    return {col: row[idx] if idx < len(row) else None for idx, col in enumerate(columns)}


def propagate_entity_ids(
    rows: Iterable[Any],
    source_name_field: str,
    target_name_field: str,
    source_entity_id_field: str,
    target_entity_id_field: str | None = None,
    *,
    columns: Sequence[str] | None = None,
) -> list[MutableMapping[str, Any]]:
    """Propagate known entity IDs to matching rows.

    The input *rows* may come from a database cursor that yields tuples.  Each
    row is normalized to a mapping before the propagation logic runs so that
    subsequent ``.get(...)`` calls are safe.
    """

    normalized_rows = [_normalize_row(row, columns) for row in rows]

    if target_entity_id_field is None:
        target_entity_id_field = source_entity_id_field

    name_to_entity: dict[str, Any] = {}
    for row in normalized_rows:
        src_name = (row.get(source_name_field) or "").strip()
        src_entity_id = row.get(source_entity_id_field)
        if src_name and src_entity_id is not None:
            name_to_entity[src_name.lower()] = src_entity_id

    for row in normalized_rows:
        current_target = row.get(target_entity_id_field)
        if current_target:
            continue
        dst_name = (row.get(target_name_field) or "").strip()
        if not dst_name:
            continue
        match = name_to_entity.get(dst_name.lower())
        if match is not None:
            row[target_entity_id_field] = match

    return normalized_rows

"""Utilities for logging learning corrections to Supabase.

The module maintains a default learning client that can be configured once at
startup and reused by subsequent calls. This mirrors the behavior expected by
``safe_log_learning`` so callers don't have to pass the client each time.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

_default_learning_client: Optional[Any] = None


def configure_default_learning_client(client: Any) -> None:
    """Register a default learning client for subsequent logging calls."""
    global _default_learning_client
    _default_learning_client = client


def _get_learning_client(client: Optional[Any] = None) -> Any:
    client_to_use = client or _default_learning_client
    if client_to_use is None:
        raise RuntimeError(
            "safe_log_learning requires a Supabase client. "
            "Pass one explicitly or call configure_default_learning_client() during startup."
        )
    return client_to_use


def safe_log_learning(
    *,
    property_id: Optional[str],
    field_name: str,
    table_name: str,
    old_value: Any,
    new_value: Any,
    notes: str,
    client: Optional[Any] = None,
) -> None:
    """Safely send a learning log entry to Supabase.

    If no client is configured, a ``RuntimeError`` will be raised. Otherwise the
    function will attempt to insert the payload using the provided client,
    falling back to a console log when a table interface is unavailable.
    """

    client_to_use = _get_learning_client(client)
    payload: Dict[str, Any] = {
        "property_id": property_id,
        "field_name": field_name,
        "table_name": table_name,
        "old_value": old_value,
        "new_value": new_value,
        "notes": notes,
    }

    # Minimal attempt to persist the payload while staying dependency-light.
    if hasattr(client_to_use, "table"):
        client_to_use.table("learning_logs").insert(payload).execute()
    elif hasattr(client_to_use, "log_learning"):
        client_to_use.log_learning(payload)
    else:
        # Fall back to a console log so callers have some visibility.
        print(f"Learning log (no-op client): {payload}")

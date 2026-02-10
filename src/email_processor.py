"""Process inbound emails and log learning updates safely."""
from __future__ import annotations

from typing import Iterable, Optional

from .correction_logger import configure_default_learning_client, safe_log_learning


class EmailRecord(dict):
    """Lightweight mapping to represent an email payload."""

    @property
    def match_id(self) -> Optional[str]:
        return self.get("match_id")


def process_emails(emails: Iterable[EmailRecord], supabase_client: Optional[object] = None) -> int:
    """Process each email and forward updates to the learning logger.

    The Supabase client is configured once up front so later calls to
    ``safe_log_learning`` have the default client available. Logging is guarded
    to avoid aborting ingestion runs when the logger misbehaves.
    """

    if supabase_client is not None:
        configure_default_learning_client(supabase_client)

    processed = 0
    for email in emails:
        processed += 1
        try:
            safe_log_learning(
                property_id=email.match_id,
                field_name=email.get("field_name", "unknown"),
                table_name=email.get("table_name", "unknown"),
                old_value=email.get("old_value"),
                new_value=email.get("new_value"),
                notes=email.get("notes", ""),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️ safe_log_learning failed (non-fatal): {exc}")
    return processed

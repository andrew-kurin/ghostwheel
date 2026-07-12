"""Shared encoding for snapshot-bound offset cursors."""

import base64

MAX_CURSOR_LENGTH = 4_096
_SNAPSHOT_DIGEST_BYTES = 12


def encode_fingerprint(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _decode_fingerprint(value: str, *, tool_name: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    try:
        return base64.b64decode(
            value + padding,
            altchars=b"-_",
            validate=True,
        )
    except (ValueError, TypeError) as error:
        raise ValueError(f"Invalid {tool_name} cursor") from error


def encode_offset_cursor(
    version: str,
    query_fingerprint: str,
    snapshot_fingerprint: str,
    offset: int,
) -> str:
    return f"{version}.{query_fingerprint}.{snapshot_fingerprint}.{offset:x}"


def decode_offset_cursor(
    cursor: str | None,
    *,
    version: str,
    query_fingerprint: str,
    tool_name: str,
) -> tuple[str, int]:
    if cursor is None:
        return "", 0
    if len(cursor) > MAX_CURSOR_LENGTH:
        raise ValueError(f"Invalid {tool_name} cursor")
    parts = cursor.split(".")
    if len(parts) != 4 or parts[0] != version or parts[1] != query_fingerprint:
        raise ValueError(f"Invalid or mismatched {tool_name} cursor")
    if (
        len(_decode_fingerprint(parts[2], tool_name=tool_name))
        != _SNAPSHOT_DIGEST_BYTES
    ):
        raise ValueError(f"Invalid {tool_name} cursor")
    try:
        offset = int(parts[3], 16)
    except ValueError as error:
        raise ValueError(f"Invalid {tool_name} cursor") from error
    if offset <= 0:
        raise ValueError(f"Invalid {tool_name} cursor")
    return parts[2], offset

"""Shared model-facing output budget helpers."""

from dataclasses import dataclass


def normalize_utf8(value: str) -> str:
    return value.encode("utf-8", errors="backslashreplace").decode("utf-8")


def truncate_utf8(value: str, max_bytes: int) -> tuple[str, bool]:
    normalized = normalize_utf8(value)
    encoded = normalized.encode("utf-8")
    if len(encoded) <= max_bytes:
        return normalized, normalized != value
    return encoded[:max_bytes].decode("utf-8", errors="ignore"), True


@dataclass(slots=True)
class OutputBudget:
    max_bytes: int
    used_bytes: int = 0

    @property
    def remaining_bytes(self) -> int:
        return max(0, self.max_bytes - self.used_bytes)

    def consume(self, value: str) -> bool:
        size = len(normalize_utf8(value).encode("utf-8"))
        if size > self.remaining_bytes:
            return False
        self.used_bytes += size
        return True

"""Framework-neutral tool configuration values."""

from dataclasses import dataclass, fields
from enum import StrEnum
import math
from numbers import Real


class ToolProfile(StrEnum):
    """Canonical capability profiles exposed by Ghostwheel's tool catalog."""

    READ_ONLY = "read-only"
    SHELL_ONLY = "shell-only"
    FULL = "full"


@dataclass(frozen=True, slots=True)
class ToolLimits:
    """Resource limits shared by all tool implementations.

    ``max_output_bytes`` bounds complete compact model text for ``read``, ``ls``,
    and ``grep``. For other tools it bounds retained variable payload; their
    small structured result envelope and field names are additional.
    """

    max_output_bytes: int = 100_000
    max_read_lines: int = 200
    max_read_scan_bytes: int = 5_000_000
    max_entries: int = 200
    max_directory_scan_entries: int = 10_000
    max_matches: int = 200
    bash_timeout_seconds: float = 30
    max_search_file_bytes: int = 5_000_000
    max_search_total_bytes: int = 50_000_000
    max_search_files: int = 10_000
    search_timeout_seconds: float = 5.0
    regex_timeout_seconds: float = 0.05

    def __post_init__(self) -> None:
        for field in fields(self):
            name = field.name
            value = getattr(self, name)
            if name in _TIMEOUT_LIMIT_FIELDS:
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise ValueError(
                        f"{name} must be a positive and finite real number"
                    )
                try:
                    finite_value = float(value)
                except OverflowError, TypeError, ValueError:
                    raise ValueError(
                        f"{name} must be a positive and finite real number"
                    ) from None
                if not math.isfinite(finite_value) or finite_value <= 0:
                    raise ValueError(
                        f"{name} must be a positive and finite real number"
                    )
            elif isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be positive integer")


_TIMEOUT_LIMIT_FIELDS = frozenset(
    {
        "bash_timeout_seconds",
        "search_timeout_seconds",
        "regex_timeout_seconds",
    }
)


DEFAULT_TOOL_LIMITS = ToolLimits()


__all__ = ["DEFAULT_TOOL_LIMITS", "ToolLimits", "ToolProfile"]

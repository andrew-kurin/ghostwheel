from collections.abc import Iterable
from dataclasses import dataclass
import math
from pathlib import Path

from ghostwheel.tools.command import CommandRunner, LocalCommandRunner
from ghostwheel.tools.workspace import Workspace


@dataclass(frozen=True, slots=True)
class ToolLimits:
    """Resource limits shared by all tool implementations.

    ``max_output_bytes`` bounds retained variable payload. The small structured
    result envelope and field names are not counted against that value.
    """

    max_output_bytes: int = 100_000
    max_entries: int = 200
    max_directory_scan_entries: int = 10_000
    max_matches: int = 200
    bash_timeout_seconds: float = 30
    max_search_file_bytes: int = 5_000_000
    max_search_files: int = 10_000
    regex_timeout_seconds: float = 0.05

    def __post_init__(self) -> None:
        positive_values = {
            "max_output_bytes": self.max_output_bytes,
            "max_entries": self.max_entries,
            "max_directory_scan_entries": self.max_directory_scan_entries,
            "max_matches": self.max_matches,
            "bash_timeout_seconds": self.bash_timeout_seconds,
            "max_search_file_bytes": self.max_search_file_bytes,
            "max_search_files": self.max_search_files,
            "regex_timeout_seconds": self.regex_timeout_seconds,
        }
        for name, value in positive_values.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")


@dataclass(frozen=True, slots=True, init=False)
class ToolDeps:
    """Runtime dependencies for tools.

    ``allowed_roots`` and the scalar limit arguments remain accepted for callers
    using the original API. New composition code should pass ``filesystem_roots``
    or prebuilt ``Workspace`` and ``ToolLimits`` values.
    """

    workspace: Workspace
    limits: ToolLimits
    dry_run: bool
    command_runner: CommandRunner

    def __init__(
        self,
        cwd: Path | str | None = None,
        filesystem_roots: Iterable[Path | str] | None = None,
        max_output_bytes: int | None = None,
        bash_timeout_seconds: float | None = None,
        dry_run: bool = False,
        *,
        max_entries: int | None = None,
        max_directory_scan_entries: int | None = None,
        max_matches: int | None = None,
        max_search_file_bytes: int | None = None,
        max_search_files: int | None = None,
        regex_timeout_seconds: float | None = None,
        allowed_roots: Iterable[Path | str] | None = None,
        workspace: Workspace | None = None,
        limits: ToolLimits | None = None,
        command_runner: CommandRunner | None = None,
    ) -> None:
        if filesystem_roots is not None and allowed_roots is not None:
            raise TypeError("Pass filesystem_roots or allowed_roots, not both")

        if workspace is None:
            if cwd is None:
                raise TypeError("cwd is required when workspace is not provided")
            roots = filesystem_roots if filesystem_roots is not None else allowed_roots
            workspace = Workspace(cwd, roots or ())
        elif (
            cwd is not None or filesystem_roots is not None or allowed_roots is not None
        ):
            raise TypeError(
                "Pass workspace or cwd/filesystem_roots/allowed_roots, not both"
            )

        scalar_limits = (
            max_output_bytes,
            max_entries,
            max_directory_scan_entries,
            max_matches,
            bash_timeout_seconds,
            max_search_file_bytes,
            max_search_files,
            regex_timeout_seconds,
        )
        if limits is not None and any(value is not None for value in scalar_limits):
            raise TypeError("Pass limits or scalar limit arguments, not both")
        if limits is None:
            limits = ToolLimits(
                max_output_bytes=(
                    100_000 if max_output_bytes is None else max_output_bytes
                ),
                max_entries=200 if max_entries is None else max_entries,
                max_directory_scan_entries=(
                    10_000
                    if max_directory_scan_entries is None
                    else max_directory_scan_entries
                ),
                max_matches=200 if max_matches is None else max_matches,
                bash_timeout_seconds=(
                    30 if bash_timeout_seconds is None else bash_timeout_seconds
                ),
                max_search_file_bytes=(
                    5_000_000
                    if max_search_file_bytes is None
                    else max_search_file_bytes
                ),
                max_search_files=(
                    10_000 if max_search_files is None else max_search_files
                ),
                regex_timeout_seconds=(
                    0.05 if regex_timeout_seconds is None else regex_timeout_seconds
                ),
            )

        object.__setattr__(self, "workspace", workspace)
        object.__setattr__(self, "limits", limits)
        object.__setattr__(self, "dry_run", dry_run)
        object.__setattr__(
            self,
            "command_runner",
            command_runner if command_runner is not None else LocalCommandRunner(),
        )

    @property
    def cwd(self) -> Path:
        return self.workspace.cwd

    @property
    def filesystem_roots(self) -> tuple[Path, ...]:
        return self.workspace.filesystem_roots

    @property
    def allowed_roots(self) -> tuple[Path, ...]:
        """Compatibility alias for ``filesystem_roots``."""
        return self.filesystem_roots

    @property
    def max_output_bytes(self) -> int:
        return self.limits.max_output_bytes

    @property
    def max_entries(self) -> int:
        return self.limits.max_entries

    @property
    def max_matches(self) -> int:
        return self.limits.max_matches

    @property
    def bash_timeout_seconds(self) -> float:
        return self.limits.bash_timeout_seconds

    def close(self) -> None:
        self.workspace.close()

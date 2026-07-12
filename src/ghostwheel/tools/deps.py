from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path

from ghostwheel.tool_config import ToolLimits
from ghostwheel.tools.command import CommandRunner, LocalCommandRunner
from ghostwheel.tools.workspace import Workspace


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
        max_read_lines: int | None = None,
        max_read_scan_bytes: int | None = None,
        max_entries: int | None = None,
        max_directory_scan_entries: int | None = None,
        max_matches: int | None = None,
        max_search_file_bytes: int | None = None,
        max_search_total_bytes: int | None = None,
        max_search_files: int | None = None,
        search_timeout_seconds: float | None = None,
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

        scalar_limits = {
            name: value
            for name, value in {
                "max_output_bytes": max_output_bytes,
                "max_read_lines": max_read_lines,
                "max_read_scan_bytes": max_read_scan_bytes,
                "max_entries": max_entries,
                "max_directory_scan_entries": max_directory_scan_entries,
                "max_matches": max_matches,
                "bash_timeout_seconds": bash_timeout_seconds,
                "max_search_file_bytes": max_search_file_bytes,
                "max_search_total_bytes": max_search_total_bytes,
                "max_search_files": max_search_files,
                "search_timeout_seconds": search_timeout_seconds,
                "regex_timeout_seconds": regex_timeout_seconds,
            }.items()
            if value is not None
        }
        if limits is not None and scalar_limits:
            raise TypeError("Pass limits or scalar limit arguments, not both")
        if limits is None:
            limits = replace(ToolLimits(), **scalar_limits)

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
    def max_read_lines(self) -> int:
        return self.limits.max_read_lines

    @property
    def max_read_scan_bytes(self) -> int:
        return self.limits.max_read_scan_bytes

    @property
    def max_entries(self) -> int:
        return self.limits.max_entries

    @property
    def max_matches(self) -> int:
        return self.limits.max_matches

    @property
    def max_search_total_bytes(self) -> int:
        return self.limits.max_search_total_bytes

    @property
    def search_timeout_seconds(self) -> float:
        return self.limits.search_timeout_seconds

    @property
    def bash_timeout_seconds(self) -> float:
        return self.limits.bash_timeout_seconds

    def close(self) -> None:
        self.workspace.close()

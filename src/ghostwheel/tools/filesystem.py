import os
import stat
from enum import Enum

from pydantic import BaseModel, Field
from pydantic_ai import RunContext

from .deps import ToolDeps
from .output import OutputBudget, normalize_utf8, truncate_utf8


class FileContents(BaseModel):
    path: str
    content: str = Field(description="File contents with line numbers prefixed")
    line_count: int
    truncated: bool = Field(
        default=False, description="True if content was truncated due to size"
    )
    total_bytes: int | None = Field(
        default=None, description="Full file size in bytes if truncation occurred"
    )


def read(ctx: RunContext[ToolDeps], path: str) -> FileContents:
    """
    Read the contents of a file and return it with line numbers prefixed.

    Files larger than the configured max_output_bytes will be truncated.
    The 'truncated' field indicates whether the returned content is partial.

    Args:
        path: Path to the file, relative to the working directory.
    """

    max_bytes = ctx.deps.limits.max_output_bytes
    with ctx.deps.workspace.open_file(path) as opened:
        file_size = opened.stat.st_size
        raw_bytes = opened.file.read(max_bytes + 1)
        display_path = normalize_utf8(
            ctx.deps.workspace.display_path(opened.path.absolute)
        )

    raw_truncated = len(raw_bytes) > max_bytes
    raw_bytes = raw_bytes[:max_bytes]

    content = raw_bytes.decode("utf-8", errors="replace")

    lines = content.splitlines()
    numbered = "\n".join(f"{i:4d} | {line}" for i, line in enumerate(lines, 1))
    numbered, numbered_truncated = truncate_utf8(numbered, max_bytes)
    truncated = raw_truncated or numbered_truncated

    return FileContents(
        path=display_path,
        content=numbered,
        line_count=len(lines),
        truncated=truncated,
        total_bytes=file_size if truncated else None,
    )


class FileKind(str, Enum):
    FILE = "file"
    DIRECTORY = "directory"
    SYMLINK = "symlink"


class DirEntry(BaseModel):
    name: str
    type: FileKind = Field(description="'file', 'directory', or 'symlink'")
    size: int | None = Field(
        default=None, description="Size in bytes for files; None for directories"
    )


class DirectoryListing(BaseModel):
    path: str = Field(
        description=("Workspace-relative path, or absolute path for an additional root")
    )
    entries: list[DirEntry]
    truncated: bool = Field(default=False, description="True if results were capped")


def ls(
    ctx: RunContext[ToolDeps], path: str = ".", show_hidden: bool = False
) -> DirectoryListing:
    """List the contents of a directory.

    Args:
        path: Directory to list, relative to the working directory. Defaults to '.'.
        show_hidden: If True, include entries starting with '.'. Defaults to False.

    Returns the directory's contents as a list of entries (name, type, size).
    Results are capped by the configured max_entries limit; if the directory is
    larger, the 'truncated' field will be True.
    """
    entries: list[DirEntry] = []
    truncated = False
    scanned_entries = 0
    output_budget = OutputBudget(ctx.deps.limits.max_output_bytes)

    with ctx.deps.workspace.open_directory(path) as opened:
        display_path = normalize_utf8(
            ctx.deps.workspace.display_path(opened.path.absolute)
        )
        with os.scandir(opened.fd) as children:
            for child in children:
                scanned_entries += 1
                if scanned_entries > ctx.deps.limits.max_directory_scan_entries:
                    truncated = True
                    break
                if not show_hidden and child.name.startswith("."):
                    continue

                if len(entries) >= ctx.deps.limits.max_entries:
                    truncated = True
                    break

                try:
                    child_stat = os.stat(
                        child.name,
                        dir_fd=opened.fd,
                        follow_symlinks=False,
                    )
                except OSError:
                    truncated = True
                    continue
                if stat.S_ISLNK(child_stat.st_mode):
                    entry_type = FileKind.SYMLINK
                    size = None
                elif stat.S_ISDIR(child_stat.st_mode):
                    entry_type = FileKind.DIRECTORY
                    size = None
                else:
                    entry_type = FileKind.FILE
                    size = child_stat.st_size

                display_name = normalize_utf8(child.name)
                if not output_budget.consume(
                    f"{display_name}:{entry_type.value}:{size}"
                ):
                    truncated = True
                    break
                entries.append(DirEntry(name=display_name, type=entry_type, size=size))

    return DirectoryListing(
        path=display_path,
        entries=entries,
        truncated=truncated,
    )

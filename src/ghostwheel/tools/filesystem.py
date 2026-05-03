from pathlib import Path
from pydantic import BaseModel, Field
from pydantic_ai import RunContext
from .deps import ToolDeps


class FileContents(BaseModel):
    path: str
    content: str = Field(description="File contents with line numbers prefixed")
    line_count: int


def read(ctx: RunContext[ToolDeps], path: str) -> FileContents:
    """Read the contents of a file and return it with line numbers prefixed."""
    target = Path(path).expanduser().resolve()

    if not any(target.is_relative_to(root) for root in ctx.deps.allowed_roots):
        raise ValueError(f"Path {target} is outside allowed roots")
    try:
        text = target.read_text()
    except FileNotFoundError:
        return FileContents(
            path=str(target), content="Error: file not found", line_count=0
        )
    except PermissionError:
        return FileContents(
            path=str(target), content="Error: permission denied", line_count=0
        )
    except OSError as exc:
        return FileContents(path=str(target), content=f"Error: {exc}", line_count=0)

    lines = text.splitlines()
    numbered = "\n".join(f"{i:4d} | {line}" for i, line in enumerate(lines, 1))
    return FileContents(path=str(target), content=numbered, line_count=len(lines))


class DirEntry(BaseModel):
    name: str
    type: str = Field(description="'file', 'dir', or 'symlink'")
    size: int | None = Field(
        default=None, description="Size in bytes for files; None for directories"
    )


class DirectoryListing(BaseModel):
    path: str = Field(description="Absolute path that was listed")
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
    Results are capped at 200 entries; if the directory is larger,
    the 'truncated' field will be True.
    """
    target = (ctx.deps.cwd / path).expanduser().resolve()

    # Path safety: must be inside an allowed root
    if not any(target.is_relative_to(root) for root in ctx.deps.allowed_roots):
        raise ValueError(f"Path {target} is outside allowed roots")

    if not target.exists():
        raise FileNotFoundError(f"Path does not exist: {target}")
    if not target.is_dir():
        raise NotADirectoryError(f"Not a directory: {target}")

    MAX_ENTRIES = 200
    entries: list[DirEntry] = []
    truncated = False

    for i, child in enumerate(sorted(target.iterdir())):
        if not show_hidden and child.name.startswith("."):
            continue

        if i >= MAX_ENTRIES:
            truncated = True
            break

        if child.is_symlink():
            entry_type = "symlink"
            size = None
        elif child.is_dir():
            entry_type = "directory"
            size = None
        else:
            entry_type = "file"
            try:
                size = child.stat().st_size
            except OSError:
                size = None

        entries.append(DirEntry(name=child.name, type=entry_type, size=size))

    return DirectoryListing(path=str(target), entries=entries, truncated=truncated)

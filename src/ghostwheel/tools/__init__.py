from ghostwheel.tool_config import ToolLimits, ToolProfile
from ghostwheel.tools.bash import bash
from ghostwheel.tools.catalog import DEFAULT_TOOL_CATALOG, ToolCatalog
from ghostwheel.tools.command import CommandRunner, LocalCommandRunner
from ghostwheel.tools.deps import ToolDeps
from ghostwheel.tools.edit import EditResult, edit
from ghostwheel.tools.listing import FileKind, ls
from ghostwheel.tools.read import read
from ghostwheel.tools.search import grep
from ghostwheel.tools.workspace import Workspace

READ_ONLY_TOOLS = DEFAULT_TOOL_CATALOG.for_profile(ToolProfile.READ_ONLY)
BASH_TOOLS = DEFAULT_TOOL_CATALOG.for_profile(ToolProfile.SHELL_ONLY)
WRITE_TOOLS = DEFAULT_TOOL_CATALOG.write
ALL_TOOLS = DEFAULT_TOOL_CATALOG.for_profile(ToolProfile.FULL)

__all__ = [
    "ALL_TOOLS",
    "BASH_TOOLS",
    "CommandRunner",
    "DEFAULT_TOOL_CATALOG",
    "EditResult",
    "FileKind",
    "LocalCommandRunner",
    "READ_ONLY_TOOLS",
    "ToolCatalog",
    "ToolDeps",
    "ToolLimits",
    "ToolProfile",
    "WRITE_TOOLS",
    "Workspace",
    "bash",
    "edit",
    "grep",
    "ls",
    "read",
]

from ghostwheel.tool_config import ToolLimits, ToolProfile
from ghostwheel.tools.bash import bash
from ghostwheel.tools.catalog import (
    DEFAULT_TOOL_CATALOG,
    ToolCatalog,
    register_tools,
)
from ghostwheel.tools.command import CommandRunner, LocalCommandRunner
from ghostwheel.tools.deps import ToolDeps
from ghostwheel.tools.listing import FileKind, ls
from ghostwheel.tools.read import read
from ghostwheel.tools.search import grep
from ghostwheel.tools.workspace import Workspace

READ_ONLY_TOOLS = DEFAULT_TOOL_CATALOG.for_profile(ToolProfile.READ_ONLY)
BASH_TOOLS = DEFAULT_TOOL_CATALOG.for_profile(ToolProfile.SHELL_ONLY)
ALL_TOOLS = DEFAULT_TOOL_CATALOG.for_profile(ToolProfile.FULL)

__all__ = [
    "ALL_TOOLS",
    "BASH_TOOLS",
    "CommandRunner",
    "DEFAULT_TOOL_CATALOG",
    "FileKind",
    "LocalCommandRunner",
    "READ_ONLY_TOOLS",
    "ToolCatalog",
    "ToolDeps",
    "ToolLimits",
    "ToolProfile",
    "Workspace",
    "bash",
    "grep",
    "ls",
    "read",
    "register_tools",
]

from typing import Any
from pydantic_ai import Agent
from .filesystem import read, ls
from .search import grep
from .bash import bash

READ_ONLY_TOOLS = [read, ls, grep]
BASH_TOOLS = [bash]
ALL_TOOLS = READ_ONLY_TOOLS + BASH_TOOLS


def register_tools(agent: Agent[Any, str], tools: list = ALL_TOOLS):
    for tool in tools:
        agent.tool(tool)

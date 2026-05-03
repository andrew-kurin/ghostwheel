from typing import Any
from pydantic_ai import Agent
from .filesystem import read, ls

READ_ONLY_TOOLS = [read, ls]
ALL_TOOLS = READ_ONLY_TOOLS


def register_tools(agent: Agent[Any, str], tools: list = ALL_TOOLS):
    for tool in tools:
        agent.tool(tool)

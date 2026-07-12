"""Owned agent definitions independent of Pydantic-AI's private internals."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast

from pydantic_ai import Agent, Tool
from pydantic_ai.settings import ModelSettings

from ghostwheel.model_config import ModelSpec
from ghostwheel.providers import build_model

DepsT = TypeVar("DepsT")
OutputT = TypeVar("OutputT")
ToolCallable = Callable[..., object]


@dataclass(frozen=True, slots=True)
class AgentBlueprint(Generic[DepsT, OutputT]):
    """Canonical inputs used to build an SDK agent and estimate static context."""

    model: ModelSpec
    instructions: str
    deps_type: type[DepsT]
    output_type: type[OutputT]
    tools: tuple[Tool[DepsT], ...] = ()
    model_settings: ModelSettings | None = None
    retries: int = 1

    @classmethod
    def from_functions(
        cls,
        *,
        model: ModelSpec,
        instructions: str,
        deps_type: type[DepsT],
        output_type: type[OutputT],
        tools: Sequence[ToolCallable] = (),
        model_settings: ModelSettings | None = None,
        retries: int = 1,
    ) -> AgentBlueprint[DepsT, OutputT]:
        prepared = tuple(
            cast(Tool[DepsT], tool if isinstance(tool, Tool) else Tool(tool))
            for tool in tools
        )
        return cls(
            model=model,
            instructions=instructions,
            deps_type=deps_type,
            output_type=output_type,
            tools=prepared,
            model_settings=model_settings,
            retries=retries,
        )

    def build(self) -> Agent[DepsT, OutputT]:
        return Agent(
            build_model(self.model),
            instructions=self.instructions,
            deps_type=self.deps_type,
            output_type=self.output_type,
            tools=self.tools,
            model_settings=self.model_settings,
            retries=self.retries,
        )

    def static_context_payload(self) -> dict[str, Any]:
        """Return owned prompt/tool data used for conservative token estimation."""

        tools: list[dict[str, Any]] = []
        for tool in self.tools:
            schema = tool.function_schema
            if schema is None:
                raise RuntimeError(f"Tool {tool.name!r} has no static function schema")
            tools.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": schema.json_schema,
                }
            )
        return {"instructions": self.instructions, "tools": tools}

    def static_context_json(self) -> str:
        return json.dumps(
            self.static_context_payload(),
            ensure_ascii=False,
            sort_keys=True,
        )

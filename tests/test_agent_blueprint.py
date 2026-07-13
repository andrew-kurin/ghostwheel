from pydantic_ai.models.test import TestModel

import ghostwheel.agent_blueprint as blueprint_module
from ghostwheel.agent_blueprint import AgentBlueprint
from ghostwheel.agent_factory import chat_agent_blueprint
from ghostwheel.config import Settings
from ghostwheel.tools.catalog import DEFAULT_TOOL_CATALOG
from ghostwheel.tools.deps import ToolDeps
from ghostwheel.tools.edit import edit
from ghostwheel.tools.filesystem import read


def test_blueprint_owns_static_instructions_and_tool_schema(monkeypatch) -> None:
    config = Settings(_env_file=None).resolve()
    blueprint = AgentBlueprint.from_functions(
        model=config.chat_model,
        instructions="Inspect before answering.",
        deps_type=ToolDeps,
        output_type=str,
        tools=(read,),
    )

    payload = blueprint.static_context_payload()

    assert payload["instructions"] == "Inspect before answering."
    assert [tool["name"] for tool in payload["tools"]] == ["read"]
    assert "path" in payload["tools"][0]["parameters"]["properties"]

    monkeypatch.setattr(blueprint_module, "build_model", lambda _spec: TestModel())
    agent = blueprint.build()

    assert agent.deps_type is ToolDeps


def test_blueprint_serialization_is_stable() -> None:
    config = Settings(_env_file=None).resolve()
    blueprint = AgentBlueprint.from_functions(
        model=config.chat_model,
        instructions="instructions",
        deps_type=type(None),
        output_type=str,
    )

    assert blueprint.static_context_json() == (
        '{"instructions": "instructions", "tools": []}'
    )


def test_default_edit_tool_keeps_sequential_execution_in_chat_blueprint() -> None:
    config = Settings(_env_file=None).resolve()
    prepared_edit = DEFAULT_TOOL_CATALOG.write[0]

    blueprint = chat_agent_blueprint(config)
    blueprint_edit = next(tool for tool in blueprint.tools if tool.name == "edit")

    assert blueprint_edit is prepared_edit
    assert blueprint_edit.function is edit
    assert blueprint_edit.sequential is True

from ghostwheel.events import (
    TextOutput,
    ThinkingOutput,
    ToolFailed,
    ToolFinished,
    ToolStarted,
)
from ghostwheel.presentation import (
    TurnState,
    duration,
    format_token_count,
    primary_argument,
)


def test_turn_state_reduces_output_and_correlates_concurrent_tools() -> None:
    ticks = iter((1.0, 2.0, 3.0, 4.0))
    state = TurnState(clock=lambda: next(ticks))

    state.apply(ThinkingOutput("reason "))
    state.apply(ThinkingOutput("continued"))
    state.apply(TextOutput("answer "))
    state.apply(TextOutput("continued"))
    state.apply(ToolStarted("read", "{'path': 'a'}", call_id="a"))
    state.apply(ToolStarted("read", "{'path': 'b'}", call_id="b"))
    state.apply(
        ToolFinished(
            "read",
            "first",
            call_id="a",
            metadata={"summary": "1 replacement"},
        )
    )
    state.apply(ToolFailed("read", "second", call_id="b"))

    assert state.thinking == "reason continued"
    assert state.answer == "answer continued"
    assert [(tool.call_id, tool.status, tool.detail) for tool in state.tools] == [
        ("a", "succeeded", "first"),
        ("b", "failed", "second"),
    ]
    assert state.tools[0].finished_at == 3.0
    assert state.tools[0].metadata == {"summary": "1 replacement"}
    assert state.tools[1].finished_at == 4.0
    assert state.status == "read failed"


def test_turn_state_handles_completion_without_a_start_and_resets() -> None:
    state = TurnState(clock=lambda: 5.0)

    activity = state.apply(ToolFinished("grep", "matches", call_id="missing"))

    assert activity is not None
    assert activity.arguments == ""
    assert activity.status == "succeeded"
    assert activity.metadata is None
    assert state.tools == [activity]

    state.reset("Reviewing…")

    assert state.status == "Reviewing…"
    assert state.answer == ""
    assert state.thinking == ""
    assert state.tools == []


def test_shared_presentation_formatters() -> None:
    assert primary_argument("{'path': 'src/main.py', 'offset': 2}") == "src/main.py"
    assert primary_argument("not  structured") == "not structured"
    assert duration(0.0001) == "<1 ms"
    assert duration(0.25) == "250 ms"
    assert duration(1.25) == "1.2 s"
    assert format_token_count(4_200) == "4.2k"

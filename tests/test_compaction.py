import asyncio
from collections.abc import Sequence
from typing import Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from ghostwheel.compaction import (
    TOOL_RESULT_SUMMARY_LIMIT,
    HistoryCompactor,
    serialize_conversation,
)
from ghostwheel.session import (
    RunOutcome,
    TurnFailed,
    TurnNoResult,
    TurnSucceeded,
    summary_message,
)
from ghostwheel.token_counting import TiktokenTokenCounter


def request(content: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content)])


class FakeRunner:
    def __init__(self, *outcomes: RunOutcome[Any]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, tuple[ModelMessage, ...], type[Any]]] = []

    async def run(
        self,
        prompt: str,
        history: Sequence[ModelMessage],
        *,
        output_type: type[Any],
    ) -> RunOutcome[Any]:
        self.calls.append((prompt, tuple(history), output_type))
        return self.outcomes.pop(0)


def test_serialize_conversation_preserves_roles_tool_ids_and_sorted_arguments() -> None:
    call_id = "call-123"
    messages = (
        request("inspect the repository"),
        ModelResponse(
            parts=[
                ThinkingPart("need to inspect first"),
                ToolCallPart(
                    "read",
                    {"path": "README.md", "line": 2},
                    tool_call_id=call_id,
                ),
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    "read",
                    {"text": "contents", "ok": True},
                    tool_call_id=call_id,
                ),
                RetryPromptPart(
                    "outside workspace",
                    tool_name="read",
                    tool_call_id="call-456",
                ),
            ]
        ),
        ModelResponse(parts=[TextPart("finished")]),
    )

    serialized = serialize_conversation(messages)

    assert "[User]: inspect the repository" in serialized
    assert "[Assistant thinking]: need to inspect first" in serialized
    assert (
        '[Assistant tool call: read (call-123)]: {"line": 2, "path": "README.md"}'
    ) in serialized
    assert (
        '[Tool result: read (call-123)]: {"ok": true, "text": "contents"}'
    ) in serialized
    assert "[Tool/model retry: read (call-456)]: outside workspace" in serialized
    assert "[Assistant]: finished" in serialized


def test_serialize_conversation_truncates_large_tool_results_with_exact_marker() -> (
    None
):
    extra = 37
    content = "x" * (TOOL_RESULT_SUMMARY_LIMIT + extra)
    messages = (
        ModelRequest(
            parts=[ToolReturnPart("bash", content, tool_call_id="call-large")]
        ),
    )

    serialized = serialize_conversation(messages)

    assert "x" * TOOL_RESULT_SUMMARY_LIMIT in serialized
    assert f"[… {extra} characters truncated]" in serialized
    assert "x" * (TOOL_RESULT_SUMMARY_LIMIT + 1) not in serialized


def test_serialize_conversation_keeps_tool_result_at_limit_unchanged() -> None:
    content = "x" * TOOL_RESULT_SUMMARY_LIMIT
    serialized = serialize_conversation(
        (
            ModelRequest(
                parts=[ToolReturnPart("bash", content, tool_call_id="call-limit")]
            ),
        )
    )

    assert content in serialized
    assert "characters truncated" not in serialized


def test_serialize_empty_conversation_has_explicit_marker() -> None:
    assert serialize_conversation(()) == "[No messages]"


def test_history_compactor_passes_rolling_summary_and_serialized_transcript() -> None:
    runner = FakeRunner(
        TurnSucceeded("  updated checkpoint\n", (request("raw compactor turn"),))
    )
    compactor = HistoryCompactor(runner)
    messages = (request("newly evicted"), ModelResponse(parts=[TextPart("done")]))

    outcome = asyncio.run(compactor.summarize("previous checkpoint", messages))

    assert outcome == TurnSucceeded("updated checkpoint", ())
    assert len(runner.calls) == 1
    prompt, history, output_type = runner.calls[0]
    assert history == ()
    assert output_type is str
    assert "<previous-summary>\nprevious checkpoint\n</previous-summary>" in prompt
    assert prompt.count("previous checkpoint") == 1
    assert "[User]: newly evicted" in prompt
    assert "[Assistant]: done" in prompt
    assert "raw compactor turn" not in prompt


def test_history_compactor_marks_absent_previous_summary() -> None:
    runner = FakeRunner(TurnSucceeded("first checkpoint", ()))
    compactor = HistoryCompactor(runner)

    outcome = asyncio.run(compactor.summarize(None, (request("first"),)))

    assert isinstance(outcome, TurnSucceeded)
    assert "<previous-summary>None</previous-summary>" in runner.calls[0][0]


def test_history_compactor_does_not_send_empty_history_over_input_budget() -> None:
    runner = FakeRunner(TurnSucceeded("should not run", ()))
    compactor = HistoryCompactor(runner, input_token_budget=1)

    outcome = asyncio.run(compactor.summarize("existing summary", ()))

    assert isinstance(outcome, TurnFailed)
    assert runner.calls == []


def test_history_compactor_converts_blank_success_to_no_result() -> None:
    runner = FakeRunner(TurnSucceeded(" \n ", (request("raw"),)))
    compactor = HistoryCompactor(runner)

    outcome = asyncio.run(compactor.summarize(None, (request("history"),)))

    assert isinstance(outcome, TurnNoResult)
    assert outcome.message == "Compaction completed without a summary."


def test_history_compactor_hard_caps_summary_with_library_tokenizer() -> None:
    counter = TiktokenTokenCounter()
    runner = FakeRunner(TurnSucceeded("summary detail " * 500, ()))
    compactor = HistoryCompactor(
        runner,
        token_counter=counter,
        summary_token_limit=20,
    )

    outcome = asyncio.run(
        compactor.summarize(None, (request("history"),), target_tokens=7)
    )

    assert isinstance(outcome, TurnSucceeded)
    empty_tokens = counter.count_messages((summary_message(""),))
    assert (
        counter.count_messages((summary_message(outcome.output),)) - empty_tokens <= 7
    )
    assert "under 7 tokens" in runner.calls[0][0]


def test_history_compactor_caps_escaped_summary_in_message_token_domain() -> None:
    counter = TiktokenTokenCounter()
    runner = FakeRunner(TurnSucceeded(('"\n' * 500), ()))
    compactor = HistoryCompactor(
        runner,
        token_counter=counter,
        summary_token_limit=100,
    )

    outcome = asyncio.run(
        compactor.summarize(None, (request("history"),), target_tokens=12)
    )

    assert isinstance(outcome, TurnSucceeded)
    empty_tokens = counter.count_messages((summary_message(""),))
    assert (
        counter.count_messages((summary_message(outcome.output),)) - empty_tokens <= 12
    )


def test_history_compactor_never_commits_an_empty_post_cap_summary() -> None:
    runner = FakeRunner(TurnSucceeded("\x00", ()))
    compactor = HistoryCompactor(
        runner,
        token_counter=TiktokenTokenCounter(),
        summary_token_limit=2,
    )

    outcome = asyncio.run(
        compactor.summarize(None, (request("history"),), target_tokens=1)
    )

    assert isinstance(outcome, TurnNoResult)


def test_history_compactor_propagates_runner_failure() -> None:
    failure = TurnFailed(RuntimeError("model unavailable"))
    runner = FakeRunner(failure)
    compactor = HistoryCompactor(runner)

    outcome = asyncio.run(compactor.summarize("old", (request("history"),)))

    assert outcome is failure


def test_history_compactor_token_bounds_an_oversized_single_message() -> None:
    counter = TiktokenTokenCounter()
    runner = FakeRunner(TurnSucceeded("bounded checkpoint", ()))
    compactor = HistoryCompactor(
        runner,
        token_counter=counter,
        input_token_budget=800,
        summary_token_limit=100,
    )
    oversized = ModelResponse(parts=[TextPart("unique-detail " * 5_000)])

    outcome = asyncio.run(compactor.summarize(None, (oversized,)))

    assert isinstance(outcome, TurnSucceeded)
    prompt = runner.calls[0][0]
    prompt_message = ModelRequest(parts=[UserPromptPart(prompt)])
    assert counter.count_messages((prompt_message,)) <= 800
    assert len(prompt) < len(str(oversized.parts[0].content))


def test_history_compactor_chunks_messages_and_rolls_summary_between_calls() -> None:
    counter = TiktokenTokenCounter()
    first = ModelResponse(parts=[TextPart("alpha " * 250)])
    second = ModelResponse(parts=[TextPart("beta " * 250)])
    probe = HistoryCompactor(FakeRunner(), token_counter=counter)
    one_message_budget = probe._count_prompt(
        probe._build_prompt(None, serialize_conversation((first,)))
    )
    assert (
        probe._count_prompt(
            probe._build_prompt(None, serialize_conversation((first, second)))
        )
        > one_message_budget
    )
    runner = FakeRunner(
        TurnSucceeded("checkpoint one", ()),
        TurnSucceeded("checkpoint two", ()),
    )
    compactor = HistoryCompactor(
        runner,
        token_counter=counter,
        input_token_budget=one_message_budget,
        summary_token_limit=100,
    )

    outcome = asyncio.run(compactor.summarize(None, (first, second)))

    assert outcome == TurnSucceeded("checkpoint two", ())
    assert len(runner.calls) == 2
    assert "checkpoint one" in runner.calls[1][0]
    for prompt, _, _ in runner.calls:
        prompt_message = ModelRequest(parts=[UserPromptPart(prompt)])
        assert counter.count_messages((prompt_message,)) <= one_message_budget


def test_history_compactor_never_chunks_between_tool_call_and_result() -> None:
    counter = TiktokenTokenCounter()
    call_id = "paired-call"
    call = ModelResponse(
        parts=[ToolCallPart("read", {"path": "README.md"}, tool_call_id=call_id)]
    )
    result = ModelRequest(
        parts=[ToolReturnPart("read", "result detail " * 100, tool_call_id=call_id)]
    )
    tail = ModelResponse(parts=[TextPart("finished")])
    probe = HistoryCompactor(FakeRunner(), token_counter=counter)
    call_budget = probe._count_prompt(
        probe._build_prompt(None, serialize_conversation((call,)))
    )
    pair_tokens = probe._count_prompt(
        probe._build_prompt(None, serialize_conversation((call, result)))
    )
    budget = call_budget + 80
    assert budget < pair_tokens
    runner = FakeRunner(
        TurnSucceeded("tool checkpoint", ()),
        TurnSucceeded("final checkpoint", ()),
    )
    compactor = HistoryCompactor(
        runner,
        token_counter=counter,
        input_token_budget=budget,
        summary_token_limit=100,
    )

    outcome = asyncio.run(compactor.summarize(None, (call, result, tail)))

    assert outcome == TurnSucceeded("final checkpoint", ())
    assert len(runner.calls) == 2
    assert "[Assistant tool call: read (paired-call)]" in runner.calls[0][0]
    assert 'result "read" ("paired-call")' in runner.calls[0][0]
    assert "[Assistant]: finished" in runner.calls[1][0]


def test_oversized_tool_atom_preserves_call_and_result_markers() -> None:
    counter = TiktokenTokenCounter()
    call_id = "large-call"
    call = ModelResponse(
        parts=[
            ToolCallPart(
                "bash",
                {"command": "x" * 20_000},
                tool_call_id=call_id,
            )
        ]
    )
    result = ModelRequest(
        parts=[ToolReturnPart("bash", "IMPORTANT RESULT", tool_call_id=call_id)]
    )
    runner = FakeRunner(TurnSucceeded("checkpoint", ()))
    compactor = HistoryCompactor(
        runner,
        token_counter=counter,
        input_token_budget=800,
        summary_token_limit=100,
    )

    outcome = asyncio.run(compactor.summarize(None, (call, result)))

    assert isinstance(outcome, TurnSucceeded)
    prompt = runner.calls[0][0]
    assert "[Assistant tool call: bash (large-call)]" in prompt
    assert "[Tool result end: bash (large-call)]" in prompt
    prompt_message = ModelRequest(parts=[UserPromptPart(prompt)])
    assert counter.count_messages((prompt_message,)) <= 800


def test_oversized_parallel_tool_atom_preserves_every_pair_in_manifest() -> None:
    counter = TiktokenTokenCounter()
    call = ModelResponse(
        parts=[
            ToolCallPart("bash", {"command": "a" * 20_000}, tool_call_id="id1"),
            ToolCallPart("bash", {"command": "b" * 20_000}, tool_call_id="id2"),
        ]
    )
    results = ModelRequest(
        parts=[
            ToolReturnPart("bash", "first", tool_call_id="id1"),
            ToolReturnPart("bash", "second", tool_call_id="id2"),
        ]
    )
    runner = FakeRunner(TurnSucceeded("checkpoint", ()))
    compactor = HistoryCompactor(
        runner,
        token_counter=counter,
        input_token_budget=800,
        summary_token_limit=100,
    )

    outcome = asyncio.run(compactor.summarize(None, (call, results)))

    assert isinstance(outcome, TurnSucceeded)
    prompt = runner.calls[0][0]
    assert 'call "bash" ("id1")' in prompt
    assert 'call "bash" ("id2")' in prompt
    assert 'result "bash" ("id1")' in prompt
    assert 'result "bash" ("id2")' in prompt
    prompt_message = ModelRequest(parts=[UserPromptPart(prompt)])
    assert counter.count_messages((prompt_message,)) <= 800


def test_tool_manifest_bounds_long_provider_ids() -> None:
    counter = TiktokenTokenCounter()
    first_id = "😀" * 1_000
    second_id = "🧪" * 1_000
    call = ModelResponse(
        parts=[
            ToolCallPart("bash", {"command": "a" * 10_000}, tool_call_id=first_id),
            ToolCallPart("bash", {"command": "b" * 10_000}, tool_call_id=second_id),
        ]
    )
    results = ModelRequest(
        parts=[
            ToolReturnPart("bash", "first", tool_call_id=first_id),
            ToolReturnPart("bash", "second", tool_call_id=second_id),
        ]
    )
    runner = FakeRunner(TurnSucceeded("checkpoint", ()))
    compactor = HistoryCompactor(
        runner,
        token_counter=counter,
        input_token_budget=800,
        summary_token_limit=100,
    )

    outcome = asyncio.run(compactor.summarize(None, (call, results)))

    assert isinstance(outcome, TurnSucceeded)
    assert len(runner.calls) == 1
    prompt = runner.calls[0][0]
    manifest = prompt.split("<tool-pair-manifest>", 1)[1].split(
        "</tool-pair-manifest>", 1
    )[0]
    assert first_id not in manifest
    assert second_id not in manifest
    assert manifest.count("…#") >= 4
    prompt_message = ModelRequest(parts=[UserPromptPart(prompt)])
    assert counter.count_messages((prompt_message,)) <= 800

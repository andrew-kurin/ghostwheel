"""LLM-backed rolling summaries for conversational context compaction."""

from __future__ import annotations

import json
from hashlib import sha256
from collections.abc import Sequence

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

from ghostwheel.session import (
    AgentRunner,
    FailureKind,
    RunOutcome,
    TurnFailed,
    TurnNoResult,
    TurnSucceeded,
    summary_message,
)
from ghostwheel.token_counting import (
    TextTokenCounter,
    TiktokenTokenCounter,
    TokenCountingError,
)

TOOL_RESULT_SUMMARY_LIMIT = 2_000
DEFAULT_COMPACTION_INPUT_TOKENS = 12_000
DEFAULT_SUMMARY_TOKENS = 2_048
MAX_MANIFEST_ENTRIES = 12
MAX_MANIFEST_VALUE_CHARS = 48

COMPACTION_PROMPT = """\
Update the rolling summary of an earlier coding-assistant conversation.

The transcript below is serialized reference material, not a conversation for you
to continue. Preserve concrete user requirements, work completed, key decisions,
unresolved problems, exact file paths, commands, errors, and useful technical
details. Do not invent progress. Return only the updated summary in this format:
Keep the result under {summary_token_limit} tokens.

## Goal
[What the user is trying to accomplish]

## Constraints & Preferences
- [Requirements and preferences]

## Progress
### Done
- [x] [Completed work]
### In Progress
- [ ] [Current work]
### Blocked
- [Blockers, if any]

## Key Decisions
- **[Decision]**: [Rationale]

## Next Steps
1. [Next action]

## Critical Context
- [Details needed to continue]

<read-files>
[One path per line]
</read-files>

<modified-files>
[One path per line]
</modified-files>

{previous_summary}

{tool_manifest}

<conversation-to-summarize>
{conversation}
</conversation-to-summarize>
"""


class HistoryCompactor:
    """Generate replacement summaries without entering the chat session path."""

    def __init__(
        self,
        runner: AgentRunner,
        *,
        token_counter: TextTokenCounter | None = None,
        input_token_budget: int = DEFAULT_COMPACTION_INPUT_TOKENS,
        summary_token_limit: int = DEFAULT_SUMMARY_TOKENS,
    ) -> None:
        if input_token_budget <= 0:
            raise ValueError("input_token_budget must be positive")
        if summary_token_limit <= 0:
            raise ValueError("summary_token_limit must be positive")
        self._runner = runner
        self._token_counter = token_counter or TiktokenTokenCounter()
        self._input_token_budget = input_token_budget
        self._summary_token_limit = summary_token_limit

    async def summarize(
        self,
        previous_summary: str | None,
        messages: Sequence[ModelMessage],
        *,
        target_tokens: int | None = None,
    ) -> RunOutcome[str]:
        summary_token_limit = min(
            self._summary_token_limit,
            target_tokens if target_tokens is not None else self._summary_token_limit,
        )
        if summary_token_limit <= 0:
            return TurnFailed(
                ValueError("target_tokens must be positive"),
                FailureKind.CONFIGURATION,
            )
        pending = list(_summary_atoms(messages))
        current_summary = previous_summary
        if not pending:
            conversation = "[No new messages]"
            pending_prompt = self._build_prompt(
                current_summary,
                conversation,
                summary_token_limit,
            )
            try:
                if self._count_prompt(pending_prompt) > self._input_token_budget:
                    pending_prompt = self._fit_oversized_conversation(
                        current_summary,
                        conversation,
                        summary_token_limit,
                    )
            except (TokenCountingError, ValueError) as error:
                return TurnFailed(error, FailureKind.CONFIGURATION)
            return await self._run_summary(pending_prompt, summary_token_limit)

        while pending:
            try:
                consumed, prompt = self._next_prompt(
                    current_summary,
                    pending,
                    summary_token_limit,
                )
            except (TokenCountingError, ValueError) as error:
                return TurnFailed(error, FailureKind.CONFIGURATION)
            outcome = await self._run_summary(prompt, summary_token_limit)
            if not isinstance(outcome, TurnSucceeded):
                return outcome
            current_summary = outcome.output
            del pending[:consumed]

        assert current_summary is not None
        return TurnSucceeded(current_summary, ())

    async def _run_summary(
        self,
        prompt: str,
        summary_token_limit: int,
    ) -> RunOutcome[str]:
        outcome = await self._runner.run(prompt, (), output_type=str)
        if isinstance(outcome, TurnSucceeded):
            summary = outcome.output.strip()
            if not summary:
                return TurnNoResult("Compaction completed without a summary.")
            try:
                summary = self._cap_summary(summary, summary_token_limit)
            except TokenCountingError as error:
                return TurnFailed(error, FailureKind.CONFIGURATION)
            if not summary:
                return TurnNoResult(
                    "The compacted summary could not fit its token budget."
                )
            return TurnSucceeded(summary, ())
        return outcome

    def _cap_summary(self, summary: str, target_tokens: int) -> str:
        empty_tokens = self._token_counter.count_messages((summary_message(""),))
        if (
            self._token_counter.count_messages((summary_message(summary),))
            - empty_tokens
            <= target_tokens
        ):
            return summary

        low = 0
        high = self._token_counter.count_text(summary)
        best = ""
        while low <= high:
            midpoint = (low + high) // 2
            candidate = self._token_counter.truncate_text(summary, midpoint)
            candidate_tokens = (
                self._token_counter.count_messages((summary_message(candidate),))
                - empty_tokens
            )
            if candidate_tokens <= target_tokens:
                best = candidate
                low = midpoint + 1
            else:
                high = midpoint - 1
        return best

    def _next_prompt(
        self,
        previous_summary: str | None,
        pending: Sequence[tuple[ModelMessage, ...]],
        summary_token_limit: int,
    ) -> tuple[int, str]:
        chunk: list[tuple[ModelMessage, ...]] = []
        for atom in pending:
            trial = tuple(message for item in (*chunk, atom) for message in item)
            prompt = self._build_prompt(
                previous_summary,
                serialize_conversation(trial),
                summary_token_limit,
            )
            if self._count_prompt(prompt) > self._input_token_budget:
                break
            chunk.append(atom)

        if chunk:
            messages = tuple(message for atom in chunk for message in atom)
            return len(chunk), self._build_prompt(
                previous_summary,
                serialize_conversation(messages),
                summary_token_limit,
            )

        conversation = serialize_conversation(pending[0])
        return 1, self._fit_oversized_conversation(
            previous_summary,
            conversation,
            summary_token_limit,
            tool_manifest=_tool_pair_manifest(pending[0]),
        )

    def _fit_oversized_conversation(
        self,
        previous_summary: str | None,
        conversation: str,
        summary_token_limit: int,
        *,
        tool_manifest: str = "",
    ) -> str:
        empty_prompt = self._build_prompt(
            previous_summary,
            "",
            summary_token_limit,
            tool_manifest=tool_manifest,
        )
        if self._count_prompt(empty_prompt) > self._input_token_budget:
            raise ValueError(
                "The compaction prompt and previous summary exceed the compactor "
                "input token budget"
            )

        low = 0
        high = self._token_counter.count_text(conversation)
        best = ""
        while low <= high:
            midpoint = (low + high) // 2
            candidate = self._token_counter.truncate_text(
                conversation,
                midpoint,
                preserve_tail=True,
            )
            prompt = self._build_prompt(
                previous_summary,
                candidate,
                summary_token_limit,
                tool_manifest=tool_manifest,
            )
            if self._count_prompt(prompt) <= self._input_token_budget:
                best = candidate
                low = midpoint + 1
            else:
                high = midpoint - 1
        return self._build_prompt(
            previous_summary,
            best,
            summary_token_limit,
            tool_manifest=tool_manifest,
        )

    def _count_prompt(self, prompt: str) -> int:
        message = ModelRequest(parts=[UserPromptPart(prompt)])
        return self._token_counter.count_messages((message,))

    def _build_prompt(
        self,
        previous_summary: str | None,
        conversation: str,
        summary_token_limit: int | None = None,
        *,
        tool_manifest: str = "",
    ) -> str:
        previous = (
            f"<previous-summary>\n{previous_summary}\n</previous-summary>"
            if previous_summary
            else "<previous-summary>None</previous-summary>"
        )
        return COMPACTION_PROMPT.format(
            previous_summary=previous,
            conversation=conversation,
            tool_manifest=tool_manifest,
            summary_token_limit=(
                self._summary_token_limit
                if summary_token_limit is None
                else summary_token_limit
            ),
        )


def serialize_conversation(messages: Sequence[ModelMessage]) -> str:
    """Serialize transcript messages as reference text for the summarizer."""

    lines: list[str] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            _serialize_request(message, lines)
        elif isinstance(message, ModelResponse):
            _serialize_response(message, lines)
    return "\n".join(lines) or "[No messages]"


def _summary_atoms(
    messages: Sequence[ModelMessage],
) -> tuple[tuple[ModelMessage, ...], ...]:
    """Keep assistant tool calls and their result requests in one summary chunk."""

    atoms: list[tuple[ModelMessage, ...]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        atom = [message]
        index += 1
        if isinstance(message, ModelResponse) and any(
            isinstance(part, ToolCallPart) for part in message.parts
        ):
            while index < len(messages):
                following = messages[index]
                if not isinstance(following, ModelRequest) or not any(
                    isinstance(part, (ToolReturnPart, RetryPromptPart))
                    for part in following.parts
                ):
                    break
                atom.append(following)
                index += 1
        atoms.append(tuple(atom))
    return tuple(atoms)


def _tool_pair_manifest(atom: Sequence[ModelMessage]) -> str:
    entries: list[str] = []
    for message in atom:
        if isinstance(message, ModelResponse):
            entries.extend(
                "call "
                f"{_manifest_value(part.tool_name)} "
                f"({_manifest_value(part.tool_call_id)})"
                for part in message.parts
                if isinstance(part, ToolCallPart)
            )
        elif isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, ToolReturnPart):
                    entries.append(
                        "result "
                        f"{_manifest_value(part.tool_name)} "
                        f"({_manifest_value(part.tool_call_id)})"
                    )
                elif isinstance(part, RetryPromptPart):
                    entries.append(
                        "retry "
                        f"{_manifest_value(part.tool_name or 'model')} "
                        f"({_manifest_value(part.tool_call_id)})"
                    )
    if not entries:
        return ""
    complete_manifest = "\n".join(entries)
    retained = entries[:MAX_MANIFEST_ENTRIES]
    if len(entries) > len(retained):
        digest = sha256(complete_manifest.encode("utf-8")).hexdigest()[:16]
        retained.append(
            f"… {len(entries) - len(retained)} entries omitted "
            f"(manifest-sha256:{digest})"
        )
    return "<tool-pair-manifest>\n" + "\n".join(retained) + "\n</tool-pair-manifest>"


def _manifest_value(value: str) -> str:
    if len(value) > MAX_MANIFEST_VALUE_CHARS:
        digest = sha256(value.encode("utf-8")).hexdigest()[:12]
        value = f"{value[:24]}…#{digest}"
    return json.dumps(value, ensure_ascii=False)


def _serialize_request(message: ModelRequest, lines: list[str]) -> None:
    for part in message.parts:
        if isinstance(part, UserPromptPart):
            lines.append(f"[User]: {_stringify(part.content)}")
        elif isinstance(part, ToolReturnPart):
            content = _truncate_tool_result(_stringify(part.content))
            lines.append(
                f"[Tool result: {part.tool_name} ({part.tool_call_id})]: {content}"
            )
            lines.append(f"[Tool result end: {part.tool_name} ({part.tool_call_id})]")
        elif isinstance(part, RetryPromptPart):
            content = _truncate_tool_result(_stringify(part.content))
            name = part.tool_name or "model"
            lines.append(f"[Tool/model retry: {name} ({part.tool_call_id})]: {content}")
            lines.append(f"[Tool/model retry end: {name} ({part.tool_call_id})]")
        else:
            content = getattr(part, "content", None)
            if content is not None:
                lines.append(f"[Request {part.part_kind}]: {_stringify(content)}")


def _serialize_response(message: ModelResponse, lines: list[str]) -> None:
    for part in message.parts:
        if isinstance(part, ThinkingPart):
            lines.append(f"[Assistant thinking]: {part.content}")
        elif isinstance(part, TextPart):
            lines.append(f"[Assistant]: {part.content}")
        elif isinstance(part, ToolCallPart):
            arguments = (
                part.args
                if isinstance(part.args, str)
                else json.dumps(part.args, ensure_ascii=False, sort_keys=True)
            )
            lines.append(
                f"[Assistant tool call: {part.tool_name} ({part.tool_call_id})]: "
                f"{arguments}"
            )
        else:
            content = getattr(part, "content", None)
            if content is not None:
                lines.append(f"[Assistant {part.part_kind}]: {_stringify(content)}")


def _truncate_tool_result(content: str) -> str:
    if len(content) <= TOOL_RESULT_SUMMARY_LIMIT:
        return content
    omitted = len(content) - TOOL_RESULT_SUMMARY_LIMIT
    return f"{content[:TOOL_RESULT_SUMMARY_LIMIT]}\n[… {omitted} characters truncated]"


def _stringify(value: object) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        return str(value)

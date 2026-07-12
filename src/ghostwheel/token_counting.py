"""Local token estimates for providers without a preflight counting API."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import tiktoken
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter


class TokenCounter(Protocol):
    """Count the estimated provider tokens represented by model messages."""

    def count_messages(self, messages: Sequence[ModelMessage]) -> int: ...


class TextTokenCounter(TokenCounter, Protocol):
    """Count and truncate plain text with the same library tokenizer."""

    def count_text(self, value: str) -> int: ...

    def truncate_text(
        self,
        value: str,
        max_tokens: int,
        *,
        preserve_tail: bool = False,
    ) -> str: ...


class TokenCountingError(RuntimeError):
    """Raised when the configured tokenizer cannot be loaded."""


_NON_PROMPT_FIELD_NAMES = {
    "usage",
    "model_name",
    "timestamp",
    "provider_name",
    "provider_url",
    "provider_details",
    "provider_response_id",
    "finish_reason",
    "run_id",
    "conversation_id",
    "metadata",
}
_NON_PROMPT_FIELDS = {
    "__all__": {
        **{name: True for name in _NON_PROMPT_FIELD_NAMES},
        "parts": {
            "__all__": {name: True for name in _NON_PROMPT_FIELD_NAMES},
        },
    }
}


@dataclass(frozen=True, slots=True)
class TiktokenTokenCounter:
    """Estimate local-model tokens with a stable library tokenizer.

    Ollama and OpenAI-compatible llama.cpp models do not expose a common
    preflight token-counting API. The estimate tokenizes the semantic Pydantic
    AI message representation while excluding response metadata that is not
    sent back to the model.
    """

    encoding_name: str = "o200k_base"
    _encoding: tiktoken.Encoding | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def count_messages(self, messages: Sequence[ModelMessage]) -> int:
        if not messages:
            return 0
        serialized = ModelMessagesTypeAdapter.dump_json(
            list(messages),
            exclude=_NON_PROMPT_FIELDS,
            exclude_defaults=True,
            exclude_none=True,
        ).decode("utf-8")
        return self.count_text(serialized)

    def count_text(self, value: str) -> int:
        if not value:
            return 0
        return len(self._get_encoding().encode(value, disallowed_special=()))

    def truncate_text(
        self,
        value: str,
        max_tokens: int,
        *,
        preserve_tail: bool = False,
    ) -> str:
        if max_tokens <= 0 or not value:
            return ""
        encoding = self._get_encoding()
        tokens = encoding.encode(value, disallowed_special=())
        if len(tokens) <= max_tokens:
            return value
        if preserve_tail and max_tokens >= 8:
            marker = "\n[… truncated …]\n"
            marker_tokens = encoding.encode(marker, disallowed_special=())
            content_tokens = max_tokens - len(marker_tokens)
            if content_tokens > 1:
                head_tokens = content_tokens // 2
                tail_tokens = content_tokens - head_tokens
                return (
                    encoding.decode(tokens[:head_tokens])
                    + marker
                    + encoding.decode(tokens[-tail_tokens:])
                )
        return encoding.decode(tokens[:max_tokens])

    def _get_encoding(self) -> tiktoken.Encoding:
        if self._encoding is None:
            try:
                encoding = tiktoken.get_encoding(self.encoding_name)
            except Exception as error:
                raise TokenCountingError(
                    f"Unable to load the tiktoken encoding {self.encoding_name!r}"
                ) from error
            object.__setattr__(self, "_encoding", encoding)
        assert self._encoding is not None
        return self._encoding

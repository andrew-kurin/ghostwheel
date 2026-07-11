import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.usage import RequestUsage

from ghostwheel.token_counting import TiktokenTokenCounter, TokenCountingError


def test_tiktoken_counter_counts_semantic_message_content() -> None:
    counter = TiktokenTokenCounter()
    short = ModelRequest(parts=[UserPromptPart("hello")])
    long = ModelRequest(parts=[UserPromptPart("hello " * 100)])

    assert counter.count_messages(()) == 0
    assert counter.count_messages((short,)) > 0
    assert counter.count_messages((long,)) > counter.count_messages((short,))


def test_tiktoken_counter_excludes_response_usage_metadata() -> None:
    counter = TiktokenTokenCounter()
    first = ModelResponse(
        parts=[TextPart("same response")],
        usage=RequestUsage(input_tokens=1, output_tokens=1),
        model_name="first-model",
    )
    second = ModelResponse(
        parts=[TextPart("same response")],
        usage=RequestUsage(input_tokens=100_000, output_tokens=50_000),
        model_name="second-model",
    )

    assert counter.count_messages((first,)) == counter.count_messages((second,))


def test_tiktoken_counter_truncates_text_with_its_library_encoding() -> None:
    counter = TiktokenTokenCounter()
    value = "alpha beta gamma delta epsilon" * 20

    truncated = counter.truncate_text(value, 12)

    assert counter.count_text(truncated) <= 12
    assert value.startswith(truncated)
    assert counter.truncate_text(value, 0) == ""

    balanced = counter.truncate_text(value, 12, preserve_tail=True)
    assert counter.count_text(balanced) <= 12
    assert "truncated" in balanced
    assert balanced.startswith("alpha")
    assert balanced.endswith(value[-5:])


def test_tiktoken_counter_reports_encoding_load_failures() -> None:
    counter = TiktokenTokenCounter("not-an-encoding")
    prompt = ModelRequest(parts=[UserPromptPart("hello")])

    assert counter.count_messages(()) == 0
    with pytest.raises(TokenCountingError, match="not-an-encoding"):
        counter.count_messages((prompt,))

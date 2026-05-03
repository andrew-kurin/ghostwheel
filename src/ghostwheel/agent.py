import logfire
import asyncio
from pathlib import Path
from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.models.openai import OpenAIChatModelSettings
from pydantic_graph import End
from pydantic_ai.messages import (
    PartDeltaEvent,
    TextPart,
    ThinkingPart,
    PartStartEvent,
    ThinkingPartDelta,
    TextPartDelta,
)
from ghostwheel.schemas import ReviewResult, SEVERITY_VALUES
from ghostwheel.rendering import render_review
from ghostwheel.tools.deps import ToolDeps
from ghostwheel.tools import register_tools, READ_ONLY_TOOLS
from rich.console import Console

logfire.configure(console=False)
logfire.instrument_pydantic_ai()

models = {
    "gemma4": "gemma4:26b",
    "glm-flash": "glm-4.7-flash:latest",
}

model = OllamaModel(
    models["gemma4"],
    provider=OllamaProvider(base_url="http://localhost:11434/v1"),
)

reviewer = Agent(
    model,
    instructions=(
        "You are a careful code reviewer. "
        "Provide specific and actionable feedback."
        "Only flag real issues, don't invent nits."
        "Don't invent the review if you haven't read the code."
        "Use read_file tool to read the contents of a file before reviewing."
    ),
    deps_type=ToolDeps,
)
register_tools(reviewer, READ_ONLY_TOOLS)

formatter = Agent(
    model,
    instructions=(
        "You convert a code review written in prose into a structured "
        "ReviewResult object. You are a transcriber, not a reviewer.\n"
        "\n"
        "Rules:\n"
        "- Include every finding from the prose. Do not omit any.\n"
        "- Do not add findings that are not in the prose.\n"
        "- Preserve the reviewer's severity assessments. If the reviewer "
        "calls something a 'bug' or 'error', it is a blocker. If they say "
        "'consider' or 'suggestion', it is a suggestion. Otherwise, warning.\n"
        "- For every finding, set severity to exactly one of: \n"
        f"{SEVERITY_VALUES}\n"
        "- Do not put severity words in category. Category should be a short issue type "
        "like 'bug', 'typing', 'runtime', 'security', 'style', or 'design'.\n"
        "- Copy file paths and line numbers exactly as written in the prose.\n"
        "- The 'message' field should restate the issue concisely.\n"
        "- The 'suggestion' field should contain the fix if the reviewer "
        "proposed one, or be omitted if they did not.\n"
        "- Set 'approve' to true only if the prose explicitly approves the "
        "code or contains no blockers and no warnings.\n"
        "- Write a two-sentence 'summary' that captures the overall verdict."
    ),
    model_settings=OpenAIChatModelSettings({"openai_reasoning_effort": "none"}),
    output_type=ReviewResult,
)


async def main():
    console = Console()

    review_message = "review the code at current folder"
    deps = ToolDeps(cwd=Path.cwd(), allowed_roots=[Path.cwd()])

    status = None

    try:
        async with reviewer.iter(review_message, deps=deps) as run:
            node = run.next_node

            while not isinstance(node, End):
                if Agent.is_model_request_node(node):
                    async with node.stream(run.ctx) as stream:
                        async for event in stream:
                            if isinstance(event, PartStartEvent):
                                part = event.part

                                if isinstance(part, ThinkingPart):
                                    console.print(part.content, end="")

                                elif isinstance(part, TextPart):
                                    if status is None:
                                        status = console.status(
                                            "[bold yellow]Writing review...[/bold yellow]",
                                            spinner="dots",
                                        )
                                        status.start()
                            elif isinstance(event, PartDeltaEvent):
                                if isinstance(event.delta, ThinkingPartDelta):
                                    content_delta = event.delta.content_delta
                                    if content_delta:
                                        console.print(content_delta, end="")
                                elif isinstance(event.delta, TextPartDelta):
                                    if event.delta.content_delta and status is None:
                                        status = console.status(
                                            "[bold yellow]Writing review...[/bold yellow]",
                                            spinner="dots",
                                        )
                                        status.start()
                    node = await run.next(node)
                else:
                    node = await run.next(node)
    finally:
        if status is not None:
            status.stop()

    console.print()

    if run.result is not None:
        structured = await formatter.run(run.result.output)
        render_review(structured.output, console)


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()

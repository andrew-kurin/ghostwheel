import logfire
import asyncio
from pathlib import Path
from pydantic_ai import Agent
from pydantic_ai.messages import (
    PartDeltaEvent,
    TextPart,
    ThinkingPart,
    PartStartEvent,
    ThinkingPartDelta,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.models.openai import OpenAIChatModelSettings
from schemas import FileContents, ReviewResult
from rendering import render_review
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
)

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


@reviewer.tool_plain
def read_file(path: str) -> FileContents:
    """Read the contents of a file and return it with line numbers prefixed."""
    p = Path(path).expanduser().resolve()
    try:
        text = p.read_text()
    except FileNotFoundError:
        return FileContents(path=str(p), content="Error: file not found", line_count=0)
    except PermissionError:
        return FileContents(
            path=str(p), content="Error: permission denied", line_count=0
        )
    lines = text.splitlines()
    numbered = "\n".join(f"{i:4d} | {line}" for i, line in enumerate(lines, 1))
    return FileContents(path=str(p), content=numbered, line_count=len(lines))


def debug_run(run, console):
    # After the run:
    for msg in run.result.all_messages():
        for part in msg.parts:
            kind = type(part).__name__
            if isinstance(part, ToolCallPart):
                console.print(
                    f"[yellow]🔧 {kind}: {part.tool_name}({part.args})[/yellow]"
                )
            elif isinstance(part, ToolReturnPart):
                console.print(
                    f"[green]← {kind}: {part.tool_name} → {str(part.content)[:200]}[/green]"
                )
            elif isinstance(part, ThinkingPart):
                console.print(f"[dim]💭 {part.content[:200]}...[/dim]")
            elif isinstance(part, TextPart):
                console.print(f"[blue]💬 {part.content[:200]}...[/blue]")


async def main():
    console = Console()
    text_started = False
    status = None
    file_paths = ["agent.py", "rendering.py", "schemas.py"]
    review_message = f"review the files at {", ".join(file_paths)}"
    async with reviewer.iter(review_message) as run:
        try:
            async for node in run:
                if Agent.is_model_request_node(node):
                    async with node.stream(run.ctx) as stream:
                        async for event in stream:
                            if isinstance(event, PartStartEvent):
                                part = event.part
                                if isinstance(part, ThinkingPart):
                                    console.print(f"\n💭 {part.content}", end="")
                                elif isinstance(part, TextPart):
                                    if not text_started:
                                        text_started = True
                                        status = console.status(
                                            "[bold yellow]Writing review...[/bold yellow]",
                                            spinner="dots",
                                        )
                                        status.start()
                            elif isinstance(event, PartDeltaEvent):
                                if isinstance(event.delta, ThinkingPartDelta):
                                    console.print(event.delta.content_delta, end="")
        finally:
            if status is not None:
                status.stop()

        console.print()

        if run.result is not None:
            structured = await formatter.run(run.result.output)
            render_review(structured.output, console)


asyncio.run(main())

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
from pydantic_ai.output import NativeOutput
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider
from schemas import FileContents, ReviewResult
from rendering import render_review
from rich.console import Console

models = {"gemma": "gemma4:26b", "qwen": "batiai/qwen3.6-35b:iq4"}

model = OllamaModel(
    models["gemma"],
    provider=OllamaProvider(base_url="http://localhost:11434/v1"),
)

agent = Agent(
    model,
    instructions=(
        "You are a careful code reviewer. "
        "Provide specific and actionable feedback."
        "Only flag real issues, don't invent nits."
        "Don't invent the review if you haven't read the code."
        "Use read_file tool to read the contents of a file before reviewing."
    ),
    output_type=NativeOutput(ReviewResult),
)


@agent.tool_plain
def read_file(path: str) -> FileContents:
    """Read the contents of a file and return it with line numbers prefixed."""
    p = Path(path).expanduser().resolve()
    text = p.read_text()
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
    file_paths = ["agent.py", "rendering.py", "schemas.py"]
    review_message = f"review the files at {", ".join(file_paths)}"
    async with agent.iter(review_message) as run:
        async for node in run:
            if Agent.is_model_request_node(node):
                async with node.stream(run.ctx) as stream:
                    async for event in stream:
                        if isinstance(event, PartStartEvent):
                            part = event.part
                            if isinstance(part, ThinkingPart):
                                print(f"\n💭 {part.content}", end="", flush=True)
                            elif isinstance(part, TextPart):
                                pass
                        elif isinstance(event, PartDeltaEvent):
                            if isinstance(event.delta, ThinkingPartDelta):
                                print(event.delta.content_delta, end="", flush=True)

        result: ReviewResult | None = (
            run.result.output if run.result is not None else None
        )
        console = Console()
        console.print()
        if result is not None:
            render_review(result, console)

        debug_run(run, console)


asyncio.run(main())

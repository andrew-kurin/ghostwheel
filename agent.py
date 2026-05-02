import asyncio
from pydantic_ai import Agent
from pydantic_ai.messages import (
    PartDeltaEvent,
    TextPart,
    ThinkingPart,
    PartStartEvent,
    ThinkingPartDelta,
)
from pydantic_ai.output import NativeOutput
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider
from schemas import ReviewResult
from rendering import render_review
from rich.console import Console

model = OllamaModel(
    "gemma4:26b",
    provider=OllamaProvider(base_url="http://localhost:11434/v1"),
)

agent = Agent(
    model,
    instructions=(
        "You are a careful code reviewer. "
        "Provide specific and actionable feedback."
        "Only flag real issues, don't nitpick."
    ),
    output_type=NativeOutput(ReviewResult),
)


async def main():
    with open("agent.py") as fh:
        file_content: str = fh.read()
    async with agent.iter(f"file_path: agent.py\ncontent:\n{file_content}") as run:
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
        if result is not None:
            console = Console()
            console.print()
            render_review(result, console)


asyncio.run(main())

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
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ToolCallPart,
    ToolReturnPart,
)
from ghostwheel.schemas import ReviewResult, SEVERITY_VALUES
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

agent = Agent(
    model,
    instructions=(
        "You are a coding assistant. The user will ask you about their code, "
        "and you have tools to read, list, and search the codebase. "
        "Investigate before answering. When you don't know something about the code, "
        "use tools to find out rather than guessing. "
        "Be specific in your answers — cite file paths and line numbers when relevant."
    ),
    deps_type=ToolDeps,
)
register_tools(agent, READ_ONLY_TOOLS)

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


async def stream_to_console(run, console: Console, stream_output: bool = True) -> None:
    status = None

    node = run.next_node

    try:
        while not isinstance(node, End):
            if Agent.is_model_request_node(node):
                async with node.stream(run.ctx) as stream:
                    async for event in stream:
                        if isinstance(event, PartStartEvent):
                            part = event.part

                            if isinstance(part, ThinkingPart):
                                console.print(f"[dim]\n💭 {part.content}[/dim]", end="")

                            elif isinstance(part, TextPart):
                                if stream_output:
                                    console.print(f"\n💬 {part.content}", end="")
                                elif status is None:
                                    status = console.status(
                                        "[bold yellow]Writing review...[/bold yellow]",
                                        spinner="dots",
                                    )
                                    status.start()
                        elif isinstance(event, PartDeltaEvent):
                            if isinstance(event.delta, ThinkingPartDelta):
                                content_delta = event.delta.content_delta
                                if content_delta:
                                    console.print(f"[dim]{content_delta}[/dim]", end="")
                            elif isinstance(event.delta, TextPartDelta):
                                if stream_output:
                                    console.print(event.delta.content_delta, end="")
                                elif event.delta.content_delta and status is None:
                                    status = console.status(
                                        "[bold yellow]Writing review...[/bold yellow]",
                                        spinner="dots",
                                    )
                                    status.start()
            elif Agent.is_call_tools_node(node):
                async with node.stream(run.ctx) as stream:
                    async for event in stream:
                        if isinstance(event, FunctionToolCallEvent):
                            part = event.part
                            if isinstance(part, ToolCallPart):
                                args_preview = str(part.args)
                                if len(args_preview) > 80:
                                    args_preview = args_preview[:80] + "..."
                                console.print(
                                    f"\n[yellow]🔧 {part.tool_name}({args_preview})[/yellow]"
                                )
                        elif isinstance(event, FunctionToolResultEvent):
                            result = event.result
                            if isinstance(result, ToolReturnPart):
                                result_preview = str(result.content)
                                if len(result_preview) > 120:
                                    result_preview = result_preview[:120] + "..."
                                console.print(
                                    f"[green]← {result.tool_name}: {result_preview}[/green]"
                                )
            node = await run.next(node)
    finally:
        if status is not None:
            status.stop()


async def run_chat(console: Console, deps: ToolDeps) -> None:
    """Interactive mode: persistent conversation"""
    history = []
    console.print(
        "[dim]Ghostwheel chat. Type 'quit' to exit, '/review path' to review code.[/dim]"
    )

    while True:
        try:
            user_input = console.input("\n[bold cyan]> [/bold cyan]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        if user_input.lower() in ["/quit"]:
            console.print("\n[dim]Goodbye![/dim]")
            break

        if user_input.lower() == "/clear":
            history = []
            console.print("[dim]History cleared.[/dim]")
            continue

        async with agent.iter(user_input, message_history=history, deps=deps) as run:
            await stream_to_console(run, console)

        if run.result is not None:
            history = run.result.all_messages()


def main() -> None:
    console = Console()
    deps = ToolDeps(
        cwd=Path.cwd(),
        allowed_roots=[Path.cwd()],
    )
    asyncio.run(run_chat(console, deps))


if __name__ == "__main__":
    main()

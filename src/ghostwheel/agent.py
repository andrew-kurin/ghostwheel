import asyncio
from pathlib import Path

import logfire
from pydantic_ai import Agent, AgentRunResult
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolReturnPart,
)
from rich.console import Console
from rich.panel import Panel

from ghostwheel.config import AppConfig, Settings
from ghostwheel.models import build_model, formatter_model_settings
from ghostwheel.rendering import render_review
from ghostwheel.schemas import ReviewResult, SEVERITY_VALUES
from ghostwheel.tools import ALL_TOOLS, register_tools
from ghostwheel.tools.deps import ToolDeps

MAIN_INSTRUCTIONS = (
    "You are a coding assistant. The user will ask you about their code, "
    "and you have tools to read, list, and search the codebase. "
    "Investigate before answering. When you don't know something about the code, "
    "use tools to find out rather than guessing. "
    "Be specific in your answers — cite file paths and line numbers when relevant. "
    "You may use bash for inspection and test commands. "
    "Do not run destructive commands, install dependencies, "
    "or modify files unless the user explicitly asks."
)

FORMATTER_INSTRUCTIONS = (
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
    "- Write a two-sentence 'summary' that captures the overall verdict.\n"
    "- The 'line' field is a single integer. If the prose cites a line range "
    "like '35-38', use the first number (35) as the line.\n"
)

REVIEW_PROMPT = (
    "Perform a careful code review of the files at {paths}. "
    "Use your tools to read each file. "
    "Identify real issues — bugs, security concerns, design problems, dead code. "
    "Don't flag stylistic nits. For each finding, cite file:line and explain why it's a problem. "
    "End with an overall verdict: approve, or changes required."
)

_observability_configured = False


def configure_observability() -> None:
    """Configure telemetry only when the application is actually started."""
    global _observability_configured

    if _observability_configured:
        return

    logfire.configure(console=False)
    logfire.instrument_pydantic_ai()
    _observability_configured = True


def create_chat_agent(config: AppConfig) -> Agent:
    chat_agent = Agent(
        build_model(config.chat_model),
        instructions=MAIN_INSTRUCTIONS,
        deps_type=ToolDeps,
    )
    register_tools(chat_agent, ALL_TOOLS)
    return chat_agent


def create_formatter(config: AppConfig) -> Agent:
    return Agent(
        build_model(config.formatter.model),
        instructions=FORMATTER_INSTRUCTIONS,
        model_settings=formatter_model_settings(config.formatter.model),
        output_type=ReviewResult,
        output_retries=config.formatter.retries,
    )


def create_tool_deps(config: AppConfig, cwd: Path | None = None) -> ToolDeps:
    root = (cwd or Path.cwd()).resolve()
    return ToolDeps(
        cwd=root,
        allowed_roots=[root],
        max_output_bytes=config.tools.max_output_bytes,
    )


async def stream_to_console(run, console: Console) -> None:
    try:
        async for node in run:
            if Agent.is_model_request_node(node):
                async with node.stream(run.ctx) as stream:
                    try:
                        async for event in stream:
                            if isinstance(event, PartStartEvent):
                                part = event.part

                                if isinstance(part, ThinkingPart):
                                    console.print(
                                        f"[dim]\n💭 {part.content}[/dim]", end=""
                                    )

                                elif isinstance(part, TextPart):
                                    console.print(f"\n💬 {part.content}", end="")
                            elif isinstance(event, PartDeltaEvent):
                                if isinstance(event.delta, ThinkingPartDelta):
                                    content_delta = event.delta.content_delta
                                    if content_delta:
                                        console.print(
                                            f"[dim]{content_delta}[/dim]", end=""
                                        )
                                elif isinstance(event.delta, TextPartDelta):
                                    console.print(event.delta.content_delta, end="")
                    except StopAsyncIteration:
                        pass
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
    except StopAsyncIteration:
        pass


async def run_agent_turn(
    chat_agent: Agent,
    prompt: str,
    history: list,
    deps: ToolDeps,
    console: Console,
) -> AgentRunResult[str] | None:
    """Run one chat-agent turn and return its canonical result.

    History is only updated from AgentRunResult.all_messages(), never by
    manually appending user text. If the model run fails or completes without a
    result, keep the previous history and tell the user explicitly.
    """
    try:
        async with chat_agent.iter(prompt, message_history=history, deps=deps) as run:
            await stream_to_console(run, console)

        if run.result is None:
            console.print(
                "[yellow]Agent completed without a result; history was not updated.[/yellow]"
            )
            return None

        return run.result
    except Exception as error:
        console.print(
            Panel(
                str(error),
                title="Agent Failed",
                border_style="red",
            )
        )
        return None


async def run_chat(
    console: Console,
    deps: ToolDeps,
    chat_agent: Agent,
    formatter: Agent,
) -> None:
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

        command, _, command_args = user_input.partition(" ")
        command_lower = command.lower()

        if command_lower == "/quit":
            console.print("\n[dim]Goodbye![/dim]")
            break
        elif command_lower == "/clear":
            history = []
            console.print("[dim]History cleared.[/dim]")
            continue
        elif command_lower == "/review":
            paths = command_args.strip() or "."
            prompt = REVIEW_PROMPT.format(paths=paths)
            result = await run_agent_turn(chat_agent, prompt, history, deps, console)
            if result is None:
                continue

            history = result.all_messages()
            prose = result.output
            try:
                with console.status(
                    "[bold yellow]Formatting review...[/bold yellow]",
                    spinner="dots",
                ):
                    structured = await formatter.run(prose)
                console.print("\n")
                render_review(structured.output, console)
            except Exception as error:
                console.print(
                    Panel(
                        f"[yellow]Couldn't format review as a structured table.[/yellow]\n"
                        f"[dim]Reason: {error}[/dim]\n\n"
                        f"[bold]Showing the raw review instead:[/bold]\n\n{prose}",
                        title="Formatter Failed",
                        border_style="yellow",
                    )
                )
            continue

        result = await run_agent_turn(chat_agent, user_input, history, deps, console)
        if result is not None:
            history = result.all_messages()


def main() -> None:
    configure_observability()
    config = Settings().resolve()
    console = Console()
    deps = create_tool_deps(config)
    chat_agent = create_chat_agent(config)
    formatter = create_formatter(config)
    asyncio.run(run_chat(console, deps, chat_agent, formatter))


def run() -> None:
    main()


if __name__ == "__main__":
    main()

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from ghostwheel.events import (
    AgentEvent,
    TextOutput,
    ThinkingOutput,
    ToolFailed,
    ToolFinished,
    ToolStarted,
)
from ghostwheel.rendering import render_review
from ghostwheel.review import RawReview, ReviewFailed, ReviewOutcome, StructuredReview
from ghostwheel.session import TurnFailed, TurnNoResult, TurnOutcome


class RichPresenter:
    """Render application events without interpreting dynamic values as markup."""

    def __init__(self, console: Console) -> None:
        self.console = console

    async def handle_event(self, event: AgentEvent) -> None:
        if isinstance(event, ThinkingOutput):
            if event.starts_part:
                self.console.print(Text("\n💭 ", style="dim"), end="")
            self.console.print(Text(event.content, style="dim"), end="")
        elif isinstance(event, TextOutput):
            if event.starts_part:
                self.console.print(Text("\n💬 "), end="")
            self.console.print(Text(event.content), end="")
        elif isinstance(event, ToolStarted):
            arguments = _preview(event.arguments, 80)
            line = Text("\n🔧 ", style="yellow")
            line.append(event.name, style="yellow")
            line.append(f"({arguments})", style="yellow")
            self.console.print(line)
        elif isinstance(event, ToolFinished):
            result = _preview(event.result, 120)
            line = Text("← ", style="green")
            line.append(event.name, style="green")
            line.append(": ", style="green")
            line.append(result, style="green")
            self.console.print(line)
        elif isinstance(event, ToolFailed):
            error = _preview(event.error, 120)
            line = Text("← ", style="red")
            line.append(event.name, style="red")
            line.append(" failed: ", style="red")
            line.append(error, style="red")
            self.console.print(line)

    def welcome(self) -> None:
        self.console.print(
            Text(
                "Ghostwheel chat. Type '/quit' to exit, '/review path' to review code.",
                style="dim",
            )
        )

    def goodbye(self) -> None:
        self.console.print(Text("\nGoodbye!", style="dim"))

    def history_cleared(self) -> None:
        self.console.print(Text("History cleared.", style="dim"))

    def history_compacted(self, dropped_turns: int) -> None:
        noun = "turn" if dropped_turns == 1 else "turns"
        self.console.print(
            Text(
                f"Context compacted: dropped {dropped_turns} {noun} to fit the budget.",
                style="dim",
            )
        )

    def turn_outcome(self, outcome: TurnOutcome) -> None:
        if isinstance(outcome, TurnNoResult):
            self.console.print(Text(outcome.message, style="yellow"))
        elif isinstance(outcome, TurnFailed):
            self.console.print(
                Panel(
                    Text(outcome.message),
                    title="Agent Failed",
                    border_style="red",
                )
            )

    def review_outcome(self, outcome: ReviewOutcome) -> None:
        if isinstance(outcome, StructuredReview):
            self.console.print("")
            if outcome.used_fallback:
                self.console.print(
                    Text(
                        "Structured-output fallback was used for this review.",
                        style="dim",
                    )
                )
            render_review(outcome.review, self.console)
        elif isinstance(outcome, RawReview):
            body = Text()
            body.append(
                "Couldn't produce a structured review.\n",
                style="yellow",
            )
            body.append("Reason: ", style="dim")
            body.append(outcome.structured_failure, style="dim")
            body.append("\n\nShowing the raw review instead:\n\n", style="bold")
            body.append(outcome.prose)
            self.console.print(
                Panel(body, title="Structured Review Failed", border_style="yellow")
            )
        elif isinstance(outcome, ReviewFailed):
            self.console.print(
                Panel(
                    Text(outcome.message),
                    title="Review Failed",
                    border_style="red",
                )
            )


def _preview(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[:limit] + "..."

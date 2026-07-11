from rich.console import Console, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ghostwheel.schemas import ReviewResult, Severity

NARROW_REVIEW_WIDTH = 100

SEVERITY_STYLE = {
    Severity.SUGGESTION: ("dim cyan", "💡"),
    Severity.WARNING: ("yellow", "⚠"),
    Severity.BLOCKER: ("bold red", "🛑"),
}

SEVERITY_ORDER = {
    Severity.BLOCKER: 0,
    Severity.WARNING: 1,
    Severity.SUGGESTION: 2,
}


def _finding_location(file: str, line: int | None, line_end: int | None) -> str:
    location = file
    if line is not None:
        location += f":{line}"
        if line_end is not None and line_end != line:
            location += f"-{line_end}"
    return location


def _stacked_findings(review: ReviewResult) -> list[RenderableType]:
    renderables: list[RenderableType] = []
    findings = sorted(
        review.findings,
        key=lambda item: SEVERITY_ORDER[item.severity],
    )
    for finding in findings:
        style, icon = SEVERITY_STYLE[finding.severity]
        title = Text(f"{icon} {finding.severity.value.upper()}", style=style)
        body = Text.assemble(
            Text("Location  ", style="bold"),
            Text(
                _finding_location(finding.file, finding.line, finding.line_end),
                style="cyan",
            ),
            "\n",
            Text("Category  ", style="bold"),
            Text(finding.category, style="magenta"),
            "\n\n",
            Text(finding.message),
        )
        if finding.suggestion:
            body.append("\n\n")
            body.append("→ Suggestion\n", style="dim bold")
            body.append(finding.suggestion, style="dim italic")

        renderables.append(Panel(body, title=title, border_style=style))
    return renderables


def review_renderables(
    review: ReviewResult,
    *,
    width: int,
) -> tuple[RenderableType, ...]:
    """Build review output for both the plain renderer and persistent TUI."""
    verdict = (
        Text("APPROVED", style="bold green")
        if review.approve
        else Text("CHANGES REQUIRED", style="bold red")
    )
    renderables: list[RenderableType] = [
        Panel(
            Text.assemble(verdict, "\n\n", Text(review.summary)),
            title="Code Review",
            border_style="green" if review.approve else "red",
        )
    ]

    if not review.findings:
        renderables.append(Text("No findings to report.", style="dim"))
        return tuple(renderables)

    if width < NARROW_REVIEW_WIDTH:
        renderables.extend(_stacked_findings(review))
        return tuple(renderables)

    findings = sorted(review.findings, key=lambda f: SEVERITY_ORDER[f.severity])

    table = Table(show_lines=True, expand=True)
    table.add_column("", width=3)
    table.add_column("Location", style="cyan", no_wrap=True)
    table.add_column("Category", style="magenta")
    table.add_column("Issue")

    for f in findings:
        style, icon = SEVERITY_STYLE[f.severity]
        location = _finding_location(f.file, f.line, f.line_end)
        issue = Text(f.message)
        if f.suggestion:
            issue.append("\n→ ", style="dim")
            issue.append(f.suggestion, style="dim italic")
        table.add_row(
            Text(icon, style=style),
            Text(location, style="cyan"),
            Text(f.category, style="magenta"),
            issue,
        )

    renderables.append(table)
    return tuple(renderables)


def render_review(review: ReviewResult, console: Console | None = None) -> None:
    console = console or Console()
    console.print(*review_renderables(review, width=console.width))

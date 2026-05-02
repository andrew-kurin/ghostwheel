from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from schemas import ReviewResult, Severity

SEVERITY_STYLE = {
    Severity.SUGGESTION: ("dim cyan", ":bulb:"),
    Severity.WARNING: ("yellow", ":warning:"),
    Severity.BLOCKER: ("bold red", ":stop_sign:"),
}


def render_review(review: ReviewResult, console: Console | None = None) -> None:
    console = console or Console()

    verdict = (
        Text("APPROVED", style="bold green")
        if review.approve
        else Text("CHANGES REQUIRED", style="bold red")
    )
    console.print(
        Panel(
            Text.assemble(verdict, "\n\n", review.summary),
            title="Code Review",
            border_style="green" if review.approve else "red",
        )
    )

    if not review.findings:
        console.print("[dim]No findings to report.[/dim]")
        return

    # findings table sorted by severity
    severity_order = {Severity.SUGGESTION: 2, Severity.WARNING: 1, Severity.BLOCKER: 0}
    findings = sorted(review.findings, key=lambda f: severity_order[f.severity])

    table = Table(show_lines=True, expand=True)
    table.add_column("", width=3)
    table.add_column("Location", style="cyan", no_wrap=True)
    table.add_column("Category", style="magenta")
    table.add_column("Issue")

    for f in findings:
        style, icon = SEVERITY_STYLE[f.severity]
        location = f"{f.file}:{f.line}" if f.line else f.file
        issue = Text(f.message)
        if f.suggestion:
            issue.append("\n→ ", style="dim")
            issue.append(f.suggestion, style="dim italic")
        table.add_row(Text.from_markup(icon, style=style), location, f.category, issue)

    console.print(table)

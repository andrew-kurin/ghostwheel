from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ghostwheel.schemas import ReviewResult, Severity

SEVERITY_STYLE = {
    Severity.SUGGESTION: ("dim cyan", "💡"),
    Severity.WARNING: ("yellow", "⚠"),
    Severity.BLOCKER: ("bold red", "🛑"),
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
            Text.assemble(verdict, "\n\n", Text(review.summary)),
            title="Code Review",
            border_style="green" if review.approve else "red",
        )
    )

    if not review.findings:
        console.print(Text("No findings to report.", style="dim"))
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
        location = f.file
        if f.line is not None:
            location += f":{f.line}"
            if f.line_end is not None and f.line_end != f.line:
                location += f"-{f.line_end}"
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

    console.print(table)

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


def sanitize_terminal_text(value: str) -> str:
    """Remove terminal controls while preserving ordinary layout characters.

    Rich treats strings as literal markup, but deliberately passes ANSI and
    other terminal escape sequences through. Model, provider, tool, and
    workspace text therefore needs an explicit trust boundary before it is
    handed to any Rich renderable.
    """

    output: list[str] = []
    index = 0
    while index < len(value):
        codepoint = _terminal_codepoint(value[index])
        if codepoint == 0x1B:
            index = _consume_escape(value, index)
        elif codepoint == 0x9B:
            index = _consume_csi(value, index + 1)
        elif codepoint in {0x90, 0x98, 0x9D, 0x9E, 0x9F}:
            index = _consume_control_string(
                value,
                index + 1,
                osc=codepoint == 0x9D,
            )
        elif value[index] in {"\n", "\t"}:
            output.append(value[index])
            index += 1
        elif codepoint < 0x20 or 0x7F <= codepoint <= 0x9F:
            index += 1
        else:
            output.append(value[index])
            index += 1
    return "".join(output)


def sanitize_terminal_line(value: str) -> str:
    """Return terminal-safe text collapsed to one display line.

    Labels, paths, and activity fields must not be able to introduce extra
    rows into a structured terminal presentation.  Normalize every kind of
    Unicode whitespace after removing terminal controls so separators such as
    line and paragraph breaks cannot bypass that boundary.
    """

    return " ".join(sanitize_terminal_text(value).split())


def _terminal_codepoint(character: str) -> int:
    """Return the byte represented by surrogateescape, or the Unicode value."""

    codepoint = ord(character)
    if 0xDC80 <= codepoint <= 0xDCFF:
        return codepoint - 0xDC00
    return codepoint


def _consume_escape(value: str, index: int) -> int:
    """Return the first index after one seven-bit terminal escape sequence."""

    index += 1
    if index >= len(value):
        return index

    introducer = value[index]
    if introducer == "[":
        return _consume_csi(value, index + 1)
    if introducer in {"P", "X", "]", "^", "_"}:
        return _consume_control_string(
            value,
            index + 1,
            osc=introducer == "]",
        )

    codepoint = _terminal_codepoint(introducer)
    if 0x20 <= codepoint <= 0x2F:
        index += 1
        while index < len(value) and 0x20 <= _terminal_codepoint(value[index]) <= 0x2F:
            index += 1
        if index < len(value) and 0x30 <= _terminal_codepoint(value[index]) <= 0x7E:
            index += 1
        return index
    if 0x30 <= codepoint <= 0x7E:
        return index + 1
    return index


def _consume_csi(value: str, index: int) -> int:
    """Consume a CSI through its final byte, or through incomplete input."""

    while index < len(value):
        codepoint = _terminal_codepoint(value[index])
        index += 1
        if 0x40 <= codepoint <= 0x7E:
            break
    return index


def _consume_control_string(value: str, index: int, *, osc: bool) -> int:
    """Consume OSC/DCS/SOS/PM/APC data through its string terminator."""

    while index < len(value):
        codepoint = _terminal_codepoint(value[index])
        if codepoint == 0x9C or (osc and codepoint == 0x07):
            return index + 1
        if codepoint == 0x1B and index + 1 < len(value) and value[index + 1] == "\\":
            return index + 2
        index += 1
    return index


def _finding_location(file: str, line: int | None, line_end: int | None) -> str:
    location = sanitize_terminal_line(file)
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
            Text(sanitize_terminal_line(finding.category), style="magenta"),
            "\n\n",
            Text(sanitize_terminal_text(finding.message)),
        )
        if finding.suggestion:
            body.append("\n\n")
            body.append("→ Suggestion\n", style="dim bold")
            body.append(
                sanitize_terminal_text(finding.suggestion),
                style="dim italic",
            )

        renderables.append(Panel(body, title=title, border_style=style))
    return renderables


def review_renderables(
    review: ReviewResult,
    *,
    width: int,
) -> tuple[RenderableType, ...]:
    """Build review output for the terminal UI at the requested width."""
    verdict = (
        Text("APPROVED", style="bold green")
        if review.approve
        else Text("CHANGES REQUIRED", style="bold red")
    )
    renderables: list[RenderableType] = [
        Panel(
            Text.assemble(
                verdict,
                "\n\n",
                Text(sanitize_terminal_text(review.summary)),
            ),
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
        issue = Text(sanitize_terminal_text(f.message))
        if f.suggestion:
            issue.append("\n→ ", style="dim")
            issue.append(sanitize_terminal_text(f.suggestion), style="dim italic")
        table.add_row(
            Text(icon, style=style),
            Text(location, style="cyan"),
            Text(sanitize_terminal_line(f.category), style="magenta"),
            issue,
        )

    renderables.append(table)
    return tuple(renderables)


def render_review(review: ReviewResult, console: Console | None = None) -> None:
    console = console or Console()
    console.print(*review_renderables(review, width=console.width))

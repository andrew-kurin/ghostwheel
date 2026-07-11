from io import StringIO

from rich.console import Console

from ghostwheel.rendering import render_review
from ghostwheel.schemas import Finding, ReviewResult, Severity


def test_render_review_preserves_line_ranges_and_literal_content() -> None:
    review = ReviewResult(
        summary="Contains [literal] markup.",
        findings=[
            Finding(
                file="src/app.py",
                line=3,
                line_end=5,
                severity=Severity.WARNING,
                category="bug",
                message="A [tag] is literal.",
            )
        ],
        approve=False,
    )
    output = StringIO()
    console = Console(file=output, color_system=None, force_terminal=False, width=120)

    render_review(review, console)

    rendered = output.getvalue()
    assert "src/app.py:3-5" in rendered
    assert "Contains [literal] markup." in rendered
    assert "A [tag] is literal." in rendered

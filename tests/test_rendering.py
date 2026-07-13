from io import StringIO

from rich.console import Console

from ghostwheel.rendering import render_review, sanitize_terminal_text
from ghostwheel.schemas import Finding, ReviewResult, Severity


def test_sanitize_terminal_text_removes_csi_osc_and_other_controls() -> None:
    value = (
        "alpha\tbeta\r\n\x00\x1b[2Jgamma\x1b]52;c;Y2xpcGJvYXJk\x1b\\delta\x9b31mepsilon"
    )

    assert sanitize_terminal_text(value) == "alpha\tbeta\ngammadeltaepsilon"


def test_sanitize_terminal_text_blocks_surrogateescaped_c1_bytes() -> None:
    value = "alpha\udc9b2Jbeta\udc9d52;c;Y2xpcGJvYXJk\udc9cgamma\udc80delta"

    safe_value = sanitize_terminal_text(value)
    encoded = safe_value.encode("utf-8", errors="surrogateescape")

    assert safe_value == "alphabetagammadelta"
    assert b"\x1b" not in encoded
    assert not any(0x80 <= byte <= 0x9F for byte in encoded)


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


def test_render_review_neutralizes_controls_in_every_model_field() -> None:
    csi = "\x1b[2J"
    osc = "\x1b]52;c;Y2xpcGJvYXJk\x1b\\"
    review = ReviewResult(
        summary=f"summary{csi} remains",
        findings=[
            Finding(
                file=f"src/{osc}app.py",
                line=7,
                severity=Severity.WARNING,
                category=f"sec{csi}urity",
                message=f"message{osc} remains",
                suggestion=f"suggestion{csi} remains",
            )
        ],
    )
    output = StringIO()
    console = Console(file=output, color_system=None, force_terminal=False, width=120)

    render_review(review, console)

    rendered = output.getvalue()
    assert csi not in rendered
    assert osc not in rendered
    assert "Y2xpcGJvYXJk" not in rendered
    assert "summary remains" in rendered
    assert "src/app.py:7" in rendered
    assert "security" in rendered
    assert "message remains" in rendered
    assert "suggestion remains" in rendered


def test_render_review_uses_stacked_findings_on_narrow_terminals() -> None:
    review = ReviewResult(
        summary="Three findings.",
        findings=[
            Finding(
                file="src/suggestion.py",
                severity=Severity.SUGGESTION,
                category="style",
                message="Consider simplifying this expression.",
                suggestion="Use the direct return value.",
            ),
            Finding(
                file="src/warning.py",
                line=8,
                severity=Severity.WARNING,
                category="bug",
                message="This warning should appear second.",
            ),
            Finding(
                file="src/[literal]-blocker-with-a-long-name.py",
                line=12,
                line_end=16,
                severity=Severity.BLOCKER,
                category="security[tag]",
                message="A [bold] blocker should appear first and wrap safely.",
                suggestion="Validate [all] input before using it.",
            ),
        ],
    )
    output = StringIO()
    console = Console(file=output, color_system=None, force_terminal=False, width=60)

    render_review(review, console)

    rendered = output.getvalue()
    assert "Location" in rendered
    assert "src/[literal]-blocker-with-a-long-name.py:12-16" in rendered
    assert "A [bold] blocker should appear first and wrap safely." in rendered
    assert "security[tag]" in rendered
    assert "Suggestion" in rendered
    assert "Validate [all] input before using it." in rendered
    assert (
        rendered.index("BLOCKER")
        < rendered.index("WARNING")
        < rendered.index("SUGGESTION")
    )
    assert "Category  security[tag]" in rendered

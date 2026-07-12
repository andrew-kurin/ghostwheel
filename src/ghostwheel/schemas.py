from enum import Enum
from typing import Self

from pydantic import BaseModel, Field, computed_field, model_validator


class Severity(str, Enum):
    SUGGESTION = "suggestion"
    WARNING = "warning"
    BLOCKER = "blocker"


def enum_values(enum_type: type[Enum]) -> str:
    return ", ".join(f"'{member.value}'" for member in enum_type)


SEVERITY_VALUES = enum_values(Severity)


class Finding(BaseModel):
    file: str = Field(description="Path to the file the findings applies to")
    line: int | None = Field(
        description="First line number if applicable",
        default=None,
        ge=1,
    )
    line_end: int | None = Field(
        description="If the issue spans a range, the last line",
        default=None,
        ge=1,
    )
    severity: Severity = Field(
        description=f"Impact level. Must be exactly: {SEVERITY_VALUES}"
    )
    category: str = Field(description="Short tag: bug, style, perf, security, etc")
    message: str = Field(description="One or two sentences describing the issue")
    suggestion: str | None = Field(description="Concrete fix if obvious", default=None)

    @model_validator(mode="after")
    def validate_line_range(self) -> Self:
        if self.line_end is not None and self.line is None:
            raise ValueError("line_end requires line")
        if self.line_end is not None and self.line_end < self.line:
            raise ValueError("line_end must be greater than or equal to line")
        return self


class ReviewResult(BaseModel):
    summary: str = Field(description="Summary of the review result")
    findings: list[Finding]

    @computed_field(
        description=(
            "Derived from findings: true exactly when there are no warnings or blockers"
        )
    )
    @property
    def approve(self) -> bool:
        """Return the verdict as a durable derivative of finding severities."""

        return not any(
            finding.severity in {Severity.WARNING, Severity.BLOCKER}
            for finding in self.findings
        )

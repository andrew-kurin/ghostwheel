from enum import Enum

from pydantic import BaseModel, Field


class Severity(str, Enum):
    SUGGESTION = "suggestion"
    WARNING = "warning"
    BLOCKER = "blocker"


def enum_values(enum_type: type[Enum]) -> str:
    return ", ".join(f"'{member.value}'" for member in enum_type)


SEVERITY_VALUES = enum_values(Severity)


class Finding(BaseModel):
    file: str = Field(description="Path to the file the findings applies to")
    line: int | None = Field(description="Line number if applicable", default=None)
    severity: Severity = Field(
        description=f"Impact level. Must be exactly: {SEVERITY_VALUES}"
    )
    category: str = Field(description="Short tag: bug, style, perf, security, etc")
    message: str = Field(description="One or two sentences describing the issue")
    suggestion: str | None = Field(description="Concrete fix if obvious", default=None)


class ReviewResult(BaseModel):
    summary: str = Field(description="Summary of the review result")
    findings: list[Finding]
    approve: bool = Field(description="True if safe to merge as-is")

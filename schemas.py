from enum import Enum

from pydantic import BaseModel, Field


class Severity(str, Enum):
    SUGGESTION = "suggestion"
    WARNING = "warning"
    BLOCKER = "blocker"


class Finding(BaseModel):
    file: str = Field(description="Path to the file the findings applies to")
    line: int | None = Field(description="Line number if applicable", default=None)
    severity: Severity
    category: str = Field(description="Short tag: bug, style, perf, security, etc")
    message: str = Field(description="One or two sentences describing the issue")
    suggestion: str | None = Field(description="Concrete fix if obvious", default=None)


class ReviewResult(BaseModel):
    summary: str = Field(description="Summary of the review result")
    findings: list[Finding]
    approve: bool = Field(description="True if safe to merge as-is")

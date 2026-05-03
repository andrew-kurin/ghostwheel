from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ToolDeps:
    cwd: Path  # Working directory; tools resolve paths relative to this
    allowed_roots: list[Path] = field(default_factory=list)  # Paths the agent can touch
    max_output_bytes: int = 100_000  # Cap on tool output size
    bash_timeout_seconds: int = 30  # Default subprocess timeout
    dry_run: bool = False

"""Backward-compatible agent factory and entry-point facade."""

from ghostwheel.agent_factory import (
    COMPACTION_INSTRUCTIONS,
    FORMATTER_INSTRUCTIONS,
    MAIN_INSTRUCTIONS,
    REVIEW_INSTRUCTIONS,
    chat_agent_blueprint,
    compaction_agent_blueprint,
    create_chat_agent,
    create_compaction_agent,
    create_formatter,
    create_review_agent,
    create_review_fallback_agent,
    create_tool_deps,
    review_agent_blueprint,
    review_fallback_agent_blueprint,
)
from ghostwheel.config import AppConfig, Settings

__all__ = [
    "COMPACTION_INSTRUCTIONS",
    "FORMATTER_INSTRUCTIONS",
    "MAIN_INSTRUCTIONS",
    "REVIEW_INSTRUCTIONS",
    "chat_agent_blueprint",
    "compaction_agent_blueprint",
    "configure_observability",
    "create_chat_agent",
    "create_compaction_agent",
    "create_formatter",
    "create_review_agent",
    "create_review_fallback_agent",
    "create_tool_deps",
    "main",
    "review_agent_blueprint",
    "review_fallback_agent_blueprint",
    "run",
]


def configure_observability(config: AppConfig | None = None) -> bool:
    """Compatibility wrapper around the opt-in telemetry module."""

    from ghostwheel.telemetry import configure_observability as configure

    resolved = config or Settings().resolve()
    return configure(resolved.observability)


def main() -> None:
    from ghostwheel.cli import main as cli_main

    cli_main()


def run() -> None:
    main()


if __name__ == "__main__":
    main()

"""Pydantic-AI agent factories.

Application/session behavior lives in dedicated modules. This module remains as
the compatibility import location for agent construction and ``main``.
"""

from pathlib import Path
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from ghostwheel.config import AppConfig, Settings
from ghostwheel.models import build_model, structured_output_model_settings
from ghostwheel.schemas import ReviewResult
from ghostwheel.tools import DEFAULT_TOOL_CATALOG, ToolCatalog, ToolProfile
from ghostwheel.tools.deps import ToolDeps, ToolLimits

MAIN_INSTRUCTIONS = (
    "You are a coding assistant. The user will ask you about their code, "
    "and you have tools to read, list, and search the codebase. "
    "Investigate before answering. When you don't know something about the code, "
    "use tools to find out rather than guessing. "
    "Be specific in your answers — cite file paths and line numbers when relevant. "
    "You may use bash for inspection and test commands when that capability is "
    "available. Do not run destructive commands, install dependencies, or modify "
    "files unless the user explicitly asks."
)

REVIEW_INSTRUCTIONS = (
    "You are a focused code reviewer. Inspect the requested files with the tools "
    "available to you. Report concrete bugs, security concerns, reliability risks, "
    "and material design problems; omit stylistic nits. Preserve exact file paths "
    "and full line ranges. A review is approved exactly when it contains no warning "
    "or blocker findings."
)

FORMATTER_INSTRUCTIONS = (
    "Transcribe review prose into ReviewResult. Include every finding and do not "
    "invent any. Preserve severity, file paths, first and last line numbers, "
    "category, message, and any proposed fix. Use suggestion when the prose only "
    "recommends an improvement, warning for required non-blocking changes, and "
    "blocker for bugs, security flaws, or merge-blocking failures. Summarize the "
    "verdict concisely; approval is derived from the resulting severities."
)

COMPACTION_INSTRUCTIONS = (
    "You maintain a concise rolling checkpoint for a coding-assistant "
    "conversation. Summarize only the serialized transcript supplied in the "
    "prompt. Preserve user requirements, completed work, decisions, unresolved "
    "issues, commands, errors, and exact file paths. Never use tools, continue "
    "the conversation, or claim work that is not present in the transcript."
)


def _profile_tools(
    profile: str,
    catalog: ToolCatalog,
) -> tuple[Any, ...]:
    return catalog.for_profile(ToolProfile(profile))


def create_chat_agent(
    config: AppConfig,
    *,
    catalog: ToolCatalog = DEFAULT_TOOL_CATALOG,
) -> Agent[ToolDeps, str]:
    return Agent(
        build_model(config.chat_model),
        instructions=MAIN_INSTRUCTIONS,
        deps_type=ToolDeps,
        tools=_profile_tools(config.tools.profile, catalog),
    )


def create_review_agent(
    config: AppConfig,
    *,
    catalog: ToolCatalog = DEFAULT_TOOL_CATALOG,
) -> Agent[ToolDeps, str]:
    return Agent(
        build_model(config.review.model),
        instructions=REVIEW_INSTRUCTIONS,
        deps_type=ToolDeps,
        tools=_profile_tools(config.tools.review_profile, catalog),
        model_settings=structured_output_model_settings(config.review.model),
        retries=config.review.retries,
    )


def create_compaction_agent(config: AppConfig) -> Agent[None, str]:
    """Create the isolated, tool-free model used for rolling summaries."""

    return Agent(
        build_model(config.chat_model),
        instructions=COMPACTION_INSTRUCTIONS,
        model_settings=ModelSettings(
            max_tokens=config.history.compaction.summary_tokens,
            temperature=0.1,
        ),
    )


def create_formatter(config: AppConfig) -> Agent[None, ReviewResult]:
    """Compatibility factory for the former prose-to-schema second pass."""
    return Agent(
        build_model(config.review.model),
        instructions=FORMATTER_INSTRUCTIONS,
        model_settings=structured_output_model_settings(config.review.model),
        output_type=ReviewResult,
        retries=config.review.retries,
    )


def create_tool_deps(config: AppConfig, cwd: Path | None = None) -> ToolDeps:
    root = (cwd or Path.cwd()).resolve()
    return ToolDeps(
        cwd=root,
        filesystem_roots=(root,),
        limits=ToolLimits(
            max_output_bytes=config.tools.max_output_bytes,
            max_entries=config.tools.max_entries,
            max_directory_scan_entries=config.tools.max_directory_scan_entries,
            max_matches=config.tools.max_matches,
            bash_timeout_seconds=config.tools.bash_timeout_seconds,
            max_search_file_bytes=config.tools.max_search_file_bytes,
            max_search_total_bytes=config.tools.max_search_total_bytes,
            max_search_files=config.tools.max_search_files,
            search_timeout_seconds=config.tools.search_timeout_seconds,
            regex_timeout_seconds=config.tools.regex_timeout_seconds,
        ),
    )


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

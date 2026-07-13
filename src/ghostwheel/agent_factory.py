"""Canonical Pydantic-AI agent blueprints and factories."""

from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from ghostwheel.agent_blueprint import AgentBlueprint
from ghostwheel.config import AppConfig
from ghostwheel.providers import structured_output_model_settings
from ghostwheel.schemas import ReviewResult
from ghostwheel.tools.catalog import DEFAULT_TOOL_CATALOG, ToolCatalog
from ghostwheel.tools.deps import ToolDeps

MAIN_INSTRUCTIONS = (
    "You are a coding assistant. The user will ask you about their code, "
    "and you have tools to read, list, search, and sometimes edit the codebase. "
    "Investigate before answering. When you don't know something about the code, "
    "use tools to find out rather than guessing. "
    "Be specific in your answers — cite file paths and line numbers when relevant. "
    "When the user asks for a change and edit is available, read the target first "
    "and prefer a precise edit over shell redirection or scripted rewrites. "
    "You may use bash for inspection and test commands when that capability is "
    "available. Do not run destructive commands, install dependencies, or modify "
    "files unless the user explicitly asks."
)

REVIEW_INSTRUCTIONS = (
    "You are a focused code reviewer. Inspect the requested files with the tools "
    "available to you. Report concrete bugs, security concerns, reliability risks, "
    "and material design problems; omit stylistic nits. Do not modify files. Preserve "
    "exact file paths and full line ranges. A review is approved exactly when it "
    "contains no warning "
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


def chat_agent_blueprint(
    config: AppConfig,
    *,
    catalog: ToolCatalog = DEFAULT_TOOL_CATALOG,
) -> AgentBlueprint[ToolDeps, str]:
    return AgentBlueprint.from_functions(
        model=config.chat_model,
        instructions=MAIN_INSTRUCTIONS,
        deps_type=ToolDeps,
        output_type=str,
        tools=catalog.for_profile(config.tools.profile),
    )


def create_chat_agent(
    config: AppConfig,
    *,
    catalog: ToolCatalog = DEFAULT_TOOL_CATALOG,
) -> Agent[ToolDeps, str]:
    return chat_agent_blueprint(config, catalog=catalog).build()


def review_agent_blueprint(
    config: AppConfig,
    *,
    catalog: ToolCatalog = DEFAULT_TOOL_CATALOG,
) -> AgentBlueprint[ToolDeps, str]:
    return AgentBlueprint.from_functions(
        model=config.review.model,
        instructions=REVIEW_INSTRUCTIONS,
        deps_type=ToolDeps,
        output_type=str,
        tools=catalog.for_profile(config.tools.review_profile),
        model_settings=structured_output_model_settings(config.review.model),
        retries=config.review.retries,
    )


def create_review_agent(
    config: AppConfig,
    *,
    catalog: ToolCatalog = DEFAULT_TOOL_CATALOG,
) -> Agent[ToolDeps, str]:
    return review_agent_blueprint(config, catalog=catalog).build()


def compaction_agent_blueprint(
    config: AppConfig,
) -> AgentBlueprint[None, str]:
    return AgentBlueprint.from_functions(
        model=config.chat_model,
        instructions=COMPACTION_INSTRUCTIONS,
        deps_type=type(None),
        output_type=str,
        model_settings=ModelSettings(
            max_tokens=config.history.compaction.summary_tokens,
            temperature=0.1,
        ),
    )


def create_compaction_agent(config: AppConfig) -> Agent[None, str]:
    """Create the isolated, tool-free model used for rolling summaries."""

    return compaction_agent_blueprint(config).build()


def review_fallback_agent_blueprint(
    config: AppConfig,
) -> AgentBlueprint[None, ReviewResult]:
    return AgentBlueprint.from_functions(
        model=config.review.model,
        instructions=FORMATTER_INSTRUCTIONS,
        deps_type=type(None),
        output_type=ReviewResult,
        model_settings=structured_output_model_settings(config.review.model),
        retries=config.review.retries,
    )


def create_review_fallback_agent(config: AppConfig) -> Agent[None, ReviewResult]:
    """Create the prose-to-structured-review fallback agent."""

    return review_fallback_agent_blueprint(config).build()


def create_formatter(config: AppConfig) -> Agent[None, ReviewResult]:
    """Compatibility factory for the former prose-to-schema second pass."""

    return create_review_fallback_agent(config)


def create_tool_deps(config: AppConfig, cwd: Path | None = None) -> ToolDeps:
    root = (cwd or Path.cwd()).resolve()
    return ToolDeps(
        cwd=root,
        filesystem_roots=(root,),
        limits=config.tools.limits,
    )


__all__ = [
    "COMPACTION_INSTRUCTIONS",
    "FORMATTER_INSTRUCTIONS",
    "MAIN_INSTRUCTIONS",
    "REVIEW_INSTRUCTIONS",
    "chat_agent_blueprint",
    "compaction_agent_blueprint",
    "create_chat_agent",
    "create_compaction_agent",
    "create_formatter",
    "create_review_agent",
    "create_review_fallback_agent",
    "create_tool_deps",
    "review_agent_blueprint",
    "review_fallback_agent_blueprint",
]

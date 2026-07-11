"""Application composition root."""

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from ghostwheel.agent import (
    create_chat_agent,
    create_compaction_agent,
    create_formatter,
    create_review_agent,
    create_tool_deps,
)
from ghostwheel.compaction import HistoryCompactor
from ghostwheel.config import COMPACTION_REQUEST_OVERHEAD_TOKENS, AppConfig
from ghostwheel.events import AgentEvent, TextOutput
from ghostwheel.pydantic_runner import PydanticAgentRunner
from ghostwheel.review import ReviewService
from ghostwheel.rich_ui import AppInfo, RichPresenter
from ghostwheel.session import ChatSession, HistoryPolicy
from ghostwheel.tools import DEFAULT_TOOL_CATALOG, ToolCatalog
from ghostwheel.tools.deps import ToolDeps
from ghostwheel.token_counting import TiktokenTokenCounter

PROVIDER_FRAMING_TOKENS = 256


@dataclass(frozen=True, slots=True)
class Application:
    session: ChatSession
    reviews: ReviewService
    presenter: RichPresenter
    tool_deps: ToolDeps

    def close(self) -> None:
        self.tool_deps.close()


def build_application(
    config: AppConfig,
    console: Console,
    *,
    cwd: Path | None = None,
    catalog: ToolCatalog = DEFAULT_TOOL_CATALOG,
    live_ui: bool = False,
    event_sink: Callable[[AgentEvent], Awaitable[None]] | None = None,
) -> Application:
    deps = create_tool_deps(config, cwd)
    presenter = RichPresenter(
        console,
        app_info=AppInfo(
            workspace=str(deps.cwd),
            provider=config.chat_model.provider.value,
            model=config.chat_model.model,
            tool_profile=config.tools.profile,
        ),
        live=live_ui,
    )

    active_event_sink = event_sink or presenter.handle_event
    token_counter = TiktokenTokenCounter()
    chat_agent = create_chat_agent(config, catalog=catalog)
    initial_overhead_tokens = _estimate_chat_overhead(chat_agent, token_counter)
    chat_runner = PydanticAgentRunner(
        chat_agent,
        deps,
        event_sink=active_event_sink,
    )
    compaction_runner = PydanticAgentRunner(create_compaction_agent(config), None)

    async def review_events(event) -> None:
        # Structured model output may be JSON text. Keep that implementation
        # detail out of the CLI while still showing thinking and tool activity.
        if not isinstance(event, TextOutput):
            await active_event_sink(event)

    review_runner = PydanticAgentRunner(
        create_review_agent(config, catalog=catalog),
        deps,
        event_sink=review_events,
    )
    fallback_runner = PydanticAgentRunner(create_formatter(config), None)
    session = ChatSession(
        chat_runner,
        history_policy=HistoryPolicy(
            context_window_tokens=config.history.context_window_tokens,
            compaction_enabled=config.history.compaction.enabled,
            reserve_tokens=config.history.compaction.reserve_tokens,
            keep_recent_tokens=config.history.compaction.keep_recent_tokens,
            summary_tokens=config.history.compaction.summary_tokens,
            token_counter=token_counter,
        ),
        compactor=HistoryCompactor(
            compaction_runner,
            token_counter=token_counter,
            input_token_budget=(
                config.history.context_window_tokens
                - config.history.compaction.summary_tokens
                - COMPACTION_REQUEST_OVERHEAD_TOKENS
            ),
            summary_token_limit=config.history.compaction.summary_tokens,
        ),
        initial_overhead_tokens=initial_overhead_tokens,
    )
    reviews = ReviewService(
        review_runner,
        raw_fallback=config.review.raw_fallback,
        fallback_runner=fallback_runner,
    )
    return Application(
        session=session,
        reviews=reviews,
        presenter=presenter,
        tool_deps=deps,
    )


def _estimate_chat_overhead(agent, token_counter: TiktokenTokenCounter) -> int:
    """Estimate static instructions, tool schemas, and provider framing."""

    tools = []
    for tool in agent._function_toolset.tools.values():
        schema = tool.function_schema
        tools.append(
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": schema.json_schema,
            }
        )
    payload = json.dumps(
        {
            "instructions": agent._instructions,
            "tools": tools,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return token_counter.count_text(payload) + PROVIDER_FRAMING_TOKENS

"""Application composition root."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from ghostwheel.agent import (
    create_chat_agent,
    create_formatter,
    create_review_agent,
    create_tool_deps,
)
from ghostwheel.config import AppConfig
from ghostwheel.events import AgentEvent, TextOutput
from ghostwheel.pydantic_runner import PydanticAgentRunner
from ghostwheel.review import ReviewService
from ghostwheel.rich_ui import AppInfo, RichPresenter
from ghostwheel.session import ChatSession, HistoryPolicy
from ghostwheel.tools import DEFAULT_TOOL_CATALOG, ToolCatalog
from ghostwheel.tools.deps import ToolDeps


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
    chat_runner = PydanticAgentRunner(
        create_chat_agent(config, catalog=catalog),
        deps,
        event_sink=active_event_sink,
    )

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
            max_turns=config.history.max_turns,
            max_messages=config.history.max_messages,
            max_bytes=config.history.max_bytes,
            response_reserve_bytes=config.history.response_reserve_bytes,
        ),
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

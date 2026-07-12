"""UI-neutral application composition root."""

from __future__ import annotations

import asyncio
import threading
import weakref
from collections.abc import Awaitable, Callable
from concurrent.futures import Future as ConcurrentFuture
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pydantic_ai import Agent

from ghostwheel.agent_factory import (
    chat_agent_blueprint,
    compaction_agent_blueprint,
    create_tool_deps,
    review_agent_blueprint,
    review_fallback_agent_blueprint,
)
from ghostwheel.agent_blueprint import AgentBlueprint
from ghostwheel.app_info import AppInfo
from ghostwheel.compaction import HistoryCompactor
from ghostwheel.config import AppConfig
from ghostwheel.event_dispatcher import EventDispatcher, EventSink, deliver_event
from ghostwheel.events import TextOutput
from ghostwheel.history import HistoryPolicy
from ghostwheel.pydantic_runner import PydanticAgentRunner
from ghostwheel.review import ReviewService
from ghostwheel.session import ChatSession
from ghostwheel.tools import DEFAULT_TOOL_CATALOG, ToolCatalog
from ghostwheel.tools.deps import ToolDeps
from ghostwheel.token_counting import TextTokenCounter, TiktokenTokenCounter

if TYPE_CHECKING:
    from rich.console import Console as _ConsoleType

    from ghostwheel.rich_ui import RichPresenter as _RichPresenterType
else:
    # Keep the composition root free of a runtime dependency on either terminal
    # adapter while leaving its public annotations resolvable through
    # ``typing.get_type_hints``.
    _ConsoleType = Any
    _RichPresenterType = Any

PROVIDER_FRAMING_TOKENS = 256


class _RuntimeState(Enum):
    NEW = auto()
    ENTERING = auto()
    RUNNING = auto()
    CLOSING = auto()
    CLOSED = auto()


@dataclass(slots=True, weakref_slot=True)
class Runtime:
    """Own application services and every external resource they use."""

    session: ChatSession
    reviews: ReviewService
    tool_deps: ToolDeps
    app_info: AppInfo
    _agents: tuple[Agent[Any, Any], ...] = field(default=(), repr=False)
    _exit_stack: AsyncExitStack | None = field(default=None, init=False, repr=False)
    _state: _RuntimeState = field(
        default=_RuntimeState.NEW,
        init=False,
        repr=False,
    )
    _state_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )
    _transition: ConcurrentFuture[None] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _owner_loop: asyncio.AbstractEventLoop | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _cleanup_task: asyncio.Task[None] | None = field(
        default=None,
        init=False,
        repr=False,
    )

    @property
    def is_closed(self) -> bool:
        with self._state_lock:
            return self._state is _RuntimeState.CLOSED

    async def __aenter__(self) -> Runtime:
        loop = asyncio.get_running_loop()
        with self._state_lock:
            if self._state in {_RuntimeState.CLOSING, _RuntimeState.CLOSED}:
                raise RuntimeError("Runtime is closed")
            if self._state is not _RuntimeState.NEW:
                raise RuntimeError("Runtime is already running")
            self._state = _RuntimeState.ENTERING
            self._owner_loop = loop
            self._transition = ConcurrentFuture()

        try:
            try:
                stack = await _enter_agents(self._agents)
            except BaseException:
                self.tool_deps.close()
                raise
        except BaseException as error:
            self._finish_close(error)
            raise

        with self._state_lock:
            transition = self._transition
            self._transition = None
            self._exit_stack = stack
            self._state = _RuntimeState.RUNNING
        assert transition is not None
        transition.set_result(None)
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            with self._state_lock:
                state = self._state
                if state is _RuntimeState.CLOSED:
                    return
                if state in {_RuntimeState.ENTERING, _RuntimeState.CLOSING}:
                    transition = self._transition
                    wait_for_entry = state is _RuntimeState.ENTERING
                    stack = None
                else:
                    if state is _RuntimeState.RUNNING and self._owner_loop is not loop:
                        raise RuntimeError(
                            "Runtime must be closed from the event loop that entered it"
                        )
                    transition = ConcurrentFuture()
                    self._transition = transition
                    self._state = _RuntimeState.CLOSING
                    self._owner_loop = loop
                    stack = self._exit_stack
                    self._exit_stack = None
                    wait_for_entry = False
                    break

            assert transition is not None
            if wait_for_entry:
                await _await_transition(transition)
                continue
            await _await_close_completion(transition)
            return

        self._cleanup_task = loop.create_task(
            self._close_owned(stack),
            name="ghostwheel-runtime-cleanup",
        )
        await _await_close_completion(transition)

    async def _close_owned(self, stack: AsyncExitStack | None) -> None:
        try:
            try:
                if stack is None:
                    stack = await _enter_agents(self._agents)
                await stack.aclose()
            finally:
                self.tool_deps.close()
        except BaseException as error:
            self._finish_close(error)
        else:
            self._finish_close(None)

    def _finish_close(self, error: BaseException | None) -> None:
        with self._state_lock:
            transition = self._transition
            self._transition = None
            self._exit_stack = None
            self._owner_loop = None
            self._cleanup_task = None
            self._state = _RuntimeState.CLOSED
        assert transition is not None
        if error is None:
            transition.set_result(None)
        else:
            transition.set_exception(error)

    def _close_unentered_synchronously(self) -> bool:
        """Claim and close a new runtime on the synchronous helper loop."""

        with self._state_lock:
            if self._state is _RuntimeState.CLOSED:
                return True
            if self._state is not _RuntimeState.NEW:
                return False
            self._state = _RuntimeState.CLOSING
            self._transition = ConcurrentFuture()

        _run_cleanup(self._start_claimed_close, suppress_errors=False)
        return True

    async def _start_claimed_close(self) -> None:
        transition = self._transition
        assert transition is not None
        self._cleanup_task = asyncio.create_task(
            self._close_owned(None),
            name="ghostwheel-runtime-cleanup",
        )
        await _await_close_completion(transition)

    def close(self) -> None:
        """Close from synchronous code; async callers must use ``aclose``."""

        if self.is_closed:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.aclose())
            return
        raise RuntimeError("Use 'await runtime.aclose()' inside an event loop")


async def _await_transition(transition: ConcurrentFuture[None]) -> None:
    """Await a lifecycle transition without letting waiter cancellation own it."""

    loop = asyncio.get_running_loop()
    waiter = loop.create_future()

    def complete(source: ConcurrentFuture[None]) -> None:
        def copy_result() -> None:
            if waiter.done():
                return
            if source.cancelled():
                waiter.cancel()
                return
            error = source.exception()
            if error is None:
                waiter.set_result(None)
            else:
                waiter.set_exception(error)

        try:
            loop.call_soon_threadsafe(copy_result)
        except RuntimeError:
            # A cancelled waiter may outlive the event loop that awaited it. The
            # shared transition itself remains available to other callers.
            pass

    transition.add_done_callback(complete)
    await waiter


async def _await_close_completion(transition: ConcurrentFuture[None]) -> None:
    """Observe shared teardown while deferring this caller's cancellation."""

    cancellation: asyncio.CancelledError | None = None
    cleanup_error: BaseException | None = None
    while not transition.done():
        try:
            await _await_transition(transition)
        except asyncio.CancelledError as error:
            task = asyncio.current_task()
            if task is None or task.cancelling() == 0:
                cleanup_error = error
                break
            cancellation = error
        except BaseException as error:
            cleanup_error = error
            break

    if transition.done():
        cleanup_error = transition.exception()

    if cancellation is not None:
        if cleanup_error is not None:
            raise cancellation from cleanup_error
        raise cancellation
    if cleanup_error is not None:
        raise cleanup_error


_APPLICATION_RUNTIMES: weakref.WeakKeyDictionary[
    ChatSession,
    weakref.ReferenceType[Runtime],
] = weakref.WeakKeyDictionary()


async def _enter_agents(
    agents: tuple[Agent[Any, Any], ...],
) -> AsyncExitStack:
    stack = AsyncExitStack()
    try:
        for agent in agents:
            await stack.enter_async_context(agent)
    except BaseException:
        await stack.aclose()
        raise
    return stack


async def _close_unentered_agents(agents: tuple[Agent[Any, Any], ...]) -> None:
    stack = await _enter_agents(agents)
    await stack.aclose()


def _run_cleanup(
    cleanup: Callable[[], Awaitable[None]],
    *,
    suppress_errors: bool = True,
) -> None:
    """Finish async cleanup before returning to a synchronous caller.

    Synchronous compatibility and failure paths cannot await when called from an
    active event loop. Unentered resources have no loop ownership, so a short-lived
    helper loop can close them before returning.
    """

    errors: list[BaseException] = []

    def run() -> None:
        try:
            asyncio.run(cleanup())
        except BaseException as error:
            errors.append(error)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        run()
        if errors and not suppress_errors:
            raise errors[0]
        return

    thread = threading.Thread(target=run, name="ghostwheel-cleanup")
    thread.start()
    thread.join()
    if errors and not suppress_errors:
        raise errors[0]


def _close_agents_after_construction_failure(
    agents: tuple[Agent[Any, Any], ...],
) -> None:
    if not agents:
        return
    _run_cleanup(lambda: _close_unentered_agents(agents))


def build_runtime(
    config: AppConfig,
    *,
    cwd: Path | None = None,
    catalog: ToolCatalog = DEFAULT_TOOL_CATALOG,
    event_sink: EventSink | None = None,
) -> Runtime:
    """Build application services without selecting or constructing a UI."""

    deps = create_tool_deps(config, cwd)
    agents: list[Agent[Any, Any]] = []
    try:
        app_info = AppInfo(
            workspace=str(deps.cwd),
            provider=config.chat_model.provider.value,
            model=config.chat_model.model,
            tool_profile=config.tools.profile.value,
        )
        token_counter = TiktokenTokenCounter()
        chat_blueprint = chat_agent_blueprint(config, catalog=catalog)
        initial_overhead_tokens = _estimate_chat_overhead(
            chat_blueprint,
            token_counter,
        )
        chat_agent = chat_blueprint.build()
        agents.append(chat_agent)
        compactor: HistoryCompactor | None = None
        if config.history.compaction.enabled:
            compaction_agent = compaction_agent_blueprint(config).build()
            agents.append(compaction_agent)
            compaction_runner = PydanticAgentRunner(compaction_agent, None)
            compactor = HistoryCompactor(
                compaction_runner,
                token_counter=token_counter,
                input_token_budget=config.history.compactor_input_tokens,
                summary_token_limit=config.history.compaction.summary_tokens,
            )
        review_agent = review_agent_blueprint(config, catalog=catalog).build()
        agents.append(review_agent)
        fallback_runner: PydanticAgentRunner | None = None
        if config.review.raw_fallback:
            fallback_agent = review_fallback_agent_blueprint(config).build()
            agents.append(fallback_agent)
            fallback_runner = PydanticAgentRunner(fallback_agent, None)

        chat_runner = PydanticAgentRunner(chat_agent, deps, event_sink=event_sink)

        async def review_events(event) -> None:
            # Structured model output may be JSON text. Keep that implementation
            # detail out of presenters while still showing thinking and tool activity.
            if event_sink is not None and not isinstance(event, TextOutput):
                await deliver_event(event_sink, event)

        review_runner = PydanticAgentRunner(
            review_agent,
            deps,
            event_sink=review_events if event_sink is not None else None,
        )
        session = ChatSession(
            chat_runner,
            history_policy=HistoryPolicy.from_config(
                config.history,
                token_counter=token_counter,
            ),
            compactor=compactor,
            initial_overhead_tokens=initial_overhead_tokens,
        )
        reviews = ReviewService(
            review_runner,
            raw_fallback=config.review.raw_fallback,
            fallback_runner=fallback_runner,
        )
        return Runtime(
            session=session,
            reviews=reviews,
            tool_deps=deps,
            app_info=app_info,
            _agents=tuple(agents),
        )
    except BaseException:
        _close_agents_after_construction_failure(tuple(agents))
        deps.close()
        raise


def _estimate_chat_overhead(
    blueprint: AgentBlueprint[Any, Any],
    token_counter: TextTokenCounter,
) -> int:
    """Estimate owned static instructions, tool schemas, and provider framing."""

    return (
        token_counter.count_text(blueprint.static_context_json())
        + PROVIDER_FRAMING_TOKENS
    )


@dataclass(frozen=True)
class Application:
    """Rich compatibility facade preserving the original dataclass fields."""

    session: ChatSession
    reviews: ReviewService
    presenter: _RichPresenterType
    tool_deps: ToolDeps

    def __post_init__(self) -> None:
        try:
            reference = _APPLICATION_RUNTIMES.get(self.session)
        except TypeError:
            # The compatibility constructor accepts arbitrary session-like values,
            # including objects that cannot participate in a weak-key registry.
            return
        runtime = reference() if reference is not None else None
        if runtime is not None and (
            self.session is runtime.session
            and self.reviews is runtime.reviews
            and self.tool_deps is runtime.tool_deps
        ):
            object.__setattr__(self, "_runtime", runtime)

    @property
    def runtime(self) -> Runtime | None:
        return cast(Runtime | None, getattr(self, "_runtime", None))

    async def __aenter__(self) -> Application:
        runtime = self.runtime
        if runtime is not None:
            await runtime.__aenter__()
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        runtime = self.runtime
        if runtime is not None:
            await runtime.aclose()
        else:
            self.tool_deps.close()

    def close(self) -> None:
        runtime = self.runtime
        if runtime is not None:
            if runtime._close_unentered_synchronously():
                # Legacy Application callers may invoke synchronous close from an
                # async embedding without ever entering the new Runtime context.
                # Those agents have no loop ownership yet, so a helper loop can
                # close them synchronously without crossing event-loop boundaries.
                return
            # An entered runtime belongs to its current event loop and must never
            # be moved to the helper thread.
            runtime.close()
        else:
            self.tool_deps.close()


def build_application(
    config: AppConfig,
    console: _ConsoleType,
    *,
    cwd: Path | None = None,
    catalog: ToolCatalog = DEFAULT_TOOL_CATALOG,
    live_ui: bool = False,
    event_sink: EventSink | None = None,
) -> Application:
    """Build the former Rich-specific application facade."""

    from ghostwheel.rich_ui import RichPresenter

    dispatcher = EventDispatcher() if event_sink is None else None
    active_sink = event_sink if event_sink is not None else dispatcher
    assert active_sink is not None
    runtime = build_runtime(
        config,
        cwd=cwd,
        catalog=catalog,
        event_sink=active_sink,
    )
    try:
        presenter = RichPresenter(
            console,
            app_info=runtime.app_info,
            live=live_ui,
        )
        if dispatcher is not None:
            dispatcher.bind(presenter.handle_event)
        _APPLICATION_RUNTIMES[runtime.session] = weakref.ref(runtime)
        return Application(
            session=runtime.session,
            reviews=runtime.reviews,
            presenter=presenter,
            tool_deps=runtime.tool_deps,
        )
    except BaseException:
        _close_runtime_after_failure(runtime)
        raise


def _close_runtime_after_failure(runtime: Runtime) -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        runtime.close()
        return
    _run_cleanup(runtime.aclose)

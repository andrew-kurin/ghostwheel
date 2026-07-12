import asyncio
from pathlib import Path

import pytest

import ghostwheel.bootstrap as bootstrap_module
from ghostwheel.app_info import AppInfo
from ghostwheel.bootstrap import Runtime, build_runtime
from ghostwheel.config import Settings
from ghostwheel.review import ReviewService
from ghostwheel.session import ChatSession


def test_build_runtime_is_ui_neutral_and_owns_resolved_metadata(tmp_path) -> None:
    config = Settings(_env_file=None).resolve()

    runtime = build_runtime(config, cwd=tmp_path)
    try:
        assert isinstance(runtime.session, ChatSession)
        assert isinstance(runtime.reviews, ReviewService)
        assert runtime.tool_deps.cwd == tmp_path.resolve()
        assert runtime.tool_deps.limits is config.tools.limits
        assert runtime.app_info.workspace == str(tmp_path.resolve())
        assert runtime.app_info.provider == config.chat_model.provider.value
        assert runtime.app_info.model == config.chat_model.model
        assert runtime.app_info.tool_profile == config.tools.profile.value
        assert runtime.session.estimated_context_tokens > 256
        assert runtime.session.context_tokens_estimated is True
    finally:
        runtime.close()


def test_runtime_async_context_owns_agent_and_tool_lifetimes() -> None:
    class ManagedAgent:
        entered = 0
        exited = 0

        async def __aenter__(self) -> "ManagedAgent":
            self.entered += 1
            return self

        async def __aexit__(self, *_exc_info: object) -> None:
            self.exited += 1

    class FakeDeps:
        closed = False

        def close(self) -> None:
            self.closed = True

    agent = ManagedAgent()
    deps = FakeDeps()
    runtime = Runtime(
        session=object(),  # type: ignore[arg-type]
        reviews=object(),  # type: ignore[arg-type]
        tool_deps=deps,  # type: ignore[arg-type]
        app_info=AppInfo(".", "provider", "model", "read-only"),
        _agents=(agent,),  # type: ignore[arg-type]
    )

    async def scenario() -> None:
        async with runtime:
            assert agent.entered == 1
            assert runtime.is_closed is False

    asyncio.run(scenario())

    assert agent.exited == 1
    assert deps.closed is True
    assert runtime.is_closed is True


def test_runtime_rejects_a_concurrent_enter_before_entry_finishes() -> None:
    class YieldingAgent:
        def __init__(self) -> None:
            self.entry_started = asyncio.Event()
            self.allow_entry = asyncio.Event()
            self.entered = 0
            self.exited = 0

        async def __aenter__(self) -> "YieldingAgent":
            self.entered += 1
            if self.entered == 1:
                self.entry_started.set()
                await self.allow_entry.wait()
            return self

        async def __aexit__(self, *_exc_info: object) -> None:
            self.exited += 1

    class FakeDeps:
        close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    async def scenario() -> None:
        agent = YieldingAgent()
        deps = FakeDeps()
        runtime = Runtime(
            session=object(),  # type: ignore[arg-type]
            reviews=object(),  # type: ignore[arg-type]
            tool_deps=deps,  # type: ignore[arg-type]
            app_info=AppInfo(".", "provider", "model", "read-only"),
            _agents=(agent,),  # type: ignore[arg-type]
        )

        entering = asyncio.create_task(runtime.__aenter__())
        await agent.entry_started.wait()

        with pytest.raises(RuntimeError, match="already running"):
            await runtime.__aenter__()

        agent.allow_entry.set()
        assert await entering is runtime
        await runtime.aclose()

        assert agent.entered == 1
        assert agent.exited == 1
        assert deps.close_calls == 1
        assert runtime.is_closed is True

    asyncio.run(scenario())


def test_runtime_close_waits_for_an_in_progress_enter() -> None:
    class YieldingAgent:
        def __init__(self) -> None:
            self.entry_started = asyncio.Event()
            self.allow_entry = asyncio.Event()
            self.entered = 0
            self.exited = 0

        async def __aenter__(self) -> "YieldingAgent":
            self.entered += 1
            self.entry_started.set()
            await self.allow_entry.wait()
            return self

        async def __aexit__(self, *_exc_info: object) -> None:
            self.exited += 1

    class FakeDeps:
        close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    async def scenario() -> None:
        agent = YieldingAgent()
        deps = FakeDeps()
        runtime = Runtime(
            session=object(),  # type: ignore[arg-type]
            reviews=object(),  # type: ignore[arg-type]
            tool_deps=deps,  # type: ignore[arg-type]
            app_info=AppInfo(".", "provider", "model", "read-only"),
            _agents=(agent,),  # type: ignore[arg-type]
        )

        entering = asyncio.create_task(runtime.__aenter__())
        await agent.entry_started.wait()
        closing = asyncio.create_task(runtime.aclose())
        await asyncio.sleep(0)
        assert closing.done() is False

        agent.allow_entry.set()
        assert await entering is runtime
        await closing

        assert agent.entered == 1
        assert agent.exited == 1
        assert deps.close_calls == 1
        assert runtime.is_closed is True

    asyncio.run(scenario())


def test_concurrent_runtime_close_is_idempotent_for_real_clients(
    tmp_path: Path,
) -> None:
    class YieldingExitAgent:
        def __init__(self) -> None:
            self.exit_started = asyncio.Event()
            self.allow_exit = asyncio.Event()
            self.entered = 0
            self.exited = 0

        async def __aenter__(self) -> "YieldingExitAgent":
            self.entered += 1
            return self

        async def __aexit__(self, *_exc_info: object) -> None:
            self.exited += 1
            self.exit_started.set()
            await self.allow_exit.wait()

    runtime = build_runtime(Settings(_env_file=None).resolve(), cwd=tmp_path)
    clients = [agent.model.client for agent in runtime._agents]
    gate = YieldingExitAgent()
    runtime._agents = (*runtime._agents, gate)  # type: ignore[assignment]

    def is_closed(client: object) -> bool:
        value = getattr(client, "is_closed")
        return bool(value() if callable(value) else value)

    async def scenario() -> None:
        await runtime.__aenter__()
        first_close = asyncio.create_task(runtime.aclose())
        await gate.exit_started.wait()
        second_close = asyncio.create_task(runtime.aclose())
        await asyncio.sleep(0)

        assert second_close.done() is False
        assert gate.entered == 1
        assert gate.exited == 1

        gate.allow_exit.set()
        await asyncio.gather(first_close, second_close)

    asyncio.run(scenario())

    assert runtime.is_closed is True
    assert all(is_closed(client) for client in clients)
    assert runtime.tool_deps.workspace.is_closed is True


def test_cancelling_close_owner_does_not_cancel_shared_teardown() -> None:
    class YieldingExitAgent:
        def __init__(self) -> None:
            self.exit_started = asyncio.Event()
            self.allow_exit = asyncio.Event()
            self.entered = 0
            self.exit_completed = 0

        async def __aenter__(self) -> "YieldingExitAgent":
            self.entered += 1
            return self

        async def __aexit__(self, *_exc_info: object) -> None:
            self.exit_started.set()
            await self.allow_exit.wait()
            self.exit_completed += 1

    class FakeDeps:
        close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    async def scenario() -> None:
        agent = YieldingExitAgent()
        deps = FakeDeps()
        runtime = Runtime(
            session=object(),  # type: ignore[arg-type]
            reviews=object(),  # type: ignore[arg-type]
            tool_deps=deps,  # type: ignore[arg-type]
            app_info=AppInfo(".", "provider", "model", "read-only"),
            _agents=(agent,),  # type: ignore[arg-type]
        )
        await runtime.__aenter__()

        owner = asyncio.create_task(runtime.aclose())
        await agent.exit_started.wait()
        waiter = asyncio.create_task(runtime.aclose())
        await asyncio.sleep(0)

        owner.cancel()
        await asyncio.sleep(0)
        assert owner.done() is False
        assert waiter.done() is False
        assert agent.exit_completed == 0

        agent.allow_exit.set()
        with pytest.raises(asyncio.CancelledError):
            await owner
        await waiter

        assert agent.entered == 1
        assert agent.exit_completed == 1
        assert deps.close_calls == 1
        assert runtime.is_closed is True
        await runtime.aclose()
        assert deps.close_calls == 1

    asyncio.run(scenario())


def test_runtime_closes_real_provider_clients(tmp_path: Path) -> None:
    runtime = build_runtime(Settings(_env_file=None).resolve(), cwd=tmp_path)
    clients = [agent.model.client for agent in runtime._agents]
    assert len(clients) == 4

    def is_closed(client: object) -> bool:
        value = getattr(client, "is_closed")
        return bool(value() if callable(value) else value)

    assert all(not is_closed(client) for client in clients)

    runtime.close()

    assert all(is_closed(client) for client in clients)


def test_disabled_compaction_omits_compactor_agent_and_accepts_tiny_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = Settings(
        history_context_window_tokens=500,
        compaction_enabled=False,
        compaction_reserve_tokens=100,
        compaction_keep_recent_tokens=100,
        compaction_summary_tokens=100,
        _env_file=None,
    ).resolve()
    assert config.history.compactor_input_tokens == -112

    def unexpected_compactor(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("disabled compaction must not build an agent")

    monkeypatch.setattr(
        bootstrap_module,
        "compaction_agent_blueprint",
        unexpected_compactor,
    )

    runtime = build_runtime(config, cwd=tmp_path)
    clients = [agent.model.client for agent in runtime._agents]

    def is_closed(client: object) -> bool:
        value = getattr(client, "is_closed")
        return bool(value() if callable(value) else value)

    assert runtime.session.compaction_enabled is False
    assert len(clients) == 3
    assert all(not is_closed(client) for client in clients)

    runtime.close()

    assert runtime.is_closed is True
    assert all(is_closed(client) for client in clients)
    assert runtime.tool_deps.workspace.is_closed is True


def test_disabled_review_fallback_omits_fallback_agent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = Settings(review_raw_fallback=False, _env_file=None).resolve()

    def unexpected_fallback(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("disabled review fallback must not build an agent")

    monkeypatch.setattr(
        bootstrap_module,
        "review_fallback_agent_blueprint",
        unexpected_fallback,
    )

    runtime = build_runtime(config, cwd=tmp_path)
    try:
        assert len(runtime._agents) == 3
        assert runtime.reviews.raw_fallback is False
        assert runtime.reviews._fallback_runner is None
    finally:
        runtime.close()


def test_build_runtime_closes_owned_dependencies_when_composition_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeDeps:
        cwd = tmp_path
        closed = False

        def close(self) -> None:
            self.closed = True

    deps = FakeDeps()
    config = Settings(_env_file=None).resolve()
    monkeypatch.setattr(
        bootstrap_module,
        "create_tool_deps",
        lambda _config, _cwd: deps,
    )

    def fail_blueprint(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("agent construction failed")

    monkeypatch.setattr(bootstrap_module, "chat_agent_blueprint", fail_blueprint)

    with pytest.raises(RuntimeError, match="agent construction failed"):
        build_runtime(config, cwd=tmp_path)

    assert deps.closed is True


def test_build_runtime_closes_agents_created_before_composition_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class ManagedAgent:
        entered = 0
        exited = 0

        async def __aenter__(self) -> "ManagedAgent":
            self.entered += 1
            return self

        async def __aexit__(self, *_exc_info: object) -> None:
            self.exited += 1

    class FakeBlueprint:
        def __init__(self, agent: ManagedAgent) -> None:
            self.agent = agent

        def static_context_json(self) -> str:
            return "{}"

        def build(self) -> ManagedAgent:
            return self.agent

    agent = ManagedAgent()
    config = Settings(_env_file=None).resolve()
    monkeypatch.setattr(
        bootstrap_module,
        "chat_agent_blueprint",
        lambda *_args, **_kwargs: FakeBlueprint(agent),
    )

    def fail_compaction_blueprint(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("later agent construction failed")

    monkeypatch.setattr(
        bootstrap_module,
        "compaction_agent_blueprint",
        fail_compaction_blueprint,
    )

    async def scenario() -> None:
        with pytest.raises(RuntimeError, match="later agent construction failed"):
            build_runtime(config, cwd=tmp_path)

        # Synchronous construction guarantees cleanup before propagating the
        # failure, even when its caller already owns an event loop.
        assert agent.entered == 1
        assert agent.exited == 1

    asyncio.run(scenario())

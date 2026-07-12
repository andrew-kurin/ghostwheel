# Ghostwheel architecture

Ghostwheel uses a small ports-and-adapters structure. Dependencies point from
entry points and adapters toward application contracts and state; compatibility
modules are not used by canonical implementation modules.

## Runtime flow

```text
cli / terminal UI
        |
        v
command controller ---- presenter port
        |
        v
ChatSession / ReviewService
        |
        v
AgentRunner / ContextCompactor contracts
        |
        v
Pydantic-AI adapter ---- agent blueprints ---- provider adapters
        |
        v
tool catalog ---- ToolDeps ---- Workspace / CommandRunner
```

## Module ownership

- `controller.py` owns command parsing and UI-neutral command orchestration.
- `bootstrap.py` composes runtime services and does not select a UI.
- `runtime_contracts.py` owns agent outcomes and runner protocols.
- `history.py` owns conversation state, message atoms, and compaction policy.
- `session.py` owns chat-session orchestration and re-exports its former public
  contract names for compatibility.
- `presentation.py` reduces runtime events into renderer-neutral turn state.
- `terminal_ui.py` is the sole presentation adapter. It uses Rich for a bounded
  transient preview plus completed output in the primary terminal buffer.
- `terminal_composer.py` owns prompt-toolkit session construction, editor key
  bindings, completion, and private prompt-history persistence.
- `terminal_io.py` owns raw tty and signal restoration, active-turn key
  monitoring, and asynchronous redirected-input buffering.
- `agent_blueprint.py` and `agent_factory.py` own SDK-independent agent inputs
  and Pydantic-AI construction. `agent.py` is a compatibility facade.
- `model_config.py` and `tool_config.py` contain lightweight resolved values;
  `providers.py` and `tools/` contain implementation adapters.
- `tools/workspace.py` is the sole owner of descriptor-safe filesystem access.

## Dependency rules

- Canonical modules do not import `agent.py` or `models.py` compatibility
  facades.
- Runtime services and adapters import `runtime_contracts.py` and `history.py`
  directly rather than importing contract types from `session.py`.
- The terminal UI does not import the CLI or runtime composition root.
- Configuration modules do not import the tool implementation graph.
- Bootstrap estimates static context from owned agent blueprints, not private
  Pydantic-AI attributes.

`tests/test_architecture.py` enforces these rules with source-level dependency
checks.

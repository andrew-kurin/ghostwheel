import importlib


def test_agent_module_import_does_not_create_runtime_singletons() -> None:
    module = importlib.import_module("ghostwheel.agent")

    assert not hasattr(module, "config")
    assert not hasattr(module, "model")
    assert not hasattr(module, "formatter_model")
    assert not hasattr(module, "agent")
    assert not hasattr(module, "formatter")

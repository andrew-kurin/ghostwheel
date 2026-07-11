import importlib

import pytest

from ghostwheel.config import ObservabilityConfig


def test_observability_is_disabled_without_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    telemetry = importlib.import_module("ghostwheel.telemetry")
    monkeypatch.setattr(telemetry, "_configured", False)
    configured: list[dict[str, object]] = []
    monkeypatch.setattr(
        telemetry.logfire,
        "configure",
        lambda **kwargs: configured.append(kwargs),
    )

    active = telemetry.configure_observability(
        ObservabilityConfig(
            enabled=False,
            include_content=False,
            send_to_logfire="if-token-present",
        )
    )

    assert active is False
    assert configured == []


def test_observability_content_capture_requires_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    telemetry = importlib.import_module("ghostwheel.telemetry")
    monkeypatch.setattr(telemetry, "_configured", False)
    configure_calls: list[dict[str, object]] = []
    instrument_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        telemetry.logfire,
        "configure",
        lambda **kwargs: configure_calls.append(kwargs),
    )
    monkeypatch.setattr(
        telemetry.logfire,
        "instrument_pydantic_ai",
        lambda **kwargs: instrument_calls.append(kwargs),
    )

    active = telemetry.configure_observability(
        ObservabilityConfig(
            enabled=True,
            include_content=False,
            send_to_logfire=False,
        )
    )

    assert active is True
    assert configure_calls == [{"console": False, "send_to_logfire": False}]
    assert instrument_calls == [
        {"include_content": False, "include_binary_content": False}
    ]

"""Optional process-level observability wiring."""

import logfire

from ghostwheel.config import ObservabilityConfig

_configured = False


def configure_observability(config: ObservabilityConfig) -> bool:
    """Configure Logfire once when explicitly enabled.

    Returns whether instrumentation is active. Model and tool content remain
    excluded unless the user separately opts into content capture.
    """
    global _configured

    if not config.enabled:
        return False
    if _configured:
        return True

    logfire.configure(
        console=False,
        send_to_logfire=config.send_to_logfire,
    )
    logfire.instrument_pydantic_ai(
        include_content=config.include_content,
        include_binary_content=False,
    )
    _configured = True
    return True

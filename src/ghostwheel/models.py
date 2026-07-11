"""Backward-compatible model/provider API.

Framework-neutral values live in :mod:`ghostwheel.model_config`; provider SDK
integration lives in :mod:`ghostwheel.providers`.
"""

from ghostwheel.model_config import ModelProvider, ModelSpec
from ghostwheel.providers import (
    LLAMA_CPP_PROFILE,
    SUPPORTED_PROVIDERS,
    ProviderRegistration,
    build_model,
    default_base_url,
    formatter_model_settings,
    normalize_provider,
    provider_registration,
    structured_output_model_settings,
    validate_provider,
)

__all__ = [
    "LLAMA_CPP_PROFILE",
    "SUPPORTED_PROVIDERS",
    "ModelProvider",
    "ModelSpec",
    "ProviderRegistration",
    "build_model",
    "default_base_url",
    "formatter_model_settings",
    "normalize_provider",
    "provider_registration",
    "structured_output_model_settings",
    "validate_provider",
]

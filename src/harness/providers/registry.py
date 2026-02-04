"""Provider registry - get_provider() returns configured provider."""

import os
from typing import TYPE_CHECKING

from .base import Provider

if TYPE_CHECKING:
    from ..logging import TraceLogger, StdoutLogger


# Global provider instance
_provider_instance: Provider | None = None
_logger_instance: "TraceLogger | StdoutLogger | None" = None
_configured_model: str | None = None


def configure(
    model: str | None = None,
    logger: "TraceLogger | StdoutLogger | None" = None,
) -> None:
    """
    Configure the global provider.
    
    Args:
        model: Model name (default: HARNESS_MODEL env var or claude-sonnet-4-5-20250514)
        logger: Logger for automatic completion logging
    """
    global _provider_instance, _logger_instance, _configured_model
    
    model = model or os.environ.get("HARNESS_MODEL", "claude-sonnet-4-5-20250514")
    
    from .litellm_provider import LiteLLMProvider
    _provider_instance = LiteLLMProvider(model, logger=logger)
    _logger_instance = logger
    _configured_model = model


def get_provider(
    model: str | None = None,
    logger: "TraceLogger | StdoutLogger | None" = None,
) -> Provider:
    """
    Get the configured provider.
    
    If model differs from configured model, creates a new provider.
    """
    global _provider_instance, _configured_model
    
    # If no model specified, use configured or auto-configure
    if model is None:
        if _provider_instance is None:
            configure(logger=logger)
        return _provider_instance
    
    # If model matches configured, return existing
    if model == _configured_model and _provider_instance is not None:
        return _provider_instance
    
    # Create new provider for this model
    from .litellm_provider import LiteLLMProvider
    return LiteLLMProvider(model, logger=logger or _logger_instance)


def reset() -> None:
    """Reset provider configuration (mainly for testing)."""
    global _provider_instance, _logger_instance, _configured_model
    _provider_instance = None
    _logger_instance = None
    _configured_model = None

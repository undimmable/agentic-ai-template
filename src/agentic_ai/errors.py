"""Exception hierarchy for the agentic_ai runtime."""

from __future__ import annotations


class AgenticError(Exception):
    """Base class for every error raised by the runtime."""


class ConfigError(AgenticError):
    """Raised when a declarative config file is missing, malformed or invalid."""


class RegistryError(AgenticError):
    """Raised for duplicate, missing or unknown registry entries."""


class ProviderError(AgenticError):
    """Raised when an LLM provider fails to produce a usable response."""


class ToolError(AgenticError):
    """Raised when a tool cannot be executed."""


class MaxIterationsError(AgenticError):
    """Raised when the agent loop hits the configured iteration budget."""

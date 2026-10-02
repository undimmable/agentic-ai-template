"""agentic_ai - a small, config-driven, extensible agent runtime."""

from __future__ import annotations

from .agent import Agent
from .config import (
    DEFAULT_SYSTEM_PROMPT,
    AgentConfig,
    ProviderConfig,
    SessionConfig,
    ToolConfig,
    dump_example_config,
    load_config,
    loads_config,
)
from .errors import (
    AgenticError,
    ConfigError,
    MaxIterationsError,
    ProviderError,
    RegistryError,
    ToolError,
)
from .messages import LLMResponse, ToolCall
from .sessions import SessionInfo, SessionStore

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "Agent",
    "AgentConfig",
    "AgenticError",
    "ConfigError",
    "LLMResponse",
    "MaxIterationsError",
    "ProviderConfig",
    "ProviderError",
    "RegistryError",
    "SessionConfig",
    "SessionInfo",
    "SessionStore",
    "ToolCall",
    "ToolConfig",
    "ToolError",
    "__version__",
    "dump_example_config",
    "load_config",
    "loads_config",
]

"""Declarative configuration for the agent runtime.

This is the "NixOS-flavoured" part of the template: a single YAML file
describes the agent, its model provider and the tools it is allowed to use.
Every value has a sensible default, so a config can grow from one line to a
full system without breaking.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .errors import ConfigError

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

DEFAULT_SYSTEM_PROMPT = (
    "You are an agentic AI collaborating with a human on a software project. "
    "You can inspect and modify the workspace using the tools you are given. "
    "Work in small, verifiable steps, keep the human informed, and finish with "
    "a concise final answer."
)

_DEFAULT_TOOLS = ["read_file", "write_file", "list_dir"]

# Defaults per provider type. Values from a config file always win; these only
# fill the gaps so a one-line config works.
PROVIDER_PROFILES: Dict[str, Dict[str, Any]] = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
    },
    "openai-compatible": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "stub": {"base_url": "", "model": "stub", "api_key_env": None},
}

DEFAULT_PROVIDER_TYPE = "openai"
_DEFAULT_TIMEOUT = 60.0


def detect_provider_type() -> str:
    """Choose a default provider from the API keys present in the environment.

    DeepSeek wins over OpenAI when ``DEEPSEEK_API_KEY`` is set. An explicit
    ``provider.type`` in the config always takes precedence over detection.
    """

    if os.environ.get("DEEPSEEK_API_KEY"):
        return "deepseek"
    return DEFAULT_PROVIDER_TYPE


# --------------------------------------------------------------------------- #
# loading helpers
# --------------------------------------------------------------------------- #
def _expand_env(value: Any, *, path: str = "$") -> Any:
    """Recursively expand ``${VAR}`` / ``${VAR:-default}`` inside strings."""

    if isinstance(value, str):

        def replace(match: "re.Match[str]") -> str:
            name, default = match.group(1), match.group(2)
            if name in os.environ:
                return os.environ[name]
            if default is not None:
                return default
            raise ConfigError(
                f"Environment variable '{name}' referenced at {path} is not set"
            )

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [_expand_env(item, path=f"{path}[{i}]") for i, item in enumerate(value)]
    if isinstance(value, dict):
        return {
            key: _expand_env(item, path=f"{path}.{key}") for key, item in value.items()
        }
    return value


def _expect_mapping(value: Any, path: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{path} must be a mapping, got {type(value).__name__}")
    return value


def _reject_unknown(data: Dict[str, Any], allowed: Any, path: str) -> None:
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        raise ConfigError(
            f"Unknown key(s) at {path}: {', '.join(unknown)}. "
            f"Allowed keys: {', '.join(sorted(allowed))}"
        )


def _as_str(value: Any, path: str, *, allow_none: bool = False) -> Optional[str]:
    if value is None and allow_none:
        return None
    if not isinstance(value, str):
        raise ConfigError(f"{path} must be a string, got {type(value).__name__}")
    return value


def _as_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{path} must be an integer, got {type(value).__name__}")
    return value


def _as_float(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path} must be a number, got {type(value).__name__}")
    return float(value)


def _as_str_list(value: Any, path: str) -> List[str]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ConfigError(f"{path} must be a list of strings")
    return list(value)


def _as_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{path} must be a boolean, got {type(value).__name__}")
    return value


# --------------------------------------------------------------------------- #
# config dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class ProviderConfig:
    type: str = "openai"
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    api_key_env: Optional[str] = "OPENAI_API_KEY"
    temperature: Optional[float] = None
    timeout: float = 60.0
    options: Dict[str, Any] = field(default_factory=dict)
    type_explicit: bool = False

    @classmethod
    def from_dict(cls, data: Any, path: str = "provider") -> "ProviderConfig":
        data = _expect_mapping(data or {}, path)
        _reject_unknown(
            data,
            {
                "type",
                "base_url",
                "model",
                "api_key",
                "api_key_env",
                "temperature",
                "timeout",
                "options",
            },
            path,
        )
        type_explicit = "type" in data
        provider_type = (
            _as_str(data["type"], f"{path}.type")
            if type_explicit
            else detect_provider_type()
        )
        profile = PROVIDER_PROFILES.get(
            provider_type, PROVIDER_PROFILES[DEFAULT_PROVIDER_TYPE]
        )
        options = _expect_mapping(data.get("options", {}) or {}, f"{path}.options")
        temperature = data.get("temperature")
        return cls(
            type=provider_type,
            base_url=_as_str(
                data.get("base_url", profile["base_url"]), f"{path}.base_url"
            ),
            model=_as_str(data.get("model", profile["model"]), f"{path}.model"),
            api_key=_as_str(data.get("api_key"), f"{path}.api_key", allow_none=True),
            api_key_env=_as_str(
                data.get("api_key_env", profile["api_key_env"]),
                f"{path}.api_key_env",
                allow_none=True,
            ),
            temperature=(
                None
                if temperature is None
                else _as_float(temperature, f"{path}.temperature")
            ),
            timeout=_as_float(data.get("timeout", _DEFAULT_TIMEOUT), f"{path}.timeout"),
            options=dict(options),
            type_explicit=type_explicit,
        )

    def resolve_api_key(self) -> Optional[str]:
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        return None


@dataclass
class ToolConfig:
    enabled: List[str] = field(default_factory=lambda: list(_DEFAULT_TOOLS))
    workspace: Optional[str] = None
    shell_timeout: float = 30.0

    @classmethod
    def from_dict(cls, data: Any, path: str = "tools") -> "ToolConfig":
        data = _expect_mapping(data or {}, path)
        _reject_unknown(data, {"enabled", "workspace", "shell_timeout"}, path)
        enabled = data.get("enabled")
        if enabled is None:
            enabled = list(_DEFAULT_TOOLS)
        return cls(
            enabled=_as_str_list(enabled, f"{path}.enabled"),
            workspace=_as_str(
                data.get("workspace"), f"{path}.workspace", allow_none=True
            ),
            shell_timeout=_as_float(
                data.get("shell_timeout", 30.0), f"{path}.shell_timeout"
            ),
        )


@dataclass
class SessionConfig:
    """Where conversations are persisted between runs.

    Sessions are stored in a small SQLite database so the TUI can list previous
    conversations and reload one. ``path`` defaults to
    :data:`~agentic_ai.sessions.DEFAULT_DB_PATH`; set ``enabled: false`` to run
    entirely in memory.
    """

    enabled: bool = True
    path: str = ".agentic/sessions.db"

    @classmethod
    def from_dict(cls, data: Any, path: str = "sessions") -> "SessionConfig":
        data = _expect_mapping(data or {}, path)
        _reject_unknown(data, {"enabled", "path"}, path)
        return cls(
            enabled=_as_bool(data.get("enabled", True), f"{path}.enabled"),
            path=_as_str(data.get("path", ".agentic/sessions.db"), f"{path}.path"),
        )


@dataclass
class AgentConfig:
    name: str = "agent"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_iterations: int = 10
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    tools: ToolConfig = field(default_factory=ToolConfig)
    sessions: SessionConfig = field(default_factory=SessionConfig)
    source: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Any) -> "AgentConfig":
        data = _expect_mapping(data or {}, "$")
        agent = _expect_mapping(data.get("agent", {}) or {}, "agent")
        _reject_unknown(agent, {"name", "system_prompt", "max_iterations"}, "agent")
        known = {"version", "agent", "provider", "tools", "sessions"}
        return cls(
            name=_as_str(agent.get("name", "agent"), "agent.name"),
            system_prompt=_as_str(
                agent.get("system_prompt", DEFAULT_SYSTEM_PROMPT), "agent.system_prompt"
            ),
            max_iterations=_as_int(
                agent.get("max_iterations", 10), "agent.max_iterations"
            ),
            provider=ProviderConfig.from_dict(data.get("provider", {})),
            tools=ToolConfig.from_dict(data.get("tools", {})),
            sessions=SessionConfig.from_dict(data.get("sessions", {})),
            extra={key: value for key, value in data.items() if key not in known},
        )


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def loads_config(text: str) -> AgentConfig:
    """Parse a YAML string into :class:`AgentConfig` (env expansion applied)."""

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML config: {exc}") from exc
    if data is None:
        data = {}
    return AgentConfig.from_dict(_expand_env(data))


def load_config(path: Any) -> AgentConfig:
    """Read and validate a config file from disk."""

    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")
    if config_path.is_dir():
        raise ConfigError(f"Config path is a directory, expected a file: {config_path}")
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Could not read config file {config_path}: {exc}") from exc
    config = loads_config(text)
    config.source = str(config_path)
    return config


def dump_example_config() -> str:
    """Return a commented example config, used by ``agentic init``."""

    return _EXAMPLE_CONFIG


_EXAMPLE_CONFIG = """\
# Declarative configuration for the agentic_ai runtime.
# Every value below is optional - the runtime ships with sensible defaults.
version: 1

agent:
  name: freyja
  system_prompt: |
    You are an agentic AI collaborating with a human on this repository.
    Inspect files before changing them, make small verifiable edits, and
    finish with a concise summary of what you did.
  max_iterations: 12

provider:
  # The provider is auto-detected from the environment:
  #   DEEPSEEK_API_KEY set -> deepseek, otherwise -> openai.
  # Uncomment `type` to pin one explicitly (an explicit type always wins).
  # type: deepseek
  # Any OpenAI-compatible /chat/completions endpoint works:
  #   base_url: https://api.deepseek.com/v1
  #   base_url: http://localhost:11434/v1   # Ollama, LM Studio, vLLM, ...
  # model: deepseek-chat
  # api_key: ${DEEPSEEK_API_KEY}          # ${VAR} and ${VAR:-default} are expanded
  temperature: 0.2
  timeout: 60

tools:
  enabled:
    - read_file
    - write_file
    - list_dir
    - shell
  workspace: .
  shell_timeout: 30

sessions:
  # Conversations are persisted in a small SQLite database so the TUI can
  # list previous sessions and reload one to continue where you left off.
  enabled: true
  path: .agentic/sessions.db
"""

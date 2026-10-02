"""Tests for the declarative config loader."""

from __future__ import annotations

import pytest

from agentic_ai.config import DEFAULT_SYSTEM_PROMPT, load_config, loads_config
from agentic_ai.errors import ConfigError


@pytest.fixture(autouse=True)
def _clear_provider_keys(monkeypatch):
    """Keep provider auto-detection deterministic across tests."""

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)


def test_minimal_config_uses_defaults():
    config = loads_config("agent:\n  name: test\n")

    assert config.name == "test"
    assert config.provider.type == "openai"
    assert config.provider.model == "gpt-4o-mini"
    assert config.tools.enabled == ["read_file", "write_file", "list_dir"]
    assert config.max_iterations == 10
    assert config.system_prompt == DEFAULT_SYSTEM_PROMPT


def test_empty_config_is_valid():
    config = loads_config("")
    assert config.name == "agent"
    assert config.provider.base_url == "https://api.openai.com/v1"


def test_full_config_roundtrip(tmp_path):
    path = tmp_path / "agent.yaml"
    path.write_text(
        """
agent:
  name: freyja
  max_iterations: 3
provider:
  type: openai
  base_url: http://localhost:11434/v1
  model: llama3
  temperature: 0.5
tools:
  enabled: [read_file, shell]
  workspace: /tmp/work
  shell_timeout: 5
""",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.source == str(path)
    assert config.name == "freyja"
    assert config.max_iterations == 3
    assert config.provider.model == "llama3"
    assert config.provider.temperature == 0.5
    assert config.tools.enabled == ["read_file", "shell"]
    assert config.tools.workspace == "/tmp/work"
    assert config.tools.shell_timeout == 5.0


def test_env_expansion(monkeypatch):
    monkeypatch.setenv("AGENTIC_TEST_KEY", "secret")
    config = loads_config("provider:\n  api_key: ${AGENTIC_TEST_KEY}\n")
    assert config.provider.api_key == "secret"


def test_env_expansion_default():
    config = loads_config("provider:\n  api_key: ${AGENTIC_TEST_MISSING:-fallback}\n")
    assert config.provider.api_key == "fallback"


def test_env_expansion_missing_raises(monkeypatch):
    monkeypatch.delenv("AGENTIC_TEST_MISSING", raising=False)
    with pytest.raises(ConfigError, match="AGENTIC_TEST_MISSING"):
        loads_config("provider:\n  api_key: ${AGENTIC_TEST_MISSING}\n")


def test_type_validation():
    with pytest.raises(ConfigError, match="max_iterations"):
        loads_config("agent:\n  max_iterations: lots\n")


def test_enabled_must_be_list_of_strings():
    with pytest.raises(ConfigError, match="enabled"):
        loads_config("tools:\n  enabled: read_file\n")


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_directory_raises(tmp_path):
    with pytest.raises(ConfigError, match="directory"):
        load_config(tmp_path)


def test_unknown_agent_key_raises():
    with pytest.raises(ConfigError, match="Unknown key.*agent"):
        loads_config("agent:\n  nam: typo\n")


def test_unknown_provider_key_raises():
    with pytest.raises(ConfigError, match="Unknown key.*provider"):
        loads_config("provider:\n  modell: gpt-4o\n")


def test_unknown_tools_key_raises():
    with pytest.raises(ConfigError, match="Unknown key.*tools"):
        loads_config("tools:\n  workpace: .\n")


def test_deepseek_selected_when_key_present(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-key")

    config = loads_config("agent:\n  name: test\n")

    assert config.provider.type == "deepseek"
    assert config.provider.base_url == "https://api.deepseek.com/v1"
    assert config.provider.model == "deepseek-chat"
    assert config.provider.api_key_env == "DEEPSEEK_API_KEY"
    assert config.provider.resolve_api_key() == "ds-key"
    assert config.provider.type_explicit is False


def test_explicit_type_wins_over_detection(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-key")

    config = loads_config("provider:\n  type: openai\n")

    assert config.provider.type == "openai"
    assert config.provider.model == "gpt-4o-mini"
    assert config.provider.type_explicit is True


def test_deepseek_profile_used_for_explicit_type(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-key")

    config = loads_config("provider:\n  type: deepseek\n")

    assert config.provider.base_url == "https://api.deepseek.com/v1"
    assert config.provider.model == "deepseek-chat"


def test_explicit_values_override_detected_profile(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-key")

    config = loads_config("provider:\n  model: deepseek-reasoner\n")

    assert config.provider.type == "deepseek"
    assert config.provider.model == "deepseek-reasoner"
    assert config.provider.base_url == "https://api.deepseek.com/v1"

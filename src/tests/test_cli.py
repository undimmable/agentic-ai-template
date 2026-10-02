"""Tests for the command line interface."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_ai import cli


@pytest.fixture(autouse=True)
def _clear_provider_keys(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)


def write_config(tmp_path: Path, text: str) -> str:
    path = tmp_path / "agent.yaml"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_no_command_prints_help(capsys):
    assert cli.main([]) == 0
    assert "usage: agentic" in capsys.readouterr().out


def test_validate_reports_auto_detected_deepseek(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-key")
    config = write_config(tmp_path, "agent:\n  name: t\n")

    assert cli.main(["-c", config, "validate"]) == 0

    out = capsys.readouterr().out
    assert "deepseek / deepseek-chat (auto-detected from DEEPSEEK_API_KEY)" in out
    assert "Config is valid." in out


def test_validate_warns_when_pinned_to_openai(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-key")
    config = write_config(tmp_path, "provider:\n  type: openai\n")

    assert cli.main(["-c", config, "validate"]) == 0

    out = capsys.readouterr().out
    assert "openai / gpt-4o-mini (pinned in config)" in out
    assert "DEEPSEEK_API_KEY is set" in out


def test_validate_defaults_to_openai_without_key(tmp_path, capsys):
    config = write_config(tmp_path, "agent:\n  name: t\n")

    assert cli.main(["-c", config, "validate"]) == 0
    assert "openai / gpt-4o-mini (default)" in capsys.readouterr().out


def test_validate_missing_config_returns_2(tmp_path, capsys):
    missing = str(tmp_path / "nope.yaml")
    assert cli.main(["-c", missing, "validate"]) == 2
    assert "configuration error" in capsys.readouterr().err


def test_providers_lists_deepseek(capsys):
    assert cli.main(["providers"]) == 0
    assert "deepseek" in capsys.readouterr().out


def test_init_writes_config_and_refuses_overwrite(tmp_path, capsys):
    target = tmp_path / "agent.yaml"

    assert cli.main(["init", str(target)]) == 0
    assert target.exists()
    assert "Wrote example config" in capsys.readouterr().out

    assert cli.main(["init", str(target)]) == 1
    assert "Refusing to overwrite" in capsys.readouterr().out

    assert cli.main(["init", str(target), "-f"]) == 0

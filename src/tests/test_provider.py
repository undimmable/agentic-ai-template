"""Tests for the OpenAI-compatible provider (no real network calls)."""

from __future__ import annotations

import json

import pytest

from agentic_ai.config import ProviderConfig
from agentic_ai.errors import ProviderError
from agentic_ai.providers import PROVIDER_REGISTRY, build_provider
from agentic_ai.providers.openai_compat import OpenAICompatibleProvider


class FakeResponse:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def make_provider(**kwargs):
    defaults = dict(base_url="http://localhost:9999/v1", model="test-model")
    defaults.update(kwargs)
    return OpenAICompatibleProvider(**defaults)


def test_endpoint_normalisation():
    assert make_provider()._endpoint() == "http://localhost:9999/v1/chat/completions"
    explicit = make_provider(base_url="http://x/v1/chat/completions")
    assert explicit._endpoint() == "http://x/v1/chat/completions"


def test_payload_includes_tools_and_temperature():
    provider = make_provider(temperature=0.1)
    payload = provider._build_payload(
        [{"role": "user", "content": "hi"}], [{"type": "function"}]
    )

    assert payload["model"] == "test-model"
    assert payload["temperature"] == 0.1
    assert payload["tools"] == [{"type": "function"}]
    assert payload["tool_choice"] == "auto"


def test_headers_include_bearer_token_when_key_present():
    provider = make_provider(api_key="secret")
    assert provider._headers()["Authorization"] == "Bearer secret"
    assert "Authorization" not in make_provider()._headers()


def test_chat_parses_tool_calls(monkeypatch):
    provider = make_provider()
    payload = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path": "a.txt"}',
                            },
                        }
                    ],
                }
            }
        ]
    }
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda request, timeout=None: FakeResponse(payload)
    )

    response = provider.chat([{"role": "user", "content": "read a.txt"}], tools=None)

    assert response.content is None
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "read_file"
    assert response.tool_calls[0].arguments == {"path": "a.txt"}


def test_chat_parses_plain_content(monkeypatch):
    provider = make_provider()
    payload = {"choices": [{"message": {"content": "hello"}}]}
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda request, timeout=None: FakeResponse(payload)
    )

    response = provider.chat([{"role": "user", "content": "hi"}])
    assert response.content == "hello"
    assert response.tool_calls == []


def test_invalid_tool_arguments_raise(monkeypatch):
    provider = make_provider()
    payload = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {"id": "1", "function": {"name": "x", "arguments": "{not json"}}
                    ]
                }
            }
        ]
    }
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda request, timeout=None: FakeResponse(payload)
    )

    with pytest.raises(ProviderError, match="invalid JSON arguments"):
        provider.chat([{"role": "user", "content": "hi"}])


def test_build_provider_from_config():
    config = ProviderConfig(api_key="k", model="m", base_url="http://x/v1")
    provider = build_provider(config)

    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.model == "m"
    assert provider.api_key == "k"


def test_registry_lists_openai_and_stub():
    assert "openai" in PROVIDER_REGISTRY
    assert "stub" in PROVIDER_REGISTRY

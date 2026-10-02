"""A scripted provider used for tests and offline runs.

It replays a list of responses (or callables producing responses) and records
every call it receives, which makes the agent loop fully deterministic to
assert against.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

from ..messages import LLMResponse
from .base import LLMProvider

Scripted = Union[LLMResponse, str, Callable[[List[Dict[str, Any]], Any], LLMResponse]]


class StubProvider(LLMProvider):
    def __init__(self, responses: Optional[List[Scripted]] = None) -> None:
        self._responses: List[Scripted] = list(responses or [])
        self.calls: List[Dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "stub"

    def queue(self, response: Scripted) -> None:
        self._responses.append(response)

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})
        if not self._responses:
            return LLMResponse(content="")
        response = self._responses.pop(0)
        if callable(response):
            return response(messages, tools)
        if isinstance(response, LLMResponse):
            return response
        return LLMResponse(content=str(response))

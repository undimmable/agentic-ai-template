"""Provider-neutral message and response types.

Messages are passed around in the OpenAI chat-completions shape because the
MVP targets OpenAI-compatible endpoints. Keeping the shape explicit here makes
it obvious what a provider has to consume and produce.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolCall:
    """A single request from the model to run a tool."""

    id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)

    def to_openai(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments or {}, ensure_ascii=False),
            },
        }


@dataclass
class LLMResponse:
    """The provider-neutral result of one model turn."""

    content: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    raw: Optional[Dict[str, Any]] = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)

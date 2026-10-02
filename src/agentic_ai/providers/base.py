"""Provider interface.

A provider is anything that can turn a conversation plus tool schemas into the
next assistant turn. The runtime only depends on this interface, so new
backends can be added without touching the agent loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..messages import LLMResponse


class LLMProvider(ABC):
    """Base class for every model backend."""

    @property
    def name(self) -> str:  # pragma: no cover - trivial default
        return type(self).__name__

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Return the next assistant turn for ``messages``."""

    def close(self) -> None:
        """Release provider resources. Default is a no-op."""

"""Tool interface and registry.

Tools are the agent's hands. Each tool exposes a JSON schema so the model can
call it, and a ``run`` method that executes it. New tools only need to
implement :class:`Tool` and be registered under a name.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

from ..registry import Registry

ToolFactory = Callable[[Any], "Tool"]

TOOL_REGISTRY: Registry[ToolFactory] = Registry("tool")


class Tool(ABC):
    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = {"type": "object", "properties": {}}

    @abstractmethod
    def run(self, **kwargs: Any) -> str:
        """Execute the tool and return a string result for the model."""

    def schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class FunctionTool(Tool):
    """Wrap a plain callable as a tool with an explicit schema."""

    def __init__(
        self,
        fn: Callable[..., Any],
        name: str,
        description: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.fn = fn
        self.name = name
        self.description = description
        self.parameters = parameters or {"type": "object", "properties": {}}

    def run(self, **kwargs: Any) -> str:
        return str(self.fn(**kwargs))


def tool(
    name: str,
    description: str,
    parameters: Optional[Dict[str, Any]] = None,
) -> Callable[[Callable[..., Any]], FunctionTool]:
    """Decorator that turns a function into a :class:`FunctionTool`."""

    def decorate(fn: Callable[..., Any]) -> FunctionTool:
        return FunctionTool(
            fn, name=name, description=description, parameters=parameters
        )

    return decorate

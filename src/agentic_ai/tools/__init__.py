"""Tool registry wiring and per-config tool construction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

from . import builtin
from .base import TOOL_REGISTRY, FunctionTool, Tool, tool

if TYPE_CHECKING:  # pragma: no cover
    from ..config import ToolConfig

builtin.register(TOOL_REGISTRY)


def build_tools(config: "ToolConfig") -> Dict[str, Tool]:
    """Instantiate the tools enabled in ``config``.

    Unknown names raise a :class:`~agentic_ai.errors.RegistryError` with the
    list of available tools, so config mistakes are obvious.
    """

    tools: Dict[str, Tool] = {}
    for name in config.enabled:
        factory = TOOL_REGISTRY.get(name)
        tools[name] = factory(config)
    return tools


__all__ = ["TOOL_REGISTRY", "FunctionTool", "Tool", "build_tools", "tool"]

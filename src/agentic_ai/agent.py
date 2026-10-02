"""The agent loop.

The loop is deliberately small: ask the provider for the next turn; if it
requests tools, run them, append the results and ask again; otherwise return
the final answer. Everything else is configuration or a plugin.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import uuid4

from .config import AgentConfig
from .errors import AgenticError, MaxIterationsError
from .logging_utils import get_logger
from .messages import ToolCall
from .providers import LLMProvider, build_provider
from .tools import Tool, build_tools


def _short(value: Any, limit: int = 120) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


class Agent:
    """A config-driven, tool-using agent."""

    def __init__(
        self,
        config: AgentConfig,
        provider: Optional[LLMProvider] = None,
        tools: Optional[Dict[str, Tool]] = None,
    ) -> None:
        self.config = config
        self.logger = get_logger("agent")
        self.provider = (
            provider if provider is not None else build_provider(config.provider)
        )
        self.tools = tools if tools is not None else build_tools(config.tools)
        self.messages: List[Dict[str, Any]] = []
        self.reset()

    # -- history ----------------------------------------------------------- #
    def reset(self) -> None:
        """Clear the conversation, keeping the system prompt."""

        self.messages = [{"role": "system", "content": self.config.system_prompt}]

    @property
    def tool_schemas(self) -> Optional[List[Dict[str, Any]]]:
        schemas = [tool.schema() for tool in self.tools.values()]
        return schemas or None

    # -- main entry point -------------------------------------------------- #
    def run(self, prompt: str) -> str:
        """Run one user turn to completion and return the final answer."""

        run_id = uuid4().hex[:8]
        self.logger.info(
            "[%s] ENTER run (agent=%s provider=%s tools=%s)",
            run_id,
            self.config.name,
            self.provider.name,
            ", ".join(sorted(self.tools)) or "none",
        )
        self.messages.append({"role": "user", "content": prompt})

        for step in range(1, self.config.max_iterations + 1):
            self.logger.debug("[%s] step %d: requesting completion", run_id, step)
            response = self.provider.chat(self.messages, self.tool_schemas)

            if response.wants_tools:
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": response.content,
                        "tool_calls": [
                            call.to_openai() for call in response.tool_calls
                        ],
                    }
                )
                for call in response.tool_calls:
                    self.logger.info(
                        "[%s] step %d: tool %s(%s)",
                        run_id,
                        step,
                        call.name,
                        _short(call.arguments),
                    )
                    result = self._execute_tool(call)
                    self.messages.append(
                        {"role": "tool", "tool_call_id": call.id, "content": result}
                    )
                continue

            content = response.content or ""
            self.messages.append({"role": "assistant", "content": content})
            self.logger.info("[%s] EXIT run after %d step(s)", run_id, step)
            return content

        raise MaxIterationsError(
            f"Agent '{self.config.name}' exceeded max_iterations="
            f"{self.config.max_iterations} without producing a final answer"
        )

    # -- internals --------------------------------------------------------- #
    def _execute_tool(self, call: ToolCall) -> str:
        tool = self.tools.get(call.name)
        if tool is None:
            available = ", ".join(sorted(self.tools)) or "none"
            self.logger.warning("unknown tool requested: %s", call.name)
            return f"Error: unknown tool '{call.name}'. Available tools: {available}."
        try:
            return str(tool.run(**call.arguments))
        except AgenticError as exc:
            self.logger.warning("tool %s failed: %s", call.name, exc)
            return f"Error executing '{call.name}': {exc}"
        except Exception as exc:  # noqa: BLE001 - surface everything to the model
            self.logger.exception("unexpected error in tool %s", call.name)
            return f"Error executing '{call.name}': {exc}"

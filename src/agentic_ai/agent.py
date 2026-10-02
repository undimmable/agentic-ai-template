"""The agent loop.

The loop is deliberately small: ask the provider for the next turn; if it
requests tools, run them, append the results and ask again; otherwise return
the final answer. Everything else is configuration or a plugin.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from .config import AgentConfig
from .errors import AgenticError, MaxIterationsError
from .logging_utils import get_logger
from .messages import ToolCall
from .providers import LLMProvider, build_provider
from .tools import Tool, build_tools

#: Stop the tool loop after the same (tool, arguments) pair repeats this many
#: times - usually a sign the model is stuck retrying a failing call.
_MAX_REPEATED_TOOL_CALLS = 3

_FINALIZE_PROMPT = (
    "Tool-use budget reached. Do not call any more tools. "
    "Summarise what you have found so far and give your best final answer now."
)

#: Event kinds emitted through the optional ``on_event`` callback.
EVENT_TOOL_CALL = "tool_call"
EVENT_TOOL_RESULT = "tool_result"
EVENT_FINAL = "final"


@dataclass
class AgentEvent:
    """A single observable step of the agent loop.

    The loop is otherwise opaque to callers that only want the final string;
    this lets an interface (the TUI, a logger, a test) watch the reasoning
    trail as it happens without reaching into ``agent.messages``.
    """

    kind: str
    name: str = ""
    arguments: Optional[Dict[str, Any]] = None
    result: str = ""
    text: str = ""


#: Signature of the observer passed to :meth:`Agent.run`.
EventCallback = Callable[[AgentEvent], None]


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
    def run(self, prompt: str, on_event: Optional[EventCallback] = None) -> str:
        """Run one user turn to completion and return the final answer.

        The loop stops as soon as the model answers without requesting tools.
        If the tool budget runs out (or the model keeps repeating the same
        call), the agent makes one final, tool-free request so the user always
        gets an answer instead of a bare error.

        ``on_event``, when given, is called with an :class:`AgentEvent` for each
        tool call, tool result and the final answer, so a front-end can render
        the reasoning trail live.
        """

        def emit(event: AgentEvent) -> None:
            if on_event is not None:
                try:
                    on_event(event)
                except Exception:  # noqa: BLE001 - observers must not break the loop
                    self.logger.exception("event observer raised; ignoring")

        run_id = uuid4().hex[:8]
        self.logger.info(
            "[%s] ENTER run (agent=%s provider=%s tools=%s)",
            run_id,
            self.config.name,
            self.provider.name,
            ", ".join(sorted(self.tools)) or "none",
        )
        self.messages.append({"role": "user", "content": prompt})

        tool_counts: Dict[str, int] = {}
        signatures: Dict[str, int] = {}

        for step in range(1, self.config.max_iterations + 1):
            self.logger.debug("[%s] step %d: requesting completion", run_id, step)
            response = self.provider.chat(self.messages, self.tool_schemas)

            if not response.wants_tools:
                content = response.content or ""
                self.messages.append({"role": "assistant", "content": content})
                self.logger.info("[%s] EXIT run after %d step(s)", run_id, step)
                emit(AgentEvent(kind=EVENT_FINAL, text=content))
                return content

            self.messages.append(
                {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": [call.to_openai() for call in response.tool_calls],
                }
            )
            repeated = False
            for call in response.tool_calls:
                tool_counts[call.name] = tool_counts.get(call.name, 0) + 1
                self.logger.info(
                    "[%s] step %d: tool %s(%s)",
                    run_id,
                    step,
                    call.name,
                    _short(call.arguments),
                )
                emit(
                    AgentEvent(
                        kind=EVENT_TOOL_CALL,
                        name=call.name,
                        arguments=dict(call.arguments or {}),
                    )
                )
                result = self._execute_tool(call)
                emit(
                    AgentEvent(
                        kind=EVENT_TOOL_RESULT, name=call.name, result=result
                    )
                )
                self.messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": result}
                )
                signature = (
                    f"{call.name}:"
                    f"{json.dumps(call.arguments, sort_keys=True, default=str)}"
                )
                signatures[signature] = signatures.get(signature, 0) + 1
                if signatures[signature] >= _MAX_REPEATED_TOOL_CALLS:
                    repeated = True

            if repeated:
                self.logger.warning(
                    "[%s] same tool call repeated %d times; stopping tool use",
                    run_id,
                    _MAX_REPEATED_TOOL_CALLS,
                )
                break

        return self._finalize(run_id, tool_counts, emit)

    def _finalize(
        self, run_id: str, tool_counts: Dict[str, int], emit: EventCallback
    ) -> str:
        """Ask for one last tool-free answer after the tool budget is spent."""

        self.logger.warning(
            "[%s] tool budget exhausted; requesting a final answer", run_id
        )
        self.messages.append({"role": "user", "content": _FINALIZE_PROMPT})
        response = self.provider.chat(self.messages, None)
        content = (response.content or "").strip()
        if content:
            self.messages.append({"role": "assistant", "content": content})
            self.logger.info("[%s] EXIT run via finalization", run_id)
            emit(AgentEvent(kind=EVENT_FINAL, text=content))
            return content

        trace = (
            ", ".join(f"{name} x{count}" for name, count in sorted(tool_counts.items()))
            or "none"
        )
        raise MaxIterationsError(
            f"Agent '{self.config.name}' exceeded max_iterations="
            f"{self.config.max_iterations} without producing a final answer "
            f"(tool calls: {trace})"
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

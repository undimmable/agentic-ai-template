"""Tests for the agent loop."""

from __future__ import annotations

import pytest

from agentic_ai.agent import Agent
from agentic_ai.config import AgentConfig
from agentic_ai.errors import MaxIterationsError, ToolError
from agentic_ai.messages import LLMResponse, ToolCall
from agentic_ai.providers.stub import StubProvider
from agentic_ai.tools.base import FunctionTool


def echo_tool() -> FunctionTool:
    return FunctionTool(
        lambda text: f"echo:{text}",
        name="echo",
        description="Echo the input.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )


def test_agent_executes_tool_then_returns_final_answer():
    provider = StubProvider(
        [
            LLMResponse(
                tool_calls=[ToolCall(id="1", name="echo", arguments={"text": "hi"})]
            ),
            LLMResponse(content="done"),
        ]
    )
    agent = Agent(AgentConfig(), provider=provider, tools={"echo": echo_tool()})

    assert agent.run("go") == "done"
    assert [m["role"] for m in agent.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert agent.messages[3]["content"] == "echo:hi"
    assert agent.messages[3]["tool_call_id"] == "1"
    # the model was given the tool schema on the first call
    assert provider.calls[0]["tools"][0]["function"]["name"] == "echo"


def test_agent_answers_without_tools():
    provider = StubProvider([LLMResponse(content="plain")])
    agent = Agent(AgentConfig(), provider=provider, tools={})

    assert agent.run("go") == "plain"
    assert agent.tool_schemas is None
    assert provider.calls[0]["tools"] is None


def test_agent_raises_when_iterations_exhausted():
    responses = [
        LLMResponse(
            tool_calls=[ToolCall(id=str(i), name="echo", arguments={"text": "x"})]
        )
        for i in range(5)
    ]
    provider = StubProvider(responses)
    agent = Agent(
        AgentConfig(max_iterations=2), provider=provider, tools={"echo": echo_tool()}
    )

    with pytest.raises(MaxIterationsError, match="max_iterations=2"):
        agent.run("go")


def test_unknown_tool_is_reported_to_the_model():
    provider = StubProvider(
        [
            LLMResponse(tool_calls=[ToolCall(id="1", name="missing", arguments={})]),
            LLMResponse(content="recovered"),
        ]
    )
    agent = Agent(AgentConfig(), provider=provider, tools={})

    assert agent.run("go") == "recovered"
    assert "unknown tool 'missing'" in agent.messages[3]["content"]


def test_tool_errors_are_returned_not_raised():
    def boom() -> str:
        raise ToolError("cannot do that")

    boom_tool = FunctionTool(
        boom, "boom", "Always fails.", {"type": "object", "properties": {}}
    )
    provider = StubProvider(
        [
            LLMResponse(tool_calls=[ToolCall(id="1", name="boom", arguments={})]),
            LLMResponse(content="ok"),
        ]
    )
    agent = Agent(AgentConfig(), provider=provider, tools={"boom": boom_tool})

    assert agent.run("go") == "ok"
    assert agent.messages[3]["content"] == "Error executing 'boom': cannot do that"


def test_reset_keeps_only_system_prompt():
    provider = StubProvider([LLMResponse(content="one"), LLMResponse(content="two")])
    agent = Agent(AgentConfig(), provider=provider, tools={})

    agent.run("first")
    assert len(agent.messages) == 3
    agent.reset()

    assert len(agent.messages) == 1
    assert agent.messages[0]["role"] == "system"


def test_multiple_tool_calls_in_one_turn():
    provider = StubProvider(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(id="1", name="echo", arguments={"text": "a"}),
                    ToolCall(id="2", name="echo", arguments={"text": "b"}),
                ]
            ),
            LLMResponse(content="done"),
        ]
    )
    agent = Agent(AgentConfig(), provider=provider, tools={"echo": echo_tool()})

    agent.run("go")

    tool_messages = [m for m in agent.messages if m["role"] == "tool"]
    assert [m["content"] for m in tool_messages] == ["echo:a", "echo:b"]

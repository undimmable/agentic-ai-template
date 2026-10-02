"""Tests for the built-in tools and their workspace confinement."""

from __future__ import annotations

import pytest

from agentic_ai.config import ToolConfig
from agentic_ai.errors import RegistryError, ToolError
from agentic_ai.tools import build_tools
from agentic_ai.tools.base import FunctionTool, tool


def test_default_tools_are_built_and_have_schemas():
    tools = build_tools(ToolConfig())

    assert set(tools) == {"read_file", "write_file", "list_dir"}
    for current in tools.values():
        schema = current.schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == current.name
        assert schema["function"]["parameters"]["type"] == "object"


def test_unknown_enabled_tool_raises():
    with pytest.raises(RegistryError, match="Unknown tool 'nope'"):
        build_tools(ToolConfig(enabled=["nope"]))


def test_write_then_read(tmp_path):
    tools = build_tools(ToolConfig(workspace=str(tmp_path)))

    message = tools["write_file"].run(path="nested/hello.txt", content="hi there")
    assert "hello.txt" in message
    assert tools["read_file"].run(path="nested/hello.txt") == "hi there"


def test_read_missing_file_raises(tmp_path):
    tools = build_tools(ToolConfig(workspace=str(tmp_path)))
    with pytest.raises(ToolError, match="not found"):
        tools["read_file"].run(path="missing.txt")


def test_read_directory_raises(tmp_path):
    tools = build_tools(ToolConfig(workspace=str(tmp_path)))
    with pytest.raises(ToolError, match="directory"):
        tools["read_file"].run(path=".")


def test_path_escape_is_blocked(tmp_path):
    workspace = tmp_path / "work"
    workspace.mkdir()
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")

    tools = build_tools(ToolConfig(workspace=str(workspace)))
    with pytest.raises(ToolError, match="escapes"):
        tools["read_file"].run(path="../secret.txt")


def test_list_dir(tmp_path):
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "a").mkdir()

    tools = build_tools(ToolConfig(workspace=str(tmp_path)))
    output = tools["list_dir"].run(path=".")

    lines = output.splitlines()
    assert "a/" in lines
    assert "b.txt" in lines


def test_list_empty_dir(tmp_path):
    tools = build_tools(ToolConfig(workspace=str(tmp_path)))
    assert tools["list_dir"].run() == "(empty directory)"


def test_shell_runs_in_workspace(tmp_path):
    tools = build_tools(ToolConfig(workspace=str(tmp_path), enabled=["shell"]))
    output = tools["shell"].run(command="echo hello")
    assert "exit_code=0" in output
    assert "hello" in output


def test_shell_timeout_reports_error(tmp_path):
    tools = build_tools(ToolConfig(workspace=str(tmp_path), enabled=["shell"]))
    with pytest.raises(ToolError, match="timed out"):
        tools["shell"].run(command="sleep 2", timeout=0.1)


def test_function_tool_decorator():
    @tool(
        name="shout",
        description="Uppercase the input.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )
    def shout(text: str) -> str:
        return text.upper()

    assert isinstance(shout, FunctionTool)
    assert shout.run(text="hi") == "HI"

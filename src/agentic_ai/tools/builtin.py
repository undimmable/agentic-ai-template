"""Built-in tools: file access plus an optional shell."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Optional

from ..errors import ToolError
from .base import TOOL_REGISTRY, Tool

_MAX_OUTPUT = 200_000


class _WorkspaceTool(Tool):
    """Base class that optionally confines file access to a workspace root."""

    def __init__(self, workspace: Optional[str] = None) -> None:
        self.workspace = Path(workspace).expanduser().resolve() if workspace else None

    def _resolve(self, path: str) -> Path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            base = self.workspace or Path.cwd()
            candidate = base / candidate
        resolved = candidate.resolve()
        if self.workspace is not None:
            try:
                resolved.relative_to(self.workspace)
            except ValueError:
                raise ToolError(
                    f"Path '{path}' escapes the configured workspace '{self.workspace}'"
                ) from None
        return resolved


class ReadFileTool(_WorkspaceTool):
    name = "read_file"
    description = "Read a UTF-8 text file and return its contents."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to read."},
            "max_bytes": {
                "type": "integer",
                "description": "Maximum number of bytes to read.",
                "default": _MAX_OUTPUT,
            },
        },
        "required": ["path"],
    }

    def run(self, path: str, max_bytes: int = _MAX_OUTPUT) -> str:
        target = self._resolve(path)
        if not target.exists():
            raise ToolError(f"File not found: {path}")
        if target.is_dir():
            raise ToolError(f"'{path}' is a directory; use list_dir instead")
        data = target.read_bytes()[: int(max_bytes)]
        return data.decode("utf-8", errors="replace")


class WriteFileTool(_WorkspaceTool):
    name = "write_file"
    description = "Write UTF-8 text to a file, creating parent directories as needed."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to write."},
            "content": {"type": "string", "description": "Full text to write."},
        },
        "required": ["path", "content"],
    }

    def run(self, path: str, content: str) -> str:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Wrote {len(content.encode('utf-8'))} bytes to {path}"


class ListDirTool(_WorkspaceTool):
    name = "list_dir"
    description = "List the entries of a directory (directories end with '/')."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory to list. Defaults to the workspace root.",
                "default": ".",
            }
        },
        "required": [],
    }

    def run(self, path: str = ".") -> str:
        target = self._resolve(path)
        if not target.exists():
            raise ToolError(f"Directory not found: {path}")
        if not target.is_dir():
            raise ToolError(f"'{path}' is not a directory")
        entries = []
        for entry in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name)):
            entries.append(f"{entry.name}/" if entry.is_dir() else entry.name)
        return "\n".join(entries) if entries else "(empty directory)"


class ShellTool(_WorkspaceTool):
    name = "shell"
    description = (
        "Run a shell command in the workspace and return exit code, stdout and stderr."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to execute."},
            "timeout": {
                "type": "number",
                "description": "Timeout in seconds.",
            },
        },
        "required": ["command"],
    }

    def __init__(self, workspace: Optional[str] = None, timeout: float = 30.0) -> None:
        super().__init__(workspace)
        self.timeout = timeout

    def run(self, command: str, timeout: Optional[float] = None) -> str:
        effective_timeout = float(timeout) if timeout is not None else self.timeout
        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=str(self.workspace) if self.workspace else None,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
            )
        except subprocess.TimeoutExpired:
            raise ToolError(
                f"Command timed out after {effective_timeout}s: {command}"
            ) from None
        stdout = (completed.stdout or "")[:_MAX_OUTPUT]
        stderr = (completed.stderr or "")[:_MAX_OUTPUT]
        return f"exit_code={completed.returncode}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"


def register(registry: Any = TOOL_REGISTRY) -> None:
    """Register every built-in tool factory into ``registry``."""

    registry.register("read_file", lambda cfg: ReadFileTool(cfg.workspace))
    registry.register("write_file", lambda cfg: WriteFileTool(cfg.workspace))
    registry.register("list_dir", lambda cfg: ListDirTool(cfg.workspace))
    registry.register(
        "shell", lambda cfg: ShellTool(cfg.workspace, timeout=cfg.shell_timeout)
    )

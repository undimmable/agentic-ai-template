"""Command line interface: ``agentic init|run|repl|validate|tools|providers``."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

from .agent import Agent
from .config import AgentConfig, dump_example_config, load_config
from .errors import AgenticError, ConfigError
from .logging_utils import configure_logging, get_logger
from .providers import PROVIDER_REGISTRY
from .tools import TOOL_REGISTRY

_REPL_HELP = """\
commands:
  /help    show this help
  /tools   list the tools enabled for this agent
  /reset   clear the conversation history
  /exit    quit
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentic",
        description="A config-driven, extensible agent runtime.",
    )
    parser.add_argument(
        "-c",
        "--config",
        default="agent.yaml",
        help="Path to the declarative config file (default: agent.yaml).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging."
    )

    sub = parser.add_subparsers(dest="command")

    init = sub.add_parser("init", help="Write an example config file.")
    init.add_argument(
        "path", nargs="?", default="agent.yaml", help="Where to write it."
    )
    init.add_argument(
        "-f", "--force", action="store_true", help="Overwrite if it exists."
    )

    run = sub.add_parser("run", help="Run a single prompt and exit.")
    run.add_argument("prompt", nargs="+", help="The prompt to send.")

    sub.add_parser("repl", help="Start an interactive session.")
    sub.add_parser("tui", help="Start the full-screen terminal interface.")
    sub.add_parser("validate", help="Validate the config and print a summary.")
    sub.add_parser("tools", help="List registered and enabled tools.")
    sub.add_parser("providers", help="List registered provider types.")
    return parser


def _load(path: str) -> AgentConfig:
    return load_config(path)


def _cmd_init(args: argparse.Namespace) -> int:
    target = Path(args.path)
    if target.exists() and not args.force:
        print(f"Refusing to overwrite existing file: {target} (use -f/--force)")
        return 1
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(dump_example_config(), encoding="utf-8")
    print(f"Wrote example config to {target}")
    print(f'Next: edit it, then run `agentic -c {target} run "hello"`')
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    config = _load(args.config)
    _print_summary(config)
    print("Config is valid.")
    return 0


def _cmd_tools(args: argparse.Namespace) -> int:
    config = _load(args.config)
    print(f"Registered tools: {', '.join(TOOL_REGISTRY.names())}")
    print(f"Enabled tools:    {', '.join(config.tools.enabled) or '(none)'}")
    print(f"Workspace:        {config.tools.workspace or '(current directory)'}")
    return 0


def _cmd_providers(_args: argparse.Namespace) -> int:
    print(f"Registered providers: {', '.join(PROVIDER_REGISTRY.names())}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    config = _load(args.config)
    agent = Agent(config)
    prompt = " ".join(args.prompt)
    try:
        answer = agent.run(prompt)
    finally:
        agent.provider.close()
    print(answer)
    return 0


def _cmd_tui(args: argparse.Namespace) -> int:
    config = _load(args.config)
    from .tui import run_tui

    return run_tui(config)


def _cmd_repl(args: argparse.Namespace) -> int:
    config = _load(args.config)
    agent = Agent(config)
    print(
        f"agentic repl - agent '{config.name}' via {agent.provider.name}. "
        f"Type /help for commands, /exit to quit."
    )
    try:
        while True:
            try:
                line = input("you> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            line = line.strip()
            if not line:
                continue
            if line in {"/exit", "/quit"}:
                break
            if line == "/help":
                print(_REPL_HELP, end="")
                continue
            if line == "/reset":
                agent.reset()
                print("(history cleared)")
                continue
            if line == "/tools":
                print(", ".join(sorted(agent.tools)) or "(no tools enabled)")
                continue
            try:
                answer = agent.run(line)
            except AgenticError as exc:
                print(f"error: {exc}", file=sys.stderr)
                continue
            print(f"{config.name}> {answer}")
    finally:
        agent.provider.close()
    return 0


def _print_summary(config: AgentConfig) -> None:
    provider_origin = (
        "pinned in config"
        if config.provider.type_explicit
        else (
            "auto-detected from DEEPSEEK_API_KEY"
            if config.provider.type == "deepseek"
            else "default"
        )
    )
    print(f"Config:        {config.source or '(none)'}")
    print(f"Agent name:    {config.name}")
    print(f"Max steps:     {config.max_iterations}")
    print(
        f"Provider:      {config.provider.type} / {config.provider.model} "
        f"({provider_origin})"
    )
    print(f"Base URL:      {config.provider.base_url}")
    print(f"API key:       {'set' if config.provider.resolve_api_key() else 'not set'}")
    print(f"Enabled tools: {', '.join(config.tools.enabled) or '(none)'}")
    if config.provider.type == "openai" and os.environ.get("DEEPSEEK_API_KEY"):
        print(
            "Note: provider is pinned to 'openai' while DEEPSEEK_API_KEY is set. "
            "Remove `provider.type` (or set it to 'deepseek') to use DeepSeek."
        )


_COMMANDS = {
    "init": _cmd_init,
    "run": _cmd_run,
    "repl": _cmd_repl,
    "tui": _cmd_tui,
    "validate": _cmd_validate,
    "tools": _cmd_tools,
    "providers": _cmd_providers,
}


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0

    configure_logging(args.verbose)
    logger = get_logger("cli")
    try:
        return _COMMANDS[args.command](args)
    except ConfigError as exc:
        logger.error("configuration error: %s", exc)
        return 2
    except AgenticError as exc:
        logger.error("%s", exc)
        return 1
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        print()
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""A terminal user interface for the agent runtime.

``agentic tui`` opens a full-screen, curses-based chat interface around the
same :class:`~agentic_ai.agent.Agent` the CLI uses. The design follows the
project's principles:

* **Config-driven** - it renders whatever ``agent.yaml`` describes (name,
  provider, model, tools) and never hard-codes a backend.
* **Transparent** - every turn is visible: the human's prompt, the agent's
  answer, and each tool call with its result, so the reasoning trail is on
  screen rather than buried in logs.
* **Responsive** - the agent loop runs on a background thread and streams
  events back through a queue, so the interface keeps redrawing (and can be
  interrupted) while a slow model is thinking.

The module is split so that everything except the thin curses shell is plain,
importable Python and therefore testable without a terminal:

* :class:`TranscriptEntry` / :class:`Transcript` - the scrollback model.
* :func:`parse_command` - slash-command parsing.
* :class:`AgentWorker` - the background agent runner and its event queue.
* :class:`TuiApp` - the curses view/controller.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .agent import (
    EVENT_FINAL,
    EVENT_TOOL_CALL,
    EVENT_TOOL_RESULT,
    Agent,
    AgentEvent,
)
from .config import AgentConfig
from .errors import AgenticError
from .logging_utils import get_logger
from .sessions import SessionError, SessionStore

# --------------------------------------------------------------------------- #
# transcript model
# --------------------------------------------------------------------------- #
#: Style tags understood by the renderer. Kept as plain strings so the model
#: layer stays independent of curses.
STYLE_USER = "user"
STYLE_AGENT = "agent"
STYLE_TOOL = "tool"
STYLE_ERROR = "error"
STYLE_SYSTEM = "system"


@dataclass
class TranscriptEntry:
    """One rendered line-group in the scrollback."""

    role: str
    text: str
    style: str = STYLE_AGENT

    def header(self) -> str:
        return {
            STYLE_USER: "you",
            STYLE_AGENT: "agent",
            STYLE_TOOL: "tool",
            STYLE_ERROR: "error",
            STYLE_SYSTEM: "system",
        }.get(self.style, self.style)


@dataclass
class Transcript:
    """An ordered, bounded list of :class:`TranscriptEntry`."""

    entries: List[TranscriptEntry] = field(default_factory=list)
    limit: int = 500

    def add(self, role: str, text: str, style: str = STYLE_AGENT) -> TranscriptEntry:
        entry = TranscriptEntry(role=role, text=text, style=style)
        self.entries.append(entry)
        if len(self.entries) > self.limit:
            del self.entries[: len(self.entries) - self.limit]
        return entry

    def clear(self) -> None:
        self.entries.clear()

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)


# --------------------------------------------------------------------------- #
# slash commands
# --------------------------------------------------------------------------- #
HELP_TEXT = """\
commands:
  /help          show this help
  /tools         list the tools enabled for this agent
  /sessions      list stored sessions
  /load <id>     load a stored session and continue it
  /save [name]   save the current conversation as a session
  /reset         clear the conversation history
  /clear         clear the on-screen transcript
  /exit, /quit   leave the interface
anything else is sent to the agent as a prompt."""


@dataclass
class Command:
    """A parsed slash command."""

    name: str
    argument: str = ""

    @property
    def is_command(self) -> bool:
        return bool(self.name)


def parse_command(line: str) -> Command:
    """Parse a raw input line into a :class:`Command`.

    A line that does not start with ``/`` is treated as a prompt and returned
    with an empty ``name`` so the caller can forward it to the agent.
    """

    stripped = line.strip()
    if not stripped.startswith("/"):
        return Command(name="", argument=stripped)
    head, _, rest = stripped.partition(" ")
    return Command(name=head[1:].lower(), argument=rest.strip())


# --------------------------------------------------------------------------- #
# background agent worker
# --------------------------------------------------------------------------- #
#: Event kinds pushed onto the worker queue.
EVENT_ANSWER = "answer"
EVENT_TOOL = "tool"
EVENT_ERROR = "error"
EVENT_DONE = "done"


@dataclass
class WorkerEvent:
    kind: str
    text: str = ""
    name: str = ""


class AgentWorker:
    """Runs agent turns on a background thread, reporting via a queue.

    The UI thread calls :meth:`submit` and then drains :attr:`events` while it
    redraws. Only one turn runs at a time; :meth:`busy` reports whether the
    agent is currently thinking.
    """

    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self.events: "queue.Queue[WorkerEvent]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._busy = False

    def busy(self) -> bool:
        with self._lock:
            return self._busy

    def submit(self, prompt: str) -> bool:
        """Start a turn. Returns ``False`` if a turn is already running."""

        with self._lock:
            if self._busy:
                return False
            self._busy = True
        self._thread = threading.Thread(
            target=self._run, args=(prompt,), name="agentic-tui-worker", daemon=True
        )
        self._thread.start()
        return True

    def _run(self, prompt: str) -> None:
        try:
            answer = self.agent.run(prompt, on_event=self._on_agent_event)
            self.events.put(WorkerEvent(EVENT_ANSWER, answer or ""))
        except AgenticError as exc:
            self.events.put(WorkerEvent(EVENT_ERROR, str(exc)))
        except Exception as exc:  # noqa: BLE001 - never kill the UI thread
            self.events.put(WorkerEvent(EVENT_ERROR, f"unexpected error: {exc}"))
        finally:
            with self._lock:
                self._busy = False
            self.events.put(WorkerEvent(EVENT_DONE))

    def _on_agent_event(self, event: AgentEvent) -> None:
        """Translate an :class:`AgentEvent` into a UI :class:`WorkerEvent`.

        Runs on the worker thread; the queue is thread-safe so the UI thread
        simply drains it on its next redraw.
        """

        if event.kind == EVENT_TOOL_CALL:
            self.events.put(
                WorkerEvent(EVENT_TOOL, _format_tool_call(event), name=event.name)
            )
        elif event.kind == EVENT_TOOL_RESULT:
            self.events.put(
                WorkerEvent(EVENT_TOOL, _format_tool_result(event), name=event.name)
            )
        # EVENT_FINAL is delivered as EVENT_ANSWER by _run, so it is ignored here.

    def drain(self) -> List[WorkerEvent]:
        """Return all events currently queued, without blocking."""

        drained: List[WorkerEvent] = []
        while True:
            try:
                drained.append(self.events.get_nowait())
            except queue.Empty:
                break
        return drained


# --------------------------------------------------------------------------- #
# rendering helpers (pure, so they can be unit-tested)
# --------------------------------------------------------------------------- #
def _format_tool_call(event: AgentEvent) -> str:
    """Render a tool call as a single, readable line."""

    args = event.arguments or {}
    if not args:
        return f"{event.name}()"
    rendered = ", ".join(f"{key}={value!r}" for key, value in args.items())
    return f"{event.name}({rendered})"


def _format_tool_result(event: AgentEvent) -> str:
    """Render a tool result, collapsing it to a single line."""

    result = (event.result or "").strip()
    if not result:
        return f"{event.name} -> (no output)"
    first_line = result.splitlines()[0]
    if len(result.splitlines()) > 1 or len(first_line) > 200:
        first_line = first_line[:197] + "..."
    return f"{event.name} -> {first_line}"


def wrap_text(text: str, width: int) -> List[str]:
    """Wrap ``text`` to ``width`` columns, preserving explicit newlines.

    Uses a simple greedy word-wrap so it has no dependency on curses or
    ``textwrap``'s locale-sensitive behaviour. Words longer than ``width`` are
    hard-split so nothing is ever lost.
    """

    if width <= 0:
        return [text]
    lines: List[str] = []
    for paragraph in text.split("\n"):
        if paragraph == "":
            lines.append("")
            continue
        current = ""
        for word in paragraph.split(" "):
            while len(word) > width:
                if current:
                    lines.append(current)
                    current = ""
                lines.append(word[:width])
                word = word[width:]
            if not current:
                current = word
            elif len(current) + 1 + len(word) <= width:
                current = f"{current} {word}"
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines or [""]


#: Indent applied to wrapped content lines when headers are shown.
_INDENT = "  "


def render_transcript(
    transcript: Transcript, width: int, *, show_headers: bool = True
) -> List[Tuple[str, str]]:
    """Flatten the transcript into ``(style, line)`` pairs for the renderer.

    Every returned line is guaranteed to be at most ``width`` columns wide,
    *including* the indent added to content lines, so the renderer can draw
    them without the terminal wrapping them onto the next row (which would
    push the body into the input line).
    """

    indent = _INDENT if show_headers else ""
    # Reserve room for the indent so indent + content never exceeds ``width``.
    content_width = max(1, width - len(indent))
    rendered: List[Tuple[str, str]] = []
    for entry in transcript:
        if show_headers:
            rendered.append((entry.style, f"{entry.header()}>"))
        for line in wrap_text(entry.text, content_width):
            rendered.append((entry.style, f"{indent}{line}"))
    return rendered


# --------------------------------------------------------------------------- #
# curses application
# --------------------------------------------------------------------------- #
class TuiApp:
    """The curses view/controller around an :class:`AgentWorker`.

    When sessions are enabled in the config, the conversation is persisted to a
    SQLite database after every completed turn, and ``/sessions`` / ``/load``
    let the human browse and resume earlier conversations.
    """

    def __init__(
        self,
        config: AgentConfig,
        agent: Agent,
        store: Optional[SessionStore] = None,
    ) -> None:
        self.config = config
        self.agent = agent
        self.worker = AgentWorker(agent)
        self.transcript = Transcript()
        self.input = ""
        self.scroll = 0  # lines scrolled up from the bottom
        self.status = "ready"
        self.logger = get_logger("tui")
        self.store = store if store is not None else self._build_store(config)
        #: Id of the session currently being persisted, once one exists.
        self.session_id: Optional[int] = None

    @staticmethod
    def _build_store(config: AgentConfig) -> Optional[SessionStore]:
        """Create a :class:`SessionStore` from the config, or ``None`` if off."""

        if not config.sessions.enabled:
            return None
        return SessionStore(config.sessions.path)

    # -- session persistence ----------------------------------------------- #
    def _session_name(self) -> str:
        """A human-friendly default name for the current conversation."""

        for entry in self.transcript:
            if entry.style == STYLE_USER:
                first_line = entry.text.strip().splitlines()[0]
                return first_line[:60] or self.config.name
        return self.config.name

    def save_session(self, name: Optional[str] = None) -> Optional[int]:
        """Persist the current conversation, returning the session id.

        Returns ``None`` when sessions are disabled or there is nothing worth
        saving (only the system prompt). Reuses the current session id so a
        resumed conversation keeps updating the same row.
        """

        if self.store is None:
            return None
        if len(self.agent.messages) <= 1:
            return None
        try:
            self.session_id = self.store.save_session(
                name or self._session_name(),
                self.agent.messages,
                session_id=self.session_id,
            )
        except SessionError as exc:
            self.logger.warning("could not save session: %s", exc)
            self.transcript.add("system", f"could not save session: {exc}", STYLE_ERROR)
            return None
        return self.session_id

    def list_sessions(self) -> List[Any]:
        """Return stored sessions, or an empty list when sessions are disabled."""

        if self.store is None:
            return []
        try:
            return self.store.list_sessions()
        except SessionError as exc:
            self.logger.warning("could not list sessions: %s", exc)
            self.transcript.add("system", f"could not list sessions: {exc}", STYLE_ERROR)
            return []

    def load_session(self, session_id: int) -> bool:
        """Load a stored session into the agent and rebuild the transcript.

        Returns ``True`` on success. The on-screen transcript is cleared and
        repopulated from the stored messages so the human sees the resumed
        conversation.
        """

        if self.store is None:
            self.transcript.add(
                "system", "sessions are disabled in the config", STYLE_ERROR
            )
            return False
        try:
            messages = self.store.load_session(session_id)
        except SessionError as exc:
            self.transcript.add("system", f"could not load session: {exc}", STYLE_ERROR)
            return False
        if messages is None:
            self.transcript.add(
                "system", f"no session with id {session_id}", STYLE_ERROR
            )
            return False

        self.agent.load_messages(messages)
        self.session_id = session_id
        self.transcript.clear()
        self._render_messages(messages)
        self.transcript.add(
            "system", f"loaded session {session_id}", STYLE_SYSTEM
        )
        return True

    def _render_messages(self, messages: List[Dict[str, Any]]) -> None:
        """Rebuild the transcript from a stored message list."""

        for message in messages:
            role = message.get("role")
            content = message.get("content") or ""
            if role == "system":
                continue
            if role == "user":
                self.transcript.add("user", content, STYLE_USER)
            elif role == "assistant":
                if content:
                    self.transcript.add("assistant", content, STYLE_AGENT)
            elif role == "tool":
                self.transcript.add("tool", content, STYLE_TOOL)

    # -- state transitions (testable without curses) ----------------------- #
    def handle_command(self, command: Command) -> bool:
        """Apply a slash command. Returns ``False`` when the UI should exit."""

        name = command.name
        if name in {"exit", "quit"}:
            return False
        if name == "help":
            self.transcript.add("system", HELP_TEXT, STYLE_SYSTEM)
        elif name == "tools":
            tools = ", ".join(sorted(self.agent.tools)) or "(no tools enabled)"
            self.transcript.add("system", f"enabled tools: {tools}", STYLE_SYSTEM)
        elif name == "sessions":
            self._command_sessions()
        elif name == "load":
            self._command_load(command.argument)
        elif name == "save":
            self._command_save(command.argument)
        elif name == "reset":
            self.agent.reset()
            self.session_id = None
            self.transcript.add("system", "conversation history cleared", STYLE_SYSTEM)
        elif name == "clear":
            self.transcript.clear()
        else:
            self.transcript.add(
                "system", f"unknown command: /{name} (try /help)", STYLE_ERROR
            )
        return True

    def _command_sessions(self) -> None:
        if self.store is None:
            self.transcript.add(
                "system", "sessions are disabled in the config", STYLE_SYSTEM
            )
            return
        sessions = self.list_sessions()
        if not sessions:
            self.transcript.add("system", "no stored sessions", STYLE_SYSTEM)
            return
        lines = ["stored sessions:"]
        for info in sessions:
            marker = "*" if info.id == self.session_id else " "
            lines.append(
                f" {marker} [{info.id}] {info.name} "
                f"({info.message_count} messages, {info.updated_at_str})"
            )
        lines.append("use /load <id> to resume one")
        self.transcript.add("system", "\n".join(lines), STYLE_SYSTEM)

    def _command_load(self, argument: str) -> None:
        argument = argument.strip()
        if not argument:
            self.transcript.add(
                "system", "usage: /load <id> (see /sessions)", STYLE_ERROR
            )
            return
        try:
            session_id = int(argument)
        except ValueError:
            self.transcript.add(
                "system", f"'{argument}' is not a session id (see /sessions)",
                STYLE_ERROR,
            )
            return
        self.load_session(session_id)

    def _command_save(self, argument: str) -> None:
        if self.store is None:
            self.transcript.add(
                "system", "sessions are disabled in the config", STYLE_SYSTEM
            )
            return
        session_id = self.save_session(argument.strip() or None)
        if session_id is not None:
            self.transcript.add(
                "system", f"saved session {session_id}", STYLE_SYSTEM
            )

    def submit_prompt(self, prompt: str) -> None:
        """Record a prompt and hand it to the worker."""

        if not prompt:
            return
        if self.worker.busy():
            self.transcript.add(
                "system", "still working on the previous prompt...", STYLE_SYSTEM
            )
            return
        self.transcript.add("user", prompt, STYLE_USER)
        self.status = "thinking..."
        self.scroll = 0
        self.worker.submit(prompt)

    def pump_events(self) -> None:
        """Drain worker events into the transcript."""

        for event in self.worker.drain():
            if event.kind == EVENT_ANSWER:
                self.transcript.add("assistant", event.text, STYLE_AGENT)
            elif event.kind == EVENT_TOOL:
                self.transcript.add("tool", event.text, STYLE_TOOL)
            elif event.kind == EVENT_ERROR:
                self.transcript.add("system", event.text, STYLE_ERROR)
            elif event.kind == EVENT_DONE:
                self.status = "ready"
                # Persist the conversation once the turn has fully completed.
                self.save_session()

    # -- curses plumbing --------------------------------------------------- #
    def run(self, stdscr: Any) -> None:  # pragma: no cover - needs a terminal
        import curses

        curses.curs_set(1)
        stdscr.nodelay(True)
        stdscr.keypad(True)
        try:
            curses.start_color()
            curses.use_default_colors()
            self._init_colors(curses)
        except curses.error:
            pass

        self.transcript.add(
            "system",
            f"agent '{self.config.name}' via {self.agent.provider.name}. "
            f"Type /help for commands, /exit to quit.",
            STYLE_SYSTEM,
        )

        while True:
            self.pump_events()
            self._draw(curses, stdscr)
            try:
                key = stdscr.get_wch()
            except curses.error:
                key = None
            if key is None:
                continue
            if not self._handle_key(curses, key):
                break

    def _init_colors(self, curses: Any) -> None:  # pragma: no cover - terminal
        curses.init_pair(1, curses.COLOR_CYAN, -1)  # user
        curses.init_pair(2, curses.COLOR_GREEN, -1)  # agent
        curses.init_pair(3, curses.COLOR_YELLOW, -1)  # tool
        curses.init_pair(4, curses.COLOR_RED, -1)  # error
        curses.init_pair(5, curses.COLOR_BLUE, -1)  # system
        curses.init_pair(6, curses.COLOR_BLACK, curses.COLOR_CYAN)  # status bar

    def _color_for(self, curses: Any, style: str) -> int:  # pragma: no cover
        pair = {
            STYLE_USER: 1,
            STYLE_AGENT: 2,
            STYLE_TOOL: 3,
            STYLE_ERROR: 4,
            STYLE_SYSTEM: 5,
        }.get(style, 2)
        try:
            return curses.color_pair(pair)
        except curses.error:
            return 0

    def _handle_key(self, curses: Any, key: Any) -> bool:  # pragma: no cover
        if key in ("\n", "\r", curses.KEY_ENTER):
            line = self.input.strip()
            self.input = ""
            if not line:
                return True
            command = parse_command(line)
            if command.is_command:
                return self.handle_command(command)
            self.submit_prompt(line)
            return True
        if key in ("\x7f", "\b", curses.KEY_BACKSPACE):
            self.input = self.input[:-1]
            return True
        if key == "\x1b":  # ESC
            return False
        if key == curses.KEY_UP:
            self.scroll += 1
            return True
        if key == curses.KEY_DOWN:
            self.scroll = max(0, self.scroll - 1)
            return True
        if key == curses.KEY_PPAGE:
            self.scroll += 10
            return True
        if key == curses.KEY_NPAGE:
            self.scroll = max(0, self.scroll - 10)
            return True
        if key == curses.KEY_RESIZE:
            return True
        if isinstance(key, str) and key.isprintable():
            self.input += key
        return True

    def _draw(self, curses: Any, stdscr: Any) -> None:  # pragma: no cover
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        if height < 4 or width < 20:
            stdscr.addnstr(0, 0, "terminal too small", max(0, width - 1))
            stdscr.refresh()
            return

        body_height = height - 3
        lines = render_transcript(self.transcript, width - 2)
        # clamp scroll to available content
        max_scroll = max(0, len(lines) - body_height)
        self.scroll = min(self.scroll, max_scroll)
        end = len(lines) - self.scroll
        start = max(0, end - body_height)
        visible = lines[start:end]

        for row, (style, text) in enumerate(visible):
            attr = self._color_for(curses, style)
            stdscr.addnstr(row, 0, text, width - 1, attr)

        # separator + input line
        sep_row = height - 3
        stdscr.addnstr(sep_row, 0, "-" * (width - 1), width - 1)
        prompt = "you> "
        stdscr.addnstr(height - 2, 0, prompt + self.input, width - 1)
        cursor_col = min(len(prompt) + len(self.input), width - 2)
        stdscr.move(height - 2, cursor_col)

        # status bar
        status = (
            f" {self.config.name} | {self.agent.provider.name} | "
            f"tools: {len(self.agent.tools)} | {self.status} "
        )
        try:
            stdscr.addnstr(
                height - 1, 0, status.ljust(width - 1), width - 1,
                self._color_for(curses, STYLE_USER),
            )
        except curses.error:
            pass
        stdscr.refresh()


def run_tui(config: AgentConfig, agent: Optional[Agent] = None) -> int:
    """Launch the curses interface. Returns a process exit code."""

    import curses

    agent = agent if agent is not None else Agent(config)
    app = TuiApp(config, agent)
    try:
        curses.wrapper(app.run)
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        pass
    finally:
        # Persist the final state, then release the provider and the database.
        app.save_session()
        if app.store is not None:
            app.store.close()
        agent.provider.close()
    return 0


__all__ = [
    "Transcript",
    "TranscriptEntry",
    "Command",
    "parse_command",
    "AgentWorker",
    "WorkerEvent",
    "EVENT_ANSWER",
    "EVENT_TOOL",
    "EVENT_ERROR",
    "EVENT_DONE",
    "wrap_text",
    "render_transcript",
    "TuiApp",
    "run_tui",
    "HELP_TEXT",
    "SessionStore",
]

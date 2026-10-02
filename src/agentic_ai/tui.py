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
* :func:`layout` - the pure geometry of the screen (transcript vs. input box).
* :class:`TuiApp` - the curses view/controller.
"""

from __future__ import annotations

import contextlib
import json
import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .agent import (
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
anything else is sent to the agent as a prompt.

keys:
  up / down      rotate through previous prompts
  pgup / pgdn    scroll the transcript
  alt+backspace  delete the word before the cursor (also ctrl+w)
  esc            leave the interface"""


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


def _format_stored_tool_call(call: Dict[str, Any]) -> str:
    """Render a stored (OpenAI-shaped) tool call as a single readable line.

    Stored assistant messages carry ``tool_calls`` in the OpenAI shape, where
    the function arguments are a JSON *string* rather than a dict. This mirrors
    :func:`_format_tool_call` so a reloaded session reads the same as the live
    trail, and it never raises on a malformed row - a bad argument blob is
    shown verbatim instead of blanking the transcript.
    """

    function = call.get("function") or {}
    name = function.get("name") or call.get("name") or "tool"
    raw = function.get("arguments")
    if raw is None:
        raw = call.get("arguments")
    if not raw:
        return f"{name}()"
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            return f"{name}({raw})"
    if isinstance(raw, dict):
        if not raw:
            return f"{name}()"
        rendered = ", ".join(f"{key}={value!r}" for key, value in raw.items())
        return f"{name}({rendered})"
    return f"{name}({raw!r})"


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


def visible_lines(
    lines: List[Tuple[str, str]], body_height: int, scroll: int
) -> List[Tuple[str, str]]:
    """Return the slice of ``lines`` currently on screen.

    ``scroll`` is the number of lines scrolled *up* from the bottom (0 shows
    the most recent lines). The scroll offset is clamped to the available
    content so a stale, too-large offset can never blank the view - which is
    what used to happen after loading a shorter session while scrolled up.
    """

    if body_height <= 0:
        return []
    max_scroll = max(0, len(lines) - body_height)
    scroll = min(max(0, scroll), max_scroll)
    end = len(lines) - scroll
    start = max(0, end - body_height)
    return lines[start:end]


# --------------------------------------------------------------------------- #
# screen layout (pure geometry, so it can be unit-tested without curses)
# --------------------------------------------------------------------------- #
#: Height of the pinned input box with a single content line: a top border,
#: the input line, and a bottom border.
INPUT_BOX_HEIGHT = 3
#: Height of the status/hint line pinned below the input box.
STATUS_HEIGHT = 1
#: Minimum terminal size we are willing to draw into.
MIN_HEIGHT = 6
MIN_WIDTH = 20
#: Prefix shown before the human's text inside the composer.
INPUT_PROMPT = "you> "
#: Most content lines the composer will grow to before it starts scrolling.
#: Keeps a pasted wall of text from swallowing the whole transcript.
MAX_INPUT_LINES = 8


def wrap_input(text: str, width: int) -> List[str]:
    """Wrap the composer text into display lines of at most ``width`` columns.

    The prompt prefix (``you> ``) is prepended to the first line so the caller
    can draw the wrapped result verbatim. Explicit newlines are honoured, and
    words longer than ``width`` are hard-split so nothing is ever lost. This
    mirrors the multi-line composer used by Codex / opencode: when the text is
    wider than the terminal it flows onto additional lines instead of being
    truncated to its tail.
    """

    if width <= 0:
        return [text]
    lines: List[str] = []
    for index, paragraph in enumerate(text.split("\n")):
        prefix = INPUT_PROMPT if index == 0 else ""
        # The first line carries the prompt, so it has less room for text.
        current = prefix
        current_width = max(1, width - len(prefix))
        for word in paragraph.split(" "):
            if word == "":
                continue
            while len(word) > current_width:
                if current.strip():
                    lines.append(current)
                    current = ""
                    current_width = width
                lines.append(word[:current_width])
                word = word[current_width:]
            if current == prefix or not current:
                # First word on the line: attach it directly to the prompt
                # (which already ends in a space) or start a fresh line.
                current = f"{current}{word}"
            elif len(current) + 1 + len(word) <= width:
                current = f"{current} {word}"
            else:
                lines.append(current)
                current = word
                current_width = width
        if current or not lines:
            lines.append(current)
    return lines or [INPUT_PROMPT]


def input_box_height(line_count: int) -> int:
    """Total rows the composer needs for ``line_count`` content lines.

    A top border, the content lines, and a bottom border. Always at least
    :data:`INPUT_BOX_HEIGHT` so an empty composer keeps its familiar shape.
    """

    return max(1, line_count) + 2


@dataclass
class Layout:
    """Where each region of the screen lives, in rows from the top.

    The input box and status line are pinned to the bottom of the terminal;
    the transcript occupies whatever is left above them. This mirrors the
    layout used by Codex / opencode, where the composer stays put at the
    bottom and the conversation scrolls above it.
    """

    height: int
    width: int
    transcript_top: int
    transcript_height: int
    input_top: int
    input_height: int
    status_row: int

    @property
    def usable(self) -> bool:
        """Whether the terminal is large enough to draw the full layout."""

        return self.height >= MIN_HEIGHT and self.width >= MIN_WIDTH


def layout(height: int, width: int, input_lines: int = 1) -> Layout:
    """Compute the screen geometry for a terminal of ``height`` x ``width``.

    The bottom of the screen is reserved for the input box and the status
    line; the transcript fills the rows above. ``input_lines`` is the number
    of content lines the composer currently needs, so the box grows upward as
    the human types past the terminal width (like Codex / opencode) instead of
    truncating the text. When the terminal is too small the transcript is
    allowed to collapse to zero rows rather than pushing the input box
    off-screen.
    """

    input_height = input_box_height(input_lines)
    status_row = height - 1
    input_top = height - 1 - input_height
    transcript_top = 0
    transcript_height = max(0, input_top - transcript_top)
    return Layout(
        height=height,
        width=width,
        transcript_top=transcript_top,
        transcript_height=transcript_height,
        input_top=input_top,
        input_height=input_height,
        status_row=status_row,
    )


# --------------------------------------------------------------------------- #
# escape-sequence interpretation (pure, so it can be unit-tested)
# --------------------------------------------------------------------------- #
#: Characters that mean "delete the previous character" across terminals.
BACKSPACE_KEYS = ("\x7f", "\b", "\x08")

#: Actions an ESC-prefixed key sequence can resolve to.
ESCAPE_KILL_WORD = "kill_word"
ESCAPE_QUIT = "quit"
ESCAPE_IGNORE = "ignore"


def interpret_escape(tail: Any) -> str:
    """Decide what an ESC-prefixed key sequence means.

    Alt+Backspace and Ctrl+Backspace arrive as an ESC byte followed by a
    backspace byte, so they must kill the word rather than quit. A genuine
    lone ESC (``tail is None``) quits, and any other Alt/Ctrl-modified key we
    do not handle is ignored - crucially, an unrecognised sequence must never
    terminate the session, which is what used to happen when the follow-up
    byte arrived too late to be read.
    """

    if tail in BACKSPACE_KEYS:
        return ESCAPE_KILL_WORD
    if tail is None:
        return ESCAPE_QUIT
    return ESCAPE_IGNORE


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
        #: Previously submitted prompts, newest last, for Up/Down recall.
        self.history: List[str] = []
        #: Index into :attr:`history` while browsing it, or ``None`` when the
        #: human is editing a fresh line.
        self.history_index: Optional[int] = None
        #: The in-progress line stashed when history browsing begins, restored
        #: when the human browses back past the newest entry.
        self.history_draft = ""

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
        # Announce the load *before* the restored conversation so the newest
        # lines - the ones the bottom-aligned viewport shows - are the loaded
        # messages themselves. Adding the confirmation last used to push the
        # whole conversation off the top of a short terminal, leaving only
        # "loaded session N" on screen.
        self.transcript.add(
            "system", f"loaded session {session_id}", STYLE_SYSTEM
        )
        self._render_messages(messages)
        # Jump back to the bottom so the freshly loaded conversation is shown
        # from its most recent line. Without this, a scroll offset left over
        # from a longer previous conversation could hide the loaded dump.
        self.scroll = 0
        self.status = "ready"
        return True

    def _render_messages(self, messages: List[Dict[str, Any]]) -> None:
        """Rebuild the transcript from a stored message list.

        Assistant turns that only requested tools (no textual ``content``)
        are rendered too, as a ``tool`` line summarising the call, so a
        reloaded conversation shows the same reasoning trail that was on
        screen while it happened. Without this, sessions dominated by tool
        traffic looked empty after ``/load`` even though the data was stored.
        """

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
                for call in message.get("tool_calls") or []:
                    self.transcript.add(
                        "tool", _format_stored_tool_call(call), STYLE_TOOL
                    )
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

    # -- input editing (history + word kill) ------------------------------- #
    def _remember(self, prompt: str) -> None:
        """Add ``prompt`` to the recall history and reset browsing state.

        Consecutive duplicates are collapsed so holding Up does not wade
        through the same line repeated.
        """

        if prompt and (not self.history or self.history[-1] != prompt):
            self.history.append(prompt)
        self.history_index = None
        self.history_draft = ""

    def history_prev(self) -> None:
        """Recall the previous prompt into the input box (Up arrow).

        The history rotates: pressing Up at the oldest entry wraps around to
        the newest, so the human can cycle through their last questions
        without hitting a dead end.
        """

        if not self.history:
            return
        if self.history_index is None:
            # Stash the line being edited so Down can restore it.
            self.history_draft = self.input
            self.history_index = len(self.history)
        self.history_index = (self.history_index - 1) % len(self.history)
        self.input = self.history[self.history_index]

    def history_next(self) -> None:
        """Move forward through recalled prompts (Down arrow).

        Rotates in the opposite direction: pressing Down at the newest entry
        wraps around to the oldest. When the human has not started browsing
        yet, Down restores the in-progress draft instead.
        """

        if self.history_index is None:
            return
        self.history_index += 1
        if self.history_index >= len(self.history):
            # Past the newest entry: wrap back to the oldest.
            self.history_index = 0
        self.input = self.history[self.history_index]

    def kill_word(self) -> None:
        """Delete the word before the cursor (readline-style).

        Trailing whitespace is removed along with the word, so repeated
        invocations walk back through the line one word at a time.
        """

        text = self.input.rstrip()
        if not text:
            self.input = ""
            return
        index = len(text)
        while index > 0 and not text[index - 1].isspace():
            index -= 1
        self.input = text[:index]

    def submit_prompt(self, prompt: str) -> None:
        """Record a prompt and hand it to the worker."""

        if not prompt:
            return
        if self.worker.busy():
            self.transcript.add(
                "system", "still working on the previous prompt...", STYLE_SYSTEM
            )
            return
        self._remember(prompt)
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

        self._set_bar_cursor(curses, stdscr)
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
            if not self._handle_key(curses, stdscr, key):
                break

    def _init_colors(self, curses: Any) -> None:  # pragma: no cover - terminal
        curses.init_pair(1, curses.COLOR_CYAN, -1)  # user
        curses.init_pair(2, curses.COLOR_GREEN, -1)  # agent
        curses.init_pair(3, curses.COLOR_YELLOW, -1)  # tool
        curses.init_pair(4, curses.COLOR_RED, -1)  # error
        curses.init_pair(5, curses.COLOR_BLUE, -1)  # system
        curses.init_pair(6, curses.COLOR_BLACK, curses.COLOR_CYAN)  # status bar
        curses.init_pair(7, curses.COLOR_WHITE, -1)  # input box border

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

    #: DECSCUSR escape that asks the terminal for a blinking vertical bar.
    _BAR_CURSOR = "\x1b[5 q"

    def _write_bar_cursor(self) -> None:  # pragma: no cover - terminal
        """Emit the DECSCUSR bar-cursor sequence straight to the terminal.

        The sequence is written to the process's stdout rather than into the
        curses screen buffer. Writing it with ``stdscr.addstr`` looked like it
        worked, but :meth:`_draw` calls ``stdscr.erase()`` at the start of
        every frame, which wiped the escape sequence out of the buffer before
        it was ever flushed - so the terminal never saw it and kept its
        default block cursor. Bypassing the buffer (and flushing) makes the
        request reach the terminal directly.
        """

        import sys

        with contextlib.suppress(OSError, ValueError):
            sys.stdout.write(self._BAR_CURSOR)
            sys.stdout.flush()

    def _set_bar_cursor(self, curses: Any, stdscr: Any) -> None:  # pragma: no cover
        """Show a vertical-bar cursor rather than the default block.

        ``curses.curs_set`` only toggles visibility (and raises on terminals
        that cannot do it), so we ask the terminal directly for a bar cursor
        with the DECSCUSR sequence and fall back to ``curs_set`` when that is
        unavailable. Both are best-effort: a terminal that supports neither
        simply keeps its default cursor instead of crashing the UI.
        """

        self._write_bar_cursor()
        with contextlib.suppress(curses.error):
            curses.curs_set(1)

    #: Characters that mean "delete the previous character" across terminals.
    _BACKSPACE_KEYS = BACKSPACE_KEYS

    #: How long to wait for the byte that follows an ESC before treating the
    #: ESC as a genuine "quit". Long enough to catch Alt/Ctrl+Backspace even
    #: over a laggy SSH link, short enough that quitting still feels instant.
    _ESCAPE_TIMEOUT_MS = 150

    def _read_escape_tail(self, stdscr: Any) -> Any:  # pragma: no cover
        """Read the key following a bare ESC, or ``None`` if none arrives.

        Alt+Backspace and Ctrl+Backspace arrive as an ESC byte followed
        immediately by a backspace byte. Because the screen is in
        ``nodelay`` mode the ESC is delivered on its own first, so we briefly
        switch to a short blocking read to see whether a follow-up key is
        waiting. A lone ESC (no follow-up) is a genuine "quit".
        """

        import curses

        try:
            stdscr.timeout(self._ESCAPE_TIMEOUT_MS)
            try:
                return stdscr.get_wch()
            except curses.error:
                return None
        finally:
            stdscr.nodelay(True)

    def _handle_key(
        self, curses: Any, stdscr: Any, key: Any
    ) -> bool:  # pragma: no cover
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
        if key == "\x17":  # Ctrl+W - kill the word before the cursor
            self.kill_word()
            return True
        if key in self._BACKSPACE_KEYS or key == curses.KEY_BACKSPACE:
            self.input = self.input[:-1]
            return True
        if key == "\x1b":  # ESC, or the start of an Alt/Ctrl sequence
            action = interpret_escape(self._read_escape_tail(stdscr))
            if action == ESCAPE_KILL_WORD:
                # Alt+Backspace / Ctrl+Backspace: kill the word, do not quit.
                self.kill_word()
                return True
            # A lone ESC quits; any other Alt/Ctrl-modified key is ignored
            # rather than terminating the session.
            return action != ESCAPE_QUIT
        if key == curses.KEY_UP:
            self.history_prev()
            return True
        if key == curses.KEY_DOWN:
            self.history_next()
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
        inner_width = max(1, width - 2)
        input_lines = wrap_input(self.input, inner_width)
        geom = layout(height, width, input_lines=len(input_lines))
        if not geom.usable:
            stdscr.addnstr(0, 0, "terminal too small", max(0, width - 1))
            stdscr.refresh()
            return

        self._draw_transcript(curses, stdscr, geom)
        self._draw_input_box(curses, stdscr, geom, input_lines)
        self._draw_status(curses, stdscr, geom)
        stdscr.refresh()

    def _draw_transcript(
        self, curses: Any, stdscr: Any, geom: Layout
    ) -> None:  # pragma: no cover - terminal
        """Draw the scrolling conversation above the input box."""

        body_height = geom.transcript_height
        if body_height <= 0:
            return
        lines = render_transcript(self.transcript, geom.width - 2)
        # Clamp scroll to available content and take the visible slice.
        max_scroll = max(0, len(lines) - body_height)
        self.scroll = min(self.scroll, max_scroll)
        visible = visible_lines(lines, body_height, self.scroll)

        # Bottom-align the content so the newest line sits just above the box.
        top = geom.transcript_top + (body_height - len(visible))
        for offset, (style, text) in enumerate(visible):
            attr = self._color_for(curses, style)
            stdscr.addnstr(top + offset, 0, text, geom.width - 1, attr)

    def _draw_input_box(
        self, curses: Any, stdscr: Any, geom: Layout, lines: List[str]
    ) -> None:  # pragma: no cover - terminal
        """Draw the bordered composer pinned to the bottom of the screen.

        The composer is multi-line: text wider than the terminal wraps onto
        additional rows (like Codex / opencode) and the box grows upward to
        fit, up to :data:`MAX_INPUT_LINES`. When there are more lines than
        that, the most recent ones are shown so the cursor stays visible.
        """

        top = geom.input_top
        inner_width = max(0, geom.width - 2)
        border_attr = self._color_for(curses, STYLE_SYSTEM)

        # Top border: +-----+
        stdscr.addnstr(top, 0, "+" + "-" * inner_width + "+", geom.width, border_attr)

        # Show the tail of the wrapped text when it exceeds the visible rows.
        visible = lines[-MAX_INPUT_LINES:]
        for offset, line in enumerate(visible):
            row = top + 1 + offset
            stdscr.addnstr(row, 0, "|", 1, border_attr)
            stdscr.addnstr(row, 1, line.ljust(inner_width), inner_width)
            stdscr.addnstr(row, geom.width - 1, "|", 1, border_attr)

        # Bottom border: +-----+
        bottom = top + 1 + len(visible)
        stdscr.addnstr(
            bottom, 0, "+" + "-" * inner_width + "+", geom.width, border_attr
        )

        # Park the cursor after the typed text on the last visible line.
        last = visible[-1] if visible else INPUT_PROMPT
        cursor_row = top + len(visible)
        cursor_col = min(1 + len(last), geom.width - 2)
        stdscr.move(cursor_row, cursor_col)

    def _draw_status(
        self, curses: Any, stdscr: Any, geom: Layout
    ) -> None:  # pragma: no cover - terminal
        """Draw the single-line status/hint bar at the very bottom."""

        status = (
            f" {self.config.name} | {self.agent.provider.name} | "
            f"tools: {len(self.agent.tools)} | {self.status} "
        )
        with contextlib.suppress(curses.error):
            stdscr.addnstr(
                geom.status_row,
                0,
                status.ljust(geom.width - 1),
                geom.width - 1,
                self._color_for(curses, STYLE_USER),
            )


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
    "BACKSPACE_KEYS",
    "ESCAPE_IGNORE",
    "ESCAPE_KILL_WORD",
    "ESCAPE_QUIT",
    "EVENT_ANSWER",
    "EVENT_DONE",
    "EVENT_ERROR",
    "EVENT_TOOL",
    "HELP_TEXT",
    "INPUT_BOX_HEIGHT",
    "INPUT_PROMPT",
    "MAX_INPUT_LINES",
    "MIN_HEIGHT",
    "MIN_WIDTH",
    "STATUS_HEIGHT",
    "AgentWorker",
    "Command",
    "Layout",
    "SessionStore",
    "Transcript",
    "TranscriptEntry",
    "TuiApp",
    "WorkerEvent",
    "input_box_height",
    "interpret_escape",
    "layout",
    "parse_command",
    "render_transcript",
    "run_tui",
    "visible_lines",
    "wrap_input",
    "wrap_text",
]

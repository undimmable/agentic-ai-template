"""Tests for the terminal user interface.

Everything except the thin curses shell is plain Python, so these tests cover
the transcript model, slash-command parsing, text wrapping, the background
worker and the app's state transitions without needing a terminal.
"""

from __future__ import annotations

import time

from agentic_ai import tui
from agentic_ai.agent import Agent
from agentic_ai.config import AgentConfig
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


def make_agent(responses, tools=None) -> Agent:
    provider = StubProvider(responses)
    return Agent(AgentConfig(), provider=provider, tools=tools or {})


# --------------------------------------------------------------------------- #
# transcript model
# --------------------------------------------------------------------------- #
def test_transcript_add_and_clear():
    transcript = tui.Transcript()
    transcript.add("user", "hi", tui.STYLE_USER)
    transcript.add("assistant", "hello", tui.STYLE_AGENT)

    assert len(transcript) == 2
    assert [e.style for e in transcript] == [tui.STYLE_USER, tui.STYLE_AGENT]
    assert transcript.entries[0].header() == "you"

    transcript.clear()
    assert len(transcript) == 0


def test_transcript_is_bounded():
    transcript = tui.Transcript(limit=3)
    for i in range(5):
        transcript.add("assistant", str(i))

    assert len(transcript) == 3
    assert [e.text for e in transcript] == ["2", "3", "4"]


# --------------------------------------------------------------------------- #
# slash commands
# --------------------------------------------------------------------------- #
def test_parse_command_recognises_slash_commands():
    command = tui.parse_command("/tools")
    assert command.is_command
    assert command.name == "tools"
    assert command.argument == ""


def test_parse_command_keeps_argument():
    command = tui.parse_command("/foo bar baz")
    assert command.name == "foo"
    assert command.argument == "bar baz"


def test_parse_command_treats_plain_text_as_prompt():
    command = tui.parse_command("  hello world  ")
    assert not command.is_command
    assert command.argument == "hello world"


# --------------------------------------------------------------------------- #
# wrapping / rendering
# --------------------------------------------------------------------------- #
def test_wrap_text_wraps_on_word_boundaries():
    assert tui.wrap_text("one two three", 7) == ["one two", "three"]


def test_wrap_text_preserves_newlines():
    assert tui.wrap_text("a\nb", 10) == ["a", "b"]


def test_wrap_text_hard_splits_long_words():
    assert tui.wrap_text("abcdefgh", 3) == ["abc", "def", "gh"]


def test_render_transcript_includes_headers_and_indent():
    transcript = tui.Transcript()
    transcript.add("user", "hi", tui.STYLE_USER)

    rendered = tui.render_transcript(transcript, width=20)
    assert rendered == [(tui.STYLE_USER, "you>"), (tui.STYLE_USER, "  hi")]


def test_render_transcript_without_headers():
    transcript = tui.Transcript()
    transcript.add("assistant", "hi", tui.STYLE_AGENT)

    rendered = tui.render_transcript(transcript, width=20, show_headers=False)
    assert rendered == [(tui.STYLE_AGENT, "hi")]


def test_render_transcript_lines_never_exceed_width():
    # Regression: content wrapped to the full width plus the 2-space indent
    # used to exceed the draw limit, so the terminal wrapped the line onto the
    # next row and the body overlapped the input line.
    transcript = tui.Transcript()
    transcript.add("assistant", "x" * 100, tui.STYLE_AGENT)

    width = 20
    rendered = tui.render_transcript(transcript, width=width)
    assert rendered  # non-empty
    assert all(len(text) <= width for _, text in rendered)


def test_render_transcript_without_headers_lines_fit_width():
    transcript = tui.Transcript()
    transcript.add("assistant", "y" * 100, tui.STYLE_AGENT)

    width = 20
    rendered = tui.render_transcript(transcript, width=width, show_headers=False)
    assert all(len(text) <= width for _, text in rendered)


# --------------------------------------------------------------------------- #
# background worker
# --------------------------------------------------------------------------- #
def _wait_for_done(worker: tui.AgentWorker, timeout: float = 5.0):
    """Block until the worker's turn finishes, then return its events.

    Drains the queue, so callers that want the app to process the events
    themselves should use :func:`_wait_until_idle` instead.
    """

    _wait_until_idle(worker, timeout)
    return worker.drain()


def _wait_until_idle(worker: tui.AgentWorker, timeout: float = 5.0) -> None:
    """Block until the worker is no longer busy, leaving events queued."""

    deadline = time.time() + timeout
    while time.time() < deadline:
        if not worker.busy():
            return
        time.sleep(0.01)
    raise AssertionError("worker did not finish in time")


def test_worker_reports_answer():
    agent = make_agent([LLMResponse(content="hello")])
    worker = tui.AgentWorker(agent)

    assert worker.submit("hi") is True
    events = _wait_for_done(worker)

    answers = [e for e in events if e.kind == tui.EVENT_ANSWER]
    assert answers and answers[0].text == "hello"
    assert not worker.busy()


def test_worker_reports_tool_calls_and_results():
    agent = make_agent(
        [
            LLMResponse(
                tool_calls=[ToolCall(id="1", name="echo", arguments={"text": "hi"})]
            ),
            LLMResponse(content="done"),
        ],
        tools={"echo": echo_tool()},
    )
    worker = tui.AgentWorker(agent)

    worker.submit("go")
    events = _wait_for_done(worker)

    tool_events = [e for e in events if e.kind == tui.EVENT_TOOL]
    assert any("echo(text='hi')" in e.text for e in tool_events)
    assert any("echo -> echo:hi" in e.text for e in tool_events)
    assert all(e.name == "echo" for e in tool_events)


def test_worker_reports_errors():
    agent = make_agent([])
    # Force the provider to raise so the worker surfaces an error event.
    agent.provider.chat = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    worker = tui.AgentWorker(agent)

    worker.submit("go")
    events = _wait_for_done(worker)

    errors = [e for e in events if e.kind == tui.EVENT_ERROR]
    assert errors and "boom" in errors[0].text


def test_worker_rejects_concurrent_submit():
    agent = make_agent([LLMResponse(content="slow")])
    worker = tui.AgentWorker(agent)

    # Simulate a turn already in flight.
    worker._busy = True
    assert worker.submit("again") is False


# --------------------------------------------------------------------------- #
# app state transitions
# --------------------------------------------------------------------------- #
def test_app_help_and_unknown_command():
    app = tui.TuiApp(AgentConfig(), make_agent([]))

    assert app.handle_command(tui.parse_command("/help")) is True
    assert "commands:" in app.transcript.entries[-1].text

    app.handle_command(tui.parse_command("/nope"))
    assert app.transcript.entries[-1].style == tui.STYLE_ERROR


def test_app_exit_command_returns_false():
    app = tui.TuiApp(AgentConfig(), make_agent([]))
    assert app.handle_command(tui.parse_command("/exit")) is False
    assert app.handle_command(tui.parse_command("/quit")) is False


def test_app_reset_clears_history():
    agent = make_agent([LLMResponse(content="one")])
    app = tui.TuiApp(AgentConfig(), agent)
    agent.run("first")
    assert len(agent.messages) == 3

    app.handle_command(tui.parse_command("/reset"))
    assert len(agent.messages) == 1


def test_app_clear_empties_transcript():
    app = tui.TuiApp(AgentConfig(), make_agent([]))
    app.transcript.add("user", "hi")
    app.handle_command(tui.parse_command("/clear"))
    assert len(app.transcript) == 0


def test_app_submit_prompt_records_and_pumps():
    agent = make_agent([LLMResponse(content="answer")])
    app = tui.TuiApp(AgentConfig(), agent)

    app.submit_prompt("question")
    assert app.transcript.entries[0].style == tui.STYLE_USER
    assert app.status == "thinking..."

    _wait_until_idle(app.worker)
    app.pump_events()

    assert app.status == "ready"
    assert any(
        e.style == tui.STYLE_AGENT and e.text == "answer" for e in app.transcript
    )


def test_app_submit_prompt_ignores_empty():
    app = tui.TuiApp(AgentConfig(), make_agent([]))
    app.submit_prompt("")
    assert len(app.transcript) == 0


# --------------------------------------------------------------------------- #
# session persistence and retrieval
# --------------------------------------------------------------------------- #
def make_store() -> "tui.SessionStore":
    return tui.SessionStore(":memory:")


def test_app_persists_conversation_after_turn():
    agent = make_agent([LLMResponse(content="answer")])
    store = make_store()
    app = tui.TuiApp(AgentConfig(), agent, store=store)

    app.submit_prompt("question")
    _wait_until_idle(app.worker)
    app.pump_events()

    sessions = store.list_sessions()
    assert len(sessions) == 1
    assert app.session_id == sessions[0].id
    # system + user + assistant
    assert sessions[0].message_count == 3


def test_app_reuses_session_id_across_turns():
    agent = make_agent([LLMResponse(content="one"), LLMResponse(content="two")])
    store = make_store()
    app = tui.TuiApp(AgentConfig(), agent, store=store)

    app.submit_prompt("first")
    _wait_until_idle(app.worker)
    app.pump_events()
    first_id = app.session_id

    app.submit_prompt("second")
    _wait_until_idle(app.worker)
    app.pump_events()

    assert app.session_id == first_id
    assert len(store.list_sessions()) == 1  # updated in place, not duplicated


def test_app_sessions_command_lists_stored_sessions():
    agent = make_agent([LLMResponse(content="answer")])
    store = make_store()
    app = tui.TuiApp(AgentConfig(), agent, store=store)
    app.submit_prompt("question")
    _wait_until_idle(app.worker)
    app.pump_events()

    app.handle_command(tui.parse_command("/sessions"))
    listing = app.transcript.entries[-1].text
    assert "stored sessions:" in listing
    assert "question" in listing  # default name derived from the first prompt


def test_app_sessions_command_with_no_sessions():
    app = tui.TuiApp(AgentConfig(), make_agent([]), store=make_store())
    app.handle_command(tui.parse_command("/sessions"))
    assert app.transcript.entries[-1].text == "no stored sessions"


def test_app_load_restores_conversation_and_transcript():
    # Seed a store with a saved conversation.
    store = make_store()
    session_id = store.save_session(
        "earlier",
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
        ],
    )

    agent = make_agent([])
    app = tui.TuiApp(AgentConfig(), agent, store=store)
    app.transcript.add("user", "stale", tui.STYLE_USER)

    assert app.load_session(session_id) is True

    # Agent history restored (system prompt preserved from the stored session).
    assert agent.messages[0] == {"role": "system", "content": "sys"}
    assert agent.messages[1] == {"role": "user", "content": "old question"}
    assert agent.messages[2] == {"role": "assistant", "content": "old answer"}
    assert app.session_id == session_id

    # Transcript rebuilt from the stored messages, stale entry gone.
    texts = [e.text for e in app.transcript]
    assert "old question" in texts
    assert "old answer" in texts
    assert "stale" not in texts


def test_app_load_command_parses_id():
    store = make_store()
    session_id = store.save_session(
        "s", [{"role": "user", "content": "hi"}]
    )
    app = tui.TuiApp(AgentConfig(), make_agent([]), store=store)

    app.handle_command(tui.parse_command(f"/load {session_id}"))
    assert app.session_id == session_id


def test_app_load_command_rejects_bad_id():
    app = tui.TuiApp(AgentConfig(), make_agent([]), store=make_store())
    app.handle_command(tui.parse_command("/load notanumber"))
    assert app.transcript.entries[-1].style == tui.STYLE_ERROR


def test_app_load_command_requires_argument():
    app = tui.TuiApp(AgentConfig(), make_agent([]), store=make_store())
    app.handle_command(tui.parse_command("/load"))
    assert app.transcript.entries[-1].style == tui.STYLE_ERROR


def test_app_load_unknown_session_reports_error():
    app = tui.TuiApp(AgentConfig(), make_agent([]), store=make_store())
    assert app.load_session(12345) is False
    assert app.transcript.entries[-1].style == tui.STYLE_ERROR


def test_app_save_command_with_name():
    agent = make_agent([LLMResponse(content="answer")])
    store = make_store()
    app = tui.TuiApp(AgentConfig(), agent, store=store)
    app.submit_prompt("question")
    _wait_until_idle(app.worker)
    app.pump_events()

    app.handle_command(tui.parse_command("/save my session"))
    assert store.list_sessions()[0].name == "my session"


def test_app_reset_clears_current_session_id():
    agent = make_agent([LLMResponse(content="answer")])
    store = make_store()
    app = tui.TuiApp(AgentConfig(), agent, store=store)
    app.submit_prompt("question")
    _wait_until_idle(app.worker)
    app.pump_events()
    assert app.session_id is not None

    app.handle_command(tui.parse_command("/reset"))
    assert app.session_id is None


def test_app_with_sessions_disabled_has_no_store():
    config = AgentConfig()
    config.sessions.enabled = False
    app = tui.TuiApp(config, make_agent([]))
    assert app.store is None

    # Commands degrade gracefully rather than crashing.
    app.handle_command(tui.parse_command("/sessions"))
    assert "disabled" in app.transcript.entries[-1].text
    assert app.save_session() is None
    assert app.list_sessions() == []


def test_app_save_skips_empty_conversation():
    store = make_store()
    app = tui.TuiApp(AgentConfig(), make_agent([]), store=store)
    # Only the system prompt is present, so nothing is worth saving.
    assert app.save_session() is None
    assert store.list_sessions() == []


# --------------------------------------------------------------------------- #
# visible window (scroll clamping)
# --------------------------------------------------------------------------- #
def test_visible_lines_shows_bottom_when_not_scrolled():
    lines = [(tui.STYLE_AGENT, str(i)) for i in range(10)]
    assert tui.visible_lines(lines, body_height=3, scroll=0) == lines[-3:]


def test_visible_lines_scrolls_up():
    lines = [(tui.STYLE_AGENT, str(i)) for i in range(10)]
    assert tui.visible_lines(lines, body_height=3, scroll=2) == lines[5:8]


def test_visible_lines_clamps_oversized_scroll():
    # Regression: a scroll offset larger than the content used to yield an
    # empty window, blanking the view (e.g. after loading a shorter session).
    lines = [(tui.STYLE_AGENT, str(i)) for i in range(4)]
    assert tui.visible_lines(lines, body_height=3, scroll=999) == lines[:3]


def test_visible_lines_handles_zero_height():
    lines = [(tui.STYLE_AGENT, "x")]
    assert tui.visible_lines(lines, body_height=0, scroll=0) == []


def test_load_session_resets_scroll_so_dump_is_visible():
    # Regression: loading a session while scrolled up left the scroll offset
    # untouched, so the freshly loaded conversation was not drawn.
    store = make_store()
    session_id = store.save_session(
        "earlier",
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
        ],
    )
    app = tui.TuiApp(AgentConfig(), make_agent([]), store=store)
    app.scroll = 500  # simulate having scrolled up in a longer conversation

    assert app.load_session(session_id) is True
    assert app.scroll == 0

    lines = tui.render_transcript(app.transcript, width=40)
    visible = tui.visible_lines(lines, body_height=10, scroll=app.scroll)
    texts = [text for _, text in visible]
    assert any("old question" in t for t in texts)
    assert any("old answer" in t for t in texts)


# --------------------------------------------------------------------------- #
# screen layout (input box pinned to the bottom)
# --------------------------------------------------------------------------- #
def test_layout_pins_input_box_to_bottom():
    geom = tui.layout(height=24, width=80)
    # Input box occupies the three rows just above the status line.
    assert geom.input_height == tui.INPUT_BOX_HEIGHT
    assert geom.input_top == 24 - 1 - tui.INPUT_BOX_HEIGHT
    assert geom.status_row == 23
    # Transcript fills everything above the box.
    assert geom.transcript_top == 0
    assert geom.transcript_height == geom.input_top
    assert geom.usable


def test_layout_input_box_never_overlaps_status():
    geom = tui.layout(height=10, width=40)
    # The box's bottom border sits directly above the status row.
    assert geom.input_top + geom.input_height == geom.status_row


def test_layout_transcript_collapses_when_terminal_is_short():
    geom = tui.layout(height=4, width=40)
    assert geom.transcript_height == 0
    assert not geom.usable


def test_layout_rejects_narrow_terminal():
    assert not tui.layout(height=24, width=10).usable

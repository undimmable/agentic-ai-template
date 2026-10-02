"""Tests for the SQLite-backed session store.

The store is plain Python with no terminal coupling, so these tests exercise
the full save/list/load/delete round-trip against an in-memory database (and a
temporary file for the on-disk path).
"""

from __future__ import annotations

import pytest

from agentic_ai.sessions import SessionError, SessionStore


def sample_messages():
    return [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "echo", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "1", "content": "echo:hi"},
        {"role": "assistant", "content": "done"},
    ]


@pytest.fixture
def store():
    store = SessionStore(":memory:")
    yield store
    store.close()


# --------------------------------------------------------------------------- #
# round-trip
# --------------------------------------------------------------------------- #
def test_save_and_load_roundtrip(store):
    session_id = store.save_session("first", sample_messages())
    assert isinstance(session_id, int)

    loaded = store.load_session(session_id)
    assert loaded is not None
    # The system prompt, user turn, tool call, tool result and answer survive.
    assert loaded[0] == {"role": "system", "content": "you are helpful"}
    assert loaded[1] == {"role": "user", "content": "hello"}
    assert loaded[2]["role"] == "assistant"
    assert loaded[2]["tool_calls"][0]["function"]["name"] == "echo"
    assert loaded[3]["role"] == "tool"
    assert loaded[4] == {"role": "assistant", "content": "done"}


def test_load_missing_session_returns_none(store):
    assert store.load_session(999) is None


def test_list_sessions_reports_counts_and_order(store):
    first = store.save_session("first", sample_messages())
    second = store.save_session("second", [{"role": "user", "content": "hi"}])

    sessions = store.list_sessions()
    assert [s.id for s in sessions] == [second, first]  # most recent first
    assert sessions[0].name == "second"
    assert sessions[0].message_count == 1
    assert sessions[1].message_count == len(sample_messages())


def test_latest_session_id(store):
    assert store.latest_session_id() is None
    store.save_session("a", [{"role": "user", "content": "1"}])
    second = store.save_session("b", [{"role": "user", "content": "2"}])
    assert store.latest_session_id() == second


# --------------------------------------------------------------------------- #
# updating and deleting
# --------------------------------------------------------------------------- #
def test_save_with_existing_id_updates_in_place(store):
    session_id = store.save_session("first", [{"role": "user", "content": "one"}])
    store.save_session(
        "renamed", [{"role": "user", "content": "two"}], session_id=session_id
    )

    sessions = store.list_sessions()
    assert len(sessions) == 1  # no new row created
    assert sessions[0].name == "renamed"
    assert store.load_session(session_id) == [{"role": "user", "content": "two"}]


def test_delete_session(store):
    session_id = store.save_session("gone", [{"role": "user", "content": "x"}])
    assert store.delete_session(session_id) is True
    assert store.load_session(session_id) is None
    assert store.list_sessions() == []
    assert store.delete_session(session_id) is False


def test_delete_cascades_messages(store):
    session_id = store.save_session("gone", sample_messages())
    store.delete_session(session_id)
    # The messages table should be empty after the cascade.
    count = store.connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    assert count == 0


# --------------------------------------------------------------------------- #
# persistence to disk
# --------------------------------------------------------------------------- #
def test_persists_across_connections(tmp_path):
    path = tmp_path / "sessions.db"
    with SessionStore(path) as store:
        session_id = store.save_session("disk", [{"role": "user", "content": "hi"}])

    # A fresh store on the same file sees the stored session.
    with SessionStore(path) as reopened:
        assert reopened.load_session(session_id) == [
            {"role": "user", "content": "hi"}
        ]


def test_creates_parent_directory(tmp_path):
    path = tmp_path / "nested" / "dir" / "sessions.db"
    with SessionStore(path) as store:
        store.save_session("nested", [{"role": "user", "content": "hi"}])
    assert path.exists()


def test_empty_message_list_is_stored(store):
    session_id = store.save_session("empty", [])
    assert store.load_session(session_id) == []
    assert store.list_sessions()[0].message_count == 0


def test_session_error_is_agentic_error():
    from agentic_ai.errors import AgenticError

    assert issubclass(SessionError, AgenticError)

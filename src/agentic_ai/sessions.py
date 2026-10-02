"""SQLite-backed storage for agent sessions.

A *session* is a named conversation: the ordered list of chat messages the
:class:`~agentic_ai.agent.Agent` accumulated while talking to a human. This
module persists those conversations in a small SQLite database so the TUI can
list previous sessions and reload one to continue where it left off.

The design follows the project's principles:

* **Stdlib only** - it uses :mod:`sqlite3` from the standard library, so no new
  dependency is introduced.
* **Config-driven** - the database path comes from ``agent.yaml`` (see
  :class:`~agentic_ai.config.SessionConfig`); nothing is hard-coded.
* **Testable** - :class:`SessionStore` is plain Python with no curses or
  terminal coupling, so it can be exercised against a temporary database.

Messages are stored in the OpenAI chat-completions shape the rest of the
runtime already uses. ``tool_calls`` (a nested list) is serialised to JSON so
the schema stays a flat, readable table.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .errors import AgenticError
from .logging_utils import get_logger

#: Default location of the session database, relative to the working directory.
DEFAULT_DB_PATH = ".agentic/sessions.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT    NOT NULL,
    created_at REAL    NOT NULL,
    updated_at REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    position   INTEGER NOT NULL,
    role       TEXT    NOT NULL,
    content    TEXT,
    tool_calls TEXT
);

CREATE INDEX IF NOT EXISTS idx_messages_session
    ON messages(session_id, position);
"""


class SessionError(AgenticError):
    """Raised when a session cannot be stored or retrieved."""


@dataclass
class SessionInfo:
    """A lightweight description of a stored session, for listing."""

    id: int
    name: str
    created_at: float
    updated_at: float
    message_count: int = 0

    @property
    def updated_at_str(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.updated_at))


def _encode_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten one chat message into a row of primitives."""

    tool_calls = message.get("tool_calls")
    return {
        "role": str(message.get("role", "")),
        "content": message.get("content"),
        "tool_calls": (
            json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None
        ),
    }


def _decode_message(row: sqlite3.Row) -> Dict[str, Any]:
    """Rebuild a chat message from a stored row."""

    message: Dict[str, Any] = {"role": row["role"]}
    if row["content"] is not None:
        message["content"] = row["content"]
    if row["tool_calls"]:
        try:
            message["tool_calls"] = json.loads(row["tool_calls"])
        except (TypeError, ValueError):  # pragma: no cover - corrupt row
            message["tool_calls"] = []
    return message


class SessionStore:
    """A small SQLite store for named conversation sessions.

    The connection is opened lazily on first use so constructing a store never
    touches the filesystem. Pass ``":memory:"`` as ``path`` for an ephemeral,
    in-process database (used by the tests).
    """

    def __init__(self, path: Any = DEFAULT_DB_PATH) -> None:
        self.path = str(path)
        self.logger = get_logger("sessions")
        self._conn: Optional[sqlite3.Connection] = None

    # -- connection management -------------------------------------------- #
    @property
    def connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = self._connect()
        return self._conn

    def _connect(self) -> sqlite3.Connection:
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        try:
            conn = sqlite3.connect(self.path)
        except sqlite3.Error as exc:  # pragma: no cover - filesystem failure
            raise SessionError(f"Could not open session database {self.path}: {exc}")
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(_SCHEMA)
        conn.commit()
        self.logger.debug("opened session database at %s", self.path)
        return conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "SessionStore":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    # -- writes ------------------------------------------------------------ #
    def save_session(
        self, name: str, messages: List[Dict[str, Any]], session_id: Optional[int] = None
    ) -> int:
        """Persist ``messages`` as a session, returning its id.

        When ``session_id`` is given the existing session is updated in place
        (its messages are replaced); otherwise a new session is created.
        """

        now = time.time()
        conn = self.connection
        try:
            with conn:  # transaction
                if session_id is None:
                    cursor = conn.execute(
                        "INSERT INTO sessions (name, created_at, updated_at) "
                        "VALUES (?, ?, ?)",
                        (name, now, now),
                    )
                    session_id = int(cursor.lastrowid)
                else:
                    conn.execute(
                        "UPDATE sessions SET name = ?, updated_at = ? WHERE id = ?",
                        (name, now, session_id),
                    )
                    conn.execute(
                        "DELETE FROM messages WHERE session_id = ?", (session_id,)
                    )
                conn.executemany(
                    "INSERT INTO messages "
                    "(session_id, position, role, content, tool_calls) "
                    "VALUES (?, ?, ?, ?, ?)",
                    [
                        (
                            session_id,
                            position,
                            encoded["role"],
                            encoded["content"],
                            encoded["tool_calls"],
                        )
                        for position, encoded in enumerate(
                            _encode_message(m) for m in messages
                        )
                    ],
                )
        except sqlite3.Error as exc:
            raise SessionError(f"Could not save session '{name}': {exc}")
        self.logger.info(
            "saved session %s ('%s', %d messages)", session_id, name, len(messages)
        )
        return session_id

    def delete_session(self, session_id: int) -> bool:
        """Delete a session and its messages. Returns whether a row was removed."""

        try:
            with self.connection as conn:
                cursor = conn.execute(
                    "DELETE FROM sessions WHERE id = ?", (session_id,)
                )
        except sqlite3.Error as exc:
            raise SessionError(f"Could not delete session {session_id}: {exc}")
        return cursor.rowcount > 0

    # -- reads ------------------------------------------------------------- #
    def list_sessions(self) -> List[SessionInfo]:
        """Return all sessions, most recently updated first."""

        try:
            rows = self.connection.execute(
                "SELECT s.id, s.name, s.created_at, s.updated_at, "
                "       COUNT(m.id) AS message_count "
                "FROM sessions s "
                "LEFT JOIN messages m ON m.session_id = s.id "
                "GROUP BY s.id "
                "ORDER BY s.updated_at DESC, s.id DESC"
            ).fetchall()
        except sqlite3.Error as exc:
            raise SessionError(f"Could not list sessions: {exc}")
        return [
            SessionInfo(
                id=row["id"],
                name=row["name"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                message_count=row["message_count"],
            )
            for row in rows
        ]

    def load_session(self, session_id: int) -> Optional[List[Dict[str, Any]]]:
        """Return the stored messages for ``session_id``, or ``None`` if absent."""

        conn = self.connection
        try:
            exists = conn.execute(
                "SELECT 1 FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if exists is None:
                return None
            rows = conn.execute(
                "SELECT role, content, tool_calls FROM messages "
                "WHERE session_id = ? ORDER BY position",
                (session_id,),
            ).fetchall()
        except sqlite3.Error as exc:
            raise SessionError(f"Could not load session {session_id}: {exc}")
        return [_decode_message(row) for row in rows]

    def latest_session_id(self) -> Optional[int]:
        """Return the id of the most recently updated session, if any."""

        sessions = self.list_sessions()
        return sessions[0].id if sessions else None


__all__ = [
    "SessionStore",
    "SessionInfo",
    "SessionError",
    "DEFAULT_DB_PATH",
]

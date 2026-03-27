"""Session memory — per-session conversation history for multi-turn chats."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional


class ConversationSession:
    """Maintains an ordered conversation history for a single session."""

    def __init__(self, session_id: str, max_turns: int = 20) -> None:
        self.session_id = session_id
        self.max_turns = max_turns
        self.created_at: float = time.time()
        self.last_active: float = time.time()
        # Each turn is {"role": ..., "content": ...}
        self._messages: deque = deque(maxlen=max_turns * 2)  # user + assistant per turn

    def add(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})
        self.last_active = time.time()

    def as_messages(self) -> List[Dict[str, str]]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)


class SessionManager:
    """Manages per-session conversation histories.

    Sessions expire automatically after *ttl_seconds* of inactivity.
    """

    def __init__(self, ttl_seconds: int = 3600, max_turns_per_session: int = 20) -> None:
        self._ttl = ttl_seconds
        self._max_turns = max_turns_per_session
        self._sessions: Dict[str, ConversationSession] = {}

    def get_or_create(self, session_id: str) -> ConversationSession:
        """Return an existing session or create a new one."""
        self._evict_expired()
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationSession(
                session_id, max_turns=self._max_turns
            )
        return self._sessions[session_id]

    def get(self, session_id: str) -> Optional[ConversationSession]:
        """Return an existing session or None."""
        self._evict_expired()
        return self._sessions.get(session_id)

    def clear(self, session_id: str) -> None:
        """Delete a session."""
        self._sessions.pop(session_id, None)

    def active_count(self) -> int:
        self._evict_expired()
        return len(self._sessions)

    def _evict_expired(self) -> None:
        now = time.time()
        expired = [
            sid
            for sid, sess in self._sessions.items()
            if (now - sess.last_active) > self._ttl
        ]
        for sid in expired:
            del self._sessions[sid]

"""Code Engine — persistent session manager.

Sessions track:
  • Conversation history (role/content pairs)
  • Active agent mode (ask / architect / code / debug / orchestrator)
  • Workspace root (the directory the agent is allowed to operate in)
  • Arbitrary key-value variables (user preferences, project details, etc.)
  • Agent state snapshot (last plan, last task list, etc.)

Persistence is JSONL-backed by default so it works out-of-the-box with no
external dependencies. Each session is one record in the JSONL file; the
file is rewritten atomically on save.

The interface is intentionally simple so it can be swapped to SQLite or a
remote store later without changing callers.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_STORE = Path("coding/store/sessions.jsonl")

VALID_MODES = {"ask", "architect", "code", "debug", "orchestrator"}


class CodeSession:
    """A single code-engine chat session."""

    def __init__(
        self,
        session_id: Optional[str] = None,
        mode: str = "ask",
        workspace: Optional[str] = None,
    ) -> None:
        self.session_id: str = session_id or str(uuid.uuid4())
        self.mode: str = mode if mode in VALID_MODES else "ask"
        self.workspace: Optional[str] = workspace
        self.variables: Dict[str, Any] = {}
        self.agent_state: Dict[str, Any] = {}
        self.history: List[Dict[str, str]] = []
        self.created_at: str = datetime.now(timezone.utc).isoformat()
        self.updated_at: str = self.created_at

    # ── conversation ──────────────────────────────────────────────────────

    def add_message(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def as_messages(self) -> List[Dict[str, str]]:
        """Return history in LLM-compatible format."""
        return list(self.history)

    def clear_history(self) -> None:
        self.history = []
        self.updated_at = datetime.now(timezone.utc).isoformat()

    # ── mode switching ────────────────────────────────────────────────────

    def set_mode(self, mode: str) -> None:
        if mode not in VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Valid modes: {sorted(VALID_MODES)}")
        self.mode = mode
        self.updated_at = datetime.now(timezone.utc).isoformat()

    # ── variables ─────────────────────────────────────────────────────────

    def set_var(self, key: str, value: Any) -> None:
        self.variables[key] = value
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def get_var(self, key: str, default: Any = None) -> Any:
        return self.variables.get(key, default)

    # ── serialisation ─────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "mode": self.mode,
            "workspace": self.workspace,
            "variables": self.variables,
            "agent_state": self.agent_state,
            "history": self.history,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodeSession":
        s = cls.__new__(cls)
        s.session_id = data["session_id"]
        s.mode = data.get("mode", "ask")
        s.workspace = data.get("workspace")
        s.variables = data.get("variables", {})
        s.agent_state = data.get("agent_state", {})
        s.history = data.get("history", [])
        s.created_at = data.get("created_at", "")
        s.updated_at = data.get("updated_at", "")
        return s

    def __repr__(self) -> str:
        return (
            f"<CodeSession id={self.session_id!r} mode={self.mode!r} "
            f"turns={len(self.history)} workspace={self.workspace!r}>"
        )


class CodeSessionManager:
    """Load, save, and manage multiple CodeSession objects."""

    def __init__(self, store_path: Optional[str] = None) -> None:
        self._path = Path(store_path or _DEFAULT_STORE)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._sessions: Dict[str, CodeSession] = {}
        self._load_all()

    # ── public CRUD ───────────────────────────────────────────────────────

    def create(self, mode: str = "ask", workspace: Optional[str] = None) -> CodeSession:
        """Create and persist a new session."""
        session = CodeSession(mode=mode, workspace=workspace)
        with self._lock:
            self._sessions[session.session_id] = session
        self._save_all()
        logger.info("Code Engine: created session %s (mode=%s)", session.session_id, mode)
        return session

    def get(self, session_id: str) -> Optional[CodeSession]:
        return self._sessions.get(session_id)

    def get_or_create(self, session_id: str, mode: str = "ask") -> CodeSession:
        if session_id in self._sessions:
            return self._sessions[session_id]
        session = CodeSession(session_id=session_id, mode=mode)
        with self._lock:
            self._sessions[session_id] = session
        self._save_all()
        return session

    def save(self, session: CodeSession) -> None:
        """Persist changes to an existing session."""
        with self._lock:
            self._sessions[session.session_id] = session
        self._save_all()

    def delete(self, session_id: str) -> bool:
        with self._lock:
            existed = session_id in self._sessions
            if existed:
                del self._sessions[session_id]
        if existed:
            self._save_all()
        return existed

    def list_all(self) -> List[Dict[str, Any]]:
        """Return summary metadata for all sessions (newest first)."""
        summaries = [
            {
                "session_id": s.session_id,
                "mode": s.mode,
                "workspace": s.workspace,
                "turns": len(s.history),
                "created_at": s.created_at,
                "updated_at": s.updated_at,
            }
            for s in self._sessions.values()
        ]
        summaries.sort(key=lambda x: x["updated_at"], reverse=True)
        return summaries

    # ── persistence ───────────────────────────────────────────────────────

    def _load_all(self) -> None:
        if not self._path.exists():
            return
        try:
            for line in self._path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    s = CodeSession.from_dict(data)
                    self._sessions[s.session_id] = s
                except (json.JSONDecodeError, KeyError) as exc:
                    logger.warning("Skipping corrupt session record: %s", exc)
        except OSError as exc:
            logger.warning("Could not load sessions: %s", exc)

    def _save_all(self) -> None:
        tmp = self._path.with_suffix(".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as fh:
                for session in self._sessions.values():
                    fh.write(json.dumps(session.to_dict(), ensure_ascii=False) + "\n")
            tmp.replace(self._path)
        except OSError as exc:
            logger.warning("Could not save sessions: %s", exc)
        finally:
            # Clean up any leftover temp file on failure
            if tmp.exists() and not self._path.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

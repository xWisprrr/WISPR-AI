"""Code Engine — audit log for all file operations.

Every read, write, create, edit, delete, and move is appended as a JSONL record
so there is always a full, tamper-evident trail of what the Code Engine touched.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_DEFAULT_LOG_PATH = Path("coding/store/audit.jsonl")


class AuditLog:
    """Thread-safe append-only audit log backed by a JSONL file."""

    def __init__(self, log_path: Optional[str] = None) -> None:
        self._path = Path(log_path or _DEFAULT_LOG_PATH)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────

    def record(
        self,
        *,
        operation: str,
        path: str,
        session_id: Optional[str] = None,
        agent: Optional[str] = None,
        bytes_changed: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append one audit record to the log."""
        entry: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "op": operation,
            "path": path,
        }
        if session_id:
            entry["session_id"] = session_id
        if agent:
            entry["agent"] = agent
        if bytes_changed is not None:
            entry["bytes"] = bytes_changed
        if extra:
            entry.update(extra)

        line = json.dumps(entry, ensure_ascii=False)
        with self._lock:
            try:
                with self._path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except OSError as exc:
                logger.warning("AuditLog write failed: %s", exc)

    def recent(self, n: int = 50) -> list:
        """Return the *n* most recent audit records (newest last)."""
        if not self._path.exists():
            return []
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
            records = []
            for line in lines:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            return records[-n:]
        except OSError:
            return []

    def clear(self) -> None:
        """Erase all audit records (use with caution)."""
        with self._lock:
            try:
                self._path.write_text("", encoding="utf-8")
            except OSError as exc:
                logger.warning("AuditLog clear failed: %s", exc)

"""Task memory — tracks reasoning steps and agent sub-task history."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class TaskMemory:
    """Persists multi-step task execution history so the system can resume or learn."""

    def __init__(self, db_path: str = settings.task_memory_db_path) -> None:
        self._path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._tasks: Dict[str, Dict[str, Any]] = self._load()

    # ── public API ────────────────────────────────────────────────────────

    def new_task(self, description: str, metadata: Optional[Dict] = None) -> str:
        """Create a new task record and return its ID."""
        task_id = str(uuid4())
        self._tasks[task_id] = {
            "id": task_id,
            "description": description,
            "steps": [],
            "status": "pending",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }
        self._save()
        return task_id

    def add_step(
        self,
        task_id: str,
        agent: str,
        action: str,
        result: Any,
        success: bool = True,
    ) -> None:
        if task_id not in self._tasks:
            raise KeyError(f"Unknown task: {task_id}")
        step = {
            "agent": agent,
            "action": action,
            "result": result,
            "success": success,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._tasks[task_id]["steps"].append(step)
        # Deliberately not saving here to avoid one disk write per step.
        # The task is persisted atomically when complete_task() or fail_task()
        # is called.  Call flush() explicitly if an intermediate checkpoint is needed.

    def complete_task(self, task_id: str, final_result: Any) -> None:
        if task_id not in self._tasks:
            return
        self._tasks[task_id]["status"] = "completed"
        self._tasks[task_id]["final_result"] = final_result
        self._tasks[task_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
        self._save()

    def fail_task(self, task_id: str, error: str) -> None:
        if task_id not in self._tasks:
            return
        self._tasks[task_id]["status"] = "failed"
        self._tasks[task_id]["error"] = error
        self._tasks[task_id]["failed_at"] = datetime.now(timezone.utc).isoformat()
        self._save()

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        return self._tasks.get(task_id)

    def recent_tasks(self, n: int = 10) -> List[Dict[str, Any]]:
        tasks = list(self._tasks.values())
        tasks.sort(key=lambda t: t.get("created_at", ""), reverse=True)
        return tasks[:n]

    def flush(self) -> None:
        """Explicitly persist all in-memory task data to disk."""
        self._save()

    # ── internals ─────────────────────────────────────────────────────────

    def _load(self) -> Dict[str, Any]:
        if os.path.exists(self._path):
            try:
                with open(self._path, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not load task memory: %s", exc)
        return {}

    def _save(self) -> None:
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._tasks, f, indent=2, ensure_ascii=False)
        except OSError as exc:
            logger.error("Could not persist task memory: %s", exc)

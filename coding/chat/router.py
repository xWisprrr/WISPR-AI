"""Code Engine — Chat router.

The ChatRouter is the single entry point for all Code Engine interactions.
It looks up (or creates) a session, resolves the active mode, dispatches
to the right specialist agent, persists the turn, and returns a unified
response dict.

Usage (async):
    router = ChatRouter()
    result = await router.chat(session_id="abc", message="Refactor my auth module")
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from llm.router import LLMRouter
from coding.session.manager import CodeSessionManager, VALID_MODES
from coding.tools.audit_log import AuditLog
from coding.agents.ask import AskAgent
from coding.agents.architect import ArchitectAgent
from coding.agents.code import CodeAgent
from coding.agents.debug import DebugAgent
from coding.agents.orchestrator import OrchestratorCodeAgent

logger = logging.getLogger(__name__)


class ChatRouter:
    """Dispatch chat messages to the correct Code Engine agent mode."""

    def __init__(
        self,
        llm_router: Optional[LLMRouter] = None,
        session_manager: Optional[CodeSessionManager] = None,
        audit: Optional[AuditLog] = None,
    ) -> None:
        self.llm = llm_router or LLMRouter()
        self.sessions = session_manager or CodeSessionManager()
        self.audit = audit or AuditLog()

    # ── primary API ───────────────────────────────────────────────────────

    async def chat(
        self,
        message: str,
        *,
        session_id: Optional[str] = None,
        mode: Optional[str] = None,
        workspace: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        auto_test: bool = False,
        test_command: Optional[str] = None,
        parallel: bool = True,
    ) -> Dict[str, Any]:
        """Process one chat turn and return the agent response.

        Args:
            message:      The user's message.
            session_id:   Existing session ID to resume, or None to create a new one.
            mode:         Override the session's current mode for this turn.
            workspace:    Override the session's workspace root.
            context:      Extra key-value context passed to the agent.
            auto_test:    (Code mode) run tests after writing and retry on failure.
            test_command: Shell command for auto_test.
            parallel:     (Orchestrator mode) run independent tasks in parallel.
        """
        # Resolve or create session
        if session_id:
            session = self.sessions.get_or_create(session_id)
        else:
            session = self.sessions.create()

        # Apply overrides
        if mode and mode != session.mode:
            session.set_mode(mode)
        if workspace:
            session.workspace = workspace

        active_mode = session.mode
        active_workspace = session.workspace

        # Build context for agent
        agent_context: Dict[str, Any] = dict(context or {})
        if active_workspace:
            agent_context["workspace"] = active_workspace

        # Dispatch
        result = await self._dispatch(
            message,
            mode=active_mode,
            history=session.as_messages(),
            context=agent_context,
            workspace=active_workspace,
            session_id=session.session_id,
            auto_test=auto_test,
            test_command=test_command,
            parallel=parallel,
        )

        # Persist turn
        session.add_message("user", message)
        session.add_message("assistant", result.get("response", result.get("error", "")))
        self.sessions.save(session)

        return {
            **result,
            "session_id": session.session_id,
            "mode": active_mode,
            "workspace": active_workspace,
        }

    async def stream_chat(
        self,
        message: str,
        *,
        session_id: Optional[str] = None,
        mode: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream the response token-by-token via the LLM stream API.

        Yields raw text chunks. Session is persisted after the last chunk.
        """
        from llm.router import TaskType

        if session_id:
            session = self.sessions.get_or_create(session_id)
        else:
            session = self.sessions.create()

        if mode and mode != session.mode:
            session.set_mode(mode)
        if workspace:
            session.workspace = workspace

        messages = session.as_messages() + [{"role": "user", "content": message}]
        full_response: List[str] = []

        async for chunk in self.llm.complete_stream(messages, task_type=TaskType.CODING):
            full_response.append(chunk)
            yield chunk

        response_text = "".join(full_response)
        session.add_message("user", message)
        session.add_message("assistant", response_text)
        self.sessions.save(session)

    # ── dispatch ──────────────────────────────────────────────────────────

    async def _dispatch(
        self,
        message: str,
        *,
        mode: str,
        history: List[Dict[str, str]],
        context: Dict[str, Any],
        workspace: Optional[str],
        session_id: str,
        auto_test: bool,
        test_command: Optional[str],
        parallel: bool,
    ) -> Dict[str, Any]:
        if mode == "ask":
            agent = AskAgent(llm_router=self.llm)
            return await agent.run(message, history=history, context=context)

        elif mode == "architect":
            agent = ArchitectAgent(llm_router=self.llm)
            return await agent.run(message, history=history, context=context, workspace=workspace)

        elif mode == "code":
            agent = CodeAgent(
                llm_router=self.llm,
                audit=self.audit,
                workspace=workspace,
                session_id=session_id,
            )
            return await agent.run(
                message,
                history=history,
                context=context,
                auto_test=auto_test,
                test_command=test_command,
            )

        elif mode == "debug":
            agent = DebugAgent(
                llm_router=self.llm,
                audit=self.audit,
                workspace=workspace,
                session_id=session_id,
            )
            return await agent.run(message, history=history, context=context)

        elif mode == "orchestrator":
            agent = OrchestratorCodeAgent(
                llm_router=self.llm,
                audit=self.audit,
                workspace=workspace,
                session_id=session_id,
            )
            return await agent.run(message, history=history, context=context, parallel=parallel)

        else:
            return {
                "success": False,
                "mode": mode,
                "error": f"Unknown mode '{mode}'. Valid modes: {sorted(VALID_MODES)}",
                "files_changed": [],
            }

"""FastAPI route definitions for the WISPR AI OS web dashboard."""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agents.orchestrator import OrchestratorAgent
from agents.react_agent import ReActAgent
from coding.engine import CodingEngine, CodeEngine, SUPPORTED_LANGUAGES
from coding.session.manager import VALID_MODES
from coding.deployment import SUPPORTED_PROVIDERS
from hallucination.reducer import HallucinationReducer
from memory.manager import MemoryManager
from plugins.plugin_manager import PluginManager
from reasoning.engine import ReasoningEngine
from search.mega_search import MegaSearch
from studio.ide import StudioIDE
from config import get_settings
from llm.router import TaskType

logger = logging.getLogger(__name__)
settings = get_settings()

# Shared Code Engine singleton — thread-safe lazy initialisation
_code_engine: Optional[CodeEngine] = None
_code_engine_lock = threading.Lock()


def _get_code_engine() -> CodeEngine:
    global _code_engine
    if _code_engine is None:
        with _code_engine_lock:
            if _code_engine is None:
                _code_engine = CodeEngine()
    return _code_engine


# ── Dependency injection helpers ──────────────────────────────────────────────

def _get_orchestrator(request: Request) -> OrchestratorAgent:
    return request.app.state.orchestrator


def _get_memory(request: Request) -> MemoryManager:
    return request.app.state.memory


def _get_plugins(request: Request) -> PluginManager:
    return request.app.state.plugins


# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., description="The user's question or task")
    context: Optional[Dict[str, Any]] = Field(default=None)
    use_reasoning: bool = Field(default=False)
    use_react: bool = Field(default=False, description="Run the ReAct iterative agent")
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for multi-turn conversation threading.",
    )


class QueryResponse(BaseModel):
    task_id: Optional[str] = None
    answer: str
    plan: Optional[List[Dict]] = None
    agent_results: Optional[List[Dict]] = None
    confidence: Optional[float] = None
    session_id: Optional[str] = None
    usage: Optional[Dict[str, int]] = None


class SearchRequest(BaseModel):
    query: str
    max_results: int = Field(default=10, ge=1, le=settings.search_max_sources)


class CodeRequest(BaseModel):
    task: str
    language: str = "python"
    code: Optional[str] = None
    action: str = Field(default="generate", description="generate | debug | optimise | translate")
    target_language: Optional[str] = None
    error: Optional[str] = None


class ExecuteRequest(BaseModel):
    code: str
    language: str = "python"
    timeout: int = Field(default=30, ge=1, le=120)


class MemoryStoreRequest(BaseModel):
    key: str
    value: Any
    tags: Optional[List[str]] = None


class PluginInvokeRequest(BaseModel):
    plugin_name: str
    task: str
    context: Optional[Dict[str, Any]] = None


class DeployRequest(BaseModel):
    project_path: str
    target: str = Field(..., description="github | vercel | netlify")
    config: Optional[Dict[str, Any]] = None


# ── Router ────────────────────────────────────────────────────────────────────

def create_router() -> APIRouter:
    router = APIRouter()

    # ── / ─────────────────────────────────────────────────────────────────
    @router.get("/api/status", tags=["system"])
    async def system_status(request: Request) -> Dict[str, Any]:
        """WISPR AI OS — system health and info."""
        plugins: PluginManager = request.app.state.plugins
        return {
            "status": "online",
            "system": settings.app_name,
            "version": settings.app_version,
            "agents": ["CoreAgent", "CoderAgent", "SearchAgent", "StudioAgent", "ReActAgent", "OrchestratorAgent"],
            "plugins": [p["name"] for p in plugins.list_plugins()],
            "supported_languages": SUPPORTED_LANGUAGES,
        }

    # ── /health ───────────────────────────────────────────────────────────
    @router.get("/health", tags=["system"])
    async def health_check(request: Request) -> Dict[str, Any]:
        """Detailed health check — reports the status of every subsystem."""
        memory: MemoryManager = request.app.state.memory
        plugins: PluginManager = request.app.state.plugins
        orchestrator: OrchestratorAgent = request.app.state.orchestrator

        checks: Dict[str, Any] = {}

        # Memory subsystems
        try:
            _ = memory.short_term.as_messages()
            _ = memory.long_term.all_entries()
            _ = memory.tasks.recent_tasks(n=1)
            checks["memory"] = {"status": "ok", "sessions": memory.sessions.active_count()}
        except Exception as exc:
            checks["memory"] = {"status": "error", "detail": str(exc)}

        # Plugin system
        try:
            plugin_list = plugins.list_plugins()
            checks["plugins"] = {"status": "ok", "loaded": len(plugin_list)}
        except Exception as exc:
            checks["plugins"] = {"status": "error", "detail": str(exc)}

        # LLM router (key configuration only — no real LLM call)
        llm_keys = {
            "openai": bool(settings.openai_api_key),
            "anthropic": bool(settings.anthropic_api_key),
            "gemini": bool(settings.gemini_api_key),
            "groq": bool(settings.groq_api_key),
        }
        checks["llm"] = {
            "status": "ok",
            "configured_providers": [k for k, v in llm_keys.items() if v],
            "routing": {
                "reasoning": settings.reasoning_model,
                "coding": settings.coding_model,
                "search": settings.search_model,
                "general": settings.general_model,
                "react": settings.reasoning_model,
                "fallback": settings.fallback_model,
            },
            "token_usage": orchestrator.llm.get_total_usage(),
        }

        # Agents
        checks["agents"] = {
            "status": "ok",
            "available": list(orchestrator._agents.keys()),
        }

        overall = "ok" if all(v.get("status") == "ok" for v in checks.values()) else "degraded"
        return {"status": overall, "timestamp": time.time(), "checks": checks}

    # ── /query ────────────────────────────────────────────────────────────
    @router.post("/query", response_model=QueryResponse, tags=["query"])
    async def query(
        body: QueryRequest,
        orchestrator: OrchestratorAgent = Depends(_get_orchestrator),
        memory: MemoryManager = Depends(_get_memory),
    ) -> QueryResponse:
        """Main AI interface — routes the task to the best agent(s)."""
        # ── Session history injection ──────────────────────────────────────
        ctx = dict(body.context or {})
        session_history: List[Dict[str, str]] = []
        if body.session_id:
            session = memory.sessions.get_or_create(body.session_id)
            session_history = session.as_messages()
            if session_history:
                ctx.setdefault("session_history", session_history)

        # ── ReAct agent ───────────────────────────────────────────────────
        if body.use_react:
            agent = ReActAgent(
                max_iterations=settings.react_max_iterations,
                llm_router=orchestrator.llm,
                memory=memory,
            )
            result = await agent.run(body.query, ctx)
            answer = result.output or (result.error or "ReAct agent returned no answer.")
            if body.session_id:
                session = memory.sessions.get_or_create(body.session_id)
                session.add("user", body.query)
                session.add("assistant", answer)
            return QueryResponse(
                answer=answer,
                session_id=body.session_id,
                usage=orchestrator.llm.get_total_usage(),
            )

        # ── Reasoning engine ──────────────────────────────────────────────
        if body.use_reasoning:
            engine = ReasoningEngine(llm_router=orchestrator.llm, memory=memory)
            result = await engine.reason(body.query, context=ctx)
            answer = result["final_answer"]
            if body.session_id:
                session = memory.sessions.get_or_create(body.session_id)
                session.add("user", body.query)
                session.add("assistant", answer)
            return QueryResponse(
                answer=answer,
                confidence=result["confidence"],
                session_id=body.session_id,
                usage=orchestrator.llm.get_total_usage(),
            )

        # ── Orchestrator ──────────────────────────────────────────────────
        result = await orchestrator.run(body.query, context=ctx)
        answer = result["final_answer"]
        if body.session_id:
            session = memory.sessions.get_or_create(body.session_id)
            session.add("user", body.query)
            session.add("assistant", answer)
        return QueryResponse(
            task_id=result["task_id"],
            answer=answer,
            plan=result["plan"],
            agent_results=result["agent_results"],
            session_id=body.session_id,
            usage=orchestrator.llm.get_total_usage(),
        )

    # ── /query/stream ─────────────────────────────────────────────────────
    @router.post("/query/stream", tags=["query"])
    async def query_stream(
        body: QueryRequest,
        orchestrator: OrchestratorAgent = Depends(_get_orchestrator),
        memory: MemoryManager = Depends(_get_memory),
    ) -> StreamingResponse:
        """Stream the AI response token-by-token via Server-Sent Events.

        The response is a text/event-stream. Each event carries a JSON payload:
        ``{"type": "token", "data": "<text chunk>"}``
        A final ``{"type": "done"}`` event signals completion.
        """
        ctx = dict(body.context or {})
        if body.session_id:
            session = memory.sessions.get_or_create(body.session_id)
            history = session.as_messages()
            if history:
                ctx.setdefault("session_history", history)

        async def event_generator() -> AsyncIterator[str]:
            # Build messages with session history
            messages = []
            if ctx.get("session_history"):
                messages.extend(ctx["session_history"])
            messages.append({"role": "user", "content": body.query})

            full_response: List[str] = []
            try:
                async for chunk in orchestrator.llm.complete_stream(
                    messages, task_type=TaskType.GENERAL
                ):
                    full_response.append(chunk)
                    payload = json.dumps({"type": "token", "data": chunk})
                    yield f"data: {payload}\n\n"
            except Exception as exc:
                error_payload = json.dumps({"type": "error", "data": str(exc)})
                yield f"data: {error_payload}\n\n"
                return

            # Persist to session memory
            if body.session_id:
                session = memory.sessions.get_or_create(body.session_id)
                session.add("user", body.query)
                session.add("assistant", "".join(full_response))

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # ── /agents ───────────────────────────────────────────────────────────
    @router.get("/agents", tags=["agents"])
    async def list_agents() -> Dict[str, Any]:
        """List all available agents with their descriptions."""
        return {
            "agents": [
                {
                    "name": "CoreAgent",
                    "type": "reasoning",
                    "description": "General intelligence, conversation, and reasoning tasks.",
                },
                {
                    "name": "CoderAgent",
                    "type": "coding",
                    "description": "Code generation, debugging, optimisation. Multi-language.",
                },
                {
                    "name": "SearchAgent",
                    "type": "search",
                    "description": "Queries 50+ search engines and synthesises results.",
                },
                {
                    "name": "StudioAgent",
                    "type": "studio",
                    "description": "Builds and deploys full applications (GitHub, Vercel, Netlify).",
                },
                {
                    "name": "ReActAgent",
                    "type": "react",
                    "description": (
                        "Autonomous Reasoning + Acting loop. Interleaves Thought, Action (search/"
                        "python/memory tools), and Observation until a Final Answer is reached."
                    ),
                },
                {
                    "name": "OrchestratorAgent",
                    "type": "orchestrator",
                    "description": "Coordinates all agents in parallel for complex tasks.",
                },
            ]
        }

    # ── /search ───────────────────────────────────────────────────────────
    @router.post("/search", tags=["search"])
    async def mega_search(body: SearchRequest) -> Dict[str, Any]:
        """Query the MegaSearch engine across 50+ sources."""
        engine = MegaSearch()
        results = await engine.search(body.query, max_results=body.max_results)
        return {
            "query": body.query,
            "num_results": len(results),
            "results": results,
        }

    # ── /code ─────────────────────────────────────────────────────────────
    @router.post("/code", tags=["code"])
    async def code_interface(
        body: CodeRequest,
        orchestrator: OrchestratorAgent = Depends(_get_orchestrator),
    ) -> Dict[str, Any]:
        """Multi-language coding interface: generate, debug, optimise, or translate."""
        engine = CodingEngine(llm_router=orchestrator.llm)

        if body.action == "generate":
            return await engine.generate(body.task, language=body.language)
        elif body.action == "debug":
            if not body.code or not body.error:
                raise HTTPException(400, "'code' and 'error' are required for debug action.")
            return await engine.debug(body.code, body.error, language=body.language)
        elif body.action == "optimise":
            if not body.code:
                raise HTTPException(400, "'code' is required for optimise action.")
            return await engine.optimise(body.code, language=body.language)
        elif body.action == "translate":
            if not body.code or not body.target_language:
                raise HTTPException(400, "'code' and 'target_language' are required for translate.")
            return await engine.translate(body.code, body.target_language, body.language)
        else:
            raise HTTPException(400, f"Unknown action: {body.action}. Use generate|debug|optimise|translate.")

    # ── /studio/execute ───────────────────────────────────────────────────
    @router.post("/studio/execute", tags=["studio"])
    async def studio_execute(body: ExecuteRequest) -> Dict[str, Any]:
        """Execute code in the sandboxed WISPR Studio IDE."""
        ide = StudioIDE()
        return await ide.execute(body.code, language=body.language, timeout=body.timeout)

    # ── /studio/deploy ────────────────────────────────────────────────────
    @router.post("/studio/deploy", tags=["studio"])
    async def studio_deploy(body: DeployRequest) -> Dict[str, Any]:
        """Generate deployment instructions for GitHub, Vercel, or Netlify."""
        ide = StudioIDE()
        result = await ide.deploy(body.project_path, body.target, body.config)
        if not result.get("success"):
            raise HTTPException(400, result.get("error", "Deploy failed"))
        return result

    # ── /memory ───────────────────────────────────────────────────────────
    @router.get("/memory", tags=["memory"])
    async def memory_overview(memory: MemoryManager = Depends(_get_memory)) -> Dict[str, Any]:
        """Overview of current memory state."""
        return {
            "short_term_entries": len(memory.short_term),
            "long_term_entries": len(memory.long_term.all_entries()),
            "recent_tasks": memory.tasks.recent_tasks(n=5),
            "active_sessions": memory.sessions.active_count(),
        }

    @router.post("/memory/store", tags=["memory"])
    async def memory_store(
        body: MemoryStoreRequest,
        memory: MemoryManager = Depends(_get_memory),
    ) -> Dict[str, str]:
        """Persist a key-value pair in long-term memory."""
        memory.long_term.store(body.key, body.value, tags=body.tags)
        return {"status": "stored", "key": body.key}

    @router.get("/memory/retrieve/{key}", tags=["memory"])
    async def memory_retrieve(
        key: str,
        memory: MemoryManager = Depends(_get_memory),
    ) -> Dict[str, Any]:
        """Retrieve a value from long-term memory."""
        value = memory.long_term.retrieve(key)
        if value is None:
            raise HTTPException(404, f"Key '{key}' not found in long-term memory.")
        return {"key": key, "value": value}

    @router.get("/memory/search", tags=["memory"])
    async def memory_search(
        q: str,
        top_k: int = 5,
        memory: MemoryManager = Depends(_get_memory),
    ) -> Dict[str, Any]:
        """Search long-term memory by keyword."""
        if not q.strip():
            raise HTTPException(400, "Search query 'q' must not be empty.")
        results = memory.long_term.search(q, top_k=top_k)
        return {"query": q, "num_results": len(results), "results": results}

    @router.get("/memory/short_term", tags=["memory"])
    async def memory_short_term(
        n: int = 10,
        memory: MemoryManager = Depends(_get_memory),
    ) -> Dict[str, Any]:
        """Return the most recent short-term (session) memory entries."""
        entries = memory.short_term.get_recent(n=n)
        return {
            "total_entries": len(memory.short_term),
            "returned": len(entries),
            "messages": [{"role": e.role, "content": e.content, "timestamp": e.timestamp} for e in entries],
        }

    @router.delete("/memory/short_term", tags=["memory"])
    async def memory_clear_short_term(
        memory: MemoryManager = Depends(_get_memory),
    ) -> Dict[str, str]:
        """Clear all short-term (session) memory entries."""
        memory.short_term.clear()
        return {"status": "cleared"}

    @router.delete("/memory/{key}", tags=["memory"])
    async def memory_delete(
        key: str,
        memory: MemoryManager = Depends(_get_memory),
    ) -> Dict[str, Any]:
        """Delete a key from long-term memory."""
        deleted = memory.long_term.delete(key)
        if not deleted:
            raise HTTPException(404, f"Key '{key}' not found in long-term memory.")
        return {"status": "deleted", "key": key}
    # ── /sessions ─────────────────────────────────────────────────────────
    @router.delete("/sessions/{session_id}", tags=["memory"])
    async def clear_session(
        session_id: str,
        memory: MemoryManager = Depends(_get_memory),
    ) -> Dict[str, str]:
        """Clear the conversation history for a session."""
        memory.sessions.clear(session_id)
        return {"status": "cleared", "session_id": session_id}

    # ── /plugins ──────────────────────────────────────────────────────────
    @router.get("/plugins", tags=["plugins"])
    async def list_plugins(plugins: PluginManager = Depends(_get_plugins)) -> Dict[str, Any]:
        """List all loaded plugins."""
        return {"plugins": plugins.list_plugins()}

    @router.post("/plugins/invoke", tags=["plugins"])
    async def invoke_plugin(
        body: PluginInvokeRequest,
        plugins: PluginManager = Depends(_get_plugins),
    ) -> Dict[str, Any]:
        """Invoke a plugin by name."""
        try:
            result = await plugins.invoke(body.plugin_name, body.task, body.context)
            return {"plugin": body.plugin_name, "result": result}
        except KeyError as exc:
            raise HTTPException(404, str(exc))
        except Exception as exc:
            raise HTTPException(500, f"Plugin execution failed: {exc}")

    # ── /reason ───────────────────────────────────────────────────────────
    @router.post("/reason", tags=["reasoning"])
    async def reason(
        body: QueryRequest,
        orchestrator: OrchestratorAgent = Depends(_get_orchestrator),
        memory: MemoryManager = Depends(_get_memory),
    ) -> Dict[str, Any]:
        """Run the autonomous multi-step reasoning engine on a task."""
        engine = ReasoningEngine(llm_router=orchestrator.llm, memory=memory)
        return await engine.reason(body.query, context=body.context)

    # ── /hallucination/check ──────────────────────────────────────────────
    @router.post("/hallucination/check", tags=["hallucination"])
    async def hallucination_check(
        body: QueryRequest,
        orchestrator: OrchestratorAgent = Depends(_get_orchestrator),
    ) -> Dict[str, Any]:
        """Run multi-vote hallucination reduction on a query."""
        reducer = HallucinationReducer(llm_router=orchestrator.llm)
        return await reducer.vote(body.query)

    # ═══════════════════════════════════════════════════════════════════════
    # Code Engine routes  /code-engine/*
    # ═══════════════════════════════════════════════════════════════════════

    # ── Request/Response models ───────────────────────────────────────────

    class CodeEngineChatRequest(BaseModel):
        message: str = Field(..., description="User message to the Code Engine")
        session_id: Optional[str] = Field(default=None, description="Session ID to resume")
        mode: Optional[str] = Field(
            default=None,
            description=f"Agent mode: {sorted(VALID_MODES)}. Overrides session mode for this turn.",
        )
        workspace: Optional[str] = Field(default=None, description="Absolute path to project root")
        auto_test: bool = Field(default=False, description="Run tests after code changes (Code mode)")
        test_command: Optional[str] = Field(default=None, description="Shell command to run tests")
        parallel: bool = Field(default=True, description="Run orchestrator tasks in parallel")

    class CodeEngineModeRequest(BaseModel):
        session_id: str
        mode: str = Field(..., description=f"New mode: {sorted(VALID_MODES)}")

    class CodeEngineWorkspaceRequest(BaseModel):
        session_id: str
        workspace: str = Field(..., description="Absolute path to workspace root")

    class CodeEngineDeployRequest(BaseModel):
        provider: str = Field(..., description=f"Deployment provider: {SUPPORTED_PROVIDERS}")
        project_path: str = Field(..., description="Absolute path to the project directory")
        project_description: Optional[str] = None
        config: Optional[Dict[str, Any]] = None

    class CodeEngineFileReadRequest(BaseModel):
        path: str
        session_id: Optional[str] = None

    class CodeEngineFileWriteRequest(BaseModel):
        path: str
        content: str
        session_id: Optional[str] = None

    class CodeEngineFileEditRequest(BaseModel):
        path: str
        old_str: str
        new_str: str
        session_id: Optional[str] = None

    class CodeEngineSearchRequest(BaseModel):
        directory: str
        pattern: str
        glob_pattern: str = "**/*"
        max_matches: int = Field(default=100, ge=1, le=1000)
        use_regex: bool = False
        session_id: Optional[str] = None

    # ── Chat ─────────────────────────────────────────────────────────────

    @router.post("/code-engine/chat", tags=["code-engine"])
    async def code_engine_chat(body: CodeEngineChatRequest) -> Dict[str, Any]:
        """Send a message to the Code Engine and receive an agentic response.

        Supports all 5 modes: ask, architect, code, debug, orchestrator.
        Unlimited prompting — no rate limits.
        """
        engine = _get_code_engine()
        try:
            return await engine.chat(
                body.message,
                session_id=body.session_id,
                mode=body.mode,
                workspace=body.workspace,
                auto_test=body.auto_test,
                test_command=body.test_command,
                parallel=body.parallel,
            )
        except Exception as exc:
            raise HTTPException(500, f"Code Engine error: {exc}")

    @router.post("/code-engine/chat/stream", tags=["code-engine"])
    async def code_engine_chat_stream(body: CodeEngineChatRequest) -> StreamingResponse:
        """Stream a Code Engine response token-by-token via Server-Sent Events."""
        engine = _get_code_engine()

        async def event_generator() -> AsyncIterator[str]:
            try:
                async for chunk in engine.stream_chat(
                    body.message,
                    session_id=body.session_id,
                    mode=body.mode,
                    workspace=body.workspace,
                ):
                    payload = json.dumps({"type": "token", "data": chunk})
                    yield f"data: {payload}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'type': 'error', 'data': str(exc)})}\n\n"
                return
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ── Sessions ──────────────────────────────────────────────────────────

    @router.get("/code-engine/sessions", tags=["code-engine"])
    async def code_engine_list_sessions() -> Dict[str, Any]:
        """List all Code Engine sessions."""
        engine = _get_code_engine()
        return {"sessions": engine.sessions.list_all()}

    @router.post("/code-engine/sessions", tags=["code-engine"])
    async def code_engine_create_session(
        mode: str = "ask",
        workspace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new Code Engine session."""
        if mode not in VALID_MODES:
            raise HTTPException(400, f"Invalid mode '{mode}'. Valid: {sorted(VALID_MODES)}")
        engine = _get_code_engine()
        session = engine.sessions.create(mode=mode, workspace=workspace)
        return session.to_dict()

    @router.get("/code-engine/sessions/{session_id}", tags=["code-engine"])
    async def code_engine_get_session(session_id: str) -> Dict[str, Any]:
        """Get a Code Engine session by ID."""
        engine = _get_code_engine()
        session = engine.sessions.get(session_id)
        if not session:
            raise HTTPException(404, f"Session '{session_id}' not found.")
        return session.to_dict()

    @router.delete("/code-engine/sessions/{session_id}", tags=["code-engine"])
    async def code_engine_delete_session(session_id: str) -> Dict[str, str]:
        """Delete a Code Engine session."""
        engine = _get_code_engine()
        if not engine.sessions.delete(session_id):
            raise HTTPException(404, f"Session '{session_id}' not found.")
        return {"status": "deleted", "session_id": session_id}

    @router.post("/code-engine/sessions/{session_id}/mode", tags=["code-engine"])
    async def code_engine_set_mode(session_id: str, body: CodeEngineModeRequest) -> Dict[str, Any]:
        """Switch the active agent mode for an existing session."""
        engine = _get_code_engine()
        session = engine.sessions.get(session_id)
        if not session:
            raise HTTPException(404, f"Session '{session_id}' not found.")
        try:
            session.set_mode(body.mode)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        engine.sessions.save(session)
        return {"session_id": session_id, "mode": session.mode}

    @router.post("/code-engine/sessions/{session_id}/workspace", tags=["code-engine"])
    async def code_engine_set_workspace(
        session_id: str, body: CodeEngineWorkspaceRequest
    ) -> Dict[str, Any]:
        """Set the workspace root for an existing session."""
        engine = _get_code_engine()
        session = engine.sessions.get(session_id)
        if not session:
            raise HTTPException(404, f"Session '{session_id}' not found.")
        session.workspace = body.workspace
        engine.sessions.save(session)
        return {"session_id": session_id, "workspace": session.workspace}

    # ── File tools ────────────────────────────────────────────────────────

    @router.post("/code-engine/files/read", tags=["code-engine"])
    async def code_engine_read_file(body: CodeEngineFileReadRequest) -> Dict[str, Any]:
        """Read a local file. Unrestricted filesystem access with audit logging."""
        from coding.tools.file_tools import FileTools
        from coding.tools.audit_log import AuditLog
        ft = FileTools(audit=_get_code_engine().audit, session_id=body.session_id)
        return await ft.read_file(body.path)

    @router.post("/code-engine/files/write", tags=["code-engine"])
    async def code_engine_write_file(body: CodeEngineFileWriteRequest) -> Dict[str, Any]:
        """Write (create or overwrite) a local file."""
        from coding.tools.file_tools import FileTools
        ft = FileTools(audit=_get_code_engine().audit, session_id=body.session_id)
        return await ft.write_file(body.path, body.content)

    @router.post("/code-engine/files/edit", tags=["code-engine"])
    async def code_engine_edit_file(body: CodeEngineFileEditRequest) -> Dict[str, Any]:
        """Apply a targeted find-and-replace edit to a local file."""
        from coding.tools.file_tools import FileTools
        ft = FileTools(audit=_get_code_engine().audit, session_id=body.session_id)
        return await ft.edit_file(body.path, body.old_str, body.new_str)

    @router.delete("/code-engine/files", tags=["code-engine"])
    async def code_engine_delete_file(path: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Delete a local file."""
        from coding.tools.file_tools import FileTools
        ft = FileTools(audit=_get_code_engine().audit, session_id=session_id)
        return await ft.delete_file(path)

    @router.get("/code-engine/files/list", tags=["code-engine"])
    async def code_engine_list_dir(path: str = ".", session_id: Optional[str] = None) -> Dict[str, Any]:
        """List the contents of a directory."""
        from coding.tools.file_tools import FileTools
        ft = FileTools(audit=_get_code_engine().audit, session_id=session_id)
        return await ft.list_dir(path)

    @router.post("/code-engine/files/search", tags=["code-engine"])
    async def code_engine_search_files(body: CodeEngineSearchRequest) -> Dict[str, Any]:
        """Search for a pattern in files under a directory."""
        from coding.tools.file_tools import FileTools
        ft = FileTools(audit=_get_code_engine().audit, session_id=body.session_id)
        return await ft.search_files(
            body.directory,
            body.pattern,
            glob_pattern=body.glob_pattern,
            max_matches=body.max_matches,
            use_regex=body.use_regex,
        )

    @router.get("/code-engine/audit", tags=["code-engine"])
    async def code_engine_audit_log(n: int = 50) -> Dict[str, Any]:
        """Return the most recent file operation audit records."""
        engine = _get_code_engine()
        records = engine.audit.recent(n=n)
        return {"count": len(records), "records": records}

    # ── Deployment ────────────────────────────────────────────────────────

    @router.post("/code-engine/deploy", tags=["code-engine"])
    async def code_engine_deploy(body: CodeEngineDeployRequest) -> Dict[str, Any]:
        """Generate deployment configuration files and instructions.

        Supported providers: github, vercel, netlify, railway, docker.
        """
        engine = _get_code_engine()
        result = await engine.deploy(
            body.provider,
            body.project_path,
            project_description=body.project_description,
            config=body.config,
        )
        if not result["success"]:
            raise HTTPException(400, result.get("error", "Deployment generation failed."))
        return result

    @router.get("/code-engine/deploy/providers", tags=["code-engine"])
    async def code_engine_deploy_providers() -> Dict[str, Any]:
        """List all supported deployment providers."""
        return {"providers": SUPPORTED_PROVIDERS}

    # ── Xencode ───────────────────────────────────────────────────────────

    class XencodeCompileRequestModel(BaseModel):
        prompt: str = Field(..., description="Project description or README content")
        project_name: str = Field(..., description="Project name (used as workspace dirname)")
        language: str = Field(default="python", description="Target language")
        workspace: Optional[str] = Field(default=None, description="Override workspace path")
        build_command: Optional[str] = Field(default=None, description="Override build command")

    class GitHubPublishRequestModel(BaseModel):
        workspace: str = Field(..., description="Absolute path to workspace to publish")
        repo_name: str = Field(..., description="GitHub repository name to create")
        private: bool = Field(default=False, description="Make repo private")
        description: str = Field(default="", description="Repository description")

    @router.post("/code-engine/xencode/compile", tags=["xencode"])
    async def xencode_compile(body: XencodeCompileRequestModel) -> Dict[str, Any]:
        """Run the full Xencode pipeline: parse spec → plan → write → validate → build → ZIP."""
        from coding.xencode.engine import XencodeEngine
        from coding.xencode.schemas import XencodeCompileRequest

        engine = XencodeEngine(llm_router=_get_code_engine().llm)
        request = XencodeCompileRequest(
            prompt=body.prompt,
            project_name=body.project_name,
            language=body.language,
            workspace=body.workspace,
            build_command=body.build_command,
        )
        result = await engine.compile(request)
        if not result.success and result.error:
            logger.error("Xencode compile failed for %s", body.project_name)
            raise HTTPException(500, "Xencode compile failed. Check server logs for details.")
        return result.model_dump()

    @router.post("/code-engine/xencode/compile/stream", tags=["xencode"])
    async def xencode_compile_stream(body: XencodeCompileRequestModel) -> StreamingResponse:
        """Stream the Xencode pipeline via Server-Sent Events.

        Events: spec_parsed, plan, files_written, validation,
                final_build_started, final_build_output, done, error
        """
        from coding.xencode.engine import XencodeEngine
        from coding.xencode.schemas import XencodeCompileRequest

        engine = XencodeEngine(llm_router=_get_code_engine().llm)
        request = XencodeCompileRequest(
            prompt=body.prompt,
            project_name=body.project_name,
            language=body.language,
            workspace=body.workspace,
            build_command=body.build_command,
        )

        async def event_generator() -> AsyncIterator[str]:
            try:
                async for event in engine.stream_compile(request):
                    payload = json.dumps(event)
                    yield f"data: {payload}\n\n"
            except Exception:
                logger.exception("Xencode stream_compile generator error")
                yield f"data: {json.dumps({'event': 'error', 'data': {'error': 'Internal stream error'}})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.post("/code-engine/github/create-and-publish", tags=["xencode"])
    async def github_create_and_publish(body: GitHubPublishRequestModel) -> Dict[str, Any]:
        """Create a GitHub repository and push the workspace using the gh CLI.

        Requires the ``gh`` CLI to be installed and authenticated (``gh auth login``).
        """
        import re as _re
        import subprocess
        import urllib.parse
        from pathlib import Path

        # Validate repo_name — GitHub only allows alphanumerics, hyphens, and underscores
        if not _re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", body.repo_name):
            raise HTTPException(
                400,
                "repo_name may only contain letters, digits, hyphens, underscores, "
                "and dots (max 100 characters).",
            )

        # Validate and resolve workspace path to prevent path traversal
        try:
            from coding.tools.file_tools import _safe_resolve
            workspace = _safe_resolve(body.workspace)
        except PermissionError as exc:
            raise HTTPException(403, str(exc))
        except Exception:
            raise HTTPException(400, "Invalid workspace path.")

        if not workspace.exists() or not workspace.is_dir():
            raise HTTPException(400, f"Workspace does not exist: {body.workspace}")

        # Check gh CLI is available
        try:
            auth_check = subprocess.run(
                ["gh", "auth", "status"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if auth_check.returncode != 0:
                return {
                    "success": False,
                    "error": (
                        "gh CLI is not authenticated. Run 'gh auth login' first."
                    ),
                }
        except FileNotFoundError:
            return {
                "success": False,
                "error": (
                    "gh CLI not found. Install it from https://cli.github.com/ "
                    "and run 'gh auth login'."
                ),
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "gh auth check timed out."}

        # Initialise git repo if not already one
        git_dir = workspace / ".git"
        if not git_dir.exists():
            subprocess.run(["git", "init"], cwd=str(workspace), capture_output=True)
            subprocess.run(
                ["git", "add", "-A"], cwd=str(workspace), capture_output=True
            )
            subprocess.run(
                ["git", "commit", "-m", "Initial commit via WISPR Xencode"],
                cwd=str(workspace),
                capture_output=True,
            )

        # Build gh repo create command (list form — no shell injection risk)
        # repo_name has already been validated against a strict alphanumeric regex above
        safe_repo_name = str(body.repo_name)  # validated: [A-Za-z0-9_.-]{1,100}
        cmd = [
            "gh", "repo", "create", safe_repo_name,
            "--source", str(workspace),
            "--push",
            "--description", body.description or "Created by WISPR Xencode",
        ]
        if body.private:
            cmd.append("--private")
        else:
            cmd.append("--public")

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(workspace),
            )
            if proc.returncode == 0:
                output = proc.stdout.strip() or proc.stderr.strip()
                # Extract repo URL: validate that netloc is exactly github.com or a subdomain
                repo_url: Optional[str] = None
                for line in output.splitlines():
                    parsed = urllib.parse.urlparse(line.strip())
                    netloc = parsed.netloc.lower()
                    if parsed.scheme in ("https", "http") and (
                        netloc == "github.com" or netloc.endswith(".github.com")
                    ):
                        repo_url = line.strip()
                        break
                return {
                    "success": True,
                    "repo_url": repo_url,
                    "message": output,
                }
            return {
                "success": False,
                "error": "Repository creation failed. Check gh CLI output.",
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "gh repo create timed out after 60s."}
        except Exception as exc:
            logger.error("GitHub publish failed: %s", exc)
            raise HTTPException(500, "GitHub publish failed. Check server logs.")

    # ── Modes info ────────────────────────────────────────────────────────

    @router.get("/code-engine/modes", tags=["code-engine"])
    async def code_engine_modes() -> Dict[str, Any]:
        """Describe the 5 Code Engine agent modes."""
        return {
            "modes": [
                {
                    "name": "ask",
                    "title": "Ask Mode",
                    "description": (
                        "Intelligent Q&A companion. Provides expert technical answers, "
                        "explanations, and guidance. Does NOT modify any files."
                    ),
                    "can_write_files": False,
                },
                {
                    "name": "architect",
                    "title": "Architect Mode",
                    "description": (
                        "Analyses the codebase, designs robust system architectures, "
                        "and produces detailed step-by-step implementation plans."
                    ),
                    "can_write_files": False,
                },
                {
                    "name": "code",
                    "title": "Code Mode",
                    "description": (
                        "Primary coding partner. Transforms natural language into "
                        "production-ready code, creating and editing local files. "
                        "Supports automatic failure recovery via test integration."
                    ),
                    "can_write_files": True,
                },
                {
                    "name": "debug",
                    "title": "Debug Mode",
                    "description": (
                        "Specialist troubleshooting expert. Systematically diagnoses "
                        "issues, identifies root causes, and applies targeted fixes."
                    ),
                    "can_write_files": True,
                },
                {
                    "name": "orchestrator",
                    "title": "Orchestrator Mode",
                    "description": (
                        "Breaks complex projects into tasks, delegates to specialist "
                        "agents, coordinates results. Supports parallel execution."
                    ),
                    "can_write_files": True,
                },
            ],
            "note": "Unlimited prompting — no rate limits.",
        }

    return router


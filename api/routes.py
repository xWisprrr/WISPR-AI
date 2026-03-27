"""FastAPI route definitions for the WISPR AI OS web dashboard."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from agents.orchestrator import OrchestratorAgent
from coding.engine import CodingEngine, SUPPORTED_LANGUAGES
from hallucination.reducer import HallucinationReducer
from memory.manager import MemoryManager
from plugins.plugin_manager import PluginManager
from reasoning.engine import ReasoningEngine
from search.mega_search import MegaSearch
from studio.ide import StudioIDE
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


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


class QueryResponse(BaseModel):
    task_id: Optional[str] = None
    answer: str
    plan: Optional[List[Dict]] = None
    agent_results: Optional[List[Dict]] = None
    confidence: Optional[float] = None


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
            "agents": ["CoreAgent", "CoderAgent", "SearchAgent", "StudioAgent", "OrchestratorAgent"],
            "plugins": [p["name"] for p in plugins.list_plugins()],
            "supported_languages": SUPPORTED_LANGUAGES,
        }

    # ── /query ────────────────────────────────────────────────────────────
    @router.post("/query", response_model=QueryResponse, tags=["query"])
    async def query(
        body: QueryRequest,
        orchestrator: OrchestratorAgent = Depends(_get_orchestrator),
        memory: MemoryManager = Depends(_get_memory),
    ) -> QueryResponse:
        """Main AI interface — routes the task to the best agent(s)."""
        if body.use_reasoning:
            engine = ReasoningEngine(llm_router=orchestrator.llm, memory=memory)
            result = await engine.reason(body.query, context=body.context)
            return QueryResponse(
                answer=result["final_answer"],
                confidence=result["confidence"],
            )

        result = await orchestrator.run(body.query, context=body.context)
        return QueryResponse(
            task_id=result["task_id"],
            answer=result["final_answer"],
            plan=result["plan"],
            agent_results=result["agent_results"],
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
    async def studio_deploy(
        project_path: str,
        target: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate deployment instructions for GitHub, Vercel, or Netlify."""
        ide = StudioIDE()
        result = await ide.deploy(project_path, target, config)
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

    return router

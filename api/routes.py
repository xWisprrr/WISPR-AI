"""
FastAPI routes — /query, /search, /code, /agents, /reason, /studio, /plugins
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agents.coder_agent import CoderAgent
from agents.core_agent import CoreAgent
from agents.orchestrator import OrchestratorAgent
from agents.search_agent import SearchAgent
from agents.studio_agent import StudioAgent
from hallucination.reducer import HallucinationReducer
from memory.store import LongTermMemory, ShortTermMemory
from plugins.manager import PluginManager
from reasoning.engine import ReasoningEngine
from search.mega_search import MegaSearch

router = APIRouter()

# ── Shared singletons ─────────────────────────────────────────────────────────
_short_mem = ShortTermMemory()
_long_mem = LongTermMemory()
_orchestrator = OrchestratorAgent()
_core = CoreAgent(memory=_short_mem)
_coder = CoderAgent()
_search_agent = SearchAgent()
_studio = StudioAgent()
_mega = MegaSearch()
_reasoning = ReasoningEngine()
_reducer = HallucinationReducer()
_plugins = PluginManager()
_plugins.discover()


# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    prompt: str
    mode: str = "auto"       # auto | reason | search | code | studio


class QueryResponse(BaseModel):
    answer: str
    agent: str
    confidence: Optional[float] = None
    metadata: dict[str, Any] = {}


class SearchRequest(BaseModel):
    query: str
    max_results: int = 10


class CodeRequest(BaseModel):
    task: str
    language: Optional[str] = None
    action: str = "generate"  # generate | debug | optimize | translate | explain


class StudioRequest(BaseModel):
    description: str
    project_name: str = "wispr-project"
    deploy_target: str = "github"


class ReasonRequest(BaseModel):
    task: str
    context: Optional[str] = None


class MemoryStoreRequest(BaseModel):
    key: str
    value: Any
    tags: list[str] = []


# ── Route handlers ────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    """Main AI interface — routes to the appropriate agent(s)."""
    if req.mode == "search":
        result = await _search_agent._timed_run(req.prompt)
    elif req.mode == "code":
        result = await _coder._timed_run(req.prompt)
    elif req.mode == "studio":
        result = await _studio._timed_run(req.prompt)
    elif req.mode == "reason":
        reasoning_result = await _reasoning.reason(req.prompt)
        return QueryResponse(
            answer=reasoning_result["answer"],
            agent="reasoning",
            confidence=reasoning_result["confidence"],
            metadata=reasoning_result,
        )
    else:
        # auto — use orchestrator
        result = await _orchestrator._timed_run(req.prompt)

    return QueryResponse(
        answer=str(result.output or ""),
        agent=result.agent_name,
        confidence=result.metadata.get("confidence"),
        metadata=result.metadata,
    )


@router.post("/search")
async def search(req: SearchRequest) -> dict[str, Any]:
    """MegaSearch endpoint — queries 50+ engines in parallel."""
    results = await _mega.search(req.query, max_results=req.max_results)
    return {
        "query": req.query,
        "result_count": len(results),
        "engine_count": _mega.engine_count,
        "results": results,
    }


@router.post("/code")
async def code(req: CodeRequest) -> dict[str, Any]:
    """Coding interface — generate, debug, optimize, or translate code."""
    result = await _coder._timed_run(
        req.task,
        context={"language": req.language, "action": req.action},
    )
    if not result.success:
        raise HTTPException(status_code=500, detail=result.error)
    return {
        "task": req.task,
        "language": result.metadata.get("language"),
        "action": result.metadata.get("action"),
        "output": result.output,
        "code_blocks": result.metadata.get("code_blocks", []),
        "elapsed": result.elapsed,
    }


@router.post("/studio")
async def studio(req: StudioRequest) -> dict[str, Any]:
    """WISPR AI Studio — generate and deploy a full project."""
    result = await _studio._timed_run(
        req.description,
        context={
            "project_name": req.project_name,
            "deploy_target": req.deploy_target,
        },
    )
    if not result.success:
        raise HTTPException(status_code=500, detail=result.error)
    return {
        "project_name": req.project_name,
        "output": result.output,
        "files_created": result.metadata.get("files_created", []),
        "run_result": result.metadata.get("run_result"),
        "deploy_instructions": result.metadata.get("deploy_instructions"),
        "elapsed": result.elapsed,
    }


@router.post("/reason")
async def reason(req: ReasonRequest) -> dict[str, Any]:
    """Autonomous reasoning endpoint — multi-step ReAct loop."""
    result = await _reasoning.reason(req.task, context=req.context)
    return result


@router.get("/agents")
async def list_agents() -> dict[str, Any]:
    """List all available agents and their status."""
    built_in = [
        {"name": "core", "description": CoreAgent.description, "status": "active"},
        {"name": "coder", "description": CoderAgent.description, "status": "active"},
        {"name": "search", "description": SearchAgent.description, "status": "active"},
        {"name": "studio", "description": StudioAgent.description, "status": "active"},
        {"name": "orchestrator", "description": OrchestratorAgent.description, "status": "active"},
    ]
    plugins = _plugins.list_plugins()
    return {
        "built_in_agents": built_in,
        "plugin_agents": plugins,
        "total": len(built_in) + len(plugins),
    }


@router.get("/memory")
async def get_memory() -> dict[str, Any]:
    """Retrieve short-term conversation memory."""
    return {
        "short_term": _short_mem.get_history(limit=20),
        "long_term_keys": _long_mem.all_keys(),
    }


@router.post("/memory")
async def store_memory_entry(req: MemoryStoreRequest) -> dict[str, Any]:
    """Store a value in long-term memory."""
    _long_mem.store(req.key, req.value, tags=req.tags)
    return {"status": "stored", "key": req.key}


@router.get("/plugins")
async def list_plugins() -> dict[str, Any]:
    """List all loaded plugins."""
    return {
        "plugins": _plugins.list_plugins(),
        "count": _plugins.plugin_count,
    }

"""
WISPR AI OS — FastAPI entry point.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.routes import router
from config import settings


@asynccontextmanager
async def lifespan(application: FastAPI):
    logger.info(f"🚀 {settings.app_name} v{settings.version} starting up")
    yield
    logger.info(f"🛑 {settings.app_name} shutting down")


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description=(
        "WISPR AI OS — a self-orchestrating AI system with multi-agent parallel execution, "
        "multi-LLM routing, MegaSearch (50+ engines), autonomous reasoning, "
        "browser IDE, and modular plugin support."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
async def root() -> dict:
    """System status and capability overview."""
    return {
        "name": settings.app_name,
        "version": settings.version,
        "status": "online",
        "capabilities": [
            "Multi-LLM intelligence routing (LiteLLM)",
            "Multi-agent parallel execution",
            "MegaSearch — 50+ search engines",
            "Autonomous reasoning (ReAct loop)",
            "Multi-language coding engine",
            "WISPR AI Studio (browser IDE + deployment)",
            "Memory system (short-term + long-term)",
            "Hallucination reduction",
            "Modular plugin system",
        ],
        "endpoints": {
            "query": "POST /query — main AI interface",
            "search": "POST /search — MegaSearch",
            "code": "POST /code — coding interface",
            "studio": "POST /studio — AI Studio",
            "reason": "POST /reason — reasoning engine",
            "agents": "GET /agents — agent status",
            "memory": "GET|POST /memory — memory system",
            "plugins": "GET /plugins — plugin list",
            "docs": "GET /docs — interactive API documentation",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=settings.debug)


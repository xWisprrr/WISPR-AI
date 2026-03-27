"""WISPR AI OS — FastAPI application entry point."""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agents.orchestrator import OrchestratorAgent
from api.routes import create_router
from config import get_settings
from memory.manager import MemoryManager
from plugins.plugin_manager import PluginManager

settings = get_settings()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # ── Startup ───────────────────────────────────────────────────────────
    plugins: PluginManager = app.state.plugins
    logger.info("🚀 %s v%s starting up…", settings.app_name, settings.app_version)
    logger.info("   Agents: CoreAgent, CoderAgent, SearchAgent, StudioAgent, OrchestratorAgent")
    logger.info("   Plugins loaded: %d", len(plugins.list_plugins()))

    yield  # application is running

    # ── Shutdown ──────────────────────────────────────────────────────────
    logger.info("🛑 %s shutting down.", settings.app_name)


def create_app() -> FastAPI:
    # Initialise shared singletons before passing to lifespan
    memory = MemoryManager()
    orchestrator = OrchestratorAgent(memory=memory)
    plugins = PluginManager()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "WISPR AI OS — a self-orchestrating multi-agent AI system that can "
            "think, code, search, build, and learn."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── Shared state (singletons) ─────────────────────────────────────────
    app.state.memory = memory
    app.state.orchestrator = orchestrator
    app.state.plugins = plugins

    # ── CORS ──────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routes ────────────────────────────────────────────────────────────
    app.include_router(create_router())

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
    )

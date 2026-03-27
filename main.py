"""WISPR AI OS — FastAPI application entry point + CLI."""

from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from agents.orchestrator import OrchestratorAgent
from api.routes import create_router
from config import get_settings
from memory.manager import MemoryManager
from plugins.plugin_manager import PluginManager

_STATIC_DIR = Path(__file__).parent / "static"

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
    logger.info("   Agents: CoreAgent, CoderAgent, SearchAgent, StudioAgent, ReActAgent, OrchestratorAgent")
    logger.info("   Code Engine: Ask · Architect · Code · Debug · Orchestrator (unlimited prompting)")
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
            "think, code, search, build, and learn. "
            "Includes a full Code Engine with 5 agentic modes, real-time file "
            "operations, seamless deployment integrations, and unlimited prompting."
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

    # ── Static files (UI) ─────────────────────────────────────────────────
    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    # ── UI root ───────────────────────────────────────────────────────────
    @app.get("/", include_in_schema=False)
    async def serve_ui() -> FileResponse:
        return FileResponse(str(_STATIC_DIR / "index.html"))

    return app


app = create_app()


# ══════════════════════════════════════════════════════════════════════════════
# Code Engine CLI
# ══════════════════════════════════════════════════════════════════════════════

def _run_async(coro):
    """Run an async coroutine from a synchronous CLI context."""
    return asyncio.run(coro)


def _print_banner():
    print(
        "\n┌─────────────────────────────────────────────┐\n"
        "│  WISPR AI — Code Engine  (unlimited prompts) │\n"
        "└─────────────────────────────────────────────┘"
    )


def _code_engine_chat_cli(
    session_id: Optional[str] = None,
    mode: str = "ask",
    workspace: Optional[str] = None,
):
    """Interactive CLI chat loop for the Code Engine."""
    from coding.engine import CodeEngine
    from coding.session.manager import VALID_MODES

    try:
        from rich.console import Console
        from rich.markdown import Markdown
        console = Console()
        use_rich = True
    except ImportError:
        console = None
        use_rich = False

    _print_banner()

    engine = CodeEngine()

    if session_id:
        session = engine.sessions.get(session_id)
        if session:
            print(f"▶ Resuming session {session.session_id} | mode={session.mode} | workspace={session.workspace}")
            mode = session.mode
            workspace = session.workspace
        else:
            print(f"Session '{session_id}' not found — starting a new one.")
            session_id = None

    if not session_id:
        session = engine.sessions.create(mode=mode, workspace=workspace)
        session_id = session.session_id
        print(f"▶ New session: {session_id} | mode={mode} | workspace={workspace or '(none)'}")

    print(f"\nCommands: /mode <{'|'.join(sorted(VALID_MODES))}> | /workspace <path> | /sessions | /quit\n")

    while True:
        try:
            prompt = input(f"[{mode}]> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not prompt:
            continue

        # Built-in CLI commands
        if prompt.startswith("/mode "):
            new_mode = prompt[6:].strip()
            if new_mode not in VALID_MODES:
                print(f"  ✗ Invalid mode. Choose from: {sorted(VALID_MODES)}")
                continue
            session = engine.sessions.get(session_id)
            session.set_mode(new_mode)
            engine.sessions.save(session)
            mode = new_mode
            print(f"  ✓ Switched to {mode} mode.")
            continue

        if prompt.startswith("/workspace "):
            new_ws = prompt[11:].strip()
            session = engine.sessions.get(session_id)
            session.workspace = new_ws
            engine.sessions.save(session)
            workspace = new_ws
            print(f"  ✓ Workspace set to: {workspace}")
            continue

        if prompt == "/sessions":
            for s in engine.sessions.list_all():
                print(f"  {s['session_id']}  mode={s['mode']}  turns={s['turns']}  updated={s['updated_at']}")
            continue

        if prompt in ("/quit", "/exit", "/q"):
            print("Goodbye.")
            break

        # Send to Code Engine
        try:
            result = _run_async(engine.chat(
                prompt,
                session_id=session_id,
                mode=mode,
                workspace=workspace,
            ))
        except Exception as exc:
            print(f"  ✗ Error: {exc}")
            continue

        response = result.get("response") or result.get("error", "(no response)")
        files = result.get("files_changed", [])

        if use_rich:
            console.print(Markdown(response))
        else:
            print(f"\n{response}\n")

        if files:
            print(f"  📁 Files changed: {', '.join(files)}")


def _list_sessions_cli():
    """Print all Code Engine sessions."""
    from coding.engine import CodeEngine
    engine = CodeEngine()
    sessions = engine.sessions.list_all()
    if not sessions:
        print("No sessions found.")
        return
    print(f"{'Session ID':<40} {'Mode':<12} {'Turns':<6} {'Updated'}")
    print("-" * 80)
    for s in sessions:
        print(f"{s['session_id']:<40} {s['mode']:<12} {s['turns']:<6} {s['updated_at']}")


def _deploy_cli(provider: str, project_path: str, description: Optional[str] = None):
    """Generate deployment files for a project."""
    from coding.engine import CodeEngine
    engine = CodeEngine()
    print(f"Generating {provider} deployment for: {project_path}")
    result = _run_async(engine.deploy(provider, project_path, project_description=description))
    if result["success"]:
        print(f"✓ {result['message']}")
        if result.get("files_generated"):
            for f in result["files_generated"]:
                print(f"  → {f}")
        print("\n" + result.get("instructions", ""))
    else:
        print(f"✗ {result.get('error', 'Unknown error')}")


def _print_cli_help():
    print("""
WISPR AI — Code Engine CLI

Usage:
  python main.py serve                          Start the API server
  python main.py code-engine chat               Start interactive chat (new session, ask mode)
  python main.py code-engine chat --mode code   Start in Code mode
  python main.py code-engine chat --session <id> Resume a session
  python main.py code-engine chat --workspace /path/to/project
  python main.py code-engine sessions           List all sessions
  python main.py code-engine deploy <provider> <project_path>
                                                Generate deployment files
                                                Providers: github vercel netlify railway docker
  python main.py code-engine help               Show this help

Agent Modes:
  ask          Q&A, explanations — read-only
  architect    Codebase analysis + architecture plans
  code         Write & edit files (production-ready code)
  debug        Diagnose bugs + apply targeted fixes
  orchestrator Break complex tasks into sub-tasks, run in parallel

Chat commands (inside interactive session):
  /mode <name>         Switch agent mode
  /workspace <path>    Set project root
  /sessions            List all sessions
  /quit                Exit

API server (default port 8000):
  POST /code-engine/chat          Chat with Code Engine
  GET  /code-engine/sessions      List sessions
  POST /code-engine/deploy        Generate deployment files
  GET  /code-engine/modes         Describe all modes
  GET  /docs                      Interactive API docs
""")


if __name__ == "__main__":
    args = sys.argv[1:]

    if not args or args[0] == "serve":
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=settings.debug,
        )

    elif args[0] == "code-engine":
        sub = args[1] if len(args) > 1 else "help"

        if sub == "chat":
            # Parse flags: --mode, --session, --workspace
            mode = "ask"
            session_id = None
            workspace = None
            i = 2
            while i < len(args):
                if args[i] == "--mode" and i + 1 < len(args):
                    mode = args[i + 1]
                    i += 2
                elif args[i] == "--session" and i + 1 < len(args):
                    session_id = args[i + 1]
                    i += 2
                elif args[i] == "--workspace" and i + 1 < len(args):
                    workspace = args[i + 1]
                    i += 2
                else:
                    i += 1
            _code_engine_chat_cli(session_id=session_id, mode=mode, workspace=workspace)

        elif sub == "sessions":
            _list_sessions_cli()

        elif sub == "deploy":
            if len(args) < 4:
                print("Usage: python main.py code-engine deploy <provider> <project_path> [description]")
                sys.exit(1)
            provider = args[2]
            project_path = args[3]
            description = args[4] if len(args) > 4 else None
            _deploy_cli(provider, project_path, description)

        else:
            _print_cli_help()

    else:
        _print_cli_help()

# WISPR AI OS

> **A self-orchestrating AI operating system** that blends ideas from Devin AI, LangChain, and OpenDevin—pushed beyond them into a modular, multi-agent, multi-LLM platform.

---

## 🧠 System Overview

WISPR AI OS is a full AI operating system that can:

| Capability | Description |
|---|---|
| **Think** | Multi-step reasoning loops with self-reflection |
| **Act** | ReAct agent: Reason → Act (search/code/memory tools) → Observe, iteratively |
| **Code** | Multi-language code generation, debugging, optimisation |
| **Search** | 50+ search engines queried in parallel |
| **Build** | Browser IDE + one-click deployment |
| **Learn** | Short-term, long-term, task, and per-session memory |
| **Stream** | Real-time token streaming via Server-Sent Events |

---

## 🏗️ Architecture

```
WISPR-AI/
├── main.py                  # FastAPI application entry point
├── config.py                # Centralised settings (pydantic-settings)
├── requirements.txt
│
├── agents/                  # Multi-Agent System
│   ├── base_agent.py        # Abstract base class
│   ├── core_agent.py        # General intelligence & reasoning
│   ├── coder_agent.py       # Code generation, debug, optimisation
│   ├── search_agent.py      # MegaSearch + LLM synthesis
│   ├── studio_agent.py      # Full-app builder & deployment
│   ├── react_agent.py       # ReAct: Reason-Act-Observe iterative loop
│   └── orchestrator.py      # Parallel agent coordination
│
├── llm/                     # Multi-LLM Intelligence Layer
│   └── router.py            # LiteLLM routing, streaming, structured outputs,
│                            #   token usage tracking
│
├── search/                  # MegaSearch Engine
│   └── mega_search.py       # 50+ sources queried concurrently
│
├── memory/                  # Memory System
│   ├── short_term.py        # In-process ring buffer
│   ├── long_term.py         # JSON-persisted key-value store
│   ├── task_memory.py       # Multi-step task history
│   ├── session.py           # Per-session conversation threading
│   └── manager.py           # Unified memory interface
│
├── reasoning/               # Autonomous Reasoning Engine
│   └── engine.py            # Decompose -> Execute -> Reflect -> Improve
│
├── hallucination/           # Hallucination Reduction System
│   └── reducer.py           # Voting, cross-checking, search verification
│
├── studio/                  # WISPR AI Studio (Browser IDE)
│   └── ide.py               # Code execution + GitHub/Vercel/Netlify deploy
│
├── coding/                  # Multi-Language Coding Engine
│   └── engine.py            # Generate, debug, optimise, translate
│
├── plugins/                 # Modular Plugin System
│   └── plugin_manager.py    # Auto-discovery & invocation
│
├── api/                     # FastAPI Web Dashboard
│   └── routes.py            # All REST endpoints
│
└── tests/
    └── test_wispr.py        # Comprehensive test suite (67 tests)
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Copy `.env.example` to `.env` and set your API keys:

```bash
cp .env.example .env
```

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
GROQ_API_KEY=...

# Override default model routing (optional)
REASONING_MODEL=gpt-4o
CODING_MODEL=gpt-4o
SEARCH_MODEL=gpt-4o-mini
REACT_MAX_ITERATIONS=8
```

### 3. Run the server

```bash
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | System status & info |
| `GET` | `/health` | Detailed subsystem health check |
| `POST` | `/query` | Main AI interface — orchestrator, reasoning, or ReAct |
| `POST` | `/query/stream` | Streaming response via Server-Sent Events |
| `GET` | `/agents` | List all agents |
| `POST` | `/search` | MegaSearch across 50+ sources |
| `POST` | `/code` | Code: generate / debug / optimise / translate |
| `POST` | `/studio/execute` | Run code in sandboxed IDE |
| `POST` | `/studio/deploy` | Deploy to GitHub / Vercel / Netlify |
| `GET` | `/memory` | Memory overview (includes active session count) |
| `POST` | `/memory/store` | Store to long-term memory |
| `GET` | `/memory/retrieve/{key}` | Retrieve from long-term memory |
| `DELETE` | `/sessions/{session_id}` | Clear a conversation session |
| `GET` | `/plugins` | List loaded plugins |
| `POST` | `/plugins/invoke` | Invoke a plugin |
| `POST` | `/reason` | Autonomous multi-step reasoning |
| `POST` | `/hallucination/check` | Multi-vote hallucination reduction |

---

## Agents

### OrchestratorAgent
Receives complex tasks, decomposes them into sub-tasks, runs all sub-agents **in parallel**, and synthesises results into a single coherent answer.

### ReActAgent *(new — state of the art)*
Implements the **ReAct (Reasoning + Acting)** pattern. The agent iterates through:

```
Thought:  <step-by-step reasoning>
Action:   tool_name(argument)
Observation: <tool result — filled in automatically>
...repeat...
Final Answer: <complete response>
```

Available tools: **`search`** (MegaSearch), **`python`** (sandboxed code execution), **`memory`** (long-term memory lookup).

Invoke via `/query` with `"use_react": true`, or route the task to `"ReActAgent"` in the orchestrator plan.

### CoreAgent
General-purpose reasoning and conversation. Uses session memory for context continuity.

### CoderAgent
Writes production-quality code in any language. Extracts and returns fenced code blocks.

### SearchAgent
Queries MegaSearch and uses the LLM to synthesise results into a factual answer with citations.

### StudioAgent
Designs full-stack applications, writes all files, and provides deployment instructions.

---

## Multi-LLM Routing

The `LLMRouter` (powered by [LiteLLM](https://github.com/BerriAI/litellm)) automatically selects the best model per task type and tracks token usage:

| Task | Default Model |
|---|---|
| Reasoning | `gpt-4o` |
| Coding | `gpt-4o` |
| Search | `gpt-4o-mini` |
| General | `gpt-4o-mini` |
| ReAct | `gpt-4o` |
| Fallback | `gpt-4o-mini` |

New in this release:
- **`complete_with_usage()`** — returns `(text, LLMUsage)` with prompt/completion/total token counts
- **`structured_complete()`** — forces JSON-mode output validated against a Pydantic schema
- **`get_total_usage()`** — exposes cumulative token usage per router instance (visible at `/health`)
- **Streaming** — `complete_stream()` yields text chunks for real-time SSE delivery

Switch to any LiteLLM-supported provider by changing the model names in `.env`.

---

## Streaming (Server-Sent Events)

Use `POST /query/stream` to receive responses token-by-token:

```python
import httpx, json

with httpx.stream("POST", "http://localhost:8000/query/stream",
                  json={"query": "Explain quantum entanglement"}) as r:
    for line in r.iter_lines():
        if line.startswith("data: "):
            event = json.loads(line[6:])
            if event["type"] == "token":
                print(event["data"], end="", flush=True)
            elif event["type"] == "done":
                break
```

---

## Session-Based Conversation Threading

Pass a `session_id` to `/query` or `/query/stream` to maintain conversation history across turns:

```json
{
  "query": "What is the capital of France?",
  "session_id": "user-42-chat"
}
```

Follow-up requests with the same `session_id` automatically receive the previous conversation as context. Sessions expire after one hour of inactivity. Clear a session with `DELETE /sessions/{session_id}`.

---

## MegaSearch Engine

Queries all sources **concurrently** and deduplicates results:

- SearXNG meta-search (Google, Bing, DuckDuckGo, Brave, StartPage, Wikipedia, Wikidata)
- DuckDuckGo HTML
- Wikipedia OpenSearch
- ArXiv (academic papers)
- GitHub Repositories
- Hacker News
- Stack Overflow
- Reddit

---

## Memory System

| Layer | Storage | Purpose |
|---|---|---|
| Short-term | In-process ring buffer | Session context for LLM |
| Long-term | JSON file | Persistent knowledge across restarts |
| Task memory | JSON file | Step-by-step task history |
| Session memory | In-process dict | Per-session conversation threading |

---

## Hallucination Reduction

1. **Majority voting** — samples the LLM N times, finds consensus answer
2. **Search verification** — checks claims against live search results
3. **Cross-checking** — compares multiple agent outputs for consistency
4. **Confidence scoring** — explicit 0.0-1.0 confidence assigned to each answer

---

## Plugin System

Drop any `.py` file in the `plugins/` directory that exports a `register()` function:

```python
# plugins/my_plugin.py
from plugins.plugin_manager import Plugin

def register() -> Plugin:
    async def handler(task: str, context: dict) -> str:
        return f"My plugin handled: {task}"

    return Plugin(
        name="MyPlugin",
        version="1.0.0",
        description="Does something cool.",
        handler=handler,
    )
```

The plugin is auto-discovered on startup and callable via `/plugins/invoke`.

---

## Supported Languages

Python, JavaScript, TypeScript, Go, Rust, Java, C, C++, Ruby, Bash, SQL, HTML, CSS

---

## Running Tests

```bash
python -m pytest tests/ -v
```

---

---

## 🤖 Code Engine

> **The agentic engineering core of WISPR AI OS** — specialized for advanced coding with real-time file operations, seamless deployment integrations, and unlimited prompting.

### Overview

The Code Engine lives under `coding/` and provides five intelligent agent modes, persistent sessions, a real-time filesystem toolkit, and one-click deployment to five cloud providers.

| Feature | Detail |
|---|---|
| **5 Agent Modes** | Ask · Architect · Code · Debug · Orchestrator |
| **Unlimited Prompting** | No rate limits of any kind |
| **Real-time File Ops** | Read · Write · Create · Edit · Delete · Search · Move |
| **Session Persistence** | JSONL-backed sessions survive restarts; resume anywhere |
| **Parallel Agents** | Orchestrator delegates tasks and runs them in parallel |
| **Auto Failure Recovery** | Code mode runs tests, feeds failures to Debug, retries up to N times |
| **Deployment** | GitHub · Vercel · Netlify · Railway · Docker |
| **Audit Log** | Every file operation is logged (who/when/session/op/path/bytes) |

---

### Agent Modes

| Mode | Description | Can Write Files |
|---|---|---|
| **ask** | Q&A companion — expert answers, explanations, guidance | ❌ No |
| **architect** | Analyses the codebase; produces architectures + step-by-step plans | ❌ No |
| **code** | Transforms natural language into production-ready code, edits local files | ✅ Yes |
| **debug** | Diagnoses bugs, identifies root causes, applies targeted fixes | ✅ Yes |
| **orchestrator** | Decomposes complex tasks, delegates to specialist agents, runs in parallel | ✅ Yes |

---

### Running the API Server

```bash
# Start the WISPR AI API server (Code Engine routes included)
python main.py serve
# or simply
python main.py

# API is available at http://localhost:8000
# Interactive docs: http://localhost:8000/docs
```

---

### CLI Chat — Quick Start

```bash
# Start a new chat session in Ask mode (default)
python main.py code-engine chat

# Start in Code mode with a workspace
python main.py code-engine chat --mode code --workspace /path/to/project

# Resume an existing session
python main.py code-engine chat --session <session-id>

# List all sessions
python main.py code-engine sessions

# Deploy a project
python main.py code-engine deploy vercel /path/to/project
python main.py code-engine deploy docker /path/to/project
python main.py code-engine deploy github /path/to/project

# Show all CLI help
python main.py code-engine help
```

**In-session commands:**

```
/mode ask|architect|code|debug|orchestrator   Switch agent mode
/workspace /path/to/project                   Set the workspace root
/sessions                                     List all sessions
/quit                                         Exit
```

---

### API Endpoints

All Code Engine endpoints are under `/code-engine/` and visible at `/docs`.

| Method | Path | Description |
|---|---|---|
| `POST` | `/code-engine/chat` | Send a message; get an agentic response |
| `POST` | `/code-engine/chat/stream` | Stream response tokens via SSE |
| `GET` | `/code-engine/sessions` | List all sessions |
| `POST` | `/code-engine/sessions` | Create a new session |
| `GET` | `/code-engine/sessions/{id}` | Get session details |
| `DELETE` | `/code-engine/sessions/{id}` | Delete a session |
| `POST` | `/code-engine/sessions/{id}/mode` | Switch mode |
| `POST` | `/code-engine/sessions/{id}/workspace` | Set workspace |
| `POST` | `/code-engine/files/read` | Read a local file |
| `POST` | `/code-engine/files/write` | Write / create a local file |
| `POST` | `/code-engine/files/edit` | Targeted find-and-replace edit |
| `DELETE` | `/code-engine/files` | Delete a file |
| `GET` | `/code-engine/files/list` | List directory contents |
| `POST` | `/code-engine/files/search` | Search file contents |
| `GET` | `/code-engine/audit` | View file operation audit log |
| `POST` | `/code-engine/deploy` | Generate deployment config + instructions |
| `GET` | `/code-engine/deploy/providers` | List supported providers |
| `GET` | `/code-engine/modes` | Describe all 5 modes |

**Example — Chat request:**

```bash
curl -X POST http://localhost:8000/code-engine/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Create a FastAPI hello-world endpoint",
    "mode": "code",
    "workspace": "/home/user/myproject"
  }'
```

**Example — Deploy to Vercel:**

```bash
curl -X POST http://localhost:8000/code-engine/deploy \
  -H "Content-Type: application/json" \
  -d '{"provider": "vercel", "project_path": "/home/user/myproject"}'
```

---

### ⚠️ Security: Unrestricted Filesystem Access

> **WARNING:** The Code Engine is configured for **unrestricted local filesystem access**.
> This means the Code and Debug agents can read and write any file on your machine
> that your user account has access to.

**What is protected by default:**

The following paths are always blocked, regardless of configuration:

- `~/.ssh` (SSH keys)
- `~/.gnupg` (GPG keys)
- `~/.aws` (AWS credentials)
- `~/.config/gcloud` (GCP credentials)
- `/etc`, `/usr`, `/bin`, `/sbin`, `/lib`, `/boot`, `/proc`, `/sys`, `/dev` (Unix system dirs)
- `/System`, `/Library`, `/private` (macOS system dirs)
- `C:\Windows`, `C:\Program Files` (Windows system dirs)

**Adding extra denied paths:**

```bash
# Unix / macOS
export WISPR_DENIED_PATHS="/home/user/secrets:/home/user/.config/sensitive"

# Windows
set WISPR_DENIED_PATHS=C:\Users\user\secrets;C:\sensitive
```

**Audit log:**

Every file operation is recorded in `coding/store/audit.jsonl`:

```jsonl
{"ts": "2026-03-27T22:00:00Z", "op": "create", "path": "/home/user/app/main.py", "session_id": "abc", "agent": "CodeAgent", "bytes": 512}
```

View recent operations:

```bash
curl http://localhost:8000/code-engine/audit
```

**Restricting to a workspace only:**

The safest way to limit the Code Engine is to set a workspace root and instruct users to always provide it when chatting. All relative paths are resolved against the workspace, so agents will naturally stay within it.

---

### Deployment Integrations

The Code Engine can generate complete deployment configurations for:

| Provider | Files Generated | CLI Command |
|---|---|---|
| **GitHub** | `.github/workflows/ci.yml` | `python main.py code-engine deploy github <path>` |
| **Vercel** | `vercel.json` | `python main.py code-engine deploy vercel <path>` |
| **Netlify** | `netlify.toml` | `python main.py code-engine deploy netlify <path>` |
| **Railway** | `railway.toml` + `Dockerfile` | `python main.py code-engine deploy railway <path>` |
| **Docker** | `Dockerfile` + `docker-compose.yml` | `python main.py code-engine deploy docker <path>` |

---

### Session Persistence

Sessions are stored in `coding/store/sessions.jsonl`. Each session records:
- Conversation history
- Active agent mode
- Workspace root
- Custom variables (project name, preferences, etc.)
- Agent state snapshots

Sessions survive server restarts and can be resumed by ID in both the CLI and API.

---

## 🖥️ Desktop App (WISPR Desktop Studio)

WISPR Desktop Studio is a native **Windows desktop application** built with [Tauri](https://tauri.app/) (Rust back-end) + React (front-end). It provides a full GUI for the WISPR AI backend — chat, Xencode project compilation, GitHub publishing — all without needing a browser.

### Prerequisites

| Requirement | Version |
|---|---|
| Node.js | 18+ |
| Rust toolchain | latest stable (`rustup update`) |
| Python + WISPR backend | running on `http://localhost:8000` |

### Running in development mode

```bash
cd desktop
npm install
npm run tauri dev
```

This starts the Vite dev server and the Tauri window simultaneously. Hot-reloading is enabled for both the React front-end and the Rust shell.

### Building a distributable

```bash
cd desktop
npm run tauri build
```

The installer (`WISPR-Desktop-Studio_x.y.z_x64-setup.exe`) is emitted to `desktop/src-tauri/target/release/bundle/`.

### Screenshot

> _Screenshot placeholder — replace with an actual screenshot once available._

---

## ⚙️ Xencode — Spec-Driven Project Compiler

Xencode turns a plain-English project description into a **complete, runnable project** — source files, build configs, README, tests, and a ready-to-ship ZIP — without writing a single line of code yourself.

### How it works

The pipeline runs through **6 sequential stages**, each emitted as a Server-Sent Event when using the streaming endpoint:

| Stage event | What happens |
|---|---|
| `spec_parsed` | Prompt is parsed (via LLM or heuristics) into a structured `XencodeSpec` |
| `plan` | A `XencodePlan` is generated: list of files with paths + content |
| `files_written` | All planned files are written to the workspace directory |
| `validation` | Static checks run: entry point exists, syntax errors, missing configs |
| `final_build_started` | The build command is executed (e.g. `pip install -r requirements.txt`) |
| `done` | Manifest persisted; ZIP deliverable created |

### API endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/code-engine/xencode/compile` | Run the full pipeline; returns `XencodeCompileResponse` |
| `POST` | `/code-engine/xencode/stream` | Same pipeline but streams stage events via SSE |

**Example — compile request:**

```bash
curl -X POST http://localhost:8000/code-engine/xencode/compile \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A FastAPI REST API with SQLite and pytest tests",
    "project_name": "my-api",
    "language": "python"
  }'
```

### Language selection

The default target language is **Python**. Xencode auto-detects the language from the prompt using keyword heuristics (no LLM required for detection). To override, set the `language` field:

| Value | Detected keywords |
|---|---|
| `python` | `python`, `flask`, `fastapi`, `django`, `pytest`, `.py` |
| `javascript` | `javascript`, `node`, `npm`, `react`, `vue`, `express` |
| `typescript` | `typescript`, `angular`, `tsx`, ` ts ` |
| `go` | `golang`, `goroutine`, `.go`, `go ` |
| `rust` | `rust`, `.rs`, `cargo`, `crate` |
| `cpp` | `c++`, `cpp`, `#include <iostream>` |
| `csharp` | `c#`, `csharp`, `.net`, `dotnet` |

### Output format

```
<workspace>/
├── main.py          # (or language equivalent)
├── requirements.txt
├── README.md
├── tests/
│   └── test_main.py
└── .xencode/
    └── manifest.json   ← build metadata
```

A `<project-name>.zip` is written to the artifacts directory (configurable via `XENCODE_ARTIFACTS_DIR`).

### Manifest format (`.xencode/manifest.json`)

```json
{
  "project_name": "my-api",
  "language": "python",
  "workspace": "/xencode/workspaces/my-api",
  "files_written": ["main.py", "requirements.txt", "README.md"],
  "build_command": "pip install -r requirements.txt",
  "build_success": true,
  "build_output": "...",
  "zip_path": "/xencode/artifacts/my-api.zip",
  "validation_errors": [],
  "repair_attempts": 0,
  "created_at": "2025-01-01T00:00:00+00:00"
}
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `XENCODE_WORKSPACE_ROOT` | `xencode/workspaces` | Root directory for generated projects |
| `XENCODE_ARTIFACTS_DIR` | `xencode/artifacts` | Where ZIP files are saved |
| `XENCODE_BUILD_TIMEOUT` | `120` | Build command timeout in seconds |
| `XENCODE_MAX_REPAIR_ATTEMPTS` | `2` | Max LLM-assisted build repair rounds |

---

## 🐙 GitHub Publishing

Publish any local workspace (including Xencode output) directly to a new GitHub repository in one API call.

### Endpoint

```
POST /code-engine/github/create-and-publish
```

**Request body:**

```json
{
  "workspace": "/absolute/path/to/project",
  "repo_name": "my-new-repo",
  "private": false,
  "description": "Created with WISPR AI Xencode"
}
```

**Response:**

```json
{
  "success": true,
  "repo_url": "https://github.com/<you>/my-new-repo",
  "message": "Repository created and pushed successfully."
}
```

### Prerequisites

- The [GitHub CLI (`gh`)](https://cli.github.com/) must be installed and authenticated (`gh auth login`).
- The workspace must be a valid directory; it will be `git init`-ed, committed, and pushed automatically.

### From WISPR Desktop Studio

After compiling a project with Xencode, click **"Publish to GitHub"** in the Desktop Studio sidebar. The app calls this endpoint automatically using the compiled workspace path.

---

## License

MIT

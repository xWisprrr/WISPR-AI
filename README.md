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

## License

MIT

# WISPR AI OS

**WISPR AI OS** is a self-orchestrating AI operating system that blends ideas from Devin AI, LangChain, and OpenDevin — and pushes beyond them into a modular, multi-agent, multi-LLM platform.

> An AI that can act like a developer, researcher, and assistant — at the same time.

---

## 🧩 Architecture

```
WISPR AI OS
├── 🧠  Multi-LLM Intelligence Layer  (llm/)
├── 🤖  Multi-Agent System            (agents/)
│   ├── Core Agent      — reasoning & conversation
│   ├── Coder Agent     — multi-language code generation
│   ├── Search Agent    — MegaSearch orchestration
│   ├── Studio Agent    — browser IDE + deployment
│   └── Orchestrator    — parallel execution & output synthesis
├── 🔄  Autonomous Reasoning Engine   (reasoning/)
├── 🌐  MegaSearch Engine             (search/)
├── 📂  Memory System                 (memory/)
├── 🧪  Hallucination Reduction       (hallucination/)
├── 💻  WISPR AI Studio               (studio/)
├── 🧰  Plugin System                 (plugins/)
└── ⚡  FastAPI Web Dashboard         (api/ + main.py)
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys (optional — the system works without LLM keys for local testing)

```bash
cp .env.example .env   # then edit with your keys
```

`.env` example:
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=...
GROQ_API_KEY=...
```

### 3. Run the server

```bash
uvicorn main:app --reload
```

Open <http://localhost:8000/docs> for the interactive API explorer.

---

## 🔌 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | System status & capability overview |
| `POST` | `/query` | Main AI interface — auto-routes to best agent(s) |
| `POST` | `/search` | MegaSearch — queries 50+ engines in parallel |
| `POST` | `/code` | Coding interface — generate / debug / optimise |
| `POST` | `/studio` | AI Studio — generate & deploy a full project |
| `POST` | `/reason` | Autonomous reasoning (ReAct loop) |
| `GET` | `/agents` | List all agents and their status |
| `GET/POST` | `/memory` | Short-term & long-term memory |
| `GET` | `/plugins` | List loaded plugins |
| `GET` | `/docs` | Interactive Swagger UI |

---

## 🧠 Components

### Multi-LLM Intelligence Layer (`llm/router.py`)

Routes tasks to the most suitable model via **LiteLLM**:

| Task Type | Default Model |
|-----------|---------------|
| Reasoning | `gpt-4o` |
| Coding | `gpt-4o` |
| Search summarisation | `gpt-4o-mini` |
| Fast / general | `gpt-4o-mini` |

Override via environment variables in `.env`.  Automatic fallback to the fast model on errors.

### Multi-Agent System (`agents/`)

All agents run in parallel via `asyncio.gather`:

- **CoreAgent** — conversational intelligence with short-term memory
- **CoderAgent** — multi-language code generation, debugging, optimisation, translation
- **SearchAgent** — MegaSearch + LLM-powered result synthesis
- **StudioAgent** — full-project generation, simulated execution, deployment instructions
- **OrchestratorAgent** — task decomposition → parallel dispatch → output synthesis → verification

### Autonomous Reasoning Engine (`reasoning/engine.py`)

ReAct-style loop (Reason → Act → Observe) with:
- Configurable maximum steps (`max_reasoning_steps`)
- Self-reflection pass on final answer
- Structured JSON step output

### MegaSearch Engine (`search/mega_search.py`)

Queries multiple engines **in parallel**:

| Engine | Type |
|--------|------|
| DuckDuckGo | General |
| Wikipedia | Encyclopaedic |
| Hacker News | Tech community |
| GitHub | Code repositories |
| arXiv | Academic papers |
| + 45 more (extensible) | Various |

Results are deduplicated by URL and ranked by relevance score.

### Memory System (`memory/store.py`)

| Store | Persistence | Capacity |
|-------|-------------|----------|
| `ShortTermMemory` | In-process (deque) | Configurable cap |
| `LongTermMemory` | JSON file on disk | Unlimited |
| `TaskMemory` | Per-task scratchpad | Task lifetime |

### Hallucination Reduction (`hallucination/reducer.py`)

1. **Confidence aggregation** — averages per-agent confidence scores
2. **Entity consistency check** — IoU of numeric tokens across agent outputs
3. **LLM self-verification** — optional second-pass critique and correction
4. **Blended final score** — weighted combination of the above

### WISPR AI Studio (`studio/ide.py`)

- Parses LLM output into a structured project file tree
- In-memory project workspace
- Simulated project execution (hooks into real WebContainer in production)
- One-command deployment instructions for **GitHub**, **Vercel**, and **Netlify**

### Plugin System (`plugins/manager.py`)

Drop a Python file into `plugins/` that exposes a `Plugin` class inheriting from `BaseAgent`:

```python
# plugins/my_agent.py
from agents.base_agent import BaseAgent, AgentResult

class Plugin(BaseAgent):
    name = "my_agent"
    description = "Does something custom"

    async def run(self, task, context=None):
        return AgentResult(agent_id=self.agent_id, agent_name=self.name,
                           success=True, output="custom result")
```

The `PluginManager` discovers and loads it automatically at startup.

### Multi-Language Coding Engine (`coding/engine.py`)

Detects and supports: Python, JavaScript, TypeScript, Rust, Go, Java, C++, C, Ruby, PHP, Swift, Kotlin, R, SQL, Bash, HTML, CSS, and more.

---

## 🧪 Testing

```bash
pytest tests/ -v
```

75 tests covering all core modules — memory, coding engine, hallucination reducer, LLM router, studio IDE, plugin manager, and all API endpoints.

---

## ⚙️ Configuration (`config.py`)

All settings are configurable via environment variables or `.env`:

```
REASONING_MODEL=gpt-4o
CODING_MODEL=gpt-4o
SEARCH_MODEL=gpt-4o-mini
FAST_MODEL=gpt-4o-mini
MAX_SEARCH_ENGINES=10
MAX_REASONING_STEPS=10
MIN_CONFIDENCE_THRESHOLD=0.6
MEMORY_DIR=memory
DEBUG=false
```

---

## 📁 Project Structure

```
WISPR-AI/
├── main.py                  # FastAPI application entry point
├── config.py                # Centralised settings (pydantic-settings)
├── requirements.txt
├── pyproject.toml           # pytest config
├── agents/
│   ├── base_agent.py        # Abstract BaseAgent + AgentResult
│   ├── core_agent.py        # General intelligence agent
│   ├── coder_agent.py       # Multi-language coding agent
│   ├── search_agent.py      # MegaSearch + synthesis agent
│   ├── studio_agent.py      # Browser IDE + deployment agent
│   └── orchestrator.py      # Parallel orchestration agent
├── llm/
│   └── router.py            # LiteLLM-based model routing
├── search/
│   └── mega_search.py       # 50+ engine parallel search
├── memory/
│   └── store.py             # Short-term, long-term, task memory
├── reasoning/
│   └── engine.py            # ReAct autonomous reasoning loop
├── coding/
│   └── engine.py            # Language detection & code utilities
├── hallucination/
│   └── reducer.py           # Multi-layer hallucination reduction
├── studio/
│   └── ide.py               # WISPR AI Studio IDE simulation
├── plugins/
│   └── manager.py           # Dynamic plugin discovery & loading
├── api/
│   └── routes.py            # FastAPI route handlers
└── tests/                   # 75 unit + integration tests
```

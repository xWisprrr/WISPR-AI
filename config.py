"""WISPR AI OS — Central configuration."""

from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── App ────────────────────────────────────────────────────────────────
    app_name: str = "WISPR AI OS"
    app_version: str = "1.0.0"
    debug: bool = False

    # ── LLM ───────────────────────────────────────────────────────────────
    # Primary API keys (optional; use env vars in production)
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    gemini_api_key: str = ""
    groq_api_key: str = ""

    # Model routing preferences
    reasoning_model: str = "gpt-4o"
    coding_model: str = "gpt-4o"
    search_model: str = "gpt-4o-mini"
    general_model: str = "gpt-4o-mini"
    fallback_model: str = "gpt-4o-mini"

    # LLM request limits
    llm_max_tokens: int = 4096
    llm_temperature: float = 0.7
    llm_timeout: int = 60

    # ── MegaSearch ────────────────────────────────────────────────────────
    search_max_sources: int = 50
    search_results_per_engine: int = 5
    search_timeout: int = 15
    searxng_url: str = "https://searx.be"

    # ── Memory ────────────────────────────────────────────────────────────
    memory_dir: str = "memory/store"
    short_term_max_entries: int = 100
    long_term_db_path: str = "memory/store/long_term.json"
    task_memory_db_path: str = "memory/store/tasks.json"

    # ── Agents ────────────────────────────────────────────────────────────
    agent_max_retries: int = 3
    agent_timeout: int = 120

    # ── Reasoning ─────────────────────────────────────────────────────────
    reasoning_max_steps: int = 10
    reasoning_confidence_threshold: float = 0.75

    # ── Studio / IDE ──────────────────────────────────────────────────────
    studio_sandbox_dir: str = "/tmp/wispr_studio"
    studio_supported_languages: List[str] = [
        "python", "javascript", "typescript", "go", "rust",
        "java", "c", "cpp", "ruby", "bash",
    ]

    # ── Hallucination ─────────────────────────────────────────────────────
    hallucination_vote_threshold: int = 2
    hallucination_confidence_min: float = 0.6

    # ── Plugins ───────────────────────────────────────────────────────────
    plugins_dir: str = "plugins"

    # ── ReAct Agent ───────────────────────────────────────────────────────
    react_max_iterations: int = 8
    session_ttl_seconds: int = 3600
    session_max_turns: int = 20
    # ── Code Engine ───────────────────────────────────────────────────────
    # Session persistence
    code_engine_sessions_path: str = "coding/store/sessions.jsonl"
    # Audit log
    code_engine_audit_path: str = "coding/store/audit.jsonl"
    # Max auto-test retry rounds for automatic failure recovery
    code_engine_max_retries: int = 3
    # Deployment: supported providers
    code_engine_deploy_providers: List[str] = [
        "github", "vercel", "netlify", "railway", "docker"
    ]
    # Unrestricted filesystem access — set to True to enable deny-list warnings in logs
    code_engine_fs_warn_unrestricted: bool = True

    # ── Xencode ───────────────────────────────────────────────────────────
    xencode_workspace_root: str = "coding/store/xencode"
    xencode_artifacts_dir: str = "coding/store/xencode/artifacts"
    xencode_max_repair_attempts: int = 2
    xencode_restrict_to_workspace: bool = True
    xencode_build_timeout: int = 120

    # ── CORS ──────────────────────────────────────────────────────────────
    cors_origins: List[str] = ["*"]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

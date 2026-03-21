"""
WISPR AI OS — Configuration
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── General ───────────────────────────────────────────────────────────────
    app_name: str = "WISPR AI OS"
    version: str = "0.1.0"
    debug: bool = False

    # ── LLM keys (any subset is sufficient) ──────────────────────────────────
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    together_api_key: Optional[str] = None
    cohere_api_key: Optional[str] = None

    # ── Default models per task type ──────────────────────────────────────────
    reasoning_model: str = "gpt-4o"
    coding_model: str = "gpt-4o"
    search_model: str = "gpt-4o-mini"
    fast_model: str = "gpt-4o-mini"

    # ── Search ────────────────────────────────────────────────────────────────
    max_search_engines: int = 10          # cap for demo (full list = 50+)
    search_timeout_seconds: float = 10.0

    # ── Memory ────────────────────────────────────────────────────────────────
    memory_dir: Path = Path("memory")
    max_short_term_entries: int = 100

    # ── Reasoning ─────────────────────────────────────────────────────────────
    max_reasoning_steps: int = 10
    min_confidence_threshold: float = 0.6

    # ── Studio ────────────────────────────────────────────────────────────────
    allowed_deployment_targets: list[str] = ["github", "vercel", "netlify"]

    # ── Plugins ───────────────────────────────────────────────────────────────
    plugins_dir: Path = Path("plugins")


settings = Settings()

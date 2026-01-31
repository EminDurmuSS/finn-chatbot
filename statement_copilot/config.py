"""
Statement Copilot - Configuration (FIXED)
=========================================
Production-ready configuration with environment variables.
bunq-aligned, LangGraph-based financial assistant.

FIXES:
- Model names updated to pinned versions (aliases may not work)
- Added fallback model option
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional, Literal, List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .log_context import LogContextFilter
from .flow_logger import FLOW_LEVEL, FlowFormatter
# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS (module-level)
# ──────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ──────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ──────────────────────────────────────────────────────────────────────────────

def setup_logging(level: str = "INFO", flow_mode: bool = True) -> None:
    """
    Configure application logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        flow_mode: If True, use visual flow logging for workflow
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Context filter for trace/session/node
    context_filter = LogContextFilter()

    if flow_mode:
        # Flow handler - visual workflow logs
        flow_handler = logging.StreamHandler()
        flow_handler.setLevel(FLOW_LEVEL)
        flow_handler.setFormatter(FlowFormatter())
        flow_handler.addFilter(context_filter)
        flow_handler.addFilter(lambda r: r.levelno == FLOW_LEVEL)
        root_logger.addHandler(flow_handler)

        # Standard handler - only warnings and above (non-flow)
        std_handler = logging.StreamHandler()
        std_handler.setLevel(logging.WARNING)
        std_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        ))
        std_handler.addFilter(context_filter)
        std_handler.addFilter(lambda r: r.levelno != FLOW_LEVEL)
        root_logger.addHandler(std_handler)
    else:
        # Debug mode - detailed logging
        handler = logging.StreamHandler()
        handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | trace=%(trace_id)s | session=%(session_id)s | node=%(node)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        handler.addFilter(context_filter)
        root_logger.addHandler(handler)

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("watchfiles").setLevel(logging.WARNING)


# ──────────────────────────────────────────────────────────────────────────────
# SETTINGS
# ──────────────────────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    """
    Application settings with environment variable support (pydantic-settings v2).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ─────────────────────────────────────────────────────────────────
    # APPLICATION
    # ─────────────────────────────────────────────────────────────────
    app_name: str = "Statement Copilot"
    app_version: str = "2.0.2"  # Version bump for fixes
    debug: bool = Field(default=False, description="Enable debug mode")
    environment: Literal["development", "staging", "production"] = "development"

    # ─────────────────────────────────────────────────────────────────
    # PATHS
    # ─────────────────────────────────────────────────────────────────
    project_root: Path = Field(default=PROJECT_ROOT)
    db_path: Path = Field(default=Path("db/statement_copilot.duckdb"))
    outputs_path: Path = Field(default=Path("outputs"))

    # ─────────────────────────────────────────────────────────────────
    # ANTHROPIC (Primary LLM)
    # ─────────────────────────────────────────────────────────────────
    anthropic_api_key: str = Field(default="", description="Anthropic API key")

    # ═══════════════════════════════════════════════════════════════════
    # FIX: Use PINNED model versions, not aliases!
    # Aliases like "claude-sonnet-4-5" may cause 500 errors
    # ═══════════════════════════════════════════════════════════════════
    
    # Primary models (Claude Sonnet 4.5 - using plain JSON, not beta structured outputs)
    model_orchestrator: str = "claude-sonnet-4-5-20250929"
    model_sql_generator: str = "claude-sonnet-4-5-20250929"
    model_search_agent: str = "claude-sonnet-4-5-20250929"
    model_action_planner: str = "claude-sonnet-4-5-20250929"
    model_synthesizer: str = "claude-sonnet-4-5-20250929"
    model_validator: str = "claude-sonnet-4-5-20250929"

    # Lightweight models (Claude Haiku 4.5 - for fast, simple tasks)
    model_guardrail: str = "claude-haiku-4-5-20251001"
    model_summarizer: str = "claude-haiku-4-5-20251001"

    # Fallback model
    model_fallback: str = "claude-sonnet-4-5-20250929"

    # Structured outputs beta header - DISABLED due to timeout issues with complex schemas
    # The beta structured outputs feature times out with complex nested schemas
    # Using plain JSON completion instead (works reliably)
    anthropic_beta_headers: List[str] = Field(
        default_factory=lambda: []  # Empty = disabled, forces plain JSON fallback
    )

    # Client behavior
    anthropic_timeout_s: float = 60.0
    anthropic_max_retries: int = 3

    # ─────────────────────────────────────────────────────────────────
    # OPENROUTER (Embeddings)
    # ─────────────────────────────────────────────────────────────────
    openrouter_api_key: str = Field(default="", description="OpenRouter API key")
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    embedding_model: str = "qwen/qwen3-embedding-8b"
    embedding_dimensions: int = 4096

    # ─────────────────────────────────────────────────────────────────
    # PINECONE (Vector Store)
    # ─────────────────────────────────────────────────────────────────
    pinecone_api_key: str = Field(default="", description="Pinecone API key")
    pinecone_index_name: str = "statement-copilot-transactions"
    pinecone_index_host: Optional[str] = None
    pinecone_namespace: str = "__default__"

    # Dense vector search settings (no BM25/hybrid)
    # Using pure semantic search with OpenRouter embeddings

    # ─────────────────────────────────────────────────────────────────
    # POSTGRESQL (Checkpointing)
    # ─────────────────────────────────────────────────────────────────
    postgres_url: str = "postgresql://postgres:postgres@localhost:5432/copilot"

    # ─────────────────────────────────────────────────────────────────
    # REDIS (Cache & Queue)
    # ─────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379"

    # ─────────────────────────────────────────────────────────────────
    # LANGGRAPH
    # ─────────────────────────────────────────────────────────────────
    checkpointer_type: Literal["memory", "postgres"] = "memory"
    max_conversation_turns: int = 50
    context_window_limit: int = 100000
    summarize_after_messages: int = 20

    # ─────────────────────────────────────────────────────────────────
    # AGENT BEHAVIOR
    # ─────────────────────────────────────────────────────────────────
    default_date_range_days: int = 30
    max_sql_results: int = 500
    max_vector_results: int = 20
    require_confirmation_default: bool = True
    max_sql_refinements: int = 3
    sql_refinement_expand_months: int = 3

    # ─────────────────────────────────────────────────────────────────
    # SECURITY & GUARDRAILS
    # ─────────────────────────────────────────────────────────────────
    max_input_length: int = 4000
    blocked_keywords: List[str] = Field(
        default_factory=lambda: [
            "ignore previous", "system prompt", "reveal instructions",
            "drop table", "delete from", "truncate", "jailbreak",
        ]
    )
    pii_patterns_enabled: bool = True
    enable_response_validator: bool = True
    enable_search_llm: bool = True

    # ─────────────────────────────────────────────────────────────────
    # OBSERVABILITY
    # ─────────────────────────────────────────────────────────────────
    langsmith_api_key: Optional[str] = None
    langsmith_project: str = "statement-copilot-prod"
    enable_tracing: bool = True
    log_level: str = "INFO"
    log_flow_mode: bool = Field(
        default=True,
        description="Use visual flow logging (True) or detailed debug logging (False)"
    )
    debug_verbose: bool = Field(
        default=False,
        description="Show full details in flow logs (no truncation, show SQL queries)"
    )

    # ─────────────────────────────────────────────────────────────────
    # TENANT
    # ─────────────────────────────────────────────────────────────────
    default_tenant_id: str = "default_tenant"
    default_user_id: str = "default_user"

    # ─────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────
    def get_db_path(self) -> Path:
        """Get absolute database path."""
        return self.db_path if self.db_path.is_absolute() else (self.project_root / self.db_path)

    def get_outputs_path(self) -> Path:
        """Get absolute outputs path (and ensure it exists)."""
        path = self.outputs_path if self.outputs_path.is_absolute() else (self.project_root / self.outputs_path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
effective_log_level = "DEBUG" if settings.debug else settings.log_level
# In debug mode, disable flow_mode to see detailed logs
flow_mode = settings.log_flow_mode and not settings.debug
setup_logging(effective_log_level, flow_mode=flow_mode)

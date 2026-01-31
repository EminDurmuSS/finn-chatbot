"""
Statement Copilot - Core Package
================================
Shared schemas, state helpers, and service clients.
"""

from .schemas import (
    IntentType,
    RiskLevel,
    MetricType,
    ActionType,
    Direction,
    Period,
    DateRange,
    Constraints,
    RouterDecision,
    MetricFilters,
    MetricRequest,
    MetricResult,
    SearchQuery,
    SearchExpansion,
    SearchResult,
    TransactionMatch,
    ActionParams,
    ActionPlan,
    ActionPlanDraft,
    ActionResult,
    DataScope,
    GuardrailResult,
    SafetyClassification,
    Evidence,
    FinalResponse,
    ResponseValidation,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ActionApprovalRequest,
)

from .state import (
    OrchestratorState,
    create_initial_state,
    create_tool_call_record,
    get_date_range_from_constraints,
    format_message_history,
    summarize_state_for_response,
)

from .database import DatabaseManager, SQLBuilder, get_db


def get_llm_client():
    """Lazy-load the LLM client to avoid import-time dependencies."""
    from .llm import get_llm_client as _get_llm_client
    return _get_llm_client()


def get_vector_store():
    """Lazy-load the vector store to avoid import-time dependencies."""
    from .vector_store import get_vector_store as _get_vector_store
    return _get_vector_store()


__all__ = [
    # Schemas
    "IntentType",
    "RiskLevel",
    "MetricType",
    "ActionType",
    "Direction",
    "Period",
    "DateRange",
    "Constraints",
    "RouterDecision",
    "MetricFilters",
    "MetricRequest",
    "MetricResult",
    "SearchQuery",
    "SearchExpansion",
    "SearchResult",
    "TransactionMatch",
    "ActionParams",
    "ActionPlan",
    "ActionPlanDraft",
    "ActionResult",
    "DataScope",
    "GuardrailResult",
    "SafetyClassification",
    "Evidence",
    "FinalResponse",
    "ResponseValidation",
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "ActionApprovalRequest",

    # State helpers
    "OrchestratorState",
    "create_initial_state",
    "create_tool_call_record",
    "get_date_range_from_constraints",
    "format_message_history",
    "summarize_state_for_response",

    # Services
    "DatabaseManager",
    "SQLBuilder",
    "get_db",
    "get_llm_client",
    "get_vector_store",
]

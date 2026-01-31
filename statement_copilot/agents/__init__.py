"""
Statement Copilot - Agents Package
==================================
AI agents for financial analysis, search, and action planning.

Primary Agents:
- Orchestrator: Central routing and intent classification
- Finance Analyst: SQL-based financial analytics
- Search Agent: Multi-stage transaction search (NEW: Professional Search Engine)
- Action Planner: Export, report, and action execution
- Synthesizer: Response generation

Supporting Components:
- Guardrails: Input/output safety
- Response Validator: Evidence-based validation
"""

# =============================================================================
# ORCHESTRATOR
# =============================================================================
from .orchestrator import (
    OrchestratorAgent,
    ResponseSynthesizer,
    get_orchestrator,
    get_synthesizer,
)

# =============================================================================
# FINANCE ANALYST
# =============================================================================
from .finance_analyst import (
    FinanceAnalystAgent,
    get_finance_analyst,
    quick_sum,
    quick_category_breakdown,
)

# =============================================================================
# SEARCH AGENT (Professional Search Engine)
# =============================================================================
# New professional search with multi-stage retrieval
from .search_agent import (
    ProfessionalSearchAgent,
    get_professional_search_agent,
    get_search_agent,  # Alias for compatibility
    quick_search,
    analyze_query,
)

# Legacy search agent (deprecated but available)
from .search_agent import SearchAgent as LegacySearchAgent

# Self-RAG Search Subgraph (NEW)
from .search_graph import (
    build_search_subgraph,
    run_self_rag_search,
    get_search_subgraph,
)

# =============================================================================
# ACTION PLANNER
# =============================================================================
from .action_planner import (
    ActionPlannerAgent,
    get_action_planner,
)

# =============================================================================
# GUARDRAILS
# =============================================================================
from ..core.guardrails import (
    Guardrails,
    RuleBasedGuard,
    LLMGuard,
    PIIMasker,
    get_guardrails,
)

# =============================================================================
# RESPONSE VALIDATOR
# =============================================================================
from .response_validator import (
    ResponseValidatorAgent,
    get_response_validator,
)

# =============================================================================
# PUBLIC API
# =============================================================================
__all__ = [
    # Orchestrator
    "OrchestratorAgent",
    "ResponseSynthesizer",
    "get_orchestrator",
    "get_synthesizer",
    
    # Finance Analyst
    "FinanceAnalystAgent",
    "get_finance_analyst",
    "quick_sum",
    "quick_category_breakdown",
    
    # Search Agent (Professional)
    "ProfessionalSearchAgent",
    "get_professional_search_agent",
    "get_search_agent",
    "quick_search",
    "analyze_query",
    
    # Legacy Search (Deprecated)
    "LegacySearchAgent",
    
    # Action Planner
    "ActionPlannerAgent",
    "get_action_planner",
    
    # Guardrails
    "Guardrails",
    "RuleBasedGuard",
    "LLMGuard",
    "PIIMasker",
    "get_guardrails",
    
    # Response Validator
    "ResponseValidatorAgent",
    "get_response_validator",
    
    # Self-RAG Search
    "build_search_subgraph",
    "run_self_rag_search",
    "get_search_subgraph",
]
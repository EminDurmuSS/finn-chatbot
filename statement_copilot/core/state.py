"""
Statement Copilot - LangGraph State
===================================
Central state shared across all agents in the orchestration graph.

UPDATED: Default date behavior changed - no date filter when not specified
"""

from typing import TypedDict, Optional, List, Dict, Any, Annotated
from datetime import datetime, date
from operator import add
import hashlib
import json


# -----------------------------------------------------------------------------
# STATE REDUCERS
# -----------------------------------------------------------------------------

def replace_value(current: Any, new: Any) -> Any:
    """Replace current value with new"""
    return new if new is not None else current


def append_list(current: Optional[List], new: Optional[List]) -> List:
    """Append new items to current list"""
    if current is None:
        current = []
    if new is None:
        return current
    return current + new


def merge_dict(current: Optional[Dict], new: Optional[Dict]) -> Dict:
    """Merge new dict into current"""
    if current is None:
        current = {}
    if new is None:
        return current
    return {**current, **new}


# -----------------------------------------------------------------------------
# ORCHESTRATOR STATE
# -----------------------------------------------------------------------------

class OrchestratorState(TypedDict, total=False):
    """
    Central state shared across all agents in the LangGraph workflow.
    
    Flow:
    1. Input Guard -> 2. Orchestrator -> 3. Primary Agents -> 4. Validator -> 5. Synthesizer
    """
# -----------------------------------------------------------------------------
    # SESSION
# -----------------------------------------------------------------------------
    session_id: str
    trace_id: str
    tenant_id: str
    user_id: str
# -----------------------------------------------------------------------------
    # INPUT
# -----------------------------------------------------------------------------
    user_message: str
    original_message: str  # Before any modifications
    message_history: Annotated[List[Dict[str, Any]], replace_value]
# -----------------------------------------------------------------------------
    # ROUTING (set by orchestrator)
# -----------------------------------------------------------------------------
    intent: Optional[str]
    confidence: float
    reasoning: Optional[str]
    
    # Agent routing
    needs_sql: bool
    needs_vector: bool
    needs_planner: bool
    pending_agents: Optional[List[str]]
    completed_agents: List[str]

    # Risk
    risk_level: str
# -----------------------------------------------------------------------------
    # CONSTRAINTS
# -----------------------------------------------------------------------------
    constraints: Dict[str, Any]
# -----------------------------------------------------------------------------
    # AGENT RESULTS
# -----------------------------------------------------------------------------
    # Finance Analyst
    sql_metric_request: Optional[Dict[str, Any]]
    sql_result: Optional[Dict[str, Any]]
    sql_error: Optional[str]
    sql_refinements: List[str]  # Not using reducer - each query gets fresh refinements
    
    # Search Agent
    search_query: Optional[Dict[str, Any]]
    vector_result: Optional[Dict[str, Any]]
    search_error: Optional[str]
    search_retry: Optional[bool]
    
    # Self-RAG Search Introspection
    search_attempts: Optional[int]  # Number of Self-RAG retry attempts
    search_critique: Optional[str]  # Final critique from reflection
    search_evidence: Optional[Dict[str, Any]]  # Detailed search evidence
    
    # Action Planner
    action_plan: Optional[Dict[str, Any]]
    action_result: Optional[Dict[str, Any]]
    action_error: Optional[str]
# -----------------------------------------------------------------------------
    # OUTPUT
# -----------------------------------------------------------------------------
    final_answer: str
    evidence: Dict[str, Any]
    suggestions: Optional[List[str]]
    follow_up_questions: Optional[List[str]]
    clarification_needed: Optional[str]
    suggested_questions: Optional[List[str]]
    previous_answer: Optional[str]
    previous_evidence: Optional[Dict[str, Any]]
    previous_intent: Optional[str]
    output_warnings: List[str]  # Not using reducer - each turn gets fresh warnings
    validation_issues: List[str]  # Not using reducer - each turn gets fresh issues
# -----------------------------------------------------------------------------
    # ACTION HANDLING (Human-in-the-Loop)
# -----------------------------------------------------------------------------
    needs_confirmation: bool
    pending_action_id: Optional[str]
    user_confirmed: Optional[bool]
    confirmation_message: Optional[str]
# -----------------------------------------------------------------------------
    # GUARDRAILS
# -----------------------------------------------------------------------------
    guardrail_passed: bool
    guardrail_warnings: List[str]  # Not using reducer - each turn gets fresh warnings
    blocked_reason: Optional[str]
# -----------------------------------------------------------------------------
    # AUDIT
# -----------------------------------------------------------------------------
    tool_calls: List[Dict[str, Any]]  # Not using reducer - each turn gets fresh calls
    errors: List[str]  # Not using reducer - each turn gets fresh errors
    
    # Timing
    started_at: datetime
    completed_at: Optional[datetime]
    total_latency_ms: Optional[int]


# -----------------------------------------------------------------------------
# STATE FACTORY
# -----------------------------------------------------------------------------

def create_initial_state(
    session_id: str,
    trace_id: str,
    tenant_id: str,
    user_id: str,
    user_message: str,
    message_history: Optional[List[Dict[str, Any]]] = None
) -> OrchestratorState:
    """Create initial state for a new orchestration run"""
    return OrchestratorState(
        # Session
        session_id=session_id,
        trace_id=trace_id,
        tenant_id=tenant_id,
        user_id=user_id,
        
        # Input
        user_message=user_message,
        original_message=user_message,
        message_history=message_history,
        
        # Routing
        intent=None,
        confidence=0.0,
        reasoning=None,
        needs_sql=False,
        needs_vector=False,
        needs_planner=False,
        pending_agents=None,
        completed_agents=[],
        risk_level="low",
        
        # Constraints
        constraints={},
        
        # Agent results
        sql_metric_request=None,
        sql_result=None,
        sql_error=None,
        sql_refinements=[],
        search_query=None,
        vector_result=None,
        search_error=None,
        search_retry=None,
        action_plan=None,
        action_result=None,
        action_error=None,
        
        # Output
        final_answer="",
        evidence={},
        suggestions=None,
        follow_up_questions=None,
        clarification_needed=None,
        suggested_questions=None,
        previous_answer=None,
        previous_evidence=None,
        previous_intent=None,
        output_warnings=[],
        validation_issues=[],
        
        # Action handling
        needs_confirmation=False,
        pending_action_id=None,
        user_confirmed=None,
        confirmation_message=None,
        
        # Guardrails
        guardrail_passed=True,
        guardrail_warnings=[],
        blocked_reason=None,
        
        # Audit
        tool_calls=[],
        errors=[],
        
        # Timing
        started_at=datetime.utcnow(),
        completed_at=None,
        total_latency_ms=None,
    )


# -----------------------------------------------------------------------------
# STATE UTILITIES
# -----------------------------------------------------------------------------

def hash_data(data: Any) -> str:
    """Create hash of data for audit logging"""
    try:
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]
    except:
        return hashlib.sha256(str(data).encode()).hexdigest()[:16]


def create_tool_call_record(
    state: OrchestratorState,
    node: str,
    tool_name: str,
    model_name: Optional[str],
    input_data: Any,
    output_data: Any,
    latency_ms: int,
    success: bool,
    error_message: Optional[str] = None
) -> Dict[str, Any]:
    """Create a tool call record for audit logging"""
    return {
        "trace_id": state.get("trace_id", ""),
        "session_id": state.get("session_id", ""),
        "node": node,
        "tool_name": tool_name,
        "model_name": model_name,
        "input_hash": hash_data(input_data),
        "output_hash": hash_data(output_data),
        "latency_ms": latency_ms,
        "success": success,
        "error_message": error_message,
        "created_at": datetime.utcnow().isoformat(),
    }


def get_date_range_from_constraints(
    constraints: Dict[str, Any],
    default_days: int = 30  # Kept for backward compatibility but NOT used by default
) -> tuple[Optional[date], Optional[date]]:
    """
    Extract date range from constraints.
    
    UPDATED BEHAVIOR:
    - If no date/period specified, returns (None, None) to search ALL data
    - Only returns actual dates when user explicitly specifies a time period
    
    Args:
        constraints: Constraints dict from orchestrator
        default_days: NOT USED anymore (kept for backward compatibility)
        
    Returns:
        Tuple of (start_date, end_date) or (None, None) for all-time search
    """
    from datetime import timedelta
    from dateutil.relativedelta import relativedelta
    
    today = date.today()
    
    # Explicit date range - user specified exact dates
    date_range = constraints.get("date_range")
    if date_range:
        if isinstance(date_range, dict):
            start = date_range.get("start")
            end = date_range.get("end")
            if isinstance(start, str):
                start = date.fromisoformat(start)
            if isinstance(end, str):
                end = date.fromisoformat(end)
            if start and end:
                return start, end
    
    # Implicit period - user mentioned a time period like "this month"
    implicit_period = constraints.get("implicit_period")
    
    # If no period specified, return None to search ALL data
    if not implicit_period:
        return None, None
    
    # Handle explicit period mentions
    if implicit_period == "today":
        return today, today
    elif implicit_period == "yesterday":
        yesterday = today - timedelta(days=1)
        return yesterday, yesterday
    elif implicit_period == "this_week":
        start = today - timedelta(days=today.weekday())
        return start, today
    elif implicit_period == "last_week":
        end = today - timedelta(days=today.weekday() + 1)
        start = end - timedelta(days=6)
        return start, end
    elif implicit_period == "this_month":
        start = today.replace(day=1)
        return start, today
    elif implicit_period == "last_month":
        first_this_month = today.replace(day=1)
        end = first_this_month - timedelta(days=1)
        start = end.replace(day=1)
        return start, end
    elif implicit_period == "last_3_months":
        start = today - relativedelta(months=3)
        return start, today
    elif implicit_period == "last_6_months":
        start = today - relativedelta(months=6)
        return start, today
    elif implicit_period == "this_year":
        start = today.replace(month=1, day=1)
        return start, today
    elif implicit_period == "last_year":
        start = today.replace(year=today.year-1, month=1, day=1)
        end = today.replace(year=today.year-1, month=12, day=31)
        return start, end

    # Unknown period - return None to search all data
    return None, None


def format_message_history(
    history: List[Dict[str, Any]],
    max_messages: int = 10
) -> str:
    """Format message history for prompt context"""
    if not history:
        return "No previous messages yet."
    
    recent = history[-max_messages:]
    formatted = []
    
    for msg in recent:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        if role == "user":
            formatted.append(f"User: {content}")
        else:
            # Truncate long assistant messages
            if len(content) > 200:
                content = content[:200] + "..."
            formatted.append(f"Assistant: {content}")
    
    return "\n".join(formatted)


def summarize_state_for_response(state: OrchestratorState) -> Dict[str, Any]:
    """Summarize state for API response"""
    return {
        "session_id": state.get("session_id"),
        "trace_id": state.get("trace_id"),
        "intent": state.get("intent"),
        "confidence": state.get("confidence"),
        "agents_used": [
            agent for agent, used in [
                ("finance_analyst", state.get("sql_result")),
                ("search_agent", state.get("vector_result")),
                ("action_planner", state.get("action_plan")),
            ] if used
        ],
        "needs_confirmation": state.get("needs_confirmation", False),
        "action_id": state.get("pending_action_id"),
        "warnings": state.get("guardrail_warnings", []),
        "total_latency_ms": state.get("total_latency_ms"),
    }
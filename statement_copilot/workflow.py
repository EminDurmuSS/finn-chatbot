"""
Statement Copilot - LangGraph Workflow
======================================
The orchestration graph that ties everything together.

Flow:
Input Guard -> Orchestrator -> [Finance Analyst | Search Agent | Action Planner] -> Synthesizer -> Output
"""

import logging
import time
import uuid
from typing import Optional, Literal, Dict, Any
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

from .config import settings
from .core import (
    OrchestratorState,
    IntentType,
    create_initial_state,
)
from .log_context import log_context, clip_text, flow_context, get_flow
from .agents import (
    get_guardrails,
    get_orchestrator,
    get_finance_analyst,
    get_search_agent,
    get_action_planner,
    get_synthesizer,
    get_response_validator,
)

logger = logging.getLogger(__name__)


def _mark_agent_done(state: OrchestratorState, agent_name: str) -> OrchestratorState:
    """Update pending/completed agent tracking."""
    pending = state.get("pending_agents") or []
    if agent_name in pending:
        pending = [a for a in pending if a != agent_name]
    state["pending_agents"] = pending

    completed = state.get("completed_agents") or []
    if agent_name not in completed:
        completed.append(agent_name)
    state["completed_agents"] = completed
    return state


# -----------------------------------------------------------------------------
# NODE FUNCTIONS
# -----------------------------------------------------------------------------

def input_guard_node(state: OrchestratorState) -> OrchestratorState:
    """
    Input validation and safety check.
    First node in the workflow.
    """
    flow = get_flow()
    with log_context(node="input_guard"):
        guardrails = get_guardrails()
        state = guardrails.process_state(state)

        passed = state.get("guardrail_passed", True)
        if flow:
            with flow.node("input_guard"):
                flow.detail("passed", passed)
                if not passed:
                    flow.detail("reason", clip_text(state.get("blocked_reason"), 60))

        return state


def orchestrator_node(state: OrchestratorState) -> OrchestratorState:
    """
    Central routing decision.
    Classifies intent and determines which agents to invoke.
    """
    flow = get_flow()
    with log_context(node="orchestrator"):
        orchestrator = get_orchestrator()
        state = orchestrator.route(state)

        if flow:
            with flow.node("orchestrator"):
                flow.detail("intent", state.get("intent"))
                flow.detail("confidence", state.get("confidence"))

                # Show reasoning (full in verbose mode)
                reasoning = state.get("reasoning")
                if reasoning:
                    flow.detail("reasoning", reasoning)

                # Show constraints extracted
                constraints = state.get("constraints") or {}
                if constraints:
                    flow.verbose_detail("constraints", constraints)

                # Show routing with reasons
                routing = []
                routing_reasons = []
                if state.get("needs_sql"):
                    routing.append("finance_analyst")
                    routing_reasons.append("needs_sql: analytics/calculation required")
                if state.get("needs_vector"):
                    routing.append("search_agent")
                    routing_reasons.append("needs_vector: transaction lookup/search required")
                if state.get("needs_planner"):
                    routing.append("action_planner")
                    routing_reasons.append("needs_planner: action/export requested")

                if routing:
                    flow.detail("routing", routing)
                    # Show why each agent was selected in verbose mode
                    for reason in routing_reasons:
                        flow.detail("routing_reason", reason)

                # Show clarification if needed
                clarification = state.get("clarification_needed")
                if clarification:
                    flow.detail("clarification", clarification)

                # Show suggestions in verbose mode
                suggestions = state.get("suggested_questions")
                if suggestions:
                    flow.verbose_detail("suggestions", suggestions)

        return state


def finance_analyst_node(state: OrchestratorState) -> OrchestratorState:
    """
    SQL-based financial analysis.
    Invoked when needs_sql=True.
    """
    flow = get_flow()
    if flow:
        with flow.node("finance_planning"):
            flow.detail("status", "querying_history")
    with log_context(node="finance_analyst"):
        analyst = get_finance_analyst()
        state = analyst.analyze(state)

        if flow:
            with flow.node("finance_analyst"):
                sql_result = state.get("sql_result") or {}
                metric_req = state.get("sql_metric_request") or {}

                flow.detail("tool", "sql_query")
                flow.detail("metric", metric_req.get("metric"))

                # Show what filters were applied (reasoning)
                filters = metric_req.get("filters") or {}
                if filters:
                    filter_summary = []
                    if filters.get("categories"):
                        filter_summary.append(f"category={filters['categories']}")
                    if filters.get("date_start"):
                        filter_summary.append(f"date>={filters['date_start']}")
                    if filters.get("date_end"):
                        filter_summary.append(f"date<={filters['date_end']}")
                    if filters.get("merchants"):
                        filter_summary.append(f"merchant={filters['merchants'][:2]}")
                    if filters.get("direction"):
                        filter_summary.append(f"direction={filters['direction']}")
                    if filter_summary:
                        flow.detail("filters", ", ".join(filter_summary))

                # Log SQL query details (verbose mode shows full query)
                sql_preview = sql_result.get("sql_preview")
                sql_params = sql_result.get("sql_params")
                sql_row_count = sql_result.get("sql_row_count")
                sql_duration = sql_result.get("sql_duration_ms")
                if sql_preview:
                    flow.sql(
                        query=sql_preview,
                        params=sql_params,
                        row_count=sql_row_count,
                        duration_ms=sql_duration
                    )

                if state.get("sql_error"):
                    flow.detail("error", clip_text(state.get("sql_error"), 40))
                else:
                    # Format result value
                    value = sql_result.get("value")
                    tx_count = sql_result.get("tx_count")
                    if value is not None:
                        flow.detail("result", f"{value:,.2f} TRY ({tx_count} transactions)" if tx_count else f"{value:,.2f}")

                # Show refinements if any
                refinements = state.get("sql_refinements")
                if refinements:
                    flow.verbose_detail("refinements", refinements)

        return _mark_agent_done(state, "finance_analyst")


def search_agent_node(state: OrchestratorState) -> OrchestratorState:
    """
    Vector-based transaction search with Self-RAG capabilities.
    Invoked when needs_vector=True.
    
    Uses Self-RAG pattern for automatic query refinement and retry.
    """
    flow = get_flow()
    if flow:
        with flow.node("search_planning"):
            flow.detail("status", "planning_retrieval")
            flow.detail("query", clip_text(state.get("user_message", ""), 60))
    with log_context(node="search_agent"):
        search = get_search_agent()
        
        # Use Self-RAG search for intelligent retry and query refinement
        state = search.search_with_self_rag(state)

        if flow:
            with flow.node("search_agent"):
                vector_result = state.get("vector_result") or {}
                search_query = state.get("search_query") or {}
                search_evidence = state.get("search_evidence") or {}

                # Determine if Self-RAG was used
                self_rag_info = search_evidence.get("self_rag_info") or {}
                if self_rag_info:
                    flow.detail("tool", "self_rag_search")
                    flow.detail("attempts", self_rag_info.get("attempts", 1))
                    flow.detail("final_quality", self_rag_info.get("final_quality"))
                    
                    # Show query evolution if there were retries
                    if self_rag_info.get("attempts", 1) > 1:
                        flow.substep(
                            "Query Refinement",
                            original_query=search_query.get("query_text"),
                            final_query=self_rag_info.get("final_query"),
                            attempts=self_rag_info.get("attempts"),
                        )
                else:
                    flow.detail("tool", "professional_search")

                # Show the query
                query_text = search_query.get("query_text", "")
                flow.detail("query", query_text)

                # Query Understanding substep
                intent = search_query.get("intent")
                strategy = search_query.get("strategy")
                confidence = search_query.get("confidence")
                entities = search_query.get("entities") or {}
                expanded = search_query.get("expanded_query")
                search_terms = search_query.get("search_terms")

                flow.substep(
                    "Query Understanding",
                    intent=intent,
                    strategy=strategy,
                    confidence=confidence,
                    entities=entities,
                )

                # Verbose details
                if expanded:
                    flow.verbose_detail("expanded_query", expanded)
                if search_terms:
                    flow.verbose_detail("search_terms", search_terms)

                # Retrieval substep
                sources_used = search_evidence.get("sources_used") or []
                filters_applied = search_evidence.get("filters_applied") or {}
                search_time_ms = search_evidence.get("search_time_ms")

                if sources_used:
                    flow.substep(
                        "Retrieval",
                        sources=sources_used,
                        filters=filters_applied,
                        time_ms=search_time_ms,
                    )

                # Show reasoning in verbose mode
                reasoning = search_evidence.get("reasoning")
                if reasoning:
                    flow.verbose_detail("reasoning", reasoning)

                if state.get("search_error"):
                    flow.detail("error", clip_text(state.get("search_error"), 40))
                else:
                    flow.detail("found", vector_result.get("total_found", 0))

        return _mark_agent_done(state, "search_agent")


def action_planner_node(state: OrchestratorState) -> OrchestratorState:
    """
    Action planning for exports, reports, etc.
    Invoked when needs_planner=True.
    """
    flow = get_flow()
    with log_context(node="action_planner"):
        planner = get_action_planner()
        state = planner.plan(state)

        if flow:
            with flow.node("action_planner"):
                plan = state.get("action_plan") or {}

                flow.detail("tool", "action_plan")
                flow.detail("action_type", plan.get("action_type"))

                # Show what the plan will do (reasoning)
                human_plan = plan.get("human_plan")
                if human_plan:
                    flow.detail("plan", clip_text(human_plan, 60))

                flow.detail("risk", plan.get("risk_level"))

                if state.get("action_error"):
                    flow.detail("error", clip_text(state.get("action_error"), 40))
                elif state.get("needs_confirmation"):
                    flow.detail("needs_confirm", True)

        return _mark_agent_done(state, "action_planner")


def action_executor_node(state: OrchestratorState) -> OrchestratorState:
    """
    Execute approved action.
    Only runs after user confirmation.
    """
    flow = get_flow()
    with log_context(node="action_executor"):
        planner = get_action_planner()
        state = planner.execute(state)

        if flow:
            with flow.node("action_executor"):
                result = state.get("action_result") or {}
                plan = state.get("action_plan") or {}

                flow.detail("action_type", plan.get("action_type"))
                flow.detail("status", result.get("status"))

                if result.get("error"):
                    flow.detail("error", clip_text(result.get("error"), 40))

        return state


def synthesizer_node(state: OrchestratorState) -> OrchestratorState:
    """
    Generate final response.
    Converts agent results to natural language.
    """
    flow = get_flow()
    with log_context(node="synthesizer"):
        synthesizer = get_synthesizer()
        state = synthesizer.synthesize(state)

        # Set completion time
        state["completed_at"] = datetime.utcnow()
        if state.get("started_at"):
            delta = state["completed_at"] - state["started_at"]
            state["total_latency_ms"] = int(delta.total_seconds() * 1000)

        if flow:
            with flow.node("synthesizer"):
                answer_len = len(state.get("final_answer", ""))
                flow.detail("answer_len", f"{answer_len} chars")

        return state


def agent_dispatcher_node(state: OrchestratorState) -> OrchestratorState:
    """
    Sequential dispatcher for agents.
    Initializes pending_agents once and routes one-by-one.
    """
    with log_context(node="agent_dispatcher"):
        if state.get("pending_agents") is None:
            pending = []
            if state.get("needs_sql"):
                pending.append("finance_analyst")
            if state.get("needs_vector"):
                pending.append("search_agent")
            if state.get("needs_planner"):
                pending.append("action_planner")
            state["pending_agents"] = pending
            state["completed_agents"] = state.get("completed_agents", [])

        return state


def post_agent_validator_node(state: OrchestratorState) -> OrchestratorState:
    """
    Validate agent outputs before synthesis/confirmation.
    Ensures required outputs exist and handles missing action params.
    """
    flow = get_flow()
    with log_context(node="post_agent_validator"):
        # Always reset validation_issues to prevent accumulation across queries
        issues = []

        if state.get("needs_sql") and not state.get("sql_result") and not state.get("sql_error"):
            issues.append("SQL result could not be produced.")
        if state.get("needs_vector") and not state.get("vector_result") and not state.get("search_error"):
            issues.append("Search result could not be produced.")
        if state.get("needs_planner") and not state.get("action_plan") and not state.get("action_error"):
            issues.append("Action plan could not be produced.")

        # Always set (even if empty) to clear previous session issues
        state["validation_issues"] = issues

        # Validate required action params
        action_plan = state.get("action_plan") or {}
        params = action_plan.get("params") or {}
        action_type = action_plan.get("action_type")

        missing_fields = []
        if action_type == "CATEGORY_UPDATE":
            if not params.get("tx_ids"):
                missing_fields.append("tx_ids")
            if not params.get("new_category"):
                missing_fields.append("new_category")
        elif action_type == "SET_BUDGET_ALERT":
            if not params.get("category"):
                missing_fields.append("category")
            if params.get("threshold_amount") is None:
                missing_fields.append("threshold_amount")
        elif action_type == "SET_REMINDER":
            if not params.get("reminder_date"):
                missing_fields.append("reminder_date")
            if not params.get("reminder_message"):
                missing_fields.append("reminder_message")

        if missing_fields:
            state["intent"] = IntentType.CLARIFY.value
            state["clarification_needed"] = (
                "Some information is missing for this action. Could you clarify: "
                + ", ".join(missing_fields)
            )
            state["suggested_questions"] = [
                "Could you provide the missing details?",
            ]
            state["action_plan"] = None
            state["needs_confirmation"] = False
            state["pending_action_id"] = None

            if flow:
                with flow.node("post_agent_validator"):
                    flow.detail("status", "clarify")
                    flow.detail("missing", missing_fields)
            return state

        # Ensure needs_confirmation is aligned with action plan
        if action_plan and action_plan.get("requires_confirmation", True):
            state["needs_confirmation"] = True

        if flow:
            with flow.node("post_agent_validator"):
                flow.detail("status", "ok")
                if state.get("needs_confirmation"):
                    flow.detail("needs_confirm", True)

        return state

def output_guard_node(state: OrchestratorState) -> OrchestratorState:
    """
    Output validation + PII masking.
    """
    flow = get_flow()
    with log_context(node="output_guard"):
        validator = get_response_validator()
        guardrails = get_guardrails()

        # Always reset output_warnings to prevent accumulation across queries
        state["output_warnings"] = []

        if state.get("final_answer"):
            validation = validator.validate(state)
            if validation.get("corrected_answer"):
                state["final_answer"] = validation["corrected_answer"]
                # If we have a corrected answer, the agent has fixed the issue.
                # We can suppress warnings to provide a cleaner UX.
                state["output_warnings"] = []
            elif validation.get("issues"):
                state["output_warnings"] = validation["issues"]

            # PII mask after validation
            state["final_answer"] = guardrails.mask_output(state["final_answer"])

        # Update message history (persistence)
        user_msg = state.get("original_message") or state.get("user_message")
        assistant_msg = state.get("final_answer")
        
        # Get existing history (safe copy)
        current_history = list(state.get("message_history") or [])
        
        # Append new interaction
        if user_msg:
             current_history.append({"role": "user", "content": user_msg})
        if assistant_msg:
             current_history.append({"role": "assistant", "content": assistant_msg})
        
        # Prepare updates to return
        # Since message_history uses replace_value reducer, we return the FULL list
        updates = {
            "final_answer": state["final_answer"],
            "output_warnings": state["output_warnings"],
            "message_history": current_history,
        }

        if flow:
            # Mark as last node before response
            flow.mark_last_node()
            with flow.node("output_guard"):
                warnings = state.get("output_warnings") or []
                if warnings:
                    flow.detail("warnings", len(warnings))
                else:
                    flow.detail("status", "ok")

        return updates


def human_confirmation_node(state: OrchestratorState):
    """
    Human-in-the-loop confirmation for actions.
    Uses LangGraph's interrupt() for pause/resume.
    """
    flow = get_flow()
    with log_context(node="human_confirmation"):
        action_plan = state.get("action_plan")

        if not action_plan:
            return state

        # Present plan to user and wait for confirmation
        confirmation_message = f"""
{action_plan.get('human_plan', 'Action planned.')}

Risk level: {action_plan.get('risk_level', 'low')}

Do you approve this action?
"""

        if flow:
            with flow.node("human_confirmation"):
                flow.detail("action_type", action_plan.get("action_type"))
                flow.detail("waiting", "user approval")

        # Interrupt and wait for user input
        user_response = interrupt({
            "type": "confirmation",
            "action_id": action_plan.get("action_id"),
            "message": confirmation_message,
            "plan": action_plan,
        })

        # Process user response
        if isinstance(user_response, dict):
            state["user_confirmed"] = user_response.get("approved", False)
            state["confirmation_message"] = user_response.get("reason")
        else:
            state["user_confirmed"] = bool(user_response)

        return state

def route_after_guard(state: OrchestratorState) -> Literal["orchestrator", "synthesizer"]:
    """Route after input guard - block or continue"""
    passed = state.get("guardrail_passed", True)
    return "orchestrator" if passed else "synthesizer"


def route_after_orchestrator(state: OrchestratorState) -> Literal["agent_dispatcher", "synthesizer"]:
    """
    Route after orchestrator - chitchat/clarify/explain go direct,
    everything else uses sequential dispatcher.
    """
    intent = state.get("intent")

    if intent in [
        IntentType.CHITCHAT.value,
        IntentType.CLARIFY.value,
        IntentType.EXPLAIN.value,
    ]:
        return "synthesizer"
    return "agent_dispatcher"


def route_after_dispatcher(state: OrchestratorState) -> Literal[
    "finance_analyst", "search_agent", "action_planner", "post_agent_validator"
]:
    """Route to next pending agent or validator."""
    pending = state.get("pending_agents") or []
    return pending[0] if pending else "post_agent_validator"


def route_after_validator(state: OrchestratorState) -> Literal["synthesizer", "human_confirmation"]:
    """Route after validation - need confirmation or synthesize"""
    needs_confirmation = bool(state.get("needs_confirmation") and state.get("action_plan"))
    return "human_confirmation" if needs_confirmation else "synthesizer"


def route_after_confirmation(state: OrchestratorState) -> Literal["action_executor", "synthesizer"]:
    """Route after user confirmation - execute or skip"""
    confirmed = bool(state.get("user_confirmed"))
    return "action_executor" if confirmed else "synthesizer"


# -----------------------------------------------------------------------------
# GRAPH BUILDER
# -----------------------------------------------------------------------------

def build_workflow() -> StateGraph:
    """
    Build the LangGraph workflow.
    
    Returns:
        Compiled StateGraph ready for execution
    """
    # Create graph
    workflow = StateGraph(OrchestratorState)
    
    # Add nodes
    workflow.add_node("input_guard", input_guard_node)
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("agent_dispatcher", agent_dispatcher_node)
    workflow.add_node("finance_analyst", finance_analyst_node)
    workflow.add_node("search_agent", search_agent_node)
    workflow.add_node("action_planner", action_planner_node)
    workflow.add_node("post_agent_validator", post_agent_validator_node)
    workflow.add_node("human_confirmation", human_confirmation_node)
    workflow.add_node("action_executor", action_executor_node)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("output_guard", output_guard_node)
    
    # Entry point
    workflow.add_edge(START, "input_guard")
    
    # After input guard
    workflow.add_conditional_edges(
        "input_guard",
        route_after_guard,
        {
            "orchestrator": "orchestrator",
            "synthesizer": "synthesizer",
        }
    )
    
    # After orchestrator
    workflow.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {
            "synthesizer": "synthesizer",
            "agent_dispatcher": "agent_dispatcher",
        }
    )
    
    # Sequential agent routing
    workflow.add_conditional_edges(
        "agent_dispatcher",
        route_after_dispatcher,
        {
            "finance_analyst": "finance_analyst",
            "search_agent": "search_agent",
            "action_planner": "action_planner",
            "post_agent_validator": "post_agent_validator",
        }
    )

    workflow.add_edge("finance_analyst", "agent_dispatcher")
    workflow.add_edge("search_agent", "agent_dispatcher")
    workflow.add_edge("action_planner", "agent_dispatcher")

    # After agent validation
    workflow.add_conditional_edges(
        "post_agent_validator",
        route_after_validator,
        {
            "synthesizer": "synthesizer",
            "human_confirmation": "human_confirmation",
        }
    )
    
    # After confirmation
    workflow.add_conditional_edges(
        "human_confirmation",
        route_after_confirmation,
        {
            "action_executor": "action_executor",
            "synthesizer": "synthesizer",
        }
    )
    
    # After action execution
    workflow.add_edge("action_executor", "synthesizer")
    
    # Output guard
    workflow.add_edge("synthesizer", "output_guard")
    workflow.add_edge("output_guard", END)
    
    return workflow


# -----------------------------------------------------------------------------
# COPILOT CLASS
# -----------------------------------------------------------------------------

class StatementCopilot:
    """
    Main entry point for the Statement Copilot.
    Manages workflow execution and checkpointing.
    """
    
    def __init__(self, checkpointer: Optional[Any] = None):
        """
        Initialize copilot.
        
        Args:
            checkpointer: Optional checkpointer for persistence
        """
        self.workflow = build_workflow()
        
        # Use provided checkpointer or create memory saver
        if checkpointer:
            self.checkpointer = checkpointer
        else:
            # Use SQLite for persistence to prevent memory leaks
            import os
            import sqlite3
            from langgraph.checkpoint.sqlite import SqliteSaver

            # Ensure db directory exists
            os.makedirs("db", exist_ok=True)
            
            # Connect to SQLite (check_same_thread=False needed for FastAPI)
            # OPTIMIZATION: Enable WAL mode for better concurrency and set busy timeout
            conn = sqlite3.connect("db/checkpoints.sqlite", check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=5000;")
            
            self.checkpointer = SqliteSaver(conn)
        
        # Compile graph
        self.graph = self.workflow.compile(checkpointer=self.checkpointer)
        
        logger.info("StatementCopilot initialized with SqliteSaver (WAL mode)")

    def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        message_history: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Process a user message.

        Args:
            message: User's message
            session_id: Session ID for checkpointing
            tenant_id: Tenant ID for data isolation
            user_id: User ID
            message_history: Previous messages for context

        Returns:
            Response dict with answer and metadata
        """
        # Generate IDs
        session_id = session_id or str(uuid.uuid4())
        trace_id = str(uuid.uuid4())[:8]
        tenant_id = tenant_id or settings.default_tenant_id
        user_id = user_id or settings.default_user_id

        with flow_context(trace_id=trace_id, session_id=session_id) as flow:
            with log_context(trace_id=trace_id, session_id=session_id):
                # Log request
                flow.request(message, tenant=tenant_id, user=user_id)

                # Config for checkpointing (used for previous state lookup too)
                config = {
                    "configurable": {
                        "thread_id": session_id,
                    }
                }

                # Fetch previous state (for EXPLAIN intent and context)
                previous_state = None
                try:
                    snapshot = self.graph.get_state(config)
                    previous_state = snapshot.values if snapshot else None
                except Exception:
                    previous_state = None

                # Create initial state
                state = create_initial_state(
                    session_id=session_id,
                    trace_id=trace_id,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    user_message=message,
                    message_history=message_history,
                )
                if previous_state:
                    state["previous_answer"] = previous_state.get("final_answer")
                    state["previous_evidence"] = previous_state.get("evidence")
                    state["previous_intent"] = previous_state.get("intent")

                try:
                    # Run workflow
                    result = self.graph.invoke(state, config)

                    # Handle LangGraph interrupts (human confirmation)
                    if isinstance(result, dict) and "__interrupt__" in result:
                        interrupts = result.get("__interrupt__") or []
                        interrupt_value = None
                        if interrupts:
                            interrupt = interrupts[0]
                            interrupt_value = interrupt.value if hasattr(interrupt, "value") else interrupt

                        action_plan = result.get("action_plan")
                        confirmation_message = None
                        if isinstance(interrupt_value, dict):
                            action_plan = interrupt_value.get("plan") or action_plan
                            confirmation_message = interrupt_value.get("message")

                        flow.response(
                            confirmation_message or (action_plan or {}).get("human_plan", ""),
                            status="pending"
                        )

                        return {
                            "answer": confirmation_message or (action_plan or {}).get("human_plan", ""),
                            "session_id": session_id,
                            "trace_id": trace_id,
                            "intent": result.get("intent"),
                            "confidence": result.get("confidence"),
                            "evidence": result.get("evidence", {}),
                            "suggestions": result.get("suggestions"),
                            "needs_confirmation": True,
                            "action_plan": action_plan,
                            "action_result": None,
                            "total_latency_ms": result.get("total_latency_ms"),
                            "warnings": (result.get("guardrail_warnings", []) + result.get("output_warnings", [])),
                        }

                    # Build response
                    response = {
                        "answer": result.get("final_answer", ""),
                        "session_id": session_id,
                        "trace_id": trace_id,
                        "intent": result.get("intent"),
                        "confidence": result.get("confidence"),
                        "evidence": result.get("evidence", {}),
                        "suggestions": result.get("suggestions"),
                        "needs_confirmation": result.get("needs_confirmation", False),
                        "action_plan": result.get("action_plan"),
                        "action_result": result.get("action_result"),
                        "total_latency_ms": result.get("total_latency_ms"),
                        "warnings": (result.get("guardrail_warnings", []) + result.get("output_warnings", [])),
                    }

                    flow.response(response.get("answer", ""), status="success")
                    return response

                except Exception as e:
                    flow.error("Workflow error", exception=e)
                    return {
                        "answer": "Sorry, something went wrong. Please try again.",
                        "session_id": session_id,
                        "trace_id": trace_id,
                        "error": str(e),
                    }

    def confirm_action(
        self,
        session_id: str,
        action_id: str,
        approved: bool,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Confirm or reject a pending action.

        Args:
            session_id: Session ID
            action_id: Action ID to confirm
            approved: Whether user approved
            reason: Optional reason for decision

        Returns:
            Response with action result
        """
        current_state = self.get_state(session_id)
        trace_id = (current_state or {}).get("trace_id", "")

        with flow_context(trace_id=trace_id, session_id=session_id) as flow:
            with log_context(trace_id=trace_id, session_id=session_id):
                flow.request(
                    f"Confirm action: {action_id}",
                    approved=approved
                )

                if not current_state:
                    flow.error("Session not found")
                    return {
                        "answer": "Session not found.",
                        "session_id": session_id,
                        "trace_id": trace_id,
                        "error": "session_not_found",
                    }

                pending_action_id = current_state.get("pending_action_id")
                if not pending_action_id or pending_action_id != action_id:
                    flow.error("Invalid action_id")
                    return {
                        "answer": "Invalid action_id for this action.",
                        "session_id": session_id,
                        "trace_id": trace_id,
                        "error": "invalid_action_id",
                    }

                config = {
                    "configurable": {
                        "thread_id": session_id,
                    }
                }

                # Resume workflow with user confirmation
                user_response = {
                    "approved": approved,
                    "reason": reason,
                    "action_id": action_id,
                }

                try:
                    # Resume from interrupt
                    result = self.graph.invoke(
                        Command(resume=user_response),
                        config
                    )

                    flow.response(
                        result.get("final_answer", ""),
                        status="success" if approved else "rejected"
                    )

                    return {
                        "answer": result.get("final_answer", ""),
                        "session_id": session_id,
                        "trace_id": result.get("trace_id", trace_id),
                        "action_result": result.get("action_result"),
                        "total_latency_ms": result.get("total_latency_ms"),
                    }

                except Exception as e:
                    flow.error("Action confirmation error", exception=e)
                    return {
                        "answer": "An error occurred while processing the action.",
                        "session_id": session_id,
                        "trace_id": trace_id,
                        "error": str(e),
                    }

    def get_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current state for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Current state or None
        """
        config = {
            "configurable": {
                "thread_id": session_id,
            }
        }
        
        try:
            snapshot = self.graph.get_state(config)
            return snapshot.values if snapshot else None
        except Exception:
            return None


# -----------------------------------------------------------------------------
# SINGLETON
# -----------------------------------------------------------------------------

_copilot: Optional[StatementCopilot] = None


def get_copilot() -> StatementCopilot:
    """Get or create copilot singleton"""
    global _copilot
    if _copilot is None:
        _copilot = StatementCopilot()
    return _copilot


# -----------------------------------------------------------------------------
# CONVENIENCE FUNCTION
# -----------------------------------------------------------------------------

def chat(message: str, **kwargs) -> Dict[str, Any]:
    """
    Quick chat function.
    
    Args:
        message: User message
        **kwargs: Additional arguments for chat()
        
    Returns:
        Response dict
    """
    return get_copilot().chat(message, **kwargs)









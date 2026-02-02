"""
Statement Copilot - Self-RAG Search Subgraph
=============================================
Implements a Self-Reflective RAG pattern for intelligent search with:
- Retrieve: Execute search using ProfessionalSearchEngine
- Grade: LLM-based evaluation of result quality
- Transform: Query rewriting for retry attempts

LangGraph 2025-2026 compatible with subgraph state mapping.
"""

import logging
import time
from typing import Dict, Any, List, Literal, Optional, TypedDict, Annotated
from datetime import datetime

from langgraph.graph import StateGraph, START, END

from ..config import settings
from ..core import get_llm_client
from .search_agent import get_professional_search_agent

logger = logging.getLogger(__name__)


# =============================================================================
# SUBGRAPH STATE
# =============================================================================

class SearchSubgraphState(TypedDict, total=False):
    """
    State for the Self-RAG Search Subgraph.
    
    This state is isolated from the parent graph and uses explicit mapping
    for input/output (LangGraph 2025-2026 best practice).
    """
    # Input (mapped from parent OrchestratorState)
    original_query: str
    tenant_id: str
    constraints: Dict[str, Any]
    
    # Internal Loop State
    current_query: str
    attempt: int
    max_attempts: int
    
    # Search Results
    results: List[Dict[str, Any]]
    total_found: int
    
    # Reflection State
    result_quality: Literal["good", "empty", "irrelevant", "pending"]
    critique: str
    refined_query: str
    
    # Metadata
    search_time_ms: int
    all_attempts: List[Dict[str, Any]]  # Log of all attempts for debugging


# =============================================================================
# PROMPTS
# =============================================================================

GRADE_RESULTS_PROMPT = """You are a search quality evaluator for a financial transaction search system.

## User's Original Query
{original_query}

## Current Search Query
{current_query}

## Search Results
{results_summary}

## Task
Evaluate whether the search results satisfy the user's intent.

Consider:
1. **Relevance**: Do the results actually match what the user asked for?
2. **Completeness**: If user asked for specific merchant/category, did we find it?
3. **Zero Results**: If no results, is it likely due to a typo or overly specific query?

Respond with a JSON object:
{{
    "quality": "good" | "empty" | "irrelevant",
    "critique": "Brief explanation of the issue if not good",
    "suggested_fix": "How to improve the query if not good"
}}
"""

TRANSFORM_QUERY_PROMPT = """You are a search query optimizer for a financial transaction search system.

## Original User Query
{original_query}

## Failed Query
{current_query}

## Current Constraints
{current_constraints}

## Available Taxonomy (Valid Categories)
{taxonomy_context}

## Critique
{critique}

## Task
Generate an improved search query and optionally adjust constraints (filters).

Strategies:
1. **Fix Typos/Keywords**: 'Netflux' -> 'Netflix'.
2. **Broaden Search**: If specific strict constraints (like category='Food') might be blocking results for a query like 'Gas Station', REMOVE that constraint.
3. **Change Category**: If the current category is clearly wrong (e.g. 'Food' for 'Shell'), SWITCH to the valid category ID from the taxonomy (e.g. 'transport', 'fuel').
4. **Remove Date Filters**: If searching 'all time', remove date constraints.
5. **Synonyms**: Add financial synonyms.

Respond with a JSON object:
{{
    "refined_query": "The improved search query",
    "updated_constraints": {{ "categories": ["valid_category_id"], "subcategories": ["valid_subcat_id"] }} or null to keep same,
    "reasoning": "Why this should work better"
}}
"""


# =============================================================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUT
# =============================================================================

from pydantic import BaseModel, Field


class GradeResult(BaseModel):
    """Result of grading search results."""
    quality: Literal["good", "empty", "irrelevant"] = Field(
        description="Quality assessment of search results"
    )
    critique: str = Field(
        description="Brief explanation of the issue if not good"
    )
    suggested_fix: str = Field(
        default="",
        description="How to improve the query if quality is not good"
    )


class TransformResult(BaseModel):
    """Result of query transformation."""
    refined_query: str = Field(
        description="The improved search query"
    )
    updated_constraints: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional updated constraints (e.g. set category to empty list to remove filter)"
    )
    reasoning: str = Field(
        description="Why this query should work better"
    )


# =============================================================================
# NODE IMPLEMENTATIONS
# =============================================================================

def retrieve_node(state: SearchSubgraphState) -> SearchSubgraphState:
    """
    Execute search using the current query.
    
    This node wraps the existing ProfessionalSearchEngine for compatibility.
    """
    start_time = time.time()
    
    current_query = state.get("current_query", state.get("original_query", ""))
    tenant_id = state.get("tenant_id", settings.default_tenant_id)
    constraints = state.get("constraints", {})
    attempt = state.get("attempt", 0) + 1
    max_attempts = state.get("max_attempts", 3)
    
    # ═══════════════════════════════════════════════════════════════════════
    # ENHANCED DEBUG LOGGING: Self-RAG Subgraph Node
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("═" * 80)
    logger.info("[SELF-RAG SUBGRAPH] Node: RETRIEVE")
    logger.info("─" * 80)
    logger.info("  Graph: Self-RAG Search Subgraph (LangGraph)")
    logger.info("  Current Query: %s", current_query)
    logger.info("  Attempt: %d / %d", attempt, max_attempts)
    logger.info("  Constraints: %s", constraints if constraints else "(none)")
    logger.info("═" * 80)
    
    try:
        # Get the search agent and execute search
        agent = get_professional_search_agent()
        result = agent.engine.search(
            query=current_query,
            tenant_id=tenant_id,
            top_k=settings.max_vector_results,
            use_llm_rerank=settings.enable_search_llm,
            overrides=constraints,  # Pass constraints as overrides for adaptive filtering
        )
        
        # Convert matches to dict format
        matches = [
            {
                "tx_id": m.tx_id,
                "score": m.score,
                "date_time": m.date_time.isoformat() if m.date_time else None,
                "amount": m.amount,
                "merchant_norm": m.merchant_norm,
                "description": m.description,
                "category": m.category,
                "match_reason": m.match_reason,
            }
            for m in result.matches
        ]
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Log this attempt
        attempt_log = {
            "attempt": attempt,
            "query": current_query,
            "constraints": constraints,  # Log constraints used
            "found": len(matches),
            "latency_ms": latency_ms,
        }
        all_attempts = state.get("all_attempts", []) + [attempt_log]
        
        logger.info(
            f"[Search Subgraph] Retrieve complete: found={len(matches)}, latency={latency_ms}ms"
        )
        
        return {
            **state,
            "current_query": current_query,
            "attempt": attempt,
            "results": matches,
            "total_found": result.total_found,
            "result_quality": "pending",  # Will be evaluated by grade_node
            "search_time_ms": latency_ms,
            "all_attempts": all_attempts,
        }
        
    except Exception as e:
        logger.error(f"[Search Subgraph] Retrieve error: {e}")
        return {
            **state,
            "attempt": attempt,
            "results": [],
            "total_found": 0,
            "result_quality": "empty",
            "critique": f"Search error: {str(e)}",
            "all_attempts": state.get("all_attempts", []) + [{
                "attempt": attempt,
                "query": current_query,
                "error": str(e),
            }],
        }


def grade_results_node(state: SearchSubgraphState) -> SearchSubgraphState:
    """
    Evaluate the quality of search results using LLM reflection.
    
    This is the "Reflection" step that determines if results are acceptable
    or if we need to retry with a different query.
    """
    results = state.get("results", [])
    original_query = state.get("original_query", "")
    current_query = state.get("current_query", "")
    constraints = state.get("constraints", {})
    
    # Fast path: if we have good results, skip LLM call
    if results and len(results) >= 1:
        # Check if results seem relevant (heuristic)
        top_result = results[0]
        score = top_result.get("score", 0)
        
        # High confidence match - skip LLM evaluation
        if score > 0.8:
            logger.info(f"[Search Subgraph] Grade: Fast path - high score ({score:.2f})")
            return {
                **state,
                "result_quality": "good",
                "critique": "",
            }
    
    # Zero results - obvious empty case
    if not results:
        logger.info("[Search Subgraph] Grade: Empty results")
        critique_msg = "No results found."
        if constraints:
            critique_msg += f" Constraints applied: {constraints}. Consider relaxing overly strict filters or checking if category is correct."
        
        return {
            **state,
            "result_quality": "empty",
            "critique": critique_msg,
        }
    
    # Use LLM to evaluate result quality
    try:
        llm = get_llm_client()
        
        # Build results summary for prompt
        results_summary = f"Found {len(results)} results:\n"
        for i, r in enumerate(results[:5]):
            results_summary += f"  {i+1}. {r.get('merchant_norm', 'N/A')} - {r.get('amount', 0):.2f} TRY ({r.get('category', 'N/A')})\n"
        
        prompt = GRADE_RESULTS_PROMPT.format(
            original_query=original_query,
            current_query=current_query,
            results_summary=results_summary,
        )
        
        grade: GradeResult = llm.complete_structured(
            prompt=prompt,
            response_model=GradeResult,
            system="You are a search quality evaluator. Be strict but fair.",
            temperature=0.0,
        )
        
        logger.info(
            f"[Search Subgraph] Grade: quality={grade.quality}, critique='{grade.critique[:50]}...'"
        )
        
        return {
            **state,
            "result_quality": grade.quality,
            "critique": grade.critique,
        }
        
    except Exception as e:
        logger.warning(f"[Search Subgraph] Grade LLM error: {e}, assuming good")
        # On error, assume results are acceptable to avoid infinite loops
        return {
            **state,
            "result_quality": "good",
            "critique": "",
        }

def _get_simplified_taxonomy() -> str:
    """Helper to get simplified taxonomy string for prompt context."""
    try:
        agent = get_professional_search_agent()
        taxonomy = agent.taxonomy.get("categories", {})
        
        lines = []
        for cat_id, cat_data in taxonomy.items():
            subcats = list(cat_data.get("subcategories", {}).keys())
            lines.append(f"- {cat_id}: {subcats}")
            
        return "\n".join(lines)
    except Exception:
        return "Taxonomy not available."

def transform_query_node(state: SearchSubgraphState) -> SearchSubgraphState:
    """
    Transform the query based on the critique.
    
    This is the "React" step that generates a better query for retry.
    Now supports CONSTRAINT RELAXATION with TAXONOMY AWARENESS.
    """
    original_query = state.get("original_query", "")
    current_query = state.get("current_query", "")
    constraints = state.get("constraints", {})
    critique = state.get("critique", "")
    
    logger.info(f"[Search Subgraph] Transform: critique='{critique[:50]}...'")
    
    try:
        llm = get_llm_client()
        
        # Get simplified taxonomy context
        taxonomy_context = _get_simplified_taxonomy()
        
        prompt = TRANSFORM_QUERY_PROMPT.format(
            original_query=original_query,
            current_query=current_query,
            current_constraints=str(constraints) if constraints else "(none)",
            taxonomy_context=taxonomy_context,
            critique=critique,
        )
        
        transform: TransformResult = llm.complete_structured(
            prompt=prompt,
            response_model=TransformResult,
            system="You are a search query optimizer. Generate concise, effective queries and manage constraints.",
            temperature=0.3,
        )
        
        new_constraints = constraints
        if transform.updated_constraints is not None:
             # Merge/Update constraints
             logger.info(f"[Search Subgraph] Constraint Update: {constraints} -> {transform.updated_constraints}")
             new_constraints = transform.updated_constraints
        
        logger.info(
            f"[Search Subgraph] Transform: refined='{transform.refined_query}', reason='{transform.reasoning[:50]}...'"
        )
        
        return {
            **state,
            "current_query": transform.refined_query,
            "constraints": new_constraints,
            "refined_query": transform.refined_query,
        }
        
    except Exception as e:
        logger.warning(f"[Search Subgraph] Transform LLM error: {e}")
        # Fallback: just use original query (will likely fail again, but max_attempts will stop loop)
        return {
            **state,
            "current_query": original_query,
            "refined_query": original_query,
        }


# =============================================================================
# CONDITIONAL ROUTING
# =============================================================================

def decide_next_step(state: SearchSubgraphState) -> Literal["end", "transform"]:
    """
    Decide whether to end or retry with a transformed query.
    """
    quality = state.get("result_quality", "pending")
    attempt = state.get("attempt", 0)
    max_attempts = state.get("max_attempts", 3)
    
    # Good results - we're done
    if quality == "good":
        logger.info("[Search Subgraph] Decision: END (good results)")
        return "end"
    
    # Max attempts reached - return best effort
    if attempt >= max_attempts:
        logger.info(f"[Search Subgraph] Decision: END (max attempts {max_attempts} reached)")
        return "end"
    
    # Need to retry
    logger.info(f"[Search Subgraph] Decision: TRANSFORM (attempt {attempt}/{max_attempts})")
    return "transform"


# =============================================================================
# BUILD SUBGRAPH
# =============================================================================

def build_search_subgraph() -> StateGraph:
    """
    Build the Self-RAG Search Subgraph.
    
    Flow:
        START -> retrieve -> grade -> [decide] -> end
                                   |
                                   v
                             transform -> retrieve (loop)
    """
    workflow = StateGraph(SearchSubgraphState)
    
    # Add nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_results_node)
    workflow.add_node("transform", transform_query_node)
    
    # Build edges
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade")
    
    # Conditional edge after grading
    workflow.add_conditional_edges(
        "grade",
        decide_next_step,
        {
            "end": END,
            "transform": "transform",
        }
    )
    
    # Transform loops back to retrieve
    workflow.add_edge("transform", "retrieve")
    
    return workflow


# =============================================================================
# PUBLIC API
# =============================================================================

_compiled_search_subgraph = None


def get_search_subgraph():
    """Get or create the compiled search subgraph singleton."""
    global _compiled_search_subgraph
    if _compiled_search_subgraph is None:
        workflow = build_search_subgraph()
        _compiled_search_subgraph = workflow.compile()
        logger.info("Self-RAG Search Subgraph compiled")
    return _compiled_search_subgraph


def run_self_rag_search(
    query: str,
    tenant_id: str,
    constraints: Optional[Dict[str, Any]] = None,
    max_attempts: int = 3,
) -> Dict[str, Any]:
    """
    Execute Self-RAG search with automatic retry and query refinement.
    
    Args:
        query: User's search query
        tenant_id: Tenant ID for data isolation
        constraints: Optional constraints from orchestrator
        max_attempts: Maximum retry attempts (default: 3)
        
    Returns:
        Dict with results, attempts info, and final quality
    """
    subgraph = get_search_subgraph()
    
    # Initialize subgraph state
    initial_state: SearchSubgraphState = {
        "original_query": query,
        "current_query": query,
        "tenant_id": tenant_id,
        "constraints": constraints or {},
        "attempt": 0,
        "max_attempts": max_attempts,
        "results": [],
        "total_found": 0,
        "result_quality": "pending",
        "critique": "",
        "refined_query": "",
        "search_time_ms": 0,
        "all_attempts": [],
    }
    
    # Execute subgraph
    start_time = time.time()
    final_state = subgraph.invoke(initial_state)
    total_time_ms = int((time.time() - start_time) * 1000)
    
    logger.info(
        f"[Search Subgraph] Complete: attempts={final_state.get('attempt')}, "
        f"quality={final_state.get('result_quality')}, "
        f"found={final_state.get('total_found')}, "
        f"total_time={total_time_ms}ms"
    )
    
    return {
        "results": final_state.get("results", []),
        "total_found": final_state.get("total_found", 0),
        "final_quality": final_state.get("result_quality"),
        "attempts": final_state.get("attempt", 1),
        "all_attempts": final_state.get("all_attempts", []),
        "final_query": final_state.get("current_query"),
        "critique": final_state.get("critique"),
        "total_time_ms": total_time_ms,
    }

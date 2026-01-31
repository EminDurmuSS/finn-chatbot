"""
Statement Copilot - Professional Search Agent
=============================================
Enhanced search agent using multi-stage retrieval with query understanding.

Key improvements:
1. Advanced query understanding with entity extraction
2. Multi-source retrieval (SQL + Vector + Merchant Dictionary)
3. Cross-encoder reranking
4. Evidence-based result assembly
"""

import logging
import time
import json
from typing import Dict, Any, Optional, List
from datetime import datetime

from ..config import settings
from ..core import (
    OrchestratorState,
    SearchResult as LegacySearchResult,
    TransactionMatch,
    create_tool_call_record,
    get_llm_client,
    get_db,
    get_vector_store,
)
from ..core.search_engine import (
    ProfessionalSearchEngine,
    QueryUnderstandingEngine,
    SearchResult,
    SearchMatch,
    SearchIntent,
    SearchStrategy,
    ExtractedEntities,
)
from ..log_context import clip_text, format_kv, format_list


logger = logging.getLogger(__name__)


# =============================================================================
# TAXONOMY LOADER
# =============================================================================

def load_taxonomy() -> Dict[str, Any]:
    """Load financial category taxonomy"""
    import json
    from pathlib import Path
    
    # Try multiple locations
    possible_paths = [
        # Primary: category_taxonomy_v1.json (current file naming)
        Path(__file__).parent.parent / "data" / "category_taxonomy_v1.json",
        Path(__file__).parent.parent.parent / "data" / "category_taxonomy_v1.json",
        Path.cwd() / "data" / "category_taxonomy_v1.json",
        # Fallback: taxonomy.json (for backward compatibility)
        Path(__file__).parent.parent / "data" / "taxonomy.json",
        Path(__file__).parent.parent.parent / "data" / "taxonomy.json",
        Path.cwd() / "data" / "taxonomy.json",
    ]
    
    for path in possible_paths:
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    categories = data.get("categories", {})
                    logger.info(f"Taxonomy loaded from {path}: {len(categories)} categories")
                    return data
            except Exception as e:
                logger.warning(f"Failed to load taxonomy from {path}: {e}")
    
    logger.warning("Taxonomy not found, using empty taxonomy")
    return {}


# =============================================================================
# PROFESSIONAL SEARCH AGENT
# =============================================================================

class ProfessionalSearchAgent:
    """
    Professional search agent with multi-stage retrieval.
    
    Features:
    1. Advanced query understanding (NLU layer)
    2. Multi-source retrieval (SQL + Vector + Merchant Dict)
    3. Intelligent reranking
    4. Evidence assembly for transparency
    """
    
    def __init__(self):
        self.llm = get_llm_client()
        self.db = get_db()
        self.vector_store = get_vector_store()
        self.taxonomy = load_taxonomy()
        
        # Initialize professional search engine
        self.engine = ProfessionalSearchEngine(
            db=self.db,
            vector_store=self.vector_store,
            llm_client=self.llm if settings.enable_search_llm else None,
            taxonomy=self.taxonomy
        )
        
        logger.info("ProfessionalSearchAgent initialized")
    
    def search(self, state: OrchestratorState) -> OrchestratorState:
        """
        Execute professional search based on state.
        
        Args:
            state: Current orchestrator state
            
        Returns:
            Updated state with vector_result
        """
        user_message = state.get("user_message", "")
        constraints = state.get("constraints", {})
        tenant_id = state.get("tenant_id", settings.default_tenant_id)
        
        start_time = time.time()
        
        try:
            # Merge constraints into query context
            query = self._build_search_query(user_message, constraints)
            
            # Execute professional search
            result = self.engine.search(
                query=query,
                tenant_id=tenant_id,
                top_k=settings.max_vector_results,
                use_llm_rerank=settings.enable_search_llm
            )
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Convert to legacy format for compatibility
            legacy_result = self._to_legacy_format(result)
            
            # Store search query details
            state["search_query"] = {
                "query_text": query,
                "expanded_query": result.query_understanding.expanded_query,
                "intent": result.query_understanding.intent.value,
                "strategy": result.query_understanding.strategy.value,
                "confidence": result.query_understanding.confidence,
                "entities": self._entities_to_dict(result.query_understanding.entities),
                "search_terms": result.query_understanding.search_terms,
            }
            
            # Store result
            state["vector_result"] = legacy_result.model_dump(mode="json")
            
            # Add detailed evidence for transparency
            state["search_evidence"] = {
                "sources_used": result.sources_used,
                "filters_applied": result.filters_applied,
                "reasoning": result.query_understanding.reasoning,
                "search_time_ms": result.search_time_ms,
            }
            
            # Add tool call record
            tool_call = create_tool_call_record(
                state=state,
                node="search_agent",
                tool_name="professional_search",
                model_name=None,
                input_data={"query": query, "intent": result.query_understanding.intent.value},
                output_data={"found": len(result.matches), "sources": result.sources_used},
                latency_ms=latency_ms,
                success=True
            )
            state["tool_calls"] = [tool_call]

            logger.info(
                f"Search completed: query='{query[:50]}...' "
                f"intent={result.query_understanding.intent.value} "
                f"found={len(result.matches)} "
                f"sources={result.sources_used} "
                f"latency={latency_ms}ms"
            )
            
            logger.debug(
                "Search details: strategy=%s confidence=%.2f entities=%s",
                result.query_understanding.strategy.value,
                result.query_understanding.confidence,
                format_kv(self._entities_to_dict(result.query_understanding.entities), max_items=6)
            )
            
            return state
            
        except Exception as e:
            logger.error(f"Search Agent error: {e}", exc_info=True)
            state["search_error"] = str(e)
            state["errors"] = [f"Search error: {e}"]
            return state
    
    def search_with_self_rag(self, state: OrchestratorState) -> OrchestratorState:
        """
        Execute search using Self-RAG pattern with automatic retry and query refinement.
        
        This method uses a LangGraph subgraph that implements:
        1. Retrieve: Execute search
        2. Grade: LLM evaluates result quality
        3. Transform: LLM refines query if results are poor
        4. Retry: Loop back to retrieve with refined query
        
        Args:
            state: Current orchestrator state
            
        Returns:
            Updated state with vector_result and self_rag_info
        """
        from .search_graph import run_self_rag_search
        
        user_message = state.get("user_message", "")
        constraints = state.get("constraints", {})
        tenant_id = state.get("tenant_id", settings.default_tenant_id)
        
        # Build search query from message and constraints
        query = self._build_search_query(user_message, constraints)
        
        logger.info(f"Starting Self-RAG search: query='{query[:50]}...'")
        
        try:
            # Execute Self-RAG search
            result = run_self_rag_search(
                query=query,
                tenant_id=tenant_id,
                constraints=constraints,
                max_attempts=3,
            )
            
            # Convert to legacy format for compatibility
            matches = result.get("results", [])
            legacy_matches = [
                TransactionMatch(
                    tx_id=m.get("tx_id", ""),
                    score=m.get("score", 0.0),
                    date_time=datetime.fromisoformat(m["date_time"]) if m.get("date_time") else None,
                    amount=m.get("amount"),
                    merchant_norm=m.get("merchant_norm"),
                    description=m.get("description"),
                    category=m.get("category"),
                )
                for m in matches
            ]
            
            legacy_result = LegacySearchResult(
                matches=legacy_matches,
                total_found=result.get("total_found", 0),
                query_text=result.get("final_query", query),
                search_type="self_rag",
            )
            
            # Store search query details
            state["search_query"] = {
                "query_text": query,
                "final_query": result.get("final_query"),
                "intent": "self_rag",
                "strategy": "self_rag",
                "confidence": 1.0 if result.get("final_quality") == "good" else 0.5,
                "attempts": result.get("attempts", 1),
            }
            
            # Store result
            state["vector_result"] = legacy_result.model_dump(mode="json")
            
            # Add Self-RAG specific evidence
            state["search_evidence"] = {
                "sources_used": ["self_rag_subgraph"],
                "filters_applied": constraints,
                "reasoning": result.get("critique", ""),
                "search_time_ms": result.get("total_time_ms", 0),
                "self_rag_info": {
                    "attempts": result.get("attempts"),
                    "all_attempts": result.get("all_attempts", []),
                    "final_quality": result.get("final_quality"),
                    "final_query": result.get("final_query"),
                },
            }
            
            # Add tool call record
            tool_call = create_tool_call_record(
                state=state,
                node="search_agent",
                tool_name="self_rag_search",
                model_name=None,
                input_data={"query": query, "max_attempts": 3},
                output_data={
                    "found": result.get("total_found", 0),
                    "attempts": result.get("attempts", 1),
                    "quality": result.get("final_quality"),
                },
                latency_ms=result.get("total_time_ms", 0),
                success=True
            )
            state["tool_calls"] = [tool_call]
            
            logger.info(
                f"Self-RAG search completed: "
                f"attempts={result.get('attempts')}, "
                f"quality={result.get('final_quality')}, "
                f"found={result.get('total_found')}, "
                f"time={result.get('total_time_ms')}ms"
            )
            
            return state
            
        except Exception as e:
            logger.error(f"Self-RAG Search error: {e}", exc_info=True)
            # Fallback to standard search
            logger.info("Falling back to standard search")
            return self.search(state)
    
    def _build_search_query(self, user_message: str, constraints: Dict[str, Any]) -> str:
        """
        Build search query for the search engine.
        
        FIXED: Return original query as-is. Do NOT augment with metadata.
        
        Previously this method appended "merchant: X", "direction: Y" etc. to the query,
        which polluted keyword extraction and caused searches to fail because
        words like 'merchant' and 'direction' were being treated as search terms.
        
        Constraints are already passed separately to the search engine via the
        constraints parameter in the search() method.
        """
        # Return original query - constraints are handled separately
        return user_message
    
    def _to_legacy_format(self, result: SearchResult) -> LegacySearchResult:
        """Convert professional search result to legacy format"""
        matches = []
        
        for match in result.matches:
            matches.append(TransactionMatch(
                tx_id=match.tx_id,
                score=match.score,
                date_time=match.date_time,
                amount=match.amount,
                merchant_norm=match.merchant_norm,
                description=match.description,
                category=match.category,
            ))
        
        return LegacySearchResult(
            matches=matches,
            total_found=result.total_found,
            query_text=result.query_understanding.expanded_query,
            search_type="hybrid"
        )
    
    def _entities_to_dict(self, entities: ExtractedEntities) -> Dict[str, Any]:
        """Convert ExtractedEntities to dict"""
        return {
            "merchants": entities.merchants,
            "categories": entities.categories,
            "subcategories": entities.subcategories,
            "amounts": entities.amounts,
            "date_range": (
                [entities.date_range[0].isoformat(), entities.date_range[1].isoformat()]
                if entities.date_range else None
            ),
            "direction": entities.direction,
            "keywords": entities.keywords,
        }
    
    def find_similar(
        self,
        tx_id: str,
        tenant_id: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find transactions similar to a given transaction.
        
        Args:
            tx_id: Reference transaction ID
            tenant_id: Tenant ID
            top_k: Number of results
            
        Returns:
            Similar transactions
        """
        result = self.engine.search_similar(tx_id, tenant_id, top_k)
        
        return [
            {
                "tx_id": m.tx_id,
                "score": m.score,
                "date_time": m.date_time.isoformat() if m.date_time else None,
                "amount": m.amount,
                "merchant_norm": m.merchant_norm,
                "description": m.description,
                "category": m.category,
            }
            for m in result.matches
        ]


# =============================================================================
# QUICK SEARCH FUNCTIONS
# =============================================================================

def quick_search(
    query: str,
    tenant_id: str,
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Quick search without full agent workflow.
    
    Args:
        query: Search query
        tenant_id: Tenant ID
        top_k: Number of results
        
    Returns:
        List of matching transactions
    """
    agent = get_professional_search_agent()
    result = agent.engine.search(query, tenant_id, top_k)
    
    return [
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


def analyze_query(query: str) -> Dict[str, Any]:
    """
    Analyze a search query without executing search.
    Useful for debugging query understanding.
    
    Args:
        query: Search query to analyze
        
    Returns:
        Query understanding details
    """
    agent = get_professional_search_agent()
    understanding = agent.engine.query_understanding.understand(query)
    
    return {
        "original_query": understanding.original_query,
        "normalized_query": understanding.normalized_query,
        "intent": understanding.intent.value,
        "confidence": understanding.confidence,
        "strategy": understanding.strategy.value,
        "expanded_query": understanding.expanded_query,
        "search_terms": understanding.search_terms,
        "entities": {
            "merchants": understanding.entities.merchants,
            "categories": understanding.entities.categories,
            "amounts": understanding.entities.amounts,
            "date_range": (
                [understanding.entities.date_range[0].isoformat(), 
                 understanding.entities.date_range[1].isoformat()]
                if understanding.entities.date_range else None
            ),
            "direction": understanding.entities.direction,
            "keywords": understanding.entities.keywords,
        },
        "reasoning": understanding.reasoning,
    }


# =============================================================================
# SINGLETON
# =============================================================================

_professional_search_agent: Optional[ProfessionalSearchAgent] = None


def get_professional_search_agent() -> ProfessionalSearchAgent:
    """Get or create professional search agent singleton"""
    global _professional_search_agent
    if _professional_search_agent is None:
        _professional_search_agent = ProfessionalSearchAgent()
    return _professional_search_agent


# Alias for backward compatibility
def get_search_agent() -> ProfessionalSearchAgent:
    """Get search agent (alias for get_professional_search_agent)"""
    return get_professional_search_agent()


# =============================================================================
# LEGACY SEARCH AGENT (Deprecated - for backward compatibility)
# =============================================================================

class SearchAgent:
    """
    Legacy search agent - wraps ProfessionalSearchAgent for compatibility.
    
    DEPRECATED: Use ProfessionalSearchAgent directly.
    """
    
    def __init__(self):
        self._professional = get_professional_search_agent()
        logger.warning("SearchAgent is deprecated. Use ProfessionalSearchAgent instead.")
    
    def search(self, state: OrchestratorState) -> OrchestratorState:
        """Delegate to professional agent"""
        return self._professional.search(state)
    
    def find_similar(self, tx_id: str, tenant_id: str, top_k: int = 5):
        """Delegate to professional agent"""
        return self._professional.find_similar(tx_id, tenant_id, top_k)
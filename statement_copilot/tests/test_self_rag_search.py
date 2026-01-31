"""
Statement Copilot - Self-RAG Search Subgraph Tests
===================================================
Comprehensive test suite for the Self-RAG search implementation.

Tests cover:
1. Subgraph state management
2. Individual node logic (retrieve, grade, transform)
3. Conditional edge routing
4. Max retry handling
5. End-to-end flow
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
from datetime import datetime

# Import the modules we're testing
from statement_copilot.agents.search_graph import (
    SearchSubgraphState,
    retrieve_node,
    grade_results_node,
    transform_query_node,
    decide_next_step,
    build_search_subgraph,
    run_self_rag_search,
    GradeResult,
    TransformResult,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_search_result():
    """Mock successful search result with good matches."""
    return {
        "matches": [
            {
                "tx_id": "tx_001",
                "score": 0.95,
                "date_time": "2026-01-15T10:30:00",
                "amount": 49.99,
                "merchant_norm": "NETFLIX",
                "description": "Netflix subscription",
                "category": "entertainment",
                "match_reason": "Exact merchant match",
            },
            {
                "tx_id": "tx_002",
                "score": 0.85,
                "date_time": "2026-01-10T14:20:00",
                "amount": 12.99,
                "merchant_norm": "NETFLIX",
                "description": "Netflix monthly",
                "category": "entertainment",
                "match_reason": "Semantic match",
            },
        ],
        "total_found": 2,
    }


@pytest.fixture
def mock_empty_result():
    """Mock empty search result."""
    return {
        "matches": [],
        "total_found": 0,
    }


@pytest.fixture
def base_state() -> SearchSubgraphState:
    """Base state for testing."""
    return SearchSubgraphState(
        original_query="Find my Netflix payment",
        current_query="Find my Netflix payment",
        tenant_id="test_tenant",
        constraints={},
        attempt=0,
        max_attempts=3,
        results=[],
        total_found=0,
        result_quality="pending",
        critique="",
        refined_query="",
        search_time_ms=0,
        all_attempts=[],
    )


# =============================================================================
# TEST: State Management
# =============================================================================

class TestStateManagement:
    """Tests for SearchSubgraphState structure and initialization."""
    
    def test_state_has_all_required_fields(self, base_state):
        """Verify state has all required fields."""
        required_fields = [
            "original_query",
            "current_query",
            "tenant_id",
            "constraints",
            "attempt",
            "max_attempts",
            "results",
            "total_found",
            "result_quality",
            "critique",
            "refined_query",
            "search_time_ms",
            "all_attempts",
        ]
        for field in required_fields:
            assert field in base_state, f"Missing field: {field}"
    
    def test_state_default_values(self, base_state):
        """Verify default values are correct."""
        assert base_state["attempt"] == 0
        assert base_state["max_attempts"] == 3
        assert base_state["result_quality"] == "pending"
        assert base_state["results"] == []
        assert base_state["all_attempts"] == []
    
    def test_state_preserves_original_query(self, base_state):
        """Verify original query is never modified."""
        original = base_state["original_query"]
        base_state["current_query"] = "modified query"
        assert base_state["original_query"] == original


# =============================================================================
# TEST: Retrieve Node
# =============================================================================

class TestRetrieveNode:
    """Tests for the retrieve_node function."""
    
    @patch('statement_copilot.agents.search_graph.get_professional_search_agent')
    def test_retrieve_increments_attempt(self, mock_get_agent, base_state):
        """Verify attempt counter is incremented."""
        # Setup mock
        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.matches = []
        mock_result.total_found = 0
        mock_agent.engine.search.return_value = mock_result
        mock_get_agent.return_value = mock_agent
        
        # Execute
        result_state = retrieve_node(base_state)
        
        # Verify
        assert result_state["attempt"] == 1
    
    @patch('statement_copilot.agents.search_graph.get_professional_search_agent')
    def test_retrieve_sets_pending_quality(self, mock_get_agent, base_state):
        """Verify result_quality is set to pending after retrieve."""
        # Setup mock
        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.matches = []
        mock_result.total_found = 0
        mock_agent.engine.search.return_value = mock_result
        mock_get_agent.return_value = mock_agent
        
        # Execute
        result_state = retrieve_node(base_state)
        
        # Verify - should be pending for grade node to evaluate
        assert result_state["result_quality"] == "pending"
    
    @patch('statement_copilot.agents.search_graph.get_professional_search_agent')
    def test_retrieve_logs_attempt(self, mock_get_agent, base_state):
        """Verify each attempt is logged in all_attempts."""
        # Setup mock
        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.matches = []
        mock_result.total_found = 0
        mock_agent.engine.search.return_value = mock_result
        mock_get_agent.return_value = mock_agent
        
        # Execute
        result_state = retrieve_node(base_state)
        
        # Verify
        assert len(result_state["all_attempts"]) == 1
        assert result_state["all_attempts"][0]["attempt"] == 1
        assert "query" in result_state["all_attempts"][0]
    
    @patch('statement_copilot.agents.search_graph.get_professional_search_agent')
    def test_retrieve_handles_error(self, mock_get_agent, base_state):
        """Verify error handling in retrieve node."""
        # Setup mock to raise error
        mock_get_agent.side_effect = Exception("Search engine unavailable")
        
        # Execute
        result_state = retrieve_node(base_state)
        
        # Verify graceful degradation
        assert result_state["result_quality"] == "empty"
        assert "error" in result_state["all_attempts"][0]


# =============================================================================
# TEST: Grade Node
# =============================================================================

class TestGradeNode:
    """Tests for the grade_results_node function."""
    
    def test_grade_empty_results(self, base_state):
        """Empty results should be graded as 'empty'."""
        base_state["results"] = []
        
        result_state = grade_results_node(base_state)
        
        assert result_state["result_quality"] == "empty"
        assert "No results found" in result_state["critique"]
    
    def test_grade_high_score_fast_path(self, base_state, mock_search_result):
        """High-scoring results should bypass LLM evaluation."""
        base_state["results"] = mock_search_result["matches"]
        
        with patch('statement_copilot.agents.search_graph.get_llm_client') as mock_llm:
            result_state = grade_results_node(base_state)
            
            # Should NOT call LLM because score > 0.8
            mock_llm.return_value.complete_structured.assert_not_called()
            assert result_state["result_quality"] == "good"
    
    @patch('statement_copilot.agents.search_graph.get_llm_client')
    def test_grade_uses_llm_for_low_scores(self, mock_llm_client, base_state):
        """Low-scoring results should use LLM evaluation."""
        base_state["results"] = [
            {"tx_id": "tx_001", "score": 0.5, "merchant_norm": "UNKNOWN"}
        ]
        
        # Setup mock LLM response
        mock_llm = MagicMock()
        mock_llm.complete_structured.return_value = GradeResult(
            quality="irrelevant",
            critique="Results don't match user intent",
            suggested_fix="Try searching for streaming services"
        )
        mock_llm_client.return_value = mock_llm
        
        result_state = grade_results_node(base_state)
        
        # Should call LLM
        mock_llm.complete_structured.assert_called_once()
        assert result_state["result_quality"] == "irrelevant"
    
    @patch('statement_copilot.agents.search_graph.get_llm_client')
    def test_grade_handles_llm_error(self, mock_llm_client, base_state):
        """LLM errors should default to 'good' to prevent infinite loops."""
        base_state["results"] = [
            {"tx_id": "tx_001", "score": 0.5, "merchant_norm": "UNKNOWN"}
        ]
        
        # Setup mock LLM to raise error
        mock_llm = MagicMock()
        mock_llm.complete_structured.side_effect = Exception("LLM unavailable")
        mock_llm_client.return_value = mock_llm
        
        result_state = grade_results_node(base_state)
        
        # Should default to "good" on error
        assert result_state["result_quality"] == "good"


# =============================================================================
# TEST: Transform Node
# =============================================================================

class TestTransformNode:
    """Tests for the transform_query_node function."""
    
    @patch('statement_copilot.agents.search_graph.get_llm_client')
    def test_transform_generates_refined_query(self, mock_llm_client, base_state):
        """Transform should generate a refined query based on critique."""
        base_state["critique"] = "Query 'Netflux' appears to be a typo"
        
        # Setup mock LLM response
        mock_llm = MagicMock()
        mock_llm.complete_structured.return_value = TransformResult(
            refined_query="Netflix",
            reasoning="Fixed typo: Netflux -> Netflix"
        )
        mock_llm_client.return_value = mock_llm
        
        result_state = transform_query_node(base_state)
        
        assert result_state["current_query"] == "Netflix"
        assert result_state["refined_query"] == "Netflix"
    
    @patch('statement_copilot.agents.search_graph.get_llm_client')
    def test_transform_handles_llm_error(self, mock_llm_client, base_state):
        """LLM errors should fallback to original query."""
        base_state["original_query"] = "Find Netflix"
        base_state["current_query"] = "Netflux"
        base_state["critique"] = "Typo detected"
        
        # Setup mock LLM to raise error
        mock_llm = MagicMock()
        mock_llm.complete_structured.side_effect = Exception("LLM unavailable")
        mock_llm_client.return_value = mock_llm
        
        result_state = transform_query_node(base_state)
        
        # Should fallback to original query
        assert result_state["current_query"] == "Find Netflix"


# =============================================================================
# TEST: Conditional Routing
# =============================================================================

class TestConditionalRouting:
    """Tests for the decide_next_step function."""
    
    def test_route_to_end_on_good_quality(self, base_state):
        """Good results should route to end."""
        base_state["result_quality"] = "good"
        base_state["attempt"] = 1
        
        decision = decide_next_step(base_state)
        
        assert decision == "end"
    
    def test_route_to_transform_on_empty(self, base_state):
        """Empty results should route to transform."""
        base_state["result_quality"] = "empty"
        base_state["attempt"] = 1
        
        decision = decide_next_step(base_state)
        
        assert decision == "transform"
    
    def test_route_to_transform_on_irrelevant(self, base_state):
        """Irrelevant results should route to transform."""
        base_state["result_quality"] = "irrelevant"
        base_state["attempt"] = 1
        
        decision = decide_next_step(base_state)
        
        assert decision == "transform"
    
    def test_route_to_end_on_max_attempts(self, base_state):
        """Should route to end when max attempts reached."""
        base_state["result_quality"] = "empty"
        base_state["attempt"] = 3
        base_state["max_attempts"] = 3
        
        decision = decide_next_step(base_state)
        
        assert decision == "end"
    
    def test_route_respects_max_attempts_boundary(self, base_state):
        """Should allow retry if under max attempts."""
        base_state["result_quality"] = "empty"
        base_state["attempt"] = 2
        base_state["max_attempts"] = 3
        
        decision = decide_next_step(base_state)
        
        assert decision == "transform"


# =============================================================================
# TEST: Subgraph Structure
# =============================================================================

class TestSubgraphStructure:
    """Tests for the subgraph build and structure."""
    
    def test_subgraph_builds_successfully(self):
        """Subgraph should build without errors."""
        workflow = build_search_subgraph()
        
        assert workflow is not None
    
    def test_subgraph_has_required_nodes(self):
        """Subgraph should have all required nodes."""
        workflow = build_search_subgraph()
        
        # Check node names exist in the graph
        assert "retrieve" in workflow.nodes
        assert "grade" in workflow.nodes
        assert "transform" in workflow.nodes
    
    def test_subgraph_compiles(self):
        """Subgraph should compile without errors."""
        workflow = build_search_subgraph()
        compiled = workflow.compile()
        
        assert compiled is not None


# =============================================================================
# TEST: End-to-End Flow
# =============================================================================

class TestEndToEndFlow:
    """End-to-end tests for the Self-RAG search flow."""
    
    @patch('statement_copilot.agents.search_graph.get_professional_search_agent')
    def test_successful_first_attempt(self, mock_get_agent):
        """Test successful search on first attempt (no retry needed)."""
        # Setup mock
        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.matches = [
            MagicMock(
                tx_id="tx_001",
                score=0.95,  # High score = good quality
                date_time=datetime.now(),
                amount=49.99,
                merchant_norm="NETFLIX",
                description="Netflix",
                category="entertainment",
                match_reason="Exact match"
            )
        ]
        mock_result.total_found = 1
        mock_agent.engine.search.return_value = mock_result
        mock_get_agent.return_value = mock_agent
        
        # Execute
        result = run_self_rag_search(
            query="Netflix",
            tenant_id="test",
            constraints={},
            max_attempts=3
        )
        
        # Verify
        assert result["attempts"] == 1  # Only one attempt needed
        assert result["final_quality"] == "good"
        assert len(result["results"]) == 1
    
    @patch('statement_copilot.agents.search_graph.get_professional_search_agent')
    @patch('statement_copilot.agents.search_graph.get_llm_client')
    def test_retry_on_empty_results(self, mock_llm_client, mock_get_agent):
        """Test retry mechanism when first search returns empty."""
        # First call returns empty, second call returns results
        mock_agent = MagicMock()
        
        empty_result = MagicMock()
        empty_result.matches = []
        empty_result.total_found = 0
        
        success_result = MagicMock()
        success_result.matches = [
            MagicMock(
                tx_id="tx_001",
                score=0.9,
                date_time=datetime.now(),
                amount=49.99,
                merchant_norm="NETFLIX",
                description="Netflix",
                category="entertainment",
                match_reason="Match"
            )
        ]
        success_result.total_found = 1
        
        mock_agent.engine.search.side_effect = [empty_result, success_result]
        mock_get_agent.return_value = mock_agent
        
        # Setup LLM for transform
        mock_llm = MagicMock()
        mock_llm.complete_structured.return_value = TransformResult(
            refined_query="Netflix streaming service",
            reasoning="Broadened search"
        )
        mock_llm_client.return_value = mock_llm
        
        # Execute
        result = run_self_rag_search(
            query="Netflux",  # Typo
            tenant_id="test",
            constraints={},
            max_attempts=3
        )
        
        # Verify retry happened
        assert result["attempts"] == 2
        assert len(result["all_attempts"]) == 2
    
    @patch('statement_copilot.agents.search_graph.get_professional_search_agent')
    @patch('statement_copilot.agents.search_graph.get_llm_client')
    def test_max_attempts_exhausted(self, mock_llm_client, mock_get_agent):
        """Test that search stops after max attempts."""
        # Always return empty results
        mock_agent = MagicMock()
        empty_result = MagicMock()
        empty_result.matches = []
        empty_result.total_found = 0
        mock_agent.engine.search.return_value = empty_result
        mock_get_agent.return_value = mock_agent
        
        # Setup LLM for transform
        mock_llm = MagicMock()
        mock_llm.complete_structured.return_value = TransformResult(
            refined_query="alternative query",
            reasoning="Trying different approach"
        )
        mock_llm_client.return_value = mock_llm
        
        # Execute with max 3 attempts
        result = run_self_rag_search(
            query="nonexistent merchant xyz123",
            tenant_id="test",
            constraints={},
            max_attempts=3
        )
        
        # Verify max attempts respected
        assert result["attempts"] == 3
        assert result["total_found"] == 0


# =============================================================================
# TEST: Integration with Workflow
# =============================================================================

class TestWorkflowIntegration:
    """Tests for integration with main workflow."""
    
    def test_search_with_self_rag_method_exists(self):
        """Verify search_with_self_rag method is accessible."""
        from statement_copilot.agents.search_agent import ProfessionalSearchAgent
        
        agent = ProfessionalSearchAgent.__new__(ProfessionalSearchAgent)
        assert hasattr(agent, 'search_with_self_rag')
    
    def test_exports_are_correct(self):
        """Verify all exports are accessible from agents package."""
        from statement_copilot.agents import (
            build_search_subgraph,
            run_self_rag_search,
            get_search_subgraph,
        )
        
        assert callable(build_search_subgraph)
        assert callable(run_self_rag_search)
        assert callable(get_search_subgraph)


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

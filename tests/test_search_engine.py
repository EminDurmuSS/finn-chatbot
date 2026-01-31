"""
Statement Copilot - Search System Tests and Examples
====================================================
Comprehensive tests for the professional search system.

Run with: python -m pytest tests/test_search_engine.py -v
Or run examples: python tests/test_search_engine.py
"""

import json
from datetime import date, datetime, timedelta
from typing import Dict, Any, List


# =============================================================================
# QUERY UNDERSTANDING TESTS
# =============================================================================

QUERY_UNDERSTANDING_TEST_CASES = [
    # Find specific
    {
        "query": "Find my Youtube payment",
        "expected_intent": "find_specific",
        "expected_merchants": ["YOUTUBE"],
        "expected_strategy": "exact_match",
    },
    {
        "query": "Where is my Getir order from last week",
        "expected_intent": "find_specific",
        "expected_merchants": ["GETIR"],
        "expected_date_range": True,
    },
    
    # Aggregate
    {
        "query": "How much did I spend on groceries this month",
        "expected_intent": "aggregate",
        "expected_categories": ["food_and_dining"],
        "expected_direction": "expense",
    },
    {
        "query": "Total spending at Starbucks",
        "expected_intent": "aggregate",
        "expected_merchants": ["STARBUCKS"],
    },
    {
        "query": "What's my average transaction amount",
        "expected_intent": "aggregate",
    },
    
    # List/Filter
    {
        "query": "Show all restaurant transactions",
        "expected_intent": "list_filter",
        "expected_categories": ["food_and_dining"],
    },
    {
        "query": "List my subscriptions",
        "expected_intent": "list_filter",
        "expected_keywords": ["subscription"],
    },
    
    # Temporal
    {
        "query": "What did I spend yesterday",
        "expected_intent": "temporal",
        "expected_date_range": True,
    },
    {
        "query": "Last month's expenses",
        "expected_intent": "temporal",
        "expected_date_range": True,
        "expected_direction": "expense",
    },
    
    # Comparative
    {
        "query": "Transactions over $100",
        "expected_intent": "comparative",
        "expected_amounts": [{"op": "gt", "value": 100}],
    },
    {
        "query": "Purchases between $50 and $200",
        "expected_intent": "comparative",
        "expected_amounts": [{"op": "between"}],
    },
    
    # Similar
    {
        "query": "Transactions like my Uber rides",
        "expected_intent": "find_similar",
        "expected_merchants": ["UBER"],
    },
    
    # Anomaly
    {
        "query": "Show me any unusual transactions",
        "expected_intent": "anomaly",
    },
    {
        "query": "Are there any suspicious charges",
        "expected_intent": "anomaly",
    },
    
    # Merchant lookup
    {
        "query": "When did I last pay at Target",
        "expected_intent": "merchant_lookup",
        "expected_merchants": ["TARGET"],
    },
    
    # Complex queries
    {
        "query": "Find all coffee purchases over $10 last week",
        "expected_intent": "list_filter",
        "expected_categories": ["food_and_dining"],
        "expected_amounts": [{"op": "gt", "value": 10}],
        "expected_date_range": True,
    },
]


def test_query_understanding():
    """Test query understanding engine"""
    from statement_copilot.core.search_engine import QueryUnderstandingEngine
    
    engine = QueryUnderstandingEngine()
    
    results = []
    for i, test in enumerate(QUERY_UNDERSTANDING_TEST_CASES):
        query = test["query"]
        understanding = engine.understand(query)
        
        result = {
            "query": query,
            "intent": understanding.intent.value,
            "confidence": understanding.confidence,
            "strategy": understanding.strategy.value,
            "merchants": understanding.entities.merchants,
            "categories": understanding.entities.categories,
            "amounts": understanding.entities.amounts,
            "date_range": understanding.entities.date_range,
            "direction": understanding.entities.direction,
            "keywords": understanding.entities.keywords,
            "expanded_query": understanding.expanded_query[:100],
        }
        
        # Check assertions
        passed = True
        errors = []
        
        if test.get("expected_intent"):
            if understanding.intent.value != test["expected_intent"]:
                passed = False
                errors.append(f"Intent: expected {test['expected_intent']}, got {understanding.intent.value}")
        
        if test.get("expected_merchants"):
            for m in test["expected_merchants"]:
                if m not in understanding.entities.merchants:
                    passed = False
                    errors.append(f"Missing merchant: {m}")
        
        if test.get("expected_direction"):
            if understanding.entities.direction != test["expected_direction"]:
                passed = False
                errors.append(f"Direction: expected {test['expected_direction']}, got {understanding.entities.direction}")
        
        if test.get("expected_date_range"):
            if understanding.entities.date_range is None:
                passed = False
                errors.append("Expected date range but got None")
        
        if test.get("expected_amounts"):
            if not understanding.entities.amounts:
                passed = False
                errors.append("Expected amounts but got empty")
        
        result["passed"] = passed
        result["errors"] = errors
        results.append(result)
        
        # Print result
        status = "✓" if passed else "✗"
        print(f"{status} Test {i+1}: {query[:50]}...")
        if errors:
            for err in errors:
                print(f"    Error: {err}")
    
    # Summary
    passed_count = sum(1 for r in results if r["passed"])
    print(f"\nPassed: {passed_count}/{len(results)}")
    
    return results


# =============================================================================
# SEARCH ENGINE INTEGRATION TESTS
# =============================================================================

def test_search_engine_mock():
    """Test search engine with mock data"""
    from statement_copilot.core.search_engine import (
        ProfessionalSearchEngine,
        QueryUnderstandingEngine,
        MultiSourceRetriever,
        SearchReranker,
    )
    
    # Create mock database
    class MockDB:
        def execute_query(self, sql, params=None, read_only=True):
            # Return mock transaction data
            return [
                {
                    "tx_id": "tx_001",
                    "date_time": datetime(2025, 1, 15, 10, 30),
                    "amount": -15.99,
                    "merchant_norm": "NETFLIX",
                    "description": "NETFLIX.COM",
                    "category": "utilities",
                    "subcategory": "tv_streaming",
                    "direction": "expense",
                },
                {
                    "tx_id": "tx_002",
                    "date_time": datetime(2025, 1, 14, 14, 22),
                    "amount": -45.67,
                    "merchant_norm": "AMAZON",
                    "description": "AMAZON.COM PURCHASE",
                    "category": "shopping",
                    "subcategory": "online_shopping",
                    "direction": "expense",
                },
                {
                    "tx_id": "tx_003",
                    "date_time": datetime(2025, 1, 13, 8, 15),
                    "amount": -5.75,
                    "merchant_norm": "STARBUCKS",
                    "description": "STARBUCKS COFFEE",
                    "category": "food_and_dining",
                    "subcategory": "cafe_coffee",
                    "direction": "expense",
                },
            ]
    
    # Create mock vector store
    class MockVectorStore:
        def search(self, query, top_k, alpha, filters):
            return [
                {"tx_id": "tx_001", "score": 0.95, "metadata": {"merchant_norm": "NETFLIX"}},
                {"tx_id": "tx_002", "score": 0.72, "metadata": {"merchant_norm": "AMAZON"}},
            ]
    
    # Initialize engine
    engine = ProfessionalSearchEngine(
        db=MockDB(),
        vector_store=MockVectorStore(),
        llm_client=None,
        taxonomy={}
    )
    
    # Test queries
    test_queries = [
        "Find my Netflix payment",
        "How much did I spend at Amazon",
        "Show me coffee purchases",
        "Transactions over $10",
    ]
    
    print("\n" + "="*60)
    print("SEARCH ENGINE INTEGRATION TEST")
    print("="*60)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = engine.search(query, "test_tenant", top_k=5)
        
        print(f"  Intent: {result.query_understanding.intent.value}")
        print(f"  Strategy: {result.query_understanding.strategy.value}")
        print(f"  Results: {len(result.matches)}")
        
        for match in result.matches[:3]:
            print(f"    - {match.merchant_norm}: ${abs(match.amount or 0):.2f} (score: {match.score:.2f})")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_basic_search():
    """Example: Basic search usage"""
    print("\n" + "="*60)
    print("EXAMPLE: Basic Search")
    print("="*60)
    
    # This would be the actual usage in production
    code = '''
from statement_copilot.agents.search_agent_v2 import quick_search, analyze_query

# Analyze a query without searching
analysis = analyze_query("Find my Netflix payment last month")
print(f"Intent: {analysis['intent']}")
print(f"Merchants: {analysis['entities']['merchants']}")
print(f"Date range: {analysis['entities']['date_range']}")

# Execute search
results = quick_search("Find my Netflix payment last month", tenant_id="my_tenant")
for r in results:
    print(f"{r['merchant_norm']}: ${abs(r['amount']):.2f} on {r['date_time']}")
'''
    print(code)


def example_advanced_search():
    """Example: Advanced search with filters"""
    print("\n" + "="*60)
    print("EXAMPLE: Advanced Search with Filters")
    print("="*60)
    
    code = '''
from statement_copilot.core.search_engine import ProfessionalSearchEngine
from statement_copilot.core import get_db, get_vector_store, get_llm_client

# Initialize search engine with all components
engine = ProfessionalSearchEngine(
    db=get_db(),
    vector_store=get_vector_store(),
    llm_client=get_llm_client(),  # Optional: enables LLM enhancement
    taxonomy=load_taxonomy()
)

# Complex query with multiple constraints
result = engine.search(
    query="Show all grocery purchases over $50 last month",
    tenant_id="my_tenant",
    top_k=20,
    use_llm_rerank=True  # Enable LLM reranking for better precision
)

# Access detailed understanding
print(f"Intent: {result.query_understanding.intent.value}")
print(f"Strategy: {result.query_understanding.strategy.value}")
print(f"Confidence: {result.query_understanding.confidence:.2f}")

# Access extracted entities
entities = result.query_understanding.entities
print(f"Categories: {entities.categories}")
print(f"Date range: {entities.date_range}")
print(f"Amount filter: {entities.amounts}")

# Process results
for match in result.matches:
    print(f"- {match.merchant_norm}: ${abs(match.amount):.2f}")
    print(f"  Score: {match.score:.2f}, Source: {match.source}")
    print(f"  Reason: {match.match_reason}")
'''
    print(code)


def example_similar_transactions():
    """Example: Finding similar transactions"""
    print("\n" + "="*60)
    print("EXAMPLE: Similar Transactions")
    print("="*60)
    
    code = '''
from statement_copilot.agents.search_agent_v2 import get_professional_search_agent

agent = get_professional_search_agent()

# Find transactions similar to a specific one
similar = agent.find_similar(
    tx_id="tx_12345",
    tenant_id="my_tenant",
    top_k=10
)

print("Transactions similar to tx_12345:")
for tx in similar:
    print(f"- {tx['merchant_norm']}: ${abs(tx['amount']):.2f} (similarity: {tx['score']:.2f})")
'''
    print(code)


def example_integration_with_workflow():
    """Example: Integration with LangGraph workflow"""
    print("\n" + "="*60)
    print("EXAMPLE: Integration with Workflow")
    print("="*60)
    
    code = '''
# In your agents/__init__.py, update the search agent import:
from .search_agent_v2 import (
    ProfessionalSearchAgent,
    get_professional_search_agent as get_search_agent,
    quick_search,
    analyze_query,
)

# In your workflow.py, the search node uses the new agent automatically:
def search_agent_node(state: OrchestratorState) -> OrchestratorState:
    """Vector-based transaction search using professional search engine."""
    with log_context(node="search_agent"):
        search = get_search_agent()  # Now returns ProfessionalSearchAgent
        state = search.search(state)
        
        # Access detailed search evidence
        evidence = state.get("search_evidence", {})
        print(f"Sources used: {evidence.get('sources_used')}")
        print(f"Reasoning: {evidence.get('reasoning')}")
        
        return state
'''
    print(code)


# =============================================================================
# PERFORMANCE BENCHMARKS
# =============================================================================

def benchmark_query_understanding():
    """Benchmark query understanding performance"""
    import time
    from statement_copilot.core.search_engine import QueryUnderstandingEngine
    
    engine = QueryUnderstandingEngine()
    
    queries = [
        "Find my Netflix payment",
        "Total spending on groceries this month",
        "Transactions over $100 last week",
        "Show all restaurant purchases",
        "When did I last pay at Target",
    ] * 20  # 100 queries
    
    print("\n" + "="*60)
    print("BENCHMARK: Query Understanding")
    print("="*60)
    
    start = time.time()
    for query in queries:
        engine.understand(query)
    elapsed = time.time() - start
    
    print(f"Total queries: {len(queries)}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Average per query: {(elapsed / len(queries)) * 1000:.2f}ms")
    print(f"Queries per second: {len(queries) / elapsed:.1f}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("STATEMENT COPILOT - SEARCH SYSTEM TESTS")
    print("="*60)
    
    # Run tests
    print("\n[1] Query Understanding Tests")
    test_query_understanding()
    
    print("\n[2] Search Engine Integration Test")
    test_search_engine_mock()
    
    print("\n[3] Usage Examples")
    example_basic_search()
    example_advanced_search()
    example_similar_transactions()
    example_integration_with_workflow()
    
    print("\n[4] Performance Benchmark")
    benchmark_query_understanding()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
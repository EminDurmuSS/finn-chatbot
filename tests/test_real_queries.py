"""
Real-World Integration Test for Self-RAG Search
================================================
This script runs actual queries through the Statement Copilot system
and displays detailed orchestration flow to verify the Self-RAG pattern.

Run with: python test_real_queries.py
"""

import sys
import os
import logging
import json
from datetime import datetime

# Add project root to sys.path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s',
    datefmt='%H:%M:%S'
)

# Reduce noise from other loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("anthropic").setLevel(logging.WARNING)

logger = logging.getLogger("real_test")


def print_header(text):
    print("\n" + "=" * 80)
    print(f" {text}")
    print("=" * 80)


def print_section(text):
    print("\n" + "-" * 60)
    print(f" {text}")
    print("-" * 60)


def print_step(step_num, text):
    print(f"\n[STEP {step_num}] {text}")


def print_json(data, indent=2):
    """Pretty print JSON data"""
    if isinstance(data, dict):
        print(json.dumps(data, indent=indent, default=str, ensure_ascii=False))
    else:
        print(data)


# =============================================================================
# TEST HARNESS
# =============================================================================

class RealQueryTester:
    """Interactive test harness for real query execution."""
    
    def __init__(self):
        self.step_count = 0
        self.results = []
    
    def test_query(self, query: str, tenant_id: str = "demo"):
        """
        Execute a query through the full system and display orchestration flow.
        """
        print_header(f"TESTING: {query}")
        
        from statement_copilot import StatementCopilot
        from statement_copilot.core import create_initial_state
        
        self.step_count = 0
        
        # Create copilot instance
        print_step(1, "Initializing Statement Copilot")
        copilot = StatementCopilot()
        print("   Copilot initialized successfully")
        
        # Create initial state
        print_step(2, "Creating initial state")
        state = create_initial_state(
            session_id=f"test_{datetime.now().strftime('%H%M%S')}",
            user_message=query,
            tenant_id=tenant_id,
        )
        print(f"   Session ID: {state.get('session_id')}")
        print(f"   User Message: {state.get('user_message')}")
        print(f"   Tenant ID: {state.get('tenant_id')}")
        
        # Run the workflow
        print_step(3, "Executing workflow...")
        print("   " + "-" * 50)
        
        try:
            result = copilot.run(state)
            
            # Analyze the result
            self._analyze_result(result, query)
            
            return result
            
        except Exception as e:
            print(f"\n   [ERROR] Workflow failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _analyze_result(self, result: dict, original_query: str):
        """Analyze and display the orchestration result."""
        
        print_section("ORCHESTRATION ANALYSIS")
        
        # 1. Intent Classification
        print_step(4, "Intent Classification")
        print(f"   Intent: {result.get('intent')}")
        print(f"   Confidence: {result.get('confidence')}")
        print(f"   Reasoning: {result.get('reasoning', 'N/A')[:100]}...")
        
        # 2. Routing Decisions
        print_step(5, "Routing Decisions")
        print(f"   needs_sql: {result.get('needs_sql')}")
        print(f"   needs_vector: {result.get('needs_vector')}")
        print(f"   needs_planner: {result.get('needs_planner')}")
        
        # 3. Constraints Extracted
        print_step(6, "Constraints Extracted")
        constraints = result.get('constraints', {})
        if constraints:
            print_json(constraints)
        else:
            print("   No constraints extracted")
        
        # 4. Search Results (Self-RAG)
        print_step(7, "Search Agent (Self-RAG)")
        search_evidence = result.get('search_evidence', {})
        search_query = result.get('search_query', {})
        vector_result = result.get('vector_result', {})
        
        if search_evidence:
            self_rag_info = search_evidence.get('self_rag_info', {})
            
            if self_rag_info:
                print("   [SELF-RAG ACTIVATED]")
                print(f"   Attempts: {self_rag_info.get('attempts', 1)}")
                print(f"   Final Quality: {self_rag_info.get('final_quality')}")
                print(f"   Final Query: {self_rag_info.get('final_query')}")
                
                # Show all attempts
                all_attempts = self_rag_info.get('all_attempts', [])
                if all_attempts:
                    print("\n   Attempt History:")
                    for attempt in all_attempts:
                        print(f"     [{attempt.get('attempt')}] Query: '{attempt.get('query', '')[:40]}...' -> Found: {attempt.get('found', 0)}")
            else:
                print("   [Standard Search]")
                print(f"   Query: {search_query.get('query_text', 'N/A')}")
        else:
            print("   No search executed")
        
        if vector_result:
            matches = vector_result.get('matches', [])
            print(f"\n   Results Found: {len(matches)}")
            for i, match in enumerate(matches[:5]):
                print(f"     {i+1}. {match.get('merchant_norm', 'N/A')} - {match.get('amount', 0):.2f} TRY ({match.get('category', 'N/A')})")
        
        # 5. SQL Results
        print_step(8, "SQL Agent")
        sql_result = result.get('sql_result')
        if sql_result:
            print(f"   Value: {sql_result.get('value')}")
            print(f"   TX Count: {sql_result.get('tx_count')}")
            print(f"   SQL Preview: {sql_result.get('sql_preview', 'N/A')[:80]}...")
        else:
            print("   No SQL query executed")
        
        # 6. Final Answer
        print_step(9, "Final Answer")
        print("-" * 60)
        final_answer = result.get('final_answer', 'No answer generated')
        print(f"\n{final_answer}\n")
        print("-" * 60)
        
        # 7. Tool Calls Summary
        print_step(10, "Tool Calls Summary")
        tool_calls = result.get('tool_calls', [])
        if tool_calls:
            for tc in tool_calls:
                print(f"   - {tc.get('tool_name')} ({tc.get('node')}) - {tc.get('latency_ms', 0)}ms")
        else:
            print("   No tool calls recorded")
        
        # 8. Errors (if any)
        errors = result.get('errors', [])
        if errors:
            print_step(11, "ERRORS")
            for err in errors:
                print(f"   [!] {err}")
        
        print_header("TEST COMPLETE")
        
        return result


def test_self_rag_only(query: str, tenant_id: str = "demo"):
    """
    Test only the Self-RAG subgraph without full orchestration.
    Useful for isolated testing.
    """
    print_header(f"SELF-RAG ONLY TEST: {query}")
    
    from statement_copilot.agents.search_graph import run_self_rag_search
    
    print_step(1, "Executing Self-RAG Search")
    result = run_self_rag_search(
        query=query,
        tenant_id=tenant_id,
        constraints={},
        max_attempts=3,
    )
    
    print_step(2, "Results")
    print(f"   Attempts: {result.get('attempts')}")
    print(f"   Final Quality: {result.get('final_quality')}")
    print(f"   Final Query: {result.get('final_query')}")
    print(f"   Total Found: {result.get('total_found')}")
    print(f"   Total Time: {result.get('total_time_ms')}ms")
    
    print_step(3, "Attempt History")
    for attempt in result.get('all_attempts', []):
        status = "OK" if attempt.get('found', 0) > 0 else "EMPTY"
        print(f"   [{attempt.get('attempt')}] {status} - Query: '{attempt.get('query', '')[:50]}...'")
    
    if result.get('critique'):
        print_step(4, "Final Critique")
        print(f"   {result.get('critique')}")
    
    print_step(5, "Top Results")
    for i, match in enumerate(result.get('results', [])[:5]):
        print(f"   {i+1}. {match.get('merchant_norm', 'N/A')} - {match.get('amount', 0):.2f} TRY")
    
    print_header("TEST COMPLETE")
    return result


# =============================================================================
# PREDEFINED TEST QUERIES
# =============================================================================

TEST_QUERIES = [
    "Find my YouTube payment",
    "How much did I spend at Green Chef",
    "Show me Netflix transactions",
    "What did I spend on food this month",
    "Netflux payment",  # Typo test - should be corrected to Netflix
    "spotify subscription",
    "amazon purchases last week",
]


def run_all_tests():
    """Run all predefined test queries."""
    print_header("RUNNING ALL TEST QUERIES")
    
    tester = RealQueryTester()
    results = []
    
    for query in TEST_QUERIES:
        try:
            result = tester.test_query(query)
            results.append({
                "query": query,
                "success": result is not None,
                "answer": result.get('final_answer', '')[:100] if result else None,
            })
        except Exception as e:
            results.append({
                "query": query,
                "success": False,
                "error": str(e),
            })
        
        print("\n" + "=" * 80 + "\n")
    
    # Summary
    print_header("TEST SUMMARY")
    for r in results:
        status = "[OK]" if r.get('success') else "[FAIL]"
        print(f"  {status} {r.get('query')}")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print_header("STATEMENT COPILOT - REAL QUERY TESTING")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if len(sys.argv) > 1:
        # Custom query from command line
        query = " ".join(sys.argv[1:])
        tester = RealQueryTester()
        tester.test_query(query)
    else:
        # Interactive mode
        print("\nOptions:")
        print("  1. Run predefined test queries")
        print("  2. Enter custom query")
        print("  3. Test Self-RAG only (isolated)")
        print("  4. Exit")
        
        try:
            choice = input("\nSelect option (1-4): ").strip()
            
            if choice == "1":
                run_all_tests()
            elif choice == "2":
                query = input("Enter query: ").strip()
                if query:
                    tester = RealQueryTester()
                    tester.test_query(query)
            elif choice == "3":
                query = input("Enter query for Self-RAG: ").strip()
                if query:
                    test_self_rag_only(query)
            else:
                print("Exiting...")
        except KeyboardInterrupt:
            print("\nTest cancelled.")

"""
COMPREHENSIVE DEEP TEST: YouTube Subscription Query
====================================================
This script performs an exhaustive test of the Statement Copilot agent
with detailed analysis of every component and comparison with expected output.

Tests:
- Intent Classification (Orchestrator)  
- Routing Decisions (SQL vs Vector vs Both)
- Self-RAG Search Pattern (attempts, transformations)
- Finance Analyst SQL Generation
- Synthesizer Response Quality
- Final Output Comparison with Expected

Expected: 18 YouTube Premium + 2 YouTube Membership transactions
"""

import sys
import os
import io

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import logging
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict

# Configure comprehensive logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)-8s | %(name)-40s | %(message)s',
    datefmt='%H:%M:%S'
)

# Reduce noise
for noisy in ["httpx", "httpcore", "anthropic", "chromadb", "sentence_transformers"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger("DEEP_TEST")


# =============================================================================
# EXPECTED OUTPUT - Ground Truth
# =============================================================================

EXPECTED_OUTPUT = {
    "has_subscription": True,
    "youtube_premium_count": 18,
    "youtube_membership_count": 2,
    "total_youtube_transactions": 20,
    "youtube_premium_total": 903.82,
    "youtube_membership_total": 60.00,
    "grand_total": 963.82,
    "missing_months": ["2024-09", "2024-10"],
    "price_changes": {
        "37.99": "May-Dec 2024 (6 payments)",
        "52.99": "Jan-Apr 2025 (4 payments)", 
        "57.99": "May-Dec 2025 (8 payments)",
    },
    "first_premium_date": "2024-05-30",
    "last_premium_date": "2025-12-13",
    "first_membership_date": "2022-08-22",
    "last_membership_date": "2022-09-22",
}

EXPECTED_TRANSACTIONS = [
    {"plan": "YouTube Membership", "date": "2022-08-22", "amount": 30.00},
    {"plan": "YouTube Membership", "date": "2022-09-22", "amount": 30.00},
    {"plan": "YouTube Premium", "date": "2024-05-30", "amount": 37.99},
    {"plan": "YouTube Premium", "date": "2024-06-29", "amount": 37.99},
    {"plan": "YouTube Premium", "date": "2024-07-28", "amount": 37.99},
    {"plan": "YouTube Premium", "date": "2024-08-28", "amount": 37.99},
    {"plan": "YouTube Premium", "date": "2024-11-09", "amount": 37.99},
    {"plan": "YouTube Premium", "date": "2024-12-13", "amount": 37.99},
    {"plan": "YouTube Premium", "date": "2025-01-15", "amount": 52.99},
    {"plan": "YouTube Premium", "date": "2025-02-13", "amount": 52.99},
    {"plan": "YouTube Premium", "date": "2025-03-13", "amount": 52.99},
    {"plan": "YouTube Premium", "date": "2025-04-16", "amount": 52.99},
    {"plan": "YouTube Premium", "date": "2025-05-13", "amount": 57.99},
    {"plan": "YouTube Premium", "date": "2025-06-13", "amount": 57.99},
    {"plan": "YouTube Premium", "date": "2025-07-13", "amount": 57.99},
    {"plan": "YouTube Premium", "date": "2025-08-13", "amount": 57.99},
    {"plan": "YouTube Premium", "date": "2025-09-13", "amount": 57.99},
    {"plan": "YouTube Premium", "date": "2025-10-13", "amount": 57.99},
    {"plan": "YouTube Premium", "date": "2025-11-18", "amount": 57.99},
    {"plan": "YouTube Premium", "date": "2025-12-13", "amount": 57.99},
]


# =============================================================================
# TEST RESULT TRACKER
# =============================================================================

@dataclass
class TestResult:
    """Tracks test execution and results."""
    test_query: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    
    # Orchestration Analysis
    intent: str = ""
    confidence: float = 0.0
    reasoning: str = ""
    needs_sql: bool = False
    needs_vector: bool = False
    needs_planner: bool = False
    constraints: Dict = field(default_factory=dict)
    
    # Search Agent (Self-RAG)
    search_attempts: int = 0
    search_quality: str = ""
    search_query_history: List[Dict] = field(default_factory=list)
    search_results: List[Dict] = field(default_factory=list)
    search_total_found: int = 0
    self_rag_active: bool = False
    
    # Finance Analyst (SQL)
    sql_executed: bool = False
    sql_metric: str = ""
    sql_filters: Dict = field(default_factory=dict)
    sql_query: str = ""
    sql_result: Dict = field(default_factory=dict)
    
    # Synthesizer
    final_answer: str = ""
    
    # Tool Call Trace
    tool_calls: List[Dict] = field(default_factory=list)
    
    # Errors
    errors: List[str] = field(default_factory=list)
    
    @property
    def duration_ms(self) -> int:
        return int((self.end_time - self.start_time) * 1000)


# =============================================================================
# PRINTING UTILITIES
# =============================================================================

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str, char: str = "="):
    width = 100
    print(f"\n{Colors.BOLD}{Colors.HEADER}{char * width}")
    print(f" {text.center(width - 2)}")
    print(f"{char * width}{Colors.ENDC}")


def print_section(text: str, color: str = Colors.CYAN):
    print(f"\n{color}{Colors.BOLD}{'─' * 80}")
    print(f" {text}")
    print(f"{'─' * 80}{Colors.ENDC}")


def print_subsection(text: str):
    print(f"\n{Colors.BLUE}▶ {text}{Colors.ENDC}")


def print_ok(text: str):
    print(f"  {Colors.GREEN}✓ {text}{Colors.ENDC}")


def print_warn(text: str):
    print(f"  {Colors.YELLOW}⚠ {text}{Colors.ENDC}")


def print_fail(text: str):
    print(f"  {Colors.RED}✗ {text}{Colors.ENDC}")


def print_info(text: str):
    print(f"  {Colors.CYAN}ℹ {text}{Colors.ENDC}")


def print_kv(key: str, value: Any, indent: int = 2):
    spaces = " " * indent
    if isinstance(value, dict):
        print(f"{spaces}{Colors.BOLD}{key}:{Colors.ENDC}")
        for k, v in value.items():
            print(f"{spaces}  {k}: {v}")
    elif isinstance(value, list) and value:
        print(f"{spaces}{Colors.BOLD}{key}:{Colors.ENDC}")
        for item in value[:10]:  # Limit to 10 items
            if isinstance(item, dict):
                summary = ", ".join(f"{k}={v}" for k, v in list(item.items())[:4])
                print(f"{spaces}  • {summary}")
            else:
                print(f"{spaces}  • {item}")
        if len(value) > 10:
            print(f"{spaces}  ... and {len(value) - 10} more")
    else:
        print(f"{spaces}{Colors.BOLD}{key}:{Colors.ENDC} {value}")


def print_json(data: Any, indent: int = 2):
    print(json.dumps(data, indent=2, default=str, ensure_ascii=False))


# =============================================================================
# DEEP TEST EXECUTOR
# =============================================================================

class DeepTestExecutor:
    """Executes comprehensive deep testing with full tracing."""
    
    def __init__(self):
        self.result = TestResult()
    
    def run(self, query: str, tenant_id: str = "demo") -> TestResult:
        """Execute the deep test."""
        print_header("COMPREHENSIVE DEEP TEST: Statement Copilot")
        print(f"\n{Colors.BOLD}Test Query:{Colors.ENDC} {query}")
        print(f"{Colors.BOLD}Tenant:{Colors.ENDC} {tenant_id}")
        print(f"{Colors.BOLD}Timestamp:{Colors.ENDC} {datetime.now().isoformat()}")
        
        self.result.test_query = query
        self.result.start_time = time.time()
        
        # Phase 1: Initialize and Run Workflow
        print_section("PHASE 1: Workflow Execution")
        state = self._execute_workflow(query, tenant_id)
        
        self.result.end_time = time.time()
        
        if state is None:
            print_fail("Workflow execution failed!")
            return self.result
        
        # Phase 2: Deep Analysis
        print_section("PHASE 2: Orchestration Deep Analysis")
        self._analyze_orchestration(state)
        
        print_section("PHASE 3: Search Agent (Self-RAG) Analysis")
        self._analyze_search(state)
        
        print_section("PHASE 4: Finance Analyst (SQL) Analysis")
        self._analyze_sql(state)
        
        print_section("PHASE 5: Synthesizer Output")
        self._analyze_synthesizer(state)
        
        # Phase 3: Comparison with Expected
        print_section("PHASE 6: Expected vs Actual Comparison")
        self._compare_with_expected(state)
        
        # Phase 4: Final Summary
        print_section("PHASE 7: Test Summary")
        self._print_summary()
        
        return self.result
    
    def _execute_workflow(self, query: str, tenant_id: str) -> Optional[Dict]:
        """Execute the full workflow and capture state."""
        print_subsection("Initializing Statement Copilot...")
        
        try:
            from statement_copilot import StatementCopilot
            
            copilot = StatementCopilot()
            print_ok("Copilot initialized successfully")
            
            # Execute workflow using chat method
            print_subsection("Executing workflow (this may take a moment)...")
            workflow_start = time.time()
            
            session_id = f"deep_test_{datetime.now().strftime('%H%M%S')}"
            result = copilot.chat(
                message=query,
                session_id=session_id,
                tenant_id=tenant_id,
                user_id="test_user",
            )
            workflow_time = (time.time() - workflow_start) * 1000
            
            print_ok(f"Workflow completed in {workflow_time:.0f}ms")
            print_kv("Response keys", list(result.keys()))
            
            # Get full state for deeper analysis
            full_state = None
            try:
                full_state = copilot.get_state(session_id)
                if full_state:
                    print_ok(f"Full state retrieved ({len(full_state)} keys)")
                    # Merge full state with result for analysis
                    result = {**result, **full_state}
            except Exception as e:
                print_warn(f"Could not retrieve full state: {e}")
            
            return result
            
        except Exception as e:
            print_fail(f"Error: {e}")
            import traceback
            traceback.print_exc()
            self.result.errors.append(str(e))
            return None
    
    def _analyze_orchestration(self, state: Dict):
        """Deep analysis of orchestration decisions."""
        print_subsection("Intent Classification")
        
        intent = state.get("intent", "UNKNOWN")
        confidence = state.get("confidence", 0.0)
        reasoning = state.get("reasoning", "N/A")
        
        self.result.intent = intent
        self.result.confidence = confidence
        self.result.reasoning = reasoning
        
        print_kv("Intent", intent)
        print_kv("Confidence", f"{confidence:.2f}")
        print_kv("Reasoning", reasoning[:200] + "..." if len(reasoning) > 200 else reasoning)
        
        # Validate intent
        if intent in ["LOOKUP", "ANALYTICS"]:
            print_ok("Intent correctly classified for transaction/subscription query")
        else:
            print_warn(f"Intent '{intent}' may not be optimal for this query")
        
        print_subsection("Routing Decisions")
        
        needs_sql = state.get("needs_sql", False)
        needs_vector = state.get("needs_vector", False)
        needs_planner = state.get("needs_planner", False)
        
        self.result.needs_sql = needs_sql
        self.result.needs_vector = needs_vector
        self.result.needs_planner = needs_planner
        
        print_kv("needs_sql", needs_sql)
        print_kv("needs_vector", needs_vector)
        print_kv("needs_planner", needs_planner)
        
        # For subscription history, we ideally want both SQL (for aggregates) and vector (for finding transactions)
        if needs_vector:
            print_ok("Vector search enabled - good for finding YouTube transactions")
        else:
            print_warn("Vector search NOT enabled - may miss specific transactions")
        
        if needs_sql:
            print_ok("SQL analysis enabled - good for aggregation/counting")
        else:
            print_warn("SQL analysis NOT enabled - may lack aggregation")
        
        print_subsection("Extracted Constraints")
        constraints = state.get("constraints", {})
        self.result.constraints = constraints
        
        if constraints:
            print_json(constraints)
        else:
            print_info("No specific constraints extracted")
    
    def _analyze_search(self, state: Dict):
        """Deep analysis of Self-RAG search behavior."""
        search_evidence = state.get("search_evidence", {})
        vector_result = state.get("vector_result", {})
        
        if not search_evidence and not vector_result:
            print_info("Search agent was not invoked")
            return
        
        # Self-RAG info
        self_rag_info = search_evidence.get("self_rag_info", {}) if search_evidence else {}
        
        if self_rag_info:
            self.result.self_rag_active = True
            print_subsection("Self-RAG Pattern Details")
            
            attempts = self_rag_info.get("attempts", 1)
            final_quality = self_rag_info.get("final_quality", "unknown")
            final_query = self_rag_info.get("final_query", "")
            all_attempts = self_rag_info.get("all_attempts", [])
            
            self.result.search_attempts = attempts
            self.result.search_quality = final_quality
            self.result.search_query_history = all_attempts
            
            print_kv("Total Attempts", attempts)
            print_kv("Final Quality", final_quality)
            print_kv("Final Query", final_query)
            
            if all_attempts:
                print_subsection("Attempt History (Query Transformation)")
                for i, attempt in enumerate(all_attempts, 1):
                    query = attempt.get("query", "")[:60]
                    found = attempt.get("found", 0)
                    latency = attempt.get("latency_ms", 0)
                    status = "✓" if found > 0 else "✗"
                    print(f"    {status} Attempt {i}: '{query}...' → Found: {found} ({latency}ms)")
            
            if attempts == 1 and final_quality == "good":
                print_ok("Query succeeded on first attempt - no transformation needed")
            elif attempts > 1 and final_quality == "good":
                print_ok(f"Self-RAG successfully refined query after {attempts} attempts")
            elif final_quality != "good":
                print_warn(f"Self-RAG ended with quality '{final_quality}' after {attempts} attempts")
        
        # Search Results Analysis
        matches = vector_result.get("matches", []) if vector_result else []
        total_found = vector_result.get("total_found", len(matches))
        
        self.result.search_results = matches
        self.result.search_total_found = total_found
        
        print_subsection(f"Search Results: {total_found} transactions found")
        
        if matches:
            # Group by merchant for analysis
            by_merchant = defaultdict(list)
            for m in matches:
                merchant = m.get("merchant_norm", "Unknown")
                by_merchant[merchant].append(m)
            
            print_kv("Unique Merchants", len(by_merchant))
            
            for merchant, txns in by_merchant.items():
                total = sum(abs(t.get("amount", 0)) for t in txns)
                print(f"    • {merchant}: {len(txns)} transactions, Total: {total:.2f} TRY")
            
            # Check for YouTube specifically
            youtube_count = sum(
                1 for m in matches 
                if "youtube" in m.get("merchant_norm", "").lower() or
                   "youtube" in m.get("description", "").lower()
            )
            
            if youtube_count > 0:
                print_ok(f"Found {youtube_count} YouTube-related transactions")
            else:
                print_warn("No YouTube transactions found in search results")
        else:
            print_warn("No transactions returned from search")
    
    def _analyze_sql(self, state: Dict):
        """Deep analysis of SQL agent behavior."""
        sql_result = state.get("sql_result")
        
        if not sql_result:
            print_info("SQL agent was not invoked")
            return
        
        self.result.sql_executed = True
        self.result.sql_result = sql_result
        
        print_subsection("SQL Execution Details")
        
        # Extract SQL info
        metric = sql_result.get("metric", "unknown")
        value = sql_result.get("value")
        tx_count = sql_result.get("tx_count")
        sql_preview = sql_result.get("sql_preview", "N/A")
        rows = sql_result.get("rows", [])
        
        self.result.sql_metric = metric
        self.result.sql_query = sql_preview
        
        print_kv("Metric Type", metric)
        print_kv("Result Value", value)
        print_kv("Transaction Count", tx_count)
        print_kv("SQL Preview", sql_preview[:150] + "..." if len(sql_preview) > 150 else sql_preview)
        
        if rows:
            print_subsection(f"SQL Result Rows: {len(rows)} rows")
            for i, row in enumerate(rows[:15], 1):
                if isinstance(row, dict):
                    summary = ", ".join(f"{k}={v}" for k, v in list(row.items())[:5])
                    print(f"    {i}. {summary}")
                else:
                    print(f"    {i}. {row}")
            
            if len(rows) > 15:
                print(f"    ... and {len(rows) - 15} more rows")
    
    def _analyze_synthesizer(self, state: Dict):
        """Analyze the synthesized response."""
        final_answer = state.get("final_answer", "No answer generated")
        self.result.final_answer = final_answer
        
        print_subsection("Final Response")
        print(f"\n{Colors.CYAN}{'─' * 80}{Colors.ENDC}")
        print(final_answer)
        print(f"{Colors.CYAN}{'─' * 80}{Colors.ENDC}\n")
        
        # Basic quality checks
        print_subsection("Response Quality Checks")
        
        answer_lower = final_answer.lower()
        
        # Check for key elements
        checks = [
            ("Mentions YouTube", "youtube" in answer_lower),
            ("Mentions subscription/premium", any(w in answer_lower for w in ["subscription", "premium", "membership"])),
            ("Contains payment amounts", any(c.isdigit() for c in final_answer) and ("." in final_answer or "," in final_answer)),
            ("Contains dates", any(str(y) in final_answer for y in range(2022, 2027))),
            ("Provides history/details", len(final_answer) > 500),
        ]
        
        for check_name, passed in checks:
            if passed:
                print_ok(check_name)
            else:
                print_warn(f"{check_name} - NOT FOUND")
    
    def _compare_with_expected(self, state: Dict):
        """Compare actual results with expected output."""
        print_subsection("Ground Truth Comparison")
        
        # Get all found transactions
        matches = []
        vector_result = state.get("vector_result", {})
        if vector_result:
            matches = vector_result.get("matches", [])
        
        sql_result = state.get("sql_result", {})
        sql_rows = sql_result.get("rows", []) if sql_result else []
        
        # Count YouTube transactions
        youtube_matches = [
            m for m in matches 
            if "youtube" in m.get("merchant_norm", "").lower() or
               "youtube" in m.get("description", "").lower()
        ]
        
        print_kv("Expected YouTube transactions", EXPECTED_OUTPUT["total_youtube_transactions"])
        print_kv("Found YouTube transactions", len(youtube_matches))
        
        if len(youtube_matches) >= EXPECTED_OUTPUT["total_youtube_transactions"] * 0.9:
            print_ok("Found >= 90% of expected YouTube transactions")
        elif len(youtube_matches) >= EXPECTED_OUTPUT["total_youtube_transactions"] * 0.7:
            print_warn("Found 70-90% of expected YouTube transactions")
        else:
            print_fail(f"Found < 70% of expected transactions ({len(youtube_matches)}/{EXPECTED_OUTPUT['total_youtube_transactions']})")
        
        # Calculate found totals
        if youtube_matches:
            total_found = sum(abs(m.get("amount", 0)) for m in youtube_matches)
            print_kv("Expected grand total", f"{EXPECTED_OUTPUT['grand_total']:.2f} TRY")
            print_kv("Found total (from search)", f"{total_found:.2f} TRY")
            
            diff_pct = abs(total_found - EXPECTED_OUTPUT["grand_total"]) / EXPECTED_OUTPUT["grand_total"] * 100
            if diff_pct < 5:
                print_ok(f"Total within 5% of expected (diff: {diff_pct:.1f}%)")
            elif diff_pct < 15:
                print_warn(f"Total within 15% of expected (diff: {diff_pct:.1f}%)")
            else:
                print_fail(f"Total differs by {diff_pct:.1f}%")
        
        # Check answer content
        print_subsection("Answer Content Verification")
        answer = self.result.final_answer.lower()
        
        content_checks = [
            ("Yes/subscription confirmation", any(w in answer for w in ["yes", "evet", "have a", "sahip"])),
            ("YouTube Premium mentioned", "youtube premium" in answer or "youtubepremium" in answer),
            ("Multiple payments listed", answer.count("202") >= 10),  # At least 10 date references
            ("Price 37.99 mentioned", "37.99" in answer or "37,99" in answer),
            ("Price 52.99 mentioned", "52.99" in answer or "52,99" in answer),
            ("Price 57.99 mentioned", "57.99" in answer or "57,99" in answer),
        ]
        
        passed = 0
        for check_name, result in content_checks:
            if result:
                print_ok(check_name)
                passed += 1
            else:
                print_warn(f"{check_name} - NOT FOUND")
        
        print(f"\n  Content Score: {passed}/{len(content_checks)} checks passed")
    
    def _print_summary(self):
        """Print final test summary."""
        print_subsection("Execution Metrics")
        print_kv("Total Duration", f"{self.result.duration_ms}ms")
        print_kv("Intent", f"{self.result.intent} (confidence: {self.result.confidence:.2f})")
        print_kv("Routing", f"SQL={self.result.needs_sql}, Vector={self.result.needs_vector}")
        
        if self.result.self_rag_active:
            print_kv("Self-RAG", f"{self.result.search_attempts} attempts, quality: {self.result.search_quality}")
        
        print_kv("Transactions Found", self.result.search_total_found)
        print_kv("SQL Executed", self.result.sql_executed)
        
        if self.result.errors:
            print_subsection("Errors Encountered")
            for err in self.result.errors:
                print_fail(err)
        
        # Overall Assessment
        print_header("TEST ASSESSMENT", char="═")
        
        score = 0
        total = 0
        
        assessments = [
            ("Intent Classification", self.result.intent in ["LOOKUP", "ANALYTICS"]),
            ("Vector Search Enabled", self.result.needs_vector),
            ("Found Transactions", self.result.search_total_found > 0),
            ("Search Quality Good", self.result.search_quality == "good"),
            ("Answer Generated", len(self.result.final_answer) > 100),
            ("Answer Contains YouTube", "youtube" in self.result.final_answer.lower()),
        ]
        
        for name, passed in assessments:
            total += 1
            if passed:
                score += 1
                print_ok(f"[PASS] {name}")
            else:
                print_fail(f"[FAIL] {name}")
        
        print(f"\n{Colors.BOLD}Overall Score: {score}/{total} ({score/total*100:.0f}%){Colors.ENDC}")
        
        if score == total:
            print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 ALL TESTS PASSED! Agent is working correctly.{Colors.ENDC}")
        elif score >= total * 0.7:
            print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠ PARTIAL SUCCESS - Some issues detected.{Colors.ENDC}")
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}❌ TEST FAILED - Critical issues found.{Colors.ENDC}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    TEST_QUERY = """Based on this dataset, do I have a YouTube subscription? If yes, when did I pay for it—please show the detailed payment history."""
    
    executor = DeepTestExecutor()
    result = executor.run(TEST_QUERY, tenant_id="default_tenant")
    
    print_header("TEST COMPLETE")
    print(f"\nTest finished at: {datetime.now().isoformat()}")
    print(f"Total execution time: {result.duration_ms}ms")

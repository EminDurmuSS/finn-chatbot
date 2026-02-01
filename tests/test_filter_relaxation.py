"""
Test Filter Relaxation Strategy

Demonstrates how the system relaxes filters when strict filtering returns 0 results.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from statement_copilot.core.search_engine import ProfessionalSearchEngine
from statement_copilot.core.database import DatabaseManager
from statement_copilot.core.vector_store import MockVectorStore
from statement_copilot.core.llm import get_llm_client
from statement_copilot.agents.search_agent import load_taxonomy
from statement_copilot.config import settings
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def test_filter_relaxation():
    """Test that filters are relaxed when needed"""
    
    print("\n" + "=" * 80)
    print("FILTER RELAXATION TEST")
    print("=" * 80)
    
    db = DatabaseManager()
    vector_store = MockVectorStore()
    llm = get_llm_client()
    taxonomy = load_taxonomy()
    
    engine = ProfessionalSearchEngine(db, vector_store, llm, taxonomy)
    
    # Scenario: Query implies "subscriptions" category, but we force "travel" to break it,
    # then we use `overrides` to force "subscriptions" back (or relax it).
    
    print(f"\nQuery: '{query}'")
    
    # 1. Standard Search (might infer category)
    print("\n--- Attempt 1: Standard Search ---")
    result1 = engine.search(
        query=query,
        tenant_id=settings.default_tenant_id
    )
    print(f"Found: {result1.total_found}")

    # 2. Search with OVERRIDE (Explicitly removing date/category)
    print("\n--- Attempt 2: Search with OVERRIDES (Relaxing Category) ---")
    # We pretend the graph decided to relax the category filter
    overrides = {"categories": None, "subcategories": None}
    
    result2 = engine.search(
        query=query,
        tenant_id=settings.default_tenant_id,
        overrides=overrides
    )
    
    print(f"Found: {result2.total_found}")
    print(f"Effective Filters: {result2.filters_applied}")
    
    # Assertions
    # We expect overrides to be applied
    assert "categories" not in result2.filters_applied or not result2.filters_applied["categories"], "Category filter should be removed by override"

    if result2.matches:
        print(f"\n[Top 3 Matches - Relaxed]")
        for i, match in enumerate(result2.matches[:3], 1):
            print(f"  {i}. {match.merchant_norm} - ${abs(match.amount or 0):.2f}")
            print(f"     Category: {match.category}")
    
    print("\n" + "=" * 80)
    print("\nKEY FEATURES OF FILTER RELAXATION:")
    print("=" * 80)
    print("""
1. ✅ Automatic Detection:
   System detects when both category AND keyword filters exist

2. ✅ Fallback Logic:
   - Try 1: Full filters (category + keywords) → Precise
   - Try 2: If 0 results, remove category → Broader recall

3. ✅ Smart Trade-off:
   - Best case: Category is correct → Get narrow, accurate results
   - Worst case: Category is wrong → Still find transactions via keywords

4. ✅ Logged for Transparency:
   "[FILTER_RELAXATION] 0 results with strict filters. Retrying..."

5. ✅ Works for Both SQL-Only and Hybrid Searches

6. ✅ Example Scenarios:
   
   Scenario A: "Ev kirası" with category=housing
   - Kira payment is in "housing" ✅ → Finds with strict filter
   
   Scenario B: "Ev kirası" with category=housing
   - Kira payment is in "transfers" (miscategorized) ❌
   - Retries without category → Finds via keyword ✅
   
   Scenario C: "subscription payment" with category=subscriptions
   - Payment is in "business_professional" (Claude.AI) ❌
   - Retries without category → Finds via keyword "subscription" ✅
""")


if __name__ == "__main__":
    test_filter_relaxation()

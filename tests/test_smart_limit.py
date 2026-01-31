"""
Test Smart Result Limiting
===========================
Verifies that COUNT(*) pre-check and dynamic limit expansion work correctly.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set encoding for Windows
os.environ['PYTHONIOENCODING'] = 'utf-8'

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

from statement_copilot.config import settings
from statement_copilot.core.database import get_db
from statement_copilot.core.vector_store import get_vector_store
from statement_copilot.core.search_engine import ProfessionalSearchEngine

def main():
    print("=" * 60)
    print("SMART RESULT LIMITING TEST")
    print("=" * 60)
    
    # Initialize components
    db = get_db()
    vector_store = get_vector_store()
    engine = ProfessionalSearchEngine(db=db, vector_store=vector_store)
    
    tenant_id = settings.default_tenant_id
    
    # Test 1: Query that should have small number of results (expand limit)
    print("\n" + "-" * 60)
    print("TEST 1: Small result set (should return ALL)")
    print("-" * 60)
    result1 = engine.search("Green Chef transactions", tenant_id, top_k=20)
    print(f"Query: 'Green Chef transactions'")
    print(f"  total_found: {result1.total_found}")
    print(f"  total_matching: {result1.total_matching}")
    print(f"  result_limited: {result1.result_limited}")
    print(f"  matches count: {len(result1.matches)}")
    
    # Test 2: Query that might have many results
    print("\n" + "-" * 60)
    print("TEST 2: Potentially large result set")
    print("-" * 60)
    result2 = engine.search("all transactions", tenant_id, top_k=20)
    print(f"Query: 'all transactions'")
    print(f"  total_found: {result2.total_found}")
    print(f"  total_matching: {result2.total_matching}")
    print(f"  result_limited: {result2.result_limited}")
    print(f"  matches count: {len(result2.matches)}")
    
    # Test 3: Query with specific merchant
    print("\n" + "-" * 60)
    print("TEST 3: Specific merchant (YouTube)")
    print("-" * 60)
    result3 = engine.search("tüm YouTube harcamalarım", tenant_id, top_k=20)
    print(f"Query: 'tüm YouTube harcamalarım'")
    print(f"  total_found: {result3.total_found}")
    print(f"  total_matching: {result3.total_matching}")
    print(f"  result_limited: {result3.result_limited}")
    print(f"  matches count: {len(result3.matches)}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()

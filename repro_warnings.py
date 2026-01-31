
import sys
import os
import logging
from datetime import datetime
import json
import uuid

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s'
)

# Reduce noise
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("org.apache.http").setLevel(logging.WARNING)
logging.getLogger("elasticsearch").setLevel(logging.WARNING)

def test_warnings_accumulation():
    from statement_copilot import StatementCopilot
    
    print("\n" + "="*80)
    print(" WARNING ACCUMULATION TEST")
    print("="*80)
    
    copilot = StatementCopilot()
    session_id = f"test_warn_{uuid.uuid4().hex[:8]}"
    
    # Message that triggers a guardrail warning "Wide scope query"
    query = "show all data" 
    
    # --- Turn 1 ---
    print(f"\n[TURN 1] Sending: '{query}'")
    result1 = copilot.chat(
        message=query,
        session_id=session_id
    )
    
    warnings1 = result1.get("warnings", [])
    print(f"Warnings (Turn 1): {warnings1}")
    
    # Check if we got the expected warning
    if not any("Wide scope query" in w for w in warnings1):
        print("WARNING: Expected 'Wide scope query' warning not found in Turn 1. Logic might be different.")
    
    len1 = len(warnings1)
    
    # --- Turn 2 ---
    print(f"\n[TURN 2] Sending: '{query}'")
    result2 = copilot.chat(
        message=query,
        session_id=session_id
    )
    
    warnings2 = result2.get("warnings", [])
    print(f"Warnings (Turn 2): {warnings2}")
    
    len2 = len(warnings2)
    
    print("\n" + "-"*50)
    print(f"Turn 1 warnings count: {len1}")
    print(f"Turn 2 warnings count: {len2}")
    
    if len2 > len1:
        print("\n[FAIL] Warnings are accumulating! (Counts should be equal or similar per turn)")
    elif len2 == len1:
        print("\n[PASS] Warnings counts are stable (not accumulating).")
    else:
        print("\n[INFO] Turn 2 has fewer warnings? That's fine too.")

if __name__ == "__main__":
    try:
        test_warnings_accumulation()
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

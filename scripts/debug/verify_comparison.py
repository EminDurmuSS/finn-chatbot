import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from statement_copilot.agents.orchestrator import OrchestratorAgent
from statement_copilot.core.state import OrchestratorState

def verify():
    print("Initializing Orchestrator...")
    orchestrator = OrchestratorAgent()
    
    # Test Case 1: Delivery vs Grocery
    query = "Is food delivery spending increasing vs grocery spending (2023-2025)?"
    print(f"\nAnalyzing Query: {query}")
    
    state = OrchestratorState(user_message=query, message_history=[])
    result = orchestrator.route(state)
    
    constraints = result.get("constraints", {})
    subcats = constraints.get("subcategories", [])
    
    print("\n--- Result ---")
    print(f"Intent: {result.get('intent')}")
    print(f"Subcategories: {subcats}")
    
    expected = {"delivery", "groceries"}
    found = set(subcats) if subcats else set()
    
    if expected.issubset(found):
        print(f"SUCCESS: Found both {expected}")
    else:
        print(f"FAILURE: Expected {expected}, found {found}")
        # Don't exit yet, run next test

    # Test Case 2: Uber vs Public Transit
    query2 = "Compare spending on uber vs public transport this year"
    print(f"\nAnalyzing Query 2: {query2}")
    
    state2 = OrchestratorState(user_message=query2, message_history=[])
    result2 = orchestrator.route(state2)
    
    constraints2 = result2.get("constraints", {})
    subcats2 = constraints2.get("subcategories", [])
    
    print("--- Result 2 ---")
    print(f"Intent: {result2.get('intent')}")
    print(f"Subcategories: {subcats2}")
    
    expected2 = {"taxi_rideshare", "public_transit"}
    found2 = set(subcats2) if subcats2 else set()
    
    if expected2.issubset(found2):
        print(f"SUCCESS: Found both {expected2}")
    else:
        print(f"FAILURE: Expected {expected2}, found {found2}")
        sys.exit(1)

if __name__ == "__main__":
    verify()

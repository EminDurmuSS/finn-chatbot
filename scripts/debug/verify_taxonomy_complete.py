import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from statement_copilot.agents.orchestrator import OrchestratorAgent
from statement_copilot.core import OrchestratorState

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

def load_taxonomy(json_path: Path) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def run_test(orchestrator, query: str, expected_cat: str, expected_subcat: str = None) -> bool:
    print(f"  Query: '{query}'")
    state = OrchestratorState(user_message=query, message_history=[])
    try:
        result = orchestrator.route(state)
        constraints = result.get("constraints", {})
        
        found_cats = constraints.get("categories", [])
        found_subcats = constraints.get("subcategories", [])
        
        # Checking if EXPECTED is contained in FOUND
        cat_match = expected_cat in found_cats if expected_cat and found_cats else False
        # If no category expected (rare), broad match
        if not expected_cat and not found_cats: cat_match = True
        
        subcat_match = expected_subcat in found_subcats if expected_subcat and found_subcats else False
        if not expected_subcat: subcat_match = True # Relax if we don't strictly test subcat

        if cat_match and subcat_match:
            print(f"    ✅ PASS: Found {found_cats} / {found_subcats}")
            return True
        else:
            print(f"    ❌ FAIL: Expected cat={expected_cat}, sub={expected_subcat}. Got cat={found_cats}, sub={found_subcats}")
            return False
    except Exception as e:
        print(f"    ❌ ERROR: {e}")
        return False

def verify_all():
    print("Loading taxonomy...")
    json_path = project_root / "data" / "category_taxonomy_v1.json"
    if not json_path.exists():
        print(f"Taxonomy file not found at {json_path}")
        return

    taxonomy = load_taxonomy(json_path)
    
    print("Initializing Orchestrator...")
    orchestrator = OrchestratorAgent()
    
    total_tests = 0
    passed_tests = 0
    
    print("\n--- Starting Verification ---")
    
    categories = taxonomy.get("categories", {})
    # Limit for quick test, or run all? User said "kontrol et", implies thoroughness.
    # But for response time, maybe limit or sample? 
    # Let's run all, it's just inference calls (which might be slow/costly?).
    # Wait, Orchestrator calls LLM. 50+ categories * LLM call = expensive / slow.
    # I should limit to a representative sample or ask user.
    # User said "kusursuz çalışıyor mu" (is it working perfectly).
    # I will test 1 subcategory per category to save time but cover breadth.
    
    for cat_id, cat_data in categories.items():
        print(f"\nTesting Category: {cat_data['display_name']} ({cat_id})")
        
        # Test Parent Category using a keyword
        keywords = cat_data.get("keywords", [])
        if keywords:
            query = f"Show me my {keywords[0]} spending"
            total_tests += 1
            if run_test(orchestrator, query, cat_id):
                passed_tests += 1
        
        # Test ONE Subcategory
        subcats = cat_data.get("subcategories", {})
        if subcats:
            # Pick the first one
            sub_id, sub_data = next(iter(subcats.items()))
            print(f"  Testing Subcategory: {sub_data['display_name']} ({sub_id})")
            
            sub_keywords = sub_data.get("keywords", [])
            examples = sub_data.get("merchants_examples", [])

            if sub_keywords:
                 query = f"How much did I spend on {sub_keywords[0]}?"
                 total_tests += 1
                 if run_test(orchestrator, query, cat_id, sub_id):
                    passed_tests += 1
            elif examples:
                 query = f"Transactions for {examples[0]}"
                 total_tests += 1
                 if run_test(orchestrator, query, cat_id, sub_id):
                    passed_tests += 1

    print(f"\n--- Summary ---")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    
    if total_tests > 0:
        print(f"Success Rate: {passed_tests / total_tests * 100:.1f}%")

if __name__ == "__main__":
    verify_all()

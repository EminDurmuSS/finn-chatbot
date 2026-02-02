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
        
        cat_match = expected_cat in found_cats if expected_cat and found_cats else False
        if not expected_cat and not found_cats: cat_match = True
        
        subcat_match = expected_subcat in found_subcats if expected_subcat and found_subcats else False
        if not expected_subcat: subcat_match = True

        if cat_match and subcat_match:
            print(f"    ✅ PASS: Found {found_cats} / {found_subcats}")
            return True
        else:
            print(f"    ❌ FAIL: Expected cat={expected_cat}, sub={expected_subcat}. Got cat={found_cats}, sub={found_subcats}")
            return False
    except Exception as e:
        print(f"    ❌ ERROR: {e}")
        return False

def verify_quick():
    print("Loading taxonomy...")
    json_path = project_root / "data" / "category_taxonomy_v1.json"
    taxonomy = load_taxonomy(json_path)
    
    print("Initializing Orchestrator...")
    orchestrator = OrchestratorAgent()
    
    # Subset of categories to test
    target_cats = ["food_and_dining", "transport", "utilities"]
    
    for cat_id in target_cats:
        cat_data = taxonomy["categories"][cat_id]
        print(f"\nTesting Category: {cat_data['display_name']} ({cat_id})")
        
        # Test Parent
        keywords = cat_data.get("keywords", [])
        if keywords:
            query = f"Show me my {keywords[0]} spending"
            run_test(orchestrator, query, cat_id)
        
        # Test ONE Subcategory
        subcats = cat_data.get("subcategories", {})
        if subcats:
            sub_id, sub_data = next(iter(subcats.items()))
            examples = sub_data.get("merchants_examples", [])
            if examples:
                 query = f"Transactions for {examples[0]}"
                 run_test(orchestrator, query, cat_id, sub_id)

if __name__ == "__main__":
    verify_quick()

import sys
import logging
from pathlib import Path
from pprint import pprint

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from statement_copilot.agents.search_agent import ProfessionalSearchAgent
from statement_copilot.core import OrchestratorState
from statement_copilot.config import settings

# Disable LLM for unit test speed & isolation
settings.enable_search_llm = False

# Configure logging to capture evidence
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_constraints_usage():
    print("Initializing ProfessionalSearchAgent...")
    agent = ProfessionalSearchAgent()
    
    # Test Case: Query without keywords, but with Orchestrator Constraints
    # If the Agent respects constraints, the result MUST have category filter applied.
    query = "items"
    forced_constraints = {
        "categories": ["food_and_dining"],
        "subcategories": ["groceries"],
        "merchants": ["TEST_MERCHANT_SHOULD_NOT_EXIST"]
    }
    
    print(f"\n--- Test: Query='{query}' with Forced Constraints ---")
    print(f"Constraints: {forced_constraints}")
    
    state = OrchestratorState(
        user_message=query,
        constraints=forced_constraints,
        tenant_id="default_tenant"
    )
    
    # Run search
    result_state = agent.search(state)
    
    # Check evidence
    evidence = result_state.get("search_evidence", {})
    filters_applied = evidence.get("filters_applied", {})
    
    print("\n--- Result Evidence ---")
    pprint(filters_applied)
    
    # Verification
    has_category = "categories" in filters_applied and "food_and_dining" in filters_applied["categories"]
    has_merchant = "merchants" in filters_applied and "TEST_MERCHANT_SHOULD_NOT_EXIST" in filters_applied["merchants"]
    
    if has_category and has_merchant:
        print("\n✅ PASS: Search Agent USED the constraints.")
    else:
        print("\n❌ FAIL: Search Agent IGNORED the constraints.")
        print(f"   Missing Categories? {not has_category}")
        print(f"   Missing Merchants? {not has_merchant}")

if __name__ == "__main__":
    verify_constraints_usage()

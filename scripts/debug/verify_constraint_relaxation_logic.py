import sys
import logging
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from statement_copilot.agents.search_graph import transform_query_node, TransformResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_relaxation_logic():
    print("--- Verifying Constraint Relaxation Logic ---")

    # Initial state with a "bad" constraint
    initial_state = {
        "original_query": "Gas Station",
        "current_query": "Gas Station",
        "constraints": {"categories": ["Food"]}, # Bad constraint
        "critique": "No results found in Food category.",
        "attempt": 1
    }

    # Mock the LLM to return a relaxation decision
    mock_llm_response = TransformResult(
        refined_query="Shell Station",
        updated_constraints={"categories": ["transport"], "subcategories": ["fuel"]}, # Switching category
        reasoning="Switching from Food to Transport/Fuel based on valid taxonomy"
    )

    # Patch the get_llm_client AND get_professional_search_agent
    with patch("statement_copilot.agents.search_graph.get_llm_client") as mock_get_llm, \
         patch("statement_copilot.agents.search_graph.get_professional_search_agent") as mock_get_agent:
        
        # Setup LLM mock
        mock_client = MagicMock()
        mock_client.complete_structured.return_value = mock_llm_response
        mock_get_llm.return_value = mock_client
        
        # Setup Taxonomy Mock
        mock_agent = MagicMock()
        mock_agent.taxonomy = {
            "categories": {
                "transport": {"subcategories": {"fuel": {}, "taxi": {}}},
                "food_and_dining": {"subcategories": {"groceries": {}}}
            }
        }
        mock_get_agent.return_value = mock_agent

        # Run the node
        print(f"Input Constraints: {initial_state['constraints']}")
        new_state = transform_query_node(initial_state)
        
        # Check results
        new_constraints = new_state.get("constraints")
        print(f"Output Constraints: {new_constraints}")
        
        if new_constraints == {"categories": ["transport"], "subcategories": ["fuel"]}:
            print("✅ PASS: Constraints were updated to new VALID category.")
        else:
            print("❌ FAIL: Constraints were NOT updated correctly.")

if __name__ == "__main__":
    test_relaxation_logic()


import asyncio
import sys
import os
from unittest.mock import MagicMock

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from statement_copilot.agents.orchestrator import ResponseSynthesizer
from statement_copilot.core import OrchestratorState

async def verify_currency_context():
    """Verify that Synthesizer respects the injected currency context."""
    print("Initializing Synthesizer...")
    synthesizer = ResponseSynthesizer()
    
    # Mock LLM to avoid actual API calls, but we want to check the PROMPT content
    # However, simpler to actually run it if we can, or mock the complete method to inspect arguments.
    # Let's mock the LLM client to inspect the prompt passed to it.
    original_llm = synthesizer.llm
    mock_llm = MagicMock()
    synthesizer.llm = mock_llm
    
    # Create a dummy state with explicit currency
    state: OrchestratorState = {
        "user_message": "How much did I spend?",
        "intent": "ANALYTICS",
        "constraints": {"currency": "TRY"},
        "sql_result": {
            "value": 1500.50,
            "tx_count": 5
        },
        "guardrail_passed": True
    }
    
    print("Running synthesis with currency=TRY...")
    try:
        synthesizer.synthesize(state)
        
        # Check call args
        call_args = mock_llm.complete.call_args
        if call_args:
            prompt_sent = call_args.kwargs.get("prompt", "")
            print(f"\n--- PROMPT SENT TO LLM ---\n{prompt_sent}\n--------------------------")
            
            if "CONTEXT: All monetary amounts are in TRY" in prompt_sent:
                print("\u2705 SUCCESS: Currency context 'TRY' found in prompt.")
            else:
                print("\u274c FAILED: Currency context 'TRY' NOT found in prompt.")
                
            if "SQL Result" in prompt_sent and "Value: 1,500.50" in prompt_sent:
                 print("\u2705 SUCCESS: SQL result included correctly.")
            else:
                 print("\u274c FAILED: SQL result integration issue.")
                 
        else:
            print("\u274c FAILED: LLM complete method was not called.")
            
    except Exception as e:
        print(f"\u274c ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(verify_currency_context())

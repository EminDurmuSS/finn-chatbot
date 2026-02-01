
import sys
import unittest
import logging
from unittest.mock import MagicMock, patch
from typing import Dict, Any

# Adjust path to find the module
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock things before importing workflow if necessary, but patching is safer
from statement_copilot.workflow import StatementCopilot
from statement_copilot.core import OrchestratorState, IntentType, Constraints

# Configure logging to see our debug prints
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_logger")

class MockOrchestrator:
    def route(self, state: OrchestratorState) -> OrchestratorState:
        state['intent'] = IntentType.CHITCHAT.value
        state['confidence'] = 0.99
        state['needs_sql'] = False
        state['needs_vector'] = False
        state['needs_planner'] = False
        return state

class MockSynthesizer:
    def synthesize(self, state: OrchestratorState) -> OrchestratorState:
        state['final_answer'] = f"Response to: {state['user_message']}"
        state['evidence'] = {}
        return state

class MockGuardrails:
    def process_state(self, state: OrchestratorState) -> OrchestratorState:
        state['guardrail_passed'] = True
        return state
    
    def mask_output(self, text: str) -> str:
        return text

class TestStateIssues(unittest.TestCase):
    
    @patch('statement_copilot.workflow.get_orchestrator')
    @patch('statement_copilot.workflow.get_synthesizer')
    @patch('statement_copilot.workflow.get_guardrails')
    @patch('statement_copilot.workflow.get_response_validator')
    def test_history_duplication(self, mock_validator, mock_guard, mock_synth, mock_orch):
        """Test if message_history gets duplicated across turns"""
        
        # Setup mocks
        mock_orch.return_value = MockOrchestrator()
        mock_synth.return_value = MockSynthesizer()
        mock_guard.return_value = MockGuardrails()
        
        mock_validator_instance = MagicMock()
        mock_validator_instance.validate.return_value = {"issues": [], "corrected_answer": None}
        mock_validator.return_value = mock_validator_instance
        
        # Initialize copilot
        copilot = StatementCopilot(checkpointer=None) # Uses MemorySaver
        session_id = "test_dup_1"
        
        print("\n--- TURN 1 ---")
        history_turn_1 = []
        result1 = copilot.chat("Hello 1", session_id=session_id, message_history=history_turn_1)
        
        # Check checkpoint state
        config = {"configurable": {"thread_id": session_id}}
        state1 = copilot.graph.get_state(config).values
        hist1 = state1.get('message_history')
        print(f"Turn 1 State History: {hist1}")
        
        # Assertion 1: Should be empty as we passed empty
        # Note: If input was empty, and no node updated it, it should be empty.
        
        print("\n--- TURN 2 ---")
        # Client sends history from previous turn
        history_turn_2 = [
            {"role": "user", "content": "Hello 1"},
            {"role": "assistant", "content": result1['answer']}
        ]
        
        result2 = copilot.chat("Hello 2", session_id=session_id, message_history=history_turn_2)
        
        state2 = copilot.graph.get_state(config).values
        hist2 = state2.get('message_history')
        print(f"Turn 2 State History: {hist2}")
        print(f"Turn 2 Input History Length: {len(history_turn_2)}")
        print(f"Turn 2 State History Length: {len(hist2)}")
        
        # If duplicated, Length will be > 2. 
        # Specifically, if hist1 was saved as [], and we append [2 items], result is 2 items. OK.
        
        print("\n--- TURN 3 ---")
        # Client sends cumulative history
        history_turn_3 = history_turn_2 + [
            {"role": "user", "content": "Hello 2"},
            {"role": "assistant", "content": result2['answer']}
        ]
        
        result3 = copilot.chat("Hello 3", session_id=session_id, message_history=history_turn_3)
        
        state3 = copilot.graph.get_state(config).values
        hist3 = state3.get('message_history')
        print(f"Turn 3 Input History Length: {len(history_turn_3)}")  # Should be 4
        print(f"Turn 3 State History Length: {len(hist3)}")
        
        # Logic Warning:
        # If Turn 2 saved state had 'hist2' (length 2).
        # And Turn 3 apps 'history_turn_3' (length 4).
        # Then resulting state will have 2 + 4 = 6 items! DUPLICATION!
        
        if len(hist3) > len(history_turn_3):
            print("!!! DUPLICATION DETECTED !!!")
        else:
            print("No duplication detected.")

    @patch('statement_copilot.workflow.get_orchestrator')
    @patch('statement_copilot.workflow.get_synthesizer')
    @patch('statement_copilot.workflow.get_guardrails')
    @patch('statement_copilot.workflow.get_response_validator')
    def test_constraint_persistence(self, mock_validator, mock_guard, mock_synth, mock_orch):
        """Test if constraints persist across turns"""
        
        # Setup mocks
        orch = MockOrchestrator()
        # Orchestrator that sets constraints in Turn 1
        def route_turn1(state):
            state['intent'] = IntentType.ANALYTICS.value
            state['constraints'] = {'category': 'Food'} # Found "Food"
            state['needs_sql'] = False
            state['needs_vector'] = False
            state['needs_planner'] = False
            state['confidence'] = 1.0
            return state
            
        def route_turn2(state):
            state['intent'] = IntentType.ANALYTICS.value
            # Orchestrator doesn't find new constraints, so it shouldn't overwrite if we want persistence?
            # actually logic says orchestrator re-evaluates. 
            # If Orchestrator returns decision.constraints as {}, it overwrites state['constraints']
            # So persistence depends on Orchestrator logic seeing history.
            # BUT let's see if the INPUT (create_initial_state) wipes it first.
            state['needs_sql'] = False
            state['needs_vector'] = False
            state['needs_planner'] = False
            state['confidence'] = 1.0
            # Note: route() usually sets state['constraints'] = constraints_dict.
            # If we don't set it in mock, we see if it was preserved from input.
            return state

        mock_orch.return_value = orch
        mock_synth.return_value = MockSynthesizer()
        mock_guard.return_value = MockGuardrails()
        mock_validator_instance = MagicMock()
        mock_validator_instance.validate.return_value = {"issues": [], "corrected_answer": None}
        mock_validator.return_value = mock_validator_instance

        copilot = StatementCopilot(checkpointer=None)
        session_id = "test_const_1"
        
        # Turn 1
        orch.route = route_turn1
        copilot.chat("Show food", session_id=session_id)
        
        state1 = copilot.graph.get_state({"configurable": {"thread_id": session_id}}).values
        print(f"\nTurn 1 Constraints: {state1.get('constraints')}") # Should be {'category': 'Food'}
        
        # Turn 2
        # Verify if create_initial_state wiped constraints before Orchestrator even runs
        # We can't easily hook into "before orchestrator" without modifying code or complex mocking.
        # But we can verify if constraints are in the state passed to orchestrator?
        # We can make mock_orch.route print the incoming state constraints.
        
        def route_check_incoming(state):
            print(f"Turn 2 Incoming Constraints to Orchestrator: {state.get('constraints')}")
            # Do nothing to constraints, see if they stay
            state['intent'] = IntentType.CHITCHAT.value
            state['needs_sql'] = False
            state['needs_vector'] = False
            state['needs_planner'] = False
            state['confidence'] = 1.0
            return state
            
        orch.route = route_check_incoming
        copilot.chat("And drinks?", session_id=session_id)
        
        state2 = copilot.graph.get_state({"configurable": {"thread_id": session_id}}).values
        print(f"Turn 2 Final Constraints: {state2.get('constraints')}")

if __name__ == '__main__':
    unittest.main()

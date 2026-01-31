"""
Statement Copilot - Response Validator Agent
=============================================
Validates synthesized responses against evidence and fixes if needed.
"""

import json
import logging
import time
from typing import Dict, Any, Optional

from ..config import settings
from ..core import (
    OrchestratorState,
    ResponseValidation,
    IntentType,
    create_tool_call_record,
    get_llm_client,
)
from ..core.prompts import get_response_validator_prompt

logger = logging.getLogger(__name__)


class ResponseValidatorAgent:
    """
    Validates the final answer using available evidence.
    If the answer contains unsupported facts, returns a corrected version.
    """

    def __init__(self):
        self.llm = get_llm_client()
        self.model = settings.model_validator

    def validate(self, state: OrchestratorState) -> Dict[str, Any]:
        """
        Validate final answer with evidence.

        Args:
            state: Orchestrator state

        Returns:
            dict with is_valid, issues, corrected_answer
        """
        if not settings.enable_response_validator:
            return {"is_valid": True, "issues": []}

        final_answer = state.get("final_answer") or ""
        if not final_answer.strip():
            return {"is_valid": True, "issues": []}

        # Build evidence context
        intent = state.get("intent")
        if intent in [
            IntentType.CHITCHAT.value,
            IntentType.CLARIFY.value,
            IntentType.EXPLAIN.value,
        ]:
            return {"is_valid": True, "issues": []}

        evidence = {
            "intent": state.get("intent"),
            "constraints": state.get("constraints"),
            "sql_result": state.get("sql_result"),
            "vector_result": state.get("vector_result"),
            "action_plan": state.get("action_plan"),
            "action_result": state.get("action_result"),
        }

        prompt = (
            "## USER MESSAGE\n"
            f"{state.get('user_message', '')}\n\n"
            "## EVIDENCE\n"
            f"{json.dumps(evidence, ensure_ascii=False)}\n\n"
            "## ANSWER\n"
            f"{final_answer}\n\n"
            "## TASK\n"
            "Validate the answer against the evidence. If it conflicts, produce a corrected answer using only the evidence."
        )

        try:
            start_time = time.time()
            result: ResponseValidation = self.llm.complete_structured(
                prompt=prompt,
                response_model=ResponseValidation,
                model=self.model,
                system=get_response_validator_prompt(),
                temperature=0.0,
            )
            latency_ms = int((time.time() - start_time) * 1000)

            # Optional audit record
            try:
                tool_call = create_tool_call_record(
                    state=state,
                    node="response_validator",
                    tool_name="validate",
                    model_name=self.model,
                    input_data={"answer": final_answer},
                    output_data=result.model_dump(),
                    latency_ms=latency_ms,
                    success=True,
                )
                state["tool_calls"] = [tool_call]
            except Exception:
                pass

            return result.model_dump()

        except Exception as e:
            logger.error(f"Response validation failed: {e}")
            return {"is_valid": True, "issues": []}


_response_validator: Optional[ResponseValidatorAgent] = None


def get_response_validator() -> ResponseValidatorAgent:
    """Get or create response validator singleton"""
    global _response_validator
    if _response_validator is None:
        _response_validator = ResponseValidatorAgent()
    return _response_validator

"""
Statement Copilot - Orchestrator Agent
======================================
Central brain that routes to appropriate agents.

bunq Alignment: Orchestrator pattern - merkezi beyin, 3-5 primary agent

UPDATED: Removed default "this_month" - now searches all time when no date specified
"""

import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime

from ..config import settings
from ..core import (
    OrchestratorState,
    RouterDecision,
    IntentType,
    Constraints,
    RiskLevel,
    Period,
    create_tool_call_record,
    get_llm_client,
    format_message_history,
)
from ..log_context import clip_text, format_kv, format_list
from ..core.prompts import get_orchestrator_prompt

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# ORCHESTRATOR AGENT
# -----------------------------------------------------------------------------

class OrchestratorAgent:
    """
    Central orchestrator that:
    1. Classifies user intent
    2. Extracts constraints (date, category, merchant)
    3. Routes to appropriate agents
    
    Uses Anthropic structured outputs for guaranteed JSON.
    """
    
    def __init__(self):
        self.llm = get_llm_client()
        self.model = settings.model_orchestrator
    
    def route(self, state: OrchestratorState) -> OrchestratorState:
        """
        Analyze user message and determine routing.
        
        Args:
            state: Current orchestrator state
            
        Returns:
            Updated state with routing decision
        """
        user_message = state.get("user_message", "")
        message_history = state.get("message_history", [])
        
        # Format history for context
        history_text = format_message_history(message_history, max_messages=5)

        logger.debug(
            "Orchestrator context: message_len=%s history_len=%s",
            len(user_message),
            len(message_history),
        )
        
        # Build prompt
        prompt = f"""
# -----------------------------------------------------------------------------
{history_text}

## USER MESSAGE
{user_message}

# -----------------------------------------------------------------------------
Analyze this message and decide the routing.
Remember: If user does NOT specify a date/time period, do NOT add any date constraints.
"""
        
        start_time = time.time()
        
        try:
            # Get structured decision from LLM
            decision: RouterDecision = self.llm.complete_structured(
                prompt=prompt,
                response_model=RouterDecision,
                model=self.model,
                system=get_orchestrator_prompt(),
                temperature=0.0
            )
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            logger.info(
                f"Orchestrator decision: intent={decision.intent.value}, "
                f"confidence={decision.confidence:.2f}, "
                f"sql={decision.needs_sql}, vector={decision.needs_vector}, "
                f"planner={decision.needs_planner}"
            )

            logger.debug(
                "Orchestrator details: reasoning=%s risk=%s constraints=%s clarify=%s suggested_questions=%s",
                clip_text(decision.reasoning, 200),
                decision.risk_level.value if decision.risk_level else None,
                format_kv(self._constraints_to_dict(decision.constraints), max_items=6, max_value_len=120),
                clip_text(decision.clarification_needed, 120),
                format_list(decision.suggested_questions or [], max_items=4, max_value_len=80),
            )
            
            # Update state with decision
            state["intent"] = decision.intent.value
            state["confidence"] = decision.confidence
            state["reasoning"] = decision.reasoning
            
            state["needs_sql"] = decision.needs_sql
            state["needs_vector"] = decision.needs_vector
            state["needs_planner"] = decision.needs_planner

            state["risk_level"] = decision.risk_level.value
            state["clarification_needed"] = decision.clarification_needed
            state["suggested_questions"] = decision.suggested_questions
            
            # Convert constraints to dict
            constraints_dict = self._constraints_to_dict(decision.constraints)
            
            # REMOVED: No longer add default "this_month" when no date specified
            # The system will now search ALL available data when no date is provided
            # This is the desired behavior for questions like "Did I ever buy from X?"
            
            # SQL-FIRST OPTIMIZATION PATCH
            # When query has explicit merchant + analytics intent, skip vector search
            # Vector: 30+ seconds, SQL: 40ms
            decision = self._apply_sql_first_optimization(decision, user_message)
            
            state["constraints"] = constraints_dict
            
            # Add tool call record
            tool_call = create_tool_call_record(
                state=state,
                node="orchestrator",
                tool_name="route",
                model_name=self.model,
                input_data={"message": user_message},
                output_data=decision.model_dump(mode="json"),
                latency_ms=latency_ms,
                success=True
            )
            state["tool_calls"] = [tool_call]

            return state

        except Exception as e:
            logger.error(f"Orchestrator error: {e}")
            
            # Fallback to CHITCHAT on error
            state["intent"] = IntentType.CHITCHAT.value
            state["confidence"] = 0.0
            state["needs_sql"] = False
            state["needs_vector"] = False
            state["needs_planner"] = False
            state["errors"] = [str(e)]

            return state
    
    def _constraints_to_dict(self, constraints: Constraints) -> Dict[str, Any]:
        """Convert Constraints model to dict, handling date serialization"""
        data = constraints.model_dump(exclude_none=True)
        
        # Convert date objects to ISO strings
        if "date_range" in data and data["date_range"]:
            dr = data["date_range"]
            if "start" in dr:
                dr["start"] = dr["start"].isoformat() if hasattr(dr["start"], 'isoformat') else dr["start"]
            if "end" in dr:
                dr["end"] = dr["end"].isoformat() if hasattr(dr["end"], 'isoformat') else dr["end"]
        
        # Handle implicit period
        if "implicit_period" in data and data["implicit_period"]:
            # Convert enum to string value
            period = data["implicit_period"]
            if hasattr(period, 'value'):
                data["implicit_period"] = period.value
        
        return data
    
    def should_clarify(self, state: OrchestratorState) -> bool:
        """Check if we need clarification from user"""
        return state.get("intent") == IntentType.CLARIFY.value
    
    def get_clarification_question(self, state: OrchestratorState) -> str:
        """Generate clarification question based on state"""
        constraints = state.get("constraints", {})
        
        questions = []
        
        # Note: We no longer ask for date range by default
        # Only ask if the request is genuinely ambiguous
        
        # Ambiguous request
        if state.get("confidence", 0) < 0.5:
            questions.append("Could you be more specific about what you're looking for?")
        
        return questions[0] if questions else "Could you provide a bit more detail?"

    def _apply_sql_first_optimization(
        self, 
        decision: RouterDecision, 
        user_message: str
    ) -> RouterDecision:
        """
        SQL-First Routing Optimization
        
        When query has:
        - Explicit merchant (YouTube, Netflix, etc.)
        - Analytics intent (total, sum, how much, spent, spending)
        - NOT semantic query (like, similar, same as)
        
        Then: Skip vector search (30+ sec) and use SQL only (40ms)
        
        This reduces response time from 47-282 seconds to 1-3 seconds
        for explicit merchant analytics queries.
        """
        msg_lower = user_message.lower()
        
        # Check if query has explicit merchant
        has_explicit_merchant = bool(
            decision.constraints.merchants or 
            decision.constraints.merchant_contains
        )
        
        # Analytics keywords that indicate aggregation queries
        analytics_keywords = [
            "total", "sum", "how much", "spent", "spending",
            "calculate", "amount", "ne kadar", "toplam", "harcama"
        ]
        is_analytics = any(kw in msg_lower for kw in analytics_keywords)
        
        # Semantic indicators that DO need vector search
        semantic_indicators = [
            "like", "similar", "same as", "benzer", "gibi", "andıran"
        ]
        is_semantic_query = any(ind in msg_lower for ind in semantic_indicators)
        
        # RULE: Explicit Merchant + Analytics + Not Semantic = SQL Only
        if has_explicit_merchant and is_analytics and not is_semantic_query:
            old_vector = decision.needs_vector
            decision.needs_vector = False
            decision.needs_sql = True
            
            logger.info(
                "[PATCH] SQL-First optimization: disabled vector search "
                f"(was={old_vector}), routing to SQL only. "
                f"merchant={decision.constraints.merchants or decision.constraints.merchant_contains}"
            )
        
        # NEW RULE: Description-based search - INTELLIGENT DETECTION
        # Instead of hardcoding keywords, we check if content_keywords were extracted
        # by QueryUnderstandingEngine. If yes, user is searching for specific content
        # that might be in descriptions (not just merchant names or categories).
        # This is DYNAMIC - LLM and extraction logic decide what's a content keyword.
        has_content_keywords = bool(
            hasattr(decision.constraints, 'content_keywords') and 
            decision.constraints.content_keywords
        )
        
        # If query has content keywords, enable DUAL search for maximum recall
        if has_content_keywords:
            decision.needs_sql = True
            decision.needs_vector = True  # Keep vector for semantic understanding
            
            logger.info(
                "[INTELLIGENT] Description-based search detected via content_keywords: %s. "
                "Enabling DUAL search (SQL + Vector) for maximum recall.",
                decision.constraints.content_keywords[:3] if len(decision.constraints.content_keywords) > 3 
                else decision.constraints.content_keywords
            )
        
        return decision


# -----------------------------------------------------------------------------
# RESPONSE SYNTHESIZER
# -----------------------------------------------------------------------------

class ResponseSynthesizer:
    """
    Synthesizes final response from agent results.
    Converts technical data to user-friendly English text.
    """
    
    def __init__(self):
        self.llm = get_llm_client()
        self.model = settings.model_synthesizer
    
    def synthesize(self, state: OrchestratorState) -> OrchestratorState:
        """
        Generate final answer from agent results.
        
        Args:
            state: State with agent results
            
        Returns:
            State with final_answer populated
        """
        intent = state.get("intent")
        user_message = state.get("user_message", "")
        
        # Check if blocked
        if not state.get("guardrail_passed", True):
            state["final_answer"] = state.get("blocked_reason", "This request cannot be processed.")
            return state
        
        # Handle different intents
        if intent == IntentType.CHITCHAT.value:
            state["final_answer"] = self._handle_chitchat(user_message)
            return state
        
        if intent == IntentType.CLARIFY.value:
            state["final_answer"] = self._handle_clarify(state)
            return state
        
        if intent == IntentType.EXPLAIN.value:
            state["final_answer"] = self._handle_explain(state)
            return state
        
        # Collect results from agents
        results = {
            "sql_result": state.get("sql_result"),
            "vector_result": state.get("vector_result"),
            "action_plan": state.get("action_plan"),
        }

        # If validator found issues and no usable results, return a clear message
        validation_issues = state.get("validation_issues", [])
        if validation_issues and not any(results.values()):
            state["final_answer"] = (
                "Some results could not be produced: "
                + "; ".join(validation_issues)
                + ". Please try again or add more detail to your request."
            )
            return state
        
        # Check for errors
        errors = []
        if state.get("sql_error"):
            errors.append(f"SQL error: {state['sql_error']}")
        if state.get("search_error"):
            errors.append(f"Search error: {state['search_error']}")
        
        if errors and not any(results.values()):
            state["final_answer"] = "Sorry, something went wrong. Please try again."
            state["errors"] = errors
            return state
        
        # Synthesize response
        try:
            response = self._synthesize_response(user_message, results, state)
            state["final_answer"] = response["answer"]
            state["suggestions"] = response.get("suggestions")
            state["follow_up_questions"] = response.get("follow_up_questions")
            
            # Build evidence
            state["evidence"] = self._build_evidence(state, results)
            
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            state["final_answer"] = self._fallback_response(results)
            state["errors"] = [str(e)]
        
        return state
    
    def _handle_chitchat(self, message: str) -> str:
        """Handle chitchat messages"""
        message_lower = message.lower()
        
        if any(g in message_lower for g in ["hello", "hi", "hey"]):
            return "Hello! How can I help? I can analyze your spending, find specific transactions, or create reports."
        
        if any(t in message_lower for t in ["thanks", "thank you", "appreciate it"]):
            return "You're welcome! Anything else I can help with?"
        
        if any(h in message_lower for h in ["how are you", "how's it going"]):
            return "I'm doing well, thanks! I'm ready to analyze your financial data. What would you like me to do?"
        
        return "How can I help? You can ask about your spending or request reports."
    
    def _handle_clarify(self, state: OrchestratorState) -> str:
        """Generate clarification request"""
        clarification_needed = state.get("clarification_needed")
        suggested_questions = state.get("suggested_questions") or []

        if clarification_needed:
            return clarification_needed
        if suggested_questions:
            return suggested_questions[0]
        
        return "Could you share more detail? What kind of analysis or action do you want?"

    def _handle_explain(self, state: OrchestratorState) -> str:
        """Explain previous result in simpler terms"""
        from ..core.prompts import get_synthesizer_prompt

        previous_answer = state.get("previous_answer")
        previous_evidence = state.get("previous_evidence")
        user_message = state.get("user_message", "")

        if not previous_answer:
            return "Which result should I explain? Could you give a brief example?"

        # Build prompt to explain previous answer with evidence if available
        context_parts = [
            f"User request: {user_message}",
            f"Previous answer: {previous_answer}",
        ]
        if previous_evidence:
            context_parts.append(f"Previous evidence: {previous_evidence}")

        prompt = "\n".join(context_parts) + "\n\nPlease explain the previous answer in simpler, clear English. Do not add new information."

        try:
            response = self.llm.complete(
                prompt=prompt,
                model=self.model,
                system=get_synthesizer_prompt(),
                temperature=0.2
            )
            return response
        except Exception as e:
            logger.error(f"Explain synthesis failed: {e}")
            return "Here is a simpler explanation: the numbers and results in the previous answer were calculated using the period and filters you specified."
    
    def _synthesize_response(
        self,
        user_message: str,
        results: Dict[str, Any],
        state: OrchestratorState
    ) -> Dict[str, Any]:
        """Use LLM to synthesize natural language response"""
        
        from ..core.prompts import get_synthesizer_prompt
        
        # Build context
        context_parts = [f"User question: {user_message}"]
        
        if results.get("sql_result"):
            context_parts.append(f"\nSQL Result:\n{self._format_sql_result(results['sql_result'])}")
        
        if results.get("vector_result"):
            context_parts.append(f"\nSearch Result:\n{self._format_vector_result(results['vector_result'])}")
        
        if results.get("action_plan"):
            context_parts.append(f"\nAction Plan:\n{self._format_action_plan(results['action_plan'])}")
        
        # Inject currency from constraints
        currency = state.get("constraints", {}).get("currency", "TRY")
        context_parts.append(f"\nCONTEXT: All monetary amounts are in {currency}. Please format response using this currency symbol/code.")
        
        prompt = "\n".join(context_parts) + "\n\nBased on these results, respond in natural, helpful English."
        
        try:
            response = self.llm.complete(
                prompt=prompt,
                model=self.model,
                system=get_synthesizer_prompt(),
                temperature=0.3
            )
            
            return {
                "answer": response,
                "suggestions": self._generate_suggestions(state),
            }
            
        except Exception as e:
            logger.error(f"LLM synthesis failed: {e}")
            return {"answer": self._fallback_response(results)}
    
    def _format_sql_result(self, result: Dict[str, Any]) -> str:
        """Format SQL result for prompt"""
        if not result:
            return "No results found"
        
        parts = []
        
        if "value" in result and result["value"] is not None:
            formatted = self._format_amount(result["value"])
            if formatted is not None:
                parts.append(f"Value: {formatted}")
        if "tx_count" in result:
            parts.append(f"Transaction count: {result['tx_count']}")
        
        if "rows" in result and result["rows"]:
            rows = result["rows"][:100]  # Increased from 10 to 100 for aggregations
            parts.append(f"Rows ({len(result['rows'])} total):")
            for row in rows:
                parts.append(f"  - {row}")
        
        return "\n".join(parts)
    

    def _format_amount(self, value: Any) -> Optional[str]:
        """Format numeric values safely for prompts/responses"""
        if value is None:
            return None
        try:
            return f"{float(value):,.2f}"
        except (TypeError, ValueError):
            return str(value)

    def _format_vector_result(self, result: Dict[str, Any]) -> str:
        """Format vector search result for prompt"""
        if not result:
            return "No results found"
        
        matches = result.get("matches", [])
        if not matches:
            return "No matching transactions found in your entire transaction history."
        
        parts = [f"Found: {len(matches)} transactions"]
        for match in matches[:5]:
            parts.append(f"  - {match.get('merchant_norm', 'N/A')}: {match.get('amount', 'N/A')} TRY ({match.get('date_time', 'N/A')}) | Desc: {match.get('description', 'N/A')}")
        
        return "\n".join(parts)
    
    def _format_action_plan(self, plan: Dict[str, Any]) -> str:
        """Format action plan for prompt"""
        if not plan:
            return "No plan"
        
        return f"""
Action: {plan.get('action_type', 'N/A')}
Plan: {plan.get('human_plan', 'N/A')}
Requires confirmation: {plan.get('requires_confirmation', True)}
"""
    
    def _generate_suggestions(self, state: OrchestratorState) -> list:
        """Generate contextual suggestions"""
        intent = state.get("intent")
        suggestions = []
        
        if intent == IntentType.ANALYTICS.value:
            suggestions = [
                "Compare with last month?",
                "See category details?",
                "Export this period to Excel?"
            ]
        elif intent == IntentType.LOOKUP.value:
            suggestions = [
                "See similar transactions?",
                "List all spending in this category?"
            ]
        elif intent == IntentType.ACTION.value:
            suggestions = [
                "Need a different report type?",
                "Generate it for another period?"
            ]
        
        return suggestions[:3]
    
    def _fallback_response(self, results: Dict[str, Any]) -> str:
        """Generate fallback response when LLM fails"""
        if results.get("sql_result"):
            result = results["sql_result"]
            if "value" in result and result["value"] is not None:
                formatted = self._format_amount(result["value"])
                if formatted is not None:
                    return f"Result: {formatted} TRY ({result.get('tx_count', 0)} transactions)"
        if results.get("vector_result"):
            matches = results["vector_result"].get("matches", [])
            if matches:
                return f"Found {len(matches)} transactions."
            else:
                return "No matching transactions found in your transaction history."
        
        if results.get("action_plan"):
            return results["action_plan"].get("human_plan", "Action planned.")
        
        return "Action completed."
    
    def _build_evidence(self, state: OrchestratorState, results: Dict[str, Any]) -> Dict[str, Any]:
        """Build evidence dict for response"""
        evidence = {
            "filters": state.get("constraints", {}),
            "agents_used": [],
        }
        
        if results.get("sql_result"):
            evidence["agents_used"].append("finance_analyst")
            sql_result = results["sql_result"]
            evidence["tx_count"] = sql_result.get("tx_count")
            evidence["sql_preview"] = sql_result.get("sql_preview")
        
        if results.get("vector_result"):
            evidence["agents_used"].append("search_agent")
            evidence["search_terms"] = [state.get("search_query", {}).get("query_text")]
        
        if results.get("action_plan"):
            evidence["agents_used"].append("action_planner")
        
        return evidence


# -----------------------------------------------------------------------------
# SINGLETON
# -----------------------------------------------------------------------------

_orchestrator: Optional[OrchestratorAgent] = None
_synthesizer: Optional[ResponseSynthesizer] = None


def get_orchestrator() -> OrchestratorAgent:
    """Get or create orchestrator singleton"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = OrchestratorAgent()
    return _orchestrator


def get_synthesizer() -> ResponseSynthesizer:
    """Get or create synthesizer singleton"""
    global _synthesizer
    if _synthesizer is None:
        _synthesizer = ResponseSynthesizer()
    return _synthesizer

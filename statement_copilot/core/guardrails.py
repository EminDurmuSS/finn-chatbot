"""
Statement Copilot - Guardrails
==============================
Input and output safety guards.
Multi-layered: Rule-based -> LLM-based -> PII masking
"""

import re
import logging
from typing import Optional, List, Tuple

from ..config import settings
from ..core import (
    OrchestratorState,
    GuardrailResult,
    SafetyClassification,
    RiskLevel,
    get_llm_client,
)
from .prompts import get_input_guard_prompt

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# PII PATTERNS
# -----------------------------------------------------------------------------

PII_PATTERNS = {
    "iban": re.compile(r'\b[A-Z]{2}\d{2}[A-Z0-9]{4,30}\b'),
    "card_number": re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'),
    "phone_tr": re.compile(r'\b(?:\+90|0)?[5][0-9]{9}\b'),
    "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    "national_id_tr": re.compile(r'\b[1-9]\d{10}\b'),
}


# -----------------------------------------------------------------------------
# RULE-BASED GUARDRAILS
# -----------------------------------------------------------------------------

class RuleBasedGuard:
    """
    Fast, deterministic rule-based checks.
    First line of defense.
    """
    
    def __init__(self):
        self.blocked_keywords = settings.blocked_keywords
        self.max_input_length = settings.max_input_length
        
        # Prompt injection patterns
        self.injection_patterns = [
            re.compile(r'ignore\s+(all\s+)?previous', re.IGNORECASE),
            re.compile(r'system\s+prompt', re.IGNORECASE),
            re.compile(r'reveal\s+(your\s+)?instructions', re.IGNORECASE),
            re.compile(r'admin\s+mode', re.IGNORECASE),
            re.compile(r'jailbreak', re.IGNORECASE),
            re.compile(r'DAN\s+mode', re.IGNORECASE),
            re.compile(r'<\|im_start\|>', re.IGNORECASE),
            re.compile(r'\[\[SYSTEM\]\]', re.IGNORECASE),
        ]
        
        # SQL injection patterns
        self.sql_injection_patterns = [
            re.compile(r';\s*(DROP|DELETE|UPDATE|INSERT)', re.IGNORECASE),
            re.compile(r'UNION\s+SELECT', re.IGNORECASE),
            re.compile(r'--\s*$', re.MULTILINE),
            re.compile(r'/\*.*\*/', re.DOTALL),
        ]
    
    def check(self, text: str) -> GuardrailResult:
        """
        Run rule-based checks on input text.
        
        Returns:
            GuardrailResult with pass/fail status
        """
        warnings = []
        
        # Length check
        if len(text) > self.max_input_length:
            return GuardrailResult(
                passed=False,
                reason=f"Message too long. Maximum {self.max_input_length} characters.",
                risk_level=RiskLevel.MEDIUM,
                blocked=True
            )
        
        # Empty check
        if not text or not text.strip():
            return GuardrailResult(
                passed=False,
                reason="Empty message is not allowed.",
                risk_level=RiskLevel.LOW,
                blocked=True
            )
        
        text_lower = text.lower()
        
        # Blocked keywords
        for keyword in self.blocked_keywords:
            if keyword.lower() in text_lower:
                logger.warning(f"Blocked keyword detected: {keyword}")
                return GuardrailResult(
                    passed=False,
                    reason="This type of request is not supported.",
                    risk_level=RiskLevel.HIGH,
                    blocked=True
                )
        
        # Prompt injection patterns
        for pattern in self.injection_patterns:
            if pattern.search(text):
                logger.warning(f"Prompt injection pattern detected")
                return GuardrailResult(
                    passed=False,
                    reason="This message cannot be processed.",
                    risk_level=RiskLevel.HIGH,
                    blocked=True
                )
        
        # SQL injection patterns
        for pattern in self.sql_injection_patterns:
            if pattern.search(text):
                logger.warning(f"SQL injection pattern detected")
                return GuardrailResult(
                    passed=False,
                    reason="This message cannot be processed.",
                    risk_level=RiskLevel.HIGH,
                    blocked=True
                )
        
        # Wide scope warning (asking for too much data)
        wide_scope_patterns = [
            r'all\s+(data|transactions|spending)',
            r'everything',
            r'show\s+all',
            r'list\s+all',
        ]
        for pattern in wide_scope_patterns:
            if re.search(pattern, text_lower):
                warnings.append("Wide scope query. Results may be limited.")
        
        return GuardrailResult(
            passed=True,
            risk_level=RiskLevel.LOW,
            warnings=warnings
        )


# -----------------------------------------------------------------------------
# LLM-BASED GUARDRAILS
# -----------------------------------------------------------------------------

class LLMGuard:
    """
    LLM-based safety classification.
    Second line of defense for edge cases.
    """
    
    def __init__(self):
        self.llm = get_llm_client()
    
    def classify(self, text: str) -> SafetyClassification:
        """
        Use LLM to classify message safety.
        
        Args:
            text: Input text to classify
            
        Returns:
            SafetyClassification result
        """
        try:
            result = self.llm.complete_structured(
                prompt=f"User message:\n{text}",
                response_model=SafetyClassification,
                model=settings.model_guardrail,
                system=get_input_guard_prompt(),
                max_tokens=500,
                temperature=0.0
            )
            return result
            
        except Exception as e:
            logger.error(f"LLM guard error: {e}")
            # Fail safe - assume safe if LLM fails
            return SafetyClassification(
                is_safe=True,
                category="safe",
                confidence=0.5,
                explanation="LLM classification failed, assuming safe"
            )
    
    def check(self, text: str) -> GuardrailResult:
        """
        Run LLM-based check and convert to GuardrailResult.
        """
        classification = self.classify(text)
        
        if not classification.is_safe:
            risk_map = {
                "prompt_injection": RiskLevel.HIGH,
                "data_extraction": RiskLevel.HIGH,
                "harmful_content": RiskLevel.HIGH,
                "off_topic": RiskLevel.LOW,
            }
            
            return GuardrailResult(
                passed=False,
                reason=classification.explanation,
                risk_level=risk_map.get(classification.category, RiskLevel.MEDIUM),
                blocked=classification.category in ["prompt_injection", "data_extraction", "harmful_content"]
            )
        
        return GuardrailResult(
            passed=True,
            risk_level=RiskLevel.LOW
        )


# -----------------------------------------------------------------------------
# PII MASKING
# -----------------------------------------------------------------------------

class PIIMasker:
    """
    Mask PII in outputs.
    Protects sensitive data in responses.
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and settings.pii_patterns_enabled
        self.patterns = PII_PATTERNS
    
    def mask(self, text: str) -> Tuple[str, List[str]]:
        """
        Mask PII patterns in text.
        
        Args:
            text: Text to mask
            
        Returns:
            Tuple of (masked_text, list of PII types found)
        """
        if not self.enabled:
            return text, []
        
        found_types = []
        masked_text = text
        
        for pii_type, pattern in self.patterns.items():
            matches = pattern.findall(masked_text)
            if matches:
                found_types.append(pii_type)
                
                # Replace with masked version
                if pii_type == "iban":
                    masked_text = pattern.sub(lambda m: m.group()[:4] + "****" + m.group()[-4:], masked_text)
                elif pii_type == "card_number":
                    masked_text = pattern.sub("****-****-****-****", masked_text)
                elif pii_type == "phone_tr":
                    masked_text = pattern.sub(lambda m: m.group()[:4] + "****" + m.group()[-2:], masked_text)
                elif pii_type == "email":
                    masked_text = pattern.sub(lambda m: m.group().split('@')[0][:2] + "****@" + m.group().split('@')[1], masked_text)
                elif pii_type == "national_id_tr":
                    masked_text = pattern.sub(lambda m: m.group()[:3] + "****" + m.group()[-2:], masked_text)
        
        if found_types:
            logger.debug(f"Masked PII types: {found_types}")
        
        return masked_text, found_types


# -----------------------------------------------------------------------------
# COMBINED GUARDRAILS
# -----------------------------------------------------------------------------

class Guardrails:
    """
    Combined guardrails manager.
    Runs checks in order: Rule-based -> LLM-based -> PII
    """
    
    def __init__(self, use_llm_guard: bool = True):
        self.rule_guard = RuleBasedGuard()
        self.llm_guard = LLMGuard() if use_llm_guard else None
        self.pii_masker = PIIMasker()
    
    def check_input(self, text: str, use_llm: bool = False) -> GuardrailResult:
        """
        Check input text through guardrails.
        
        Args:
            text: Input text to check
            use_llm: Whether to use LLM guard for edge cases
            
        Returns:
            GuardrailResult
        """
        # Step 1: Rule-based (fast)
        rule_result = self.rule_guard.check(text)
        
        if not rule_result.passed:
            logger.info(f"Input blocked by rule guard: {rule_result.reason}")
            return rule_result
        
        # Step 2: LLM-based (slower but more robust)
        if use_llm and self.llm_guard:
            llm_result = self.llm_guard.check(text)
            if not llm_result.passed:
                logger.info(f"Input blocked by LLM guard: {llm_result.reason}")
                return llm_result

            # Combine warnings
            rule_result.warnings.extend(llm_result.warnings)
        
        return rule_result
    
    def mask_output(self, text: str) -> str:
        """
        Mask PII in output text.
        
        Args:
            text: Output text to mask
            
        Returns:
            Masked text
        """
        masked, _ = self.pii_masker.mask(text)
        return masked
    
    def process_state(self, state: OrchestratorState) -> OrchestratorState:
        """
        Process state through guardrails.
        Used as a LangGraph node.
        """
        user_message = state.get("user_message", "")
        
        # Check input
        result = self.check_input(user_message, use_llm=True)
        
        # Update state
        state["guardrail_passed"] = result.passed
        
        # Always set warnings (empty list if none) to ensure previous warnings are cleared
        state["guardrail_warnings"] = result.warnings or []
        
        if not result.passed:
            state["blocked_reason"] = result.reason
            state["risk_level"] = result.risk_level.value
        
        return state


# -----------------------------------------------------------------------------
# SINGLETON
# -----------------------------------------------------------------------------

_guardrails: Optional[Guardrails] = None


def get_guardrails() -> Guardrails:
    """Get or create guardrails singleton"""
    global _guardrails
    if _guardrails is None:
        _guardrails = Guardrails(use_llm_guard=True)
    return _guardrails

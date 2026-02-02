"""
Statement Copilot - LLM Client (FIXED)
======================================
Anthropic client with structured outputs support.

FIXES:
- Added model fallback support
- Better error logging
- Timeout handling improved
- Schema validation before sending
"""

from __future__ import annotations

import json
import logging
import random
import time
from typing import Optional, Type, TypeVar, Any, Dict, List

import anthropic
from pydantic import BaseModel

from ..config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Retryable HTTP statuses
_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}

# Try to import SDK helpers
try:
    from anthropic import transform_schema
except Exception:
    transform_schema = None

try:
    APIStatusError = anthropic.APIStatusError
except Exception:
    APIStatusError = Exception

try:
    RateLimitError = anthropic.RateLimitError
except Exception:
    RateLimitError = Exception

try:
    APITimeoutError = anthropic.APITimeoutError
except Exception:
    APITimeoutError = Exception


class LLMClient:
    """
    Anthropic client wrapper with improved error handling.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.anthropic_api_key
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is required")

        self.client = anthropic.Anthropic(
            api_key=self.api_key,
            timeout=settings.anthropic_timeout_s,
        )
        self.beta_headers: List[str] = list(settings.anthropic_beta_headers)
        
        # Fallback model for when primary fails
        self.fallback_model = getattr(settings, 'model_fallback', 'claude-3-5-sonnet-20241022')

        # Token tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
        logger.info(f"LLMClient initialized. Beta headers: {self.beta_headers}")

    def _extract_text(self, response: Any) -> str:
        """Safely extract text from Anthropic response."""
        try:
            blocks = getattr(response, "content", None) or []
            parts: List[str] = []
            for b in blocks:
                txt = getattr(b, "text", None)
                if isinstance(txt, str) and txt.strip():
                    parts.append(txt)
                    continue
                if isinstance(b, dict) and b.get("type") == "text":
                    t = b.get("text", "")
                    if isinstance(t, str) and t.strip():
                        parts.append(t)
            return "".join(parts).strip()
        except Exception:
            return str(getattr(response, "content", "")).strip()

    def _track_usage(self, response: Any) -> None:
        """Accumulate token usage."""
        usage = getattr(response, "usage", None)
        if not usage:
            return
        in_toks = getattr(usage, "input_tokens", 0) or 0
        out_toks = getattr(usage, "output_tokens", 0) or 0
        self.total_input_tokens += int(in_toks)
        self.total_output_tokens += int(out_toks)

    def _status_code(self, exc: Exception) -> Optional[int]:
        """Extract HTTP status code from exception."""
        for attr in ("status_code", "status", "code"):
            v = getattr(exc, attr, None)
            if isinstance(v, int):
                return v
        resp = getattr(exc, "response", None)
        if resp is not None:
            v = getattr(resp, "status_code", None)
            if isinstance(v, int):
                return v
        return None

    def _should_retry(self, exc: Exception) -> bool:
        if isinstance(exc, RateLimitError):
            return True
        if isinstance(exc, APITimeoutError):
            return True
        if isinstance(exc, APIStatusError):
            sc = self._status_code(exc)
            return sc in _RETRYABLE_STATUS
        sc = self._status_code(exc)
        return sc in _RETRYABLE_STATUS if sc is not None else False

    def _call_with_retries(
        self,
        fn,
        *,
        max_retries: int = 3,
        base_delay: float = 1.0,
        jitter: float = 0.5,
        log_label: str = "anthropic_call",
    ) -> Any:
        """Execute with exponential backoff retries."""
        last_exc: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                return fn()
            except Exception as e:
                last_exc = e
                status = self._status_code(e)
                
                # Log detailed error info
                logger.warning(
                    f"{log_label} failed (attempt {attempt+1}/{max_retries+1}): "
                    f"status={status}, error={type(e).__name__}: {str(e)[:200]}"
                )
                
                if attempt >= max_retries or not self._should_retry(e):
                    raise
                    
                delay = base_delay * (2 ** attempt) + random.random() * jitter
                logger.info(f"{log_label} retrying in {delay:.2f}s...")
                time.sleep(delay)
                
        raise last_exc or RuntimeError("Unknown error")

    def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """Simple text completion with fallback support."""
        model = model or settings.model_orchestrator
        messages = [{"role": "user", "content": prompt}]

        logger.debug(
            "LLM complete request: model=%s prompt_len=%s system_len=%s temp=%.2f max_tokens=%s stop_count=%s",
            model,
            len(prompt),
            len(system or ""),
            float(temperature),
            max_tokens,
            len(stop_sequences) if stop_sequences else 0,
        )

        def _do(use_model: str):
            kwargs: Dict[str, Any] = dict(
                model=use_model,
                max_tokens=max_tokens,
                messages=messages,
                temperature=temperature,
            )
            if system is not None:
                kwargs["system"] = system
            if stop_sequences:
                kwargs["stop_sequences"] = stop_sequences
            return self.client.messages.create(**kwargs)

        start_time = time.time()
        
        # Try primary model first
        try:
            response = self._call_with_retries(
                lambda: _do(model), 
                log_label=f"llm.complete[{model}]"
            )
        except Exception as e:
            # Try fallback model
            logger.warning(f"Primary model {model} failed, trying fallback {self.fallback_model}")
            try:
                response = self._call_with_retries(
                    lambda: _do(self.fallback_model),
                    log_label=f"llm.complete[fallback]"
                )
            except Exception as e2:
                logger.error(f"Both primary and fallback models failed: {e2}")
                raise

        latency_ms = int((time.time() - start_time) * 1000)
        self._track_usage(response)
        text = self._extract_text(response)

        usage = getattr(response, "usage", None)
        in_tokens = getattr(usage, "input_tokens", None) if usage else None
        out_tokens = getattr(usage, "output_tokens", None) if usage else None

        logger.debug(
            "LLM complete response: latency=%sms output_len=%s input_tokens=%s output_tokens=%s",
            latency_ms,
            len(text),
            in_tokens,
            out_tokens,
        )
        return text

    def complete_structured(
        self,
        prompt: str,
        response_model: Type[T],
        model: Optional[str] = None,
        system: Optional[str] = None,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        stop_sequences: Optional[List[str]] = None,
    ) -> T:
        """Structured output completion with fallback support."""
        model = model or settings.model_orchestrator
        messages = [{"role": "user", "content": prompt}]

        logger.debug(
            "LLM structured request: model=%s schema=%s prompt_len=%s system_len=%s temp=%.2f max_tokens=%s",
            model,
            response_model.__name__,
            len(prompt),
            len(system or ""),
            float(temperature),
            max_tokens,
        )

        # Build schema
        if transform_schema is not None:
            schema_payload = {
                "type": "json_schema",
                "schema": transform_schema(response_model),
            }
            logger.debug(f"Using transform_schema for {response_model.__name__}")
        else:
            schema_payload = {
                "type": "json_schema",
                "schema": response_model.model_json_schema(),
            }
            logger.debug(f"Using raw Pydantic schema for {response_model.__name__}")

        def _do_structured(use_model: str):
            kwargs: Dict[str, Any] = dict(
                model=use_model,
                max_tokens=max_tokens,
                messages=messages,
                temperature=temperature,
                betas=self.beta_headers,
                output_format=schema_payload,
            )
            if system is not None:
                kwargs["system"] = system
            if stop_sequences:
                kwargs["stop_sequences"] = stop_sequences
            return self.client.beta.messages.create(**kwargs)

        def _do_plain(use_model: str):
            """Fallback: plain completion with JSON instruction."""
            schema_str = json.dumps(response_model.model_json_schema(), indent=2)
            json_prompt = (
                f"{prompt}\n\n"
                f"Respond ONLY with valid JSON matching this schema:\n{schema_str}\n"
                f"No markdown, no explanation, just the JSON object."
            )
            kwargs: Dict[str, Any] = dict(
                model=use_model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": json_prompt}],
                temperature=temperature,
            )
            if system is not None:
                kwargs["system"] = system
            return self.client.messages.create(**kwargs)

        start_time = time.time()
        response = None

        # If beta headers are disabled, skip structured output and use plain JSON directly
        if not self.beta_headers:
            logger.info("Beta headers disabled, using plain JSON completion")
            response = self._call_with_retries(
                lambda: _do_plain(model),
                log_label=f"llm.plain_json[{model}]"
            )
        else:
            # Try structured output first
            try:
                response = self._call_with_retries(
                    lambda: _do_structured(model),
                    log_label=f"llm.structured[{model}]"
                )
            except Exception as e:
                logger.warning(f"Structured output failed with {model}: {e}")

                # Try fallback model with structured output
                try:
                    logger.info(f"Trying fallback model {self.fallback_model} with structured output")
                    response = self._call_with_retries(
                        lambda: _do_structured(self.fallback_model),
                        log_label=f"llm.structured[fallback]"
                    )
                except Exception as e2:
                    logger.warning(f"Fallback structured also failed: {e2}")

                    # Last resort: plain completion with JSON instruction
                    try:
                        logger.info("Falling back to plain completion with JSON instruction")
                        response = self._call_with_retries(
                            lambda: _do_plain(self.fallback_model),
                            log_label="llm.plain_json"
                        )
                    except Exception as e3:
                        logger.error(f"All attempts failed: {e3}")
                        raise

        latency_ms = int((time.time() - start_time) * 1000)

        # Check for refusal
        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason == "refusal":
            text = self._extract_text(response)
            raise RuntimeError(f"Model refusal: {text}")

        self._track_usage(response)
        raw_text = self._extract_text(response)
        
        if not raw_text:
            raise RuntimeError("Empty response from model")

        # Clean up potential markdown
        clean_text = raw_text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        if clean_text.startswith("```"):
            clean_text = clean_text[3:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
        clean_text = clean_text.strip()

        try:
            data = json.loads(clean_text)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode failed. Raw: {raw_text[:500]}")
            raise RuntimeError(f"Invalid JSON: {e}") from e

        try:
            parsed = response_model.model_validate(data)
        except Exception as e:
            logger.error(f"Validation failed for {response_model.__name__}. Data: {str(data)[:500]}")
            raise

        usage = getattr(response, "usage", None)
        in_tokens = getattr(usage, "input_tokens", None) if usage else None
        out_tokens = getattr(usage, "output_tokens", None) if usage else None

        logger.debug(
            "LLM structured response: schema=%s latency=%sms output_len=%s input_tokens=%s output_tokens=%s",
            response_model.__name__,
            latency_ms,
            len(raw_text),
            in_tokens,
            out_tokens,
        )
        return parsed

    def complete_with_messages(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        system: Optional[str] = None,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """Multi-turn completion."""
        model = model or settings.model_orchestrator
        
        # Normalize messages
        normalized = []
        for m in messages:
            role = m.get("role")
            content = m.get("content")
            if role not in ("user", "assistant"):
                raise ValueError(f"Invalid role: {role}")
            normalized.append({"role": role, "content": content})

        def _do(use_model: str):
            kwargs: Dict[str, Any] = dict(
                model=use_model,
                max_tokens=max_tokens,
                messages=normalized,
                temperature=temperature,
            )
            if system:
                kwargs["system"] = system
            if stop_sequences:
                kwargs["stop_sequences"] = stop_sequences
            return self.client.messages.create(**kwargs)

        total_chars = sum(len(str(m.get("content", ""))) for m in normalized)
        logger.debug(
            "LLM multi-turn request: model=%s messages=%s total_chars=%s system_len=%s temp=%.2f max_tokens=%s",
            model,
            len(normalized),
            total_chars,
            len(system or ""),
            float(temperature),
            max_tokens,
        )

        start_time = time.time()
        try:
            response = self._call_with_retries(lambda: _do(model), log_label="llm.multi_turn")
        except Exception:
            response = self._call_with_retries(
                lambda: _do(self.fallback_model), 
                log_label="llm.multi_turn[fallback]"
            )

        latency_ms = int((time.time() - start_time) * 1000)
        self._track_usage(response)
        text = self._extract_text(response)

        usage = getattr(response, "usage", None)
        in_tokens = getattr(usage, "input_tokens", None) if usage else None
        out_tokens = getattr(usage, "output_tokens", None) if usage else None
        logger.debug(
            "LLM multi-turn response: latency=%sms output_len=%s input_tokens=%s output_tokens=%s",
            latency_ms,
            len(text),
            in_tokens,
            out_tokens,
        )
        return text

    def complete_structured_with_messages(
        self,
        messages: List[Dict[str, Any]],
        response_model: Type[T],
        model: Optional[str] = None,
        system: Optional[str] = None,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        stop_sequences: Optional[List[str]] = None,
    ) -> T:
        """Multi-turn structured output completion."""
        model = model or settings.model_orchestrator
        
        # Normalize messages
        normalized = []
        for m in messages:
            role = m.get("role")
            content = m.get("content")
            if role not in ("user", "assistant"):
                raise ValueError(f"Invalid role: {role}")
            normalized.append({"role": role, "content": content})

        if transform_schema is not None:
            schema_payload = {"type": "json_schema", "schema": transform_schema(response_model)}
        else:
            schema_payload = {"type": "json_schema", "schema": response_model.model_json_schema()}

        def _do(use_model: str):
            kwargs: Dict[str, Any] = dict(
                model=use_model,
                max_tokens=max_tokens,
                messages=normalized,
                temperature=temperature,
                betas=self.beta_headers,
                output_format=schema_payload,
            )
            if system:
                kwargs["system"] = system
            if stop_sequences:
                kwargs["stop_sequences"] = stop_sequences
            return self.client.beta.messages.create(**kwargs)

        total_chars = sum(len(str(m.get("content", ""))) for m in normalized)
        logger.debug(
            "LLM structured multi-turn request: model=%s schema=%s messages=%s total_chars=%s system_len=%s temp=%.2f max_tokens=%s",
            model,
            response_model.__name__,
            len(normalized),
            total_chars,
            len(system or ""),
            float(temperature),
            max_tokens,
        )

        start_time = time.time()
        try:
            response = self._call_with_retries(lambda: _do(model), log_label="llm.structured_multi")
        except Exception:
            response = self._call_with_retries(
                lambda: _do(self.fallback_model),
                log_label="llm.structured_multi[fallback]"
            )

        latency_ms = int((time.time() - start_time) * 1000)

        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason == "refusal":
            raise RuntimeError(f"Model refusal: {self._extract_text(response)}")

        self._track_usage(response)
        raw_text = self._extract_text(response)
        
        if not raw_text:
            raise RuntimeError("Empty response")

        data = json.loads(raw_text)
        parsed = response_model.model_validate(data)

        usage = getattr(response, "usage", None)
        in_tokens = getattr(usage, "input_tokens", None) if usage else None
        out_tokens = getattr(usage, "output_tokens", None) if usage else None
        logger.debug(
            "LLM structured multi-turn response: schema=%s latency=%sms output_len=%s input_tokens=%s output_tokens=%s",
            response_model.__name__,
            latency_ms,
            len(raw_text),
            in_tokens,
            out_tokens,
        )
        return parsed

    def get_token_usage(self) -> Dict[str, int]:
        """Get cumulative token usage."""
        return {
            "input_tokens": int(self.total_input_tokens),
            "output_tokens": int(self.total_output_tokens),
            "total_tokens": int(self.total_input_tokens + self.total_output_tokens),
        }

    def reset_token_usage(self) -> None:
        """Reset token counters."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0


# Singleton
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get or create LLM client singleton."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client

"""
Statement Copilot - Professional Search Engine (Production Ready, EN-only)
=========================================================================

Multi-stage retrieval with query understanding, optional LLM extraction,
SQL+Vector retrieval, fusion, and deterministic reranking.

Architecture:
1) Query Understanding (NLU) - extract entities, classify intent
2) Strategy Selection - choose SQL / vector / hybrid
3) Multi-Source Retrieval - SQL + Vector
4) Result Fusion & Reranking - deterministic scoring (+ optional LLM rerank)
5) Evidence Assembly - traceable results

Key Guarantees:
- LLM-first with rule-based fallback
- Merchant validation against known merchants (prevents hallucination)
- Confidence gating (low LLM confidence triggers fallback)
- Soft categories by default (prevents over-filtering)
- Keywords as reranker boosts, not hard SQL filters
- Timezone-safe recency scoring
- Date filters optional (no date mentioned = no restriction)

Assumptions / interfaces:
- db.execute_query(sql: str, params: list[Any]) -> list[dict]
- vector_store.search(query: str, top_k: int, alpha: float, filters: dict) -> list[dict]
- llm_client.complete_structured(prompt: str, response_model: BaseModel, system: str, temperature: float) -> BaseModel
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Protocol, Set, Tuple

from pydantic import BaseModel, Field

from ..config import settings
from .prompts import (
    get_query_understanding_prompt,
    get_query_expansion_prompt,
    get_result_reranking_prompt,
)

logger = logging.getLogger(__name__)


# =============================================================================
# REQUIRED INTERFACES (DOCUMENTATION)
# =============================================================================

class DBClient(Protocol):
    def execute_query(self, sql: str, params: List[Any]) -> List[Dict[str, Any]]:
        ...


class VectorStore(Protocol):
    def search(self, query: str, top_k: int, alpha: float, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        ...


class LLMClient(Protocol):
    def complete_structured(
        self,
        prompt: str,
        response_model: type[BaseModel],
        system: str,
        temperature: float = 0.0,
    ) -> BaseModel:
        ...


# =============================================================================
# MINIMAL STOPWORDS (Simplified - LLM handles semantic understanding)
# =============================================================================

STOPWORDS: Set[str] = {
    # Articles / pronouns / auxiliaries
    "the", "a", "an", "i", "me", "my", "we", "us", "you", "your", "it", "its",
    "is", "are", "was", "were", "be", "been", "being",
    # Prepositions / connectors
    "on", "in", "at", "to", "for", "of", "with", "by", "from",
    "and", "or", "but", "so", "than",
    # Question words
    "what", "where", "when", "how", "why", "which", "who",
    # Common query verbs (not helpful as search terms)
    "show", "find", "get", "list", "tell", "see", "display",
    "have", "has", "had", "do", "does", "did",
}

_MONTHS: Dict[str, int] = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _is_all_digits_or_money(s: str) -> bool:
    s2 = s.strip().replace(",", "")
    if not s2:
        return True
    if s2.isdigit():
        return True
    return bool(re.fullmatch(r"\$?\d+(?:\.\d{1,2})?", s2))


def _dedup_keep_order(items: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _levenshtein_ratio(s1: str, s2: str) -> float:
    """Calculate Levenshtein similarity ratio (0.0 to 1.0)."""
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0

    len1, len2 = len(s1), len(s2)
    if len1 < len2:
        s1, s2 = s2, s1
        len1, len2 = len2, len1

    # Quick check for very different lengths
    if len1 - len2 > max(len1, len2) * 0.5:
        return 0.0

    # Simple Levenshtein distance
    prev_row = list(range(len2 + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    distance = prev_row[-1]
    max_len = max(len1, len2)
    return 1.0 - (distance / max_len)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class SearchIntent(str, Enum):
    FIND_SPECIFIC = "find_specific"
    FIND_SIMILAR = "find_similar"
    AGGREGATE = "aggregate"
    LIST_FILTER = "list_filter"
    TEMPORAL = "temporal"
    COMPARATIVE = "comparative"
    ANOMALY = "anomaly"
    MERCHANT_LOOKUP = "merchant_lookup"


class SearchStrategy(str, Enum):
    EXACT_MATCH = "exact_match"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    SQL_ONLY = "sql_only"


def _today_utc_date() -> date:
    return datetime.utcnow().date()


TEMPORAL_KEYWORDS: Dict[str, Any] = {
    "today": lambda today: (today, today),
    "yesterday": lambda today: (today - timedelta(days=1), today - timedelta(days=1)),
    "this week": lambda today: (today - timedelta(days=today.weekday()), today),
    "last week": lambda today: (
        today - timedelta(days=today.weekday() + 7),
        today - timedelta(days=today.weekday() + 1),
    ),
    "this month": lambda today: (today.replace(day=1), today),
    "last month": lambda today: (
        (today.replace(day=1) - timedelta(days=1)).replace(day=1),
        today.replace(day=1) - timedelta(days=1),
    ),
    "last 7 days": lambda today: (today - timedelta(days=7), today),
    "last 30 days": lambda today: (today - timedelta(days=30), today),
    "last 90 days": lambda today: (today - timedelta(days=90), today),
    "this year": lambda today: (today.replace(month=1, day=1), today),
    "last year": lambda today: (
        today.replace(year=today.year - 1, month=1, day=1),
        today.replace(year=today.year - 1, month=12, day=31),
    ),
}

# LLM confidence threshold - below this, fall back to rules
LLM_CONFIDENCE_THRESHOLD = 0.6


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SearchEngineConfig:
    """Centralized configuration for search engine."""
    
    # LLM thresholds
    llm_confidence_threshold: float = 0.6
    merchant_fuzzy_threshold: float = 0.7
    
    # Query processing
    min_keyword_length: int = 3
    max_query_expansion_terms: int = 10
    
    # Reranking
    llm_rerank_top_k: int = 20
    llm_score_weight: float = 0.7  # 70% LLM, 30% deterministic
    deterministic_score_weight: float = 0.3
    
    # Scoring boosts (deterministic reranker)
    merchant_boost: float = 0.30
    category_boost: float = 0.15
    subcategory_boost: float = 0.10
    keyword_boost_base: float = 0.18
    keyword_boost_per_match: float = 0.03
    recency_max_boost: float = 0.20
    
    # Penalties
    non_matching_merchant_penalty: float = 0.10


# Default config instance
DEFAULT_CONFIG = SearchEngineConfig()


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ExtractedEntities:
    merchants: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    subcategories: List[str] = field(default_factory=list)
    amounts: List[Dict[str, Any]] = field(default_factory=list)
    date_range: Optional[Tuple[date, date]] = None
    direction: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    content_keywords: List[str] = field(default_factory=list)


@dataclass
class QueryUnderstanding:
    original_query: str
    normalized_query: str
    intent: SearchIntent
    confidence: float
    entities: ExtractedEntities
    strategy: SearchStrategy
    expanded_query: str
    search_terms: List[str]
    reasoning: str


@dataclass
class SearchMatch:
    tx_id: str
    score: float
    source: str  # 'sql' | 'vector'

    date_time: Optional[datetime] = None
    amount: Optional[float] = None
    merchant_norm: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    subcategory: Optional[str] = None
    direction: Optional[str] = None

    match_reason: str = ""
    highlight_terms: List[str] = field(default_factory=list)


@dataclass
class SearchResult:
    query_understanding: QueryUnderstanding
    matches: List[SearchMatch]
    total_found: int

    search_time_ms: int = 0
    sources_used: List[str] = field(default_factory=list)
    filters_applied: Dict[str, Any] = field(default_factory=dict)

    total_matching: int = 0
    result_limited: bool = False

    sql_query: Optional[str] = None
    vector_query: Optional[str] = None


# =============================================================================
# PYDANTIC MODELS FOR LLM
# =============================================================================

class UnifiedQueryAnalysis(BaseModel):
    """Single LLM call response for complete query understanding."""
    intent: Literal[
        "find_specific", "find_similar", "aggregate", "list_filter",
        "temporal", "comparative", "anomaly", "merchant_lookup"
    ] = Field(description="The primary user intent")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the classification")

    merchants: List[str] = Field(default_factory=list, description="Business/brand names like Netflix, YouTube, Starbucks")
    content_keywords: List[str] = Field(default_factory=list, description="Meaningful search terms like subscription, premium, rent")
    direction: Optional[Literal["expense", "income", "transfer"]] = Field(
        default=None, description="Transaction direction: expense (spent), income (received), transfer"
    )

    search_strategy: Literal["exact_match", "semantic", "hybrid", "sql_only"] = Field(
        default="hybrid", description="Recommended search approach"
    )

    reasoning: str = Field(description="Brief explanation of the classification")


class RerankedResult(BaseModel):
    tx_id: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    match_explanation: str


class RerankedResults(BaseModel):
    results: List[RerankedResult]
    overall_quality: str


# =============================================================================
# QUERY UNDERSTANDING ENGINE
# =============================================================================

class QueryUnderstandingEngine:
    """
    LLM-first query understanding with rule-based fallback.

    Flow:
    1. Rule-based extraction for structured data (amounts, dates, categories)
    2. LLM for semantic understanding (intent, merchants, keywords, direction)
    3. Confidence gating - low confidence triggers rule-based fallback
    4. Merchant validation against known merchants
       - FIX: rejected merchants are preserved as keywords (signal not lost)
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        taxonomy: Optional[Dict[str, Any]] = None,
        known_merchants: Optional[Set[str]] = None,
        today_func=_today_utc_date,
    ):
        self.llm = llm_client
        self.taxonomy = taxonomy or {}
        self.known_merchants = known_merchants or set()
        self.today_func = today_func
        self._build_category_index()

    def _build_category_index(self) -> None:
        """Build keyword -> category/subcategory mappings from taxonomy."""
        self.category_keywords: Dict[str, str] = {}
        self.subcategory_keywords: Dict[str, Tuple[str, str]] = {}

        categories = self.taxonomy.get("categories", {}) or {}
        for cat_id, cat_data in categories.items():
            keywords = cat_data.get("keywords", []) or []
            display_name = (cat_data.get("display_name") or "").lower()
            all_keywords = list(keywords) + [display_name, str(cat_id)]

            for kw in all_keywords:
                if kw:
                    self.category_keywords[kw.lower()] = cat_id

            subcats = cat_data.get("subcategories", {}) or {}
            for subcat_id, subcat_data in subcats.items():
                sub_keywords = subcat_data.get("keywords", []) or []
                sub_display = (subcat_data.get("display_name") or "").lower()
                merchant_examples = subcat_data.get("merchants_examples", []) or []
                all_sub_kw = list(sub_keywords) + [sub_display, str(subcat_id)] + [m.lower() for m in merchant_examples if m]

                for kw in all_sub_kw:
                    if kw:
                        self.subcategory_keywords[kw.lower()] = (cat_id, subcat_id)

    def understand(self, query: str) -> QueryUnderstanding:
        """Main entry point for query understanding."""
        start = time.time()

        # Minimal normalization (preserve semantics for LLM)
        normalized = self._normalize_query(query)

        # Rule-based extraction for structured data
        entities = ExtractedEntities()
        entities.amounts = self._extract_amounts(normalized)
        entities.date_range = self._extract_date_range(normalized)
        entities.categories, entities.subcategories = self._extract_categories(normalized)
        entities.keywords = self._extract_keywords(normalized)

        # LLM-powered classification (with confidence gating)
        if self.llm:
            llm_result = self._classify_unified_llm(query, entities)
            if llm_result and llm_result["confidence"] >= LLM_CONFIDENCE_THRESHOLD:
                intent = llm_result["intent"]
                confidence = llm_result["confidence"]

                # ✅ FIX #1: Capture both validated and rejected merchants
                validated_merchants, rejected_merchants = self._validate_merchants(llm_result["merchants"])
                entities.merchants = validated_merchants

                # ✅ FIX #1: Add rejected merchants to content_keywords (signal preserved)
                entities.content_keywords = llm_result["content_keywords"] + rejected_merchants

                entities.direction = llm_result["direction"]
                strategy = llm_result["strategy"]
                reasoning = llm_result["reasoning"]
            else:
                # LLM failed or low confidence - use rules
                if llm_result:
                    logger.info(
                        "[QUERY] LLM confidence %.2f < %.2f threshold, using rule fallback",
                        llm_result["confidence"], LLM_CONFIDENCE_THRESHOLD
                    )
                entities.merchants = self._extract_merchants_fallback(normalized)
                entities.content_keywords = self._extract_content_keywords_rules(normalized, entities)
                entities.direction = None  # LLM handles this, fallback just leaves it None
                intent, confidence = self._classify_intent_rules(normalized, entities)
                strategy = self._select_strategy(intent, entities)
                reasoning = self._get_strategy_reasoning(strategy)
        else:
            # No LLM available
            entities.merchants = self._extract_merchants_fallback(normalized)
            entities.content_keywords = self._extract_content_keywords_rules(normalized, entities)
            entities.direction = None  # No LLM, no sophisticated direction detection
            intent, confidence = self._classify_intent_rules(normalized, entities)
            strategy = self._select_strategy(intent, entities)
            reasoning = self._get_strategy_reasoning(strategy)

        # Handle quoted strings as explicit merchant hints
        for quoted in re.findall(r"\"([^\"]+)\"", query):
            qd = quoted.strip()
            if qd and len(qd) >= 3 and not _is_all_digits_or_money(qd):
                up = qd.upper()
                if up not in entities.merchants:
                    entities.merchants.append(up)

        # ✅ Extra safety: validate final merchant list (includes quoted/fallback merchants too)
        validated_final, rejected_final = self._validate_merchants(entities.merchants)
        entities.merchants = validated_final
        if rejected_final:
            entities.content_keywords = (entities.content_keywords or []) + rejected_final

        entities.merchants = _dedup_keep_order(entities.merchants)
        entities.content_keywords = _dedup_keep_order([k.lower() for k in entities.content_keywords if k])

        expanded, search_terms = self._build_search_query(normalized, entities)

        latency_ms = int((time.time() - start) * 1000)

        logger.info(
            "[QUERY] intent=%s conf=%.2f strategy=%s latency_ms=%d merchants=%s cats=%s date=%s",
            intent.value, confidence, strategy.value, latency_ms,
            entities.merchants or None, entities.categories or None, entities.date_range or None,
        )

        return QueryUnderstanding(
            original_query=query,
            normalized_query=normalized,
            intent=intent,
            confidence=confidence,
            entities=entities,
            strategy=strategy,
            expanded_query=expanded,
            search_terms=search_terms,
            reasoning=reasoning,
        )

    # ----------------------------- Normalization -----------------------------

    def _normalize_query(self, query: str) -> str:
        """Minimal normalization - preserve semantics for LLM."""
        return " ".join(query.lower().strip().split())

    # ----------------------------- LLM Classification -----------------------------

    def _classify_unified_llm(self, query: str, entities: ExtractedEntities) -> Optional[Dict[str, Any]]:
        """
        Single LLM call for complete query understanding.
        Returns None on failure.
        """
        try:
            # Provide context about pre-extracted entities
            context_parts = []
            if entities.date_range:
                context_parts.append(f"Date: {entities.date_range[0]} to {entities.date_range[1]}")
            if entities.amounts:
                context_parts.append(f"Amount filter: {entities.amounts}")
            if entities.categories:
                context_parts.append(f"Category: {entities.categories}")

            context = "\n".join(context_parts) if context_parts else "No filters detected"

            # Use advanced prompt from prompts.py
            system = get_query_understanding_prompt()

            prompt = f"""Query: "{query}"

Pre-extracted (via rules):
{context}

Return structured data following the system prompt instructions.
Focus on: intent, merchants, content_keywords, direction, search_strategy."""

            result: UnifiedQueryAnalysis = self.llm.complete_structured(
                prompt=prompt,
                response_model=UnifiedQueryAnalysis,
                system=system,
                temperature=0.0,
            )

            # Map to enums
            intent_map = {
                "find_specific": SearchIntent.FIND_SPECIFIC,
                "find_similar": SearchIntent.FIND_SIMILAR,
                "aggregate": SearchIntent.AGGREGATE,
                "list_filter": SearchIntent.LIST_FILTER,
                "temporal": SearchIntent.TEMPORAL,
                "comparative": SearchIntent.COMPARATIVE,
                "anomaly": SearchIntent.ANOMALY,
                "merchant_lookup": SearchIntent.MERCHANT_LOOKUP,
            }

            strategy_map = {
                "exact_match": SearchStrategy.EXACT_MATCH,
                "semantic": SearchStrategy.SEMANTIC,
                "hybrid": SearchStrategy.HYBRID,
                "sql_only": SearchStrategy.SQL_ONLY,
            }

            intent = intent_map.get(result.intent, SearchIntent.LIST_FILTER)
            strategy = strategy_map.get(result.search_strategy, SearchStrategy.HYBRID)

            # Process merchants (uppercase for consistency)
            merchants = [m.strip().upper() for m in result.merchants if m and m.strip()]

            # Process keywords (exclude merchants)
            merchants_lower = {m.lower() for m in merchants}
            content_keywords = [
                k.strip().lower()
                for k in result.content_keywords
                if k and k.strip() and k.strip().lower() not in merchants_lower
            ]

            return {
                "intent": intent,
                "confidence": float(result.confidence),
                "merchants": _dedup_keep_order(merchants),
                "content_keywords": _dedup_keep_order(content_keywords),
                "direction": result.direction,
                "strategy": strategy,
                "reasoning": result.reasoning,
            }

        except Exception as e:
            logger.warning("LLM classification failed: %s", e)
            return None

    # ----------------------------- Merchant Validation -----------------------------

    def _validate_merchants(self, llm_merchants: List[str]) -> Tuple[List[str], List[str]]:
        """
        Validate extracted merchants against known merchants.

        Returns:
            (validated_merchants, rejected_merchants)
            - validated: merchants found in DB or confidently fuzzy-matched (safe for SQL filter)
            - rejected: merchants not found (must be preserved as keywords to avoid signal loss)
        """
        if not llm_merchants:
            return [], []

        if not self.known_merchants:
            # No known merchants loaded - trust extraction, nothing rejected
            # Keep consistent uppercase formatting.
            trusted = [m.strip().upper() for m in llm_merchants if m and m.strip()]
            return _dedup_keep_order(trusted), []

        validated: List[str] = []
        rejected: List[str] = []

        for merchant in llm_merchants:
            if not merchant:
                continue
            m_upper = merchant.strip().upper()
            if not m_upper:
                continue

            # Exact match
            if m_upper in self.known_merchants:
                validated.append(m_upper)
                continue

            # Fuzzy match against known merchants
            best_match = self._fuzzy_match_merchant(m_upper)
            if best_match:
                validated.append(best_match)
                logger.debug("[MERCHANT] Fuzzy matched '%s' -> '%s'", merchant, best_match)
            else:
                # No match - convert to keyword (signal preserved)
                rejected.append(m_upper.lower())
                logger.info("[MERCHANT] '%s' not in DB - converting to keyword", merchant)

        return _dedup_keep_order(validated), _dedup_keep_order(rejected)

    def _fuzzy_match_merchant(self, merchant: str, threshold: float = 0.7) -> Optional[str]:
        """Token-based fuzzy matching against known merchants."""
        m_lower = merchant.lower()
        m_tokens = set(m_lower.split())

        best_score = 0.0
        best_match: Optional[str] = None

        for known in self.known_merchants:
            k_lower = known.lower()

            # Substring check (fast path)
            if m_lower in k_lower or k_lower in m_lower:
                return known

            # Token overlap (Jaccard similarity)
            k_tokens = set(k_lower.split())
            if m_tokens and k_tokens:
                intersection = len(m_tokens & k_tokens)
                union = len(m_tokens | k_tokens)
                score = intersection / union if union > 0 else 0

                if score > best_score and score >= threshold:
                    best_score = score
                    best_match = known

        return best_match

    # ----------------------------- Rule-based Extraction -----------------------------

    def _extract_amounts(self, query: str) -> List[Dict[str, Any]]:
        """Extract amount filters from query."""
        amounts: List[Dict[str, Any]] = []

        over_match = re.search(r"(?:over|above|more than|greater than|exceeding)\s*\$?(\d+(?:,\d{3})*(?:\.\d{1,2})?)", query)
        if over_match:
            amounts.append({"op": "gt", "value": float(over_match.group(1).replace(",", ""))})

        under_match = re.search(r"(?:under|below|less than|smaller than)\s*\$?(\d+(?:,\d{3})*(?:\.\d{1,2})?)", query)
        if under_match:
            amounts.append({"op": "lt", "value": float(under_match.group(1).replace(",", ""))})

        exact_match = re.search(r"(?:exactly|precisely)\s*\$?(\d+(?:,\d{3})*(?:\.\d{1,2})?)", query)
        if exact_match:
            amounts.append({"op": "eq", "value": float(exact_match.group(1).replace(",", ""))})

        between_match = re.search(
            r"between\s*\$?(\d+(?:,\d{3})*(?:\.\d{1,2})?)\s*(?:and|to|-)\s*\$?(\d+(?:,\d{3})*(?:\.\d{1,2})?)",
            query,
        )
        if between_match:
            low = float(between_match.group(1).replace(",", ""))
            high = float(between_match.group(2).replace(",", ""))
            amounts.append({"op": "between", "low": low, "high": high})

        return amounts

    def _extract_date_range(self, query: str) -> Optional[Tuple[date, date]]:
        """
        Extract date range from query.
        Returns None if no date specified (= no restriction).
        """
        q = query.lower()
        today = self.today_func()

        # 1) Explicit ISO date: 2025-12-13 or 2025/12/13
        iso_dates = re.findall(r"\b(\d{4})[-/](\d{2})[-/](\d{2})\b", q)
        parsed_iso: List[date] = []
        for y, m, d in iso_dates:
            try:
                parsed_iso.append(date(int(y), int(m), int(d)))
            except Exception:
                continue

        # 2) US date: 12/13/2025
        us_dates = re.findall(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", q)
        parsed_us: List[date] = []
        for m, d, y in us_dates:
            try:
                parsed_us.append(date(int(y), int(m), int(d)))
            except Exception:
                continue

        parsed_all = parsed_iso + parsed_us

        # Range: from X to Y
        if len(parsed_all) >= 2:
            d1, d2 = parsed_all[0], parsed_all[1]
            return (d1, d2) if d1 <= d2 else (d2, d1)

        # Single explicit date
        if len(parsed_all) == 1:
            return (parsed_all[0], parsed_all[0])

        # 3) Temporal keywords
        for keyword, fn in TEMPORAL_KEYWORDS.items():
            if keyword in q:
                return fn(today)

        # 4) Month name
        for month_name, month_num in _MONTHS.items():
            if month_name in q:
                year = today.year if month_num <= today.month else today.year - 1
                start = date(year, month_num, 1)
                if month_num == 12:
                    end = date(year, 12, 31)
                else:
                    end = date(year, month_num + 1, 1) - timedelta(days=1)
                return (start, end)

        # 5) All-time patterns
        all_time_patterns = [
            r"\bever\b", r"\bnever\b", r"\ball time\b", r"\bat any time\b",
            r"\bentire history\b", r"\bfull history\b", r"\ball transactions\b", r"\bhistory\b",
        ]
        if any(re.search(p, q) for p in all_time_patterns):
            return None

        # No date mentioned = no restriction
        return None


    def _extract_merchants_fallback(self, query: str) -> List[str]:
        """
        Simplified merchant extraction fallback.
        Only handles quoted strings, for complex patterns LLM should be used.
        """
        merchants: List[str] = []
        
        # Just extract quoted strings as merchant hints
        for quoted in re.findall(r'"([^"]+)"', query):
            cand = quoted.strip()
            if len(cand) >= 3 and not _is_all_digits_or_money(cand):
                merchants.append(cand.upper())
        
        return _dedup_keep_order(merchants)


    def _extract_categories(self, query: str) -> Tuple[List[str], List[str]]:
        """Extract categories from query using taxonomy keywords."""
        categories: List[str] = []
        subcategories: List[str] = []
        q = query.lower()
        q_words = set(re.findall(r"\b\w+\b", q))

        # Simple plural normalization
        q_words_norm = set(q_words)
        for w in q_words:
            if w.endswith("s") and len(w) > 3:
                q_words_norm.add(w[:-1])
            if w.endswith("ies") and len(w) > 4:
                q_words_norm.add(w[:-3] + "y")

        def matches_keyword(keyword: str) -> bool:
            kw = keyword.lower()
            if " " not in kw and "_" not in kw and "-" not in kw:
                return kw in q_words_norm
            if kw in q:
                return True
            kw_norm = kw.replace("_", " ").replace("-", " ")
            if kw_norm in q:
                return True
            kw_words = set(kw_norm.split())
            return bool(kw_words) and kw_words.issubset(q_words_norm)

        for kw, (cat_id, subcat_id) in self.subcategory_keywords.items():
            if matches_keyword(kw):
                if cat_id not in categories:
                    categories.append(cat_id)
                if subcat_id not in subcategories:
                    subcategories.append(subcat_id)

        for kw, cat_id in self.category_keywords.items():
            if matches_keyword(kw) and cat_id not in categories:
                categories.append(cat_id)

        return categories, subcategories

    def _extract_keywords(self, query: str, exclude_terms: Optional[Set[str]] = None) -> List[str]:
        """
        Universal keyword extractor.
        Args:
            query: Query string
            exclude_terms: Optional set of terms to exclude (e.g., merchants, categories)
        """
        words = re.findall(r"\b\w+\b", query.lower())
        exclude = exclude_terms or set()
        
        keywords = []
        for w in words:
            if (len(w) >= DEFAULT_CONFIG.min_keyword_length
                and w not in STOPWORDS
                and w not in exclude
                and not w.isdigit()):
                keywords.append(w)
        
        return _dedup_keep_order(keywords)

    def _extract_content_keywords_rules(self, query: str, entities: ExtractedEntities) -> List[str]:
        """Extract content keywords excluding already extracted entities."""
        # Build exclusion set from entities
        exclude = set()
        for m in entities.merchants:
            exclude.add(m.lower())
        for c in entities.categories:
            exclude.add(c.lower())
        for s in entities.subcategories:
            exclude.add(s.lower())
        
        return self._extract_keywords(query, exclude)


    # ----------------------------- Intent Classification (Simplified Fallback) -----------------------------

    def _classify_intent_rules(self, query: str, entities: ExtractedEntities) -> Tuple[SearchIntent, float]:
        """
        Simplified rule-based intent classification (last resort fallback).
        Only used when LLM is completely unavailable.
        """
        q = query.lower()

        # Only the most obvious patterns - if uncertain, default to LIST_FILTER
        if any(p in q for p in ["total", "sum", "average", "avg"]):
            return SearchIntent.AGGREGATE, 0.90

        if re.search(r"\b(transactions?\s+like|similar\s+to|same\s+as)\b", q):
            return SearchIntent.FIND_SIMILAR, 0.85

        if entities.merchants:
            return SearchIntent.MERCHANT_LOOKUP, 0.70

        # Default: safe fallback
        return SearchIntent.LIST_FILTER, 0.50


    def _select_strategy(self, intent: SearchIntent, entities: ExtractedEntities) -> SearchStrategy:
        """Select search strategy based on intent and entities."""
        if intent == SearchIntent.AGGREGATE:
            return SearchStrategy.SQL_ONLY
        if intent in (SearchIntent.MERCHANT_LOOKUP, SearchIntent.FIND_SPECIFIC) and entities.merchants:
            return SearchStrategy.EXACT_MATCH
        if intent == SearchIntent.FIND_SIMILAR:
            return SearchStrategy.SEMANTIC
        return SearchStrategy.HYBRID

    def _get_strategy_reasoning(self, strategy: SearchStrategy) -> str:
        """Get human-readable reasoning for strategy."""
        return {
            SearchStrategy.SQL_ONLY: "Aggregate intent detected — SQL-only is optimal.",
            SearchStrategy.EXACT_MATCH: "Specific merchant lookup detected — SQL exact/LIKE matching.",
            SearchStrategy.SEMANTIC: "Similarity intent detected — vector semantic search.",
            SearchStrategy.HYBRID: "General intent — use SQL filters + vector semantic search.",
        }.get(strategy, "Default strategy selection.")

    # ----------------------------- Query Building -----------------------------

    def _build_search_query(self, query: str, entities: ExtractedEntities) -> Tuple[str, List[str]]:
        """
        Build expanded query for vector search.
        Now includes optional LLM-based expansion.
        """
        terms: List[str] = [query]
        search_terms: List[str] = []

        # Add LLM-extracted content keywords
        search_terms.extend(entities.content_keywords)

        # Add merchants
        for m in entities.merchants:
            terms.append(m.lower())
            search_terms.append(m)

        # Add category display names from taxonomy
        for cat in entities.categories:
            cat_data = (self.taxonomy.get("categories", {}) or {}).get(cat, {}) or {}
            display_name = (cat_data.get("display_name") or cat).lower()
            if display_name != cat.lower():
                terms.append(display_name)
                search_terms.append(display_name)

        # ✨ PHASE 2: LLM-based query expansion
        if self.llm:
            expansion_terms = self._expand_query_with_llm(query, entities)
            for exp_term in expansion_terms:
                if exp_term and exp_term not in terms:
                    terms.append(exp_term)
                    search_terms.append(exp_term)

        expanded = " ".join(_dedup_keep_order(terms))
        return expanded, _dedup_keep_order(search_terms)

    def _expand_query_with_llm(self, query: str, entities: ExtractedEntities) -> List[str]:
        """
        Use LLM to expand query with synonyms and related terms.
        Uses advanced query expansion prompt from prompts.py.
        
        Returns:
            List of expansion terms (empty if LLM fails)
        """
        if not self.llm:
            return []
        
        try:
            system = get_query_expansion_prompt()
            
            # Build context
            context_parts = []
            if entities.merchants:
                context_parts.append(f"Merchants: {', '.join(entities.merchants)}")
            if entities.categories:
                context_parts.append(f"Categories: {', '.join(entities.categories)}")
            if entities.content_keywords:
                context_parts.append(f"Keywords: {', '.join(entities.content_keywords)}")
            
            context = "\n".join(context_parts) if context_parts else "No entities extracted"
            
            prompt = f"""Expand this financial search query with relevant synonyms and related terms.

Query: "{query}"

Extracted Entities:
{context}

Return 5-10 relevant expansion terms as a comma-separated list.
Focus on:
- Merchant variations (official names, abbreviations, domain names)
- Category synonyms
- Financial terminology
- Action synonyms

Example: If query is "Netflix subscription", return: netflix.com, NFLX, streaming, monthly, recurring, membership"""

            class QueryExpansion(BaseModel):
                expansion_terms: List[str] = Field(
                    description="5-10 relevant expansion terms",
                    max_length=10
                )
                reasoning: str = Field(description="Brief explanation of expansion strategy")

            result: QueryExpansion = self.llm.complete_structured(
                prompt=prompt,
                response_model=QueryExpansion,
                system=system,
                temperature=0.3,  # Slightly higher for creativity
            )
            
            # Clean and deduplicate
            expansions = [t.strip().lower() for t in result.expansion_terms if t and t.strip()]
            logger.debug("[EXPANSION] Query='%s' → Expansions=%s", query, expansions)
            
            return _dedup_keep_order(expansions[:10])  # Limit to 10
            
        except Exception as e:
            logger.warning("Query expansion failed: %s", e)
            return []


# =============================================================================
# MULTI-SOURCE RETRIEVER
# =============================================================================

class MultiSourceRetriever:
    """
    Retrieves from SQL and Vector sources.
    Includes smart limit expansion and filter relaxation.
    """

    SMART_LIMIT_THRESHOLD = 100

    def __init__(self, db: DBClient, vector_store: VectorStore):
        self.db = db
        self.vector_store = vector_store

    def retrieve(
        self,
        understanding: QueryUnderstanding,
        tenant_id: str,
        top_k: int = 50,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[SearchMatch], Dict[str, Any]]:
        all_matches: List[SearchMatch] = []
        metadata: Dict[str, Any] = {
            "sources_used": [],
            "filters_applied": {},
            "sql_query": None,
            "vector_query": None,
            "filter_relaxed": False,
            "relaxation_type": None,
            "total_matching": 0,
            "result_limited": False,
        }

        # Build base filters from entities
        filters = self._build_filters(understanding.entities, tenant_id, understanding.intent)
        
        # Apply overrides (Graph-driven constraints take precedence)
        if overrides:
            for k, v in overrides.items():
                if v is None:
                    filters.pop(k, None)  # Explicit removal (e.g. relax date)
                else:
                    filters[k] = v  # Explicit set (e.g. force category)
            
            # If overrides modified the filters, log it
            logger.info("[RETRIEVER] Applied overrides: %s -> Effective: %s", overrides, filters)

        metadata["filters_applied"] = dict(filters)

        strategy = understanding.strategy

        # Smart count (only when selective filters exist)
        total_count = self._count_matching_if_reasonable(filters, tenant_id)
        metadata["total_matching"] = total_count

        effective_top_k = top_k
        if total_count != -1 and 0 < total_count <= self.SMART_LIMIT_THRESHOLD:
            effective_top_k = total_count
            metadata["result_limited"] = False
        elif total_count > self.SMART_LIMIT_THRESHOLD:
            metadata["result_limited"] = True

        # Filter relaxation setup
        has_categories = bool(filters.get("categories"))

        def relax_category(f: Dict[str, Any]) -> Dict[str, Any]:
            nf = dict(f)
            nf.pop("categories", None)
            nf.pop("subcategories", None)
            return nf

        # ===============================
        # SQL_ONLY / EXACT_MATCH
        # ===============================
        if strategy in (SearchStrategy.SQL_ONLY, SearchStrategy.EXACT_MATCH):
            matches, sql = self._retrieve_sql(filters, tenant_id, effective_top_k, normalize_for_hybrid=False)

            if not matches and has_categories:
                logger.info("[RELAX] 0 SQL results -> removing category filters.")
                relaxed = relax_category(filters)
                matches, sql = self._retrieve_sql(relaxed, tenant_id, effective_top_k, normalize_for_hybrid=False)
                if matches:
                    metadata["filter_relaxed"] = True
                    metadata["relaxation_type"] = "category_removed"
                    metadata["filters_applied"] = relaxed

            all_matches.extend(matches)
            metadata["sources_used"].append("sql")
            metadata["sql_query"] = sql

            return self._finalize(all_matches), metadata

        # ===============================
        # SEMANTIC (vector-only)
        # ===============================
        if strategy == SearchStrategy.SEMANTIC:
            v = self._retrieve_vector(understanding.expanded_query, filters, tenant_id, top_k, alpha=0.90)
            all_matches.extend(v)
            metadata["sources_used"].append("vector")
            metadata["vector_query"] = understanding.expanded_query
            return self._finalize(all_matches), metadata

        # ===============================
        # HYBRID
        # ===============================
        # Fetch 2x results to ensure recall before fusion
        effective_sql_limit = effective_top_k * 2
        effective_vector_limit = effective_top_k * 2

        sql_matches, sql = self._retrieve_sql(filters, tenant_id, effective_sql_limit, normalize_for_hybrid=True)
        effective_filters_for_vector = dict(filters)

        if not sql_matches and has_categories:
            logger.info("[RELAX] HYBRID: 0 SQL results -> removing category filters.")
            relaxed = relax_category(filters)
            sql_matches, sql = self._retrieve_sql(relaxed, tenant_id, effective_sql_limit, normalize_for_hybrid=True)
            if sql_matches:
                metadata["filter_relaxed"] = True
                metadata["relaxation_type"] = "category_removed"
                metadata["filters_applied"] = relaxed
            effective_filters_for_vector = relaxed

        all_matches.extend(sql_matches)
        metadata["sources_used"].append("sql")
        metadata["sql_query"] = sql

        vector_matches = self._retrieve_vector(
            understanding.expanded_query,
            effective_filters_for_vector,
            tenant_id,
            effective_vector_limit,
            alpha=0.70,
        )

        # Smart Relaxation for Vector (Independent of SQL)
        # If vector search yielded nothing but we had strict filters, try again without them.
        if not vector_matches and effective_filters_for_vector.get("categories"):
            logger.info("[RELAX] HYBRID: 0 Vector results -> retrying without category filters.")
            relaxed_vec_filters = relax_category(effective_filters_for_vector)
            vector_matches = self._retrieve_vector(
                understanding.expanded_query,
                relaxed_vec_filters,
                tenant_id,
                effective_vector_limit,
                alpha=0.70,
            )
            # Mark metadata if we found something this way
            if vector_matches:
                metadata["filter_relaxed"] = True
                metadata["relaxation_type"] = "vector_category_removed"
                # Note: We don't overwrite metadata["filters_applied"] here to preserve SQL context,
                # but we could track it separately if needed.
        all_matches.extend(vector_matches)
        metadata["sources_used"].append("vector")
        metadata["vector_query"] = understanding.expanded_query

        return self._finalize(all_matches), metadata

    def _build_filters(self, entities: ExtractedEntities, tenant_id: str, intent: Optional[SearchIntent]) -> Dict[str, Any]:
        """
        Build filters from entities.
        Categories are SOFT by default (prevents over-filtering).
        """
        filters: Dict[str, Any] = {"tenant_id": tenant_id}

        if entities.date_range is not None:
            filters["date_start"] = entities.date_range[0]
            filters["date_end"] = entities.date_range[1]

        if entities.direction:
            filters["direction"] = entities.direction

        # Categories: soft by default, hard only for explicit category queries without merchant
        is_explicit_category_query = (
            intent == SearchIntent.LIST_FILTER and
            not entities.merchants and
            len(entities.categories) == 1
        )

        if entities.categories:
            if is_explicit_category_query:
                filters["categories"] = list(entities.categories)
            else:
                filters["soft_categories"] = list(entities.categories)

        if entities.subcategories:
            if is_explicit_category_query:
                filters["subcategories"] = list(entities.subcategories)
            else:
                filters["soft_subcategories"] = list(entities.subcategories)

        if entities.merchants:
            filters["merchants"] = list(entities.merchants)

        for amount in entities.amounts:
            op = amount.get("op")
            if op == "gt":
                filters["min_amount"] = amount["value"]
            elif op == "lt":
                filters["max_amount"] = amount["value"]
            elif op == "between":
                filters["min_amount"] = amount["low"]
                filters["max_amount"] = amount["high"]
            elif op == "eq":
                filters["exact_amount"] = amount["value"]

        # Keywords as soft boosts
        if entities.content_keywords:
            filters["soft_keywords"] = list(entities.content_keywords)

        return filters

    def _count_matching_if_reasonable(self, filters: Dict[str, Any], tenant_id: str) -> int:
        """Count matching rows only when selective filters exist."""
        selective_keys = {"date_start", "date_end", "direction", "merchants", "categories", "min_amount", "max_amount", "exact_amount"}
        if not any(k in filters for k in selective_keys):
            return -1
        return self._count_matching(filters, tenant_id)

    def _count_matching(self, filters: Dict[str, Any], tenant_id: str) -> int:
        """Execute COUNT query."""
        where_parts = ["tenant_id = ?"]
        params: List[Any] = [tenant_id]

        if filters.get("date_start") is not None:
            where_parts.append("date_time >= ?")
            params.append(datetime.combine(filters["date_start"], datetime.min.time()))

        if filters.get("date_end") is not None:
            where_parts.append("date_time <= ?")
            params.append(datetime.combine(filters["date_end"], datetime.max.time()))

        if filters.get("direction"):
            where_parts.append("direction = ?")
            params.append(filters["direction"])

        if filters.get("merchants"):
            conds = []
            for m in filters["merchants"]:
                # Search in both merchant_norm AND description (for P2P transfers)
                conds.append("(UPPER(merchant_norm) LIKE ? OR UPPER(description) LIKE ?)")
                params.append(f"%{str(m).upper()}%")
                params.append(f"%{str(m).upper()}%")
            where_parts.append(f"({' OR '.join(conds)})")

        if filters.get("categories"):
            conds = []
            for c in filters["categories"]:
                cl = str(c).lower().strip()
                conds.append("LOWER(COALESCE(category_final, category)) LIKE ?")
                params.append(f"%{cl}%")
            where_parts.append(f"({' OR '.join(conds)})")

        if filters.get("exact_amount") is not None:
            where_parts.append("ABS(amount) = ?")
            params.append(filters["exact_amount"])
        else:
            if filters.get("min_amount") is not None:
                where_parts.append("ABS(amount) >= ?")
                params.append(filters["min_amount"])
            if filters.get("max_amount") is not None:
                where_parts.append("ABS(amount) <= ?")
                params.append(filters["max_amount"])

        sql = f"SELECT COUNT(*) as cnt FROM transactions WHERE {' AND '.join(where_parts)}"
        try:
            rows = self.db.execute_query(sql, params)
            return int(rows[0]["cnt"]) if rows else 0
        except Exception as e:
            logger.warning("COUNT failed: %s", e)
            return -1

    def _retrieve_sql(
        self,
        filters: Dict[str, Any],
        tenant_id: str,
        limit: int,
        normalize_for_hybrid: bool,
    ) -> Tuple[List[SearchMatch], str]:
        """Retrieve from SQL with filters."""
        where_parts = ["tenant_id = ?"]
        params: List[Any] = [tenant_id]

        if filters.get("date_start") is not None:
            where_parts.append("date_time >= ?")
            params.append(datetime.combine(filters["date_start"], datetime.min.time()))

        if filters.get("date_end") is not None:
            where_parts.append("date_time <= ?")
            params.append(datetime.combine(filters["date_end"], datetime.max.time()))

        if filters.get("direction"):
            where_parts.append("direction = ?")
            params.append(filters["direction"])

        if filters.get("categories"):
            cats = filters["categories"]
            cat_conditions: List[str] = []
            for cat in cats:
                cat_lower = str(cat).lower().strip()
                cat_norm = (
                    cat_lower.replace(" & ", "_and_")
                    .replace("&", "_and_")
                    .replace(" ", "_")
                    .replace("-", "_")
                )
                cat_singular = cat_norm.rstrip("s") if cat_norm.endswith("s") and len(cat_norm) > 3 else cat_norm

                cat_conditions.append(
                    "("
                    "LOWER(COALESCE(category_final, category)) LIKE ? OR "
                    "LOWER(COALESCE(category_final, category)) LIKE ? OR "
                    "LOWER(COALESCE(category_final, category)) LIKE ? OR "
                    "LOWER(COALESCE(subcategory_final, subcategory)) LIKE ? OR "
                    "LOWER(COALESCE(subcategory_final, subcategory)) LIKE ? OR "
                    "LOWER(COALESCE(subcategory_final, subcategory)) LIKE ?"
                    ")"
                )
                params.extend([
                    f"%{cat_lower}%", f"%{cat_norm}%", f"%{cat_singular}%",
                    f"%{cat_lower}%", f"%{cat_norm}%", f"%{cat_singular}%",
                ])
            where_parts.append(f"({' OR '.join(cat_conditions)})")

        if filters.get("merchants"):
            conds = []
            for m in filters["merchants"]:
                # Search in both merchant_norm AND description (for P2P transfers)
                conds.append("(UPPER(merchant_norm) LIKE ? OR UPPER(description) LIKE ?)")
                params.append(f"%{str(m).upper()}%")
                params.append(f"%{str(m).upper()}%")
            where_parts.append(f"({' OR '.join(conds)})")

        if filters.get("exact_amount") is not None:
            where_parts.append("ABS(amount) = ?")
            params.append(filters["exact_amount"])
        else:
            if filters.get("min_amount") is not None:
                where_parts.append("ABS(amount) >= ?")
                params.append(filters["min_amount"])
            if filters.get("max_amount") is not None:
                where_parts.append("ABS(amount) <= ?")
                params.append(filters["max_amount"])

        sql = f"""
            SELECT
                tx_id,
                date_time,
                amount,
                merchant_norm,
                description,
                COALESCE(category_final, category) as category,
                COALESCE(subcategory_final, subcategory) as subcategory,
                direction
            FROM transactions
            WHERE {' AND '.join(where_parts)}
            ORDER BY date_time DESC
            LIMIT {int(limit)}
        """.strip()

        try:
            rows = self.db.execute_query(sql, params)
            out: List[SearchMatch] = []

            for i, row in enumerate(rows):
                # Hybrid normalization: modest SQL baseline so vector can compete
                if normalize_for_hybrid:
                    score = 0.55 - (i * 0.005)
                else:
                    score = 0.90 - (i * 0.01)

                out.append(
                    SearchMatch(
                        tx_id=row["tx_id"],
                        score=_clamp(float(score)),
                        source="sql",
                        date_time=row.get("date_time"),
                        amount=row.get("amount"),
                        merchant_norm=row.get("merchant_norm"),
                        description=row.get("description"),
                        category=row.get("category"),
                        subcategory=row.get("subcategory"),
                        direction=row.get("direction"),
                        match_reason="SQL filter match",
                    )
                )

            return out, sql

        except Exception as e:
            logger.error("SQL retrieval error: %s", e)
            return [], sql

    def _retrieve_vector(
        self,
        query: str,
        filters: Dict[str, Any],
        tenant_id: str,
        limit: int,
        alpha: float,
    ) -> List[SearchMatch]:
        """Retrieve from vector store."""
        pinecone_filters: Dict[str, Any] = {"tenant_id": {"$eq": tenant_id}}

        if filters.get("direction"):
            pinecone_filters["direction"] = {"$eq": filters["direction"]}

        # Hard categories only (soft_categories not applied to vector)
        if filters.get("categories"):
            cats = filters["categories"]
            normalized = set()
            for cat in cats:
                cl = str(cat).lower().strip()
                cn = (
                    cl.replace(" & ", "_and_")
                    .replace("&", "_and_")
                    .replace(" ", "_")
                    .replace("-", "_")
                )
                cs = cn.rstrip("s") if cn.endswith("s") and len(cn) > 3 else cn
                normalized.update([cl, cn, cs])

            all_cats = list(normalized)
            pinecone_filters["category"] = {"$in": all_cats} if len(all_cats) > 1 else {"$eq": all_cats[0]}

        try:
            results = self.vector_store.search(
                query=query,
                top_k=int(limit),
                alpha=float(alpha),
                filters=pinecone_filters,
            )

            matches: List[SearchMatch] = []
            for r in results:
                md = r.get("metadata", {}) or {}
                raw = float(r.get("score", 0.0) or 0.0)
                score = _clamp(0.35 + 0.65 * _clamp(raw))

                matches.append(
                    SearchMatch(
                        tx_id=r["tx_id"],
                        score=score,
                        source="vector",
                        date_time=self._parse_datetime(md.get("date_str") or md.get("date_time")),
                        amount=md.get("amount"),
                        merchant_norm=md.get("merchant_norm"),
                        description=md.get("description"),
                        category=md.get("category"),
                        subcategory=md.get("subcategory"),
                        direction=md.get("direction"),
                        match_reason="Vector semantic match",
                    )
                )

            return matches

        except Exception as e:
            logger.error("Vector retrieval error: %s", e)
            return []

    @staticmethod
    def _parse_datetime(value: Any) -> Optional[datetime]:
        if not value:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except Exception:
                try:
                    return datetime.strptime(value, "%Y-%m-%d")
                except Exception:
                    return None
        return None

    @staticmethod
    def _finalize(matches: List[SearchMatch]) -> List[SearchMatch]:
        """Deduplicate by tx_id, keep best score."""
        best: Dict[str, SearchMatch] = {}
        for m in matches:
            if m.tx_id not in best or (m.score or 0.0) > (best[m.tx_id].score or 0.0):
                best[m.tx_id] = m
        return sorted(best.values(), key=lambda x: x.score, reverse=True)


# =============================================================================
# RERANKER
# =============================================================================

class SearchReranker:
    """
    Deterministic reranking with improved merchant matching.

    Features:
    - Token-based fuzzy merchant matching
    - Edit distance fallback for typos
    - Category and keyword boosts
    - Temporal recency boost
    - Optional LLM rerank
    """

    def __init__(self, llm_client: Optional[LLMClient] = None, today_func=_today_utc_date):
        self.llm = llm_client
        self.today_func = today_func

    def rerank(
        self,
        query: str,
        matches: List[SearchMatch],
        understanding: QueryUnderstanding,
        filters: Optional[Dict[str, Any]] = None,
        use_llm: bool = False,
        top_k: int = 20,
    ) -> List[SearchMatch]:
        if not matches:
            return []

        scored = self._score_rerank(matches, understanding, filters or {})

        if use_llm and self.llm and len(scored) > 5:
            scored = self._llm_rerank(query, scored[:30], understanding)

        return scored[:top_k]

    def _safe_days_ago(self, dt: datetime) -> int:
        try:
            return (self.today_func() - dt.date()).days
        except Exception:
            return 999999

    def _score_rerank(
        self,
        matches: List[SearchMatch],
        understanding: QueryUnderstanding,
        filters: Dict[str, Any],
    ) -> List[SearchMatch]:
        e = understanding.entities
        merchants_q = [m.lower() for m in e.merchants]

        hard_cats = [c.lower() for c in e.categories]
        soft_cats = [c.lower() for c in filters.get("soft_categories", [])]
        soft_subcats = [s.lower() for s in filters.get("soft_subcategories", [])]
        all_cats = hard_cats + soft_cats

        soft_kws = [k.lower() for k in filters.get("soft_keywords", [])]
        keywords_q = _dedup_keep_order([k.lower() for k in (e.content_keywords or e.keywords)] + soft_kws)

        for m in matches:
            base = _clamp(float(m.score or 0.0))
            boost = 0.0
            penalty = 0.0

            merchant_text = (m.merchant_norm or "").lower()
            desc_text = (m.description or "").lower()
            cat_text = (m.category or "").lower()
            subcat_text = (m.subcategory or "").lower()

            # ============= IMPROVED MERCHANT MATCHING =============
            if merchants_q:
                matched_merchant = False
                best_score = 0.0

                for query_merchant in merchants_q:
                    # 1. Exact substring (highest confidence)
                    if query_merchant in merchant_text:
                        matched_merchant = True
                        best_score = max(best_score, 1.0)
                        continue

                    # 2. Token overlap
                    q_tokens = set(query_merchant.split())
                    a_tokens = set(merchant_text.split()) if merchant_text else set()

                    # Remove noise suffixes
                    noise_suffixes = {"premium", "pro", "plus", "basic", "free", "subscription", "music", "tv"}
                    q_core = q_tokens - noise_suffixes
                    a_core = a_tokens - noise_suffixes

                    if q_core and a_core:
                        overlap = len(q_core & a_core) / len(q_core)
                        if overlap >= 0.5:
                            matched_merchant = True
                            best_score = max(best_score, overlap)
                            continue
                    elif q_tokens and a_tokens:
                        overlap = len(q_tokens & a_tokens) / len(q_tokens)
                        if overlap >= 0.5:
                            matched_merchant = True
                            best_score = max(best_score, overlap)
                            continue

                    # 3. Edit distance for typos (lower threshold)
                    if merchant_text:
                        edit_ratio = _levenshtein_ratio(query_merchant, merchant_text)
                        if edit_ratio >= 0.75:
                            matched_merchant = True
                            best_score = max(best_score, edit_ratio * 0.8)

                if matched_merchant:
                    boost += 0.30 * best_score
                    for mm in e.merchants:
                        if mm.lower() in merchant_text or any(t in merchant_text for t in mm.lower().split()):
                            m.highlight_terms.append(mm)
                elif merchant_text:
                    # Softer penalty
                    penalty += 0.10

            # ============= CATEGORY BOOSTS =============
            if all_cats and cat_text and any(c in cat_text for c in all_cats):
                boost += 0.15

            if soft_subcats and subcat_text and any(s in subcat_text for s in soft_subcats):
                boost += 0.10

            # ============= KEYWORD BOOSTS =============
            if keywords_q:
                combined_text = f"{desc_text} {merchant_text}"
                matched_keywords = [k for k in keywords_q if k in combined_text]
                if matched_keywords:
                    boost += 0.18 + (0.03 * min(len(matched_keywords), 3))
                    for kk in matched_keywords:
                        m.highlight_terms.append(kk)

            # ============= TEMPORAL RECENCY =============
            if understanding.intent == SearchIntent.TEMPORAL and m.date_time:
                days_ago = self._safe_days_ago(m.date_time)
                boost += max(0.0, 0.20 - (days_ago * 0.01))

            m.highlight_terms = _dedup_keep_order(m.highlight_terms)
            m.score = _clamp(base + boost - penalty)

        return sorted(matches, key=lambda x: x.score, reverse=True)

    def _llm_rerank(
        self,
        query: str,
        matches: List[SearchMatch],
        understanding: QueryUnderstanding,
    ) -> List[SearchMatch]:
        """
        Optional LLM-based reranking using advanced prompt.
        Uses get_result_reranking_prompt() from prompts.py.
        """
        try:
            items: List[str] = []
            for i, m in enumerate(matches[:20]):
                items.append(
                    f"{i+1}. [{m.tx_id}] {m.merchant_norm or 'Unknown'} - "
                    f"${abs(float(m.amount or 0.0)):.2f} - {m.category or 'Uncategorized'} - "
                    f"{m.date_time.strftime('%Y-%m-%d') if m.date_time else 'Unknown date'} - "
                    f"{m.description[:50] if m.description else 'No description'}"
                )

            # Use advanced prompt from prompts.py
            system = get_result_reranking_prompt()

            prompt = f"""Rerank these financial transactions by relevance.

Query: "{query}"
Intent: {understanding.intent.value}
Search Strategy: {understanding.strategy.value}

Transactions to rank:
{chr(10).join(items)}

For each transaction, provide:
1. tx_id (from the bracketed ID)
2. relevance_score (0.0 = irrelevant, 1.0 = perfect match)
3. brief match_explanation

Focus on:
- Query-transaction semantic match
- Merchant relevance
- Intent alignment
- Date relevance (if temporal query)
- Amount relevance (if amount mentioned)"""

            result: RerankedResults = self.llm.complete_structured(
                prompt=prompt,
                response_model=RerankedResults,
                system=system,
                model=settings.model_reranker,
                temperature=0.0,
            )

            score_map = {r.tx_id: float(r.relevance_score) for r in result.results}
            for m in matches:
                if m.tx_id in score_map:
                    llm_score = score_map[m.tx_id]
                    # Blend LLM score with deterministic score (70% LLM, 30% deterministic)
                    m.score = _clamp(0.7 * llm_score + 0.3 * m.score)
                    
                    # Add LLM reasoning to match_reason
                    for r in result.results:
                        if r.tx_id == m.tx_id:
                            m.match_reason = f"{m.match_reason} | LLM: {r.match_explanation}"
                            break

            logger.debug("[LLM_RERANK] Reranked %d results with LLM scores", len(score_map))
            return sorted(matches, key=lambda x: x.score, reverse=True)

        except Exception as e:
            logger.warning("LLM rerank failed: %s", e)
            return matches


# =============================================================================
# MAIN SEARCH ENGINE
# =============================================================================

class ProfessionalSearchEngine:
    """
    Production-ready search engine.

    Usage:
        engine = ProfessionalSearchEngine(db, vector_store, llm_client, taxonomy)
        result = engine.search("Did I ever pay YouTube Premium?", tenant_id="default_tenant")
    """

    def __init__(
        self,
        db: DBClient,
        vector_store: VectorStore,
        llm_client: Optional[LLMClient] = None,
        taxonomy: Optional[Dict[str, Any]] = None,
        known_merchants: Optional[Set[str]] = None,
        today_func=_today_utc_date,
    ):
        self.db = db
        self.vector_store = vector_store
        self.llm = llm_client
        self.taxonomy = taxonomy or {}
        self.today_func = today_func

        # Load known merchants if not provided
        self.known_merchants = known_merchants or set()

        self.query_understanding = QueryUnderstandingEngine(
            llm_client=self.llm,
            taxonomy=self.taxonomy,
            known_merchants=self.known_merchants,
            today_func=self.today_func,
        )
        self.retriever = MultiSourceRetriever(db=self.db, vector_store=self.vector_store)
        self.reranker = SearchReranker(llm_client=self.llm, today_func=self.today_func)

    def search(
        self,
        query: str,
        tenant_id: str,
        top_k: int = 20,
        use_llm_rerank: bool = False,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> SearchResult:
        """Execute search and return results."""
        start = time.time()

        understanding = self.query_understanding.understand(query)

        matches, meta = self.retriever.retrieve(
            understanding=understanding,
            tenant_id=tenant_id,
            top_k=max(top_k * 2, 50),
            overrides=overrides,
        )

        # Enrich BEFORE reranking
        enriched = self._enrich_missing_fields(matches, tenant_id)

        reranked = self.reranker.rerank(
            query=query,
            matches=enriched,
            understanding=understanding,
            filters=meta.get("filters_applied", {}),
            use_llm=use_llm_rerank,
            top_k=top_k,
        )

        elapsed_ms = int((time.time() - start) * 1000)

        raw_total = meta.get("total_matching", 0)
        safe_total = max(0, int(raw_total)) if raw_total != -1 else 0

        return SearchResult(
            query_understanding=understanding,
            matches=reranked,
            total_found=len(reranked),
            search_time_ms=elapsed_ms,
            sources_used=meta.get("sources_used", []),
            filters_applied=meta.get("filters_applied", {}),
            total_matching=safe_total,
            result_limited=bool(meta.get("result_limited", False)),
            sql_query=meta.get("sql_query"),
            vector_query=meta.get("vector_query"),
        )

    def _enrich_missing_fields(self, matches: List[SearchMatch], tenant_id: str) -> List[SearchMatch]:
        """Fetch missing fields from SQL for vector results."""
        needs = [m for m in matches if m.description is None or m.merchant_norm is None or m.date_time is None]
        if not needs:
            return matches

        tx_ids = [m.tx_id for m in needs]
        placeholders = ",".join(["?"] * len(tx_ids))

        sql = f"""
            SELECT
                tx_id, date_time, amount, merchant_norm, description,
                COALESCE(category_final, category) as category,
                COALESCE(subcategory_final, subcategory) as subcategory,
                direction
            FROM transactions
            WHERE tx_id IN ({placeholders})
              AND tenant_id = ?
        """.strip()

        try:
            rows = self.db.execute_query(sql, tx_ids + [tenant_id])
            row_map = {r["tx_id"]: r for r in rows}

            for m in matches:
                r = row_map.get(m.tx_id)
                if not r:
                    continue
                m.date_time = m.date_time or r.get("date_time")
                m.amount = m.amount if m.amount is not None else r.get("amount")
                m.merchant_norm = m.merchant_norm or r.get("merchant_norm")
                m.description = m.description or r.get("description")
                m.category = m.category or r.get("category")
                m.subcategory = m.subcategory or r.get("subcategory")
                m.direction = m.direction or r.get("direction")

        except Exception as e:
            logger.error("Enrichment failed: %s", e)

        return matches

    def search_similar(self, tx_id: str, tenant_id: str, top_k: int = 10) -> SearchResult:
        """Find transactions similar to a given transaction."""
        sql = """
            SELECT merchant_norm, description, COALESCE(category_final, category) as category
            FROM transactions
            WHERE tx_id = ? AND tenant_id = ?
        """.strip()

        rows = self.db.execute_query(sql, [tx_id, tenant_id])
        if not rows:
            uq = QueryUnderstanding(
                original_query=f"Similar to {tx_id}",
                normalized_query=f"similar to {tx_id}",
                intent=SearchIntent.FIND_SIMILAR,
                confidence=0.0,
                entities=ExtractedEntities(),
                strategy=SearchStrategy.SEMANTIC,
                expanded_query="",
                search_terms=[],
                reasoning="Transaction not found.",
            )
            return SearchResult(query_understanding=uq, matches=[], total_found=0)

        tx = rows[0]
        parts: List[str] = []
        if tx.get("merchant_norm"):
            parts.append(str(tx["merchant_norm"]))
        if tx.get("description"):
            parts.append(str(tx["description"])[:120])
        if tx.get("category"):
            parts.append(str(tx["category"]))

        return self.search(" ".join(parts), tenant_id=tenant_id, top_k=top_k, use_llm_rerank=False)

    def load_known_merchants(self, tenant_id: str, min_count: int = 3, limit: int = 1000) -> None:
        """Load known merchants from database for validation (DuckDB compatible)."""

        # ✅ FIX #2: DuckDB-friendly SQL (no DISTINCT + GROUP BY combo; stable ORDER BY alias)
        sql = """
            SELECT
                UPPER(merchant_norm) as merchant,
                COUNT(*) as tx_count
            FROM transactions
            WHERE tenant_id = ?
              AND merchant_norm IS NOT NULL
              AND merchant_norm <> ''
            GROUP BY UPPER(merchant_norm)
            HAVING COUNT(*) >= ?
            ORDER BY tx_count DESC
            LIMIT ?
        """.strip()

        try:
            rows = self.db.execute_query(sql, [tenant_id, min_count, limit])
            self.known_merchants = {r["merchant"] for r in rows if r.get("merchant")}
            self.query_understanding.known_merchants = self.known_merchants
            logger.info(f"Loaded {len(self.known_merchants)} known merchants for tenant {tenant_id}")
        except Exception as e:
            logger.warning(f"Failed to load known merchants: {e}")


# =============================================================================
# FACTORY (SINGLETON)
# =============================================================================

_search_engine: Optional[ProfessionalSearchEngine] = None


def get_search_engine(
    db: Optional[DBClient] = None,
    vector_store: Optional[VectorStore] = None,
    llm_client: Optional[LLMClient] = None,
    taxonomy: Optional[Dict[str, Any]] = None,
    known_merchants: Optional[Set[str]] = None,
) -> ProfessionalSearchEngine:
    """Get or create search engine singleton."""
    global _search_engine

    if _search_engine is None:
        if db is None or vector_store is None:
            raise ValueError("db and vector_store are required for first initialization.")
        _search_engine = ProfessionalSearchEngine(
            db=db,
            vector_store=vector_store,
            llm_client=llm_client,
            taxonomy=taxonomy,
            known_merchants=known_merchants,
        )

    return _search_engine

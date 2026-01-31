"""
Statement Copilot - Pydantic Schemas
====================================
Structured output schemas for Anthropic's guaranteed JSON feature.
All schemas use `extra="forbid"` for strict validation.

bunq Alignment: Strict JSON and structured outputs.
"""

from typing import Optional, List, Dict, Any, Literal, Union
from datetime import datetime, date
from pydantic import BaseModel, Field, ConfigDict, field_validator
from enum import Enum
import uuid


# -----------------------------------------------------------------------------
# ENUMS
# -----------------------------------------------------------------------------

class IntentType(str, Enum):
    """User intent classification"""
    ANALYTICS = "ANALYTICS"      # SQL-based calculations
    LOOKUP = "LOOKUP"            # Vector search + SQL fetch
    ACTION = "ACTION"            # Export, report, reminder
    EXPLAIN = "EXPLAIN"          # Explain previous result
    CLARIFY = "CLARIFY"          # Need more info from user
    CHITCHAT = "CHITCHAT"        # General conversation


class RiskLevel(str, Enum):
    """Risk assessment for actions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class MetricType(str, Enum):
    """Available SQL metric types"""
    SUM_AMOUNT = "sum_amount"
    COUNT_TX = "count_tx"
    AVG_AMOUNT = "avg_amount"
    MEDIAN_AMOUNT = "median_amount"
    MIN_MAX_AMOUNT = "min_max_amount"
    TOP_MERCHANTS = "top_merchants"
    TOP_CATEGORIES = "top_categories"
    CATEGORY_BREAKDOWN = "category_breakdown"
    MERCHANT_BREAKDOWN = "merchant_breakdown"
    LARGEST_TRANSACTIONS = "largest_transactions"
    SMALLEST_TRANSACTIONS = "smallest_transactions"
    DAILY_TREND = "daily_trend"
    WEEKLY_TREND = "weekly_trend"
    MONTHLY_TREND = "monthly_trend"
    MONTHLY_COMPARISON = "monthly_comparison"
    YEAR_OVER_YEAR = "year_over_year"
    SUBSCRIPTION_LIST = "subscription_list"
    RECURRING_PAYMENTS = "recurring_payments"
    ANOMALY_DETECTION = "anomaly_detection"
    BALANCE_HISTORY = "balance_history"
    CASHFLOW_SUMMARY = "cashflow_summary"


class ActionType(str, Enum):
    """Available action types"""
    EXPORT_XLSX = "EXPORT_XLSX"
    EXPORT_CSV = "EXPORT_CSV"
    EXPORT_PDF = "EXPORT_PDF"
    MONTHLY_REPORT = "MONTHLY_REPORT"
    ANNUAL_REPORT = "ANNUAL_REPORT"
    SUBSCRIPTION_REVIEW = "SUBSCRIPTION_REVIEW"
    SET_BUDGET_ALERT = "SET_BUDGET_ALERT"
    CATEGORY_UPDATE = "CATEGORY_UPDATE"
    SET_REMINDER = "SET_REMINDER"
    GENERATE_CHART = "GENERATE_CHART"
    EMAIL_REPORT = "EMAIL_REPORT"


class Direction(str, Enum):
    """Transaction direction"""
    EXPENSE = "expense"
    INCOME = "income"
    TRANSFER = "transfer"
    NEUTRAL = "neutral"
    ALL = "all"


class Period(str, Enum):
    """Time periods"""
    TODAY = "today"
    YESTERDAY = "yesterday"
    THIS_WEEK = "this_week"
    LAST_WEEK = "last_week"
    THIS_MONTH = "this_month"
    LAST_MONTH = "last_month"
    LAST_3_MONTHS = "last_3_months"
    LAST_6_MONTHS = "last_6_months"
    THIS_YEAR = "this_year"
    LAST_YEAR = "last_year"
    CUSTOM = "custom"


# -----------------------------------------------------------------------------
# BASE SCHEMAS
# -----------------------------------------------------------------------------

class StrictModel(BaseModel):
    """Base model with strict validation"""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class DateRange(StrictModel):
    """Date range filter"""
    start: date
    end: date
    
    @field_validator('end')
    @classmethod
    def end_after_start(cls, v, info):
        if 'start' in info.data and v < info.data['start']:
            raise ValueError('end date must be after start date')
        return v


# -----------------------------------------------------------------------------
# ORCHESTRATOR SCHEMAS
# -----------------------------------------------------------------------------

class Constraints(StrictModel):
    """Query constraints extracted by orchestrator"""
    
    # Date handling
    date_range: Optional[DateRange] = None
    implicit_period: Optional[Period] = None
    
    # Filters
    categories: Optional[List[str]] = None
    subcategories: Optional[List[str]] = None
    merchants: Optional[List[str]] = None
    merchant_contains: Optional[str] = None
    direction: Optional[Direction] = None
    
    # Amount range
    min_amount: Optional[float] = None
    max_amount: Optional[float] = None
    
    # Content-based search
    content_keywords: Optional[List[str]] = Field(
        default=None,
        description="Keywords for description/merchant search (e.g., 'kira', 'iban', 'subscription')"
    )
    
    # Other
    currency: str = "TRY"
    limit: int = Field(default=50, ge=1, le=500)
    
    # Comparison
    compare_with_period: Optional[Period] = None


class RouterDecision(StrictModel):
    """
    Orchestrator's routing decision.
    This is THE critical schema for intent classification.
    """
    
    intent: IntentType
    confidence: float = Field(ge=0.0, le=1.0)
    
    # Which agents to invoke
    needs_sql: bool = False
    needs_vector: bool = False
    needs_planner: bool = False
    
    # Risk assessment
    risk_level: RiskLevel = RiskLevel.LOW
    
    # Extracted constraints
    constraints: Constraints = Field(default_factory=Constraints)
    
    # Reasoning
    reasoning: str = Field(description="Brief explanation of routing decision")
    
    # Clarification
    clarification_needed: Optional[str] = None
    suggested_questions: Optional[List[str]] = None


# -----------------------------------------------------------------------------
# FINANCE ANALYST SCHEMAS
# -----------------------------------------------------------------------------

class MetricFilters(StrictModel):
    """Filters for SQL metric queries"""
    
    date_start: Optional[date] = None
    date_end: Optional[date] = None
    categories: Optional[List[str]] = None
    subcategories: Optional[List[str]] = None
    merchants: Optional[List[str]] = None
    merchant_contains: Optional[str] = None
    direction: Optional[Direction] = None
    min_amount: Optional[float] = None
    max_amount: Optional[float] = None
    exclude_transfers: bool = True


class MetricRequest(StrictModel):
    """
    SQL metric parameters.
    bunq Alignment: LLM generates these, deterministic code executes.
    """
    
    metric: MetricType
    filters: MetricFilters = Field(default_factory=MetricFilters)
    group_by: Optional[List[str]] = None
    order_by: Optional[str] = None
    order_direction: Literal["ASC", "DESC"] = "DESC"
    limit: int = Field(default=50, ge=1, le=500)
    
    # For comparison queries
    compare_with: Optional[Period] = None
    
    # Additional context
    user_question: Optional[str] = None


class MetricResult(StrictModel):
    """Result from SQL metric calculation"""

    metric: MetricType
    value: Optional[float] = None
    rows: Optional[List[Dict[str, Any]]] = None
    tx_count: int = 0

    # Metadata
    filters_applied: Dict[str, Any] = Field(default_factory=dict)
    sql_preview: Optional[str] = None
    execution_time_ms: Optional[int] = None

    # Debug info for verbose logging
    sql_params: Optional[List[str]] = None
    sql_row_count: Optional[int] = None
    sql_duration_ms: Optional[int] = None

    # For comparisons
    comparison_value: Optional[float] = None
    change_percent: Optional[float] = None
    change_direction: Optional[Literal["up", "down", "same"]] = None


# -----------------------------------------------------------------------------
# SEARCH AGENT SCHEMAS
# -----------------------------------------------------------------------------

class SearchQuery(StrictModel):
    """Vector search query parameters"""
    
    query_text: str = Field(min_length=1, max_length=500)
    
    # Metadata filters
    filters: Optional[Dict[str, Any]] = None
    
    # Search settings
    top_k: int = Field(default=10, ge=1, le=50)
    alpha: float = Field(
        default=0.70, 
        ge=0.0, 
        le=1.0,
        description="Hybrid: 0=sparse, 1=dense"
    )
    
    # Optional: similar transaction
    similar_to_tx_id: Optional[str] = None
    
    # Reranking
    enable_reranking: bool = False


class SearchExpansion(StrictModel):
    """LLM-based search query expansion"""
    
    expanded_query: str = Field(min_length=1, max_length=800)
    strategy: Literal["exact", "semantic", "hybrid"] = "hybrid"
    alpha: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=1, le=50)
    must_include: Optional[List[str]] = None
    related_terms: Optional[List[str]] = None
    notes: Optional[str] = None


class TransactionMatch(StrictModel):
    """Single transaction match from search"""
    
    tx_id: str
    score: float
    date_time: Optional[datetime] = None
    amount: Optional[float] = None
    merchant_norm: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None


class SearchResult(StrictModel):
    """Vector search result"""
    
    matches: List[TransactionMatch] = Field(default_factory=list)
    total_found: int = 0
    query_text: str = ""
    search_type: Literal["hybrid", "dense", "sparse", "self_rag"] = "hybrid"


# -----------------------------------------------------------------------------
# ACTION PLANNER SCHEMAS
# -----------------------------------------------------------------------------

class ExportParams(StrictModel):
    """Parameters for export actions"""
    format: Literal["xlsx", "csv", "pdf"] = "xlsx"
    include_charts: bool = False
    include_summary: bool = True
    date_start: Optional[date] = None
    date_end: Optional[date] = None
    categories: Optional[List[str]] = None


class ReportParams(StrictModel):
    """Parameters for report generation"""
    report_type: Literal["monthly", "annual", "custom"] = "monthly"
    year: Optional[int] = None
    month: Optional[int] = None
    sections: List[str] = Field(default_factory=lambda: [
        "summary", "categories", "merchants", "trends"
    ])
    include_charts: bool = True


class BudgetAlertParams(StrictModel):
    """Parameters for budget alerts"""
    category: str
    threshold_amount: float
    period: Literal["daily", "weekly", "monthly"] = "monthly"
    notification_type: Literal["email", "push", "both"] = "both"


class CategoryUpdateParams(StrictModel):
    """Parameters for category updates"""
    tx_ids: List[str] = Field(min_length=1)
    new_category: str
    new_subcategory: Optional[str] = None
    apply_to_similar: bool = False


class ReminderParams(StrictModel):
    """Parameters for reminders"""
    reminder_date: datetime
    message: str
    recurring: bool = False
    recurrence_pattern: Optional[Literal["daily", "weekly", "monthly"]] = None


class ActionParams(StrictModel):
    """Union of all action parameters"""
    
    # Common
    date_start: Optional[date] = None
    date_end: Optional[date] = None
    
    # Export
    export_format: Optional[Literal["xlsx", "csv", "pdf"]] = None
    include_charts: Optional[bool] = None
    
    # Report
    report_type: Optional[str] = None
    sections: Optional[List[str]] = None
    
    # Budget
    category: Optional[str] = None
    threshold_amount: Optional[float] = None
    
    # Category update
    tx_ids: Optional[List[str]] = None
    new_category: Optional[str] = None
    new_subcategory: Optional[str] = None
    
    # Reminder
    reminder_date: Optional[datetime] = None
    reminder_message: Optional[str] = None


class DataScope(StrictModel):
    """What data an action will access"""
    
    tables: List[str] = Field(default_factory=lambda: ["transactions"])
    date_range: Optional[DateRange] = None
    estimated_rows: Optional[int] = None
    categories_affected: Optional[List[str]] = None
    read_only: bool = True


class ActionPlan(StrictModel):
    """
    Action execution plan - requires user confirmation.
    bunq Alignment: Plan -> Confirm -> Execute
    """
    
    action_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    action_type: ActionType
    
    # Human-readable plan (English)
    human_plan: str = Field(
        description="User-friendly description",
        min_length=10,
        max_length=1000
    )

    # Optional plan steps
    plan_steps: Optional[List[str]] = None
    
    # Machine-readable params
    params: ActionParams
    
    # Scope and safety
    data_scope: DataScope
    requires_confirmation: bool = True
    risk_level: RiskLevel = RiskLevel.LOW
    
    # Warnings
    warnings: List[str] = Field(default_factory=list)
    
    # Estimation
    estimated_time_seconds: Optional[int] = None


class ActionPlanDraft(StrictModel):
    """Structured draft for action planning"""
    
    action_type: ActionType
    params: ActionParams = Field(default_factory=ActionParams)
    missing_fields: Optional[List[str]] = None
    assumptions: Optional[List[str]] = None
    requires_confirmation: bool = True
    risk_level: RiskLevel = RiskLevel.LOW
    estimated_time_seconds: Optional[int] = None


class ActionResult(StrictModel):
    """Result of action execution"""
    
    action_id: str
    status: Literal["success", "failed", "cancelled", "pending"]
    
    # Output artifacts
    artifacts: Dict[str, str] = Field(default_factory=dict)
    
    # Info
    message: Optional[str] = None
    error: Optional[str] = None
    execution_time_ms: Optional[int] = None


# -----------------------------------------------------------------------------
# RESPONSE SCHEMAS
# -----------------------------------------------------------------------------

class Evidence(StrictModel):
    """Evidence for response traceability"""
    
    # Filters applied
    filters: Dict[str, Any] = Field(default_factory=dict)
    
    # Data stats
    tx_count: Optional[int] = None
    date_range: Optional[Dict[str, str]] = None
    total_amount: Optional[float] = None
    
    # Debug info
    sql_preview: Optional[str] = None
    search_terms: Optional[List[str]] = None
    
    # Agent trace
    agents_used: List[str] = Field(default_factory=list)
    model_used: Optional[str] = None


class FinalResponse(StrictModel):
    """Final response structure"""
    
    answer: str = Field(min_length=1)
    evidence: Evidence = Field(default_factory=Evidence)
    
    # Suggestions
    suggestions: Optional[List[str]] = None
    follow_up_questions: Optional[List[str]] = None
    
    # Action handling
    needs_confirmation: bool = False
    action_plan: Optional[ActionPlan] = None


class ResponseValidation(StrictModel):
    """Validation result for assistant responses"""
    
    is_valid: bool = True
    issues: List[str] = Field(default_factory=list)
    corrected_answer: Optional[str] = None


# -----------------------------------------------------------------------------
# GUARDRAIL SCHEMAS
# -----------------------------------------------------------------------------

class GuardrailResult(StrictModel):
    """Result from guardrail check"""
    
    passed: bool
    reason: Optional[str] = None
    risk_level: RiskLevel = RiskLevel.LOW
    
    blocked: bool = False
    modified_input: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)


class SafetyClassification(StrictModel):
    """LLM-based safety classification"""
    
    is_safe: bool
    category: Literal[
        "prompt_injection",
        "data_extraction",
        "harmful_content",
        "off_topic",
        "safe"
    ] = "safe"
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str


# -----------------------------------------------------------------------------
# CHAT SCHEMAS
# -----------------------------------------------------------------------------

class ChatMessage(StrictModel):
    """Chat message structure"""
    
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Metadata
    tool_calls: Optional[List[Dict[str, Any]]] = None
    evidence: Optional[Evidence] = None


class ChatRequest(StrictModel):
    """API chat request"""
    
    message: str = Field(min_length=1, max_length=4000)
    session_id: Optional[str] = None
    
    # Overrides
    force_intent: Optional[IntentType] = None
    skip_confirmation: bool = False


class ChatResponse(StrictModel):
    """API chat response"""
    
    answer: str
    evidence: Evidence = Field(default_factory=Evidence)
    
    # Action handling
    needs_confirmation: bool = False
    action_id: Optional[str] = None
    action_plan: Optional[ActionPlan] = None
    
    # Session info
    session_id: str
    trace_id: str
    
    # Suggestions
    suggestions: Optional[List[str]] = None


class ActionApprovalRequest(StrictModel):
    """Request to approve/reject an action"""
    
    action_id: str
    approved: bool
    reason: Optional[str] = None


# -----------------------------------------------------------------------------
# AUDIT SCHEMAS
# -----------------------------------------------------------------------------

class ToolCall(StrictModel):
    """Audit record for tool calls"""
    
    trace_id: str
    session_id: str
    node: str
    tool_name: str
    model_name: Optional[str] = None
    
    input_hash: str
    output_hash: str
    
    latency_ms: int
    success: bool
    error_message: Optional[str] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)

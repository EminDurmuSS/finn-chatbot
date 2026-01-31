"""
Statement Copilot - Finance Analyst Agent
=========================================
SQL-based financial analytics.

bunq Alignment: LLM generates parameters, deterministic code executes SQL.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from datetime import date, datetime
from decimal import Decimal

from ..config import settings
from ..core import (
    OrchestratorState,
    MetricRequest,
    MetricFilters,
    MetricResult,
    MetricType,
    Direction,
    create_tool_call_record,
    get_llm_client,
    get_db,
    SQLBuilder,
    get_date_range_from_constraints,
)
from ..log_context import clip_text, format_kv, format_list
from ..core.prompts import get_finance_analyst_prompt

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# FINANCE ANALYST AGENT
# -----------------------------------------------------------------------------

class FinanceAnalystAgent:
    """
    Finance Analyst that:
    1. Converts natural language to MetricRequest (via LLM)
    2. Executes deterministic SQL (via SQLBuilder)
    3. Returns structured MetricResult
    
    bunq Alignment: LLM never calculates, only generates parameters.
    """
    
    def __init__(self):
        self.llm = get_llm_client()
        self.db = get_db()
        self.model = settings.model_sql_generator
    
    def analyze(self, state: OrchestratorState) -> OrchestratorState:
        """
        Run financial analysis based on state.
        
        Args:
            state: Current orchestrator state with constraints
            
        Returns:
            Updated state with sql_result
        """
        user_message = state.get("user_message", "")
        constraints = state.get("constraints", {})
        tenant_id = state.get("tenant_id", settings.default_tenant_id)
        
        start_time = time.time()
        
        try:
            logger.info("")
            logger.info("╔" + "═" * 78 + "╗")
            logger.info("║" + " " * 25 + "FINANCE ANALYST - SQL AGENT" + " " * 25 + "║")
            logger.info("╚" + "═" * 78 + "╝")
            
            # Step 1: Generate MetricRequest from LLM
            logger.info("")
            logger.info("[STEP 1] Generating MetricRequest from LLM...")
            metric_request = self._generate_metric_request(user_message, constraints)

            logger.info("✓ MetricRequest generated:")
            logger.info("  • Metric Type: %s", metric_request.metric.value)
            logger.info("  • Limit: %s", metric_request.limit)
            logger.info("  • Order: %s", metric_request.order_direction)
            logger.info("  • Filters: %s", format_kv(self._filters_to_dict(metric_request.filters), max_items=10, max_value_len=200))
            
            # Step 2: Apply constraints to filters
            logger.info("")
            logger.info("[STEP 2] Applying orchestrator constraints...")
            logger.info("  Constraints received: %s", constraints)
            metric_request = self._apply_constraints(metric_request, constraints)

            logger.info("✓ Filters after constraint application:")
            logger.info("  %s", format_kv(self._filters_to_dict(metric_request.filters), max_items=10, max_value_len=200))
            
            # Step 3: Build and execute SQL (with refinement loop)
            logger.info("")
            logger.info("[STEP 3] Building and executing SQL query...")
            result = self._execute_metric(metric_request, tenant_id)
            refinements: List[str] = []

            if self._result_is_empty(metric_request.metric, result):
                logger.debug("SQL result empty, attempting refinements")
                for _ in range(settings.max_sql_refinements):
                    refined_filters, note = self._refine_filters(metric_request.filters)
                    if not refined_filters:
                        break
                    metric_request.filters = refined_filters
                    result = self._execute_metric(metric_request, tenant_id)
                    if note:
                        refinements.append(note)
                        logger.debug(
                            "SQL refinement applied: %s filters=%s",
                            note,
                            format_kv(self._filters_to_dict(metric_request.filters), max_items=6, max_value_len=120),
                        )
                    if not self._result_is_empty(metric_request.metric, result):
                        break

            # Store request in state (after refinements)
            state["sql_metric_request"] = metric_request.model_dump(mode="json")
            
            latency_ms = int((time.time() - start_time) * 1000)
            result.execution_time_ms = latency_ms
            
            # Store result in state
            state["sql_result"] = result.model_dump(mode="json")
            if refinements:
                state["sql_refinements"] = refinements
            
            # Add tool call record
            tool_call = create_tool_call_record(
                state=state,
                node="finance_analyst",
                tool_name="analyze",
                model_name=self.model,
                input_data={"message": user_message, "constraints": constraints},
                output_data=result.model_dump(mode="json"),
                latency_ms=latency_ms,
                success=True
            )
            state["tool_calls"] = [tool_call]

            logger.info(
                f"Finance Analyst: metric={metric_request.metric.value}, "
                f"tx_count={result.tx_count}, latency={latency_ms}ms"
            )
            
            return state
            
        except Exception as e:
            logger.error(f"Finance Analyst error: {e}")
            state["sql_error"] = str(e)
            state["errors"] = [f"SQL analysis error: {e}"]
            return state
    
    def _generate_metric_request(
        self,
        user_message: str,
        constraints: Dict[str, Any]
    ) -> MetricRequest:
        """
        Use LLM to generate MetricRequest from natural language.
        
        Args:
            user_message: User's question
            constraints: Extracted constraints from orchestrator
            
        Returns:
            MetricRequest with appropriate metric and filters
        """
        # Build prompt with context
        prompt = f"""
## USER QUESTION
{user_message}

## CURRENT CONSTRAINTS
{self._format_constraints(constraints)}

## TASK
Create the appropriate MetricRequest for this question.
- Choose the correct metric type
- Use filters from constraints
- Set limit and ordering
"""
        
        try:
            result = self.llm.complete_structured(
                prompt=prompt,
                response_model=MetricRequest,
                model=self.model,
                system=get_finance_analyst_prompt(),
                temperature=0.0
            )
            return result
            
        except Exception as e:
            logger.error(f"MetricRequest generation failed: {e}")
            # Fallback to sum_amount
            return MetricRequest(
                metric=MetricType.SUM_AMOUNT,
                filters=MetricFilters(),
                user_question=user_message
            )
    
    def _apply_constraints(
        self,
        request: MetricRequest,
        constraints: Dict[str, Any]
    ) -> MetricRequest:
        """
        Apply orchestrator constraints to MetricRequest filters.
        
        Args:
            request: Generated MetricRequest
            constraints: Constraints from orchestrator
            
        Returns:
            MetricRequest with applied constraints
        """
        filters = request.filters
        
        # Apply date range from constraints
        date_start, date_end = get_date_range_from_constraints(
            constraints,
            default_days=settings.default_date_range_days
        )
        
        if not filters.date_start:
            filters.date_start = date_start
        if not filters.date_end:
            filters.date_end = date_end
        
        # Apply categories if not set
        if not filters.categories and constraints.get("categories"):
            filters.categories = constraints["categories"]
        
        # Apply subcategories
        if not filters.subcategories and constraints.get("subcategories"):
            filters.subcategories = constraints["subcategories"]
        
        # Apply merchants - OVERRIDE, not just fallback!
        # Orchestrator's merchant constraint is always applied
        # This fixes the bug where LLM misses the merchant and it's not filtered
        if constraints.get("merchants"):
            existing = filters.merchants or []
            new_merchants = constraints["merchants"]
            # Merge: both LLM-detected and constraint merchants
            merged = list(set(existing + new_merchants))
            filters.merchants = merged
            logger.debug(
                "[PATCH] Merchant override: existing=%s, constraint=%s, merged=%s",
                existing, new_merchants, merged
            )
        
        if not filters.merchant_contains and constraints.get("merchant_contains"):
            filters.merchant_contains = constraints["merchant_contains"]
        
        # Apply direction
        if not filters.direction and constraints.get("direction"):
            direction_str = constraints["direction"]
            if direction_str in [d.value for d in Direction]:
                filters.direction = Direction(direction_str)
        
        # Apply amount range
        if filters.min_amount is None and constraints.get("min_amount") is not None:
            filters.min_amount = constraints["min_amount"]
        
        if filters.max_amount is None and constraints.get("max_amount") is not None:
            filters.max_amount = constraints["max_amount"]
        
        request.filters = filters
        return request
    
    def _execute_metric(
        self,
        request: MetricRequest,
        tenant_id: str
    ) -> MetricResult:
        """
        Execute SQL metric and return result.
        
        Args:
            request: MetricRequest with metric and filters
            tenant_id: Tenant ID for data isolation
            
        Returns:
            MetricResult with calculated values
        """
        # Convert filters to dict for SQLBuilder
        filters_dict = self._filters_to_dict(request.filters)
        
        # Build SQL
        logger.info("")
        logger.info("┌" + "─" * 78 + "┐")
        logger.info("│" + " " * 20 + "SQL QUERY SELECTION & BUILDING" + " " * 27 + "│")
        logger.info("└" + "─" * 78 + "┘")
        logger.info("")
        logger.info("WHY THIS QUERY WAS SELECTED:")
        logger.info("  • Metric Type: %s", request.metric.value)
        logger.info("  • Reason: User query requires %s calculation", request.metric.value.replace('_', ' '))
        logger.info("  • Strategy: SQL-first deterministic calculation")
        logger.info("")
        
        sql, params = SQLBuilder.build(
            metric=request.metric.value,
            filters=filters_dict,
            tenant_id=tenant_id,
            limit=request.limit,
            order_direction=request.order_direction
        )

        logger.info("SQL BUILD SUMMARY:")
        logger.info("  • Metric: %s", request.metric.value)
        logger.info("  • Filters applied: %s", format_kv(filters_dict, max_items=10, max_value_len=150))
        logger.info("  • Limit: %s", request.limit)
        logger.info("  • Order: %s", request.order_direction)
        logger.info("")
        logger.info("GENERATED SQL QUERY:")
        logger.info("%s", sql.strip())
        logger.info("")
        logger.info("QUERY PARAMETERS (%d total):", len(params))
        for i, param in enumerate(params, 1):
            logger.info("  [%d] %s = %s", i, type(param).__name__, param)
        logger.info("")

        # Validate SQL
        if not SQLBuilder.validate_sql(sql):
            raise ValueError("Generated SQL failed validation")
        logger.info("✓ SQL validation passed")
        logger.info("")

        # Execute and track timing
        logger.info("Executing SQL query...")
        sql_start = time.time()
        rows = self.db.execute_query(sql, params, read_only=True)
        sql_duration_ms = int((time.time() - sql_start) * 1000)
        
        logger.info("")
        logger.info("✓ SQL execution completed:")
        logger.info("  • Rows returned: %d", len(rows))
        logger.info("  • Execution time: %d ms", sql_duration_ms)
        logger.info("")
        
        # Parse result based on metric type
        result = self._parse_result(request.metric, rows, filters_dict)

        # Add SQL details for debugging
        result.sql_preview = sql  # Store full SQL for verbose mode
        result.filters_applied = filters_dict
        result.sql_params = [str(p) for p in params]  # Store params as strings
        result.sql_row_count = len(rows)
        result.sql_duration_ms = sql_duration_ms
        
        logger.info("✓ Result parsed successfully")
        logger.info("  • Transaction count: %d", result.tx_count)
        logger.info("  • Result value: %s", result.value)
        logger.info("")

        return result
    
    def _filters_to_dict(self, filters: MetricFilters) -> Dict[str, Any]:
        """Convert MetricFilters to dict for SQLBuilder"""
        data = filters.model_dump(exclude_none=True)
        
        # Convert enums to values
        if "direction" in data and data["direction"]:
            if hasattr(data["direction"], "value"):
                data["direction"] = data["direction"].value
        
        return data

    def _result_is_empty(self, metric: MetricType, result: MetricResult) -> bool:
        """Determine if result is effectively empty."""
        if metric == MetricType.COUNT_TX:
            return (result.value or 0) == 0
        if result.tx_count == 0:
            return True
        if result.rows:
            return False
        if result.value is None:
            return True
        return False

    def _refine_filters(self, filters: MetricFilters) -> tuple[Optional[MetricFilters], Optional[str]]:
        """
        Relax filters when no results are found.
        Returns (refined_filters, note). If no refinement possible, returns (None, None).
        """
        from dateutil.relativedelta import relativedelta

        refined = filters.model_copy(deep=True)

        # 1) Relax merchant_contains by shortening
        if refined.merchant_contains and " " in refined.merchant_contains:
            refined.merchant_contains = refined.merchant_contains.split()[0]
            return refined, "merchant_contains shortened"

        # 2) Remove merchant filters
        if refined.merchants:
            refined.merchants = None
            return refined, "merchant filter removed"
        if refined.merchant_contains:
            refined.merchant_contains = None
            return refined, "merchant_contains filter removed"

        # 3) Remove amount bounds
        if refined.min_amount is not None or refined.max_amount is not None:
            refined.min_amount = None
            refined.max_amount = None
            return refined, "amount range filter removed"

        # 4) Drop subcategories then categories
        if refined.subcategories:
            refined.subcategories = None
            return refined, "subcategory filter removed"
        if refined.categories:
            refined.categories = None
            return refined, "category filter removed"

        # 5) Expand date range
        if refined.date_end or refined.date_start:
            end = refined.date_end or date.today()
            start = end - relativedelta(months=settings.sql_refinement_expand_months)
            if refined.date_start != start or refined.date_end != end:
                refined.date_start = start
                refined.date_end = end
                return refined, f"date range expanded to {settings.sql_refinement_expand_months} months"

        # 6) Relax direction
        if refined.direction and refined.direction != Direction.ALL:
            refined.direction = Direction.ALL
            return refined, "direction filter relaxed"

        # 7) Include transfers
        if refined.exclude_transfers:
            refined.exclude_transfers = False
            return refined, "exclude transfers filter removed"

        return None, None
    
    def _parse_result(
        self,
        metric: MetricType,
        rows: List[Dict[str, Any]],
        filters: Dict[str, Any]
    ) -> MetricResult:
        """
        Parse SQL result into MetricResult.
        
        Args:
            metric: Metric type
            rows: SQL result rows
            filters: Applied filters
            
        Returns:
            Structured MetricResult
        """
        result = MetricResult(
            metric=metric,
            filters_applied=filters
        )
        
        if not rows:
            return result
        
        # Single value metrics
        if metric in [MetricType.SUM_AMOUNT, MetricType.COUNT_TX, MetricType.AVG_AMOUNT, MetricType.MEDIAN_AMOUNT]:
            row = rows[0]
            result.value = row.get("value")
            result.tx_count = row.get("tx_count", 0)
        
        # Min/max
        elif metric == MetricType.MIN_MAX_AMOUNT:
            row = rows[0]
            result.rows = [{
                "min": row.get("min_value"),
                "max": row.get("max_value"),
                "avg": row.get("avg_value"),
            }]
            result.tx_count = row.get("tx_count", 0)
        
        # Top lists and breakdowns
        elif metric in [
            MetricType.TOP_MERCHANTS, MetricType.TOP_CATEGORIES,
            MetricType.CATEGORY_BREAKDOWN, MetricType.MERCHANT_BREAKDOWN,
            MetricType.LARGEST_TRANSACTIONS, MetricType.SMALLEST_TRANSACTIONS
        ]:
            result.rows = rows
            if metric in [MetricType.LARGEST_TRANSACTIONS, MetricType.SMALLEST_TRANSACTIONS]:
                result.tx_count = len(rows)
            else:
                result.tx_count = sum(r.get("tx_count", 0) for r in rows)
        
        # Trends
        elif metric in [MetricType.DAILY_TREND, MetricType.WEEKLY_TREND, MetricType.MONTHLY_TREND]:
            result.rows = self._serialize_rows(rows)
            result.tx_count = sum(r.get("tx_count", 0) for r in rows)
        
        # Comparison
        elif metric == MetricType.MONTHLY_COMPARISON:
            row = rows[0] if rows else {}
            result.value = row.get("current_expense")
            result.comparison_value = row.get("previous_expense")
            
            if result.comparison_value and result.comparison_value > 0:
                change = row.get("expense_change_percent")
                result.change_percent = change
                if change is not None:
                    result.change_direction = "up" if change > 0 else ("down" if change < 0 else "same")
            
            result.rows = [row]
            result.tx_count = row.get("current_tx_count", 0)
        
        # Cashflow
        elif metric == MetricType.CASHFLOW_SUMMARY:
            row = rows[0] if rows else {}
            result.rows = [row]
            result.value = row.get("net_flow")
            result.tx_count = row.get("tx_count", 0)
        
        # Subscriptions and recurring
        elif metric in [MetricType.SUBSCRIPTION_LIST, MetricType.RECURRING_PAYMENTS]:
            result.rows = rows
            result.tx_count = len(rows)
        
        # Anomaly detection
        elif metric == MetricType.ANOMALY_DETECTION:
            result.rows = self._serialize_rows(rows)
            result.tx_count = len(rows)
        
        # Balance history
        elif metric == MetricType.BALANCE_HISTORY:
            result.rows = self._serialize_rows(rows)
            result.tx_count = len(rows)
        
        else:
            # Default: return all rows
            result.rows = self._serialize_rows(rows)
            result.tx_count = len(rows)
        
        return result
    
    def _serialize_rows(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Serialize rows, converting non-JSON-serializable types"""
        serialized = []
        for row in rows:
            new_row = {}
            for key, value in row.items():
                if isinstance(value, (datetime, date)):
                    new_row[key] = value.isoformat()
                elif isinstance(value, bool):
                    new_row[key] = value
                elif isinstance(value, Decimal):
                    new_row[key] = float(value)
                elif isinstance(value, (int, float)):
                    new_row[key] = value
                elif hasattr(value, '__float__'):
                    new_row[key] = float(value)
                else:
                    new_row[key] = value
            serialized.append(new_row)
        return serialized
    
    def _format_constraints(self, constraints: Dict[str, Any]) -> str:
        """Format constraints for prompt"""
        if not constraints:
            return "None"
        
        parts = []
        for key, value in constraints.items():
            if value is not None:
                parts.append(f"- {key}: {value}")
        
        return "\n".join(parts) if parts else "Yok"


# -----------------------------------------------------------------------------
# QUICK ANALYTICS (Shortcut functions)
# -----------------------------------------------------------------------------

def quick_sum(
    tenant_id: str,
    date_start: Optional[date] = None,
    date_end: Optional[date] = None,
    categories: Optional[List[str]] = None,
    direction: str = "expense"
) -> float:
    """Quick sum calculation without LLM"""
    db = get_db()
    
    filters = {
        "date_start": date_start,
        "date_end": date_end,
        "categories": categories,
        "direction": direction,
        "exclude_transfers": True,
    }
    
    sql, params = SQLBuilder.build(
        metric="sum_amount",
        filters=filters,
        tenant_id=tenant_id
    )
    
    rows = db.execute_query(sql, params)
    if rows and rows[0].get("value"):
        return float(rows[0]["value"])
    return 0.0


def quick_category_breakdown(
    tenant_id: str,
    date_start: Optional[date] = None,
    date_end: Optional[date] = None,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """Quick category breakdown without LLM"""
    db = get_db()
    
    filters = {
        "date_start": date_start,
        "date_end": date_end,
        "direction": "expense",
        "exclude_transfers": True,
    }
    
    sql, params = SQLBuilder.build(
        metric="category_breakdown",
        filters=filters,
        tenant_id=tenant_id,
        limit=limit
    )
    
    return db.execute_query(sql, params)


# -----------------------------------------------------------------------------
# SINGLETON
# -----------------------------------------------------------------------------

_finance_analyst: Optional[FinanceAnalystAgent] = None


def get_finance_analyst() -> FinanceAnalystAgent:
    """Get or create finance analyst singleton"""
    global _finance_analyst
    if _finance_analyst is None:
        _finance_analyst = FinanceAnalystAgent()
    return _finance_analyst

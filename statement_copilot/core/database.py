"""
Statement Copilot - Database Manager
====================================
DuckDB connection management and SQL building.
SQL-first truth layer - all calculations happen here.

bunq Alignment: LLM never calculates, it only produces parameters.
"""

import duckdb
import logging
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, List, Dict, Any, Generator, Tuple
from datetime import date, datetime

from ..config import settings
from ..log_context import clip_text, format_list

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# DATABASE MANAGER
# -----------------------------------------------------------------------------

class DatabaseManager:
    """
    DuckDB connection manager with thread-safe operations.
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or settings.get_db_path()
        self._ensure_db_exists()
    
    def _ensure_db_exists(self):
        """Ensure database exists, raise error if not"""
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Database not found at {self.db_path}. "
                f"Please run 'python scripts/setup/duckdb_setup.py' first to create the database."
            )
    
    @contextmanager
    def connection(self, read_only: bool = True) -> Generator[duckdb.DuckDBPyConnection, None, None]:
        """Context manager for database connections"""
        con = duckdb.connect(str(self.db_path), read_only=read_only)
        try:
            yield con
            if not read_only:
                con.commit()
        except Exception as e:
            if not read_only:
                con.rollback()
            raise
        finally:
            con.close()
    
    def execute_query(
        self,
        sql: str,
        params: Optional[List[Any]] = None,
        read_only: bool = True
    ) -> List[Dict[str, Any]]:
        """Execute query and return results as list of dicts"""
        logger.info("")
        logger.info("═" * 80)
        logger.info("[DATABASE EXECUTION]")
        logger.info("Read-only: %s", read_only)
        logger.info("─" * 80)
        logger.info("SQL QUERY:")
        logger.info("%s", sql.strip())
        if params:
            logger.info("─" * 80)
            logger.info("PARAMETERS (%d total):", len(params))
            for i, param in enumerate(params, 1):
                logger.info("  [%d] %s", i, param)
        logger.info("═" * 80)
        logger.info("")
        
        with self.connection(read_only=read_only) as con:
            if params:
                result = con.execute(sql, params)
            else:
                result = con.execute(sql)
            
            columns = [desc[0] for desc in result.description]
            rows = result.fetchall()
            
            result_count = len(rows)
            logger.info("[DATABASE RESULT] Returned %d rows", result_count)
            
            return [dict(zip(columns, row)) for row in rows]
    
    def execute_scalar(
        self,
        sql: str,
        params: Optional[List[Any]] = None
    ) -> Any:
        """Execute query and return single scalar value"""
        logger.debug(
            "DB scalar query: sql=%s params=%s",
            clip_text(sql, 300),
            format_list(params or [], max_items=6, max_value_len=80),
        )
        with self.connection(read_only=True) as con:
            if params:
                result = con.execute(sql, params).fetchone()
            else:
                result = con.execute(sql).fetchone()
            return result[0] if result else None
    
    def get_transaction_by_ids(
        self,
        tx_ids: List[str],
        tenant_id: str
    ) -> List[Dict[str, Any]]:
        """Get full transaction details by IDs"""
        if not tx_ids:
            return []
        
        placeholders = ",".join(["?" for _ in tx_ids])
        sql = f"""
            SELECT * FROM transactions
            WHERE tx_id IN ({placeholders})
            AND tenant_id = ?
            ORDER BY date_time DESC
        """
        params = tx_ids + [tenant_id]
        
        return self.execute_query(sql, params)


# -----------------------------------------------------------------------------
# SQL BUILDER
# -----------------------------------------------------------------------------

class SQLBuilder:
    """
    Build safe SQL queries from structured parameters.
    
    bunq Alignment: LLM generates params, this code builds safe SQL.
    """
    
    ALLOWED_COLUMNS = {
        "tx_id", "file_id", "tenant_id", "user_id",
        "date_time", "value_date", "amount", "balance",
        "direction", "description", "merchant_norm", "merchant_key",
        "category", "subcategory", "category_final", "subcategory_final",
        "channel", "transaction_type", "month_year", "currency",
        # Added for compatibility
        "day_of_week", "hour_of_day", "tags_arr", "reference", 
        "confidence", "source",
        # Full schema compatibility
        "merchant_raw", "tags", "tags_final", "taxonomy_version",
        "category_updated_at", "transaction_code", "reasoning",
        "overdraft_balance", "source_file"
    }
    
    ALLOWED_GROUP_BY = {
        "category", "subcategory", "category_final", "subcategory_final",
        "merchant_norm", "merchant_key", "direction", "channel", 
        "month_year", "currency", "day_of_week", "hour_of_day"
    }
    
    METRIC_TEMPLATES = {
        "sum_amount": """
            SELECT 
                SUM(amount) as value,
                COUNT(*) as tx_count,
                MIN(date_time) as date_start,
                MAX(date_time) as date_end
            FROM transactions
            {where}
        """,
        
        "count_tx": """
            SELECT COUNT(*) as value
            FROM transactions
            {where}
        """,
        
        "avg_amount": """
            SELECT 
                AVG(amount) as value,
                COUNT(*) as tx_count,
                MIN(amount) as min_amount,
                MAX(amount) as max_amount
            FROM transactions
            {where}
        """,

        "median_amount": """
            SELECT 
                MEDIAN(amount) as value,
                COUNT(*) as tx_count
            FROM transactions
            {where}
        """,
        
        "min_max_amount": """
            SELECT 
                MIN(amount) as min_value,
                MAX(amount) as max_value,
                AVG(amount) as avg_value,
                COUNT(*) as tx_count
            FROM transactions
            {where}
        """,
        
        "top_merchants": """
            SELECT 
                merchant_norm,
                SUM(ABS(amount)) as total,
                COUNT(*) as tx_count,
                AVG(amount) as avg_amount,
                MIN(date_time) as first_tx,
                MAX(date_time) as last_tx
            FROM transactions
            {where}
            AND merchant_norm IS NOT NULL
            AND merchant_norm != ''
            GROUP BY merchant_norm
            ORDER BY total {order_direction}
            LIMIT {limit}
        """,
        
        "top_categories": """
            SELECT 
                COALESCE(category_final, category, 'Diğer') as category,
                SUM(ABS(amount)) as total,
                COUNT(*) as tx_count
            FROM transactions
            {where}
            GROUP BY COALESCE(category_final, category, 'Diğer')
            ORDER BY total {order_direction}
            LIMIT {limit}
        """,
        
        "category_breakdown": """
            SELECT 
                COALESCE(category_final, category, 'Diğer') as category,
                COALESCE(subcategory_final, subcategory, 'Diğer') as subcategory,
                SUM(amount) as total,
                COUNT(*) as tx_count,
                SUM(CASE WHEN direction = 'expense' THEN ABS(amount) ELSE 0 END) as expense_total,
                SUM(CASE WHEN direction = 'income' THEN amount ELSE 0 END) as income_total
            FROM transactions
            {where}
            GROUP BY 
                COALESCE(category_final, category, 'Diğer'),
                COALESCE(subcategory_final, subcategory, 'Diğer')
            ORDER BY ABS(total) DESC
            LIMIT {limit}
        """,
        
        "merchant_breakdown": """
            SELECT 
                merchant_norm,
                COALESCE(category_final, category, 'Diğer') as category,
                SUM(amount) as total,
                COUNT(*) as tx_count,
                AVG(amount) as avg_amount
            FROM transactions
            {where}
            AND merchant_norm IS NOT NULL
            GROUP BY merchant_norm, COALESCE(category_final, category, 'Diğer')
            ORDER BY ABS(total) DESC
            LIMIT {limit}
        """,

        "largest_transactions": """
            SELECT 
                tx_id,
                date_time,
                amount,
                merchant_norm,
                description,
                COALESCE(category_final, category, 'Diğer') as category,
                direction
            FROM transactions
            {where}
            ORDER BY ABS(amount) DESC
            LIMIT {limit}
        """,

        "smallest_transactions": """
            SELECT 
                tx_id,
                date_time,
                amount,
                merchant_norm,
                description,
                COALESCE(category_final, category, 'Diğer') as category,
                direction
            FROM transactions
            {where}
            ORDER BY ABS(amount) ASC
            LIMIT {limit}
        """,
        
        "daily_trend": """
            SELECT 
                CAST(date_time AS DATE) as date,
                SUM(amount) as total,
                SUM(CASE WHEN direction = 'expense' THEN ABS(amount) ELSE 0 END) as expense,
                SUM(CASE WHEN direction = 'income' THEN amount ELSE 0 END) as income,
                COUNT(*) as tx_count
            FROM transactions
            {where}
            GROUP BY CAST(date_time AS DATE)
            ORDER BY date {order_direction}
            LIMIT {limit}
        """,
        
        "weekly_trend": """
            SELECT 
                DATE_TRUNC('week', date_time) as week_start,
                SUM(amount) as total,
                SUM(CASE WHEN direction = 'expense' THEN ABS(amount) ELSE 0 END) as expense,
                SUM(CASE WHEN direction = 'income' THEN amount ELSE 0 END) as income,
                COUNT(*) as tx_count
            FROM transactions
            {where}
            GROUP BY DATE_TRUNC('week', date_time)
            ORDER BY week_start {order_direction}
            LIMIT {limit}
        """,
        
        "monthly_trend": """
            SELECT 
                month_year,
                SUM(amount) as total,
                SUM(CASE WHEN direction = 'expense' THEN ABS(amount) ELSE 0 END) as expense,
                SUM(CASE WHEN direction = 'income' THEN amount ELSE 0 END) as income,
                COUNT(*) as tx_count
            FROM transactions
            {where}
            GROUP BY month_year
            ORDER BY month_year {order_direction}
            LIMIT {limit}
        """,
        
        "monthly_comparison": """
            WITH current_period AS (
                SELECT 
                    SUM(amount) as total,
                    SUM(CASE WHEN direction = 'expense' THEN ABS(amount) ELSE 0 END) as expense,
                    SUM(CASE WHEN direction = 'income' THEN amount ELSE 0 END) as income,
                    COUNT(*) as tx_count
                FROM transactions
                {where}
            ),
            previous_period AS (
                SELECT 
                    SUM(amount) as total,
                    SUM(CASE WHEN direction = 'expense' THEN ABS(amount) ELSE 0 END) as expense,
                    SUM(CASE WHEN direction = 'income' THEN amount ELSE 0 END) as income,
                    COUNT(*) as tx_count
                FROM transactions
                {where_previous}
            )
            SELECT 
                c.total as current_total,
                c.expense as current_expense,
                c.income as current_income,
                c.tx_count as current_tx_count,
                p.total as previous_total,
                p.expense as previous_expense,
                p.income as previous_income,
                p.tx_count as previous_tx_count,
                CASE WHEN p.expense > 0 THEN 
                    ROUND(((c.expense - p.expense) / p.expense) * 100, 2)
                ELSE NULL END as expense_change_percent,
                CASE WHEN p.income > 0 THEN 
                    ROUND(((c.income - p.income) / p.income) * 100, 2)
                ELSE NULL END as income_change_percent
            FROM current_period c, previous_period p
        """,

        "year_over_year": """
            WITH current_period AS (
                SELECT 
                    SUM(amount) as total,
                    SUM(CASE WHEN direction = 'expense' THEN ABS(amount) ELSE 0 END) as expense,
                    SUM(CASE WHEN direction = 'income' THEN amount ELSE 0 END) as income,
                    COUNT(*) as tx_count
                FROM transactions
                {where}
            ),
            previous_period AS (
                SELECT 
                    SUM(amount) as total,
                    SUM(CASE WHEN direction = 'expense' THEN ABS(amount) ELSE 0 END) as expense,
                    SUM(CASE WHEN direction = 'income' THEN amount ELSE 0 END) as income,
                    COUNT(*) as tx_count
                FROM transactions
                {where_previous}
            )
            SELECT 
                c.total as current_total,
                c.expense as current_expense,
                c.income as current_income,
                c.tx_count as current_tx_count,
                p.total as previous_total,
                p.expense as previous_expense,
                p.income as previous_income,
                p.tx_count as previous_tx_count,
                CASE WHEN p.expense > 0 THEN 
                    ROUND(((c.expense - p.expense) / p.expense) * 100, 2)
                ELSE NULL END as expense_change_percent,
                CASE WHEN p.income > 0 THEN 
                    ROUND(((c.income - p.income) / p.income) * 100, 2)
                ELSE NULL END as income_change_percent
            FROM current_period c, previous_period p
        """,
        
        "subscription_list": """
            SELECT
                merchant_norm,
                amount_median,
                period_days,
                last_seen,
                next_estimate,
                confidence,
                status
            FROM subscriptions
            WHERE tenant_id = ?
            AND status = 'active'
            ORDER BY ABS(amount_median) DESC
            LIMIT {limit}
        """,
        
        "recurring_payments": """
            SELECT 
                merchant_norm,
                COUNT(*) as occurrence_count,
                AVG(ABS(amount)) as avg_amount,
                STDDEV(ABS(amount)) as std_amount,
                MIN(date_time) as first_tx,
                MAX(date_time) as last_tx
            FROM transactions
            {where}
            AND merchant_norm IS NOT NULL
            GROUP BY merchant_norm
            HAVING COUNT(*) >= 3
            ORDER BY occurrence_count DESC
            LIMIT {limit}
        """,
        
        "anomaly_detection": """
            WITH stats AS (
                SELECT 
                    AVG(ABS(amount)) as avg_amount,
                    STDDEV(ABS(amount)) as std_amount
                FROM transactions
                {where}
            )
            SELECT 
                t.tx_id,
                t.date_time,
                t.amount,
                t.merchant_norm,
                t.category,
                t.description,
                (ABS(t.amount) - s.avg_amount) / NULLIF(s.std_amount, 0) as z_score
            FROM transactions t, stats s
            {where}
            AND ABS((ABS(t.amount) - s.avg_amount) / NULLIF(s.std_amount, 0)) > 2.5
            ORDER BY ABS(t.amount) DESC
            LIMIT {limit}
        """,
        
        "balance_history": """
            SELECT 
                CAST(date_time AS DATE) as date,
                balance,
                amount,
                merchant_norm,
                direction
            FROM transactions
            {where}
            AND balance IS NOT NULL
            AND balance IS NOT NULL
            ORDER BY {order_by} {order_direction}
            LIMIT {limit}
        """,
        
        "cashflow_summary": """
            SELECT 
                SUM(CASE WHEN direction = 'income' THEN amount ELSE 0 END) as total_income,
                SUM(CASE WHEN direction = 'expense' THEN ABS(amount) ELSE 0 END) as total_expense,
                SUM(CASE WHEN direction = 'transfer' THEN ABS(amount) ELSE 0 END) as total_transfer,
                SUM(amount) as net_flow,
                COUNT(*) as tx_count,
                COUNT(DISTINCT CAST(date_time AS DATE)) as active_days
            FROM transactions
            {where}
        """,

        "generic_trend": """
            SELECT 
                {date_col} as date_group,
                SUM(amount) as total,
                SUM(CASE WHEN direction = 'expense' THEN ABS(amount) ELSE 0 END) as expense,
                SUM(CASE WHEN direction = 'income' THEN amount ELSE 0 END) as income,
                COUNT(*) as tx_count
            FROM transactions
            {where}
            GROUP BY 1
            ORDER BY 1 {order_direction}
            LIMIT {limit}
        """,

        "refund_analysis": """
            SELECT 
                merchant_norm,
                SUM(CASE WHEN direction = 'expense' THEN ABS(amount) ELSE 0 END) as spend_total,
                SUM(CASE WHEN direction = 'income' AND (category LIKE '%refund%' OR description LIKE '%refund%' OR category LIKE '%iade%') THEN amount ELSE 0 END) as refund_total,
                CASE WHEN SUM(CASE WHEN direction = 'expense' THEN ABS(amount) ELSE 0 END) > 0 THEN
                   (SUM(CASE WHEN direction = 'income' AND (category LIKE '%refund%' OR description LIKE '%refund%' OR category LIKE '%iade%') THEN amount ELSE 0 END) / 
                    SUM(CASE WHEN direction = 'expense' THEN ABS(amount) ELSE 0 END)) * 100
                ELSE 0 END as refund_rate,
                COUNT(*) as tx_count
            FROM transactions
            {where}
            GROUP BY merchant_norm
            HAVING refund_total > 0
            ORDER BY refund_total DESC
            LIMIT {limit}
        """,

        "settlement_lag": """
            SELECT 
                tx_id,
                date_time,
                value_date,
                DATEDIFF('day', value_date, CAST(date_time AS DATE)) as lag_days,
                amount,
                merchant_norm,
                description,
                channel
            FROM transactions
            {where}
            AND ABS(DATEDIFF('day', value_date, CAST(date_time AS DATE))) >= 2
            ORDER BY ABS(lag_days) DESC
            LIMIT {limit}
        """,

        "auth_hold_reconciliation": """
             SELECT 
                COUNT(*) as hold_count,
                SUM(amount) as hold_total,
                SUM(CASE WHEN description LIKE '%AUTH%' THEN 1 ELSE 0 END) as auth_count
             FROM transactions
             {where}
             AND (description LIKE '%AUTH%' OR category = 'authorization_hold')
        """,

        "daily_spike_detection": """
             WITH daily_spend AS (
                 SELECT 
                    CAST(date_time AS DATE) as date,
                    SUM(ABS(amount)) as daily_total
                 FROM transactions
                 {where}
                 AND direction = 'expense'
                 GROUP BY CAST(date_time AS DATE)
             ),
             stats AS (
                 SELECT MEDIAN(daily_total) as median_daily FROM daily_spend
             )
             SELECT 
                 d.date,
                 d.daily_total,
                 s.median_daily,
                 (d.daily_total / NULLIF(s.median_daily, 0)) as multiple
             FROM daily_spend d, stats s
             WHERE d.daily_total > (s.median_daily * 3)
             ORDER BY d.daily_total DESC
             LIMIT {limit}
        """,

        "channel_breakdown": """
            SELECT 
                channel,
                COUNT(*) as tx_count,
                SUM(ABS(amount)) as total_spend,
                AVG(ABS(amount)) as avg_spend
            FROM transactions
            {where}
            GROUP BY channel
            ORDER BY total_spend DESC
            LIMIT {limit}
        """,

        "p2p_flow": """
            SELECT 
                CASE 
                    WHEN merchant_norm IN ('P2P_TRANSFER', 'INTERNAL_TRANSFER') THEN 
                        UPPER(TRIM(split_part(description, '*', 1)))
                    ELSE merchant_norm 
                END as counterparty,
                SUM(CASE WHEN direction = 'transfer' AND amount < 0 THEN ABS(amount) ELSE 0 END) as sent_total,
                SUM(CASE WHEN direction = 'transfer' AND amount > 0 THEN amount ELSE 0 END) as received_total,
                SUM(amount) as net_flow,
                COUNT(*) as tx_count
            FROM transactions
            {where}
            AND (
                subcategory IN ('p2p_sent', 'p2p_received', 'internal_transfer')
                OR (merchant_norm IN ('P2P_TRANSFER', 'INTERNAL_TRANSFER'))
            )
            GROUP BY 1
            ORDER BY ABS(net_flow) DESC
            LIMIT {limit}
        """,

        "fx_analysis": """
            SELECT 
                merchant_norm,
                COUNT(*) as tx_count,
                SUM(ABS(amount)) as total_amount
            FROM transactions
            {where}
            AND (category LIKE '%currency%' OR category LIKE '%fx%' OR description LIKE '%exchange%' OR description LIKE '%döviz%')
            GROUP BY merchant_norm
            ORDER BY total_amount DESC
            LIMIT {limit}
        """,
        
        "ledger_reconciliation": """
            SELECT 
                strftime(date_time, '%Y-%m-%d %H:%M:%S') as ts_group,
                COUNT(*) as tx_count,
                SUM(amount) as net_amount_change,
                MAX(balance) - MIN(balance) as balance_diff
            FROM transactions
            {where}
            GROUP BY ts_group
            HAVING COUNT(*) > 1
            LIMIT {limit}
        """,

        "low_confidence_audit": """
            SELECT 
                 tx_id,
                 merchant_norm,
                 description,
                 amount,
                 confidence,
                 category
            FROM transactions
            {where}
            AND confidence < 0.60
            ORDER BY confidence ASC
            LIMIT {limit}
        """,

        "business_spend": """
            SELECT 
                merchant_norm,
                COALESCE(category_final, category) as category,
                SUM(ABS(amount)) as total_spend
            FROM transactions
            {where}
            AND (category LIKE '%business%' OR category LIKE '%software%' OR category LIKE '%office%')
            GROUP BY merchant_norm, COALESCE(category_final, category)
            ORDER BY total_spend DESC
            LIMIT {limit}
        """,

        "atm_analysis": """
             SELECT 
                CAST(date_time AS DATE) as date,
                COUNT(*) as tx_count,
                SUM(ABS(amount)) as total_withdrawn
             FROM transactions
             {where}
             AND (category LIKE '%atm%' OR merchant_norm LIKE '%ATM%')
             GROUP BY CAST(date_time AS DATE)
             ORDER BY total_withdrawn DESC
             LIMIT {limit}
        """
    }
    
    @classmethod
    def build(
        cls,
        metric: str,
        filters: Dict[str, Any],
        tenant_id: str,
        limit: int = 50,
        order_direction: str = "DESC",
        time_grain: Optional[str] = None,
        order_by: Optional[str] = None
    ) -> Tuple[str, List[Any]]:
        """
        Build safe SQL from metric type and filters.
        
        Returns:
            Tuple of (sql, params)
        """
        if metric not in cls.METRIC_TEMPLATES:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Validate order direction
        order_direction = order_direction.upper()
        if order_direction not in ("ASC", "DESC"):
            order_direction = "DESC"
        
        # Validate limit
        limit = min(max(1, limit), 3000)
        
        # Build WHERE clause
        where_parts = ["WHERE tenant_id = ?"]
        params: List[Any] = [tenant_id]
        
        # Date range
        if filters.get("date_start"):
            where_parts.append("AND date_time >= ?")
            params.append(filters["date_start"])
        
        if filters.get("date_end"):
            where_parts.append("AND date_time <= ?")
            # Add time to include full day
            date_end = filters["date_end"]
            if isinstance(date_end, date) and not isinstance(date_end, datetime):
                date_end = datetime.combine(date_end, datetime.max.time())
            params.append(date_end)
        
        # Categories - normalize for fuzzy matching
        if filters.get("categories"):
            cats = [c for c in filters["categories"] if c]
            if cats:
                cat_conditions = []
                for cat in cats:
                    # Normalize: lowercase, replace spaces/special chars with underscore
                    cat_lower = cat.lower().strip()
                    cat_normalized = (
                        cat_lower
                        .replace(" & ", "_and_")
                        .replace("&", "_and_")
                        .replace(" ", "_")
                        .replace("-", "_")
                    )
                    # Remove trailing 's' for singular/plural match
                    cat_singular = cat_normalized.rstrip("s") if cat_normalized.endswith("s") and len(cat_normalized) > 3 else cat_normalized
                    
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
                        f"%{cat_lower}%",
                        f"%{cat_normalized}%",
                        f"%{cat_singular}%",
                        f"%{cat_lower}%",
                        f"%{cat_normalized}%",
                        f"%{cat_singular}%",
                    ])
                where_parts.append(f"AND ({' OR '.join(cat_conditions)})")
        
        # Subcategories - normalize for fuzzy matching
        if filters.get("subcategories"):
            subcats = [s for s in filters["subcategories"] if s]
            if subcats:
                subcat_conditions = []
                for subcat in subcats:
                    subcat_lower = subcat.lower().strip()
                    subcat_normalized = (
                        subcat_lower
                        .replace(" & ", "_and_")
                        .replace("&", "_and_")
                        .replace(" ", "_")
                        .replace("-", "_")
                    )
                    subcat_singular = subcat_normalized.rstrip("s") if subcat_normalized.endswith("s") and len(subcat_normalized) > 3 else subcat_normalized
                    
                    subcat_conditions.append(
                        "("
                        "LOWER(COALESCE(subcategory_final, subcategory)) LIKE ? OR "
                        "LOWER(COALESCE(subcategory_final, subcategory)) LIKE ? OR "
                        "LOWER(COALESCE(subcategory_final, subcategory)) LIKE ?"
                        ")"
                    )
                    params.extend([
                        f"%{subcat_lower}%",
                        f"%{subcat_normalized}%",
                        f"%{subcat_singular}%",
                    ])
                where_parts.append(f"AND ({' OR '.join(subcat_conditions)})")
        
        # Merchants - FUZZY MATCHING with LIKE pattern
        # Fixes: "YouTube" matches "YOUTUBE PREMIUM", "Google YouTubePremium", etc.
        if filters.get("merchants"):
            merchants = [m for m in filters["merchants"] if m]
            if merchants:
                merchant_conditions = []
                for merchant in merchants:
                    # Normalize merchant name for matching
                    m_upper = merchant.upper().strip()
                    m_lower = merchant.lower().strip()
                    
                    # Match against:
                    # 1. merchant_norm (case-insensitive, partial match)
                    # 2. description (case-insensitive, partial match)
                    # 3. Handle multi-word merchants with % wildcards
                    merchant_conditions.append(
                        "("
                        "UPPER(merchant_norm) LIKE ? OR "
                        "UPPER(merchant_norm) LIKE ? OR "
                        "LOWER(description) LIKE ? OR "
                        "LOWER(description) LIKE ?"
                        ")"
                    )
                    params.extend([
                        f"%{m_upper}%",                    # YOUTUBE matches YOUTUBE PREMIUM
                        f"%{m_upper.replace(' ', '%')}%", # YOUTUBE PREMIUM matches variations
                        f"%{m_lower}%",                   # youtube matches Google YouTubePremium
                        f"%{m_lower.replace(' ', '%')}%",
                    ])
                
                where_parts.append(f"AND ({' OR '.join(merchant_conditions)})")
                
                logger.debug(
                    "[PATCH] Fuzzy merchant filter applied: %s -> %d conditions",
                    merchants, len(merchant_conditions)
                )
        
        if filters.get("merchant_contains"):
            # Search in both merchant_norm AND description (for P2P transfers)
            where_parts.append("AND (UPPER(merchant_norm) LIKE ? OR UPPER(description) LIKE ?)")
            params.append(f"%{filters['merchant_contains'].upper()}%")
            params.append(f"%{filters['merchant_contains'].upper()}%")
        
        # Keyword Search - Search across both merchant_norm AND description
        # Enables queries like "Ev kirası" or "IBAN" to match in either field
        if filters.get("keyword_search"):
            keywords = filters["keyword_search"] if isinstance(filters["keyword_search"], list) else [filters["keyword_search"]]
            keywords = [k for k in keywords if k]
            if keywords:
                keyword_conditions = []
                for keyword in keywords:
                    k_upper = keyword.upper().strip()
                    k_lower = keyword.lower().strip()
                    
                    # Match in merchant_norm OR description
                    keyword_conditions.append(
                        "("
                        "UPPER(merchant_norm) LIKE ? OR "
                        "LOWER(description) LIKE ?"
                        ")"
                    )
                    params.extend([
                        f"%{k_upper}%",
                        f"%{k_lower}%",
                    ])
                
                # All keywords must match (AND between keywords, OR within each keyword's fields)
                where_parts.append(f"AND ({' AND '.join(keyword_conditions)})")
                
                logger.debug(
                    "[NEW] Keyword search filter applied: %s -> %d keyword(s)",
                    keywords, len(keywords)
                )
        
        # Direction
        if filters.get("direction") and filters["direction"] != "all":
            where_parts.append("AND direction = ?")
            params.append(filters["direction"])
        
        # Exclude transfers (default)
        if filters.get("exclude_transfers", True):
            where_parts.append("AND direction != 'transfer'")
        
        # Amount range
        if filters.get("min_amount") is not None:
            where_parts.append("AND ABS(amount) >= ?")
            params.append(abs(filters["min_amount"]))
        
        if filters.get("max_amount") is not None:
            where_parts.append("AND ABS(amount) <= ?")
            params.append(abs(filters["max_amount"]))
        
        where_clause = " ".join(where_parts)
        
        # Handle time_grain override (Dynamic Trend)
        # CRITICAL: Only apply override for scalar metrics. 
        # Specialized metrics (subscription_list, breakdowns, etc.) MUST NOT be converted to generic trends.
        TRENDABLE_METRICS = {
            "sum_amount", "count_tx", "avg_amount", "median_amount", "min_max_amount",
            "daily_trend", "weekly_trend", "monthly_trend", "cashflow_summary"
        }
        
        if time_grain and metric == "category_breakdown":
            grains = {
                "day": "CAST(date_time AS DATE)",
                "week": "DATE_TRUNC('week', date_time)",
                "month": "DATE_TRUNC('month', date_time)",
                "quarter": "DATE_TRUNC('quarter', date_time)",
                "year": "DATE_TRUNC('year', date_time)",
            }
            # Default to month if unknown
            date_col = grains.get(str(time_grain).lower(), "DATE_TRUNC('month', date_time)")
            
            # [PATCH] Trended Category Breakdown
            # Groups by BOTH time and category
            sql = f"""
                SELECT 
                    {date_col} as date_group,
                    COALESCE(category_final, category, 'Diğer') as category,
                    COALESCE(subcategory_final, subcategory, 'Diğer') as subcategory,
                    SUM(amount) as total,
                    COUNT(*) as tx_count,
                    SUM(CASE WHEN direction = 'expense' THEN ABS(amount) ELSE 0 END) as expense_total,
                    SUM(CASE WHEN direction = 'income' THEN amount ELSE 0 END) as income_total
                FROM transactions
                {where_clause}
                GROUP BY 
                    1,
                    COALESCE(category_final, category, 'Diğer'),
                    COALESCE(subcategory_final, subcategory, 'Diğer')
                ORDER BY 1 DESC, ABS(total) DESC
                LIMIT {limit}
            """
            return sql, params

        if time_grain and metric in TRENDABLE_METRICS:
            grains = {
                "day": "CAST(date_time AS DATE)",
                "week": "DATE_TRUNC('week', date_time)",
                "month": "DATE_TRUNC('month', date_time)",
                "quarter": "DATE_TRUNC('quarter', date_time)",
                "year": "DATE_TRUNC('year', date_time)",
            }
            # Default to month if unknown
            date_col = grains.get(str(time_grain).lower(), "DATE_TRUNC('month', date_time)")
            
            sql = cls.METRIC_TEMPLATES["generic_trend"].format(
                date_col=date_col,
                where=where_clause,
                limit=limit,
                order_direction=order_direction
            )
            return sql, params

        # Handle monthly comparison (needs two WHERE clauses)
        if metric == "monthly_comparison":
            from dateutil.relativedelta import relativedelta
            
            date_start = filters.get("date_start")
            date_end = filters.get("date_end")
            
            if date_start and date_end:
                if isinstance(date_start, datetime):
                    date_start = date_start.date()
                if isinstance(date_end, datetime):
                    date_end = date_end.date()
                    
                prev_start = date_start - relativedelta(months=1)
                prev_end = date_end - relativedelta(months=1)
            else:
                today = date.today()
                date_start = today.replace(day=1)
                date_end = today
                # MTD Logic: Previous period ends exactly 1 month before current period ends
                prev_end = date_end - relativedelta(months=1)
                prev_start = prev_end.replace(day=1)
            
            # Build previous period WHERE
            where_prev_parts = ["WHERE tenant_id = ?", "AND date_time >= ?", "AND date_time <= ?"]
            params_prev = [tenant_id, prev_start, datetime.combine(prev_end, datetime.max.time())]
            
            if filters.get("categories"):
                cats = [c for c in filters["categories"] if c]
                if cats:
                    cat_conditions = []
                    for cat in cats:
                        cat_lower = cat.lower().strip()
                        cat_normalized = (
                            cat_lower
                            .replace(" & ", "_and_")
                            .replace("&", "_and_")
                            .replace(" ", "_")
                            .replace("-", "_")
                        )
                        cat_singular = cat_normalized.rstrip("s") if cat_normalized.endswith("s") and len(cat_normalized) > 3 else cat_normalized
                        
                        cat_conditions.append(
                            "("
                            "LOWER(COALESCE(category_final, category)) LIKE ? OR "
                            "LOWER(COALESCE(category_final, category)) LIKE ? OR "
                            "LOWER(COALESCE(category_final, category)) LIKE ?"
                            ")"
                        )
                        params_prev.extend([
                            f"%{cat_lower}%",
                            f"%{cat_normalized}%",
                            f"%{cat_singular}%",
                        ])
                    where_prev_parts.append(f"AND ({' OR '.join(cat_conditions)})")
            
            if filters.get("direction") and filters["direction"] != "all":
                where_prev_parts.append("AND direction = ?")
                params_prev.append(filters["direction"])
            
            if filters.get("exclude_transfers", True):
                where_prev_parts.append("AND direction != 'transfer'")
            
            where_prev_clause = " ".join(where_prev_parts)
            
            sql = cls.METRIC_TEMPLATES[metric].format(
                where=where_clause,
                where_previous=where_prev_clause,
                limit=limit,
                order_direction=order_direction
            )
            params.extend(params_prev)
        
        elif metric == "year_over_year":
            from dateutil.relativedelta import relativedelta
            
            date_start = filters.get("date_start")
            date_end = filters.get("date_end")
            
            if date_start and date_end:
                if isinstance(date_start, datetime):
                    date_start = date_start.date()
                if isinstance(date_end, datetime):
                    date_end = date_end.date()
                    
                prev_start = date_start - relativedelta(years=1)
                prev_end = date_end - relativedelta(years=1)
            else:
                today = date.today()
                # Default to this year vs last year if no date range
                date_start = today.replace(month=1, day=1)
                date_end = today
                prev_start = date_start - relativedelta(years=1)
                prev_end = date_end - relativedelta(years=1)
            
            # Build previous period WHERE
            where_prev_parts = ["WHERE tenant_id = ?", "AND date_time >= ?", "AND date_time <= ?"]
            params_prev = [tenant_id, prev_start, datetime.combine(prev_end, datetime.max.time())]
            
            # Re-apply filters for previous period
            if filters.get("categories"):
                cats = [c for c in filters["categories"] if c]
                if cats:
                    cat_conditions = []
                    for cat in cats:
                        cat_lower = cat.lower().strip()
                        cat_normalized = (
                            cat_lower
                            .replace(" & ", "_and_")
                            .replace("&", "_and_")
                            .replace(" ", "_")
                            .replace("-", "_")
                        )
                        cat_singular = cat_normalized.rstrip("s") if cat_normalized.endswith("s") and len(cat_normalized) > 3 else cat_normalized
                        
                        cat_conditions.append(
                            "("
                            "LOWER(COALESCE(category_final, category)) LIKE ? OR "
                            "LOWER(COALESCE(category_final, category)) LIKE ? OR "
                            "LOWER(COALESCE(category_final, category)) LIKE ?"
                            ")"
                        )
                        params_prev.extend([
                            f"%{cat_lower}%",
                            f"%{cat_normalized}%",
                            f"%{cat_singular}%",
                        ])
                    where_prev_parts.append(f"AND ({' OR '.join(cat_conditions)})")
            
            if filters.get("direction") and filters["direction"] != "all":
                where_prev_parts.append("AND direction = ?")
                params_prev.append(filters["direction"])
            
            if filters.get("exclude_transfers", True):
                where_prev_parts.append("AND direction != 'transfer'")
            
            where_prev_clause = " ".join(where_prev_parts)
            
            sql = cls.METRIC_TEMPLATES[metric].format(
                where=where_clause,
                where_previous=where_prev_clause,
                limit=limit,
                order_direction=order_direction
            )
            params.extend(params_prev)
        elif metric == "subscription_list":
            # Special case: uses different parameter
            sql = cls.METRIC_TEMPLATES[metric].format(limit=limit)
            params = [tenant_id]
        else:
            # Validate order_by if provided
            safe_order_by = "date_time"
            if order_by and order_by in cls.ALLOWED_COLUMNS:
                safe_order_by = order_by

            sql = cls.METRIC_TEMPLATES[metric].format(
                where=where_clause,
                limit=limit,
                order_direction=order_direction,
                order_by=safe_order_by
            )
        
        return sql, params
    
    @classmethod
    def validate_sql(cls, sql: str) -> bool:
        """Validate SQL is safe (SELECT only)"""
        sql_upper = sql.strip().upper()
        
        # Must start with SELECT or WITH
        if not (sql_upper.startswith("SELECT") or sql_upper.startswith("WITH")):
            return False
        
        # Forbidden keywords
        forbidden = [
            "DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE",
            "TRUNCATE", "EXEC", "EXECUTE", "GRANT", "REVOKE", 
            "COPY", "ATTACH", "DETACH"
        ]
        
        for keyword in forbidden:
            # Check for keyword as whole word
            if f" {keyword} " in sql_upper or sql_upper.startswith(f"{keyword} "):
                return False
        
        return True


# -----------------------------------------------------------------------------
# SINGLETON
# -----------------------------------------------------------------------------

_db_manager: Optional[DatabaseManager] = None


def get_db() -> DatabaseManager:
    """Get or create database manager singleton"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager

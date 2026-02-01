"""
Statement Copilot - DuckDB Setup Script (UPDATED)
=================================================
This script creates the DuckDB database and loads CSV data.

Updates:
- STORED/VIRTUAL compatibility for generated columns (runtime test + fallback)
- Indexes moved to after import (faster ingest)
- date_time tamamen bozuksa crash engellendi (df.empty guard)
- Auto-increment for tool_audit: SEQUENCE + DEFAULT nextval(...)
- Subscription detection no longer assumes amount sign (ABS/normalize)
- file_id daha stabil: tenant + sha256(file_bytes)

Usage:
    pip install duckdb>=0.7.0 pandas
    python scripts/setup/duckdb_setup.py
"""

import duckdb
import pandas as pd
from pathlib import Path
import hashlib
import json
import os

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "db" / "statement_copilot.duckdb"
DATA_PATH = PROJECT_ROOT / "data"
DEFAULT_TENANT_ID = os.getenv("TENANT_ID", "default_tenant")


# -----------------------------
# Utilities
# -----------------------------
def _parse_version(v: str) -> tuple[int, int, int]:
    parts = (v.split("+")[0]).split(".")
    nums = []
    for p in parts[:3]:
        try:
            nums.append(int(p))
        except ValueError:
            nums.append(0)
    while len(nums) < 3:
        nums.append(0)
    return tuple(nums)  # type: ignore


def ensure_min_duckdb_version(min_version: str = "0.7.0"):
    cur = _parse_version(getattr(duckdb, "__version__", "0.0.0"))
    req = _parse_version(min_version)
    if cur < req:
        raise RuntimeError(
            f"DuckDB version too old: {duckdb.__version__}. "
            f"Please use duckdb>={min_version} kullan."
        )


def pick_generated_mode(con: duckdb.DuckDBPyConnection) -> str | None:
    """
    DuckDB'de generated columns destek durumunu test eder.
    Try VIRTUAL first, then STORED, otherwise return None.
    """
    # VIRTUAL test
    try:
        con.execute("DROP TABLE IF EXISTS __gen_test")
        con.execute(
            "CREATE TABLE __gen_test(a INT, b INT GENERATED ALWAYS AS (a + 1) VIRTUAL)"
        )
        con.execute("DROP TABLE IF EXISTS __gen_test")
        return "VIRTUAL"
    except Exception:
        pass

    # STORED test
    try:
        con.execute("DROP TABLE IF EXISTS __gen_test")
        con.execute(
            "CREATE TABLE __gen_test(a INT, b INT GENERATED ALWAYS AS (a + 1) STORED)"
        )
        con.execute("DROP TABLE IF EXISTS __gen_test")
        return "STORED"
    except Exception:
        pass

    # No generated support
    try:
        con.execute("DROP TABLE IF EXISTS __gen_test")
    except Exception:
        pass
    return None


def ensure_column(con: duckdb.DuckDBPyConnection, table_name: str, column_name: str, ddl: str):
    existing = [row[1] for row in con.execute(f"PRAGMA table_info('{table_name}')").fetchall()]
    if column_name not in existing:
        con.execute(f"ALTER TABLE {table_name} ADD COLUMN {ddl}")


def backfill_tenant_id(con: duckdb.DuckDBPyConnection, table_name: str, tenant_id: str):
    con.execute(
        f"UPDATE {table_name} SET tenant_id = ? WHERE tenant_id IS NULL OR tenant_id = ''",
        [tenant_id],
    )


def try_backfill_time_buckets(con: duckdb.DuckDBPyConnection, tenant_id: str | None = None, file_id: str | None = None):
    """
    Fill month_year / day_of_week / hour_of_day columns if they are normal columns.
    If generated columns, update fails; we catch and ignore.
    """
    where = "WHERE date_time IS NOT NULL"
    params: list[str] = []
    if tenant_id:
        where += " AND tenant_id = ?"
        params.append(tenant_id)
    if file_id:
        where += " AND file_id = ?"
        params.append(file_id)

    updates = [
        ("month_year", "strftime(date_time, '%Y-%m')"),
        ("day_of_week", "dayofweek(date_time)"),
        ("hour_of_day", "hour(date_time)"),
    ]

    for col, expr in updates:
        try:
            con.execute(
                f"""
                UPDATE transactions
                SET {col} = {expr}
                {where}
                  AND ({col} IS NULL OR CAST({col} AS VARCHAR) = '')
                """,
                params,
            )
        except Exception:
            # If column missing or generated, update is not possible; that's ok.
            pass


# -----------------------------
# Schema
# -----------------------------
def create_schema(con: duckdb.DuckDBPyConnection):
    """Create all tables - aligned to Statement Copilot spec"""

    gen_mode = pick_generated_mode(con)  # "VIRTUAL" | "STORED" | None

    # 1) raw_files
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS raw_files (
            file_id VARCHAR PRIMARY KEY,
            tenant_id VARCHAR DEFAULT 'default_tenant',
            user_id VARCHAR DEFAULT 'default_user',
            file_name VARCHAR,
            sha256 VARCHAR,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status VARCHAR DEFAULT 'uploaded',  -- uploaded/ingesting/enriched/failed
            parser_hint VARCHAR,               -- isbank_xls / garanti_xls / auto
            row_count INTEGER,
            date_range_start DATE,
            date_range_end DATE
        )
        """
    )

    # 2) transactions
    if gen_mode:
        time_bucket_cols = f"""
            month_year VARCHAR GENERATED ALWAYS AS (strftime(date_time, '%Y-%m')) {gen_mode},
            day_of_week INTEGER GENERATED ALWAYS AS (dayofweek(date_time)) {gen_mode},
            hour_of_day INTEGER GENERATED ALWAYS AS (hour(date_time)) {gen_mode}
        """
    else:
        # For older versions without generated support, create normal columns.
        time_bucket_cols = """
            month_year VARCHAR,
            day_of_week INTEGER,
            hour_of_day INTEGER
        """

    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS transactions (
            tx_id VARCHAR PRIMARY KEY,
            file_id VARCHAR,
            tenant_id VARCHAR DEFAULT 'default_tenant',
            user_id VARCHAR DEFAULT 'default_user',
            currency VARCHAR DEFAULT 'TRY',

            -- Zaman bilgileri
            date_time TIMESTAMP,
            value_date DATE,

            -- Tutar bilgileri
            amount DOUBLE,
            balance DOUBLE,
            direction VARCHAR,              -- expense/income/neutral/transfer

            -- Transaction details
            description VARCHAR,
            merchant_norm VARCHAR,
            merchant_raw VARCHAR,
            merchant_key VARCHAR,

            -- Categorization
            category VARCHAR,
            subcategory VARCHAR,
            tags VARCHAR,
            tags_arr VARCHAR[],
            taxonomy_version VARCHAR DEFAULT '1.0.0',
            category_final VARCHAR,
            subcategory_final VARCHAR,
            tags_final VARCHAR[],
            category_updated_at TIMESTAMP,

            -- Kanal ve tip bilgileri
            channel VARCHAR,
            transaction_type VARCHAR,
            transaction_code VARCHAR,

            -- Ek bilgiler
            reference VARCHAR,
            confidence DOUBLE,
            reasoning VARCHAR,
            overdraft_balance DOUBLE,
            source VARCHAR,
            source_file VARCHAR,

            -- Time buckets (read-only if generated)
            {time_bucket_cols}
        )
        """
    )

    # 3) merchant_dictionary
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS merchant_dictionary (
            tenant_id VARCHAR DEFAULT 'default_tenant',
            pattern VARCHAR,
            merchant_norm VARCHAR,
            category_default VARCHAR,
            confidence DOUBLE DEFAULT 1.0,
            source VARCHAR DEFAULT 'manual',  -- manual/rule/fuzzy/llm_suggest
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (tenant_id, pattern)
        )
        """
    )

    # 4) category_rules
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS category_rules (
            tenant_id VARCHAR DEFAULT 'default_tenant',
            rule_id VARCHAR,
            priority INTEGER DEFAULT 100,
            match_type VARCHAR,             -- contains/regex/merchant/amount_range
            pattern VARCHAR,
            category VARCHAR,
            subcategory VARCHAR,
            is_active BOOLEAN DEFAULT true,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (tenant_id, rule_id)
        )
        """
    )

    # 5) subscriptions
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS subscriptions (
            sub_id VARCHAR PRIMARY KEY,
            tenant_id VARCHAR DEFAULT 'default_tenant',
            merchant_norm VARCHAR,
            amount_median DOUBLE,
            amount_variance DOUBLE,
            period_days INTEGER,
            last_seen TIMESTAMP,
            next_estimate TIMESTAMP,
            confidence DOUBLE,
            evidence_tx_ids VARCHAR,        -- JSON array of tx_ids
            status VARCHAR DEFAULT 'active', -- active/cancelled/paused
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    # 6) budgets
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS budgets (
            budget_id VARCHAR PRIMARY KEY,
            user_id VARCHAR DEFAULT 'default_user',
            category VARCHAR,
            subcategory VARCHAR,
            period VARCHAR,                 -- monthly/weekly/yearly
            limit_amount DOUBLE,
            alert_threshold DOUBLE DEFAULT 0.8,
            is_active BOOLEAN DEFAULT true,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    # 7) alerts
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS alerts (
            alert_id VARCHAR PRIMARY KEY,
            user_id VARCHAR DEFAULT 'default_user',
            alert_type VARCHAR,
            condition_json VARCHAR,
            notification_method VARCHAR,
            is_active BOOLEAN DEFAULT true,
            last_triggered TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    # 8) actions
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS actions (
            action_id VARCHAR PRIMARY KEY,
            session_id VARCHAR,
            action_type VARCHAR,
            plan_json VARCHAR,
            status VARCHAR DEFAULT 'planned',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            approved_at TIMESTAMP,
            completed_at TIMESTAMP,
            result_artifacts_json VARCHAR,
            error_message VARCHAR
        )
        """
    )

    # 9) tool_audit (auto-id)
    con.execute("CREATE SEQUENCE IF NOT EXISTS tool_audit_id_seq START 1;")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS tool_audit (
            id BIGINT PRIMARY KEY DEFAULT nextval('tool_audit_id_seq'),
            trace_id VARCHAR,
            session_id VARCHAR,
            node VARCHAR,
            tool_name VARCHAR,
            model_name VARCHAR,
            input_hash VARCHAR,
            output_hash VARCHAR,
            latency_ms INTEGER,
            success BOOLEAN,
            error_message VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    # 10) chat_sessions
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_sessions (
            session_id VARCHAR PRIMARY KEY,
            user_id VARCHAR DEFAULT 'default_user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            message_count INTEGER DEFAULT 0,
            context_summary VARCHAR
        )
        """
    )

    # 11) chat_messages
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_messages (
            message_id VARCHAR PRIMARY KEY,
            session_id VARCHAR,
            role VARCHAR,
            content VARCHAR,
            tool_calls_json VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    print(" All tables created")


def migrate_v2(con: duckdb.DuckDBPyConnection):
    """V2 migration - yeni kolonlar"""
    con.execute("ALTER TABLE transactions ADD COLUMN IF NOT EXISTS user_id VARCHAR DEFAULT 'default_user'")
    con.execute("ALTER TABLE transactions ADD COLUMN IF NOT EXISTS currency VARCHAR DEFAULT 'TRY'")
    con.execute("ALTER TABLE transactions ADD COLUMN IF NOT EXISTS merchant_raw VARCHAR")
    con.execute("ALTER TABLE transactions ADD COLUMN IF NOT EXISTS merchant_key VARCHAR")
    con.execute("ALTER TABLE transactions ADD COLUMN IF NOT EXISTS tags_arr VARCHAR[]")
    con.execute("ALTER TABLE transactions ADD COLUMN IF NOT EXISTS taxonomy_version VARCHAR DEFAULT '1.0.0'")
    con.execute("ALTER TABLE transactions ADD COLUMN IF NOT EXISTS category_final VARCHAR")
    con.execute("ALTER TABLE transactions ADD COLUMN IF NOT EXISTS subcategory_final VARCHAR")
    con.execute("ALTER TABLE transactions ADD COLUMN IF NOT EXISTS tags_final VARCHAR[]")
    con.execute("ALTER TABLE transactions ADD COLUMN IF NOT EXISTS category_updated_at TIMESTAMP")

    # time buckets (for older DBs)
    ensure_column(con, "transactions", "month_year", "month_year VARCHAR")
    ensure_column(con, "transactions", "day_of_week", "day_of_week INTEGER")
    ensure_column(con, "transactions", "hour_of_day", "hour_of_day INTEGER")

    con.execute("UPDATE transactions SET user_id = 'default_user' WHERE user_id IS NULL OR user_id = ''")
    con.execute("UPDATE transactions SET currency = 'TRY' WHERE currency IS NULL OR currency = ''")


def create_indexes(con: duckdb.DuckDBPyConnection):
    """
    Indexes are created after import.
    Note: DuckDB indexes may not help every workload.
    """
    con.execute("CREATE INDEX IF NOT EXISTS idx_tx_tenant_user ON transactions(tenant_id, user_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_tx_date ON transactions(date_time)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_tx_merchant_norm ON transactions(merchant_norm)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_tx_merchant_key ON transactions(merchant_key)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_tx_file ON transactions(file_id)")
    print(" Indexes created (post import)")


# -----------------------------
# Loading
# -----------------------------
def generate_tx_id(row: pd.Series, tenant_id: str) -> str:
    """Create unique ID for transaction"""
    key_parts = [
        tenant_id,
        str(row.get("date_time", "")),
        str(row.get("amount", "")),
        str(row.get("description", "")),
        str(row.get("reference", "")),
    ]
    return hashlib.sha256("|".join(key_parts).encode()).hexdigest()[:16]


def load_transactions_from_csv(con: duckdb.DuckDBPyConnection, csv_path: Path, tenant_id: str):
    """Load transactions from CSV"""

    print(f" Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    user_id = os.getenv("USER_ID", "default_user")

    # date_time kolon kontrol
    if "date_time" not in df.columns:
        print(" CSV missing 'date_time' column. File skipped.")
        # raw_files failed record (traceability)
        file_sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        file_id = hashlib.sha256(f"{tenant_id}:{file_sha}".encode()).hexdigest()[:12]
        con.execute(
            """
            INSERT OR REPLACE INTO raw_files
            (file_id, tenant_id, user_id, file_name, sha256, status, row_count)
            VALUES (?, ?, ?, ?, ?, 'failed', ?)
            """,
            [file_id, tenant_id, user_id, csv_path.name, file_sha, int(len(df))],
        )
        return None

    # file_sha + file_id (stabil)
    file_sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    file_id = hashlib.sha256(f"{tenant_id}:{file_sha}".encode()).hexdigest()[:12]

    # date_time parse + invalid drop
    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    invalid_date_rows = int(df["date_time"].isna().sum())
    if invalid_date_rows:
        df = df[df["date_time"].notna()].copy()
        print(f"  {invalid_date_rows} rows skipped (invalid date_time)")

    # guard: tamamen bo kaldysa crash olmasn
    if df.empty:
        print("  date_time completely invalid. File marked as 'failed' and skipped.")
        con.execute(
            """
            INSERT OR REPLACE INTO raw_files
            (file_id, tenant_id, user_id, file_name, sha256, status, row_count)
            VALUES (?, ?, ?, ?, ?, 'failed', 0)
            """,
            [file_id, tenant_id, user_id, csv_path.name, file_sha],
        )
        return None

    date_min = df["date_time"].min()
    date_max = df["date_time"].max()

    # raw_files kayd
    con.execute(
        """
        INSERT OR REPLACE INTO raw_files
        (file_id, tenant_id, user_id, file_name, sha256, status, row_count, date_range_start, date_range_end)
        VALUES (?, ?, ?, ?, ?, 'enriched', ?, ?, ?)
        """,
        [file_id, tenant_id, user_id, csv_path.name, file_sha, int(len(df)), date_min.date(), date_max.date()],
    )

    # tx_id
    df["tenant_id"] = tenant_id
    df["user_id"] = user_id

    if "tx_id" not in df.columns:
        df["tx_id"] = df.apply(lambda r: generate_tx_id(r, tenant_id), axis=1)
    else:
        missing = df["tx_id"].isna() | (df["tx_id"].astype(str).str.strip() == "")
        if missing.any():
            df.loc[missing, "tx_id"] = df[missing].apply(lambda r: generate_tx_id(r, tenant_id), axis=1)

    df["file_id"] = file_id

    # value_date parse (fallback to date_time)
    if "value_date" in df.columns:
        df["value_date"] = pd.to_datetime(df["value_date"], errors="coerce").dt.date
    else:
        df["value_date"] = df["date_time"].dt.date

    # amount normalize
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    else:
        df["amount"] = None

    # Sign normalization by direction (expense negative, income positive)
    if "direction" in df.columns and "amount" in df.columns:
        dir_lower = df["direction"].astype(str).str.strip().str.lower()
        exp = dir_lower.eq("expense")
        inc = dir_lower.eq("income")
        df.loc[exp, "amount"] = -df.loc[exp, "amount"].abs()
        df.loc[inc, "amount"] = df.loc[inc, "amount"].abs()

    # merchant normalize
    if "merchant_norm" in df.columns:
        df["merchant_raw"] = df["merchant_norm"]
        df["merchant_norm"] = df["merchant_norm"].astype(str).str.strip().str.upper()
        df.loc[df["merchant_norm"].isin(["", "NONE", "NAN"]), "merchant_norm"] = None
        df["merchant_key"] = df["merchant_norm"]
    else:
        df["merchant_norm"] = None
        df["merchant_raw"] = None
        df["merchant_key"] = None

    # currency
    if "currency" not in df.columns:
        df["currency"] = "TRY"

    # tags_arr
    def parse_tags(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return []
        s = str(x)
        parts = [p.strip().lower() for p in s.split(",") if p.strip()]
        return sorted(set(parts))

    if "tags" in df.columns:
        df["tags_arr"] = df["tags"].apply(parse_tags)
    else:
        df["tags"] = None
        df["tags_arr"] = [[] for _ in range(len(df))]

    # taxonomy defaults
    if "taxonomy_version" not in df.columns:
        df["taxonomy_version"] = "1.0.0"

    # insert cols (filter to existing)
    column_order = [
        "tx_id",
        "file_id",
        "tenant_id",
        "user_id",
        "currency",
        "date_time",
        "value_date",
        "amount",
        "balance",
        "direction",
        "description",
        "merchant_norm",
        "merchant_raw",
        "merchant_key",
        "category",
        "subcategory",
        "tags",
        "tags_arr",
        "taxonomy_version",
        "category_final",
        "subcategory_final",
        "tags_final",
        "category_updated_at",
        "channel",
        "transaction_type",
        "transaction_code",
        "reference",
        "confidence",
        "reasoning",
        "overdraft_balance",
        "source",
        "source_file",
        # month_year/day_of_week/hour_of_day insert etmiyoruz (generated olabilir)
    ]

    available_cols = [c for c in column_order if c in df.columns]
    df_insert = df[available_cols].copy()

    # Bulk replace (same file_id+tenant)
    con.execute("DELETE FROM transactions WHERE file_id = ? AND tenant_id = ?", [file_id, tenant_id])
    con.register("df_insert", df_insert)

    insert_cols = ", ".join(available_cols)
    con.execute(
        f"""
        INSERT INTO transactions ({insert_cols})
        SELECT {insert_cols} FROM df_insert
        """
    )

    # If time bucket columns are normal columns, fill them
    try_backfill_time_buckets(con, tenant_id=tenant_id, file_id=file_id)

    print(f" {len(df)} transactions loaded")
    print(f"    Date range: {date_min.date()} - {date_max.date()}")

    return file_id


# -----------------------------
# Enrichment
# -----------------------------
def populate_merchant_dictionary(con: duckdb.DuckDBPyConnection, tenant_id: str):
    merchants = con.execute(
        """
        SELECT merchant_norm, category, COUNT(*) as cnt
        FROM transactions
        WHERE tenant_id = ?
          AND merchant_norm IS NOT NULL AND merchant_norm != ''
        GROUP BY merchant_norm, category
        ORDER BY cnt DESC
        """,
        [tenant_id],
    ).fetchall()

    for merchant, category, _cnt in merchants:
        con.execute(
            """
            INSERT OR IGNORE INTO merchant_dictionary
            (tenant_id, pattern, merchant_norm, category_default, source)
            VALUES (?, ?, ?, ?, 'auto_detected')
            """,
            [tenant_id, merchant, merchant, category],
        )

    print(f" {len(merchants)} merchant kaydedildi")


def detect_subscriptions(con: duckdb.DuckDBPyConnection, tenant_id: str):
    """
    Detect recurring payments.
    NOT: amount iaretine gvenmiyoruz. direction='expense' + ABS(amount) zerinden gidiyoruz.
    """

    subscription_candidates = con.execute(
        """
        WITH merchant_payments AS (
            SELECT
                merchant_norm,
                date_time,
                ABS(amount) as amount_abs,
                tx_id,
                LAG(date_time) OVER (PARTITION BY merchant_norm ORDER BY date_time) as prev_date
            FROM transactions
            WHERE tenant_id = ?
              AND direction = 'expense'
              AND merchant_norm IS NOT NULL
              AND amount IS NOT NULL
              AND ABS(amount) > 0
        ),
        intervals AS (
            SELECT
                merchant_norm,
                DATEDIFF('day', prev_date, date_time) as days_between,
                amount_abs,
                tx_id,
                date_time
            FROM merchant_payments
            WHERE prev_date IS NOT NULL
        ),
        -- 1. Interval Based (Strict period match)
        interval_candidates AS (
            SELECT
                merchant_norm,
                MEDIAN(days_between) as median_interval,
                MEDIAN(amount_abs) as median_amount_abs,
                STDDEV(amount_abs) as amount_stddev,
                COUNT(*) as occurrence_count,
                LIST(tx_id) as tx_ids
            FROM intervals
            WHERE days_between BETWEEN 25 AND 35   -- Monthly
               OR days_between BETWEEN 6 AND 8     -- Weekly
               OR days_between BETWEEN 360 AND 370 -- Yearly
            GROUP BY merchant_norm
            HAVING COUNT(*) >= 2
        ),
        -- 2. Frequency Based (Approximate monthly match for irregular dates)
        stats_per_merchant AS (
            SELECT
                merchant_norm,
                MIN(date_time) as first_date,
                MAX(date_time) as last_date,
                COUNT(*) as total_count,
                MEDIAN(amount_abs) as median_amount_abs,
                STDDEV(amount_abs) as amount_stddev,
                LIST(tx_id) as tx_ids
            FROM merchant_payments
            GROUP BY merchant_norm
        ),
        frequency_candidates AS (
            SELECT
                merchant_norm,
                30 as median_interval, -- Assume monthly
                median_amount_abs,
                amount_stddev,
                total_count as occurrence_count,
                tx_ids
            FROM stats_per_merchant
            WHERE total_count >= 3 -- Need more evidence for loose matching
              AND DATEDIFF('day', first_date, last_date) >= 50 -- Spam at least ~2 months
              -- Check density: Transactions per month is roughly 1 (0.6 - 1.8)
              AND (CAST(total_count AS FLOAT) / (GREATEST(DATEDIFF('day', first_date, last_date), 1) / 30.0)) BETWEEN 0.6 AND 1.8
        ),
        -- Combine (Prioritize Interval Based)
        combined_results AS (
            SELECT * FROM interval_candidates
            UNION ALL
            SELECT * FROM frequency_candidates
            WHERE merchant_norm NOT IN (SELECT merchant_norm FROM interval_candidates)
        )
        SELECT 
            merchant_norm, 
            median_interval, 
            median_amount_abs, 
            amount_stddev, 
            occurrence_count, 
            tx_ids
        FROM combined_results
        ORDER BY occurrence_count DESC
        """,
        [tenant_id],
    ).fetchall()

    for merchant, interval, median_abs, stddev, count, tx_ids in subscription_candidates:
        sub_id = hashlib.sha256(f"{tenant_id}_{merchant}_{interval}".encode()).hexdigest()[:12]

        last_payment = con.execute(
            """
            SELECT MAX(date_time)
            FROM transactions
            WHERE tenant_id = ? AND merchant_norm = ? AND direction = 'expense'
            """,
            [tenant_id, merchant],
        ).fetchone()[0]

        if last_payment:
            next_estimate = pd.Timestamp(last_payment) + pd.Timedelta(days=int(interval))

            # Standardize amount_median as negative for expense (or keep positive if you prefer)
            amount_median = -float(median_abs) if median_abs is not None else None

            con.execute(
                """
                INSERT OR REPLACE INTO subscriptions
                (sub_id, tenant_id, merchant_norm, amount_median, amount_variance, period_days,
                 last_seen, next_estimate, confidence, evidence_tx_ids)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    sub_id,
                    tenant_id,
                    merchant,
                    amount_median,
                    float(stddev) if stddev is not None else 0.0,
                    int(interval),
                    last_payment,
                    next_estimate.to_pydatetime(),
                    min(0.5 + int(count) * 0.1, 0.95),
                    json.dumps(list(tx_ids)[:10]),
                ],
            )

    print(f" {len(subscription_candidates)} abonelik tespit edildi")


def insert_default_category_rules(con: duckdb.DuckDBPyConnection, tenant_id: str):
    rules = [
        ("rule_001", 10, "merchant", "MIGROS", "food_and_dining", "groceries"),
        ("rule_002", 10, "merchant", "A101", "food_and_dining", "groceries"),
        ("rule_003", 10, "merchant", "BIM", "food_and_dining", "groceries"),
        ("rule_004", 10, "merchant", "CARREFOUR", "food_and_dining", "groceries"),
        ("rule_005", 20, "contains", "FATURA", "utilities", "other_utilities"),
        ("rule_006", 20, "contains", "TRAKYAGAZ", "utilities", "gas_heating"),
        ("rule_007", 20, "contains", "TREPA", "utilities", "electricity"),
        ("rule_008", 30, "merchant", "YOUTUBE", "utilities", "tv_streaming"),
        ("rule_009", 30, "merchant", "NETFLIX", "utilities", "tv_streaming"),
        ("rule_010", 30, "merchant", "SPOTIFY", "utilities", "tv_streaming"),
        ("rule_011", 30, "merchant", "OPENAI", "business_professional", "software_subscriptions"),
        ("rule_012", 30, "merchant", "GITHUB", "business_professional", "software_subscriptions"),
        ("rule_013", 30, "merchant", "AWS", "business_professional", "domain_hosting"),
        ("rule_014", 40, "contains", "ATM", "financial_services", "atm_withdrawal"),
        ("rule_015", 40, "contains", "FAST", "transfers", "p2p_sent"),
        ("rule_016", 40, "contains", "HAVALE", "transfers", "p2p_sent"),
        ("rule_017", 50, "contains", "BLK", "internal_banking", "authorization_hold"),
    ]

    for rule in rules:
        con.execute(
            """
            INSERT OR IGNORE INTO category_rules
            (tenant_id, rule_id, priority, match_type, pattern, category, subcategory)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (tenant_id, *rule),
        )

    print(f" {len(rules)} categorization rules added")


def print_database_stats(con: duckdb.DuckDBPyConnection, tenant_id: str):
    print("\n" + "=" * 60)
    print(" DATABASE STATISTICS")
    print("=" * 60)

    tx_count = con.execute(
        "SELECT COUNT(*) FROM transactions WHERE tenant_id = ?", [tenant_id]
    ).fetchone()[0]
    print(f"\n Toplam Transaction: {tx_count:,}")

    date_range = con.execute(
        """
        SELECT MIN(date_time), MAX(date_time)
        FROM transactions
        WHERE tenant_id = ?
        """,
        [tenant_id],
    ).fetchone()
    print(f" Date Range: {date_range[0]} - {date_range[1]}")

    print("\n Category Distribution:")
    categories = con.execute(
        """
        SELECT category, COUNT(*) as cnt, SUM(ABS(amount)) as total
        FROM transactions
        WHERE tenant_id = ?
        GROUP BY category
        ORDER BY cnt DESC
        LIMIT 10
        """,
        [tenant_id],
    ).fetchall()
    for cat, cnt, total in categories:
        print(f"   {cat or 'NULL':<25} {cnt:>5} transactions  {total:>12,.2f} TL")

    print("\n Income/Expense Distribution:")
    directions = con.execute(
        """
        SELECT direction, COUNT(*) as cnt, SUM(amount) as total
        FROM transactions
        WHERE tenant_id = ?
        GROUP BY direction
        """,
        [tenant_id],
    ).fetchall()
    for dir_, cnt, total in directions:
        print(f"   {str(dir_):<15} {cnt:>5} transactions  {total:>12,.2f} TL")

    print("\n Top 10 Merchant:")
    merchants = con.execute(
        """
        SELECT merchant_norm, COUNT(*) as cnt, SUM(ABS(amount)) as total
        FROM transactions
        WHERE tenant_id = ?
          AND merchant_norm IS NOT NULL
        GROUP BY merchant_norm
        ORDER BY total DESC
        LIMIT 10
        """,
        [tenant_id],
    ).fetchall()
    for merchant, cnt, total in merchants:
        print(f"   {merchant:<30} {cnt:>5} transactions  {total:>12,.2f} TL")

    sub_count = con.execute(
        "SELECT COUNT(*) FROM subscriptions WHERE tenant_id = ?", [tenant_id]
    ).fetchone()[0]
    print(f"\n Tespit Edilen Abonelik: {sub_count}")

    if sub_count > 0:
        subs = con.execute(
            """
            SELECT merchant_norm, amount_median, period_days
            FROM subscriptions
            WHERE tenant_id = ?
            ORDER BY ABS(amount_median) DESC
            LIMIT 5
            """,
            [tenant_id],
        ).fetchall()
        for merchant, amount, period in subs:
            period_str = (
                "monthly" if 25 <= period <= 35 else "weekly" if 6 <= period <= 8 else f"{period} days"
            )
            print(f"   {merchant:<25} {amount:>10,.2f} TL ({period_str})")

    print("\n" + "=" * 60)


# -----------------------------
# Main
# -----------------------------
def main():
    ensure_min_duckdb_version("0.7.0")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    tenant_id = (DEFAULT_TENANT_ID or "").strip() or "default_tenant"

    print(" Statement Copilot DuckDB Kurulumu (UPDATED)")
    print("=" * 60)
    print(f"Tenant: {tenant_id}")

    con = duckdb.connect(str(DB_PATH))
    try:
        print("\n Creating schema...")
        create_schema(con)
        migrate_v2(con)

        # Ensure tenant_id columns (for older DBs)
        ensure_column(con, "raw_files", "tenant_id", "tenant_id VARCHAR DEFAULT 'default_tenant'")
        ensure_column(con, "transactions", "tenant_id", "tenant_id VARCHAR DEFAULT 'default_tenant'")
        ensure_column(con, "merchant_dictionary", "tenant_id", "tenant_id VARCHAR DEFAULT 'default_tenant'")
        ensure_column(con, "category_rules", "tenant_id", "tenant_id VARCHAR DEFAULT 'default_tenant'")
        ensure_column(con, "subscriptions", "tenant_id", "tenant_id VARCHAR DEFAULT 'default_tenant'")

        backfill_tenant_id(con, "raw_files", tenant_id)
        backfill_tenant_id(con, "transactions", tenant_id)
        backfill_tenant_id(con, "merchant_dictionary", tenant_id)
        backfill_tenant_id(con, "category_rules", tenant_id)
        backfill_tenant_id(con, "subscriptions", tenant_id)

        print("\n Loading CSV files...")
        csv_files = list(DATA_PATH.glob("*.csv"))
        if csv_files:
            for csv_file in csv_files:
                load_transactions_from_csv(con, csv_file, tenant_id)
        else:
            print("  No CSV files found in data/ folder")
            print("   Please copy the enriched CSV file into the data/ folder")

        # Backfill time buckets if they are normal columns
        try_backfill_time_buckets(con, tenant_id=tenant_id)

        print("\n Merchant dictionary dolduruluyor...")
        populate_merchant_dictionary(con, tenant_id)

        print("\n Abonelikler tespit ediliyor...")
        detect_subscriptions(con, tenant_id)

        print("\n Adding default rules...")
        insert_default_category_rules(con, tenant_id)

        # indeksleri en sona
        print("\n Creating indexes...")
        create_indexes(con)

        print_database_stats(con, tenant_id)

        con.commit()
        print(f"\n Database created successfully: {DB_PATH}")

    finally:
        con.close()


if __name__ == "__main__":
    main()
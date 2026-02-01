
import duckdb
import pandas as pd

def test_logic():
    con = duckdb.connect(":memory:")
    
    # Create dummy transactions
    con.execute("CREATE TABLE transactions (tenant_id VARCHAR, merchant_norm VARCHAR, date_time TIMESTAMP, amount DOUBLE, direction VARCHAR, tx_id VARCHAR)")
    
    # Case 1: Regular monthly (Original logic should catch)
    # 30 days interval
    dates1 = ["2024-01-01", "2024-01-31", "2024-03-02"] 
    for i, d in enumerate(dates1):
        con.execute(f"INSERT INTO transactions VALUES ('t1', 'NETFLIX', '{d}', -100, 'expense', 'id1_{i}')")
        
    # Case 2: Irregular dates but monthly frequency (New logic should catch)
    # Jan 5, Feb 20, Mar 2 (Close enough density: 3 tx in ~2 months)
    dates2 = ["2024-01-05", "2024-02-20", "2024-03-02"]
    for i, d in enumerate(dates2):
        con.execute(f"INSERT INTO transactions VALUES ('t1', 'ISKANI', '{d}', -500, 'expense', 'id2_{i}')")

    # Case 3: Random spam (Should NOT catch)
    # 5 tx in 10 days
    dates3 = ["2024-01-01", "2024-01-03", "2024-01-05", "2024-01-07", "2024-01-09"]
    for i, d in enumerate(dates3):
        con.execute(f"INSERT INTO transactions VALUES ('t1', 'MARKET', '{d}', -50, 'expense', 'id3_{i}')")

    print("Data inserted.")
    
    query = """
    WITH merchant_payments AS (
            SELECT
                merchant_norm,
                date_time,
                ABS(amount) as amount_abs,
                tx_id,
                LAG(date_time) OVER (PARTITION BY merchant_norm ORDER BY date_time) as prev_date
            FROM transactions
            WHERE tenant_id = 't1'
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
              AND DATEDIFF('day', first_date, last_date) >= 50 -- Spam at least ~2 months (relaxed for test)
              -- Check density: Transactions per month is roughly 1 (0.75 - 1.5)
              AND (CAST(total_count AS FLOAT) / (GREATEST(DATEDIFF('day', first_date, last_date), 1) / 30.0)) BETWEEN 0.75 AND 1.5
        ),
        -- Combine (Prioritize Interval Based)
        combined_results AS (
            SELECT * FROM interval_candidates
            UNION ALL
            SELECT * FROM frequency_candidates
            WHERE merchant_norm NOT IN (SELECT merchant_norm FROM interval_candidates)
        )
        SELECT * FROM combined_results
    """
    
    res = con.execute(query).fetchall()
    print("Results:")
    for r in res:
        print(r)

if __name__ == "__main__":
    test_logic()

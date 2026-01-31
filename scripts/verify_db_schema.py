
import duckdb
import sys
import os

# Define expected columns based on duckdb_setup.py
EXPECTED_COLUMNS = {
    'tx_id', 'file_id', 'tenant_id', 'user_id', 'currency',
    'date_time', 'value_date',
    'amount', 'balance', 'direction',
    'description', 'merchant_norm', 'merchant_raw', 'merchant_key',
    'category', 'subcategory', 'tags', 'tags_arr',
    'taxonomy_version', 'category_final', 'subcategory_final', 'tags_final',
    'category_updated_at',
    'channel', 'transaction_type', 'transaction_code',
    'reference', 'confidence', 'reasoning',
    'overdraft_balance', 'source', 'source_file',
    'month_year', 'day_of_week', 'hour_of_day'
}

DB_PATH = os.path.join(os.getcwd(), "db", "statement_copilot.duckdb")

def verify_schema():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at: {DB_PATH}")
        return

    print(f"Checking database at: {DB_PATH}")
    try:
        con = duckdb.connect(DB_PATH, read_only=True)
        
        # Check tables
        tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
        print(f"Found tables: {tables}")
        
        if 'transactions' not in tables:
            print("CRITICAL: 'transactions' table missing!")
            return

        # Check columns
        columns_info = con.execute("PRAGMA table_info('transactions')").fetchall()
        # row: (cid, name, type, notnull, dflt_value, pk)
        actual_columns = {r[1] for r in columns_info}
        
        print("\nColumn Analysis for 'transactions':")
        missing_cols = EXPECTED_COLUMNS - actual_columns
        extra_cols = actual_columns - EXPECTED_COLUMNS
        
        if missing_cols:
            print(f"MISSING columns ({len(missing_cols)}):")
            for c in sorted(missing_cols):
                print(f"  - {c}")
        else:
            print("All expected columns operate properly.")
            
        if extra_cols:
            print(f"EXTRA columns ({len(extra_cols)}):")
            for c in sorted(extra_cols):
                print(f"  + {c}")
                
        # Check generated column status for time buckets
        print("\nTime Bucket Columns:")
        for col in ['month_year', 'day_of_week', 'hour_of_day']:
            if col in actual_columns:
                print(f"  {col}: Present")
            else:
                print(f"  {col}: MISSING")

    except Exception as e:
        print(f"Error inspecting database: {e}")
    finally:
        if 'con' in locals():
            con.close()

if __name__ == "__main__":
    verify_schema()

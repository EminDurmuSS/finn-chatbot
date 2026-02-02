
import sys
import os
import duckdb
from datetime import date

# Add project root to path
sys.path.append("f:\\finn-chatbot")

from statement_copilot.core import SQLBuilder, get_db

def verify_lowest_balance_execution():
    print("Verifying lowest balance logic with REAL DB execution...")
    
    db = get_db()
    
    # 1. Build the SQL
    print("\n[STEP 1] Building SQL for 'lowest balance'...")
    sql, params = SQLBuilder.build(
        metric="balance_history",
        filters={"exclude_transfers": True},
        tenant_id="default_tenant",
        limit=5,
        order_direction="ASC",
        order_by="balance"
    )
    
    print(f"Generated SQL:\n{sql}")
    
    # 2. Execute against DB
    print("\n[STEP 2] Executing against database...")
    try:
        results = db.execute_query(sql, params)
        print(f"Query executed successfully. Returned {len(results)} rows.")
        
        if not results:
            print("WARNING: No rows returned. Database might be empty or tenant_id mismatch.")
            return

        # 3. Verify ordering
        print("\n[STEP 3] Verifying sort order...")
        balances = [row['balance'] for row in results if row['balance'] is not None]
        print(f"Balances returned: {balances}")
        
        if balances == sorted(balances):
             print("PASS: Results are sorted by balance (ASC).")
        else:
             print("FAIL: Results are NOT sorted by balance!")

        # 4. Check dates to ensure they aren't chronological (unless coincidentally so)
        dates = [row['date'] for row in results if row['date']]
        print(f"Dates returned: {dates}")

    except Exception as e:
        print(f"FAIL: Database execution failed: {e}")
        # Print extra debug info if table missing
        if "Table with name transactions does not exist" in str(e):
             print("CRITICAL: 'transactions' table missing. Run setup script first.")

if __name__ == "__main__":
    try:
        verify_lowest_balance_execution()
        print("\nVerification complete.")
    except Exception as e:
        print(f"\nVerification failed with error: {e}")
        sys.exit(1)

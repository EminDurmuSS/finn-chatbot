
import duckdb
import pandas as pd
from pathlib import Path

# Connect to the database
db_path = Path("db/statement_copilot.duckdb")
con = duckdb.connect(str(db_path), read_only=True)

# 1. Check if we have 'expense' transactions with merchant_norm
print("Checking expense transactions...")
try:
    count = con.execute("SELECT COUNT(*) FROM transactions WHERE direction = 'expense' AND merchant_norm IS NOT NULL").fetchone()[0]
    print(f"Expense transactions with merchant_norm: {count}")
    
    if count == 0:
        print("WARNING: No expense transactions with merchant_norm found!")
        # Check if we have ANY transactions with merchant_norm
        total_merchants = con.execute("SELECT COUNT(*) FROM transactions WHERE merchant_norm IS NOT NULL").fetchone()[0]
        print(f"Total transactions with merchant_norm: {total_merchants}")
        # Check directions
        directions = con.execute("SELECT DISTINCT direction FROM transactions").fetchall()
        print(f"Available directions: {directions}")

    # 2. Look for potential subscriptions (Netflix, Spotify, etc.)
    print("\nChecking for common subscriptions (Netflix, Spotify, YouTube)...")
    common_subs = ['NETFLIX', 'SPOTIFY', 'YOUTUBE', 'APPLE', 'AMAZON PRIME', 'DISNEY']
    
    for sub in common_subs:
        rows = con.execute(f"SELECT date_time, amount, direction, merchant_norm FROM transactions WHERE merchant_norm LIKE '%{sub}%' ORDER BY date_time").fetchall()
        if rows:
            print(f"\nFound {sub}: {len(rows)} transactions")
            for row in rows:
                print(f"  {row[0]} | {row[1]} | {row[2]} | {row[3]}")
                
            # Calculate intervals if > 1
            if len(rows) > 1:
                dates = pd.to_datetime([r[0] for r in rows])
                for i in range(1, len(dates)):
                    diff = (dates[i] - dates[i-1]).days
                    print(f"    Interval {i}: {diff} days")

except Exception as e:
    print(f"Error: {e}")

finally:
    con.close()

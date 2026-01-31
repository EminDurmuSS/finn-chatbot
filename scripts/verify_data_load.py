
import duckdb
import os

DB_PATH = os.path.join(os.getcwd(), "db", "statement_copilot.duckdb")

if os.path.exists(DB_PATH):
    con = duckdb.connect(DB_PATH, read_only=True)
    count = con.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    print(f"Transaction count: {count}")
    
    # Check if merchant dictionary has data
    m_count = con.execute("SELECT COUNT(*) FROM merchant_dictionary").fetchone()[0]
    print(f"Merchant dictionary count: {m_count}")
else:
    print("Database not found.")

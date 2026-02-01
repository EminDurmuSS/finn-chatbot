
import duckdb
from pathlib import Path

DB_PATH = Path("f:/finn-chatbot/db/statement_copilot.duckdb")

if not DB_PATH.exists():
    print(f"ERROR: Database file not found at {DB_PATH}")
    exit(1)

print(f"Connecting to {DB_PATH}...")
con = duckdb.connect(str(DB_PATH), read_only=True)
print("Connected.")

expected_tables = [
    "raw_files",
    "transactions",
    "merchant_dictionary",
    "category_rules",
    "subscriptions",
    "budgets",
    "alerts",
    "actions",
    "tool_audit",
    "chat_sessions",
    "chat_messages"
]

print("Checking tables...")
existing_tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]

missing_tables = []
for index, table in enumerate(expected_tables):
    if table in existing_tables:
        count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"[OK] Table '{table}' exists. Rows: {count}")
    else:
        print(f"[MISSING] Table '{table}' does NOT exist.")
        missing_tables.append(table)

if missing_tables:
    print(f"\nMissing tables: {missing_tables}")
else:
    print("\nAll expected tables exist.")

con.close()


import duckdb
con = duckdb.connect('db/statement_copilot.duckdb', read_only=True)
res = con.execute("SELECT merchant_norm FROM transactions WHERE merchant_norm LIKE '%YOUTUBE%' LIMIT 5").fetchall()
print("Results:", res)

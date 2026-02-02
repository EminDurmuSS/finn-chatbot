
import duckdb
from pathlib import Path

# Path to the database
REPO_ROOT = Path("f:/finn-chatbot")
DB_PATH = REPO_ROOT / "db" / "statement_copilot.duckdb"
TENANT_ID = "default_tenant"

def verify_p2p_fix():
    if not DB_PATH.exists():
        print(f"Database not found at {DB_PATH}")
        return

    print(f"Connecting to {DB_PATH}...")
    try:
        con = duckdb.connect(str(DB_PATH), read_only=True)
        
        print("\n--- Verifying P2P Flow Metric ---")
        
        # This duplicates the logic we just added to database.py for testing
        query = """
            SELECT 
                CASE 
                    WHEN merchant_norm = 'P2P_TRANSFER' THEN split_part(description, '*', 1)
                    ELSE merchant_norm 
                END as counterparty,
                SUM(CASE WHEN direction = 'transfer' AND amount < 0 THEN ABS(amount) ELSE 0 END) as sent_total,
                SUM(CASE WHEN direction = 'transfer' AND amount > 0 THEN amount ELSE 0 END) as received_total,
                SUM(amount) as net_flow
            FROM transactions
            WHERE tenant_id = ?
            AND (category LIKE '%p2p%' OR category LIKE '%transfer%' OR description LIKE '%transfer%')
            GROUP BY 1
            ORDER BY ABS(net_flow) DESC
            LIMIT 20
        """
        
        params = [TENANT_ID]
        results = con.execute(query, params).fetchall()
        columns = [desc[0] for desc in con.description]
        
        if not results:
            print("No P2P transactions found.")
        else:
            print(f"Found {len(results)} distinct counterparties.")
            print(f"{'Counterparty':<30} | {'Sent':>15} | {'Received':>15} | {'Net Flow':>15}")
            print("-" * 85)
            for row in results:
                r = dict(zip(columns, row))
                print(f"{r['counterparty'][:30]:<30} | {r['sent_total']:>15.2f} | {r['received_total']:>15.2f} | {r['net_flow']:>15.2f}")
                    
    except Exception as e:
        print(f"Error querying database: {e}")
    finally:
        con.close()

if __name__ == "__main__":
    verify_p2p_fix()

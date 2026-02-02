
import duckdb
from pathlib import Path

# Path to the database
REPO_ROOT = Path("f:/finn-chatbot")
DB_PATH = REPO_ROOT / "db" / "statement_copilot.duckdb"

def inspect_p2p():
    if not DB_PATH.exists():
        print(f"Database not found at {DB_PATH}")
        return

    print(f"Connecting to {DB_PATH}...")
    try:
        con = duckdb.connect(str(DB_PATH), read_only=True)
        
        print("\n--- P2P Transaction Inspection ---")
        query = """
            SELECT 
                tx_id,
                date_time,
                amount,
                merchant_norm,
                description,
                category,
                subcategory,
                direction
            FROM transactions
            WHERE (category LIKE '%p2p%' OR category LIKE '%transfer%' OR description LIKE '%transfer%')
            LIMIT 10
        """
        
        results = con.execute(query).fetchall()
        columns = [desc[0] for desc in con.description]
        
        if not results:
            print("No P2P transactions found.")
        else:
            for row in results:
                print("\n--------------------------------")
                row_dict = dict(zip(columns, row))
                for k, v in row_dict.items():
                    print(f"{k}: {v}")
                    
    except Exception as e:
        print(f"Error querying database: {e}")
    finally:
        con.close()

if __name__ == "__main__":
    inspect_p2p()

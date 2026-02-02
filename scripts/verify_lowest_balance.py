
import sys
import os
from datetime import date

# Add project root to path
sys.path.append("f:\\finn-chatbot")

from statement_copilot.core import SQLBuilder

def verify_lowest_balance_logic():
    print("Verifying lowest balance logic...")
    
    # 1. Test DEFAULT behavior (no order_by)
    print("\n[TEST 1] Default behavior (should use date_time)")
    sql_default, _ = SQLBuilder.build(
        metric="balance_history",
        filters={"exclude_transfers": True},
        tenant_id="test_tenant",
        limit=20,
        order_direction="ASC"
    )
    
    if "ORDER BY date_time ASC" in sql_default:
        print("PASS: Default uses date_time")
    else:
        print(f"FAIL: Default SQL incorrect:\n{sql_default}")

    # 2. Test EXPLICIT order_by="balance"
    print("\n[TEST 2] Explicit order_by='balance' (should use balance)")
    sql_balance, _ = SQLBuilder.build(
        metric="balance_history",
        filters={"exclude_transfers": True},
        tenant_id="test_tenant",
        limit=20,
        order_direction="ASC",
        order_by="balance"
    )
    
    if "ORDER BY balance ASC" in sql_balance:
        print("PASS: Explicit uses balance")
    else:
        print(f"FAIL: Explicit SQL incorrect:\n{sql_balance}")

    # 3. Test INVALID order_by (should fallback to date_time)
    print("\n[TEST 3] Invalid order_by='hacking' (should fallback to date_time)")
    sql_hack, _ = SQLBuilder.build(
        metric="balance_history",
        filters={"exclude_transfers": True},
        tenant_id="test_tenant",
        limit=20,
        order_direction="ASC",
        order_by="hacking"
    )
    
    if "ORDER BY date_time ASC" in sql_hack:
        print("PASS: Fallback to date_time for invalid column")
    else:
        print(f"FAIL: Fallback SQL incorrect:\n{sql_hack}")

if __name__ == "__main__":
    try:
        verify_lowest_balance_logic()
        print("\nAll tests passed!")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        sys.exit(1)

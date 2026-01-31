
import sys
import os
from datetime import date, datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from statement_copilot.core.database import SQLBuilder
except ImportError:
    print("Could not import SQLBuilder. Check python path.")
    sys.exit(1)

def test_compatibility():
    print("Testing SQLBuilder compatibility...")
    
    # 1. Test YEAR_OVER_YEAR Metric
    try:
        sql, params = SQLBuilder.build(
            metric="year_over_year",
            filters={
                "date_start": date(2025, 1, 1), 
                "date_end": date(2025, 12, 31)
            },
            tenant_id="test_tenant"
        )
        print("[PASS] YEAR_OVER_YEAR SQL generated successfully.")
        
        # Simple string check to ensure key parts are present
        if "previous_period" in sql and "current_period" in sql:
             print("[PASS] SQL contains CTEs for comparison.")
        else:
             print("[FAIL] SQL missing CTEs.")
             
    except Exception as e:
        print(f"[FAIL] YEAR_OVER_YEAR generation failed: {e}")

    # 2. Test Allowed Columns (implicit check via build or just checking the set)
    required_columns = ["day_of_week", "hour_of_day", "tags_arr", "reference", "confidence", "source"]
    missing = [c for c in required_columns if c not in SQLBuilder.ALLOWED_COLUMNS]
    
    if not missing:
        print(f"[PASS] All required columns present: {required_columns}")
    else:
        print(f"[FAIL] Missing columns: {missing}")

    # 3. Test Allowed Group By
    required_group = ["day_of_week", "hour_of_day"]
    missing_group = [g for g in required_group if g not in SQLBuilder.ALLOWED_GROUP_BY]
    
    if not missing_group:
         print(f"[PASS] All required group_by fields present: {required_group}")
    else:
         print(f"[FAIL] Missing group_by fields: {missing_group}")

if __name__ == "__main__":
    test_compatibility()

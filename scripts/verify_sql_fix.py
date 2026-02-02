
import sys
import os
from datetime import date, datetime

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from statement_copilot.core.database import SQLBuilder

def test_trended_breakdown():
    print("Testing SQLBuilder.build for trended category breakdown...")
    
    tenant_id = "default_tenant"
    metric = "category_breakdown"
    time_grain = "year"
    
    filters = {
        "categories": ["food_and_dining"],
        "subcategories": ["delivery"],
        "direction": "expense",
        "date_start": date(2023, 1, 1),
        "date_end": date(2025, 12, 31),
        "time_grain": "year", # This is what triggers the fix
    }
    
    try:
        sql, params = SQLBuilder.build(
            metric=metric,
            filters=filters,
            tenant_id=tenant_id,
            limit=50,
            time_grain=time_grain
        )
        
        print("\nGenerated SQL:")
        print(sql)
        print("\nParams:", params)
        
        # assertions
        sql_lower = sql.lower()
        if "date_trunc('year', date_time)" in sql_lower and "group by" in sql_lower:
             # Check if it groups by both date and category
             if "coalesce(category_final" in sql_lower:
                 print("\nSUCCESS: SQL includes time grouping and category grouping!")
             else:
                 print("\nFAILURE: Category grouping missing?")
        else:
            print("\nFAILURE: SQL does not look like a trended breakdown.")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_trended_breakdown()

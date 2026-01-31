"""
Quick test to verify enhanced SQL debug logging
"""
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging to INFO level to see SQL debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s | %(name)s | %(message)s',
    stream=sys.stdout
)

from statement_copilot.core import get_db, SQLBuilder
from datetime import date, datetime

def test_sql_logging():
    """Test enhanced SQL logging with a simple query"""
    print("\n" + "="*80)
    print("TESTING ENHANCED SQL DEBUG LOGGING")
    print("="*80 + "\n")
    
    db = get_db()
    
    # Example filters
    filters = {
        "date_start": date(2024, 1, 1),
        "date_end": date(2024, 1, 31),
        "direction": "expense",
        "merchants": ["YOUTUBE"],
        "categories": ["Entertainment"],
        "keyword_search": ["subscription"]
    }
    
    print("\n📋 TEST: Building SQL with multiple filters...")
    print(f"Filters: {filters}\n")
    
    # Build SQL - this will show detailed logs
    sql, params = SQLBuilder.build(
        metric="sum_amount",
        filters=filters,
        tenant_id="test_tenant_001",
        limit=50,
        order_direction="DESC"
    )
    
    print("\n✅ SQL Building Complete!")
    print("\n" + "="*80)
    print("Check the logs above to see:")
    print("  ✓ Filter breakdown with normalization")
    print("  ✓ Full SQL query")
    print("  ✓ All parameters with types")
    print("="*80 + "\n")

if __name__ == "__main__":
    test_sql_logging()

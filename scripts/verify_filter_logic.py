
import sys
import logging
from typing import Dict, Any, Optional

# Mock logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def mock_build_filters(entities, tenant_id, intent):
    return {"date_start": "2023-01-01", "category": "Food"}

def retrieve(overrides: Optional[Dict[str, Any]] = None):
    # This simulates the logic we just added to search_engine.py
    filters = mock_build_filters(None, "default", None)
    print(f"Base filters: {filters}")
    
    if overrides:
        for k, v in overrides.items():
            if v is None:
                filters.pop(k, None)
            else:
                filters[k] = v
        print(f"Effective filters: {filters}")
    return filters

if __name__ == "__main__":
    print("--- Test 1: No overrides ---")
    retrieve(None)
    
    print("\n--- Test 2: Override category ---")
    f = retrieve({"category": "Travel"})
    assert f["category"] == "Travel"
    
    print("\n--- Test 3: Remove date (Relaxation) ---")
    f = retrieve({"date_start": None})
    assert "date_start" not in f
    
    print("\n--- Test 4: Add new filter ---")
    f = retrieve({"min_amount": 100})
    assert f["min_amount"] == 100
    assert f["category"] == "Food" # Original remains
    
    print("\nSUCCESS: Filter override logic works as expected.")

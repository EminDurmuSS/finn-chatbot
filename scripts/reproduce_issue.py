
import sys
import os
from datetime import date
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from statement_copilot.agents.finance_analyst import FinanceAnalystAgent
from statement_copilot.core.schemas import MetricRequest, MetricType, MetricFilters, Direction, TimeGrain

def run_test():
    print("Initializing Finance Analyst...")
    try:
        agent = FinanceAnalystAgent()
    except Exception as e:
        print(f"Failed to init agent: {e}")
        return
    
    # Simulate the state that caused the issue (or the new request)
    user_message = "Compare my 2023, 2024, and 2025 delivery spending"
    
    # Constraints as seen in the logs (or expected)
    constraints = {
        "categories": ["food_and_dining"],
        "subcategories": ["delivery"],
        "direction": "expense",
        "time_grain": "year",
        "date_range": {"start": "2023-01-01", "end": "2025-12-31"} 
    }
    
    state = {
        "user_message": user_message,
        "constraints": constraints,
        "tenant_id": "default_tenant"
    }
    
    print(f"\nRunning analysis for: '{user_message}'")
    print(f"Constraints: {constraints}")
    
    try:
        result_state = agent.analyze(state)
        
        if "sql_error" in result_state:
            print(f"\nERROR: {result_state['sql_error']}")
            if "errors" in result_state:
                print(f"Errors list: {result_state['errors']}")
            return
            
        sql_result = result_state.get("sql_result", {})
        rows = sql_result.get("rows", [])
        
        print(f"\nMetric: {sql_result.get('metric')}")
        print(f"Transaction Count: {sql_result.get('tx_count')}")
        print(f"Generated SQL Preview: \n{sql_result.get('sql_preview')}")
        
        print(f"\nRows returned ({len(rows)}):")
        for row in rows:
            print(row)
            
    except Exception as e:
        print(f"Execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()

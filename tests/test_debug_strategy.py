"""
Debug Smart Limiting - Check Strategy Selection
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['PYTHONIOENCODING'] = 'utf-8'

import logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

from statement_copilot.config import settings
from statement_copilot.core.database import get_db
from statement_copilot.core.vector_store import get_vector_store
from statement_copilot.core.search_engine import (
    ProfessionalSearchEngine, 
    QueryUnderstandingEngine,
    SearchStrategy
)

def main():
    print("=" * 60)
    print("DEBUG: Strategy Selection")
    print("=" * 60)
    
    # Initialize query understanding
    query_engine = QueryUnderstandingEngine()
    
    queries = [
        "Green Chef transactions",
        "all transactions", 
        "tüm YouTube harcamalarım",
        "Show all my Netflix payments",
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        understanding = query_engine.understand(query)
        print(f"  Intent: {understanding.intent.value}")
        print(f"  Strategy: {understanding.strategy.value}")
        print(f"  Merchants: {understanding.entities.merchants}")
        print(f"  Categories: {understanding.entities.categories}")
        
        # Check if strategy uses count
        uses_count = understanding.strategy in (
            SearchStrategy.SQL_ONLY, 
            SearchStrategy.EXACT_MATCH, 
            SearchStrategy.HYBRID
        )
        print(f"  Uses COUNT: {uses_count}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()

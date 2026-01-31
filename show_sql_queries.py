"""
Show SQL queries generated for description-based searches
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from statement_copilot.core.search_engine import QueryUnderstandingEngine, MultiSourceRetriever
from statement_copilot.core.database import DatabaseManager
from statement_copilot.core.vector_store import MockVectorStore
from statement_copilot.core.llm import get_llm_client
from statement_copilot.agents.search_agent import load_taxonomy
from statement_copilot.config import settings
import logging

logging.basicConfig(level=logging.WARNING)  # Suppress info logs

def show_sql_queries():
    """Show SQL queries for test cases"""
    
    db = DatabaseManager()
    vector_store = MockVectorStore()
    llm = get_llm_client()
    taxonomy = load_taxonomy()
    
    qe = QueryUnderstandingEngine(llm, taxonomy)
    retriever = MultiSourceRetriever(db, vector_store)
    
    test_queries = [
        "Ev kirası olarak ödediğim bir iban varmı",
        "kira ödemesi",
        "iban transfer",
        "subscription payment",
        "fatura ödeme",
    ]
    
    for query in test_queries:
        print("\n" + "=" * 80)
        print(f"Query: {query}")
        print("=" * 80)
        
        # Understand query
        understanding = qe.understand(query)
        
        print(f"\n[Extracted Entities]")
        print(f"  Merchants: {understanding.entities.merchants}")
        print(f"  Categories: {understanding.entities.categories}")
        print(f"  Content Keywords: {understanding.entities.content_keywords}")
        
        # Build filters
        filters = retriever._build_filters(understanding.entities, settings.default_tenant_id)
        
        print(f"\n[Filters Built]")
        for key, value in filters.items():
            print(f"  {key}: {value}")
        
        # Get SQL query (retrieve but show query)
        matches, sql = retriever._retrieve_sql(
            filters=filters,
            tenant_id=settings.default_tenant_id,
            limit=10,
            normalize_for_hybrid=True
        )
        
        print(f"\n[SQL Query Generated]")
        print(sql)
        
        print(f"\n[Results]")
        print(f"  Total Matches: {len(matches)}")
        if matches:
            for i, m in enumerate(matches[:3], 1):
                print(f"  {i}. {m.merchant_norm} - ${abs(m.amount or 0):.2f}")
                if m.description:
                    print(f"     Description: {m.description[:80]}...")


if __name__ == "__main__":
    show_sql_queries()

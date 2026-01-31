"""
Test Description-Based Search Implementation

Tests the new content_keywords extraction and keyword_search filter
to ensure queries like "Ev kirası" (house rent) can find transactions
based on description content.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from statement_copilot.core.database import DatabaseManager
from statement_copilot.core.search_engine import ProfessionalSearchEngine, QueryUnderstandingEngine
from statement_copilot.core.vector_store import MockVectorStore
from statement_copilot.core.llm import get_llm_client
from statement_copilot.agents.search_agent import load_taxonomy
from statement_copilot.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_description_search():
    """Test description-based keyword search"""
    
    print("=" * 80)
    print("TEST: Description-Based Search")
    print("=" * 80)
    
    # Initialize components
    db = DatabaseManager()
    vector_store = MockVectorStore()
    llm = get_llm_client()
    taxonomy = load_taxonomy()
    
    # Create search engine
    engine = ProfessionalSearchEngine(
        db=db,
        vector_store=vector_store,
        llm_client=llm,
        taxonomy=taxonomy
    )
    
    # Test queries
    test_queries = [
        "Ev kirası olarak ödediğim bir iban varmı",
        "kira ödemesi",
        "iban transfer",
        "subscription payment",
        "fatura ödeme",
    ]
    
    for query in test_queries:
        print(f"\n{'=' * 80}")
        print(f"Query: {query}")
        print(f"{'=' * 80}")
        
        # Test query understanding
        qe = QueryUnderstandingEngine(llm, taxonomy)
        understanding = qe.understand(query)
        
        print(f"\n[Query Understanding]")
        print(f"  Intent: {understanding.intent.value}")
        print(f"  Strategy: {understanding.strategy.value}")
        print(f"  Confidence: {understanding.confidence:.2f}")
        print(f"  Merchants: {understanding.entities.merchants}")
        print(f"  Categories: {understanding.entities.categories}")
        print(f"  Keywords: {understanding.entities.keywords}")
        print(f"  Content Keywords: {understanding.entities.content_keywords}")
        
        # Execute search
        result = engine.search(
            query=query,
            tenant_id=settings.default_tenant_id,
            top_k=5,
            use_llm_rerank=False
        )
        
        print(f"\n[Search Results]")
        print(f"  Total Found: {result.total_found}")
        print(f"  Sources Used: {result.sources_used}")
        print(f"  Search Time: {result.search_time_ms}ms")
        
        if result.sql_query:
            print(f"\n[SQL Query]")
            # Simplify output - just show WHERE clause
            if "WHERE" in result.sql_query:
                where_part = result.sql_query.split("WHERE")[1].split("ORDER BY")[0]
                print(f"  WHERE {where_part.strip()[:200]}...")
        
        print(f"\n[Top Matches]")
        for i, match in enumerate(result.matches[:3], 1):
            print(f"  {i}. {match.merchant_norm or 'Unknown'} - ${abs(match.amount or 0):.2f}")
            print(f"     Category: {match.category or 'N/A'}")
            if match.description:
                desc = match.description[:100] + "..." if len(match.description) > 100 else match.description
                print(f"     Description: {desc}")
            print(f"     Score: {match.score:.3f} | Source: {match.source}")
            print(f"     Reason: {match.match_reason}")
    
    print(f"\n{'=' * 80}")
    print("Test Complete!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    test_description_search()


import logging
import time
from datetime import datetime
from typing import List

from statement_copilot.core.search_engine import ProfessionalSearchEngine, SearchMatch, SearchIntent, SearchStrategy, QueryUnderstanding, RerankedResults
from statement_copilot.core.llm import get_llm_client
from statement_copilot.config import settings
from statement_copilot.core.prompts import get_result_reranking_prompt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def verify_reranker():
    print("="*60)
    print("VERIFYING RERANKER QUALITY & SPEED (Model: {})".format(settings.model_reranker))
    print("="*60)

    # 1. Setup Mock Data
    query = "Have I ever paid for YouTube?"
    understanding = QueryUnderstanding(
        intent=SearchIntent.MERCHANT_LOOKUP,
        strategy=SearchStrategy.HYBRID,
        confidence=0.9,
        original_query=query,
        normalized_query=query.lower(),
        entities={},
        expanded_query=query,
        search_terms=[],
        reasoning="Test reasoning"
    )
    
    # Mix of relevant and irrelevant transactions
    matches = [
        SearchMatch(tx_id="tx_1", score=0.5, merchant_norm="YOUTUBE PREMIUM", amount=-59.99, date_time=datetime(2025, 1, 15), description="Google *YouTube Premium", category="utilities", source="test"),
        SearchMatch(tx_id="tx_2", score=0.5, merchant_norm="NETFLIX", amount=-100.00, date_time=datetime(2025, 1, 10), description="Netflix Subscription", category="entertainment", source="test"),
        SearchMatch(tx_id="tx_3", score=0.5, merchant_norm="Migros", amount=-250.00, date_time=datetime(2025, 1, 5), description="Migros Sanal Market", category="groceries", source="test"),
        SearchMatch(tx_id="tx_4", score=0.5, merchant_norm="GOOGLE YOUTUBE MEMBER", amount=-30.00, date_time=datetime(2024, 12, 20), description="YouTube Membership", category="entertainment", source="test"),
        SearchMatch(tx_id="tx_5", score=0.5, merchant_norm="Spotify", amount=-40.00, date_time=datetime(2025, 1, 1), description="Spotify Premium", category="music", source="test"),
    ]

    # 2. Initialize Search Engine with real LLM
    llm = get_llm_client()
    engine = ProfessionalSearchEngine(db=None, vector_store=None, llm_client=llm)
    
    # 3. Time the Reranking
    print("\n[Running Reranker...]")
    start_time = time.time()
    
    # Call the internal _llm_rerank method
    # Note: We need to bypass the method's dependency on self.llm if it's not set in init, 
    # but ProfessionalSearchEngine initializes self.llm in __init__ using get_llm_client()
    
    try:
        # Access the reranker instance attached to the engine
        reranked_matches = engine.reranker._llm_rerank(query, matches, understanding)
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Reranking finished in {duration:.2f} seconds")
        
        # 4. Analyze Results
        print("\n[Results]")
        print(f"{'TX_ID':<10} | {'MERCHANT':<25} | {'SCORE':<10} | {'EXPLANATION'}")
        print("-" * 80)
        
        relevant_found = False
        irrelevant_filtered = True
        
        for m in reranked_matches:
            print(f"{m.tx_id:<10} | {m.merchant_norm:<25} | {m.score:<10.4f} | {m.highlight_terms}") # Using highlight_terms slot for explanation/debug if needed, but score is blended
            
            # Simple assertions based on expected behavior
            if "YOUTUBE" in m.merchant_norm:
                if m.score < 0.7:
                    print(f"❌ FAIL: Relevant item {m.merchant_norm} got low score: {m.score}")
                    relevant_found = False
                else:
                    relevant_found = True
            else:
                if m.score > 0.6: # Allow some leeway, but shouldn't be high
                    print(f"⚠️ WARNING: Irrelevant item {m.merchant_norm} got high score: {m.score}")
                    irrelevant_filtered = False

        # 5. Speed Assertion
        if duration > 5.0:
            print(f"\n❌ FAIL: Reranking took too long ({duration:.2f}s > 5.0s)")
        else:
            print(f"\n✅ PASS: Speed check passed ({duration:.2f}s)")

        # 6. Quality Assertion
        if relevant_found: 
             print("✅ PASS: Quality check passed (Relevant items boosted)")
        else:
             print("❌ FAIL: Quality check failed (Relevant items NOT boosted)")

    except Exception as e:
        print(f"\n❌ ERROR during test execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_reranker()

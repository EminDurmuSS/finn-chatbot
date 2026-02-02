import unittest
from unittest.mock import MagicMock, ANY
import sys
import os
from datetime import datetime

# --- MOCK DEPENDENCIES TO RUN IN ISOLATION ---
# This prevents ModuleNotFoundError for dependencies not installed in the test env
sys.modules["duckdb"] = MagicMock()
sys.modules["pinecone"] = MagicMock()
sys.modules["pinecone.grpc"] = MagicMock()
sys.modules["pinecone_text"] = MagicMock()
sys.modules["pinecone_text.sparse"] = MagicMock()
sys.modules["pinecone_text.hybrid"] = MagicMock()
sys.modules["openai"] = MagicMock()

# Adjust path to find the module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from statement_copilot.core.search_engine import MultiSourceRetriever, QueryUnderstanding, ExtractedEntities, SearchStrategy, SearchIntent

class TestSearchIntegration(unittest.TestCase):
    def setUp(self):
        self.mock_db = MagicMock()
        self.mock_vector_store = MagicMock()
        self.retriever = MultiSourceRetriever(self.mock_db, self.mock_vector_store)
        
        # Setup common understanding object
        self.understanding = QueryUnderstanding(
            original_query="test",
            normalized_query="test",
            intent=SearchIntent.FIND_SPECIFIC,
            confidence=1.0,
            entities=ExtractedEntities(),
            strategy=SearchStrategy.HYBRID,
            expanded_query="test expanded",
            search_terms=["test"],
            reasoning="test"
        )

    def test_top_k_doubling(self):
        """Verify that retrieve() requests 2x top_k from underlying sources"""
        print("\n--- Test: Top-K Doubling ---")
        
        # Mock returns
        self.mock_db.execute_query.return_value = [] # No SQL results
        self.mock_vector_store.search.return_value = [] # No Vector results
        
        # Call retrieve with top_k=10
        self.retriever.retrieve(
            understanding=self.understanding,
            tenant_id="test_tenant",
            top_k=10
        )
        
        # Check Vector Store call
        # _retrieve_vector calls vector_store.search
        # effective_vector_limit should be 20
        
        self.assertTrue(self.mock_vector_store.search.called)
        
        call_kwargs = self.mock_vector_store.search.call_args.kwargs
        cutoff = call_kwargs.get('top_k')
        
        print(f"Requested Top-K: 10 -> Actual Vector Limit: {cutoff}")
        self.assertEqual(cutoff, 20, "Vector search should request 2x top_k")

    def test_vector_relaxation_integration(self):
        """Verify that retrieve triggers vector relaxation when initial search is empty"""
        print("\n--- Test: Vector Relaxation Integration ---")
        
        # Setup: Query with Category
        self.understanding.entities.categories = ["Food"]
        # Must be LIST_FILTER to trigger "hard" category filter in _build_filters
        self.understanding.intent = SearchIntent.LIST_FILTER
        
        # Mock SQL to return 1 result so independent SQL relaxation is NOT triggered
        # This forces the Vector logic to handle its own relaxation if needed
        # We need a robust side_effect to handle both COUNT and SELECT queries
        mock_sql_row = {
            "tx_id": "sql1",
            "date_time": datetime(2023, 1, 1),
            "amount": 50.0,
            "merchant_norm": "TEST",
            "description": "Test SQL",
            "category": "Food",
            "subcategory": "Groceries",
            "direction": "expense"
        }
        
        def db_side_effect(sql, params):
            # Check if this is the count query
            if "COUNT" in sql.upper():
                return [{"cnt": 5}]
            return [mock_sql_row]
            
        self.mock_db.execute_query.side_effect = db_side_effect
        
        # Mock Vector Store to return:
        # 1. Empty list (Strict search)
        # 2. List with 1 match (Relaxed search)
        
        mock_vec_match = {"tx_id": "vec1", "score": 0.9, "metadata": {"category": "Food", "description": "Burger"}}
        self.mock_vector_store.search.side_effect = [
            [],             # First call (Strict) -> Empty
            [mock_vec_match]    # Second call (Relaxed) -> Match
        ]
        
        # Call retrieve
        matches, metadata = self.retriever.retrieve(
            understanding=self.understanding,
            tenant_id="test_tenant",
            top_k=10
        )
        
        # Verify result count: 1 SQL + 1 Vector = 2
        print(f"Matches found: {len(matches)}")
        self.assertEqual(len(matches), 2)
        
        # Verify metadata indicates relaxation
        # Note: Vector relaxation updates metadata['filter_relaxed']
        print(f"Filter Relaxed: {metadata.get('filter_relaxed')}")
        self.assertTrue(metadata.get("filter_relaxed"), "Metadata should indicate filter relaxation")
        self.assertEqual(metadata.get("relaxation_type"), "vector_category_removed")
        
        # Verify 2 calls to vector store
        self.assertEqual(self.mock_vector_store.search.call_count, 2)

if __name__ == '__main__':
    unittest.main()

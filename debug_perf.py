"""
Performance Debug Script - Vector Search Bottleneck Analysis
"""
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set encoding for Windows
os.environ['PYTHONIOENCODING'] = 'utf-8'

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def time_it(name, func, *args, **kwargs):
    """Helper to time a function call"""
    start = time.time()
    result = func(*args, **kwargs)
    elapsed = (time.time() - start) * 1000
    print(f"  {name}: {elapsed:.0f}ms")
    return result, elapsed

def main():
    print("=" * 70)
    print("PERFORMANCE DEBUG - Vector Search Bottleneck Analysis")
    print("=" * 70)
    
    total_times = {}
    
    # Step 1: Settings load
    print("\n[1] Loading settings...")
    start = time.time()
    from statement_copilot.config import settings
    total_times['settings'] = (time.time() - start) * 1000
    print(f"  Settings load: {total_times['settings']:.0f}ms")
    
    # Step 2: BM25 Encoder initialization (OFTEN SLOW)
    print("\n[2] Loading BM25 Encoder (often slow)...")
    start = time.time()
    from statement_copilot.core.embeddings import get_bm25_encoder
    bm25 = get_bm25_encoder()
    total_times['bm25_init'] = (time.time() - start) * 1000
    print(f"  BM25 Encoder init: {total_times['bm25_init']:.0f}ms")
    
    # Step 3: Embeddings client init
    print("\n[3] Loading Embeddings Client...")
    start = time.time()
    from statement_copilot.core.embeddings import get_embeddings_client
    embeddings = get_embeddings_client()
    total_times['embeddings_init'] = (time.time() - start) * 1000
    print(f"  Embeddings Client init: {total_times['embeddings_init']:.0f}ms")
    
    # Step 4: Database connection
    print("\n[4] Getting database connection...")
    start = time.time()
    from statement_copilot.core.database import get_db
    db = get_db()
    total_times['db_init'] = (time.time() - start) * 1000
    print(f"  Database init: {total_times['db_init']:.0f}ms")
    
    # Step 5: Vector store init
    print("\n[5] Getting vector store...")
    start = time.time()
    from statement_copilot.core.vector_store import get_vector_store
    vector_store = get_vector_store()
    total_times['vector_store_init'] = (time.time() - start) * 1000
    print(f"  Vector Store init: {total_times['vector_store_init']:.0f}ms")
    
    # Step 6: Search engine init
    print("\n[6] Initializing Search Engine...")
    start = time.time()
    from statement_copilot.core.search_engine import ProfessionalSearchEngine
    engine = ProfessionalSearchEngine(db=db, vector_store=vector_store)
    total_times['search_engine_init'] = (time.time() - start) * 1000
    print(f"  Search Engine init: {total_times['search_engine_init']:.0f}ms")
    
    tenant_id = settings.default_tenant_id
    
    # Step 7: Embedding generation timing
    print("\n[7] Testing embedding generation speed...")
    test_query = "YouTube payment"
    
    start = time.time()
    dense = embeddings.embed(test_query)
    total_times['embed_dense'] = (time.time() - start) * 1000
    print(f"  Dense embedding (API call): {total_times['embed_dense']:.0f}ms")
    
    start = time.time()
    sparse = bm25.encode(test_query)
    total_times['embed_sparse'] = (time.time() - start) * 1000
    print(f"  Sparse BM25 encoding: {total_times['embed_sparse']:.0f}ms")
    
    # Step 8: Pure SQL query
    print("\n[8] Testing pure SQL query...")
    start = time.time()
    rows = db.execute_query(
        "SELECT * FROM transactions WHERE tenant_id = ? LIMIT 20",
        [tenant_id]
    )
    total_times['sql_query'] = (time.time() - start) * 1000
    print(f"  SQL query (20 rows): {total_times['sql_query']:.0f}ms, got {len(rows)} rows")
    
    # Step 9: Vector search (if Pinecone available)
    print("\n[9] Testing vector search...")
    start = time.time()
    try:
        vector_results = vector_store.search(test_query, top_k=10)
        total_times['vector_search'] = (time.time() - start) * 1000
        print(f"  Vector search: {total_times['vector_search']:.0f}ms, got {len(vector_results)} results")
    except Exception as e:
        total_times['vector_search'] = (time.time() - start) * 1000
        print(f"  Vector search FAILED: {e} (took {total_times['vector_search']:.0f}ms)")
    
    # Step 10: Full search
    print("\n[10] Testing full search (ProfessionalSearchEngine)...")
    start = time.time()
    result = engine.search("YouTube ödemeleri", tenant_id, top_k=20)
    total_times['full_search'] = (time.time() - start) * 1000
    print(f"  Full search: {total_times['full_search']:.0f}ms, got {len(result.matches)} matches")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - Time Breakdown")
    print("=" * 70)
    
    total = sum(total_times.values())
    sorted_times = sorted(total_times.items(), key=lambda x: x[1], reverse=True)
    
    for name, ms in sorted_times:
        pct = (ms / total) * 100 if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {name:25s}: {ms:>6.0f}ms ({pct:>5.1f}%) {bar}")
    
    print(f"\n  {'TOTAL':25s}: {total:>6.0f}ms")
    
    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    if total_times['bm25_init'] > 2000:
        print(f"  ⚠️  BM25 init is slow ({total_times['bm25_init']:.0f}ms) - Consider lazy loading or caching")
    
    if total_times.get('embed_dense', 0) > 500:
        print(f"  ⚠️  Dense embedding API call is slow ({total_times['embed_dense']:.0f}ms) - Network latency issue")
    
    if total_times.get('vector_search', 0) > 1000:
        print(f"  ⚠️  Vector search is slow ({total_times['vector_search']:.0f}ms) - Consider Pinecone region or caching")
    
    print("\nDone!")

if __name__ == "__main__":
    main()

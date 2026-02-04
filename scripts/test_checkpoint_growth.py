"""
Checkpoint Growth Test
======================
Tests checkpoint file growth with sample queries.
"""

import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_file_size_mb(path):
    """Get file size in MB"""
    if os.path.exists(path):
        return os.path.getsize(path) / (1024 * 1024)
    return 0

def main():
    from statement_copilot.workflow import StatementCopilot
    
    # Checkpoint file path
    checkpoint_path = "db/checkpoints.sqlite"
    wal_path = "db/checkpoints.sqlite-wal"
    
    # Clear existing checkpoint
    for p in [checkpoint_path, wal_path, "db/checkpoints.sqlite-shm"]:
        if os.path.exists(p):
            os.remove(p)
            print(f"✓ Deleted: {p}")
    
    print("\n" + "="*60)
    print("CHECKPOINT GROWTH TEST")
    print("="*60)
    
    # Initialize copilot (creates new checkpoint file)
    copilot = StatementCopilot()
    session_id = "test-checkpoint-growth"
    
    # Test queries
    queries = [
        "YouTube üyeliğim var mı?",
        "Bu ay toplam ne kadar harcadım?",
        "En büyük 10 harcamam nedir?",
        "Spotify aboneliğim ne kadar?",
    ]
    
    print(f"\nInitial checkpoint size: {get_file_size_mb(checkpoint_path):.2f} MB")
    print(f"Initial WAL size: {get_file_size_mb(wal_path):.2f} MB")
    print("-"*60)
    
    for i, query in enumerate(queries, 1):
        print(f"\n[Query {i}] {query}")
        start = time.time()
        
        result = copilot.chat(
            message=query,
            session_id=session_id,
        )
        
        elapsed = time.time() - start
        answer = result.get("answer", "")[:100]
        
        # Check sizes
        cp_size = get_file_size_mb(checkpoint_path)
        wal_size = get_file_size_mb(wal_path)
        total_size = cp_size + wal_size
        
        print(f"  → Answer: {answer}...")
        print(f"  → Time: {elapsed:.1f}s")
        print(f"  → Checkpoint: {cp_size:.2f} MB | WAL: {wal_size:.2f} MB | Total: {total_size:.2f} MB")
        print("-"*60)
    
    # Final summary
    final_cp = get_file_size_mb(checkpoint_path)
    final_wal = get_file_size_mb(wal_path)
    final_total = final_cp + final_wal
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Queries sent: {len(queries)}")
    print(f"Checkpoint file: {final_cp:.2f} MB")
    print(f"WAL file: {final_wal:.2f} MB")
    print(f"TOTAL: {final_total:.2f} MB")
    print(f"Average per query: {final_total / len(queries):.2f} MB")
    
    if final_total > 10:
        print(f"\n⚠️  WARNING: {final_total:.1f} MB for just {len(queries)} queries!")
        print("   Projected for 10 queries: ~{:.1f} MB".format(final_total / len(queries) * 10))
        print("   Projected for 100 queries: ~{:.1f} MB".format(final_total / len(queries) * 100))

if __name__ == "__main__":
    main()

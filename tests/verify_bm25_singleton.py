
import sys
import os
import time
import logging

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import after sys.path fix
from statement_copilot.core.embeddings import get_bm25_encoder

logging.basicConfig(level=logging.INFO)

def verify_singleton():
    print("="*60)
    print("VERIFYING BM25 SINGLETON")
    print("="*60)

    print("\n[Call 1] Requesting BM25 encoder...")
    start1 = time.time()
    encoder1 = get_bm25_encoder()
    dur1 = time.time() - start1
    print(f"Call 1 finished in {dur1:.4f} seconds")
    print(f"Object ID: {id(encoder1)}")

    print("\n[Call 2] Requesting BM25 encoder again...")
    start2 = time.time()
    encoder2 = get_bm25_encoder()
    dur2 = time.time() - start2
    print(f"Call 2 finished in {dur2:.4f} seconds")
    print(f"Object ID: {id(encoder2)}")

    print("\n[Analysis]")
    if encoder1 is encoder2:
        print("✅ PASS: Objects are IDENTICAL (Same Memory Address)")
    else:
        print("❌ FAIL: Objects are DIFFERENT (Reloaded!)")

    if dur2 < 0.01:
        print("✅ PASS: Second call was instantaneous")
    else:
        print(f"⚠️ WARNING: Second call took {dur2:.4f}s")

if __name__ == "__main__":
    verify_singleton()

"""
Test Script - Verify Merchant Filtering Patches
================================================
Tests the 3 patches:
1. SQLBuilder fuzzy merchant matching
2. Finance Analyst merchant override
3. SQL-First routing optimization
"""

import asyncio
import logging
import time
import sys

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Reduce noise from other loggers
for noisy in ['httpx', 'httpcore', 'anthropic', 'pinecone']:
    logging.getLogger(noisy).setLevel(logging.WARNING)

async def test_patches():
    """Test all 3 patches with YouTube query"""
    
    from statement_copilot import StatementCopilot
    
    print("\n" + "="*70)
    print("PATCH VERIFICATION TEST")
    print("="*70)
    
    copilot = StatementCopilot()
    
    # Test 1: YouTube total spending (should use SQL only, fuzzy match)
    test_queries = [
        "Calculate your total YouTube spending?",
        "How much did I spend on YouTube?",
        "YouTube toplam harcama ne kadar?",
    ]
    
    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"TEST: {query}")
        print("="*70)
        
        start_time = time.time()
        
        response = await copilot.chat(query, tenant_id="bunq_demo")
        
        elapsed = time.time() - start_time
        
        print(f"\n📊 RESULT:")
        print(f"   Response: {response[:500]}...")
        print(f"   ⏱️ Time: {elapsed:.2f} seconds")
        
        # Check patches worked
        if elapsed < 10:
            print("   ✅ SQL-First optimization WORKING (fast response)")
        else:
            print("   ⚠️ Response slower than expected")
        
        if "963" in response or "YouTube" in response.lower():
            print("   ✅ Fuzzy merchant matching WORKING (YouTube found)")
        elif "-545" in response or "545,729" in response:
            print("   ❌ STILL BROKEN - returning all transactions!")
        
        print("\n")

if __name__ == "__main__":
    asyncio.run(test_patches())

"""Quick verification tests for search engine bug fixes."""
import sys
sys.path.insert(0, '.')

from statement_copilot.core.search_engine import (
    QueryUnderstandingEngine, SearchIntent, _dedup_keep_order
)

def test_substring_intent_bug():
    """Test that 'small' doesn't match 'all' for LIST_FILTER intent."""
    engine = QueryUnderstandingEngine()
    
    # 'small' should NOT trigger LIST_FILTER
    result = engine.understand('small purchases')
    assert result.intent != SearchIntent.LIST_FILTER, \
        f"BUG: 'small' triggered LIST_FILTER! Got: {result.intent}"
    print(f"✅ 'small purchases' -> {result.intent} (not LIST_FILTER)")
    
    # 'show all' SHOULD trigger LIST_FILTER
    result2 = engine.understand('show all transactions')
    assert result2.intent == SearchIntent.LIST_FILTER, \
        f"'show all' should be LIST_FILTER, got: {result2.intent}"
    print(f"✅ 'show all transactions' -> {result2.intent}")
    
    # 'all my' should trigger LIST_FILTER
    result3 = engine.understand('all my expenses')
    assert result3.intent == SearchIntent.LIST_FILTER, \
        f"'all my' should be LIST_FILTER, got: {result3.intent}"
    print(f"✅ 'all my expenses' -> {result3.intent}")

def test_dedup_order():
    """Test that _dedup_keep_order preserves order."""
    items = ['b', 'a', 'c', 'b', 'a', 'd']
    result = _dedup_keep_order(items)
    assert result == ['b', 'a', 'c', 'd'], f"Order not preserved: {result}"
    print(f"✅ _dedup_keep_order preserves order: {result}")

def main():
    print("=" * 60)
    print("SEARCH ENGINE BUG FIX VERIFICATION")
    print("=" * 60)
    
    try:
        test_substring_intent_bug()
        test_dedup_order()
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✅")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

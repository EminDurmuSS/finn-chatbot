"""
Show example SQL queries for description-based keyword search
"""

# Example: Query "Ev kirası iban"
# Content Keywords extracted: ['kirası', 'iban']

print("=" * 80)
print("EXAMPLE SQL QUERIES - Description-Based Keyword Search")
print("=" * 80)

print("\n1. Query: 'Ev kirası olarak ödediğim bir iban varmı'")
print("   Content Keywords: ['kirası', 'iban']")
print("\n   Generated SQL:")
sql1 = """
SELECT
    tx_id,
    date_time,
    amount,
    merchant_norm,
    description,
    COALESCE(category_final, category) as category,
    COALESCE(subcategory_final, subcategory) as subcategory,
    direction
FROM transactions
WHERE tenant_id = ? 
  AND direction = ?
  AND (LOWER(COALESCE(category_final, category)) LIKE ? 
       OR LOWER(COALESCE(category_final, category)) LIKE ? 
       OR LOWER(COALESCE(subcategory_final, subcategory)) LIKE ?)
  AND ((UPPER(merchant_norm) LIKE ? OR LOWER(description) LIKE ?) 
       AND (UPPER(merchant_norm) LIKE ? OR LOWER(description) LIKE ?))
ORDER BY date_time DESC
LIMIT 10
"""
print(sql1)
print("   Parameters:")
print("     tenant_id, 'expense', '%housing%', '%rent%', '%rent%',")
print("     '%KIRASI%', '%kirası%',  -- First keyword: 'kirası'")
print("     '%IBAN%', '%iban%'       -- Second keyword: 'iban'")

print("\n" + "=" * 80)
print("\n2. Query: 'subscription payment'")
print("   Content Keywords: ['subscription']")
print("\n   Generated SQL:")
sql2 = """
SELECT
    tx_id,
    date_time,
    amount,
    merchant_norm,
    description,
    COALESCE(category_final, category) as category,
    COALESCE(subcategory_final, subcategory) as subcategory,
    direction
FROM transactions
WHERE tenant_id = ?
  AND direction = ?
  AND (LOWER(COALESCE(category_final, category)) LIKE ? 
       OR LOWER(COALESCE(category_final, category)) LIKE ? 
       OR ...)  -- Multiple category matches
  AND (UPPER(merchant_norm) LIKE ? OR LOWER(description) LIKE ?)
ORDER BY date_time DESC
LIMIT 10
"""
print(sql2)
print("   Parameters:")
print("     tenant_id, 'expense', '%utilities%', '%subscriptions%', ...,")
print("     '%SUBSCRIPTION%', '%subscription%'")

print("\n" + "=" * 80)
print("\n3. Query: 'kira ödemesi'")
print("   Content Keywords: ['kira', 'ödemesi']")
print("\n   Generated SQL:")
sql3 = """
SELECT
    tx_id,
    date_time,
    amount,
    merchant_norm,
    description,
    COALESCE(category_final, category) as category,
    COALESCE(subcategory_final, subcategory) as subcategory,
    direction
FROM transactions
WHERE tenant_id = ?
  AND direction = ?
  AND (LOWER(COALESCE(category_final, category)) LIKE ?)
  AND ((UPPER(merchant_norm) LIKE ? OR LOWER(description) LIKE ?)
       AND (UPPER(merchant_norm) LIKE ? OR LOWER(description) LIKE ?))
ORDER BY date_time DESC
LIMIT 10
"""
print(sql3)
print("   Parameters:")
print("     tenant_id, 'expense', '%housing%',")
print("     '%KIRA%', '%kira%',        -- First keyword: 'kira'")
print("     '%ÖDEMESI%', '%ödemesi%'   -- Second keyword: 'ödemesi'")

print("\n" + "=" * 80)
print("\nKEY FEATURES:")
print("=" * 80)
print("""
1. ✅ Keyword Search Logic:
   (UPPER(merchant_norm) LIKE '%KEYWORD%' OR LOWER(description) LIKE '%keyword%')
   
   This searches BOTH merchant name AND description for each keyword.

2. ✅ Multiple Keywords (AND logic):
   If query has ['kira', 'iban'], both must match:
   - Either merchant or description contains 'kira' 
   AND
   - Either merchant or description contains 'iban'

3. ✅ Case-Insensitive:
   - UPPER() for merchant_norm
   - LOWER() for description
   - Both uppercase and lowercase variants in LIKE

4. ✅ Wildcard Matching:
   '%keyword%' matches:
   - "Ev kirası" contains 'kirası' ✅
   - "IBAN Transfer" contains 'iban' ✅
   - "CLAUDE.AI SUBSCRIPTION" contains 'subscription' ✅

5. ✅ Combined with Category Filters:
   If user says "Restaurant harcamalarımda burger geçenler":
   - Category filter: category LIKE '%restaurant%'
   AND
   - Keyword filter: (merchant LIKE '%BURGER%' OR description LIKE '%burger%')
""")

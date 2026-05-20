"""Quick check of Marqo-GS-10M splits and fashion200k/polyvore structure."""
from datasets import load_dataset

for split in ["in_domain", "novel_document", "novel_query", "zero_shot"]:
    ds = load_dataset("Marqo/marqo-GS-10M", split=split, streaming=True)
    count = 0
    has_fashion = 0
    fashion_kw = {"dress", "shirt", "pants", "jacket", "shoes", "bag", "hat",
                  "skirt", "coat", "sweater", "blouse", "jeans", "boots", "scarf",
                  "fashion", "clothing", "wear", "outfit", "top", "shorts", "heel",
                  "sandal", "sneaker", "denim", "leather", "silk", "cotton", "hoodie"}
    for row in ds:
        count += 1
        if count == 1:
            print(f"GS-10M {split}: cols={list(row.keys())}")
            print(f"  title: {str(row.get('title', ''))[:80]}")
            print(f"  query: {str(row.get('query', ''))[:80]}")
            has_img = row.get("image") is not None
            print(f"  has image: {has_img}")
        title = str(row.get("title", "")).lower()
        query = str(row.get("query", "")).lower()
        if any(kw in title or kw in query for kw in fashion_kw):
            has_fashion += 1
        if count >= 2000:
            break
    print(f"  scanned {count} rows, {has_fashion} fashion-related")
    print()

print("=== fashion200k structure ===")
ds = load_dataset("Marqo/fashion200k", split="data", streaming=True)
row = next(iter(ds))
print(f"cols: {list(row.keys())}")
print(f"text: {str(row.get('text', ''))[:80]}")
print(f"item_ID: {row.get('item_ID')}")
print()

print("=== polyvore structure ===")
ds = load_dataset("Marqo/polyvore", split="data", streaming=True)
row = next(iter(ds))
print(f"cols: {list(row.keys())}")
print(f"text: {str(row.get('text', ''))[:80]}")
print(f"item_ID: {row.get('item_ID')}")

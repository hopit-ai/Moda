# deepfashion_multimodal — Benchmark Pattern Analysis

- **HuggingFace:** `Marqo/deepfashion-multimodal`
- **Total rows:** 42,537
- **Columns:** category1, category2, category3, text, item_ID
- **Tasks (2):** text-to-image, category-to-product

---

## Text Fields

### `text`

- Non-null: 42,537 / 42,537 (100.0%)
- Unique values: 40,769 (uniqueness: 95.8%)
- **Duplication:** 4.2% of texts are duplicates

**Word count distribution:** min=4, p10=22, median=42, mean=41.4, p90=59, max=95

**Length buckets:**
  - very_long (21+ words): 38,984 (91.6%)
  - long (11-20 words): 3,473 (8.2%)
  - medium (6-10 words): 78 (0.2%)
  - short (3-5 words): 2 (0.0%)

**Text style:**
  - Has commas: 45.4%
  - Has periods: 100.0%
  - Has numbers: 0.0%
  - Starts uppercase: 100.0%
  - All lowercase: 0.0%

**Vocabulary:** 104 unique tokens across 1,759,778 total

**Top 30 tokens:**
  - `the`: 121,842
  - `is`: 118,379
  - `a`: 82,364
  - `fabric`: 73,406
  - `with`: 73,098
  - `patterns`: 70,136
  - `and`: 68,623
  - `wears`: 52,329
  - `cotton`: 49,586
  - `her`: 48,779
  - `has`: 48,517
  - `this`: 46,441
  - `color`: 45,285
  - `shirt`: 40,665
  - `on`: 35,107
  - `there`: 34,173
  - `tank`: 30,363
  - `it`: 28,190
  - `neckline`: 27,035
  - `ring`: 25,596
  - `an`: 25,271
  - `person`: 23,531
  - `accessory`: 23,445
  - `pure`: 22,179
  - `solid`: 21,946
  - `sleeves`: 21,767
  - `wearing`: 21,746
  - `lady`: 21,590
  - `female`: 21,509
  - `of`: 21,374

**Top 20 bigrams:**
  - `fabric and`: 50,267
  - `cotton fabric`: 43,285
  - `color patterns`: 42,394
  - `there is`: 34,173
  - `on her`: 34,048
  - `with cotton`: 33,775
  - `patterns the`: 32,165
  - `wears a`: 31,742
  - `is with`: 28,090
  - `a ring`: 25,596
  - `is an`: 23,445
  - `an accessory`: 23,445
  - `pure color`: 22,179
  - `solid color`: 21,946
  - `is wearing`: 21,746
  - `wearing a`: 20,890
  - `are with`: 18,102
  - `it has`: 17,631
  - `ring on`: 17,065
  - `accessory on`: 17,030

**Sample texts (first 20):**
  1. "The lower clothing is of long length. The fabric is cotton and it has plaid patterns."
  2. "His tank top has sleeves cut off, cotton fabric and pure color patterns. The neckline of it is round. The pants this man wears is of long length. The ..."
  3. "His sweater has long sleeves, cotton fabric and stripe patterns. The neckline of it is lapel. The gentleman wears a long pants. The pants are with cot..."
  4. "His shirt has short sleeves, cotton fabric and pure color patterns. It has a crew neckline. The person wears a long pants. The pants are with cotton f..."
  5. "The sweater the person wears has long sleeves, its fabric is denim, and it has solid color patterns. The sweater has a lapel neckline. The person wear..."
  6. "The person wears a short-sleeve T-shirt with solid color patterns and long pants. The T-shirt is with cotton fabric and it has a round neckline. The p..."
  7. "This man is wearing a long-sleeve sweater with plaid patterns. The sweater is with cotton fabric and its neckline is lapel. This man wears a long pant..."
  8. "The upper clothing has short sleeves, cotton fabric and solid color patterns. The neckline of it is lapel. The lower clothing is of long length. The f..."
  9. "The gentleman wears a tank tank top with pure color patterns. The tank top is with cotton fabric. It has a square neckline. The gentleman wears a long..."
  10. "This guy wears a long-sleeve shirt with solid color patterns and a long trousers. The shirt is with cotton fabric. The neckline of the shirt is round...."
  11. "The upper clothing has medium sleeves, cotton fabric and pure color patterns. It has a lapel neckline. The lower clothing is of long length. The fabri..."
  12. "His shirt has medium sleeves, cotton fabric and pure color patterns. It has a lapel neckline. The guy wears a long trousers. The trousers are with cot..."
  13. "This gentleman is wearing a medium-sleeve shirt with pure color patterns. The shirt is with cotton fabric and its neckline is lapel. This gentleman we..."
  14. "His shirt has long sleeves, cotton fabric and solid color patterns. The neckline of it is round. The trousers this person wears is of long length. The..."
  15. "This man is wearing a short-sleeve shirt with pure color patterns. The shirt is with cotton fabric and its neckline is lapel. The trousers this man we..."
  16. "His T-shirt has short sleeves, cotton fabric and solid color patterns. The neckline of it is lapel. The trousers this man wears is of long length. The..."
  17. "This man wears a short-sleeve T-shirt with pure color patterns. The T-shirt is with cotton fabric. It has a lapel neckline. The trousers this man wear..."
  18. "This guy is wearing a short-sleeve T-shirt with pure color patterns. The T-shirt is with cotton fabric and its neckline is v-shape. The pants this guy..."
  19. "This person is wearing a short-sleeve shirt with solid color patterns. The shirt is with cotton fabric. It has a crew neckline. The pants this person ..."
  20. "His T-shirt has short sleeves, cotton fabric and pure color patterns. It has a round neckline. The shorts the guy wears is of medium length. The short..."

**Most repeated texts (top 15):**
  - (29×) "The upper clothing has short sleeves, cotton fabric and pure color patterns."
  - (27×) "The upper clothing has long sleeves, cotton fabric and pure color patterns."
  - (25×) "The upper clothing has no sleeves, cotton fabric and solid color patterns."
  - (25×) "The upper clothing has sleeves cut off, cotton fabric and pure color patterns."
  - (23×) "The upper clothing has sleeves cut off, cotton fabric and solid color patterns."
  - (22×) "The upper clothing has no sleeves, cotton fabric and pure color patterns."
  - (19×) "The upper clothing has long sleeves, cotton fabric and solid color patterns."
  - (16×) "The upper clothing has short sleeves, cotton fabric and solid color patterns."
  - (15×) "The upper clothing has short sleeves, cotton fabric and graphic patterns."
  - (14×) "Her shirt has long sleeves, cotton fabric and solid color patterns."
  - (13×) "The upper clothing has short sleeves, cotton fabric and graphic patterns. The neckline of it is lapel."
  - (12×) "Her tank top has no sleeves, cotton fabric and pure color patterns."
  - (10×) "His shirt has long sleeves, cotton fabric and pure color patterns."
  - (10×) "The upper clothing has long sleeves, cotton fabric and graphic patterns."
  - (10×) "The upper clothing has sleeves cut off, cotton fabric and graphic patterns."

---

## Category Fields

### `category1`

- Non-null: 42,537 / 42,537
- Unique values: 2
- Top-5 concentration: 100.0%
- Top-10 concentration: 100.0%

**Full value distribution:**
  - `women`: 37,577 (88.3%) ████████████████████████████████████████████
  - `men`: 4,960 (11.7%) █████

### `category2`

- Non-null: 42,537 / 42,537
- Unique values: 16
- Top-5 concentration: 77.6%
- Top-10 concentration: 94.1%

**Full value distribution:**
  - `tees`: 13,356 (31.4%) ███████████████
  - `blouses`: 7,503 (17.6%) ████████
  - `dresses`: 6,680 (15.7%) ███████
  - `sweaters`: 3,322 (7.8%) ███
  - `jackets`: 2,164 (5.1%) ██
  - `rompers`: 1,627 (3.8%) █
  - `shorts`: 1,514 (3.6%) █
  - `sweatshirts`: 1,398 (3.3%) █
  - `cardigans`: 1,382 (3.2%) █
  - `graphic`: 1,096 (2.6%) █
  - `skirts`: 813 (1.9%) █
  - `pants`: 744 (1.7%) █
  - `shirts`: 569 (1.3%) █
  - `denim`: 243 (0.6%) █
  - `leggings`: 101 (0.2%) █
  - `suiting`: 25 (0.1%) █

---

## Text ↔ Category Overlap

How often does the `text` field contain (or share words with) category labels?

### text vs `category1`
  - Text contains exact category string: 0.0%
  - Text shares words with category: 0.0%
  - **Total overlap: 0.0%**

### text vs `category2`
  - Text contains exact category string: 66.2%
  - Text shares words with category: 0.0%
  - **Total overlap: 66.2%**

---

## Cross-Field Correlations

### category1 × category2
- Unique combinations: 23

**Top 20 combinations:**
  - `women | tees`: 11,018
  - `women | blouses`: 7,503
  - `women | dresses`: 6,680
  - `women | sweaters`: 2,852
  - `men | tees`: 2,338
  - `women | jackets`: 1,801
  - `women | rompers`: 1,627
  - `women | cardigans`: 1,382
  - `women | shorts`: 1,286
  - `women | graphic`: 1,096
  - `women | skirts`: 813
  - `women | sweatshirts`: 767
  - `men | sweatshirts`: 631
  - `men | shirts`: 569
  - `women | pants`: 505
  - `men | sweaters`: 470
  - `men | jackets`: 363
  - `men | pants`: 239
  - `men | shorts`: 228
  - `women | denim`: 146

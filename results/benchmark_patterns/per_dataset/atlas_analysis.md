# atlas — Benchmark Pattern Analysis

- **HuggingFace:** `Marqo/atlas`
- **Total rows:** 78,370
- **Columns:** gender, category, sub-category, text, item_ID
- **Tasks (2):** text-to-image, sub-category-to-product

---

## Text Fields

### `text`

- Non-null: 78,370 / 78,370 (100.0%)
- Unique values: 16,720 (uniqueness: 21.3%)
- **Duplication:** 78.7% of texts are duplicates

**Word count distribution:** min=1, p10=5, median=7, mean=7.0, p90=9, max=21

**Length buckets:**
  - medium (6-10 words): 65,903 (84.1%)
  - short (3-5 words): 10,610 (13.5%)
  - long (11-20 words): 1,842 (2.4%)
  - very_short (1-2 words): 11 (0.0%)
  - very_long (21+ words): 4 (0.0%)

**Text style:**
  - Has commas: 6.6%
  - Has periods: 3.1%
  - Has numbers: 10.2%
  - Starts uppercase: 93.9%
  - All lowercase: 0.0%

**Vocabulary:** 4,327 unique tokens across 549,597 total

**Top 30 tokens:**
  - `men's`: 41,439
  - `solid`: 33,192
  - `blue`: 17,443
  - `sleeve`: 14,100
  - `shirt`: 13,761
  - `casual`: 12,981
  - `printed`: 12,295
  - `full`: 11,555
  - `men`: 9,932
  - `women's`: 9,817
  - `fit`: 9,784
  - `shorts`: 9,702
  - `slim`: 9,459
  - `black`: 9,008
  - `sweatshirt`: 8,326
  - `jeans`: 7,570
  - `trousers`: 7,243
  - `jacket`: 6,643
  - `multicolor`: 5,350
  - `grey`: 5,331
  - `white`: 5,106
  - `regular`: 4,724
  - `with`: 4,617
  - `design`: 4,612
  - `self`: 4,582
  - `formal`: 4,543
  - `dark`: 4,334
  - `track`: 4,252
  - `pants`: 4,243
  - `top`: 3,790

**Top 20 bigrams:**
  - `solid men's`: 13,103
  - `full sleeve`: 11,539
  - `sleeve solid`: 8,361
  - `fit men's`: 6,187
  - `slim fit`: 6,106
  - `blue jeans`: 4,871
  - `solid women's`: 4,708
  - `men's blue`: 4,640
  - `self design`: 4,545
  - `track pants`: 4,243
  - `dark blue`: 3,975
  - `sleeve printed`: 3,949
  - `solid men`: 3,796
  - `men's jacket`: 3,787
  - `campus sutra`: 3,478
  - `men's sweatshirt`: 3,338
  - `solid casual`: 3,298
  - `men's solid`: 3,246
  - `regular fit`: 3,245
  - `sutra full`: 3,187

**Sample texts (first 20):**
  1. "Plain Dupion Silk Dhoti Kurta in Fawn"
  2. "Solid Color Dupion Silk Dhoti Kurta in Royal Blue"
  3. "Solid Color Rayon Asymmetric Dhoti Kurta in Black"
  4. "Solid Color Art Silk Kurta Set in Beige"
  5. "Solid Color Rayon Kurta Set in Black"
  6. "Printed Dupion Silk Dhoti Kurta in Red"
  7. "Plain Dupion Silk Dhoti Kurta with Jacket in Maroon and Black"
  8. "Plain Kurta Set in Navy Blue"
  9. "Woven Pure Silk Dhoti with Kurta in Cream"
  10. "Solid Color Dupion Silk Dhoti Kurta in White"
  11. "Contrast Patch Border Dupion Silk Short Dhoti Kurta in Off White"
  12. "Solid Color Rayon Kurta Set in Peach Orange"
  13. "Plain Art Dupion Silk Dhoti Kurta in Red"
  14. "Mens dhoti with black boarder"
  15. "Plain Cotton Silk Churidar in Off White"
  16. "Mens dhoti with mejantha boarder"
  17. "Mens dhoti with gold boarder"
  18. "Mens dhoti with silver boarder"
  19. "Dupion Silk Dhoti in Beige"
  20. "Dupion Silk Dhoti in Beige"

**Most repeated texts (top 15):**
  - (1,700×) "Campus Sutra Full Sleeve Printed Men Sweatshirt"
  - (712×) "Campus Sutra Full Sleeve Solid Men Sweatshirt"
  - (429×) "Sports 52 Wear Men's Cargos"
  - (341×) "Campus Sutra Full Sleeve Solid Men Jacket"
  - (212×) "ShopyBucket Self Design Men's Waistcoat"
  - (191×) "Pepe Jeans Full Sleeve Solid Men's Jacket"
  - (179×) "Lee Skinny Men's Blue Jeans"
  - (178×) "Beevee Men Cargos"
  - (176×) "Rakshita Collection Solid Men's Multicolor Sports Shorts"
  - (173×) "Flying Machine Slim Men's Blue Jeans"
  - (170×) "Kotty Printed Women's Multicolor Night Shorts"
  - (164×) "ManQ Solid Men's Waistcoat"
  - (158×) "Christy World Solid Men's Multicolor Basic Shorts"
  - (151×) "Plutus Full Sleeve Solid Men's Jacket"
  - (150×) "Spykar Skinny Men's Blue Jeans"

---

## Category Fields

### `category`

- Non-null: 78,370 / 78,370
- Unique values: 3
- Top-5 concentration: 100.0%
- Top-10 concentration: 100.0%

**Full value distribution:**
  - `Western Wear`: 70,116 (89.5%) ████████████████████████████████████████████
  - `Ethnic Wear`: 7,880 (10.1%) █████
  - `Inner Wear`: 374 (0.5%) █

### `sub-category`

- Non-null: 78,370 / 78,370
- Unique values: 34
- Top-5 concentration: 49.6%
- Top-10 concentration: 83.5%

**Full value distribution:**
  - `Shorts`: 10,025 (12.8%) ██████
  - `Shirts`: 7,513 (9.6%) ████
  - `Tops&Tees`: 7,393 (9.4%) ████
  - `Trousers`: 7,237 (9.2%) ████
  - `Sweatshirts&Hoodies`: 6,733 (8.6%) ████
  - `Blazers&Suits`: 6,678 (8.5%) ████
  - `Jeans`: 6,381 (8.1%) ████
  - `Jackets`: 5,869 (7.5%) ███
  - `Tracks&Joggers`: 4,243 (5.4%) ██
  - `Abayas&Burqas`: 3,333 (4.3%) ██
  - `Formal Shirts`: 2,788 (3.6%) █
  - `Cargos`: 2,046 (2.6%) █
  - `Sarees`: 2,016 (2.6%) █
  - `Sweatshirts&Shirts`: 1,616 (2.1%) █
  - `Salwar Kameez&Kurtis`: 1,162 (1.5%) █
  - `Jacket&Shrugs&Cardigans`: 1,085 (1.4%) █
  - `Kurta`: 617 (0.8%) █
  - `Lehenga Choli`: 556 (0.7%) █
  - `Boxers`: 228 (0.3%) █
  - `Blazers&Waistcoats`: 200 (0.3%) █
  - `Dungarees`: 157 (0.2%) █
  - `Pyjama`: 109 (0.1%) █
  - `Palazzo pants`: 71 (0.1%) █
  - `Pants&Trousers`: 59 (0.1%) █
  - `Dhoti Pants`: 58 (0.1%) █
  - `Skirts`: 57 (0.1%) █
  - `Petticoats`: 37 (0.0%) █
  - `Galabiyyas`: 36 (0.0%) █
  - `Patialas`: 25 (0.0%) █
  - `Gowns&Dresses`: 20 (0.0%) █
  - `Dhoti Kurta`: 13 (0.0%) █
  - `Sherwani`: 5 (0.0%) █
  - `Stoles&Scarves`: 3 (0.0%) █
  - `Bandhgala`: 1 (0.0%) █

---

## Attribute Fields

### `gender`

- Non-null: 78,370 / 78,370
- Unique values: 2
- Top-5 concentration: 100.0%

**Full value distribution:**
  - `Men`: 54,901 (70.1%)
  - `Women`: 23,469 (29.9%)

---

## Text ↔ Category Overlap

How often does the `text` field contain (or share words with) category labels?

### text vs `category`
  - Text contains exact category string: 0.0%
  - Text shares words with category: 2.3%
  - **Total overlap: 2.3%**

### text vs `sub-category`
  - Text contains exact category string: 34.4%
  - Text shares words with category: 21.0%
  - **Total overlap: 55.4%**

### text vs `gender`
  - Text contains exact category string: 89.8%
  - Text shares words with category: 0.0%
  - **Total overlap: 89.8%**

---

## Cross-Field Correlations

### gender × category
- Unique combinations: 6

**Top 20 combinations:**
  - `Men | Western Wear`: 53,796
  - `Women | Western Wear`: 16,320
  - `Women | Ethnic Wear`: 7,112
  - `Men | Ethnic Wear`: 768
  - `Men | Inner Wear`: 337
  - `Women | Inner Wear`: 37

### gender × sub-category
- Unique combinations: 35

**Top 20 combinations:**
  - `Men | Shirts`: 7,513
  - `Women | Tops&Tees`: 7,393
  - `Men | Trousers`: 7,237
  - `Men | Sweatshirts&Hoodies`: 6,733
  - `Men | Blazers&Suits`: 6,678
  - `Men | Jeans`: 6,381
  - `Men | Jackets`: 5,869
  - `Women | Shorts`: 5,679
  - `Men | Shorts`: 4,346
  - `Men | Tracks&Joggers`: 4,243
  - `Women | Abayas&Burqas`: 3,333
  - `Men | Formal Shirts`: 2,788
  - `Men | Cargos`: 2,046
  - `Women | Sarees`: 2,016
  - `Women | Sweatshirts&Shirts`: 1,616
  - `Women | Salwar Kameez&Kurtis`: 1,162
  - `Women | Jacket&Shrugs&Cardigans`: 1,085
  - `Men | Kurta`: 617
  - `Women | Lehenga Choli`: 556
  - `Men | Boxers`: 228

### category × sub-category
- Unique combinations: 35

**Top 20 combinations:**
  - `Western Wear | Shorts`: 10,025
  - `Western Wear | Shirts`: 7,513
  - `Western Wear | Tops&Tees`: 7,393
  - `Western Wear | Trousers`: 7,237
  - `Western Wear | Sweatshirts&Hoodies`: 6,733
  - `Western Wear | Blazers&Suits`: 6,678
  - `Western Wear | Jeans`: 6,381
  - `Western Wear | Jackets`: 5,831
  - `Western Wear | Tracks&Joggers`: 4,243
  - `Ethnic Wear | Abayas&Burqas`: 3,333
  - `Western Wear | Formal Shirts`: 2,788
  - `Western Wear | Cargos`: 2,046
  - `Ethnic Wear | Sarees`: 2,016
  - `Western Wear | Sweatshirts&Shirts`: 1,616
  - `Ethnic Wear | Salwar Kameez&Kurtis`: 1,162
  - `Western Wear | Jacket&Shrugs&Cardigans`: 1,085
  - `Ethnic Wear | Kurta`: 617
  - `Ethnic Wear | Lehenga Choli`: 556
  - `Inner Wear | Boxers`: 228
  - `Western Wear | Blazers&Waistcoats`: 200

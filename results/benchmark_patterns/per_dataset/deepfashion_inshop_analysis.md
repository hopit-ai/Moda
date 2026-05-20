# deepfashion_inshop — Benchmark Pattern Analysis

- **HuggingFace:** `Marqo/deepfashion-inshop`
- **Total rows:** 52,591
- **Columns:** category1, category2, category3, color, description, text, item_ID
- **Tasks (4):** text-to-image, category-to-product, sub-category-to-product, color-to-product

---

## Text Fields

### `text`

- Non-null: 52,591 / 52,591 (100.0%)
- Unique values: 7,959 (uniqueness: 15.1%)
- **Duplication:** 84.9% of texts are duplicates

**Word count distribution:** min=24, p10=46, median=69, mean=69.8, p90=93, max=162

**Length buckets:**
  - very_long (21+ words): 52,591 (100.0%)

**Text style:**
  - Has commas: 100.0%
  - Has periods: 100.0%
  - Has numbers: 100.0%
  - Starts uppercase: 98.8%
  - All lowercase: 0.0%

**Vocabulary:** 8,172 unique tokens across 3,670,700 total

**Top 30 tokens:**
  - `a`: 127,391
  - `and`: 91,000
  - `from`: 72,506
  - `length`: 71,149
  - `this`: 64,941
  - `waist`: 64,109
  - `the`: 60,397
  - `with`: 59,420
  - `measured`: 52,516
  - `wash`: 50,792
  - `to`: 50,312
  - `cold`: 49,092
  - `chest`: 44,920
  - `small`: 42,851
  - `full`: 42,021
  - `for`: 36,483
  - `lightweight`: 35,774
  - `100`: 33,054
  - `5`: 32,795
  - `of`: 31,199
  - `knit`: 30,974
  - `it`: 30,136
  - `cotton`: 29,284
  - `hand`: 28,344
  - `is`: 28,305
  - `polyester`: 26,487
  - `sleeve`: 25,390
  - `in`: 24,937
  - `woven`: 23,455
  - `your`: 22,378

**Top 20 bigrams:**
  - `measured from`: 52,516
  - `wash cold`: 48,550
  - `from small`: 42,581
  - `full length`: 41,301
  - `hand wash`: 27,922
  - `sleeve length`: 24,780
  - `small hand`: 24,584
  - `length measured`: 23,096
  - `waist measured`: 21,792
  - `machine wash`: 21,557
  - `with a`: 20,364
  - `and a`: 14,981
  - `small machine`: 11,860
  - `100 polyester`: 11,366
  - `for a`: 10,026
  - `100 cotton`: 9,267
  - `lightweight knit`: 9,237
  - `lightweight woven`: 8,999
  - `to hem`: 8,839
  - `5 full`: 8,607

**Sample texts (first 20):**
  1. "Give your trusty blues the day off. In a clean wash, these skinny jeans are a slick, sharp option. Also, their classic five-pocket construction and sl..."
  2. "Give your trusty blues the day off. In a clean wash, these skinny jeans are a slick, sharp option. Also, their classic five-pocket construction and sl..."
  3. "Give your trusty blues the day off. In a clean wash, these skinny jeans are a slick, sharp option. Also, their classic five-pocket construction and sl..."
  4. "Give your trusty blues the day off. In a clean wash, these skinny jeans are a slick, sharp option. Also, their classic five-pocket construction and sl..."
  5. "Classic in their clean wash and slim fit, these jeans are ideal for work and off-duty weekend looks alike. Complete with a five-pocket construction to..."
  6. "Classic in their clean wash and slim fit, these jeans are ideal for work and off-duty weekend looks alike. Complete with a five-pocket construction to..."
  7. "Classic in their clean wash and slim fit, these jeans are ideal for work and off-duty weekend looks alike. Complete with a five-pocket construction to..."
  8. "Classic in their clean wash and slim fit, these jeans are ideal for work and off-duty weekend looks alike. Complete with a five-pocket construction to..."
  9. "Classic in their clean wash and slim fit, these jeans are ideal for work and off-duty weekend looks alike. Complete with a five-pocket construction to..."
  10. "Classic in their clean wash and slim fit, these jeans are ideal for work and off-duty weekend looks alike. Complete with a five-pocket construction to..."
  11. "Classic in their clean wash and slim fit, these jeans are ideal for work and off-duty weekend looks alike. Complete with a five-pocket construction to..."
  12. "Classic in their clean wash and slim fit, these jeans are ideal for work and off-duty weekend looks alike. Complete with a five-pocket construction to..."
  13. "Classic in their clean wash and slim fit, these jeans are ideal for work and off-duty weekend looks alike. Complete with a five-pocket construction to..."
  14. "Classic in their clean wash and slim fit, these jeans are ideal for work and off-duty weekend looks alike. Complete with a five-pocket construction to..."
  15. "Classic in their clean wash and slim fit, these jeans are ideal for work and off-duty weekend looks alike. Complete with a five-pocket construction to..."
  16. "Classic in their clean wash and slim fit, these jeans are ideal for work and off-duty weekend looks alike. Complete with a five-pocket construction to..."
  17. "Classic in their clean wash and slim fit, these jeans are ideal for work and off-duty weekend looks alike. Complete with a five-pocket construction to..."
  18. "Classic in their clean wash and slim fit, these jeans are ideal for work and off-duty weekend looks alike. Complete with a five-pocket construction to..."
  19. "Classic in their clean wash and slim fit, these jeans are ideal for work and off-duty weekend looks alike. Complete with a five-pocket construction to..."
  20. "Classic in their clean wash and slim fit, these jeans are ideal for work and off-duty weekend looks alike. Complete with a five-pocket construction to..."

**Most repeated texts (top 15):**
  - (168×) "A V-neck tee featuring ribbed trim. Short sleeves. Knit. Lightweight.  ||  27" approx. length from high point shoulder t..."
  - (162×) "A knit tee featuring a V-neckline. Short sleeves. Finished trim. Lightweight.  || 25.5" approx. length from high point s..."
  - (131×) "A knit cami featuring a V-neckline. Stretch-fit. Finished trim. Lightweight.  || 24" approx. length from high point shou..."
  - (128×) "A basic cami featuring a V-neckline. Finished trim. Knit. Lightweight.  || 24.5" approx. length from high point shoulder..."
  - (121×) "Does your search for the perfect cami know no end? Your search will come to a close with this one. Its lightweight feel,..."
  - (111×) "A basic mini skirt. Banded waist. Stretch knit. Lightweight.  || 16" approx. length from waist to hem, 26" waist || Meas..."
  - (102×) "A lightweight tank featuring a round neckline. Knit.  || 29" approx. length from high point shoulder to hem, 42" chest, ..."
  - (89×) "Fall is quickly approaching, which means we can stop daydreaming about this classic button-front V-neck cardigan and fin..."
  - (89×) "A layering bandeau featuring a ruched front. Elasticized trim. Knit. Lightweight.  || 4" approx. length from bust to hem..."
  - (79×) "Like your favorite knit cami, only made better by a slightly longer silhouette for added coverage. With its soft-to-the-..."
  - (77×) "Instantly classic and versatile in its simplicity, this tank dress will be your new anytime, anywhere piece. With a flat..."
  - (76×) "Style Deals - A cami featuring spaghetti straps. Round neckline. Knit. Lightweight.  || 20" approx. length from high poi..."
  - (75×) "A knit cami featuring a scoop neckline. Stretch-fit. Finished trim. Lightweight.  || 23.5" approx. length from high poin..."
  - (74×) "A basic tank top featuring a scoop neck and back. Sleeveless. Stretch knit. Lightweight.  || 25" approx. length from hig..."
  - (73×) "Classic in their clean wash and slim fit, these jeans are ideal for work and off-duty weekend looks alike. Complete with..."

### `description`

- Non-null: 52,591 / 52,591 (100.0%)
- Unique values: 7,959 (uniqueness: 15.1%)
- **Duplication:** 84.9% of texts are duplicates

**Word count distribution:** min=25, p10=47, median=71, mean=71.5, p90=95, max=163

**Length buckets:**
  - very_long (21+ words): 52,591 (100.0%)

**Text style:**
  - Has commas: 100.0%
  - Has periods: 100.0%
  - Has numbers: 100.0%
  - Starts uppercase: 0.0%
  - All lowercase: 0.0%

**Vocabulary:** 8,177 unique tokens across 3,760,691 total

**Top 30 tokens:**
  - `a`: 127,391
  - `and`: 91,000
  - `from`: 72,506
  - `length`: 71,149
  - `this`: 64,941
  - `waist`: 64,109
  - `the`: 60,416
  - `with`: 59,441
  - `measured`: 52,516
  - `wash`: 50,812
  - `to`: 50,317
  - `cold`: 49,104
  - `chest`: 44,920
  - `small`: 42,851
  - `in`: 42,484
  - `full`: 42,021
  - `for`: 36,483
  - `lightweight`: 35,774
  - `imported`: 34,104
  - `100`: 33,054
  - `5`: 32,795
  - `of`: 31,199
  - `knit`: 30,974
  - `it`: 30,631
  - `cotton`: 29,284
  - `hand`: 28,356
  - `is`: 28,305
  - `polyester`: 26,487
  - `sleeve`: 25,390
  - `woven`: 23,447

**Top 20 bigrams:**
  - `measured from`: 52,516
  - `wash cold`: 48,562
  - `from small`: 42,581
  - `full length`: 41,301
  - `hand wash`: 27,934
  - `cold imported`: 27,520
  - `sleeve length`: 24,780
  - `small hand`: 24,596
  - `length measured`: 23,096
  - `waist measured`: 21,792
  - `machine wash`: 21,565
  - `with a`: 20,364
  - `made in`: 17,798
  - `cold made`: 16,413
  - `and a`: 14,981
  - `small machine`: 11,860
  - `100 polyester`: 11,366
  - `for a`: 10,026
  - `100 cotton`: 9,267
  - `lightweight knit`: 9,237

**Sample texts (first 20):**
  1. "['Give your trusty blues the day off. In a clean wash, these skinny jeans are a slick, sharp option. Also, their classic five-pocket construction and ..."
  2. "['Give your trusty blues the day off. In a clean wash, these skinny jeans are a slick, sharp option. Also, their classic five-pocket construction and ..."
  3. "['Give your trusty blues the day off. In a clean wash, these skinny jeans are a slick, sharp option. Also, their classic five-pocket construction and ..."
  4. "['Give your trusty blues the day off. In a clean wash, these skinny jeans are a slick, sharp option. Also, their classic five-pocket construction and ..."
  5. "['Classic in their clean wash and slim fit, these jeans are ideal for work and off-duty weekend looks alike. Complete with a five-pocket construction ..."
  6. "['Classic in their clean wash and slim fit, these jeans are ideal for work and off-duty weekend looks alike. Complete with a five-pocket construction ..."
  7. "['Classic in their clean wash and slim fit, these jeans are ideal for work and off-duty weekend looks alike. Complete with a five-pocket construction ..."
  8. "['Classic in their clean wash and slim fit, these jeans are ideal for work and off-duty weekend looks alike. Complete with a five-pocket construction ..."
  9. "['Classic in their clean wash and slim fit, these jeans are ideal for work and off-duty weekend looks alike. Complete with a five-pocket construction ..."
  10. "['Classic in their clean wash and slim fit, these jeans are ideal for work and off-duty weekend looks alike. Complete with a five-pocket construction ..."
  11. "['Classic in their clean wash and slim fit, these jeans are ideal for work and off-duty weekend looks alike. Complete with a five-pocket construction ..."
  12. "['Classic in their clean wash and slim fit, these jeans are ideal for work and off-duty weekend looks alike. Complete with a five-pocket construction ..."
  13. "['Classic in their clean wash and slim fit, these jeans are ideal for work and off-duty weekend looks alike. Complete with a five-pocket construction ..."
  14. "['Classic in their clean wash and slim fit, these jeans are ideal for work and off-duty weekend looks alike. Complete with a five-pocket construction ..."
  15. "['Classic in their clean wash and slim fit, these jeans are ideal for work and off-duty weekend looks alike. Complete with a five-pocket construction ..."
  16. "['Classic in their clean wash and slim fit, these jeans are ideal for work and off-duty weekend looks alike. Complete with a five-pocket construction ..."
  17. "['Classic in their clean wash and slim fit, these jeans are ideal for work and off-duty weekend looks alike. Complete with a five-pocket construction ..."
  18. "['Classic in their clean wash and slim fit, these jeans are ideal for work and off-duty weekend looks alike. Complete with a five-pocket construction ..."
  19. "['Classic in their clean wash and slim fit, these jeans are ideal for work and off-duty weekend looks alike. Complete with a five-pocket construction ..."
  20. "['Classic in their clean wash and slim fit, these jeans are ideal for work and off-duty weekend looks alike. Complete with a five-pocket construction ..."

**Most repeated texts (top 15):**
  - (168×) "['A V-neck tee featuring ribbed trim. Short sleeves. Knit. Lightweight. ', ' 27" approx. length from high point shoulder..."
  - (162×) "['A knit tee featuring a V-neckline. Short sleeves. Finished trim. Lightweight. ', '25.5" approx. length from high point..."
  - (131×) "['A knit cami featuring a V-neckline. Stretch-fit. Finished trim. Lightweight. ', '24" approx. length from high point sh..."
  - (128×) "['A basic cami featuring a V-neckline. Finished trim. Knit. Lightweight. ', '24.5" approx. length from high point should..."
  - (121×) "['Does your search for the perfect cami know no end? Your search will come to a close with this one. Its lightweight fee..."
  - (111×) "['A basic mini skirt. Banded waist. Stretch knit. Lightweight. ', '16" approx. length from waist to hem, 26" waist', 'Me..."
  - (102×) "['A lightweight tank featuring a round neckline. Knit. ', '29" approx. length from high point shoulder to hem, 42" chest..."
  - (89×) "['Fall is quickly approaching, which means we can stop daydreaming about this classic button-front V-neck cardigan and f..."
  - (89×) "['A layering bandeau featuring a ruched front. Elasticized trim. Knit. Lightweight. ', '4" approx. length from bust to h..."
  - (79×) "['Like your favorite knit cami, only made better by a slightly longer silhouette for added coverage. With its soft-to-th..."
  - (77×) "["Instantly classic and versatile in its simplicity, this tank dress will be your new anytime, anywhere piece. With a fl..."
  - (76×) "['Style Deals - A cami featuring spaghetti straps. Round neckline. Knit. Lightweight. ', '20" approx. length from high p..."
  - (75×) "['A knit cami featuring a scoop neckline. Stretch-fit. Finished trim. Lightweight. ', '23.5" approx. length from high po..."
  - (74×) "['A basic tank top featuring a scoop neck and back. Sleeveless. Stretch knit. Lightweight. ', '25" approx. length from h..."
  - (73×) "['Classic in their clean wash and slim fit, these jeans are ideal for work and off-duty weekend looks alike. Complete wi..."

---

## Category Fields

### `category1`

- Non-null: 52,591 / 52,591
- Unique values: 2
- Top-5 concentration: 100.0%
- Top-10 concentration: 100.0%

**Full value distribution:**
  - `women`: 44,753 (85.1%) ██████████████████████████████████████████
  - `men`: 7,838 (14.9%) ███████

### `category2`

- Non-null: 52,591 / 52,591
- Unique values: 16
- Top-5 concentration: 71.2%
- Top-10 concentration: 91.2%

**Full value distribution:**
  - `tees`: 14,375 (27.3%) █████████████
  - `blouses`: 7,964 (15.1%) ███████
  - `dresses`: 6,990 (13.3%) ██████
  - `shorts`: 4,476 (8.5%) ████
  - `sweaters`: 3,641 (6.9%) ███
  - `pants`: 2,821 (5.4%) ██
  - `jackets`: 2,319 (4.4%) ██
  - `skirts`: 2,045 (3.9%) █
  - `rompers`: 1,696 (3.2%) █
  - `sweatshirts`: 1,631 (3.1%) █
  - `cardigans`: 1,436 (2.7%) █
  - `graphic`: 1,297 (2.5%) █
  - `denim`: 804 (1.5%) █
  - `shirts`: 722 (1.4%) █
  - `leggings`: 335 (0.6%) █
  - `suiting`: 39 (0.1%) █

### `category3`

- Non-null: 30,004 / 52,591
- Unique values: 8
- Top-5 concentration: 91.9%
- Top-10 concentration: 100.0%

**Full value distribution:**
  - `tanks`: 14,375 (47.9%) ███████████████████████
  - `shirts`: 7,964 (26.5%) █████████████
  - `coats`: 1,895 (6.3%) ███
  - `jumpsuits`: 1,696 (5.7%) ██
  - `hoodies`: 1,631 (5.4%) ██
  - `tees`: 1,297 (4.3%) ██
  - `polos`: 722 (2.4%) █
  - `vests`: 424 (1.4%) █

---

## Attribute Fields

### `color`

- Non-null: 52,591 / 52,591
- Unique values: 804
- Top-5 concentration: 23.5%

**Full value distribution:**
  - `Black`: 3,409 (6.5%)
  - `Cream`: 3,112 (5.9%)
  - `Burgundy`: 2,492 (4.7%)
  - `Cream-black`: 1,721 (3.3%)
  - `Black-cream`: 1,637 (3.1%)
  - `Heather grey`: 1,576 (3.0%)
  - `Olive`: 1,195 (2.3%)
  - `White-black`: 1,118 (2.1%)
  - `Rust`: 1,006 (1.9%)
  - `Black-white`: 945 (1.8%)
  - `Blush`: 898 (1.7%)
  - `Navy`: 875 (1.7%)
  - `Mustard`: 814 (1.5%)
  - `Taupe`: 691 (1.3%)
  - `Teal`: 685 (1.3%)
  - `Wine`: 661 (1.3%)
  - `Amber`: 654 (1.2%)
  - `White`: 634 (1.2%)
  - `Light denim`: 610 (1.2%)
  - `Oatmeal`: 583 (1.1%)
  - `Denim`: 581 (1.1%)
  - `Indigo`: 529 (1.0%)
  - `Grey`: 491 (0.9%)
  - `Denim washed`: 460 (0.9%)
  - `Coral`: 451 (0.9%)
  - `Mauve`: 432 (0.8%)
  - `Blue-cream`: 392 (0.7%)
  - `Navy-cream`: 361 (0.7%)
  - `Navy-white`: 353 (0.7%)
  - `Heather grey-black`: 353 (0.7%)
  - `Pink`: 348 (0.7%)
  - `Light blue`: 339 (0.6%)
  - `Khaki`: 335 (0.6%)
  - `Aqua`: 316 (0.6%)
  - `Rose`: 314 (0.6%)
  - `Tomato`: 303 (0.6%)
  - `Black-multi`: 286 (0.5%)
  - `Black-grey`: 280 (0.5%)
  - `Peacock`: 270 (0.5%)
  - `Blue-white`: 269 (0.5%)
  - `Black-red`: 263 (0.5%)
  - `Cream-multi`: 261 (0.5%)
  - `Periwinkle`: 261 (0.5%)
  - `Red-cream`: 238 (0.5%)
  - `Royal`: 234 (0.4%)
  - `Peach`: 228 (0.4%)
  - `Blue`: 226 (0.4%)
  - `Red`: 226 (0.4%)
  - `Georgia peach`: 220 (0.4%)
  - `Brown`: 217 (0.4%)
  - ... and 754 more

---

## Text ↔ Category Overlap

How often does the `text` field contain (or share words with) category labels?

### text vs `category1`
  - Text contains exact category string: 11.1%
  - Text shares words with category: 0.0%
  - **Total overlap: 11.1%**

### text vs `category2`
  - Text contains exact category string: 32.7%
  - Text shares words with category: 0.0%
  - **Total overlap: 32.7%**

### text vs `category3`
  - Text contains exact category string: 3.7%
  - Text shares words with category: 0.0%
  - **Total overlap: 3.7%**

### text vs `color`
  - Text contains exact category string: 83.0%
  - Text shares words with category: 16.9%
  - **Total overlap: 99.9%**

---

## Cross-Field Correlations

### category1 × category2
- Unique combinations: 23

**Top 20 combinations:**
  - `women | tees`: 11,530
  - `women | blouses`: 7,964
  - `women | dresses`: 6,990
  - `women | shorts`: 3,476
  - `women | sweaters`: 3,036
  - `men | tees`: 2,845
  - `women | skirts`: 2,045
  - `women | jackets`: 1,895
  - `women | pants`: 1,804
  - `women | rompers`: 1,696
  - `women | cardigans`: 1,436
  - `women | graphic`: 1,297
  - `men | pants`: 1,017
  - `men | shorts`: 1,000
  - `women | sweatshirts`: 856
  - `men | sweatshirts`: 775
  - `men | shirts`: 722
  - `men | sweaters`: 605
  - `men | jackets`: 424
  - `men | denim`: 411

### category1 × category3
- Unique combinations: 10

**Top 20 combinations:**
  - `women | tanks`: 11,530
  - `women | shirts`: 7,964
  - `men | tanks`: 2,845
  - `women | coats`: 1,895
  - `women | jumpsuits`: 1,696
  - `women | tees`: 1,297
  - `women | hoodies`: 856
  - `men | hoodies`: 775
  - `men | polos`: 722
  - `men | vests`: 424

### category1 × color
- Unique combinations: 951

**Top 20 combinations:**
  - `women | Cream`: 3,079
  - `women | Black`: 2,861
  - `women | Cream-black`: 1,604
  - `women | Black-cream`: 1,591
  - `women | Burgundy`: 1,553
  - `women | Heather grey`: 1,256
  - `women | Rust`: 989
  - `women | Olive`: 951
  - `men | Burgundy`: 939
  - `women | Blush`: 898
  - `women | White-black`: 879
  - `women | Mustard`: 770
  - `women | Black-white`: 706
  - `women | Navy`: 697
  - `women | Taupe`: 677
  - `women | Amber`: 654
  - `women | Wine`: 641
  - `women | Teal`: 597
  - `women | White`: 580
  - `women | Denim`: 572

### category2 × category3
- Unique combinations: 8

**Top 20 combinations:**
  - `tees | tanks`: 14,375
  - `blouses | shirts`: 7,964
  - `jackets | coats`: 1,895
  - `rompers | jumpsuits`: 1,696
  - `sweatshirts | hoodies`: 1,631
  - `graphic | tees`: 1,297
  - `shirts | polos`: 722
  - `jackets | vests`: 424

### category2 × color
- Unique combinations: 2,228

**Top 20 combinations:**
  - `blouses | Cream`: 931
  - `tees | Burgundy`: 921
  - `tees | Heather grey`: 766
  - `tees | Cream`: 764
  - `tees | Black`: 677
  - `dresses | Cream`: 502
  - `tees | Cream-black`: 491
  - `dresses | Black`: 478
  - `blouses | Black`: 464
  - `tees | White-black`: 435
  - `graphic | White-black`: 421
  - `blouses | Cream-black`: 404
  - `tees | Amber`: 388
  - `tees | Oatmeal`: 386
  - `tees | Olive`: 377
  - `blouses | Blush`: 368
  - `tees | Rust`: 345
  - `jackets | Black`: 343
  - `shorts | Black`: 321
  - `tees | Black-cream`: 313

### category3 × color
- Unique combinations: 1,129

**Top 20 combinations:**
  - `shirts | Cream`: 931
  - `tanks | Burgundy`: 921
  - `tanks | Heather grey`: 766
  - `tanks | Cream`: 764
  - `tanks | Black`: 677
  - `tanks | Cream-black`: 491
  - `shirts | Black`: 464
  - `tanks | White-black`: 435
  - `tees | White-black`: 421
  - `shirts | Cream-black`: 404
  - `tanks | Amber`: 388
  - `tanks | Oatmeal`: 386
  - `tanks | Olive`: 377
  - `shirts | Blush`: 368
  - `tanks | Rust`: 345
  - `tanks | Black-cream`: 313
  - `coats | Black`: 279
  - `shirts | Black-cream`: 271
  - `tanks | Mustard`: 256
  - `hoodies | Burgundy`: 254

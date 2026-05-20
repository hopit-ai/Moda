# KAGL — Benchmark Pattern Analysis

- **HuggingFace:** `Marqo/KAGL`
- **Total rows:** 44,434
- **Columns:** gender, category1, category2, category3, baseColour, season, year, usage, text, item_ID
- **Tasks (7):** text-to-image, category-to-product, sub-category-to-product, fine-category-to-product, color-to-product, season-to-product, usage-to-product

---

## Text Fields

### `text`

- Non-null: 44,434 / 44,434 (100.0%)
- Unique values: 31,131 (uniqueness: 70.1%)
- **Duplication:** 29.9% of texts are duplicates

**Word count distribution:** min=1, p10=4, median=6, mean=5.9, p90=8, max=15

**Length buckets:**
  - medium (6-10 words): 24,829 (55.9%)
  - short (3-5 words): 19,325 (43.5%)
  - long (11-20 words): 272 (0.6%)
  - very_short (1-2 words): 8 (0.0%)

**Text style:**
  - Has commas: 0.0%
  - Has periods: 1.7%
  - Has numbers: 10.6%
  - Starts uppercase: 99.3%
  - All lowercase: 0.0%

**Vocabulary:** 8,762 unique tokens across 263,499 total

**Top 30 tokens:**
  - `men`: 19,270
  - `women`: 14,746
  - `black`: 9,680
  - `shirt`: 7,598
  - `blue`: 6,601
  - `white`: 5,805
  - `t`: 5,487
  - `shoes`: 4,088
  - `printed`: 3,437
  - `brown`: 3,242
  - `red`: 3,103
  - `grey`: 3,044
  - `watch`: 2,413
  - `men's`: 2,361
  - `nike`: 2,326
  - `green`: 2,321
  - `casual`: 2,296
  - `of`: 2,111
  - `puma`: 2,102
  - `adidas`: 2,081
  - `unisex`: 2,035
  - `navy`: 1,846
  - `pink`: 1,766
  - `dial`: 1,743
  - `kurta`: 1,688
  - `polo`: 1,651
  - `top`: 1,588
  - `purple`: 1,564
  - `united`: 1,513
  - `sports`: 1,465

**Top 20 bigrams:**
  - `t shirt`: 5,207
  - `men black`: 2,691
  - `women black`: 1,584
  - `navy blue`: 1,553
  - `of benetton`: 1,426
  - `united colors`: 1,407
  - `colors of`: 1,402
  - `dial watch`: 1,292
  - `sports shoes`: 1,238
  - `men white`: 1,172
  - `puma men`: 1,154
  - `casual shoes`: 1,151
  - `nike men`: 1,139
  - `women printed`: 1,058
  - `adidas men`: 1,022
  - `men blue`: 1,016
  - `men brown`: 976
  - `women white`: 884
  - `women blue`: 843
  - `men grey`: 810

**Sample texts (first 20):**
  1. "Palm Tree Girls Sp Jace Sko White Skirts"
  2. "Palm Tree Kids Girls Sp Jema Skt Blue Skirts"
  3. "Palm Tree Kids Sp Jema Skt Blue Skirts"
  4. "Nike Women As Nike Eleme White T-Shirt"
  5. "Nike Men As 7 Sw Temp Grey Shorts"
  6. "Nike Men As Ss Trainin Blue T-Shirts"
  7. "Nike Men AS T90 Black Tshirts"
  8. "Nike Women As Trophy Swo White T-Shirts"
  9. "Nike Men Town Navy Blue T-Shirts"
  10. "Nike Men Town Round Red Neck T-Shirts"
  11. "Nike Men As Showdown F Black T-Shirts"
  12. "Nike Men Grey Melange Track Pants"
  13. "Nike Men's Breakli Navy Blue Track Pants"
  14. "Nike Men As Ss Cruiser Yellow T-Shirts"
  15. "Nike Unisex Feather lite Blue Caps"
  16. "Locomotive Men American Glory White Tshirts"
  17. "Palm Tree Girl's Cr Blossom Drs Pink Dresses"
  18. "Nike Women As The Windru Blue Jackets"
  19. "Nike Men Black Three Quarter Track Pants"
  20. "Nike Women As Miler Ss White T-Shirts"

**Most repeated texts (top 15):**
  - (82×) "Lucera Women Silver Earrings"
  - (56×) "Lucera Women Silver Pendant"
  - (50×) "Lucera Women Silver Ring"
  - (48×) "Catwalk Women Black Heels"
  - (42×) "Q&Q Men Black Dial Watch"
  - (41×) "Fastrack Men Black Dial Watch"
  - (38×) "Maxima Men White Dial Watch"
  - (37×) "Fabindia Women Anusuya Silver Earrings"
  - (32×) "FNF Multi Coloured Printed Sari"
  - (31×) "Coolers Men Black Sandals"
  - (31×) "Miami Blues Women Sunglasses"
  - (28×) "Murcia Women Brown Handbag"
  - (28×) "Murcia Women Black Handbag"
  - (28×) "Lucera Women Silver Pendant with Chain"
  - (27×) "Locomotive Men Washed Blue Jeans"

---

## Category Fields

### `category1`

- Non-null: 44,434 / 44,434
- Unique values: 7
- Top-5 concentration: 99.9%
- Top-10 concentration: 100.0%

**Full value distribution:**
  - `Apparel`: 21,395 (48.2%) ████████████████████████
  - `Accessories`: 11,287 (25.4%) ████████████
  - `Footwear`: 9,222 (20.8%) ██████████
  - `Personal Care`: 2,399 (5.4%) ██
  - `Free Items`: 105 (0.2%) █
  - `Sporting Goods`: 25 (0.1%) █
  - `Home`: 1 (0.0%) █

### `category2`

- Non-null: 44,434 / 44,434
- Unique values: 45
- Top-5 concentration: 69.8%
- Top-10 concentration: 83.2%

**Full value distribution:**
  - `Topwear`: 15,401 (34.7%) █████████████████
  - `Shoes`: 7,344 (16.5%) ████████
  - `Bags`: 3,053 (6.9%) ███
  - `Bottomwear`: 2,693 (6.1%) ███
  - `Watches`: 2,542 (5.7%) ██
  - `Innerwear`: 1,808 (4.1%) ██
  - `Jewellery`: 1,080 (2.4%) █
  - `Eyewear`: 1,073 (2.4%) █
  - `Fragrance`: 1,007 (2.3%) █
  - `Sandal`: 963 (2.2%) █
  - `Wallets`: 933 (2.1%) █
  - `Flip Flops`: 915 (2.1%) █
  - `Belts`: 811 (1.8%) █
  - `Socks`: 698 (1.6%) █
  - `Lips`: 527 (1.2%) █
  - `Dress`: 478 (1.1%) █
  - `Loungewear and Nightwear`: 470 (1.1%) █
  - `Saree`: 427 (1.0%) █
  - `Nails`: 329 (0.7%) █
  - `Makeup`: 307 (0.7%) █
  - `Headwear`: 293 (0.7%) █
  - `Ties`: 258 (0.6%) █
  - `Accessories`: 143 (0.3%) █
  - `Scarves`: 118 (0.3%) █
  - `Cufflinks`: 108 (0.2%) █
  - `Apparel Set`: 106 (0.2%) █
  - `Free Gifts`: 104 (0.2%) █
  - `Stoles`: 90 (0.2%) █
  - `Skin Care`: 77 (0.2%) █
  - `Skin`: 69 (0.2%) █
  - `Eyes`: 43 (0.1%) █
  - `Mufflers`: 38 (0.1%) █
  - `Shoe Accessories`: 24 (0.1%) █
  - `Sports Equipment`: 21 (0.0%) █
  - `Gloves`: 20 (0.0%) █
  - `Hair`: 19 (0.0%) █
  - `Bath and Body`: 12 (0.0%) █
  - `Water Bottle`: 7 (0.0%) █
  - `Umbrellas`: 6 (0.0%) █
  - `Perfumes`: 6 (0.0%) █
  - `Wristbands`: 4 (0.0%) █
  - `Beauty Accessories`: 4 (0.0%) █
  - `Sports Accessories`: 3 (0.0%) █
  - `Home Furnishing`: 1 (0.0%) █
  - `Vouchers`: 1 (0.0%) █

### `category3`

- Non-null: 44,434 / 44,434
- Unique values: 142
- Top-5 concentration: 39.9%
- Top-10 concentration: 57.3%

**Full value distribution:**
  - `Tshirts`: 7,069 (15.9%) ███████
  - `Shirts`: 3,215 (7.2%) ███
  - `Casual Shoes`: 2,846 (6.4%) ███
  - `Watches`: 2,542 (5.7%) ██
  - `Sports Shoes`: 2,036 (4.6%) ██
  - `Kurtas`: 1,844 (4.1%) ██
  - `Tops`: 1,762 (4.0%) █
  - `Handbags`: 1,759 (4.0%) █
  - `Heels`: 1,323 (3.0%) █
  - `Sunglasses`: 1,073 (2.4%) █
  - `Wallets`: 936 (2.1%) █
  - `Flip Flops`: 916 (2.1%) █
  - `Sandals`: 897 (2.0%) █
  - `Briefs`: 849 (1.9%) █
  - `Belts`: 813 (1.8%) █
  - `Backpacks`: 724 (1.6%) █
  - `Socks`: 686 (1.5%) █
  - `Formal Shoes`: 637 (1.4%) █
  - `Perfume and Body Mist`: 609 (1.4%) █
  - `Jeans`: 608 (1.4%) █
  - `Shorts`: 547 (1.2%) █
  - `Trousers`: 530 (1.2%) █
  - `Flats`: 500 (1.1%) █
  - `Bra`: 477 (1.1%) █
  - `Dresses`: 464 (1.0%) █
  - `Sarees`: 427 (1.0%) █
  - `Earrings`: 417 (0.9%) █
  - `Deodorant`: 347 (0.8%) █
  - `Nail Polish`: 329 (0.7%) █
  - `Lipstick`: 315 (0.7%) █
  - `Track Pants`: 304 (0.7%) █
  - `Clutches`: 288 (0.6%) █
  - `Sweatshirts`: 285 (0.6%) █
  - `Caps`: 283 (0.6%) █
  - `Sweaters`: 277 (0.6%) █
  - `Ties`: 263 (0.6%) █
  - `Jackets`: 258 (0.6%) █
  - `Innerwear Vests`: 242 (0.5%) █
  - `Kurtis`: 234 (0.5%) █
  - `Tunics`: 229 (0.5%) █
  - `Nightdress`: 189 (0.4%) █
  - `Leggings`: 177 (0.4%) █
  - `Pendant`: 176 (0.4%) █
  - `Capris`: 175 (0.4%) █
  - `Necklace and Chains`: 160 (0.4%) █
  - `Lip Gloss`: 144 (0.3%) █
  - `Night suits`: 141 (0.3%) █
  - `Trunk`: 140 (0.3%) █
  - `Skirts`: 128 (0.3%) █
  - `Scarves`: 119 (0.3%) █
  - `Ring`: 118 (0.3%) █
  - `Dupatta`: 116 (0.3%) █
  - `Accessory Gift Set`: 111 (0.2%) █
  - `Cufflinks`: 106 (0.2%) █
  - `Kajal and Eyeliner`: 102 (0.2%) █
  - `Kurta Sets`: 94 (0.2%) █
  - `Free Gifts`: 91 (0.2%) █
  - `Stoles`: 90 (0.2%) █
  - `Duffel Bag`: 88 (0.2%) █
  - `Bangle`: 85 (0.2%) █
  - `Laptop Bag`: 82 (0.2%) █
  - `Foundation and Primer`: 76 (0.2%) █
  - `Sports Sandals`: 67 (0.2%) █
  - `Bracelet`: 66 (0.1%) █
  - `Lounge Pants`: 61 (0.1%) █
  - `Face Moisturisers`: 61 (0.1%) █
  - `Jewellery Set`: 58 (0.1%) █
  - `Fragrance Gift Set`: 57 (0.1%) █
  - `Highlighter and Blush`: 53 (0.1%) █
  - `Boxers`: 52 (0.1%) █
  - `Compact`: 49 (0.1%) █
  - `Lip Liner`: 48 (0.1%) █
  - `Mobile Pouch`: 47 (0.1%) █
  - `Messenger Bag`: 44 (0.1%) █
  - `Eyeshadow`: 42 (0.1%) █
  - `Suspenders`: 40 (0.1%) █
  - `Camisoles`: 39 (0.1%) █
  - `Mufflers`: 38 (0.1%) █
  - `Patiala`: 38 (0.1%) █
  - `Lounge Shorts`: 34 (0.1%) █
  - `Jeggings`: 34 (0.1%) █
  - `Stockings`: 32 (0.1%) █
  - `Salwar`: 32 (0.1%) █
  - `Churidar`: 30 (0.1%) █
  - `Tracksuits`: 29 (0.1%) █
  - `Face Wash and Cleanser`: 28 (0.1%) █
  - `Sunscreen`: 25 (0.1%) █
  - `Shoe Accessories`: 23 (0.1%) █
  - `Gloves`: 20 (0.0%) █
  - `Bath Robe`: 20 (0.0%) █
  - `Hair Colour`: 19 (0.0%) █
  - `Rain Jacket`: 18 (0.0%) █
  - `Waist Pouch`: 17 (0.0%) █
  - `Swimwear`: 17 (0.0%) █
  - `Travel Accessory`: 16 (0.0%) █
  - `Jumpsuit`: 16 (0.0%) █
  - `Baby Dolls`: 16 (0.0%) █
  - `Lip Care`: 16 (0.0%) █
  - `Waistcoat`: 15 (0.0%) █
  - `Basketballs`: 13 (0.0%) █
  - `Mascara`: 13 (0.0%) █
  - `Booties`: 12 (0.0%) █
  - `Rompers`: 12 (0.0%) █
  - `Mask and Peel`: 12 (0.0%) █
  - `Rucksacks`: 11 (0.0%) █
  - `Water Bottle`: 11 (0.0%) █
  - `Concealer`: 11 (0.0%) █
  - `Tights`: 9 (0.0%) █
  - `Shapewear`: 9 (0.0%) █
  - `Footballs`: 8 (0.0%) █
  - `Blazers`: 8 (0.0%) █
  - `Clothing Set`: 8 (0.0%) █
  - `Headband`: 7 (0.0%) █
  - `Wristbands`: 7 (0.0%) █
  - `Salwar and Dupatta`: 7 (0.0%) █
  - `Umbrellas`: 6 (0.0%) █
  - `Shrug`: 6 (0.0%) █
  - `Eye Cream`: 6 (0.0%) █
  - `Nail Essentials`: 6 (0.0%) █
  - `Body Lotion`: 6 (0.0%) █
  - `Nehru Jackets`: 5 (0.0%) █
  - `Toner`: 5 (0.0%) █
  - `Face Scrub and Exfoliator`: 5 (0.0%) █
  - `Lehenga Choli`: 4 (0.0%) █
  - `Robe`: 4 (0.0%) █
  - `Lip Plumper`: 4 (0.0%) █
  - `Beauty Accessory`: 4 (0.0%) █
  - `Makeup Remover`: 4 (0.0%) █
  - `Hat`: 3 (0.0%) █
  - `Tablet Sleeve`: 3 (0.0%) █
  - `Trolley Bag`: 3 (0.0%) █
  - `Lounge Tshirts`: 3 (0.0%) █
  - `Ties and Cufflinks`: 2 (0.0%) █
  - `Key chain`: 2 (0.0%) █
  - `Rain Trousers`: 2 (0.0%) █
  - `Face Serum and Gel`: 2 (0.0%) █
  - `Cushion Covers`: 1 (0.0%) █
  - `Body Wash and Scrub`: 1 (0.0%) █
  - `Ipad`: 1 (0.0%) █
  - `Mens Grooming Kit`: 1 (0.0%) █
  - `Hair Accessory`: 1 (0.0%) █
  - `Shoe Laces`: 1 (0.0%) █

---

## Attribute Fields

### `gender`

- Non-null: 44,434 / 44,434
- Unique values: 5
- Top-5 concentration: 100.0%

**Full value distribution:**
  - `Men`: 22,157 (49.9%)
  - `Women`: 18,628 (41.9%)
  - `Unisex`: 2,164 (4.9%)
  - `Boys`: 830 (1.9%)
  - `Girls`: 655 (1.5%)

### `baseColour`

- Non-null: 44,424 / 44,434
- Unique values: 46
- Top-5 concentration: 59.5%

**Full value distribution:**
  - `Black`: 9,731 (21.9%)
  - `White`: 5,540 (12.5%)
  - `Blue`: 4,921 (11.1%)
  - `Brown`: 3,493 (7.9%)
  - `Grey`: 2,741 (6.2%)
  - `Red`: 2,456 (5.5%)
  - `Green`: 2,116 (4.8%)
  - `Pink`: 1,861 (4.2%)
  - `Navy Blue`: 1,791 (4.0%)
  - `Purple`: 1,643 (3.7%)
  - `Silver`: 1,090 (2.5%)
  - `Yellow`: 779 (1.8%)
  - `Beige`: 749 (1.7%)
  - `Gold`: 629 (1.4%)
  - `Maroon`: 580 (1.3%)
  - `Orange`: 530 (1.2%)
  - `Olive`: 410 (0.9%)
  - `Multi`: 394 (0.9%)
  - `Cream`: 389 (0.9%)
  - `Steel`: 315 (0.7%)
  - `Charcoal`: 228 (0.5%)
  - `Peach`: 195 (0.4%)
  - `Off White`: 182 (0.4%)
  - `Skin`: 179 (0.4%)
  - `Lavender`: 162 (0.4%)
  - `Grey Melange`: 146 (0.3%)
  - `Khaki`: 139 (0.3%)
  - `Magenta`: 129 (0.3%)
  - `Teal`: 120 (0.3%)
  - `Tan`: 114 (0.3%)
  - `Mustard`: 97 (0.2%)
  - `Bronze`: 95 (0.2%)
  - `Copper`: 86 (0.2%)
  - `Turquoise Blue`: 69 (0.2%)
  - `Rust`: 66 (0.1%)
  - `Burgundy`: 45 (0.1%)
  - `Metallic`: 43 (0.1%)
  - `Coffee Brown`: 31 (0.1%)
  - `Mauve`: 29 (0.1%)
  - `Rose`: 28 (0.1%)
  - `Nude`: 23 (0.1%)
  - `Sea Green`: 22 (0.0%)
  - `Mushroom Brown`: 16 (0.0%)
  - `Taupe`: 11 (0.0%)
  - `Lime Green`: 6 (0.0%)
  - `Fluorescent Green`: 5 (0.0%)

### `season`

- Non-null: 44,413 / 44,434
- Unique values: 4
- Top-5 concentration: 100.0%

**Full value distribution:**
  - `Summer`: 21,472 (48.3%)
  - `Fall`: 11,445 (25.8%)
  - `Winter`: 8,517 (19.2%)
  - `Spring`: 2,979 (6.7%)

### `usage`

- Non-null: 44,122 / 44,434
- Unique values: 8
- Top-5 concentration: 99.9%

**Full value distribution:**
  - `Casual`: 34,407 (78.0%)
  - `Sports`: 4,025 (9.1%)
  - `Ethnic`: 3,208 (7.3%)
  - `Formal`: 2,359 (5.3%)
  - `Smart Casual`: 67 (0.2%)
  - `Party`: 29 (0.1%)
  - `Travel`: 26 (0.1%)
  - `Home`: 1 (0.0%)

---

## Text ↔ Category Overlap

How often does the `text` field contain (or share words with) category labels?

### text vs `category1`
  - Text contains exact category string: 0.1%
  - Text shares words with category: 0.2%
  - **Total overlap: 0.2%**

### text vs `category2`
  - Text contains exact category string: 17.1%
  - Text shares words with category: 9.7%
  - **Total overlap: 26.8%**

### text vs `category3`
  - Text contains exact category string: 27.0%
  - Text shares words with category: 30.0%
  - **Total overlap: 57.0%**

### text vs `gender`
  - Text contains exact category string: 90.8%
  - Text shares words with category: 0.0%
  - **Total overlap: 90.8%**

### text vs `baseColour`
  - Text contains exact category string: 48.9%
  - Text shares words with category: 44.0%
  - **Total overlap: 92.8%**

### text vs `season`
  - Text contains exact category string: 0.4%
  - Text shares words with category: 0.0%
  - **Total overlap: 0.4%**

### text vs `usage`
  - Text contains exact category string: 5.0%
  - Text shares words with category: 4.9%
  - **Total overlap: 10.0%**

---

## Cross-Field Correlations

### gender × category1
- Unique combinations: 23

**Top 20 combinations:**
  - `Men | Apparel`: 11,350
  - `Women | Apparel`: 8,623
  - `Men | Footwear`: 5,751
  - `Women | Accessories`: 5,319
  - `Men | Accessories`: 4,426
  - `Women | Footwear`: 2,836
  - `Women | Personal Care`: 1,807
  - `Unisex | Accessories`: 1,497
  - `Boys | Apparel`: 759
  - `Men | Personal Care`: 577
  - `Girls | Apparel`: 567
  - `Unisex | Footwear`: 521
  - `Unisex | Apparel`: 96
  - `Girls | Footwear`: 60
  - `Boys | Footwear`: 54
  - `Men | Free Items`: 53
  - `Women | Free Items`: 43
  - `Girls | Accessories`: 28
  - `Unisex | Sporting Goods`: 25
  - `Boys | Accessories`: 17

### gender × category2
- Unique combinations: 120

**Top 20 combinations:**
  - `Men | Topwear`: 8,840
  - `Women | Topwear`: 5,499
  - `Men | Shoes`: 4,477
  - `Women | Shoes`: 2,555
  - `Women | Bags`: 2,069
  - `Men | Watches`: 1,473
  - `Men | Bottomwear`: 1,398
  - `Women | Bottomwear`: 1,044
  - `Women | Jewellery`: 1,015
  - `Men | Innerwear`: 988
  - `Women | Watches`: 902
  - `Unisex | Bags`: 891
  - `Women | Innerwear`: 811
  - `Men | Sandal`: 778
  - `Boys | Topwear`: 633
  - `Men | Eyewear`: 598
  - `Men | Fragrance`: 572
  - `Men | Belts`: 539
  - `Men | Socks`: 535
  - `Women | Lips`: 527

### gender × category3
- Unique combinations: 280

**Top 20 combinations:**
  - `Men | Tshirts`: 5,245
  - `Men | Shirts`: 2,842
  - `Men | Casual Shoes`: 2,247
  - `Women | Kurtas`: 1,761
  - `Women | Handbags`: 1,689
  - `Men | Sports Shoes`: 1,590
  - `Women | Tops`: 1,532
  - `Men | Watches`: 1,473
  - `Women | Heels`: 1,322
  - `Women | Tshirts`: 1,116
  - `Women | Watches`: 902
  - `Men | Sandals`: 749
  - `Men | Formal Shoes`: 637
  - `Unisex | Backpacks`: 631
  - `Men | Sunglasses`: 598
  - `Men | Briefs`: 564
  - `Men | Belts`: 541
  - `Men | Socks`: 535
  - `Boys | Tshirts`: 531
  - `Women | Flats`: 500

### gender × baseColour
- Unique combinations: 164

**Top 20 combinations:**
  - `Men | Black`: 5,884
  - `Men | White`: 3,103
  - `Women | Black`: 2,949
  - `Men | Blue`: 2,679
  - `Women | White`: 2,029
  - `Men | Brown`: 1,951
  - `Men | Grey`: 1,810
  - `Women | Blue`: 1,718
  - `Women | Pink`: 1,488
  - `Women | Brown`: 1,399
  - `Men | Navy Blue`: 1,309
  - `Men | Red`: 1,159
  - `Women | Purple`: 1,036
  - `Women | Green`: 963
  - `Women | Red`: 961
  - `Men | Green`: 911
  - `Women | Silver`: 823
  - `Unisex | Black`: 774
  - `Women | Grey`: 751
  - `Women | Gold`: 530

### gender × season
- Unique combinations: 20

**Top 20 combinations:**
  - `Men | Summer`: 10,875
  - `Women | Summer`: 8,432
  - `Men | Fall`: 7,087
  - `Women | Winter`: 4,620
  - `Women | Fall`: 3,660
  - `Men | Winter`: 3,225
  - `Women | Spring`: 1,909
  - `Men | Spring`: 962
  - `Unisex | Summer`: 938
  - `Boys | Summer`: 691
  - `Unisex | Winter`: 617
  - `Girls | Summer`: 536
  - `Unisex | Fall`: 504
  - `Boys | Fall`: 113
  - `Unisex | Spring`: 99
  - `Girls | Fall`: 81
  - `Girls | Winter`: 35
  - `Boys | Winter`: 20
  - `Boys | Spring`: 6
  - `Girls | Spring`: 3

### gender × usage
- Unique combinations: 25

**Top 20 combinations:**
  - `Men | Casual`: 16,768
  - `Women | Casual`: 14,365
  - `Women | Ethnic`: 3,083
  - `Men | Sports`: 2,954
  - `Men | Formal`: 2,249
  - `Unisex | Casual`: 1,830
  - `Boys | Casual`: 799
  - `Women | Sports`: 765
  - `Girls | Casual`: 645
  - `Unisex | Sports`: 283
  - `Women | Formal`: 109
  - `Men | Ethnic`: 107
  - `Men | Smart Casual`: 54
  - `Women | Party`: 27
  - `Boys | Sports`: 21
  - `Unisex | Travel`: 20
  - `Women | Smart Casual`: 13
  - `Boys | Ethnic`: 10
  - `Girls | Ethnic`: 8
  - `Women | Travel`: 4

### category1 × category2
- Unique combinations: 47

**Top 20 combinations:**
  - `Apparel | Topwear`: 15,401
  - `Footwear | Shoes`: 7,344
  - `Accessories | Bags`: 3,053
  - `Apparel | Bottomwear`: 2,693
  - `Accessories | Watches`: 2,542
  - `Apparel | Innerwear`: 1,808
  - `Accessories | Jewellery`: 1,080
  - `Accessories | Eyewear`: 1,073
  - `Personal Care | Fragrance`: 1,007
  - `Footwear | Sandal`: 963
  - `Accessories | Wallets`: 933
  - `Footwear | Flip Flops`: 915
  - `Accessories | Belts`: 811
  - `Accessories | Socks`: 686
  - `Personal Care | Lips`: 527
  - `Apparel | Dress`: 478
  - `Apparel | Loungewear and Nightwear`: 470
  - `Apparel | Saree`: 427
  - `Personal Care | Nails`: 329
  - `Personal Care | Makeup`: 307

### category1 × category3
- Unique combinations: 153

**Top 20 combinations:**
  - `Apparel | Tshirts`: 7,068
  - `Apparel | Shirts`: 3,215
  - `Footwear | Casual Shoes`: 2,846
  - `Accessories | Watches`: 2,542
  - `Footwear | Sports Shoes`: 2,036
  - `Apparel | Kurtas`: 1,844
  - `Apparel | Tops`: 1,762
  - `Accessories | Handbags`: 1,757
  - `Footwear | Heels`: 1,323
  - `Accessories | Sunglasses`: 1,073
  - `Accessories | Wallets`: 935
  - `Footwear | Flip Flops`: 916
  - `Footwear | Sandals`: 897
  - `Apparel | Briefs`: 849
  - `Accessories | Belts`: 810
  - `Accessories | Backpacks`: 722
  - `Accessories | Socks`: 686
  - `Footwear | Formal Shoes`: 637
  - `Apparel | Jeans`: 608
  - `Personal Care | Perfume and Body Mist`: 608

### category1 × baseColour
- Unique combinations: 198

**Top 20 combinations:**
  - `Apparel | Blue`: 3,435
  - `Accessories | Black`: 3,305
  - `Apparel | Black`: 3,202
  - `Footwear | Black`: 2,935
  - `Apparel | White`: 2,795
  - `Apparel | Grey`: 1,610
  - `Footwear | White`: 1,528
  - `Apparel | Green`: 1,467
  - `Apparel | Red`: 1,430
  - `Footwear | Brown`: 1,338
  - `Accessories | Brown`: 1,320
  - `Apparel | Navy Blue`: 1,270
  - `Apparel | Pink`: 1,107
  - `Apparel | Purple`: 1,001
  - `Accessories | White`: 984
  - `Accessories | Silver`: 812
  - `Accessories | Blue`: 712
  - `Apparel | Brown`: 565
  - `Apparel | Yellow`: 556
  - `Footwear | Grey`: 552

### category1 × season
- Unique combinations: 22

**Top 20 combinations:**
  - `Apparel | Summer`: 12,666
  - `Apparel | Fall`: 7,722
  - `Accessories | Winter`: 5,381
  - `Accessories | Summer`: 4,566
  - `Footwear | Summer`: 4,147
  - `Footwear | Fall`: 2,480
  - `Personal Care | Spring`: 2,361
  - `Footwear | Winter`: 2,208
  - `Accessories | Fall`: 1,231
  - `Apparel | Winter`: 869
  - `Footwear | Spring`: 367
  - `Apparel | Spring`: 137
  - `Accessories | Spring`: 109
  - `Free Items | Winter`: 56
  - `Free Items | Summer`: 44
  - `Personal Care | Summer`: 37
  - `Sporting Goods | Summer`: 12
  - `Sporting Goods | Fall`: 11
  - `Free Items | Spring`: 5
  - `Sporting Goods | Winter`: 2

### category1 × usage
- Unique combinations: 31

**Top 20 combinations:**
  - `Apparel | Casual`: 15,600
  - `Accessories | Casual`: 10,095
  - `Footwear | Casual`: 6,492
  - `Apparel | Ethnic`: 2,960
  - `Personal Care | Casual`: 2,138
  - `Footwear | Sports`: 2,031
  - `Apparel | Sports`: 1,595
  - `Apparel | Formal`: 1,172
  - `Footwear | Formal`: 654
  - `Accessories | Formal`: 527
  - `Accessories | Sports`: 375
  - `Accessories | Ethnic`: 207
  - `Free Items | Casual`: 78
  - `Accessories | Smart Casual`: 28
  - `Accessories | Travel`: 25
  - `Footwear | Ethnic`: 23
  - `Apparel | Party`: 22
  - `Apparel | Smart Casual`: 21
  - `Sporting Goods | Sports`: 21
  - `Footwear | Smart Casual`: 18

### category2 × category3
- Unique combinations: 168

**Top 20 combinations:**
  - `Topwear | Tshirts`: 7,068
  - `Topwear | Shirts`: 3,215
  - `Shoes | Casual Shoes`: 2,846
  - `Watches | Watches`: 2,542
  - `Shoes | Sports Shoes`: 2,036
  - `Topwear | Kurtas`: 1,844
  - `Topwear | Tops`: 1,762
  - `Bags | Handbags`: 1,757
  - `Shoes | Heels`: 1,323
  - `Eyewear | Sunglasses`: 1,073
  - `Wallets | Wallets`: 933
  - `Flip Flops | Flip Flops`: 915
  - `Sandal | Sandals`: 895
  - `Innerwear | Briefs`: 849
  - `Belts | Belts`: 810
  - `Bags | Backpacks`: 722
  - `Socks | Socks`: 686
  - `Shoes | Formal Shoes`: 637
  - `Bottomwear | Jeans`: 608
  - `Fragrance | Perfume and Body Mist`: 603

### category2 × baseColour
- Unique combinations: 852

**Top 20 combinations:**
  - `Topwear | Blue`: 2,449
  - `Shoes | Black`: 2,319
  - `Topwear | White`: 2,240
  - `Topwear | Black`: 2,023
  - `Shoes | White`: 1,448
  - `Topwear | Green`: 1,270
  - `Topwear | Red`: 1,189
  - `Topwear | Grey`: 1,101
  - `Shoes | Brown`: 1,035
  - `Watches | Black`: 987
  - `Topwear | Purple`: 833
  - `Bags | Black`: 823
  - `Topwear | Navy Blue`: 807
  - `Topwear | Pink`: 780
  - `Bottomwear | Blue`: 694
  - `Bottomwear | Black`: 611
  - `Watches | White`: 579
  - `Topwear | Yellow`: 492
  - `Jewellery | Silver`: 484
  - `Shoes | Grey`: 452

### category2 × season
- Unique combinations: 122

**Top 20 combinations:**
  - `Topwear | Summer`: 8,830
  - `Topwear | Fall`: 6,247
  - `Shoes | Summer`: 3,192
  - `Watches | Winter`: 2,474
  - `Shoes | Winter`: 1,933
  - `Shoes | Fall`: 1,920
  - `Bags | Summer`: 1,683
  - `Bottomwear | Summer`: 1,608
  - `Innerwear | Summer`: 1,397
  - `Eyewear | Winter`: 1,039
  - `Fragrance | Spring`: 1,003
  - `Bags | Winter`: 997
  - `Bottomwear | Fall`: 970
  - `Socks | Summer`: 569
  - `Belts | Summer`: 543
  - `Lips | Spring`: 516
  - `Jewellery | Summer`: 510
  - `Wallets | Summer`: 495
  - `Sandal | Summer`: 484
  - `Flip Flops | Summer`: 471

### category2 × usage
- Unique combinations: 108

**Top 20 combinations:**
  - `Topwear | Casual`: 11,077
  - `Shoes | Casual`: 4,647
  - `Bags | Casual`: 2,929
  - `Watches | Casual`: 2,450
  - `Topwear | Ethnic`: 2,282
  - `Shoes | Sports`: 2,010
  - `Bottomwear | Casual`: 1,822
  - `Innerwear | Casual`: 1,784
  - `Topwear | Sports`: 1,085
  - `Eyewear | Casual`: 1,065
  - `Fragrance | Casual`: 1,000
  - `Sandal | Casual`: 937
  - `Jewellery | Casual`: 933
  - `Topwear | Formal`: 921
  - `Flip Flops | Casual`: 908
  - `Wallets | Casual`: 892
  - `Belts | Casual`: 676
  - `Shoes | Formal`: 645
  - `Socks | Casual`: 516
  - `Bottomwear | Sports`: 463

### category3 × baseColour
- Unique combinations: 1,882

**Top 20 combinations:**
  - `Tshirts | White`: 1,119
  - `Tshirts | Black`: 1,049
  - `Tshirts | Blue`: 1,046
  - `Watches | Black`: 987
  - `Sports Shoes | White`: 898
  - `Casual Shoes | Black`: 875
  - `Shirts | Blue`: 785
  - `Tshirts | Grey`: 628
  - `Tshirts | Red`: 601
  - `Tshirts | Green`: 591
  - `Watches | White`: 579
  - `Sports Shoes | Black`: 554
  - `Shirts | White`: 526
  - `Casual Shoes | Brown`: 511
  - `Jeans | Blue`: 445
  - `Tshirts | Navy Blue`: 438
  - `Casual Shoes | White`: 435
  - `Formal Shoes | Black`: 397
  - `Sunglasses | Black`: 392
  - `Sandals | Black`: 349

### category3 × season
- Unique combinations: 341

**Top 20 combinations:**
  - `Tshirts | Summer`: 4,388
  - `Tshirts | Fall`: 2,508
  - `Watches | Winter`: 2,474
  - `Shirts | Fall`: 1,785
  - `Casual Shoes | Summer`: 1,417
  - `Shirts | Summer`: 1,336
  - `Tops | Summer`: 1,274
  - `Sports Shoes | Summer`: 1,191
  - `Kurtas | Summer`: 1,157
  - `Sunglasses | Winter`: 1,039
  - `Heels | Winter`: 1,021
  - `Handbags | Summer`: 997
  - `Casual Shoes | Fall`: 881
  - `Sports Shoes | Fall`: 724
  - `Briefs | Summer`: 691
  - `Kurtas | Fall`: 685
  - `Handbags | Winter`: 639
  - `Perfume and Body Mist | Spring`: 609
  - `Socks | Summer`: 557
  - `Belts | Summer`: 542

### category3 × usage
- Unique combinations: 249

**Top 20 combinations:**
  - `Tshirts | Casual`: 6,112
  - `Casual Shoes | Casual`: 2,813
  - `Watches | Casual`: 2,450
  - `Shirts | Casual`: 2,313
  - `Sports Shoes | Sports`: 1,996
  - `Kurtas | Ethnic`: 1,837
  - `Handbags | Casual`: 1,746
  - `Tops | Casual`: 1,700
  - `Heels | Casual`: 1,287
  - `Sunglasses | Casual`: 1,065
  - `Tshirts | Sports`: 955
  - `Flip Flops | Casual`: 909
  - `Wallets | Casual`: 895
  - `Sandals | Casual`: 874
  - `Shirts | Formal`: 866
  - `Briefs | Casual`: 847
  - `Belts | Casual`: 675
  - `Backpacks | Casual`: 630
  - `Jeans | Casual`: 608
  - `Formal Shoes | Formal`: 607

### baseColour × season
- Unique combinations: 173

**Top 20 combinations:**
  - `Black | Summer`: 4,472
  - `White | Summer`: 2,840
  - `Black | Winter`: 2,555
  - `Blue | Summer`: 2,527
  - `Black | Fall`: 2,289
  - `Blue | Fall`: 1,595
  - `White | Fall`: 1,518
  - `Grey | Summer`: 1,416
  - `Brown | Summer`: 1,395
  - `Red | Summer`: 1,272
  - `Brown | Winter`: 1,093
  - `Green | Summer`: 1,085
  - `Navy Blue | Summer`: 1,025
  - `Pink | Summer`: 916
  - `White | Winter`: 887
  - `Grey | Fall`: 886
  - `Purple | Summer`: 728
  - `Red | Fall`: 704
  - `Brown | Fall`: 673
  - `Green | Fall`: 672

### baseColour × usage
- Unique combinations: 181

**Top 20 combinations:**
  - `Black | Casual`: 7,473
  - `Blue | Casual`: 3,867
  - `White | Casual`: 3,792
  - `Brown | Casual`: 2,873
  - `Grey | Casual`: 2,123
  - `Red | Casual`: 1,972
  - `Green | Casual`: 1,612
  - `Pink | Casual`: 1,529
  - `Navy Blue | Casual`: 1,439
  - `Purple | Casual`: 1,247
  - `White | Sports`: 1,199
  - `Black | Sports`: 1,188
  - `Silver | Casual`: 947
  - `Black | Formal`: 749
  - `Yellow | Casual`: 658
  - `Beige | Casual`: 581
  - `Gold | Casual`: 542
  - `Maroon | Casual`: 416
  - `Grey | Sports`: 407
  - `Orange | Casual`: 395

### season × usage
- Unique combinations: 28

**Top 20 combinations:**
  - `Summer | Casual`: 16,309
  - `Winter | Casual`: 7,993
  - `Fall | Casual`: 7,552
  - `Spring | Casual`: 2,553
  - `Summer | Sports`: 2,135
  - `Summer | Ethnic`: 1,887
  - `Fall | Sports`: 1,642
  - `Fall | Ethnic`: 1,212
  - `Summer | Formal`: 1,079
  - `Fall | Formal`: 994
  - `Winter | Formal`: 247
  - `Winter | Sports`: 117
  - `Spring | Sports`: 110
  - `Winter | Ethnic`: 105
  - `Spring | Formal`: 39
  - `Winter | Smart Casual`: 32
  - `Fall | Smart Casual`: 22
  - `Winter | Travel`: 12
  - `Summer | Travel`: 12
  - `Fall | Party`: 11

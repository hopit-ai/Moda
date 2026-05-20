# polyvore — Benchmark Pattern Analysis

- **HuggingFace:** `Marqo/polyvore`
- **Total rows:** 94,096
- **Columns:** category, text, item_ID
- **Tasks (2):** text-to-image, category-to-product

---

## Text Fields

### `text`

- Non-null: 94,096 / 94,096 (100.0%)
- Unique values: 94,096 (uniqueness: 100.0%)
- **Duplication:** 0.0% of texts are duplicates

**Word count distribution:** min=0, p10=3, median=5, mean=5.4, p90=8, max=32

**Length buckets:**
  - short (3-5 words): 47,101 (50.1%)
  - medium (6-10 words): 36,070 (38.3%)
  - very_short (1-2 words): 7,531 (8.0%)
  - long (11-20 words): 3,317 (3.5%)
  - very_long (21+ words): 77 (0.1%)

**Text style:**
  - Has commas: 0.0%
  - Has periods: 1.6%
  - Has numbers: 3.6%
  - Starts uppercase: 0.0%
  - All lowercase: 99.9%

**Vocabulary:** 2,664 unique tokens across 508,987 total

**Top 30 tokens:**
  - `leather`: 7,795
  - `black`: 7,711
  - `bag`: 5,669
  - `women's`: 4,676
  - `dress`: 4,306
  - `gold`: 4,103
  - `top`: 3,723
  - `white`: 3,351
  - `earrings`: 3,231
  - `necklace`: 3,132
  - `skirt`: 2,979
  - `iphone`: 2,615
  - `jeans`: 2,600
  - `shirt`: 2,589
  - `lace`: 2,541
  - `blue`: 2,500
  - `boots`: 2,411
  - `jacket`: 2,381
  - `suede`: 2,371
  - `high`: 2,366
  - `print`: 2,312
  - `mini`: 2,251
  - `ring`: 2,186
  - `shoulder`: 2,179
  - `bracelet`: 2,166
  - `clutch`: 2,159
  - `case`: 2,158
  - `women`: 2,156
  - `long`: 2,138
  - `denim`: 2,057

**Top 20 bigrams:**
  - `pre owned`: 1,762
  - `shoulder bag`: 1,327
  - `t shirt`: 1,082
  - `long sleeve`: 1,025
  - `shein sheinside`: 977
  - `saint laurent`: 935
  - `case iphone`: 867
  - `plus size`: 856
  - `michael kors`: 844
  - `ankle boots`: 812
  - `dolce gabbana`: 801
  - `iphone plus`: 775
  - `new york`: 740
  - `stud earrings`: 736
  - `iphone case`: 722
  - `kate spade`: 702
  - `skinny jeans`: 697
  - `lace up`: 635
  - `crop top`: 626
  - `river island`: 614

**Sample texts (first 20):**
  1. "tibi knit long sleeve dress"
  2. "michael kors leather over-the-knee boots"
  3. "givenchy leather medium antigona duffel black"
  4. "bottega veneta acetate leather sunglasses"
  5. "pier imports stem"
  6. "miranda coat"
  7. "three pocket blazer"
  8. "givenchy skinny jean"
  9. "guess black silver-tone chronograph watch"
  10. "ray-ban original wayfarer sunglasses"
  11. "barrel clip key"
  12. "design key ring magnetic"
  13. "contrast trimmed cotton shirt"
  14. "rochas dress"
  15. "phillip lim red leather satchel"
  16. "classic drop military metal aviator sunglasses"
  17. "river island blue embroidered cut playsuit"
  18. "beige crystal sandals"
  19. "madden girl polka dot backpack"
  20. "naked strawberry banana oz"

**Most repeated texts (top 15):**
  - (1×) "tibi knit long sleeve dress"
  - (1×) "michael kors leather over-the-knee boots"
  - (1×) "givenchy leather medium antigona duffel black"
  - (1×) "bottega veneta acetate leather sunglasses"
  - (1×) "pier imports stem"
  - (1×) "miranda coat"
  - (1×) "three pocket blazer"
  - (1×) "givenchy skinny jean"
  - (1×) "guess black silver-tone chronograph watch"
  - (1×) "ray-ban original wayfarer sunglasses"
  - (1×) "barrel clip key"
  - (1×) "design key ring magnetic"
  - (1×) "contrast trimmed cotton shirt"
  - (1×) "rochas dress"
  - (1×) "phillip lim red leather satchel"

---

## Category Fields

### `category`

- Non-null: 94,096 / 94,096
- Unique values: 377
- Top-5 concentration: 17.1%
- Top-10 concentration: 30.0%

**Full value distribution:**
  - `Earrings`: 3,606 (3.8%) █
  - `Shoulder Bags`: 3,525 (3.7%) █
  - `Necklaces`: 3,379 (3.6%) █
  - `Tops`: 2,851 (3.0%) █
  - `Sandals`: 2,689 (2.9%) █
  - `Day Dresses`: 2,525 (2.7%) █
  - `Bracelets & Bangles`: 2,505 (2.7%) █
  - `Pumps`: 2,422 (2.6%) █
  - `Clutches`: 2,377 (2.5%) █
  - `Ankle Booties`: 2,355 (2.5%) █
  - `Rings`: 2,116 (2.2%) █
  - `Jackets`: 2,095 (2.2%) █
  - `Sweaters`: 2,037 (2.2%) █
  - `Sunglasses`: 2,021 (2.1%) █
  - `Coats`: 1,986 (2.1%) █
  - `Handbags`: 1,863 (2.0%) █
  - `Tech Accessories`: 1,693 (1.8%) █
  - `Knee Length Skirts`: 1,692 (1.8%) █
  - `Sneakers`: 1,578 (1.7%) █
  - `Tote Bags`: 1,512 (1.6%) █
  - `Blouses`: 1,481 (1.6%) █
  - `T-Shirts`: 1,478 (1.6%) █
  - `Shorts`: 1,451 (1.5%) █
  - `Hats`: 1,417 (1.5%) █
  - `Skinny Jeans`: 1,373 (1.5%) █
  - `Pants`: 1,235 (1.3%) █
  - `Tank Tops`: 1,158 (1.2%) █
  - `Clothing`: 1,146 (1.2%) █
  - `Backpacks`: 1,109 (1.2%) █
  - `Watches`: 1,099 (1.2%) █
  - `Cocktail Dresses`: 1,064 (1.1%) █
  - `Lipstick`: 962 (1.0%) █
  - `Fragrance`: 843 (0.9%) █
  - `Shoes`: 837 (0.9%) █
  - `Mini Skirts`: 825 (0.9%) █
  - `Flats`: 786 (0.8%) █
  - `Eyeshadow`: 759 (0.8%) █
  - `Scarves`: 745 (0.8%) █
  - `Hair Accessories`: 714 (0.8%) █
  - `Nail Polish`: 687 (0.7%) █
  - `Blazers`: 625 (0.7%) █
  - `Boots`: 585 (0.6%) █
  - `Floral Decor`: 580 (0.6%) █
  - `Home Decor`: 565 (0.6%) █
  - `Cardigans`: 526 (0.6%) █
  - `Sweatshirts`: 518 (0.6%) █
  - `Jeans`: 512 (0.5%) █
  - `Wallets`: 423 (0.4%) █
  - `Belts`: 412 (0.4%) █
  - `Dresses`: 383 (0.4%) █
  - `Leggings`: 370 (0.4%) █
  - `Mascara`: 369 (0.4%) █
  - `Jewelry`: 367 (0.4%) █
  - `Beauty Products`: 343 (0.4%) █
  - `Eyeliner`: 342 (0.4%) █
  - `Accessories`: 330 (0.4%) █
  - `Makeup`: 309 (0.3%) █
  - `Gowns`: 293 (0.3%) █
  - `Stationery`: 289 (0.3%) █
  - `Office Accessories`: 286 (0.3%) █
  - `Capri & Cropped Pants`: 281 (0.3%) █
  - `Boyfriend Jeans`: 278 (0.3%) █
  - `Lip Gloss`: 278 (0.3%) █
  - `Eye Makeup`: 276 (0.3%) █
  - `Throw Pillows`: 275 (0.3%) █
  - `Accent Tables`: 267 (0.3%) █
  - `Vests`: 264 (0.3%) █
  - `Hair Styling Tools`: 248 (0.3%) █
  - `Makeup Brushes`: 247 (0.3%) █
  - `Long Skirts`: 245 (0.3%) █
  - `Hoodies`: 240 (0.3%) █
  - `Brooches`: 232 (0.2%) █
  - `Outerwear`: 227 (0.2%) █
  - `Loafers & Moccasins`: 225 (0.2%) █
  - `Skirts`: 223 (0.2%) █
  - `Eyeglasses`: 221 (0.2%) █
  - `Socks`: 220 (0.2%) █
  - `Holiday Decorations`: 213 (0.2%) █
  - `Drinkware`: 210 (0.2%) █
  - `Bags`: 206 (0.2%) █
  - `Bras`: 206 (0.2%) █
  - `Ceiling Lights`: 205 (0.2%) █
  - `Knee High Boots`: 193 (0.2%) █
  - `Rugs`: 186 (0.2%) █
  - `Lip Makeup`: 185 (0.2%) █
  - `Blush`: 180 (0.2%) █
  - `Gloves`: 173 (0.2%) █
  - `Oxfords`: 171 (0.2%) █
  - `Men's Jeans`: 168 (0.2%) █
  - `Candles & Candleholders`: 163 (0.2%) █
  - `Men's Watches`: 161 (0.2%) █
  - `Over-The-Knee Boots`: 159 (0.2%) █
  - `Font`: 158 (0.2%) █
  - `Nail Treatments`: 155 (0.2%) █
  - `Sofas`: 154 (0.2%) █
  - `Accent Chairs`: 153 (0.2%) █
  - `Messenger Bags`: 153 (0.2%) █
  - `Men's Sneakers`: 153 (0.2%) █
  - `Rompers`: 152 (0.2%) █
  - `Lip Treatments`: 151 (0.2%) █
  - `Food & Drink`: 150 (0.2%) █
  - `Foundation`: 149 (0.2%) █
  - `Electronics`: 148 (0.2%) █
  - `Face Powder`: 142 (0.2%) █
  - `Tights`: 141 (0.1%) █
  - `Men's T-Shirts`: 141 (0.1%) █
  - `Charms & Pendants`: 139 (0.1%) █
  - `Luggage`: 136 (0.1%) █
  - `Baby`: 129 (0.1%) █
  - `Blankets`: 128 (0.1%) █
  - `Kitchen & Dining`: 127 (0.1%) █
  - `Straight Leg Jeans`: 126 (0.1%) █
  - `Table Lamps`: 123 (0.1%) █
  - `Body Moisturizers`: 123 (0.1%) █
  - `Men's Jackets`: 122 (0.1%) █
  - `Men's Casual Shirts`: 121 (0.1%) █
  - `Small Storage`: 120 (0.1%) █
  - `Activewear Pants`: 116 (0.1%) █
  - `Bikinis`: 116 (0.1%) █
  - `Kids`: 113 (0.1%) █
  - `Body Cleansers`: 113 (0.1%) █
  - `Books`: 110 (0.1%) █
  - `Sports & Outdoors`: 105 (0.1%) █
  - `Jumpsuits`: 104 (0.1%) █
  - `Panties & Thongs`: 103 (0.1%) █
  - `Men's Casual Pants`: 103 (0.1%) █
  - `Vases`: 102 (0.1%) █
  - `Face Makeup`: 100 (0.1%) █
  - `Tunics`: 100 (0.1%) █
  - `Men's Hats`: 98 (0.1%) █
  - `Slippers`: 96 (0.1%) █
  - `Men's Sunglasses`: 95 (0.1%) █
  - `Bags & Cases`: 92 (0.1%) █
  - `Shapewear`: 91 (0.1%) █
  - `Mid Calf Boots`: 89 (0.1%) █
  - `Clocks`: 88 (0.1%) █
  - `Serveware`: 86 (0.1%) █
  - `Flip Flops`: 85 (0.1%) █
  - `Outdoor Decor`: 85 (0.1%) █
  - `Body Art`: 83 (0.1%) █
  - `Girls`: 81 (0.1%) █
  - `Men's Wallets`: 78 (0.1%) █
  - `Bikini Tops`: 76 (0.1%) █
  - `One Piece Swimsuits`: 74 (0.1%) █
  - `Men's Fashion`: 74 (0.1%) █
  - `Face Care`: 74 (0.1%) █
  - `False Eyelashes`: 72 (0.1%) █
  - `Flared Jeans`: 71 (0.1%) █
  - `Wallpaper`: 68 (0.1%) █
  - `Chairs`: 68 (0.1%) █
  - `Men's Shoes`: 68 (0.1%) █
  - `Men's Clothing`: 67 (0.1%) █
  - `Men's Fragrance`: 66 (0.1%) █
  - `Cheek Bronzer`: 65 (0.1%) █
  - `Sports Bras`: 65 (0.1%) █
  - `Men's Backpacks`: 63 (0.1%) █
  - `Gift Sets & Kits`: 61 (0.1%) █
  - `Concealer`: 60 (0.1%) █
  - `Men's Tech Accessories`: 60 (0.1%) █
  - `Floor Lamps`: 59 (0.1%) █
  - `Nail Care`: 59 (0.1%) █
  - `Sun Care`: 58 (0.1%) █
  - `Men's Boots`: 57 (0.1%) █
  - `Toys`: 56 (0.1%) █
  - `Mirrors`: 55 (0.1%) █
  - `Lighting`: 55 (0.1%) █
  - `Men's Belts`: 52 (0.1%) █
  - `Haircare`: 51 (0.1%) █
  - `Bootcut Jeans`: 51 (0.1%) █
  - `Men's Bracelets`: 49 (0.1%) █
  - `Beds`: 48 (0.1%) █
  - `Lip Pencils`: 48 (0.1%) █
  - `Frames`: 47 (0.0%) █
  - `Athletic Shoes`: 47 (0.0%) █
  - `Brushes & Combs`: 46 (0.0%) █
  - `Dinnerware`: 45 (0.0%) █
  - `Men's Dress Shirts`: 44 (0.0%) █
  - `Bar Tools`: 43 (0.0%) █
  - `Bikini Bottoms`: 42 (0.0%) █
  - `Makeup Primer`: 42 (0.0%) █
  - `Costumes`: 41 (0.0%) █
  - `Styling Products`: 41 (0.0%) █
  - `Kitchen Gadgets & Tools`: 40 (0.0%) █
  - `Face Masks`: 40 (0.0%) █
  - `Curtains`: 37 (0.0%) █
  - `Dining Chairs`: 35 (0.0%) █
  - `Face Cleansers`: 35 (0.0%) █
  - `Bath Towels`: 35 (0.0%) █
  - `Face Moisturizers`: 35 (0.0%) █
  - `Men's Dress Shoes`: 34 (0.0%) █
  - `Flatware`: 33 (0.0%) █
  - `Blow Dryers & Irons`: 33 (0.0%) █
  - `Men's Shorts`: 32 (0.0%) █
  - `Ottomans`: 32 (0.0%) █
  - `Men's Coats`: 32 (0.0%) █
  - `Hair Color`: 32 (0.0%) █
  - `Cover-ups`: 32 (0.0%) █
  - `Boys`: 31 (0.0%) █
  - `Intimates`: 30 (0.0%) █
  - `Media`: 30 (0.0%) █
  - `Table Linens`: 30 (0.0%) █
  - `Men's Dress Pants`: 30 (0.0%) █
  - `Men's Scarves`: 30 (0.0%) █
  - `Camisoles`: 29 (0.0%) █
  - `Beach Towels`: 28 (0.0%) █
  - `Dining Tables`: 28 (0.0%) █
  - `Sheets & Pillowcases`: 28 (0.0%) █
  - `Pajamas`: 28 (0.0%) █
  - `Men's Bags`: 27 (0.0%) █
  - `Wide Leg Jeans`: 27 (0.0%) █
  - `Outdoor Lighting`: 26 (0.0%) █
  - `Pets`: 25 (0.0%) █
  - `Umbrellas`: 25 (0.0%) █
  - `Food Storage Containers`: 24 (0.0%) █
  - `Ties`: 24 (0.0%) █
  - `Eye Care`: 24 (0.0%) █
  - `Wall Lights`: 22 (0.0%) █
  - `Eyewear`: 22 (0.0%) █
  - `Beauty Accessories`: 22 (0.0%) █
  - `Tinted Moisturizer`: 22 (0.0%) █
  - `Clogs`: 21 (0.0%) █
  - `Bed Accessories`: 21 (0.0%) █
  - `Men's Shirts`: 20 (0.0%) █
  - `Dressers`: 20 (0.0%) █
  - `Activewear Tank Tops`: 20 (0.0%) █
  - `Robes`: 20 (0.0%) █
  - `Activewear Shorts`: 20 (0.0%) █
  - `Men's Messenger Bags`: 20 (0.0%) █
  - `Men's Sweaters`: 20 (0.0%) █
  - `Bookcases`: 19 (0.0%) █
  - `Bath & Body`: 19 (0.0%) █
  - `Men's Eyeglasses`: 19 (0.0%) █
  - `Home Fragrance`: 19 (0.0%) █
  - `Men's Grooming`: 19 (0.0%) █
  - `Men's Key Rings & Chains`: 19 (0.0%) █
  - `Activewear`: 18 (0.0%) █
  - `Men's Sportcoats & Blazers`: 18 (0.0%) █
  - `Manicure Tools`: 18 (0.0%) █
  - `Men's Polos`: 18 (0.0%) █
  - `Men's Rings`: 18 (0.0%) █
  - `Outdoors`: 17 (0.0%) █
  - `Duvet Covers`: 17 (0.0%) █
  - `Desks`: 17 (0.0%) █
  - `Briefcases`: 16 (0.0%) █
  - `Office Chairs`: 16 (0.0%) █
  - `Men's Loafers & Moccasins`: 16 (0.0%) █
  - `Makeup Remover`: 15 (0.0%) █
  - `Men's Necklaces`: 15 (0.0%) █
  - `Swimwear`: 15 (0.0%) █
  - `Teapots`: 15 (0.0%) █
  - `Bath Accessories`: 14 (0.0%) █
  - `Men's Socks`: 14 (0.0%) █
  - `Men's Work Boots`: 14 (0.0%) █
  - `Bedding`: 14 (0.0%) █
  - `Men's Oxfords`: 14 (0.0%) █
  - `Furniture`: 13 (0.0%) █
  - `Men's Suits`: 13 (0.0%) █
  - `Hair Shampoo`: 13 (0.0%) █
  - `Storage & Organization`: 13 (0.0%) █
  - `Desk Lamps`: 13 (0.0%) █
  - `Lip Stain`: 13 (0.0%) █
  - `Tables`: 12 (0.0%) █
  - `Hair Conditioner`: 12 (0.0%) █
  - `Wedding Dresses`: 11 (0.0%) █
  - `Home Improvement`: 11 (0.0%) █
  - `Men's Tank Tops`: 11 (0.0%) █
  - `Bow Ties`: 11 (0.0%) █
  - `Patio Umbrellas`: 11 (0.0%) █
  - `Men's Briefcases`: 11 (0.0%) █
  - `Activewear Tops`: 11 (0.0%) █
  - `Men's Gloves`: 11 (0.0%) █
  - `Sideboards`: 10 (0.0%) █
  - `Eyelash Curlers`: 10 (0.0%) █
  - `Small Appliances`: 10 (0.0%) █
  - `Patio Furniture`: 10 (0.0%) █
  - `Outdoor Chairs`: 10 (0.0%) █
  - `Party Supplies`: 10 (0.0%) █
  - `Men's Shaving`: 10 (0.0%) █
  - `Men's Vests`: 9 (0.0%) █
  - `Men's Activewear Pants`: 9 (0.0%) █
  - `Hammocks & Swings`: 9 (0.0%) █
  - `Kitchen Linens`: 9 (0.0%) █
  - `Activewear Jackets`: 9 (0.0%) █
  - `Cheek Makeup`: 9 (0.0%) █
  - `Men's Accessories`: 9 (0.0%) █
  - `Storage & Shelves`: 8 (0.0%) █
  - `Hosiery`: 8 (0.0%) █
  - `Nightstands`: 8 (0.0%) █
  - `Cabinets`: 8 (0.0%) █
  - `Stools`: 8 (0.0%) █
  - `Children's Decor`: 8 (0.0%) █
  - `Jewelry Storage`: 8 (0.0%) █
  - `Fireplace Accessories`: 8 (0.0%) █
  - `Bed Pillows`: 8 (0.0%) █
  - `Suspenders`: 8 (0.0%) █
  - `Cuff Links`: 8 (0.0%) █
  - `Men's Hoodies`: 8 (0.0%) █
  - `Men's Watches & Jewelry`: 7 (0.0%) █
  - `Comforters`: 7 (0.0%) █
  - `Men's Bags & Wallets`: 7 (0.0%) █
  - `Men's Athletic Shoes`: 7 (0.0%) █
  - `Makeup Tools`: 7 (0.0%) █
  - `Outdoor Tables`: 6 (0.0%) █
  - `Men's Underwear`: 6 (0.0%) █
  - `Cookbooks`: 6 (0.0%) █
  - `Skincare`: 6 (0.0%) █
  - `Cookware`: 6 (0.0%) █
  - `Men's Pants`: 6 (0.0%) █
  - `Garden Tools`: 6 (0.0%) █
  - `Juniors`: 5 (0.0%) █
  - `Men's Sweatshirts`: 5 (0.0%) █
  - `Aprons`: 5 (0.0%) █
  - `Cleaning`: 5 (0.0%) █
  - `Men's Deodorant`: 5 (0.0%) █
  - `Children's Furniture`: 5 (0.0%) █
  - `Cutlery`: 5 (0.0%) █
  - `Men's Outerwear`: 5 (0.0%) █
  - `Men's Sandals`: 5 (0.0%) █
  - `Benches`: 4 (0.0%) █
  - `Outdoor Stools`: 4 (0.0%) █
  - `Panel Screens`: 4 (0.0%) █
  - `Bath`: 4 (0.0%) █
  - `Handkerchiefs`: 4 (0.0%) █
  - `Barstools`: 4 (0.0%) █
  - `Face Toners`: 4 (0.0%) █
  - `Baby Bedding`: 4 (0.0%) █
  - `Paint`: 4 (0.0%) █
  - `Maternity`: 4 (0.0%) █
  - `Deodorant`: 4 (0.0%) █
  - `Men's Swimwear`: 4 (0.0%) █
  - `Gift Cards`: 4 (0.0%) █
  - `Bar Cabinets`: 3 (0.0%) █
  - `Shower Curtains`: 3 (0.0%) █
  - `Napkin Rings`: 3 (0.0%) █
  - `Bath Rugs & Mats`: 3 (0.0%) █
  - `Armoires`: 3 (0.0%) █
  - `Entertainment Units`: 3 (0.0%) █
  - `Outdoor Loungers & Day Beds`: 3 (0.0%) █
  - `File Cabinets`: 3 (0.0%) █
  - `Children's Room`: 3 (0.0%) █
  - `Bakeware`: 3 (0.0%) █
  - `Suits`: 3 (0.0%) █
  - `Quilts & Coverlets`: 3 (0.0%) █
  - `Health`: 3 (0.0%) █
  - `Men's Gift Sets & Kits`: 3 (0.0%) █
  - `Men's Money Clips`: 3 (0.0%) █
  - `Men's Skincare`: 3 (0.0%) █
  - `Tweezers & Brow Tools`: 3 (0.0%) █
  - `Men's Umbrellas`: 3 (0.0%) █
  - `Standing Fans`: 2 (0.0%) █
  - `Children's Bedding`: 2 (0.0%) █
  - `Chemises`: 2 (0.0%) █
  - `Men's Activewear Tops`: 2 (0.0%) █
  - `Men's Activewear Shorts`: 2 (0.0%) █
  - `Fans`: 2 (0.0%) █
  - `Window Treatments`: 2 (0.0%) █
  - `Outdoor Patio Sets`: 2 (0.0%) █
  - `Window Blinds`: 2 (0.0%) █
  - `Men's Sleepwear`: 2 (0.0%) █
  - `Men's Slippers`: 2 (0.0%) █
  - `Men's Flip Flops`: 2 (0.0%) █
  - `Sharpeners`: 2 (0.0%) █
  - `Sports Accessories`: 2 (0.0%) █
  - `Outdoor Benches`: 1 (0.0%) █
  - `Men's Activewear Jackets`: 1 (0.0%) █
  - `Decorative Hardware`: 1 (0.0%) █
  - `Nursery Furniture`: 1 (0.0%) █
  - `Oral Care`: 1 (0.0%) █
  - `Bedspreads`: 1 (0.0%) █
  - `Hair Removal`: 1 (0.0%) █
  - `Men's Activewear`: 1 (0.0%) █
  - `Men's Grooming Bags`: 1 (0.0%) █
  - `Flooring`: 1 (0.0%) █
  - `Activewear Skirts`: 1 (0.0%) █
  - `Men's Hair Care`: 1 (0.0%) █
  - `Curtain Rods`: 1 (0.0%) █
  - `Sleepwear`: 1 (0.0%) █

---

## Text ↔ Category Overlap

How often does the `text` field contain (or share words with) category labels?

### text vs `category`
  - Text contains exact category string: 12.4%
  - Text shares words with category: 43.2%
  - **Total overlap: 55.6%**

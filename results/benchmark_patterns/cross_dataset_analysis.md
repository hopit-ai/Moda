# Cross-Dataset Benchmark Pattern Analysis

Comparative analysis across all Marqo benchmark datasets.

## 1. Dataset Overview

| Dataset | Rows | Columns | Tasks | Text Field? | Category Depth |
|---------|------|---------|-------|-------------|----------------|
| atlas | 78,370 | 5 | 2 | Yes | 2 levels |
| fashion200k | 201,624 | 5 | 4 | Yes | 3 levels |
| KAGL | 44,434 | 10 | 7 | Yes | 3 levels |
| polyvore | 94,096 | 3 | 2 | Yes | 1 levels |
| deepfashion_inshop | 52,591 | 7 | 4 | Yes | 3 levels |
| deepfashion_multimodal | 42,537 | 5 | 2 | Yes | 2 levels |
| iMaterialist | ~721K | ? | 3 | No (attribute only) | N/A |

## 2. Task Coverage Matrix

| Dataset | category-to-image | category-to-product | color-to-product | fine-category-to-product | neckline-to-image | season-to-product | style-to-image | sub-category-to-product | text-to-image | usage-to-product |
|---------|---|---|---|---|---|---|---|---|---|---|
| atlas | - | - | - | - | - | - | - | Y | Y | - |
| fashion200k | - | Y | - | Y | - | - | - | Y | Y | - |
| KAGL | - | Y | Y | Y | - | Y | - | Y | Y | Y |
| polyvore | - | Y | - | - | - | - | - | - | Y | - |
| deepfashion_inshop | - | Y | Y | - | - | - | - | Y | Y | - |
| deepfashion_multimodal | - | Y | - | - | - | - | - | - | Y | - |
| iMaterialist | Y | - | - | - | Y | - | Y | - | - | - |

## 3. Text Description Patterns

| Dataset | Median Words | Mean Words | P10 | P90 | Unique Texts | Vocab Size |
|---------|-------------|------------|-----|-----|-------------|------------|
| atlas | 7 | 7.0 | 5 | 9 | 16,720 | 4,327 |
| fashion200k | 37 | 38.0 | 23 | 56 | 187,550 | 5,181 |
| KAGL | 6 | 5.9 | 4 | 8 | 31,131 | 8,762 |
| polyvore | 5 | 5.4 | 3 | 8 | 94,096 | 2,664 |
| deepfashion_inshop | 69 | 69.8 | 46 | 93 | 7,959 | 8,172 |
| deepfashion_multimodal | 42 | 41.4 | 22 | 59 | 40,769 | 104 |
| iMaterialist | — | — | — | — | — | — |

## 4. Category Concentration

How concentrated are category distributions? (Higher = more skewed = easier task)

### atlas
- `category`: 3 unique, top-5 = 100.0%, top-10 = 100.0%
- `sub-category`: 34 unique, top-5 = 49.6%, top-10 = 83.5%

### fashion200k
- `category1`: 5 unique, top-5 = 100.0%, top-10 = 100.0%
- `category2`: 31 unique, top-5 = 26.2%, top-10 = 48.3%
- `category3`: 61676 unique, top-5 = 1.7%, top-10 = 2.8%

### KAGL
- `category1`: 7 unique, top-5 = 99.9%, top-10 = 100.0%
- `category2`: 45 unique, top-5 = 69.8%, top-10 = 83.2%
- `category3`: 142 unique, top-5 = 39.9%, top-10 = 57.3%

### polyvore
- `category`: 377 unique, top-5 = 17.1%, top-10 = 30.0%

### deepfashion_inshop
- `category1`: 2 unique, top-5 = 100.0%, top-10 = 100.0%
- `category2`: 16 unique, top-5 = 71.2%, top-10 = 91.2%
- `category3`: 8 unique, top-5 = 91.9%, top-10 = 100.0%

### deepfashion_multimodal
- `category1`: 2 unique, top-5 = 100.0%, top-10 = 100.0%
- `category2`: 16 unique, top-5 = 77.6%, top-10 = 94.1%

## 5. Shared Vocabulary Across Datasets

**Tokens in top-50 of ALL datasets:** []

**Unique to atlas:** ['abaya', 'basic', 'blazer', 'breasted', 'campus', 'checkered', 'dark', 'multicolor', 'regular', 'self', 'single', 'skinny', 'slim', 'sutra', 'sweatshirt', 'track', 'trousers', 'van', 'waistcoat']
**Unique to fashion200k:** ['be', 'by', 'collar', 'flowy', 'have', 'loose', 'made', 'material', 'pair', 'pattern', 'stretchy', 'worn', 'zipper']
**Unique to KAGL:** ['adidas', 'benetton', 'boys', 'brown', 'check', 'colors', 'dial', 'flip', 'handbag', 'kids', 'kurta', 'navy', 'nike', 'polo', 'puma', 'purple', 'shirts', 'shoe', 'shoes', 'striped', 'tshirts', 'unisex', 'united', 'wallet', 'watch', 'yellow']
**Unique to polyvore:** ['bag', 'boots', 'bracelet', 'case', 'clutch', 'coat', 'earrings', 'faux', 'gold', 'iphone', 'leather', 'mini', 'necklace', 'new', 'owned', 'plus', 'pre', 'set', 'shoulder', 'size', 'suede', 'vintage', 'yoins']
**Unique to deepfashion_inshop:** ['100', '25', '5', 'back', 'chest', 'cold', 'features', 'from', 'hand', 'hem', 'knit', 'lightweight', 'machine', 'measured', 'piece', 'polyester', 'rayon', 'shell', 'small', 'spandex', 'unlined', 'wash', 'woven', 'you', 'your']
**Unique to deepfashion_multimodal:** ['accessory', 'clothing', 'fabric', 'female', 'finger', 'graphic', 'her', 'his', 'lady', 'patterns', 'person', 'point', 'pure', 'short', 'tank', 'there', 'three', 'wearing', 'wears', 'wrist']

## 6. Category Taxonomy Comparison

**Category overlap between datasets:**

- atlas ∩ fashion200k: 3 shared (['jackets', 'shirts', 'skirts'])
- atlas ∩ KAGL: 9 shared (['boxers', 'jackets', 'jeans', 'lehenga choli', 'sarees', 'shirts', 'shorts', 'skirts', 'trousers'])
- atlas ∩ polyvore: 4 shared (['jackets', 'jeans', 'shorts', 'skirts'])
- atlas ∩ deepfashion_inshop: 4 shared (['jackets', 'shirts', 'shorts', 'skirts'])
- atlas ∩ deepfashion_multimodal: 4 shared (['jackets', 'shirts', 'shorts', 'skirts'])
- fashion200k ∩ KAGL: 6 shared (['dresses', 'jackets', 'leggings', 'shirts', 'skirts', 'tops'])
- fashion200k ∩ polyvore: 12 shared (['blouses', 'cocktail dresses', 'dresses', 'gowns', 'jackets', 'knee length skirts', 'leggings', 'mini skirts', 'pants', 'skirts', 't-shirts', 'tops'])
- fashion200k ∩ deepfashion_inshop: 7 shared (['blouses', 'dresses', 'jackets', 'leggings', 'pants', 'shirts', 'skirts'])
- fashion200k ∩ deepfashion_multimodal: 7 shared (['blouses', 'dresses', 'jackets', 'leggings', 'pants', 'shirts', 'skirts'])
- KAGL ∩ polyvore: 49 shared (['backpacks', 'belts', 'clutches', 'dresses', 'earrings', 'flats', 'flip flops', 'jackets', 'jeans', 'leggings', 'lip gloss', 'scarves', 'shoes', 'socks', 'sunglasses', 'suspenders', 'ties', 'tights', 'tunics', 'wallets']...)
- KAGL ∩ deepfashion_inshop: 9 shared (['dresses', 'jackets', 'leggings', 'rompers', 'shirts', 'shorts', 'skirts', 'sweaters', 'sweatshirts'])
- KAGL ∩ deepfashion_multimodal: 9 shared (['dresses', 'jackets', 'leggings', 'rompers', 'shirts', 'shorts', 'skirts', 'sweaters', 'sweatshirts'])
- polyvore ∩ deepfashion_inshop: 15 shared (['blouses', 'cardigans', 'coats', 'dresses', 'hoodies', 'jackets', 'jumpsuits', 'leggings', 'pants', 'rompers', 'shorts', 'skirts', 'sweaters', 'sweatshirts', 'vests'])
- polyvore ∩ deepfashion_multimodal: 11 shared (['blouses', 'cardigans', 'dresses', 'jackets', 'leggings', 'pants', 'rompers', 'shorts', 'skirts', 'sweaters', 'sweatshirts'])
- deepfashion_inshop ∩ deepfashion_multimodal: 18 shared (['blouses', 'cardigans', 'denim', 'dresses', 'graphic', 'jackets', 'leggings', 'men', 'pants', 'rompers', 'shirts', 'shorts', 'skirts', 'suiting', 'sweaters', 'sweatshirts', 'tees', 'women'])

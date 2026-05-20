# Training Data Design Implications

Based on deep analysis of all 7 Marqo benchmark datasets.

## Key Principle

We learn the **patterns** (query types, category structures, text styles) not the actual data.

Training data must cover these patterns using **non-benchmark sources** only.

---

## 1. Query Type Coverage Required

The benchmarks test these distinct query types:

### category-to-image
Tested in: iMaterialist

### category-to-product
Tested in: fashion200k, KAGL, polyvore, deepfashion_inshop, deepfashion_multimodal

### color-to-product
Tested in: KAGL, deepfashion_inshop

### fine-category-to-product
Tested in: fashion200k, KAGL

### neckline-to-image
Tested in: iMaterialist

### season-to-product
Tested in: KAGL

### style-to-image
Tested in: iMaterialist

### sub-category-to-product
Tested in: atlas, fashion200k, KAGL, deepfashion_inshop

### text-to-image
Tested in: atlas, fashion200k, KAGL, polyvore, deepfashion_inshop, deepfashion_multimodal

### usage-to-product
Tested in: KAGL

---

## 2. Text Description Patterns to Teach

(To be filled after analysis run with actual data patterns)

---

## 3. Category Granularity Levels

(To be filled after analysis run with actual category distributions)

---

## 4. Attribute Coverage Requirements

(To be filled after analysis run with actual attribute distributions)

---

## 5. Recommended Training Data Sources

| Source | What it covers | Benchmark overlap risk |
|--------|---------------|----------------------|
| Marqo-GS-10M (fashion 5M) | Real search queries + ranked results | None with eval benchmarks |
| DeepFashion (In-Shop) | Retail catalog + descriptions | LEAKAGE: used in benchmark |
| DeepFashion (Multimodal) | Multi-attribute fashion | LEAKAGE: used in benchmark |
| Open Images (fashion subset) | Diverse real-world images | None |
| LLM-generated captions | Any pattern we want to teach | None |

**Safe sources:** Marqo-GS-10M, Open Images, LLM-generated

**Must exclude from training:** All 7 benchmark datasets

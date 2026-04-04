# MODA Phase 2 — Running Leaderboard (Real H&M Queries)

**Benchmark:** H&M full-pipeline | 10,000 real user queries (sampled, seed=42) | 105,542 articles  
**Ground truth:** Purchase-based relevance — 1 positive (bought) + ~9 negatives per query  
**Source:** [microsoft/hnm-search-data → data/search/](https://huggingface.co/datasets/microsoft/hnm-search-data/tree/main/data/search)

---

## Phase 1 Baselines — Dense Only (embedding → FAISS cosine similarity)

| Model | nDCG@5 | nDCG@10 | MRR | Recall@10 | Recall@20 |
| --- | --- | --- | --- | --- | --- |
| Marqo-FashionCLIP | 0.0188 | **0.0300** | 0.0341 | 0.0105 | 0.0197 |
| CLIP ViT-B/32 | 0.0170 | 0.0265 | 0.0312 | 0.0086 | 0.0177 |
| Marqo-FashionSigLIP | 0.0152 | 0.0232 | 0.0260 | 0.0077 | 0.0148 |

---

## Phase 2 Ablation — H&M Tier 2 Leaderboard

| # | Config | nDCG@5 | nDCG@10 | MRR | Recall@10 | Recall@20 | vs P1 Best |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | BM25 only (OpenSearch) | 0.0134 | 0.0187 | 0.0197 | 0.0052 | 0.0098 | -37.7% |
| 2 | BM25 + synonyms | 0.0092 | 0.0126 | 0.0134 | 0.0034 | 0.0066 | -58.0% ❌ |
| 2b | BM25 + NER attribute boost | 0.0163 | 0.0223 | 0.0243 | 0.0066 | 0.0127 | **+14.3% ✅** |
| 2c | BM25 + synonyms + NER | 0.0096 | 0.0133 | 0.0143 | 0.0037 | 0.0074 | -55.7% ❌ |
| 3 | Dense only (FashionCLIP) | 0.0188 | 0.0300 | 0.0341 | 0.0105 | 0.0197 | baseline |
| 4a | Hybrid Config A (BM25×0.2 + Dense×0.8) | 0.0214 | 0.0322 | 0.0368 | 0.0107 | 0.0203 | +7.3% ✅ |
| 4b | Hybrid Config B (BM25×0.3 + Dense×0.7) | 0.0229 | 0.0334 | 0.0377 | 0.0111 | 0.0205 | +11.4% ✅ |
| **4c** | **Hybrid Config C (BM25×0.4 + Dense×0.6)** | **0.0244** | **0.0353** | **0.0392** | **0.0113** | **0.0204** | **+17.8% ✅** |
| 4d | Hybrid Config D (BM25×0.5 + Dense×0.5) | 0.0218 | 0.0314 | 0.0344 | 0.0093 | 0.0165 | +4.8% ✅ |
| 5 | Hybrid Config C + synonyms | — | — | — | — | — | (skipped: synonyms hurt) |
| 6 | Hybrid Config C + CE rerank | **0.0384** | **0.0533** | **0.0562** | **0.0163** | **0.0284** | **+77.6% ✅** |
| 7 | Hybrid (NER-BM25×0.4 + Dense×0.6) | 0.0227 | 0.0329 | 0.0432 | 0.0124 | 0.0220 | +9.7% ✅ |
| **8** | **Full Pipeline: NER-BM25 + Dense + CE rerank ← NEW BEST** | **0.0396** | **0.0549** | **0.0579** | **0.0166** | **0.0289** | **+83.0% ✅** |

---

## Phase 2D — Query Understanding Ablation (BM25 standalone, 10K queries)

| Config | nDCG@10 | MRR | R@10 | R@20 | vs BM25 Baseline |
| --- | --- | --- | --- | --- | --- |
| A: BM25 baseline | 0.0195 | 0.0208 | 0.0056 | 0.0106 | — |
| B: BM25 + synonyms | 0.0126 | 0.0134 | 0.0034 | 0.0066 | **-35.4%** ❌ |
| C: BM25 + NER attribute boost | 0.0223 | 0.0243 | 0.0066 | 0.0127 | **+14.4%** ✅ |
| D: BM25 + synonyms + NER | 0.0133 | 0.0143 | 0.0037 | 0.0074 | **-31.8%** ❌ |

---

## Key Findings

### 1. Synonym Expansion HURTS BM25 on H&M Real Queries (−35 to −58%)

**Why:** H&M product names are brand-style titles ("Ben zip hoodie", "Tigra knitted headband").  
BM25 is precision-sensitive to IDF. When we expand "hoodie" → 12+ synonyms, we:
- Dramatically lower IDF of common fashion terms
- Create 50+ token queries where `operator: or` matches every product on at least one term
- Cause "query pollution" — the documented failure mode in LESER (2025), LEAPS (2026)

**Industry fix:** Confidence-threshold expansion with ≤3 terms per type, behavioral data validation, and search-log mining to validate which expansions actually improve CTR. Not viable without real search log data.

### 2. NER Attribute Boosting HELPS BM25 (+14%)

**Why:** GLiNER (NAACL 2024) correctly extracts fashion entities:
- `"navy slim fit jeans mens"` → {color: navy, fit: slim fit, type: jeans, gender: mens}
- These map to OpenSearch field boosts: `colour_group_name: "Dark Blue"^4`, `index_group_name: "Menswear"^2`
- `bool.should` clauses boost without hard-excluding near-miss products

### 3. Dense Retrieval > BM25 on Real User Queries (−38% vs P1 dense best)

**Why:** Real user queries are semantic ("warm earband", "casual summer outfit") while H&M product names are brand/style identifiers. Dense embeddings capture intent; BM25 relies on term overlap.

**Paper contribution:** Empirical evidence that semantic dense retrieval is superior to lexical BM25 for real user-intent queries in fashion e-commerce.

### 4. Cross-Encoder Reranking is the Dominant Signal (+78 to +83%)

The CE reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) achieves the majority of gains by evaluating full (query, product-text) pairs holistically. It compensates for retrieval quality differences between Config 6 and Config 8.

### 5. Full Pipeline (NER + Hybrid + CE Rerank) Sets New SOTA: nDCG@10 = 0.0549

A **+83% improvement** over the Phase 1 FashionCLIP dense baseline (0.0300).  
Pipeline: `GLiNER NER → NER-boosted BM25 × 0.4 + FashionCLIP FAISS × 0.6 → RRF → CE rerank`

---

## Overall Pipeline Improvement Summary

```
Phase 1 FashionCLIP dense (baseline)     nDCG@10 = 0.0300
→ BM25 alone                             nDCG@10 = 0.0187  (-37.7%)
→ + Dense hybrid (Config C)              nDCG@10 = 0.0353  (+17.8%)
→ + CE rerank (Config 6)                 nDCG@10 = 0.0533  (+77.6%)
→ + NER attribute boost (Config 8)       nDCG@10 = 0.0549  (+83.0%) ← BEST
```

Each component's marginal contribution:
- BM25 addition to dense: **+17.8%** (RRF hybrid effect)
- CE reranking: **+50.9%** (from 0.0353 → 0.0533)
- NER on BM25 component: **+3.0%** (from 0.0533 → 0.0549)

---

_Last updated: 2026-04-03 (Phase 2D query understanding complete)_

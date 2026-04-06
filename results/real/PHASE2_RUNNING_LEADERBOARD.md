# MODA — Running Leaderboard (Real H&M Queries)

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
| 8 | Full Pipeline: NER-BM25 + Dense + CE rerank | 0.0396 | 0.0549 | 0.0579 | 0.0166 | 0.0289 | +83.0% ✅ |
| **10** | **ColBERT→CE cascade ← NEW BEST** | **0.0401** | **0.0553** | **0.0578** | **0.0165** | **—** | **+84.3% ✅** |

---

## Phase 2E — Reranker Comparison: ColBERT v2 Late Interaction (10K queries)

**Model:** `colbert-ir/colbertv2.0` (BERT-base + 768→128 projection, MaxSim scoring)  
**Pipeline:** NER-boosted Hybrid (BM25×0.4 + Dense×0.6) → RRF top-100 → rerank → top-50  
**Comparison:** Same data split (10K queries, seed=42), same retrieval pipeline as Config 7/8

| # | Config | nDCG@5 | nDCG@10 | MRR | Recall@10 | Recall@50 | vs P1 Best |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 7 | Hybrid NER baseline (no rerank) | 0.0227 | 0.0329 | 0.0432 | 0.0124 | 0.0464 | +9.7% |
| 9 | Hybrid NER → ColBERT@50 | 0.0334 | 0.0480 | 0.0513 | 0.0149 | 0.0546 | +60.0% ✅ |
| 8 | Hybrid NER → CE@50 (reference) | 0.0396 | 0.0549 | 0.0579 | 0.0166 | 0.0563 | +83.0% |
| **10** | **ColBERT@100 → CE@50 cascade** | **0.0401** | **0.0553** | **0.0578** | **0.0165** | **0.0546** | **+84.3% ✅** |

### Reranker Head-to-Head (vs Hybrid NER baseline)

| Reranker | nDCG@10 | Lift vs baseline | Lift vs P1 |
| --- | --- | --- | --- |
| None (baseline) | 0.0329 | — | +9.7% |
| ColBERT v2 (late interaction) | 0.0480 | **+45.9%** | **+60.0%** |
| Cross-Encoder (full cross-attn) | 0.0549 | **+66.9%** | **+83.0%** |
| ColBERT→CE cascade | **0.0553** | **+68.1%** | **+84.3%** |

> **Finding:** ColBERT v2 delivers a strong +45.9% lift over the hybrid baseline. The Cross-Encoder wins by +21pp on its own. However, the **ColBERT→CE cascade** (ColBERT narrows 100→50, then CE re-scores the top-50) edges out CE-alone by +0.8%, establishing a new best nDCG@10 = 0.0553. ColBERT's MaxSim pre-filtering surfaces slightly better candidates for CE to score.

---

## Phase 2F — Mixture of Encoders: Superlinked-style Structured Retrieval (10K queries) — EXPLORATORY, NOT CORE BENCHMARK

> **Note:** This experiment is exploratory and is **excluded from the core Phase 1–2 benchmark results**. Superlinked's MoE concept requires type-specific trained encoders. Our implementation reuses FashionCLIP for all four fields, which is not a fair test of the architecture. These numbers are included for transparency only.

**Architecture:** Multi-field FashionCLIP encoding with query-time NER-adaptive weighting  
**Product vector:** 4 parallel FashionCLIP embeddings per product:
- `text` = FashionCLIP(prod_name + detail_desc) — 512-dim
- `color` = FashionCLIP(colour_group_name) — 512-dim (50 unique categories)
- `type` = FashionCLIP(product_type_name) — 512-dim (131 unique categories)
- `group` = FashionCLIP(product_group_name) — 512-dim (19 unique categories)

**Query scoring:** `score = w_text·cos(q,p_text) + w_color·cos(NER_color,p_color) + w_type·cos(NER_type,p_type) + w_group·cos(NER_group,p_group)`  
**Adaptive weights:** w_color=0.25 / w_type=0.30 / w_group=0.15 (activated when NER detects entity)

| # | Config | nDCG@5 | nDCG@10 | MRR | Recall@10 | Recall@50 | vs P1 Best |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | Dense only (FashionCLIP) | 0.0161 | 0.0256 | 0.0356 | 0.0105 | 0.0461 | −14.7% |
| 11 | MoE retrieval (structured) | 0.0167 | 0.0264 | 0.0370 | 0.0109 | 0.0481 | −12.0% |
| 7 | Hybrid NER + Dense (baseline) | 0.0227 | 0.0329 | 0.0432 | 0.0124 | 0.0464 | +9.7% |
| 12 | Hybrid NER + MoE | 0.0223 | 0.0330 | 0.0437 | 0.0129 | 0.0481 | +10.0% |
| 8 | Hybrid NER + Dense + CE@50 | 0.0396 | 0.0549 | 0.0579 | 0.0166 | 0.0563 | +83.0% |
| 13 | Hybrid NER + MoE + CE@50 | 0.0393 | 0.0541 | 0.0582 | 0.0164 | 0.0578 | +80.3% |

### MoE vs Single-Vector Dense (retrieval stage only)

| Retriever | nDCG@10 | Lift vs Dense | Key Advantage |
| --- | --- | --- | --- |
| Dense (FashionCLIP) | 0.0256 | — | Semantic text similarity |
| MoE (structured) | 0.0264 | **+3.1%** | NER-driven category matching |
| Hybrid + Dense | 0.0329 | — | BM25 + semantic fusion |
| Hybrid + MoE | 0.0330 | **+0.3%** | Marginal gain from structure |

> **Finding:** The Mixture of Encoders approach provides a modest +3.1% lift over single-vector dense retrieval standalone. In the hybrid setting (BM25+retriever), the gain shrinks to +0.3% because BM25's NER-boosted field matching already captures much of the same categorical signal. With CE reranking, MoE+CE (0.0541) is within 1.5% of Dense+CE (0.0549) — the CE reranker equalizes retrieval-stage differences. MoE's main benefit is improved Recall@50 (0.0481 vs 0.0461), suggesting it surfaces a more diverse candidate pool.

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

### 5. ColBERT→CE Cascade Sets New SOTA: nDCG@10 = 0.0553

ColBERT's per-token MaxSim pre-filtering (100→50) followed by CE's full cross-attention scoring edges out CE-alone by +0.8%. ColBERT surfaces slightly better candidates for CE to evaluate.

### 6. Mixture of Encoders: Exploratory (Not Core Benchmark)

The Superlinked-style structured retrieval was tested with the same FashionCLIP encoder for all four fields — this is not a fair test of the MoE concept, which requires type-specific trained encoders. Results are included for transparency but excluded from the core Phase 1–2 benchmark. We plan to revisit with trained field-specific encoders in future work.

---

## Overall Pipeline Improvement Summary

```
Phase 1 FashionCLIP dense (baseline)     nDCG@10 = 0.0300
→ BM25 alone                             nDCG@10 = 0.0187  (-37.7%)
→ + Dense hybrid (Config C)              nDCG@10 = 0.0353  (+17.8%)
→ + CE rerank (Config 6)                 nDCG@10 = 0.0533  (+77.6%)
→ + NER attribute boost (Config 8)       nDCG@10 = 0.0549  (+83.0%)
→ + ColBERT→CE cascade (Config 10)       nDCG@10 = 0.0553  (+84.3%) ← BEST
```

Each component's marginal contribution:
- BM25 addition to dense: **+17.8%** (RRF hybrid effect)
- CE reranking: **+50.9%** (from 0.0353 → 0.0533)
- NER on BM25 component: **+3.0%** (from 0.0533 → 0.0549)
- ColBERT pre-filter for CE: **+0.7%** (from 0.0549 → 0.0553)

---

## Phase 3A — Fine-Tuned Cross-Encoder Evaluation (22,855 held-out test queries)

**Leakage prevention:** Evaluated exclusively on held-out test split (22,855 queries). Train/val/test split by *unique query text* — all query IDs sharing the same text go into the same split, preventing both direct and semantic leakage.  
**Models compared:** Off-the-shelf `cross-encoder/ms-marco-MiniLM-L-6-v2` vs fine-tuned `moda-fashion-ce-best` (trained on H&M purchase pairs from train split)  
**Pipeline:** NER-boosted Hybrid (BM25×0.4 + Dense×0.6) → RRF top-100 → CE rerank → top-50

| # | Config | nDCG@5 | nDCG@10 | MRR | Recall@10 | Recall@50 | vs Off-shelf CE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| — | Hybrid NER baseline (no rerank) | 0.0324 | 0.0422 | 0.0558 | 0.0142 | 0.0515 | −34.7% |
| 8' | Hybrid NER + Off-shelf CE@50 | 0.0442 | **0.0646** | **0.0671** | **0.0195** | 0.0620 | baseline |
| **14** | **Hybrid NER + Fine-tuned CE@50** | **0.0480** | **0.0654** | 0.0644 | 0.0183 | **0.0616** | **+1.2%** |

### Head-to-Head Analysis

| Metric | Off-shelf CE | Fine-tuned CE | Delta | Winner |
| --- | --- | --- | --- | --- |
| nDCG@5 | 0.0442 | **0.0480** | **+8.6%** | Fine-tuned ✅ |
| nDCG@10 | 0.0646 | **0.0654** | **+1.2%** | Fine-tuned ✅ |
| MRR | **0.0671** | 0.0644 | −4.0% | Off-shelf ✅ |
| Recall@10 | **0.0195** | 0.0183 | −6.2% | Off-shelf ✅ |
| Recall@50 | **0.0620** | 0.0616 | −0.6% | Off-shelf ✅ |

> **Finding:** The fine-tuned CE shows a **mixed result**. It improves nDCG@5 (+8.6%) and nDCG@10 (+1.2%), suggesting better ranking of top positions. However, MRR (−4.0%) and Recall@10 (−6.2%) are worse — indicating the fine-tuned model occasionally ranks the true positive outside the top-10 when the off-shelf model would catch it. The fine-tuning appears to sharpen discrimination at the very top of the ranked list (nDCG@5) at the cost of recall breadth. Overall, the off-the-shelf `ms-marco-MiniLM-L-6-v2` remains remarkably competitive — a strong zero-shot baseline that domain-specific fine-tuning only marginally improves on nDCG while slightly hurting other metrics.

---

## Phase 3B — LLM-Trained Cross-Encoder (22,855 held-out test queries) ← NEW SOTA

**Key insight:** Phase 3A showed fine-tuning on noisy purchase labels barely helped (+1.2%). Phase 3B replaces those labels with **42,800 LLM-judged relevance scores** (GPT-4o-mini via PaleblueDot API, graded 0-3) and retrains the CE with MSE loss on normalized labels.

**Training data:** 42,800 query-product pairs labeled by GPT-4o-mini with a fashion relevance rubric (0=irrelevant, 1=partial, 2=good, 3=exact match). Score distribution: 27.7% score-0, 21.1% score-1, 25.0% score-2, 26.2% score-3.  
**Training:** 3 epochs, batch=32, lr=2e-5, CECorrelationEvaluator → Spearman=0.903  
**Pipeline:** NER-boosted Hybrid (BM25×0.4 + Dense×0.6) → RRF top-100 → CE rerank → top-50

| # | Config | nDCG@10 | MRR | Recall@10 | vs Off-shelf CE |
| --- | --- | --- | --- | --- | --- |
| — | Hybrid NER baseline (no rerank) | 0.0422 | 0.0558 | 0.0142 | −34.7% |
| 8' | Hybrid NER + Off-shelf CE@50 | 0.0646 | 0.0671 | 0.0195 | baseline |
| 14 | Hybrid NER + Fine-tuned CE@50 (Phase 3A) | 0.0654 | 0.0644 | 0.0183 | +1.2% |
| **15** | **Hybrid NER + LLM-trained CE@50 (Phase 3B)** | **0.0747** | **0.0755** | **0.0217** | **+15.7% ✅** |

### Phase 3 Head-to-Head (all models, same 22,855 test queries)

| Metric | Off-shelf CE | Fine-tuned CE (3A) | LLM-trained CE (3B) | 3B vs Off-shelf |
| --- | --- | --- | --- | --- |
| nDCG@10 | 0.0646 | 0.0654 (+1.2%) | **0.0747** | **+15.7% ✅** |
| MRR | 0.0671 | 0.0644 (−4.0%) | **0.0755** | **+12.5% ✅** |
| Recall@10 | 0.0195 | 0.0183 (−6.2%) | **0.0217** | **+11.3% ✅** |

> **Finding:** LLM-judged labels are a **game-changer**. The LLM-trained CE improves **every metric** substantially over both the off-the-shelf and Phase 3A models:
> - **+15.7% nDCG@10** vs off-the-shelf (0.0646 → 0.0747)
> - **+14.2% nDCG@10** vs Phase 3A fine-tuned (0.0654 → 0.0747)
> - **+12.5% MRR** — better top-1 ranking
> - **+11.3% Recall@10** — catches more relevant items
>
> This proves the Phase 3A hypothesis: **data quality, not model capacity, was the bottleneck**. The same 22M-param MiniLM-L6 architecture, trained on 42.8K clean graded labels instead of 2.5M noisy binary purchase labels, delivers dramatically better results. GPT-4o-mini's semantic relevance judgments are a far more reliable training signal than purchase logs.

---

## Phase 3C — Fine-Tuned Bi-Encoder (22,855 held-out test queries)

**Key insight:** Instead of just improving the reranker, Phase 3C improves the *retriever* itself. We fine-tune FashionCLIP's text encoder with InfoNCE contrastive loss, using **retriever-mined hard negatives** labeled by GPT-4o-mini.

**Data generation:**
1. Sample 5,000 unique train queries (leakage-free — disjoint from test)
2. Retrieve top-20 from FashionCLIP for each → 100,000 (query, product) pairs
3. Label each with GPT-4o-mini (0-3 relevance scale) → 100K labels, 0 failures
4. Extract contrastive triplets: positive (score 2-3) + hard negative (score 0) → 24,433 triplets

**Training:** 5 epochs, batch=64 (grad_accum=4), lr=1e-6, InfoNCE + in-batch negatives + mined hard negatives, FP16 autocast on Apple M4 Max MPS — 41.6 min total  
**Val accuracy:** 66.5% (baseline) → 99.4% (fine-tuned) on triplet ranking task

| # | Config | nDCG@10 | MRR | Recall@10 | vs Baseline |
| --- | --- | --- | --- | --- | --- |
| 3 | Baseline FashionCLIP (dense only) | 0.0229 | 0.0208 | 0.0433 | baseline |
| **16** | **Fine-tuned FashionCLIP (dense only)** | **0.0444** | **0.0405** | **0.0811** | **+94.2% ✅** |

### Dense Retrieval Head-to-Head

| Metric | Baseline FashionCLIP | Fine-tuned FashionCLIP | Delta |
| --- | --- | --- | --- |
| nDCG@10 | 0.0229 | **0.0444** | **+94.2%** |
| MRR | 0.0208 | **0.0405** | **+94.7%** |
| Recall@10 | 0.0433 | **0.0811** | **+87.3%** |

> **Finding:** Fine-tuning FashionCLIP on retriever-mined hard negatives with LLM labels **nearly doubles** dense retrieval quality across all metrics. The model learns exactly where it previously failed — products that it ranked highly but were irrelevant according to GPT-4o-mini. This is the largest single-stage improvement in the entire project (+94.2%), and it happens at the *retrieval stage* rather than the reranking stage. Improved retrieval means the downstream reranker receives better candidates, potentially compounding gains.

---

## Phase 3 — Combined Pipeline Evaluation (2×3 Factorial, 22,855 test queries)

**Apples-to-apples comparison:** All 6 configs share the same BM25-NER retrieval, same test queries, same qrels.

### Retriever × Reranker Matrix (nDCG@10)

|  | No Rerank | Off-shelf CE | LLM-trained CE (3B) |
| --- | --- | --- | --- |
| Baseline FashionCLIP | 0.0422 | 0.0646 | 0.0747 |
| Fine-tuned FashionCLIP (3C) | 0.0515 (+22.0%) | 0.0650 (+0.6%) | **0.0757 (+1.3%)** |

### Full Results

| # | Config | nDCG@10 | MRR | R@10 | vs A0 baseline |
| --- | --- | --- | --- | --- | --- |
| A0 | Baseline Hybrid (no rerank) | 0.0422 | 0.0558 | 0.0142 | baseline |
| B0 | Fine-tuned Hybrid (no rerank) | 0.0515 | 0.0740 | 0.0188 | +22.0% |
| A1 | Baseline + Off-shelf CE | 0.0646 | 0.0671 | 0.0195 | +53.1% |
| B1 | Fine-tuned + Off-shelf CE | 0.0650 | 0.0723 | 0.0207 | +54.0% |
| A2 | Baseline + LLM-trained CE | 0.0747 | 0.0755 | 0.0217 | +77.0% |
| **B2** | **Fine-tuned + LLM-trained CE** | **0.0757** | **0.0799** | **0.0243** | **+79.4%** |

### Key Findings from Combined Evaluation

1. **B2 is the new project SOTA** (nDCG@10 = 0.0757), combining the fine-tuned retriever with LLM-trained CE
2. **LLM-trained CE is the dominant component**: A2 vs A1 = +15.7% nDCG, vs B1 vs A1 = +0.6% from retriever alone with off-shelf CE
3. **Retriever fine-tuning compounds with LLM CE**: B2 vs A2 = +1.3% nDCG, but +5.8% MRR and +12.0% Recall@10 — the fine-tuned retriever surfaces better candidates that the LLM CE can score
4. **Off-shelf CE equalises retriever gains**: B1 vs A1 = +0.6% nDCG — a generic CE compensates for retriever weaknesses, but gains appear in MRR (+7.7%) and Recall (+6.2%)
5. **Gains are sub-additive on nDCG but additive on Recall**: The retriever's +22% hybrid lift compresses to +1.3% after LLM CE on nDCG, but Recall@10 improves +12.0% — better candidates enter the pool even if CE re-orders similarly

---

## Overall Pipeline Improvement Summary

```
Phase 1 FashionCLIP dense (baseline)     nDCG@10 = 0.0300
→ BM25 alone                             nDCG@10 = 0.0187  (-37.7%)
→ + Dense hybrid (Config C)              nDCG@10 = 0.0353  (+17.8%)
→ + CE rerank (Config 6)                 nDCG@10 = 0.0533  (+77.6%)
→ + NER attribute boost (Config 8)       nDCG@10 = 0.0549  (+83.0%)
→ + ColBERT→CE cascade (Config 10)       nDCG@10 = 0.0553  (+84.3%) ← BEST Phase 2
→ + Fine-tuned CE (Phase 3A, purchase)   nDCG@10 = 0.0654  (+1.2% vs off-shelf CE)
→ + LLM-trained CE (Phase 3B, GPT-4o)   nDCG@10 = 0.0747  (+15.7% vs off-shelf CE)
→ + Fine-tuned bi-encoder (Phase 3C)     nDCG@10 = 0.0444  (+94.2% dense retrieval)
→ + Combined B2 (3C retriever + 3B CE)   nDCG@10 = 0.0757  (+1.3% vs 3B) ← NEW BEST
```

Each component's marginal contribution:
- BM25 addition to dense: **+17.8%** (RRF hybrid effect)
- CE reranking: **+50.9%** (from 0.0353 → 0.0533)
- NER on BM25 component: **+3.0%** (from 0.0533 → 0.0549)
- ColBERT pre-filter for CE: **+0.7%** (from 0.0549 → 0.0553)
- Fine-tuned CE on purchase data: **+1.2%** (marginal, noisy labels)
- **LLM-trained CE on GPT-4o labels: +15.7%** (clean graded labels — biggest reranker gain)
- **Fine-tuned bi-encoder: +94.2%** dense retrieval, **+1.3%** end-to-end with LLM CE
- **Combined best (B2): +79.4%** over baseline hybrid (from 0.0422 → 0.0757)

---

_Last updated: 2026-04-06 (Phase 3 combined eval complete — B2 SOTA nDCG@10=0.0757, MRR=0.0799, R@10=0.0243 on 22,855 test queries)_

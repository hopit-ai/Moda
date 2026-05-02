# MODA

**The first open-source, end-to-end benchmark for fashion search with a full component-by-component breakdown.**  
253,685 purchase-grounded queries · 105,542 H&M products · 40+ pipeline configs · nDCG@10 = 0.1063 on H&M text retrieval (+301% over dense baseline) · Fine R@1 = 67.68 on LookBench image-to-image retrieval (+3.84 over FashionSigLIP)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What is this?

Nobody has published open-source, full-pipeline fashion search benchmarks on purchase-grounded queries.  
Marqo has great embeddings. Algolia/Bloomreach are proprietary. Nobody has put it all together and measured what each piece contributes.

**MODA fills that gap.** We built a complete retrieval pipeline (BM25 + SPLADE + dense + hybrid + NER + cross-encoder reranking), ran it against 253,685 H&M purchase-grounded queries, isolated the contribution of every component, and published everything: code, results, methodology.

> **Note on query provenance:** The queries are synthetically generated from real H&M purchase data ([Microsoft's H&M Search Data](https://huggingface.co/datasets/microsoft/hnm-search-data)), not captured from actual search logs. The purchases are real; the queries are reconstructed. This is a known limitation. See the [blog post](blog_post.md) for details.

---

## Blog series

We are publishing this work as a series of technical blog posts, each covering one phase of the pipeline:

| Blog | Title | Focus | Key result |
|------|-------|-------|------------|
| [Blog 1](blog_post.md) | Building a zero-shot fashion search pipeline | BM25 + Dense + CE reranking | nDCG@10 = 0.0543 |
| [Blog 2](blog_post_phase2b.md) | The one swap that beat weeks of tuning | Replacing BM25 with SPLADE | nDCG@10 = 0.0748 (+38%) |
| [Blog 3](blog_post_phase3a_3b.md) | $25 beat everything we had built | Training the cross-encoder with LLM-graded labels | nDCG@10 = 0.0976 (+31%) |
| [Blog 4](blog_post_phase3c.md) | Training the retriever on its own mistakes | Fine-tuning FashionCLIP and SPLADE on hard negatives | nDCG@10 = 0.1063 (+9%) |
| [Blog 5](blog_post_phase4.md) | Adding eyes to the search engine | Three-Tower multimodal retriever (text + image) | nDCG@10 = 0.0833 (lateral) |
| [Blog 6](blog_post_phase5.md) | Beating FashionSigLIP | Cross-domain fine-tuning on LookBench | Fine R@1 = 67.68 (+3.84 over baseline) |

---

## Key results

### Phase 1: Zero-shot pipeline (253,685 queries, 105,542 products)

| Config | nDCG@10 | 95% CI | MRR | Recall@10 | Recall@50 |
|--------|---------|--------|-----|-----------|-----------|
| BM25 only | 0.0186 | [.0183-.0190] | 0.0227 | 0.0059 | 0.0251 |
| Dense only (FashionCLIP) | 0.0265 | [.0261-.0269] | 0.0369 | 0.0106 | 0.0462 |
| Hybrid (BM25x0.4 + Densex0.6) | 0.0328 | [.0324-.0333] | 0.0429 | 0.0121 | 0.0457 |
| **Full pipeline (Hybrid + CE rerank)** | **0.0543** | **[.0537-.0550]** | **0.0569** | **0.0164** | **0.0559** |

### Phase 2B: SPLADE swap (22,855 held-out test queries)

| Config | nDCG@10 | MRR | Recall@10 |
|--------|---------|-----|-----------|
| SPLADE standalone | 0.0464 | 0.0695 | 0.0189 |
| Dense standalone (FashionCLIP) | 0.0265 | 0.0369 | 0.0106 |
| SPLADE + Dense (0.5/0.5) hybrid | 0.0556 | 0.0662 | 0.0201 |
| **SPLADE + Dense + off-shelf CE** | **0.0748** | **0.0738** | **0.0215** |

SPLADE standalone beats both BM25 (+149%) and dense retrieval (+75%) on fashion queries. The full pipeline with SPLADE reaches **nDCG@10 = 0.0748**, +38% over the Blog 1 best. Zero training. Same cross-encoder. One component swap.

### SPLADE vs BM25 on 253K queries

| Config | nDCG@10 | MRR | Recall@10 |
|--------|---------|-----|-----------|
| BM25 standalone | 0.0186 | 0.0227 | 0.0059 |
| **SPLADE standalone** | **0.0412** | **0.0695** | **0.0189** |

+121% nDCG, +206% MRR, +220% Recall@10. SPLADE's learned expansion does what manual synonym lists and NER boosts attempted to do for BM25, but better and without manual rules.

### Phase 3: Training the cross-encoder with LLM labels (22,855 held-out test queries)

| Config | nDCG@10 | MRR | Recall@10 | Cost |
|--------|---------|-----|-----------|------|
| Off-shelf CE (ms-marco-MiniLM-L6) | 0.0639 | 0.0648 | 0.0172 | $0 |
| FT on 1.5M purchase labels | 0.0664 | 0.0678 | 0.0180 | $0 (flat) |
| FT on 9.8K GPT-4o-mini labels | 0.0689 | 0.0701 | 0.0190 | ~$2 |
| **FT-CE-L12 on 194K Sonnet labels** | **0.0735** | **0.0751** | **0.0217** | **~$25** |
| **Best hybrid + LLM CE (SPLADE 0.4 + Dense 0.6)** | **0.0976** | **0.0931** | **0.0268** | **~$25** |

LLM-graded labels crushed purchase labels. 9.8K GPT-4o-mini labels (cost: $2) beat 1.5M purchase labels. Scaling to 194K Sonnet-graded labels (cost: $25) lifted the full pipeline to nDCG@10 = 0.0976, a +31% gain over the Blog 2 headline at essentially zero cost.

### Phase 3C: Training the retrievers on their own mistakes (22,855 held-out test queries)

**Standalone FashionCLIP (dense retriever) before and after fine-tuning:**

| Metric | Baseline | FT-FashionCLIP | Delta |
|--------|----------|----------------|-------|
| nDCG@10 | 0.0229 | 0.0542 | +137% |
| MRR | 0.0208 | 0.0505 | +143% |
| Recall@10 | 0.0433 | 0.0811 | +87% |
| Recall@100 | 0.168 | 0.244 | +45% |

Fine-tuned on 24K contrastive triplets mined from the retriever's own top-20 failures, graded by GPT-4o-mini. Cost: $3 in LLM calls, 45 minutes of training on an M-series MacBook.

**Full pipeline across retriever fine-tuning states (LLM CE fixed):**

| Retrievers | nDCG@10 | MRR | Recall@10 |
|-----------|---------|-----|-----------|
| Baseline SPLADE + Baseline CLIP | 0.0946 | 0.0660 | 0.0253 |
| **Baseline SPLADE + FT-CLIP** | **0.1063** | **0.0766** | **0.0265** |
| FT-SPLADE + Baseline CLIP | 0.0983 | 0.0925 | 0.0268 |
| FT-SPLADE + FT-CLIP | 0.1017 | 0.0741 | 0.0258 |

Best config: **SPLADE(0.3) + FT-FashionCLIP(0.7) + LLM CE = nDCG@10 = 0.1063**, 95% CI [0.1023, 0.1103]. The project best.

Training both retrievers is worse than training one. Their errors start correlating and the hybrid stops extracting diversity. The winning pair always has one baseline retriever and one fine-tuned. The optimal fusion weight shifted again with the stronger dense retriever, from SPLADE(0.4)+Dense(0.6) in Blog 3 to SPLADE(0.3)+FT-Dense(0.7) in Blog 4. As the dense side gets stronger, it deserves more weight in the fusion.

### Phase 4: Three-Tower multimodal retriever (22,855 held-out test queries)

A separate experiment that did not extend the project best, but is included for honest reporting.

| Config | nDCG@10 | MRR | Recall@10 |
|--------|---------|-----|-----------|
| Text tower only (FashionCLIP) | 0.0355 | 0.0657 | 0.0196 |
| Image tower only (FashionCLIP vision) | 0.0305 | 0.0560 | 0.0148 |
| Combined towers (alpha=0.3) | 0.0350 | 0.0656 | 0.0185 |
| 3T + SPLADE + LLM CE (3T best) | **0.0833** | 0.0899 | 0.0272 |
| Blog 4 best (text-only, SPLADE+FT-CLIP+LLM CE) | **0.1063** | 0.0766 | 0.0265 |

The Three-Tower model (separate query / text / image encoders, frozen FashionCLIP towers, contrastive learning on the query tower) does not beat the text-only Blog 4 pipeline on aggregate nDCG@10. It wins on MRR and ties on Recall@10, suggesting the image channel helps at position 1 but hurts the top-10 ordering.

The image channel adds qualitative value on visually-specific queries ("red dress", "floral pattern", "striped shirt") where titles underspecify appearance. On H&M's merchandising-written attribute-rich titles, text retrieval done well beats multimodal retrieval done adequately. On catalogs with noisier text (user-uploaded photos, resale listings, streetwear with visual motifs), the trade-off likely flips.

This is a lateral experiment, not a successor to Blog 4. We report it because the failure mode is itself a finding.

### Phase 5: Beating FashionSigLIP on LookBench (image-to-image retrieval, 2,345 queries)

A different task from Blog 1-5. No text queries, no H&M catalog, no reranker. LookBench is pure image-to-image retrieval: query image of a person wearing an outfit, gallery of 60K+ product images, find the matching product. The state-of-the-art is Marqo's FashionSigLIP. We beat it by cross-domain fine-tuning on DeepFashion2.

| Model | Dim | Fine R@1 | Coarse R@1 | nDCG@5 | Δ vs SigLIP |
|---|---|---|---|---|---|
| FashionSigLIP (paper) | 768 | 62.77 | 82.77 | 49.44 | – |
| FashionSigLIP (our reproduction) | 768 | 63.84 | 83.67 | 49.63 | +1.07 vs paper (clean reproduction) |
| FashionCLIP (our reproduction) | 512 | 59.36 | 78.46 | 45.20 | -4.48 |
| MODA-SigLIP-DeepFashion2 (single model) | 768 | 66.52 | 85.67 | 52.46 | **+2.68** |
| **MODA-SigLIP-DF2 + FashionCLIP ensemble** | **2048** | **67.68** | **86.74** | **53.85** | **+3.84** |

Win-loss across all metric × subset cells: 14W / 1T / 0L for the standalone fine-tuned model; 12W / 0T / 1L for the ensemble (the loss is a 1-query flip on a saturated subset). Data leakage check passed on every training pool.

**What didn't work** (all on LookBench, all underperform the SigLIP baseline):

| Failed approach | Fine R@1 | Δ vs SigLIP | Reason |
|---|---|---|---|
| DINOv2-Base (general vision FM) | 39.49 | -24.35 | General self-supervised features miss fashion's fine-grained attributes |
| MODA-SigLIP-Vision-FT (H&M only) | 58.85 | -4.99 | Single-domain fine-tuning narrowed the model |
| MODA-FashionCLIP-Phase4F (joint text-image FT on H&M) | 54.80 | -9.04 | Multimodal joint training pulled vision encoder toward text alignment |

The winning recipe was specifically cross-domain (DeepFashion2 shop ↔ consumer pairs) plus an ensemble with a different model family at test time. Three of four intuitive approaches failed before we found it.

### Latency (Apple M-series, per query)

| Stage | Mean | p50 | p95 |
|-------|------|-----|-----|
| SPLADE encode + retrieve | ~28ms | ~24ms | ~45ms |
| Dense lookup (FAISS, pre-computed) | <1ms | <1ms | <1ms |
| RRF fusion | 0.1ms | 0.1ms | 0.2ms |
| CE rerank (100 candidates) | 50.9ms | 47.7ms | 73.3ms |
| **Full pipeline end-to-end** | **~80ms** | **~73ms** | **~120ms** |

> Absolute nDCG values are low because ground truth is purchase-based (1 bought item per query against 105K products). This is a benchmark and component breakdown. The relative gains between configs are the finding.

---

## Key findings

1. **Dense > BM25 on fashion queries (-30%)** -- H&M product names are brand-style identifiers ("Ben zip hoodie"). Real users search semantically ("zip hoodie"). This contradicts general e-commerce benchmarks like WANDS where BM25 is competitive.

2. **SPLADE beats both BM25 and dense retrieval** -- Off-shelf SPLADE (`naver/splade-cocondenser-ensembledistil`) standalone (0.0464) beats BM25 (0.0186) by 149% and dense (0.0265) by 75%. Learned sparse retrieval is not obsolete even on a semantics-heavy task.

3. **Cross-encoder reranking is the dominant signal** -- [ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) at 50ms latency is the single most impactful component in both the BM25 and SPLADE eras.

4. **NER helps BM25 (+14%) but does nothing on SPLADE** -- SPLADE's learned expansion already captures what NER attribute boosting was doing manually. When the retriever gets smarter, the tricks you built around the dumb one become dead weight.

5. **Synonym expansion hurts (-35%)** -- Confirms LESER (2025) and LEAPS (2026) query pollution failure mode. Aggressive expansion collapses IDF weights.

6. **FashionCLIP > FashionSigLIP on H&M** -- Short brand-style product titles match CLIP's training distribution better than SigLIP's caption-optimized encoder.

7. **LLM-graded labels >> purchase labels (+15.7%)** -- Fine-tuning on 1.5M purchase labels barely moved the number. A $2 pilot on 9.8K GPT-4o-mini graded labels beat it. Scaling to 194K Sonnet labels (total cost: $25) lifted the trained cross-encoder to nDCG@10 = 0.0735 standalone and 0.0976 in the full pipeline. Data quality is the bottleneck, not model capacity.

8. **The pool size trap** -- Bigger candidate pools for the reranker make things worse. 100 candidates beats 200 beats 500. Cross-encoders have a signal-to-noise floor, and false positives leak in faster than true positives at larger pool sizes.

9. **Fine-tuning the retriever on its own mistakes doubles dense retrieval (+137%)** -- We mined hard negatives from FashionCLIP's top-20 failures, graded them with GPT-4o-mini ($3), and trained on 24K contrastive triplets. Standalone dense nDCG@10 went from 0.0229 to 0.0542. Recall@100 went from 16.8% to 24.4%, which is the ceiling for downstream reranking.

10. **Diversity collapses when both retrievers are fine-tuned** -- The best full pipeline pairs one fine-tuned retriever with one baseline retriever, not both. Training both on the same hard negatives makes their errors correlate and the hybrid stops adding variety. Whichever pair is mismatched in training state beats the fully-matched pair across every column of the factorial.

11. **Fusion weights track the capability gap** -- As the dense retriever got better across phases, the optimal SPLADE-vs-Dense weight shifted from 0.5/0.5 (Blog 2) to 0.4/0.6 (Blog 3) to 0.3/0.7 (Blog 4). RRF fusion is essentially "whichever retriever is more trustworthy on this query gets more vote share." The weight is tracking the real capability delta.

12. **Multimodal retrieval is not free on text-rich catalogs** -- Adding an image channel to the H&M pipeline did not improve aggregate nDCG@10. The text-only pipeline (0.1063) beats the three-tower multimodal pipeline (0.0833). Image features help on visually-specific queries ("floral midi dress") but hurt on text-dominated queries where the title already carries the answer. On H&M's clean merchandising-written titles, text-only retrieval done well beats multimodal retrieval done adequately. On catalogs with noisier text, the trade-off likely flips.

13. **Beating FashionSigLIP on LookBench by +3.84 points** -- Cross-domain fine-tuning on DeepFashion2 (shop ↔ consumer pairs) plus a test-time ensemble with FashionCLIP lifts Fine R@1 from 63.84 to 67.68. The standalone fine-tuned model alone reaches 66.52. Reproduced FashionSigLIP's paper number within 1.07 points before measuring the delta. All training pools passed leakage audits against the LookBench evaluation set.

14. **Fashion-specific pretraining is essential for image-to-image retrieval** -- DINOv2-Base (general self-supervised vision features) scored 39.49 Fine R@1 on LookBench, 24 points below FashionSigLIP. General vision foundation models do not capture fashion's fine-grained attributes (sleeve length, neckline, print pattern, fabric).

15. **Single-domain fine-tuning narrows, cross-domain fine-tuning broadens** -- Fine-tuning the SigLIP vision encoder on H&M images alone (single-domain studio flat-lay) regressed Fine R@1 by -4.99 points vs the baseline. Fine-tuning on DeepFashion2 (shop image ↔ consumer image pairs across diverse settings) lifted the same model by +2.68. The training data distribution decides whether fine-tuning generalizes or specializes.

16. **Joint text-image fine-tuning hurts pure image-to-image retrieval** -- Phase 4F's multimodal joint training, which improved text-to-product retrieval on H&M, regressed LookBench Fine R@1 by -9.04 points. Multi-task joint embeddings fight themselves: text alignment and image-image retrieval pull in different directions. Single-task models win when the task is narrow.

17. **~80ms full pipeline on $0 hardware** -- Everything runs on Apple Silicon with no cloud GPU cost.

---

## Models & Components

| Component | Model / Library | Source | Role |
|---|---|---|---|
| Sparse retrieval (SPLADE) | [naver/splade-cocondenser-ensembledistil](https://huggingface.co/naver/splade-cocondenser-ensembledistil) | `transformers` | Learned sparse retrieval via MLM head |
| Dense retrieval | [Marqo-FashionCLIP](https://huggingface.co/Marqo/marqo-fashionCLIP) (ViT-B/32, 512-dim) | `open_clip` | Bi-encoder embedding + FAISS index |
| Lexical retrieval | [OpenSearch 2.11](https://opensearch.org/) BM25 | Docker | Keyword matching with field boosts |
| NER | [GLiNER2](https://github.com/fastino-ai/GLiNER2) `fastino/gliner2-base-v1` (EMNLP 2025) | `gliner>=0.1.0` | Zero-shot fashion attribute extraction |
| Cross-encoder reranker (off-shelf) | [ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) (22M params) | `sentence-transformers` | Pair-wise reranking baseline |
| Cross-encoder reranker (trained) | MiniLM-L-12 (33M params), trained on 194K Sonnet-graded labels | `sentence-transformers` | +15.7% over off-shelf CE |
| Hybrid fusion | Reciprocal Rank Fusion (RRF) | Custom | Combines SPLADE + dense ranked lists |
| LLM labeling | `openai/gpt-4o-mini` and `claude-sonnet-4.6` via [PaleblueDot AI](https://palebluedot.ai) | REST API | Graded relevance scores (0-3) |

---

## Architecture

```
Query
  |
  +---> SPLADE (learned sparse retrieval)   --+
  |                                           |
  +---> Dense (FashionCLIP -> FAISS)         -+---> RRF Hybrid Fusion
  |                                           |         |
  +---> BM25 (OpenSearch)                   --+         v
  |     [optional: NER attribute boosts]    Cross-Encoder Reranker
  |                                         (ms-marco-MiniLM-L6-v2)
  |                                                     |
  +----------------------------------------------> Top-10 Results
```

---

## Reproducing the results

### Prerequisites

```bash
# Python 3.10+
pip install -r requirements.txt

# OpenSearch (required for BM25)
docker run -d -p 9200:9200 -e "discovery.type=single-node" \
  -e "OPENSEARCH_INITIAL_ADMIN_PASSWORD=<your-password>" \
  opensearchproject/opensearch:2.11.0
```

### Step 1 -- Download data

```bash
# H&M purchase-grounded queries + purchase data (253K queries, ~200MB)
python scripts/build_hnm_benchmark.py

# Or manually from HuggingFace:
# https://huggingface.co/datasets/microsoft/hnm-search-data
# Place in: data/raw/hnm_real/{articles.csv, queries.csv, qrels.csv}
```

### Step 2 -- Index articles in OpenSearch

```bash
python benchmark/index_hnm_opensearch.py
# Indexes 105,542 H&M articles with field-weighted BM25
# Takes ~5 minutes
```

### Step 3 -- Embed articles (FAISS index)

```bash
python benchmark/embed_hnm.py --model fashion-clip
# Embeds 105,542 articles with Marqo-FashionCLIP
# Saves: data/processed/embeddings/fashion-clip_{faiss.index, article_ids.json}
# Takes ~15 min on Apple MPS / ~5 min on GPU
```

### Step 4 -- Run full 253K evaluation pipeline (Blog 1)

```bash
# All stages (overnight run, ~18 hrs total)
python benchmark/eval_full_253k.py --stages all

# Or stage by stage:
python benchmark/eval_full_253k.py --stages 1      # BM25 + FAISS retrieval (~30 min)
python benchmark/eval_full_253k.py --stages 2      # NER pre-compute + BM25+NER (~3 hrs)
python benchmark/eval_full_253k.py --stages 3      # CE reranking (~8.5 hrs) run overnight
python benchmark/eval_full_253k.py --stages 4      # Metrics + final table (~3 min)
```

### Step 5 -- Run SPLADE evaluation (Blog 2)

```bash
# SPLADE + FashionCLIP hybrid on test split (22,855 queries, ~4 hrs)
python -m benchmark.eval_splade_pipeline

# SPLADE configs on full 253K queries
python -m benchmark.eval_full_253k_splade
```

### Step 6 -- Train the cross-encoder on LLM labels (Blog 3)

```bash
# Generate LLM-graded labels via PaleblueDot (requires API key)
python benchmark/generate_llm_labels.py --n_pairs 194000 --model claude-sonnet-4.6

# Train CE on LLM-graded labels (~2h17m on Apple MPS)
python benchmark/train_ce_llm_labels.py --labels data/processed/ce_labels.jsonl --base MiniLM-L12
```

### Step 6 -- Reproduce 10K sample breakdown (faster, for iteration)

```bash
python benchmark/eval_hybrid.py           # BM25 + dense + hybrid (~15 min)
python benchmark/eval_full_pipeline.py    # Full pipeline with CE rerank (~2 hrs)
python benchmark/eval_query_understanding.py  # Synonyms vs NER comparison
```

---

## Project structure

```
MODA/
├── benchmark/
│   ├── article_text.py            <- Canonical article text builder (train-eval consistent)
│   ├── splade_retriever.py        <- SPLADE sparse retrieval wrapper
│   ├── eval_full_253k.py          <- Main 253K evaluation pipeline (staged, checkpointed)
│   ├── eval_full_253k_splade.py   <- SPLADE configs on full 253K benchmark
│   ├── eval_splade_pipeline.py    <- SPLADE + FashionCLIP hybrid eval (test split)
│   ├── eval_hybrid.py             <- BM25 + hybrid breakdown (10K sample)
│   ├── eval_full_pipeline.py      <- Full pipeline with CE rerank (10K sample)
│   ├── embed_hnm.py               <- Article embedding + FAISS index builder
│   ├── index_hnm_opensearch.py    <- OpenSearch indexing
│   ├── models.py                  <- FashionCLIP / FashionSigLIP loaders
│   ├── metrics.py                 <- nDCG, MRR, Recall, AP
│   ├── compute_confidence_intervals.py <- Bootstrap CIs
│   ├── leakage_guard.py           <- Train/test leakage validation
│   ├── _faiss_flat_worker.py      <- FAISS subprocess
│   ├── eval_lookbench_baseline.py <- Phase 5: FashionSigLIP / FashionCLIP baselines on LookBench
│   ├── eval_lookbench_dinov2.py   <- Phase 5: DINOv2 vision foundation model baseline
│   ├── eval_lookbench_ensemble.py <- Phase 5: ensemble + test-time augmentation eval
│   ├── train_deepfashion2_contrastive.py <- Phase 5: cross-domain DeepFashion2 contrastive fine-tune
│   ├── marqo_clean_leakage_audit.py <- Phase 5: cross-benchmark leakage audit
│   └── data_leakage_check_extended.py <- Phase 5: extended train/eval leakage check across all pools
│
├── scripts/
│   ├── build_hnm_benchmark.py     <- Download + prepare H&M data
│   ├── download_datasets.py       <- Download Tier 1 datasets
│   └── verify_setup.py            <- Environment sanity check
│
├── results/
│   ├── full/full_ablation.json         <- 253K final results
│   └── real/
│       ├── phase1_2_splade_eval.json   <- SPLADE Phase 1-2 results
│       ├── all_experiments_with_ci.json <- All configs with bootstrap CIs
│       └── gliner2_ablation.json       <- GLiNER v1 vs GLiNER2 results
│   ├── lookbench/                       <- Phase 5 image-to-image retrieval results
│   │   ├── baseline_eval.json               <- FashionSigLIP / FashionCLIP baselines
│   │   ├── deepfashion2_eval.json           <- MODA-SigLIP-DeepFashion2 (single model FT)
│   │   ├── ensemble_tta_eval.json           <- Ensemble (DF2 + FashionCLIP) — project best 67.68
│   │   ├── dinov2_eval.json                 <- DINOv2 zero-shot (failure mode comparison)
│   │   ├── data_leakage_check.json          <- Train/eval leakage audit
│   │   ├── data_leakage_check_v2.json       <- Extended leakage audit (all training pools)
│   │   └── phase5_summary.json              <- Full Phase 5 ranked results + win/loss matrix
│   └── marqo_bench/
│       └── leakage_audit_clean.json     <- Marqo benchmark cross-leakage audit
│
├── blog_post.md                   <- Blog 1: Zero-shot pipeline
├── blog_post_phase2b.md           <- Blog 2: SPLADE swap
├── requirements.txt
└── README.md
```

---

## Ground truth

**Source:** [microsoft/hnm-search-data](https://huggingface.co/datasets/microsoft/hnm-search-data)

Each query in `qrels.csv` has:
- `positive_ids`: the article the user **purchased** after this search (grade = 2)
- `negative_ids`: articles **shown** in the same session but **not** bought (grade = 1)
- Everything else in the 105K catalogue: unlabeled (grade = 0)

**nDCG@k** uses the full grade scale (2 > 1 > 0). **MRR/Recall/P@k** treat any labeled item (grade > 0) as relevant.

> Purchase does not equal perfect relevance. A user searching "black dress" sees 20 good options but buys one. The other 19 are scored as negatives. This suppresses all absolute metric values. The _relative_ ordering between pipeline configs is what matters.

---

## Phase roadmap

| Phase | Focus | Status |
|---|---|---|
| **1** | Zero-shot pipeline: BM25 + dense + hybrid + NER + CE rerank (253K queries) | Done |
| **2B** | SPLADE swap: replace BM25 with learned sparse retrieval (+38%) | Done |
| **3A/3B** | Training the cross-encoder with LLM-graded labels (+31%) | Done |
| **3C** | Fine-tuning the retriever on its own mistakes (+9% to project best 0.1063) | Done |
| **4** | Three-Tower multimodal retriever on H&M (lateral experiment, 0.0833 aggregate) | Done |
| **5** | LookBench image-to-image retrieval: cross-domain fine-tuning beats FashionSigLIP by +3.84 | Done |

---

## Citation

If you use this benchmark or code, please cite:

```bibtex
@misc{moda2026,
  title  = {MODA: The First Open Benchmark for End-to-End Fashion Search},
  author = {Hopit AI},
  year   = {2026},
  url    = {https://github.com/hopit-ai/moda}
}
```

---

## License

MIT -- see [LICENSE](LICENSE).

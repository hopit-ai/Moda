# MODA

**The first open-source, end-to-end benchmark for fashion search with a full component-by-component breakdown.**  
253,685 purchase-grounded queries · 105,542 H&M products · 20+ pipeline configs · nDCG@10 = 0.0748 (+183% over dense baseline)

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
| Blog 3 | *Coming soon* | | |
| Blog 4 | *Coming soon* | | |
| Blog 5 | *Coming soon* | | |

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

7. **~80ms full pipeline on $0 hardware** -- Everything runs on Apple Silicon with no cloud GPU cost.

---

## Models & Components

| Component | Model / Library | Source | Role |
|---|---|---|---|
| Sparse retrieval (SPLADE) | [naver/splade-cocondenser-ensembledistil](https://huggingface.co/naver/splade-cocondenser-ensembledistil) | `transformers` | Learned sparse retrieval via MLM head |
| Dense retrieval | [Marqo-FashionCLIP](https://huggingface.co/Marqo/marqo-fashionCLIP) (ViT-B/32, 512-dim) | `open_clip` | Bi-encoder embedding + FAISS index |
| Lexical retrieval | [OpenSearch 2.11](https://opensearch.org/) BM25 | Docker | Keyword matching with field boosts |
| NER | [GLiNER2](https://github.com/fastino-ai/GLiNER2) `fastino/gliner2-base-v1` (EMNLP 2025) | `gliner>=0.1.0` | Zero-shot fashion attribute extraction |
| Cross-encoder reranker | [ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) (22M params) | `sentence-transformers` | Pair-wise reranking |
| Hybrid fusion | Reciprocal Rank Fusion (RRF) | Custom | Combines SPLADE + dense ranked lists |

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
│   └── _faiss_flat_worker.py      <- FAISS subprocess
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
| **3** | Training the cross-encoder and retriever | Coming soon |
| **4** | Multimodal retrieval: images as a search signal | Coming soon |

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

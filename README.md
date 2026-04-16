# MODA

**The first open-source, end-to-end benchmark for fashion search with a full component-by-component breakdown.**  
253,685 purchase-grounded queries · 105,542 H&M products · 49+ pipeline configs · nDCG@10 = 0.1063 (+301% over dense baseline)

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
| Blog 3 | *Coming soon* | Training the cross-encoder with LLM labels | |
| Blog 4 | *Coming soon* | Fine-tuning the retriever on its own mistakes | |
| Blog 5 | *Coming soon* | Adding eyes to the search engine (multimodal) | |

---

## Key results (253,685 queries, 105,542 products)

| Config | nDCG@10 | 95% CI | MRR | AP | Recall@10 | Recall@50 | P@10 |
|--------|---------|--------|-----|----|-----------|-----------|----|
| BM25 only | 0.0186 | [.0183-.0190] | 0.0227 | 0.0040 | 0.0059 | 0.0251 | 0.0058 |
| BM25 + NER boost | 0.0204 | [.0200-.0207] | 0.0260 | 0.0048 | 0.0069 | 0.0298 | 0.0068 |
| Dense only (FashionCLIP) | 0.0265 | [.0261-.0269] | 0.0369 | 0.0071 | 0.0106 | 0.0462 | 0.0105 |
| Hybrid (BM25x0.4 + Dense x0.6) | 0.0328 | [.0324-.0333] | 0.0429 | 0.0075 | 0.0121 | 0.0457 | 0.0121 |
| Hybrid + NER boost | 0.0333 | [.0329-.0338] | 0.0438 | 0.0078 | 0.0124 | 0.0470 | 0.0124 |
| **Full pipeline (Hybrid + CE rerank)** | **0.0543** | **[.0537-.0550]** | **0.0569** | **0.0091** | **0.0164** | **0.0559** | **0.0163** |

Full pipeline vs dense baseline: +105% nDCG@10, +54% MRR, +55% Recall@10, +21% Recall@50.

### Latency (Apple M-series, per query)

| Stage | Mean | p50 | p95 |
|-------|------|-----|-----|
| BM25 (OpenSearch) | 11.5ms | 9.7ms | 18.2ms |
| Dense lookup (FAISS, pre-computed) | <1ms | <1ms | <1ms |
| RRF fusion | 0.1ms | 0.1ms | 0.2ms |
| CE rerank (100 candidates) | 50.9ms | 47.7ms | 73.3ms |
| **Full pipeline end-to-end** | **62.5ms** | **~58ms** | **~92ms** |

### SPLADE retrieval (22,855 held-out test queries)

| Config | nDCG@10 | 95% CI | MRR | R@10 | Notes |
|--------|---------|--------|-----|------|-------|
| SPLADE standalone | 0.0464 | [0.0437, 0.0491] | 0.0602 | 0.0167 | Off-shelf `naver/splade-cocondenser-ensembledistil` |
| SPLADE + LLM CE | 0.0903 | [0.0869, 0.0937] | 0.0640 | 0.0243 | Sparse-only with CE rerank |
| SPLADE + FashionCLIP (0.5/0.5) + LLM CE | 0.0933 | [0.0899, 0.0967] | 0.0667 | 0.0250 | RRF hybrid |
| SPLADE + FashionCLIP (0.4/0.6) + LLM CE | 0.0976 | [0.0941, 0.1011] | 0.0710 | 0.0249 | |
| SPLADE + FT-CLIP (best) + LLM CE | 0.1017 | [0.0981, 0.1053] | 0.0741 | 0.0258 | FT-FashionCLIP dense channel |
| **SPLADE + FT-CLIP (0.3/0.7) + LLM CE** | **0.1063** | **[0.1023, 0.1103]** | **0.0766** | **0.0265** | **Project best** |

> **SPLADE replaces BM25** as the lexical backbone. Off-shelf SPLADE standalone (0.0464) already beats BM25 (0.0186) by 149%. Adding SPLADE to the hybrid fusion with FT-FashionCLIP and LLM CE reranking reaches **nDCG@10 = 0.1063**, +301% over the Phase 1 dense baseline (0.0265).

> Absolute nDCG values are low because ground truth is purchase-based (1 bought item per query against 105K products). This is a benchmark and component breakdown. The relative gains between configs are the finding.
>
> **Evaluation splits:** Phase 2 results use all **253,685 queries** (zero-shot components only, nothing was trained on this data). Phase 3+ results (LLM-trained CE, fine-tuned retrievers, SPLADE) use a held-out **22,855-query test split** to prevent data leakage, since these models were trained on a disjoint subset of the queries.

---

## Key findings

1. **Dense > BM25 on fashion queries (-30%)** -- H&M product names are brand-style identifiers ("Ben zip hoodie"). Real users search semantically ("zip hoodie"). This contradicts general e-commerce benchmarks like WANDS where BM25 is competitive.

2. **Cross-encoder reranking is the dominant signal (+51% marginal)** -- [ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) at 50ms latency is the single most impactful component.

3. **Synonym expansion hurts (-35%)** -- Confirms LESER (2025) and LEAPS (2026) query pollution failure mode. Aggressive expansion collapses IDF weights.

4. **NER attribute boosting helps (+14% on BM25)** -- [GLiNER2](https://github.com/fastino-ai/GLiNER2) zero-shot NER extracts fashion attributes and maps them to H&M field boosts. +16% improvement over GLiNER v1 at the NER stage.

5. **SPLADE replaces BM25 as the lexical backbone** -- Off-shelf SPLADE (`naver/splade-cocondenser-ensembledistil`) standalone (0.0464) beats BM25 (0.0186) by 149%. SPLADE + FT-FashionCLIP + LLM CE reaches **nDCG@10 = 0.1063**, the project best.

6. **FashionCLIP > FashionSigLIP on H&M** -- Short brand-style product titles match CLIP's training distribution better than SigLIP's caption-optimized encoder.

7. **62.5ms full pipeline on $0 hardware** -- Everything runs on Apple Silicon with no cloud GPU cost.

---

## Models & Components

| Component | Model / Library | Source | Role |
|---|---|---|---|
| Dense retrieval | [Marqo-FashionCLIP](https://huggingface.co/Marqo/marqo-fashionCLIP) (ViT-B/32, 512-dim) | `open_clip` | Bi-encoder embedding + FAISS index |
| Sparse retrieval (SPLADE) | [naver/splade-cocondenser-ensembledistil](https://huggingface.co/naver/splade-cocondenser-ensembledistil) | `transformers` | Learned sparse retrieval via MLM head |
| Lexical retrieval | [OpenSearch 2.11](https://opensearch.org/) BM25 | Docker | Keyword matching with field boosts |
| NER | [GLiNER2](https://github.com/fastino-ai/GLiNER2) `fastino/gliner2-base-v1` (EMNLP 2025) | `gliner>=0.1.0` | Zero-shot fashion attribute extraction |
| Cross-encoder reranker | [ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) (22M params) | `sentence-transformers` | L2 pair-wise reranking |
| Hybrid fusion | Reciprocal Rank Fusion (RRF) | Custom | Combines SPLADE + dense (+ BM25) ranked lists |
| LLM labeling (Phase 3+) | `openai/gpt-4o-mini` via [PaleblueDot AI](https://palebluedot.ai) | REST API | Graded relevance scores (0-3) |

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
  |                                         (LLM-trained MiniLM-L6)
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

### Step 4 -- Run full 253K evaluation pipeline

```bash
# All stages (overnight run, ~18 hrs total)
python benchmark/eval_full_253k.py --stages all

# Or stage by stage:
python benchmark/eval_full_253k.py --stages 1      # BM25 + FAISS retrieval (~30 min)
python benchmark/eval_full_253k.py --stages 2      # NER pre-compute + BM25+NER (~3 hrs)
python benchmark/eval_full_253k.py --stages 3      # CE reranking (~8.5 hrs) run overnight
python benchmark/eval_full_253k.py --stages 4      # Metrics + final table (~3 min)
```

### Step 4b -- Run SPLADE evaluation

```bash
# SPLADE + FashionCLIP hybrid on test split (22,855 queries, ~4 hrs)
python -m benchmark.eval_splade_pipeline

# SPLADE configs on full 253K queries
python -m benchmark.eval_full_253k_splade
```

### Step 5 -- Reproduce 10K sample breakdown (faster, for iteration)

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
│   ├── train_cross_encoder.py     <- CE fine-tuning
│   ├── train_ce_llm_labels.py     <- CE training with LLM-graded labels
│   ├── train_biencoder.py         <- Bi-encoder fine-tuning (hard negatives)
│   ├── train_splade.py            <- SPLADE fine-tuning
│   ├── train_three_tower.py       <- Three-tower multimodal training
│   ├── eval_three_tower.py        <- Three-tower benchmark eval
│   ├── generate_llm_labels.py     <- LLM label generation
│   ├── generate_biencoder_labels.py <- Hard negative mining for bi-encoder
│   ├── generate_splade_labels.py  <- Hard negative mining for SPLADE
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
│       ├── splade_pipeline_eval.json   <- SPLADE pipeline results (project best)
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

| Phase | Focus | What | Status |
|---|---|---|---|
| **1** | Benchmark validation | Reproduce Marqo's 7-dataset embedding benchmark (<1% delta). Build eval harness. | Done |
| **2** | Zero-shot pipeline | BM25 + dense + hybrid + NER + CE rerank. 253K queries, 11 configs, component-by-component breakdown. | Done |
| **2B** | SPLADE retrieval | Replace BM25 with SPLADE learned sparse retrieval. 20+ fusion configs. Best: SPLADE + FT-CLIP + LLM CE = **0.1063**. | Done |
| **3** | Trained models | LLM-judged labels for CE. Fine-tuned bi-encoder on retriever-mined hard negatives. Fine-tuned SPLADE. | Done |
| **4** | Multimodal retrieval | Image embeddings, text-to-image retrieval, joint fine-tuning. Three-Tower architecture. | Done |

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

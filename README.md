# MODA

**Open benchmark for end-to-end fashion search, with a full component-by-component breakdown.**
253,685 purchase-grounded queries · 105,542 H&M products · 11 pipeline configs · +81% over embedding-only retrieval

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What is this?

Nobody has published open, full-pipeline fashion search benchmarks on purchase-grounded queries at scale. Marqo has great embeddings. Algolia and Bloomreach are proprietary. Nobody has put it all together and measured what each piece contributes.

MODA fills that gap. We built a complete retrieval pipeline (BM25 + dense + hybrid + NER + cross-encoder reranking), ran it against 253,685 queries derived from H&M purchase transactions, isolated the contribution of every component, and published everything.

> **Note on query provenance:** The queries are synthetically generated from real H&M purchase data ([Microsoft's H&M Search Data](https://huggingface.co/datasets/microsoft/hnm-search-data)), not captured from actual search logs. The purchases are real; the queries are reconstructed. This is a known limitation. See the [blog post](blog_post.md) for details.

Read the full write-up: [blog_post.md](blog_post.md)

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

> Absolute nDCG values are low because ground truth is purchase-based (1 bought item per query against 105K products). This is a benchmark and component breakdown. The relative gains between configs are the finding.

---

## Key findings

1. **Dense > BM25 on fashion queries (-30%)** — H&M product names are brand-style identifiers ("Ben zip hoodie"). Real users search semantically ("zip hoodie"). This contradicts general e-commerce benchmarks like WANDS where BM25 is competitive.

2. **Cross-encoder reranking is the dominant signal (+51% marginal)** — [ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) at 50ms latency is the single most impactful component.

3. **Synonym expansion hurts (-35%)** — Confirms LESER (2025) and LEAPS (2026) query pollution failure mode. Aggressive expansion collapses IDF weights.

4. **NER attribute boosting helps (+14% on BM25)** — [GLiNER2](https://github.com/fastino-ai/GLiNER2) zero-shot NER extracts fashion attributes and maps them to H&M field boosts. +16% improvement over GLiNER v1 at the NER stage.

5. **FashionCLIP > FashionSigLIP on H&M** — Short brand-style product titles match CLIP's training distribution better than SigLIP's caption-optimized encoder.

6. **62.5ms full pipeline on $0 hardware** — Everything runs on Apple Silicon with no cloud GPU cost.

---

## Models and components

| Component | Model / Library | Role |
|-----------|----------------|------|
| Dense retrieval | [Marqo-FashionCLIP](https://huggingface.co/Marqo/marqo-fashionCLIP) (ViT-B/32, 512-dim) | Bi-encoder embedding + FAISS index |
| Lexical retrieval | [OpenSearch 2.11](https://opensearch.org/) BM25 | Keyword matching with field boosts |
| NER | [GLiNER2](https://github.com/fastino-ai/GLiNER2) (EMNLP 2025) | Zero-shot fashion attribute extraction |
| Cross-encoder | [ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) (22M params) | Pairwise reranking |
| Hybrid fusion | Reciprocal Rank Fusion (RRF) | Combines BM25 + dense ranked lists |

---

## Architecture

```
Query
  |
  +---> BM25 (OpenSearch)                 --+
  |     [optional: NER attribute boosts]    |
  |                                         +---> RRF Hybrid Fusion
  +---> Dense (FashionCLIP -> FAISS)      --+          |
  |                                                    v
  |                                Cross-Encoder Reranker (MiniLM-L6)
  |                                                    |
  +----------------------------------------------> Top-50 Results
```

---

## Quick start

### Prerequisites

```bash
# Python 3.10+
git clone https://github.com/hopit-ai/Moda.git && cd Moda
pip install -r requirements.txt

# OpenSearch (required for BM25)
docker run -d -p 9200:9200 -e "discovery.type=single-node" \
  -e "OPENSEARCH_INITIAL_ADMIN_PASSWORD=Admin@12345" \
  opensearchproject/opensearch:2.11.0
```

### Run on the H&M benchmark (reproduce our results)

```bash
# 1. Download H&M data (253K queries, 105K products, ~200MB)
python scripts/build_hnm_benchmark.py

# 2. Index in OpenSearch for BM25 (~5 min)
python benchmark/index_hnm_opensearch.py

# 3. Build dense embeddings + FAISS index (~15 min on MPS, ~5 min on GPU)
python benchmark/embed_hnm.py --model fashion-clip

# 4. Run the full evaluation pipeline (~18 hrs, checkpointed — safe to interrupt and resume)
python benchmark/eval_full_253k.py --stages all

# Or run stages individually:
python benchmark/eval_full_253k.py --stages 1    # BM25 + FAISS retrieval (~30 min)
python benchmark/eval_full_253k.py --stages 2    # NER pre-compute (~3 hrs)
python benchmark/eval_full_253k.py --stages 3    # CE reranking (~8.5 hrs, run overnight)
python benchmark/eval_full_253k.py --stages 4    # Aggregate metrics (~3 min)
```

Results land in `results/full/full_ablation.json`.

### Run on a faster 10K sample (for iteration)

```bash
# BM25 + dense + hybrid breakdown (~15 min, no CE)
python benchmark/eval_hybrid.py

# Full pipeline including CE rerank (~2 hrs)
python benchmark/eval_full_pipeline.py

# Synonym vs NER comparison
python benchmark/eval_query_understanding.py
```

### Run on your own catalog

The pipeline expects a CSV with product data. At minimum you need an ID column and a text column (product title or description). Additional columns (color, category, brand) improve NER boosting and faceted search.

```bash
# 1. Prepare your catalog as CSV
#    Required: article_id, prod_name
#    Optional: product_type_name, colour_group_name, detail_desc, ...

# 2. Index your catalog in OpenSearch
python benchmark/index_hnm_opensearch.py --csv path/to/your_catalog.csv

# 3. Embed your catalog
python benchmark/embed_hnm.py --model fashion-clip --csv path/to/your_catalog.csv

# 4. Search
#    Use the scripts as a starting point — the retrieval, fusion,
#    and reranking logic is in benchmark/eval_hybrid.py and
#    benchmark/eval_full_pipeline.py
```

To evaluate, you'll need query-relevance pairs (qrels). If you have search logs with click or purchase data, those work. If not, you can generate relevance labels with an LLM (we show how in Phase 3).

### Reproduce Marqo's 7-dataset embedding benchmark (Phase 1)

```bash
python scripts/download_datasets.py
python benchmark/eval_marqo_7dataset.py --models fashion-clip fashion-siglip
```

---

## Ground truth

**Source:** [microsoft/hnm-search-data](https://huggingface.co/datasets/microsoft/hnm-search-data)

Each query in `qrels.csv` has:
- `positive_ids` — the article the user purchased (grade = 2)
- `negative_ids` — articles shown but not bought (grade = 1)
- Everything else — unlabeled (grade = 0)

> **Limitations:** (1) Purchase is not perfect relevance. A user searching "black dress" sees 20 good options but buys one. The other 19 are scored as negatives. This suppresses absolute metric values. (2) The queries are synthetically generated from purchase transactions, not captured from actual search logs. The relative ordering between configs is what matters; absolute values are not comparable to benchmarks with different relevance structures.

---

## Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| **1** | Reproduce Marqo 7-dataset embedding benchmark (<1% delta) | Done |
| **2** | Zero-shot pipeline: BM25 + dense + hybrid + NER + CE rerank + ColBERT (253K queries, 11 configs) | Done |
| **3** | Fine-tuned models: LLM-judged labels, trained bi-encoder, MoE with per-field encoders | Next |
| **4** | Multimodal: image embeddings, three-way hybrid, [LookBench](https://arxiv.org/abs/2601.14706) visual retrieval benchmark | Planned |
| **5** | Search experience: data augmentation, faceted navigation, partitioned indexes, demo | Planned |

---

## Citation

```bibtex
@misc{moda2026,
  title  = {MODA: The First Open Benchmark for End-to-End Fashion Search},
  author = {Hopit AI},
  year   = {2026},
  url    = {https://github.com/hopit-ai/Moda}
}
```

---

## License

MIT — see [LICENSE](LICENSE).

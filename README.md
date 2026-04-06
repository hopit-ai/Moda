# MODA — Modular Open-Source Discovery Architecture

**The first open-source, end-to-end, ablation-complete benchmark for fashion search.**  
253,685 real user queries · 105,542 H&M products · 18 pipeline configs · nDCG@10 = 0.0757 (+152% over baseline)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

---

## What is this?

Nobody has published open-source, full-pipeline fashion search benchmarks on real user queries.  
Marqo has great embeddings. Algolia/Bloomreach are proprietary. Nobody has put it all together and ablated it.

**MODA fills that gap.** We built a complete retrieval pipeline (BM25 + dense + hybrid + NER + cross-encoder reranking), ran it against 253,685 real H&M purchase queries, isolated the contribution of every component, and published everything — code, results, methodology.

---

## Key Results (253,685 real queries × 105,542 articles)

| Config | nDCG@10 | 95% CI | MRR | Latency | vs Dense Baseline |
|---|---|---|---|---|---|
| BM25 only | 0.0186 | [0.0183–0.0190] | 0.0227 | 11.5ms | −37.9% |
| Dense only (FashionCLIP) | 0.0265 | [0.0261–0.0269] | 0.0369 | <1ms* | baseline |
| Hybrid (BM25×0.4 + Dense×0.6) | 0.0328 | [0.0324–0.0333] | 0.0429 | 11.6ms | +23.8% |
| Hybrid + NER boost | 0.0333 | [0.0329–0.0338] | 0.0438 | ~18ms | +25.7% |
| **Full Pipeline (Hybrid + CE rerank)** | **0.0543** | **[0.0537–0.0550]** | **0.0569** | **62.5ms** | **+104.9%** |

### Phase 3 — LLM-Guided Training (22,855 held-out test queries)

| Config | nDCG@10 | MRR | R@10 | vs Baseline |
|---|---|---|---|---|
| Off-the-shelf CE@50 | 0.0646 | 0.0671 | 0.0195 | CE baseline |
| Fine-tuned CE@50 (purchase labels) | 0.0654 | 0.0644 | 0.0183 | +1.2% |
| LLM-trained CE@50 (GPT-4o-mini labels) | 0.0747 | 0.0755 | 0.0217 | +15.7% |
| Fine-tuned FashionCLIP (dense only) | 0.0444 | 0.0405 | 0.0811 | +94.2% (dense) |
| **Fine-tuned retriever + LLM CE (B2)** | **0.0757** | **0.0799** | **0.0243** | **+79.4% ✅ SOTA** |

### Retriever × Reranker Matrix (nDCG@10)

| | No Rerank | Off-shelf CE | LLM-trained CE |
|---|---|---|---|
| Baseline FashionCLIP | 0.0422 | 0.0646 | 0.0747 |
| Fine-tuned FashionCLIP | 0.0515 (+22%) | 0.0650 (+0.6%) | **0.0757 (+1.3%)** |

> **Phase 3 breakthroughs:** LLM-judged labels transformed both stages. Cross-encoder: 42.8K GPT-4o-mini graded labels → **+15.7% nDCG@10**. Bi-encoder: 100K retriever-mined hard negatives + LLM labels → **+94.2% dense retrieval**. Combined (B2): **nDCG@10 = 0.0757, MRR = 0.0799, Recall@10 = 0.0243** — new project SOTA. Gains are sub-additive on nDCG but additive on Recall (+12%).

### Phase 4 — Multimodal Retrieval

| Step | What | Command | Status |
|------|------|---------|--------|
| **4A** | Download H&M product images (105K) | `python scripts/download_hnm_images.py` | Done |
| **4B** | Embed images with FashionCLIP vision encoder, build FAISS index | `python benchmark/embed_hnm_images.py` | Done |
| **4C** | Text-to-image retrieval channel | (integrated in eval scripts) | Done |
| **4D** | Zero-shot multimodal eval (baseline + fine-tuned text encoder) | `python benchmark/eval_multimodal_pipeline.py` | Done |
| **4E** | LLM labels for image hard negatives (PaleblueDot GPT-4o-mini) | `python benchmark/generate_image_labels.py` | Done |
| **4F** | Joint text+image fine-tuning (both CLIP encoders, contrastive + alignment) | `python benchmark/train_multimodal.py` | Running |
| **4G** | Re-embed images with 4F model + multimodal pipeline eval | `bash scripts/run_phase4g_multimodal_eval.sh` | Waiting for 4F |
| **4H** | Three-Tower training (query/text/image towers, novel architecture) | `python benchmark/train_three_tower.py` | Pending |
| **4I** | Three-Tower full benchmark evaluation | `python benchmark/eval_three_tower.py` | Pending |

**4E** produces `data/processed/image_retriever_labels.jsonl` via PaleblueDot (`openai/gpt-4o-mini`). Both **4F** and **4H** consume this file alongside Phase 3C text labels (`biencoder_retriever_labels.jsonl`).

**4F** fine-tunes both FashionCLIP encoders with contrastive loss + alignment regularisation. **4G** re-embeds 105K images with the 4F checkpoint and reruns multimodal pipeline eval (auto-starts after 4F saves `best/`). Run 4G standalone: `bash scripts/run_phase4g_multimodal_eval.sh`

**4H** is a novel **Three-Tower Fashion Retriever (3TFR)**: dedicated **query tower** (CLIP text encoder + projection MLP), frozen **text tower**, frozen **image tower** — all projecting into a single 512-dim space. Product embeddings are precomputed offline; only the query tower trains. **4I** evaluates 4H on the full 22,855 test queries with BM25 hybrid + CE reranking.

*Dense vectors are pre-computed; online latency is dict lookup.  
Latency measured on Apple M-series (MPS).

> **Framing note:** Absolute nDCG values are low because ground truth is purchase-based (1 bought item per query against 105K products). This is a **benchmark and ablation study** — the relative gains between configs are the finding.

---

## Key Findings

1. **Dense > BM25 on real user queries (−38%)** — H&M product names are brand-style identifiers ("Ben zip hoodie"). Real users search semantically ("zip hoodie"). This contradicts general e-commerce benchmarks (WANDS) where BM25 wins — the difference is real-user intent vs curated queries.

2. **Cross-encoder reranking is the dominant signal (+105%)** — `ms-marco-MiniLM-L-6-v2` running over 100 candidates per query at 62.5ms end-to-end is the single most impactful component.

3. **Synonym expansion hurts (−32 to −58%)** — Confirms LESER (2025) / LEAPS-Taobao (2026) "query pollution" failure mode. Aggressive expansion causes IDF collapse.

4. **NER attribute boosting helps (+9%)** — GLiNER zero-shot NER maps query entities to H&M fields via `bool.should` boosts without hard-filtering.

5. **LLM-judged labels >> purchase labels for CE training (+15.7%)** — Fine-tuning on purchase data barely helped (+1.2%). Replacing with 42.8K GPT-4o-mini graded relevance scores yields nDCG@10=0.0747. Data quality is the bottleneck, not model capacity.

6. **Retriever-mined hard negatives + LLM labels = +94.2% dense retrieval** — Fine-tuning FashionCLIP with contrastive loss on 24K triplets (mined from its own top-20 failures, labeled by GPT-4o-mini) nearly doubles retrieval quality. The model learns exactly where it was wrong.

7. **FashionCLIP > FashionSigLIP on H&M** — Short brand-style product titles match CLIP's training distribution better than SigLIP's caption-optimized encoder.

---

## Architecture

```
Query
  │
  ├─► BM25 (OpenSearch)                  ─┐
  │   [optional: NER attribute boosts]    │
  │                                       ├─► RRF Hybrid Fusion
  ├─► Dense (FashionCLIP → FAISS)        ─┘        │
  │                                                  ▼
  │                               Cross-Encoder Reranker (MiniLM-L6)
  │                                                  │
  └──────────────────────────────────────────► Top-50 Results
```

---

## Reproducing the Results

### Prerequisites

```bash
# Python 3.10+
pip install -r requirements.txt

# OpenSearch (required for BM25)
docker run -d -p 9200:9200 -e "discovery.type=single-node" \
  -e "OPENSEARCH_INITIAL_ADMIN_PASSWORD=Admin@12345" \
  opensearchproject/opensearch:2.11.0
```

### Step 1 — Download data

```bash
# H&M real user queries + purchase data (253K queries, ~200MB)
python scripts/build_hnm_benchmark.py

# Or manually from HuggingFace:
# https://huggingface.co/datasets/microsoft/hnm-search-data
# Place in: data/raw/hnm_real/{articles.csv, queries.csv, qrels.csv}
```

### Step 2 — Index articles in OpenSearch

```bash
python benchmark/index_hnm_opensearch.py
# Indexes 105,542 H&M articles with field-weighted BM25
# Takes ~5 minutes
```

### Step 3 — Embed articles (FAISS index)

```bash
python benchmark/embed_hnm.py --model fashion-clip
# Embeds 105,542 articles with Marqo-FashionCLIP
# Saves: data/processed/embeddings/fashion-clip_{faiss.index, article_ids.json}
# Takes ~15 min on Apple MPS / ~5 min on GPU
```

### Step 4 — Run full 253K evaluation pipeline

The pipeline is broken into 4 stages, each checkpointed to disk. You can run all at once or stage by stage:

```bash
# All stages (overnight run, ~18 hrs total)
python benchmark/eval_full_253k.py --stages all

# Or stage by stage:
python benchmark/eval_full_253k.py --stages 1      # BM25 + FAISS retrieval (~30 min)
python benchmark/eval_full_253k.py --stages 2      # NER pre-compute + BM25+NER (~3 hrs)
python benchmark/eval_full_253k.py --stages 3      # CE reranking (~8.5 hrs) 🌙 run overnight
python benchmark/eval_full_253k.py --stages 4      # Metrics + final table (~3 min)
```

**Stage 3 tip:** Keep your laptop lid open and run `caffeinate -i &` before starting to prevent sleep.

If interrupted, re-run the same command — each stage resumes from its checkpoint automatically.

**Output:** `results/full/full_ablation.json`

### Step 5 — Reproduce 10K sample ablation (faster, for iteration)

```bash
# BM25 + dense + hybrid configs (no CE, ~15 min)
python benchmark/eval_hybrid.py

# Full pipeline including CE rerank (~2 hrs on 10K)
python benchmark/eval_full_pipeline.py

# Query understanding ablation (synonyms vs NER)
python benchmark/eval_query_understanding.py
```

### Step 6 — Reproduce Phase 1 (Marqo 7-dataset embedding benchmark)

```bash
# Download Tier 1 datasets
python scripts/download_datasets.py

# Run embedding benchmark (reproduces Marqo's published numbers)
python benchmark/eval_marqo_7dataset.py --models fashion-clip fashion-siglip
```

---

## Project Structure

```
MODA/
├── benchmark/
│   ├── eval_full_253k.py       ← Main 253K evaluation pipeline (staged, checkpointed)
│   ├── eval_hybrid.py          ← BM25 + hybrid ablation (10K sample)
│   ├── eval_full_pipeline.py   ← Full pipeline with CE rerank (10K sample)
│   ├── eval_query_understanding.py  ← Synonym vs NER ablation
│   ├── eval_marqo_7dataset.py  ← Phase 1 Marqo benchmark reproduction
│   ├── embed_hnm.py            ← Article embedding + FAISS index builder
│   ├── index_hnm_opensearch.py ← OpenSearch indexing
│   ├── query_expansion.py      ← Synonym dictionary + GLiNER NER
│   ├── models.py               ← FashionCLIP / FashionSigLIP loaders
│   ├── metrics.py              ← nDCG, MRR, Recall, AP (with unit tests)
│   ├── train_cross_encoder.py  ← Phase 3: CE fine-tuning on H&M data
│   ├── train_multimodal.py     ← Phase 4F: joint text+image fine-tuning
│   ├── train_three_tower.py    ← Phase 4H: three-tower architecture training
│   ├── eval_multimodal_pipeline.py ← Phase 4D/4G: multimodal pipeline eval
│   ├── eval_three_tower.py     ← Phase 4I: three-tower benchmark eval
│   └── _faiss_search_worker.py ← FAISS subprocess (avoids BLAS conflicts)
│
├── scripts/
│   ├── build_hnm_benchmark.py  ← Download + prepare H&M data
│   ├── download_datasets.py    ← Download Tier 1 datasets
│   ├── generate_report_pdf.py  ← PDF report generator
│   └── verify_setup.py         ← Environment sanity check
│
├── results/
│   ├── full/
│   │   ├── full_ablation.json       ← 253K final results (all 8 configs)
│   │   ├── ner_cache_253k.json      ← Pre-computed GLiNER extractions (253K)
│   │   ├── latency_results.json     ← Latency measurements (500-query sample)
│   │   └── PHASE2_FULL_LEADERBOARD.md  ← Full results leaderboard
│   └── real/
│       ├── PHASE2_RUNNING_LEADERBOARD.md  ← 10K sample leaderboard
│       └── ner_cache_10k.json             ← Pre-computed GLiNER (10K)
│
├── data/
│   ├── raw/                    ← Downloaded datasets (gitignored, ~8.8 GB)
│   │   └── hnm_real/           ← articles.csv, queries.csv, qrels.csv
│   └── processed/
│       └── embeddings/         ← FAISS indexes + article ID lists (gitignored, ~1.4 GB)
│
├── MODA_Phase0_to_Phase2_Report.pdf  ← Full research report
├── requirements.txt
└── README.md
```

---

## Ground Truth — How Relevance is Defined

**Source:** [microsoft/hnm-search-data](https://huggingface.co/datasets/microsoft/hnm-search-data)

Each query in `qrels.csv` has:
- `positive_ids` — the article the user **purchased** after this search (grade = 2)
- `negative_ids` — articles **shown** in the same session but **not** bought (grade = 1)
- Everything else in the 105K catalogue — unlabeled (grade = 0)

**nDCG@k** uses the full grade scale (2 > 1 > 0). **MRR/Recall/P@k** treat any labeled item (grade > 0) as relevant.

> **Limitation acknowledged:** Purchase ≠ perfect relevance. A user searching "black dress" sees 20 good options but buys one — the other 19 are scored as negatives. This suppresses all absolute metric values. The _relative_ ordering between pipeline configs is what matters. Phase 3 will introduce LLM-judged relevance labels to validate.

---

## Phase Roadmap

| Phase | Description | Status |
|---|---|---|
| **0** | Data acquisition + OpenSearch + FAISS infrastructure | ✅ Complete |
| **1** | Marqo 7-dataset embedding benchmark reproduction (<1% delta) | ✅ Complete |
| **2** | Zero-shot full pipeline: BM25 + NER + Hybrid + CE rerank (253K queries) | ✅ Complete |
| **3** | LLM-guided cross-encoder + bi-encoder fine-tuning (nDCG@10 = 0.0757 SOTA) | ✅ Complete |
| **4A–4G** | Multimodal search: image embeddings, joint fine-tuning (4F), re-embed + eval (4G) | 🔄 In Progress |
| **4H** | Three-Tower Fashion Retriever training (novel architecture) | 🔜 Pending |
| **4I** | Three-Tower full benchmark evaluation | 🔜 Pending |
| **5** | Paper + HuggingFace release | 🔜 Planned |

---

## Citation

If you use this benchmark or code, please cite:

```bibtex
@misc{moda2026,
  title  = {MODA: The First Open Benchmark for End-to-End Fashion Search},
  author = {Rohit Anand},
  year   = {2026},
  url    = {https://github.com/your-username/moda}
}
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

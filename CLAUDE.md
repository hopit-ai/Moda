# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## What This Project Is

MODA is an open-source, end-to-end fashion search project with three parallel workstreams:

1. **H&M full-pipeline search benchmark** (`benchmark/` + `scripts/`) — measures every retrieval component (BM25, SPLADE, dense/CLIP, hybrid fusion, NER, cross-encoder reranking) against 253,685 purchase-grounded H&M queries × 105,542 products.
2. **Beat-FashionSigLIP track** (`scripts/v4/`) — fine-tunes Marqo-FashionSigLIP via GCL with pattern-targeted + synthetic data, with the explicit goal of exceeding FashionSigLIP's published MRR numbers on Marqo's own 7-dataset benchmark.
3. **HuggingFace model publishing** (`hf_repos/`) — five distilled fashion embedding models, each self-contained with its own `inference.py`.

---

## Environment Setup

```bash
# Python 3.10+, virtualenv at .venv/
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For GPU-only training (A100 runs)
pip install -r scripts/requirements_gpu.txt

# Start OpenSearch (required for any BM25 or SPLADE pipeline)
docker compose up -d
# OR one-liner without compose:
docker run -d -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e DISABLE_SECURITY_PLUGIN=true \
  opensearchproject/opensearch:2.11.0

# Sanity check the full environment
python scripts/verify_setup.py
```

---

## Track 1 — H&M Search Pipeline

### Architecture

```
Query
  ├─► SPLADE (naver/splade-cocondenser-ensembledistil)  ─┐
  ├─► Dense  (Marqo-FashionCLIP → FAISS)                ─┼─► RRF Fusion → CE Reranker → Top-10
  └─► BM25   (OpenSearch 2.11, optional)                ─┘
      [optional: GLiNER NER attribute boosts]
```

Best config: `SPLADE ⊕ FT-FashionCLIP (0.3/0.7) + LLM-trained CE` → nDCG@10 = 0.1063

### Data Preparation

```bash
# Download H&M data (253K queries → data/raw/hnm_real/)
python scripts/build_hnm_benchmark.py

# Index articles in OpenSearch
python benchmark/index_hnm_opensearch.py

# Embed 105K articles with FashionCLIP → FAISS index
python benchmark/embed_hnm.py --model fashion-clip

# Embed images for multimodal pipeline
python benchmark/embed_hnm_images.py
```

### Evaluation

```bash
# Full 253K overnight run (staged, checkpointed — can resume)
python benchmark/eval_full_253k.py --stages all
python benchmark/eval_full_253k.py --stages 1    # BM25 + FAISS only
python benchmark/eval_full_253k.py --stages 3    # CE reranking (8.5 hrs)

# Faster 10K sample (for iteration)
python benchmark/eval_hybrid.py
python benchmark/eval_full_pipeline.py

# SPLADE evaluation
python -m benchmark.eval_splade_pipeline          # test split (22K queries)
python -m benchmark.eval_full_253k_splade         # full 253K

# Multimodal pipeline
python benchmark/eval_multimodal_pipeline.py
bash scripts/run_phase4g_multimodal_eval.sh

# Specific ablations
python benchmark/eval_gliner2_ablation.py         # GLiNER v1 vs GLiNER2
python benchmark/eval_lookbench_baseline.py       # LookBench zero-shot baselines
```

### Training

```bash
# Fine-tune cross-encoder with LLM labels
python benchmark/train_cross_encoder.py

# Fine-tune bi-encoder (FashionCLIP) with hard negatives
python benchmark/train_biencoder.py

# Three-tower architecture training
python benchmark/train_three_tower.py --quick    # smoke test
python benchmark/train_three_tower.py            # full run
```

### Key Modules

| File | Role |
|------|------|
| `benchmark/metrics.py` | All IR metrics: nDCG, MRR, Recall, AP — used by all eval scripts |
| `benchmark/models.py` | `MODEL_REGISTRY` + loaders for `open_clip` and `sentence-transformers` |
| `benchmark/query_expansion.py` | `FashionNER` (GLiNER) + `SynonymExpander`; `build_boosted_query()` → OpenSearch |
| `benchmark/splade_retriever.py` | `SpladeRetriever` — encode articles/queries into sparse vectors, retrieve via dot product |
| `benchmark/article_text.py` | `build_article_text()` — canonical text representation of H&M articles for indexing |
| `benchmark/_faiss_flat_worker.py` | FAISS in subprocess to avoid MPS/BLAS conflicts on Apple Silicon |

### Evaluation Splits

- **Phase 2 (zero-shot configs):** all 253,685 queries — nothing was trained on this data
- **Phase 3+ (trained models):** 22,855-query held-out test split — disjoint from training to prevent leakage
- Data leakage checks: `benchmark/data_leakage_check_extended.py`

### Output Locations

- `results/full/full_ablation.json` — 253K final ablation results
- `results/real/` — 10K sample and test-split results
- `results/lookbench/` — LookBench per-subset JSON files

Long-running stages checkpoint to disk automatically. Re-running `eval_full_253k.py` with the same `--stages` flag resumes without repeating completed work. Run `caffeinate -i &` before overnight jobs on macOS to prevent sleep.

---

## Track 2 — Beat-FashionSigLIP

**Goal:** exceed Marqo-FashionSigLIP's published MRR numbers on Marqo's own 7-dataset benchmark using GCL fine-tuning with pattern-targeted + synthetic training data.

### Targets (FashionSigLIP zero-shot baseline, MRR)

From `results/v4_gcl/baseline_v4/full_results.json` — these are the numbers we have to beat:

| Dataset | text_to_image MRR | category_to_product MRR |
|---|---|---|
| polyvore | **0.7402** | 0.6939 |
| iMaterialist | — | 0.6502 |
| KAGL | 0.5805 | — |
| fashion200k | 0.4551 | — |
| atlas | 0.4226 | 1.0000 (saturated) |
| deepfashion_inshop | 0.4077 | — |
| deepfashion_multimodal | 0.0329 | — |

Latest fine-tuned checkpoint (`checkpoints/v4_gcl/final_model.pt`, ~50 steps) still trails on every gap dataset — see `results/v4_gcl/ft_v4/gap_analysis.json`. Track is early; longer training on more data is the next step.

### Method

- **Backbone:** FashionSigLIP (`hf-hub:Marqo/marqo-fashionSigLIP`, ViT-B-16-SigLIP, 203M params)
- **Loss:** GCL (Generalized Contrastive Learning) with `score_to_weight: inverse_sqrt`
- **Encoding:** LHS = query text · RHS = weighted (image × 0.9, title × 0.1) — multi-field RHS
- **Anchor regularization:** prevents catastrophic forgetting of baseline capabilities
- **Early stopping:** per-benchmark eval every N steps; halt if any benchmark regresses >3%
- **Recommended retrain LR:** ~1e-6 for 1 epoch from best checkpoint

### Data Pipeline

Located in `data/processed/v4_pattern_targeted/`. Current totals: 118,852 mined + 37,000 synthetic ≈ 155K pairs.

**Step 1 — Mine `Marqo/marqo-GS-10M` into pattern buckets** (`phase1_build_pattern_dataset.py`)

Streams the dataset (no full download), classifies into 9 buckets, saves images at 224px:

| Bucket | Target | Current |
|---|---:|---:|
| apparel | 50,000 | 49,785 |
| short_title | 40,000 | 39,927 |
| long_description | 30,000 | 52 ← gap |
| accessories | 20,000 | 19,867 |
| non_fashion | 15,000 | 15,570 |
| compound_attr | 15,000 | 14,890 |
| color_centric | 15,000 | 14,890 |
| footwear | 15,000 | 15,045 |
| brand_product | 10,000 | 9,963 |

**Step 2 — Synthetic gap-fill via LLM** (`phase1b_synthetic_gapfill.py`)

Uses anchor images from step 1 and generates new text pairings via PaleblueDot API (`qwen/qwen3.5-flash`). Without `PALEBLUEDOT_API_KEY` set, falls back to deterministic templates with `--local-only`.

Current 37,000 synthetic pairs:

| Type | Count | Targets which gap |
|---|---:|---|
| long_description | 10,000 | DF-Multimodal-style attribute templates |
| lifestyle | 8,000 | Atlas/Polyvore home-decor + non-fashion |
| paraphrase | 6,000 | Diversity — paraphrase existing queries |
| template_description | 5,000 | "The upper clothing has..." templates |
| category_query | 5,000 | Short categorical labels (KAGL/Atlas) |
| compound_color | 3,000 | "dark navy blue", "muted dusty pink" |

**Step 3 — Leakage check** (`phase1c_leakage_check.py`)

Cross-checks `pairs.jsonl` and `synthetic_pairs.jsonl` against all 7 Marqo benchmarks for product_id, query, and title overlaps. Currently flags Polyvore generic queries (290 pairs removed); `stats.json` carries `leakage_clean: true` post-removal. Re-run after **any** synthetic data change.

### Commands

```bash
# Step 1 — Mine GS-10M (~120K pairs into 9 pattern buckets)
python scripts/v4/phase1_build_pattern_dataset.py

# Step 2 — Synthetic gap-fill
PALEBLUEDOT_API_KEY=... python scripts/v4/phase1b_synthetic_gapfill.py
python scripts/v4/phase1b_synthetic_gapfill.py --local-only   # no API

# Step 3 — Leakage check (run after every synthetic change)
python scripts/v4/phase1c_leakage_check.py

# Step 4 — GCL fine-tuning of FashionSigLIP
python scripts/v4/phase2_train_gcl.py

# Step 5 — Evaluate on all 7 Marqo benchmarks (full settings, NOT --fast)
python scripts/v4/phase3_eval_all_benchmarks.py

# Step 6 — Per-dataset gap analysis + next-iteration recipes
python scripts/v4/phase3b_gap_analysis.py
python scripts/v4/phase4_iterate.py
```

### Retrain Loop

When evaluation shows a gap on a specific dataset, the recipe is in `data/processed/v4_pattern_targeted/iteration_plan.md` (auto-generated from `gap_analysis.json`). Each gap maps to a data strategy — e.g. *"deepfashion_inshop −28%: more in-shop product photos with brand+color queries"*. Loop:

1. Re-run step 1 with mining biased toward the gap bucket
2. Optionally extend step 2 templates for the weakest dataset
3. Re-run step 3 leakage check
4. Resume training from `final_model.pt` at LR ~1e-6 for 1 epoch
5. Re-run step 5 (full eval, not `--fast`)

### Key Modules

| File | Role |
|------|------|
| `scripts/v4/phase1.yaml` | Dataset list, model registry, eval config (`ks`, `top_k_retrieval`, batch_size, device) |
| `scripts/v4/phase1_build_pattern_dataset.py` | GS-10M streaming + bucket classifier (9 keyword dictionaries: APPAREL_KW, ACCESSORIES_KW, etc.) |
| `scripts/v4/phase1b_synthetic_gapfill.py` | PaleblueDot API client + 6 generation templates; `--local-only` skips the API |
| `scripts/v4/phase1c_leakage_check.py` | Cross-checks against Marqo benchmark HF datasets (`Marqo/deepfashion-inshop`, etc.) |
| `scripts/v4/phase2_train_gcl.py` | `PatternTargetedDataset` + GCL loss + anchor reg + per-benchmark early stopping |
| `scripts/v4/phase3_eval_all_benchmarks.py` | Evals on 7 Marqo HF datasets × 3 task types (text_to_image, category_to_product, subcategory_to_product) |
| `scripts/v4/phase3b_gap_analysis.py` | Diff vs baseline, emit per-dataset recipes |
| `scripts/v4/phase4_iterate.py` | Drives the next iteration based on gap_analysis output |

### Output Locations

- `results/v4_gcl/baseline_v4/full_results.json` — FashionSigLIP zero-shot numbers (the targets)
- `results/v4_gcl/ft_v4/full_results.json` — current fine-tuned checkpoint numbers
- `results/v4_gcl/ft_v4/gap_analysis.json` — per-(dataset,task) MRR delta + mining recipes
- `results/benchmark_patterns/cross_dataset_analysis.md` — pattern overlap analysis across the 7 benchmarks
- `checkpoints/v4_gcl/final_model.pt` — current best fine-tuned weights
- `data/processed/v4_pattern_targeted/iteration_plan.md` — actionable next-mining-round recipes

---

## Track 3 — HuggingFace Model Publishing

Five published models in `hf_repos/`, each self-contained with `inference.py`:

| Repo | Architecture | Dim | Notes |
|------|-------------|-----|-------|
| `moda-fashion-distilled` | ViT-B-16-SigLIP | 768 | Best LookBench single model (67.63% Fine R@1) |
| `moda-fashion-distilled-512d` | ViT-B-16-SigLIP | 512 | Smaller, fast |
| `moda-fashion-matryoshka` | ViT-B-16-SigLIP | 768→64 | Truncatable embeddings |
| `moda-fashion-vision-fp16` | ViT-B-16-SigLIP | 768 | Vision-only, FP16 |
| `moda-fashion-deepfashion2` | ViT-B-16-SigLIP | 768 | DeepFashion2 fine-tuned |

All load via `open_clip.create_model_and_transforms("ViT-B-16-SigLIP", pretrained=str(weights_path))`.

### Commands

```bash
# Validate all 5 published models load and produce sane embeddings
python benchmark/test_hf_models_inference.py

# Distillation source scripts
python benchmark/distill_matryoshka.py
python benchmark/distill_512d_native.py
# Other distillation variants in benchmark/distill_*.py
```

### Phase 1 Marqo Benchmark Reproduction

```bash
python scripts/download_datasets.py
python benchmark/eval_marqo_7dataset.py --models fashion-clip fashion-siglip
```

---

## Data Layout

```
data/
├── raw/hnm_real/                ← H&M articles.csv, queries.csv, qrels.csv (~200MB, gitignored)
│                                  Download: python scripts/build_hnm_benchmark.py
├── raw/{deepfashion_inshop,fashion200k,atlas,polyvore,KAGL,...}  (~8.8GB, gitignored)
│                                  Download: python scripts/download_datasets.py
└── processed/
    ├── embeddings/              ← FAISS indexes + article_ids.json (~1.4GB, gitignored)
    ├── v4_pattern_targeted/     ← Track 2 training data
    │   ├── pairs.jsonl                ← 118,852 GS-10M mined pairs (leakage_clean)
    │   ├── synthetic_pairs.jsonl      ← 37,000 LLM/template synthetic pairs
    │   ├── images/                    ← 224px product images keyed by pair records
    │   ├── stats.json                 ← bucket counts + leakage status
    │   ├── synthetic_stats.json       ← per-template synthetic counts
    │   ├── leakage_results.json       ← per-benchmark overlap report
    │   ├── quality_report.md          ← dataset size summary
    │   └── iteration_plan.md          ← per-dataset gap recipes for next mining round
    ├── synthetic/               ← Combined synthetic data + phase 10/11 caption sets
    └── labels_v1*/              ← LLM-judged H&M relevance labels
```

H&M ground truth (`qrels.csv`): purchased article = grade 2, shown-but-not-bought = grade 1, rest = 0.

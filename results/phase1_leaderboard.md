# MODA Phase 1 — Benchmark Leaderboard
_Generated: 2026-04-02 18:39 UTC_

## Overview

| Tier | Description | Status |
| --- | --- | --- |
| Tier 1 | Marqo 7-dataset Text-to-Image Recall | Partial (deepfashion_inshop done) |
| Tier 2 | H&M Full-Pipeline (MODA's new benchmark) | ✅ All 3 models |
| Tier 3 | FashionIQ Composed Retrieval | 🔜 Phase 2 |

---

## Tier 2 — H&M Full-Pipeline Benchmark (MODA Contribution)

> **Benchmark design:** Category-based relevance over 105,542 H&M articles.
> Queries are `product_type_name` labels. Positive articles share the same type,
> negative articles share garment group but differ in type.
> 4,078 evaluation queries.

| Model | nDCG@5 | nDCG@10 | nDCG@20 | MRR | Recall@10 | Recall@20 | P@10 | Latency (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CLIP ViT-B/32 | 0.0751 | 0.0966 | 0.1287 | 0.1154 | 0.0125 | 0.0222 | 0.0485 | 0.03 |
| marqo-fashionCLIP | 0.0668 | 0.0886 | 0.1190 | 0.1088 | 0.0125 | 0.0219 | 0.0477 | 0.04 |
| marqo-fashionSigLIP | 0.0579 | 0.0765 | 0.1040 | 0.0904 | 0.0100 | 0.0174 | 0.0377 | 0.03 |

---

## Tier 1 — Marqo 7-Dataset Text-to-Image Embedding Benchmark

> Reproduces Marqo FashionCLIP paper evaluation. Metrics: Recall@1, Recall@10, MRR.

### Atlas (sub-category-to-product)

| Model | MAP@1 | MAP@10 | MRR | NDCG@1 | NDCG@10 | P@1 | P@10 | Recall@1 | Recall@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| marqo-fashionCLIP | 0.0048 | 0.0333 | 0.7284 | 0.7059 | 0.6584 | 0.7059 | 0.6382 | 0.0048 | 0.0547 |
| marqo-fashionSigLIP | 0.0146 | 0.0556 | 0.7439 | 0.7059 | 0.6854 | 0.7059 | 0.6706 | 0.0146 | 0.0647 |

### Atlas (text-to-image)

| Model | MAP@1 | MAP@10 | MRR | NDCG@1 | NDCG@10 | P@1 | P@10 | Recall@1 | Recall@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| marqo-fashionCLIP | 0.0735 | 0.1289 | 0.1691 | 0.1090 | 0.1693 | 0.1090 | 0.0464 | 0.0735 | 0.2466 |
| marqo-fashionSigLIP | 0.1072 | 0.1826 | 0.2337 | 0.1640 | 0.2338 | 0.1640 | 0.0654 | 0.1072 | 0.3239 |

### DeepFashion InShop (category-to-product)

| Model | MAP@1 | MAP@10 | MRR | NDCG@1 | NDCG@10 | P@1 | P@10 | Recall@1 | Recall@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| marqo-fashionCLIP | 0.0003 | 0.0034 | 0.7225 | 0.6250 | 0.6842 | 0.6250 | 0.7000 | 0.0003 | 0.0053 |
| marqo-fashionSigLIP | 0.0020 | 0.0081 | 0.8646 | 0.8125 | 0.7886 | 0.8125 | 0.7875 | 0.0020 | 0.0102 |

### DeepFashion InShop (color-to-product)

| Model | MAP@1 | MAP@10 | MRR | NDCG@1 | NDCG@10 | P@1 | P@10 | Recall@1 | Recall@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| marqo-fashionCLIP | 0.0013 | 0.0051 | 0.0547 | 0.0348 | 0.0324 | 0.0348 | 0.0290 | 0.0013 | 0.0092 |
| marqo-fashionSigLIP | 0.0021 | 0.0075 | 0.0666 | 0.0423 | 0.0386 | 0.0423 | 0.0342 | 0.0021 | 0.0139 |

### DeepFashion InShop (sub-category-to-product)

| Model | MAP@1 | MAP@10 | MRR | NDCG@1 | NDCG@10 | P@1 | P@10 | Recall@1 | Recall@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| marqo-fashionCLIP | 0.0008 | 0.0062 | 0.8750 | 0.8750 | 0.7668 | 0.8750 | 0.7625 | 0.0008 | 0.0068 |
| marqo-fashionSigLIP | 0.0008 | 0.0057 | 0.8750 | 0.8750 | 0.7435 | 0.8750 | 0.7250 | 0.0008 | 0.0061 |

### DeepFashion InShop (text-to-image)

| Model | MAP@1 | MAP@10 | MRR | NDCG@1 | NDCG@10 | P@1 | P@10 | Recall@1 | Recall@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| marqo-fashionCLIP | 0.0482 | 0.1475 | 0.3307 | 0.2370 | 0.2276 | 0.2370 | 0.1280 | 0.0482 | 0.2386 |
| marqo-fashionSigLIP | 0.0499 | 0.1587 | 0.3300 | 0.2435 | 0.2367 | 0.2435 | 0.1331 | 0.0499 | 0.2477 |

### DeepFashion Multimodal (category-to-product)

| Model | MAP@1 | MAP@10 | MRR | NDCG@1 | NDCG@10 | P@1 | P@10 | Recall@1 | Recall@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| marqo-fashionCLIP | 0.0008 | 0.0043 | 0.8125 | 0.8125 | 0.6921 | 0.8125 | 0.6813 | 0.0008 | 0.0048 |
| marqo-fashionSigLIP | 0.0032 | 0.0112 | 0.7969 | 0.7500 | 0.6923 | 0.7500 | 0.6813 | 0.0032 | 0.0141 |

### DeepFashion Multimodal (text-to-image)

| Model | MAP@1 | MAP@10 | MRR | NDCG@1 | NDCG@10 | P@1 | P@10 | Recall@1 | Recall@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| marqo-fashionCLIP | 0.0045 | 0.0084 | 0.0085 | 0.0045 | 0.0119 | 0.0045 | 0.0024 | 0.0045 | 0.0235 |
| marqo-fashionSigLIP | 0.0080 | 0.0148 | 0.0149 | 0.0080 | 0.0187 | 0.0080 | 0.0032 | 0.0080 | 0.0312 |

### Fashion200K (category-to-product)

| Model | MAP@1 | MAP@10 | MRR | NDCG@1 | NDCG@10 | P@1 | P@10 | Recall@1 | Recall@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| marqo-fashionCLIP | 0.0000 | 0.0003 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0003 |
| marqo-fashionSigLIP | 0.0000 | 0.0002 | 1.0000 | 1.0000 | 0.9264 | 1.0000 | 0.9200 | 0.0000 | 0.0003 |

### Fashion200K (fine-category-to-product)

| Model | MAP@1 | MAP@10 | MRR | NDCG@1 | NDCG@10 | P@1 | P@10 | Recall@1 | Recall@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| marqo-fashionCLIP | 0.0418 | 0.0902 | 0.1322 | 0.0865 | 0.1233 | 0.0865 | 0.0425 | 0.0418 | 0.1710 |
| marqo-fashionSigLIP | 0.0721 | 0.1460 | 0.2029 | 0.1475 | 0.1871 | 0.1475 | 0.0600 | 0.0721 | 0.2430 |

### Fashion200K (sub-category-to-product)

| Model | MAP@1 | MAP@10 | MRR | NDCG@1 | NDCG@10 | P@1 | P@10 | Recall@1 | Recall@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| marqo-fashionCLIP | 0.0002 | 0.0016 | 0.7968 | 0.7419 | 0.6806 | 0.7419 | 0.6645 | 0.0002 | 0.0017 |
| marqo-fashionSigLIP | 0.0002 | 0.0014 | 0.8521 | 0.7419 | 0.6555 | 0.7419 | 0.6452 | 0.0002 | 0.0017 |

### Fashion200K (text-to-image)

| Model | MAP@1 | MAP@10 | MRR | NDCG@1 | NDCG@10 | P@1 | P@10 | Recall@1 | Recall@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| marqo-fashionCLIP | 0.0660 | 0.1244 | 0.1247 | 0.0660 | 0.1611 | 0.0660 | 0.0286 | 0.0660 | 0.2796 |
| marqo-fashionSigLIP | 0.1108 | 0.1858 | 0.1868 | 0.1115 | 0.2317 | 0.1115 | 0.0389 | 0.1108 | 0.3784 |

### KAGL (category-to-product)

| Model | MAP@1 | MAP@10 | MRR | NDCG@1 | NDCG@10 | P@1 | P@10 | Recall@1 | Recall@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| marqo-fashionCLIP | 0.0058 | 0.0185 | 0.7143 | 0.7143 | 0.6039 | 0.7143 | 0.5857 | 0.0058 | 0.0238 |
| marqo-fashionSigLIP | 0.0001 | 0.0362 | 0.6429 | 0.5714 | 0.6738 | 0.5714 | 0.6857 | 0.0001 | 0.0467 |

### KAGL (color-to-product)

| Model | MAP@1 | MAP@10 | MRR | NDCG@1 | NDCG@10 | P@1 | P@10 | Recall@1 | Recall@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| marqo-fashionCLIP | 0.0037 | 0.0150 | 0.6115 | 0.5000 | 0.4954 | 0.5000 | 0.4913 | 0.0037 | 0.0293 |
| marqo-fashionSigLIP | 0.0039 | 0.0170 | 0.6643 | 0.5870 | 0.5175 | 0.5870 | 0.5022 | 0.0039 | 0.0278 |

### KAGL (fine-category-to-product)

| Model | MAP@1 | MAP@10 | MRR | NDCG@1 | NDCG@10 | P@1 | P@10 | Recall@1 | Recall@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| marqo-fashionCLIP | 0.0417 | 0.1655 | 0.7728 | 0.7254 | 0.6735 | 0.7254 | 0.6120 | 0.0417 | 0.2103 |
| marqo-fashionSigLIP | 0.0403 | 0.1756 | 0.7632 | 0.7042 | 0.6850 | 0.7042 | 0.6296 | 0.0403 | 0.2198 |

### KAGL (season-to-product)

| Model | MAP@1 | MAP@10 | MRR | NDCG@1 | NDCG@10 | P@1 | P@10 | Recall@1 | Recall@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| marqo-fashionCLIP | 0.0001 | 0.0003 | 0.5417 | 0.2500 | 0.4059 | 0.2500 | 0.4250 | 0.0001 | 0.0005 |
| marqo-fashionSigLIP | 0.0000 | 0.0002 | 0.5000 | 0.5000 | 0.3657 | 0.5000 | 0.3500 | 0.0000 | 0.0002 |

### KAGL (sub-category-to-product)

| Model | MAP@1 | MAP@10 | MRR | NDCG@1 | NDCG@10 | P@1 | P@10 | Recall@1 | Recall@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| marqo-fashionCLIP | 0.0168 | 0.0767 | 0.6911 | 0.6667 | 0.6383 | 0.6667 | 0.6111 | 0.0168 | 0.0939 |
| marqo-fashionSigLIP | 0.0152 | 0.1093 | 0.7565 | 0.7111 | 0.7169 | 0.7111 | 0.6867 | 0.0152 | 0.1296 |

### KAGL (text-to-image)

| Model | MAP@1 | MAP@10 | MRR | NDCG@1 | NDCG@10 | P@1 | P@10 | Recall@1 | Recall@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| marqo-fashionCLIP | 0.1580 | 0.2473 | 0.2645 | 0.1760 | 0.3001 | 0.1760 | 0.0537 | 0.1580 | 0.4471 |
| marqo-fashionSigLIP | 0.1794 | 0.2769 | 0.2932 | 0.1975 | 0.3326 | 0.1975 | 0.0587 | 0.1794 | 0.4888 |

### KAGL (usage-to-product)

| Model | MAP@1 | MAP@10 | MRR | NDCG@1 | NDCG@10 | P@1 | P@10 | Recall@1 | Recall@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| marqo-fashionCLIP | 0.0000 | 0.0004 | 0.3542 | 0.2500 | 0.2732 | 0.2500 | 0.2625 | 0.0000 | 0.0005 |
| marqo-fashionSigLIP | 0.0001 | 0.0009 | 0.4375 | 0.3750 | 0.4011 | 0.3750 | 0.4000 | 0.0001 | 0.0010 |

### Polyvore (category-to-product)

| Model | MAP@1 | MAP@10 | MRR | NDCG@1 | NDCG@10 | P@1 | P@10 | Recall@1 | Recall@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| marqo-fashionCLIP | 0.0308 | 0.1064 | 0.6286 | 0.5146 | 0.4812 | 0.5146 | 0.4294 | 0.0308 | 0.1550 |
| marqo-fashionSigLIP | 0.0419 | 0.1336 | 0.6965 | 0.5968 | 0.5438 | 0.5968 | 0.4735 | 0.0419 | 0.1885 |

### Polyvore (text-to-image)

| Model | MAP@1 | MAP@10 | MRR | NDCG@1 | NDCG@10 | P@1 | P@10 | Recall@1 | Recall@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| marqo-fashionCLIP | 0.2130 | 0.3004 | 0.3004 | 0.2130 | 0.3509 | 0.2130 | 0.0513 | 0.2130 | 0.5135 |
| marqo-fashionSigLIP | 0.2695 | 0.3664 | 0.3664 | 0.2695 | 0.4184 | 0.2695 | 0.0584 | 0.2695 | 0.5840 |

---

## Tier 3 — FashionIQ Composed Retrieval

_Planned for Phase 2. Metric: Recall@{10,50}._

---

## Notes

- **Tier 2 query design:** synthetic category-based queries from H&M article metadata.
  Phase 2 will extend with purchase-signal-based relevance using transaction data.
- **Tier 1 status:** Only `deepfashion_inshop × marqo-fashionSigLIP` completed in Phase 1
  (due to 1hr+ embedding times per dataset per model). Remaining runs are queued for Phase 2.
- **Device:** Apple MPS (M-series) used for article embedding; CPU used for query encoding.
- All models used zero-shot (no fine-tuning on H&M data).

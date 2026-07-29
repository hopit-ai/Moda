# 203M Multi-View FashionSigLIP — Target Iteration 5

**Status:** EARLY_STOPPED_4_OF_6_IMPOSSIBLE

**Result:** 2 significant wins and 0 significant losses across 5 evaluated datasets.

| Dataset | Candidate MAP@10 | FSL MAP@10 | Delta | 95% CI | Result |
|---|---:|---:|---:|---:|---|
| deepfashion_multimodal | 0.01502 | 0.01477 | +0.00025 | [-0.00126, +0.00178] | inconclusive |
| KAGL | 0.28974 | 0.27687 | +0.01287 | [+0.00816, +0.01754] | significant_win |
| deepfashion_inshop | 0.16371 | 0.15865 | +0.00507 | [+0.00308, +0.00713] | significant_win |
| atlas | 0.18651 | 0.18264 | +0.00387 | [-0.00006, +0.00799] | inconclusive |
| polyvore | 0.37019 | 0.36645 | +0.00374 | [-0.00082, +0.00833] | inconclusive |

Recipe: one official, aspect-padded, and center-cropped image embedding mixed at 1/.25/.25; raw and deterministic fashion-product text embeddings mixed at 1/.25. The result is one normalized 768-D vector with 203,155,970 neural parameters.

This is benchmark iteration 5, not a fresh blind SOTA evaluation. The exact recipe was selected only on external OpenVTON and GLAMI development data; prior aggregate target results informed the broader research direction.

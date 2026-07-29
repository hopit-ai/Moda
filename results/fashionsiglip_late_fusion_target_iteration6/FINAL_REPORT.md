# 203M Three-Route Late-Fusion FashionSigLIP — Target Iteration 6

**Status:** AT_LEAST_4_OF_6_ACHIEVED

**Result:** 4 significant wins and 0 significant losses across 6 evaluated datasets.

| Dataset | Candidate MAP@10 | FSL MAP@10 | Delta | 95% CI | Result |
|---|---:|---:|---:|---:|---|
| deepfashion_multimodal | 0.01504 | 0.01477 | +0.00028 | [-0.00121, +0.00192] | inconclusive |
| KAGL | 0.29074 | 0.27687 | +0.01387 | [+0.00894, +0.01883] | significant_win |
| deepfashion_inshop | 0.16371 | 0.15865 | +0.00507 | [+0.00304, +0.00710] | significant_win |
| atlas | 0.18637 | 0.18264 | +0.00374 | [-0.00017, +0.00775] | inconclusive |
| polyvore | 0.37191 | 0.36645 | +0.00547 | [+0.00086, +0.01014] | significant_win |
| fashion200k | 0.19510 | 0.18577 | +0.00932 | [+0.00382, +0.01477] | significant_win |

Recipe: the locked iteration-5 multi-view query and gallery descriptor plus a 10% max-over-view score correction from the aspect-padded and center-cropped image routes.

Deployment: 203,155,970 neural parameters, one query vector, three stored 768-D gallery vectors, and three ANN routes.

This is benchmark iteration 6, not a fresh blind SOTA evaluation. Previous aggregate target results were known, but this exact late-fusion blend was selected only on external OpenVTON and GLAMI development features.

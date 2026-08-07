# AGENTS.md

Orientation for coding agents working in this repository.

## What this repository is

MODA is an open-source fashion retrieval benchmark and model family by Hopit AI. It covers image-to-image retrieval (find a product from a photo) and text-to-image retrieval (find a product from a description). Results are reproducible from the scripts here; every headline number has an artifact behind it.

## Layout

| Path | Contents |
|---|---|
| `benchmark/` | Evaluation and training scripts. One file per experiment, named `eval_*` or `train_*`. |
| `results/` | Frozen result artifacts (JSON and Markdown). Treat these as receipts, not scratch space. |
| `hf_repos/` | Packaged model repositories mirrored to Hugging Face. Small files only; weights are never committed. |
| `scripts/` | Repository tooling, data preparation, and packaging. |
| `train/`, `eval/`, `configs/` | Training entry points, evaluation configs, and experiment configuration. |
| `blog_post*.md` | The published write-up for each phase. |

## Rules that matter here

**Never commit model weights.** `.safetensors`, `.bin`, `.pt` and `.pth` are gitignored on purpose. Model repositories under `hf_repos/` carry code, configs, and result summaries only. The text-to-image system downloads its base checkpoint (`Marqo/marqo-fashionSigLIP`) at runtime rather than vendoring it.

**Frozen results are immutable.** Directories such as `results/fashionsiglip_late_fusion_target_iteration6/` are locked receipts for a published claim. Do not edit their numbers. A new result gets a new directory.

**Claims must match artifacts.** Every performance number in the README, a model card, or a blog post should be traceable to a file under `results/`. If you change a number in prose, change it because the artifact changed.

**Stay inside the claim boundary.** The text-to-image result is a retrieval system over frozen weights, evaluated non-blind at benchmark iteration six. Do not describe it as a new checkpoint, a six-of-six win, or an independently verified state-of-the-art result. `FSL_203M_SYSTEM_ARCHITECTURE.md` states the exact defensible wording.

**No secrets.** Tokens, keys, `.env` files, and licensed source data stay out of the repository. Several fashion datasets used in research here are research-only or noncommercial and must not be redistributed.

## Running things

```bash
pip install -r requirements.txt

# text-to-image retrieval
pip install "git+https://huggingface.co/HopitAI/moda-fashionsiglip-multiview-203m"
python inference.py --gallery ./my_catalog --query "red floral summer dress"

# image-to-image embeddings
python inference.py --image path/to/image.jpg
```

Full reproduction steps, including the OpenSearch and FAISS setup for the H&M pipeline, are in the README under "Reproducing the results". The heavy evaluations are designed to run on Apple Silicon without a GPU; several take hours and checkpoint to disk so they can resume.

## Conventions

Python, standard library plus the pinned requirements. Match the surrounding style of the file you are editing. Evaluation scripts print a result table and write a JSON artifact under `results/`; keep both when adding a new experiment.

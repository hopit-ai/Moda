# Publishing the MODA Multi-View Retrieval System

This document is an operator guide only. Nothing in the repository uploads
files automatically.

## Local artifacts

| Hugging Face artifact | Local source |
|---|---|
| Model/system repository | `hf_repos/moda-fashionsiglip-multiview-203m/` |
| Interactive Gradio Space | `hf_spaces/moda-fashionsiglip-multiview-demo/` |

The system repository contains code, the frozen recipe, tests, and compact
benchmark results. It does **not** duplicate the Marqo FashionSigLIP weights;
the package downloads `Marqo/marqo-fashionSigLIP` when first used.

## Pre-publication checklist

1. Run the package tests in a clean environment.
2. Run `example_retrieval.py` against a small local image folder.
3. Confirm the result table matches
   `results/fashionsiglip_late_fusion_target_iteration6/FINAL_SUMMARY.json`.
4. Search both upload folders for tokens, private paths, raw datasets, images,
   checkpoints, and generated indexes.
5. Review the model card disclosure. Do not call this a new checkpoint, a
   six-of-six significant win, or an independent blind SOTA result.
6. Create private repositories first and inspect their file lists before
   making them public.

## Authenticate safely

Install the Hub client and authenticate interactively:

```bash
python -m pip install --upgrade huggingface_hub
hf auth login
```

Never put a Hugging Face token in source code, a shell script, a model card,
Git history, or an uploaded `.env` file.

## Upload to private staging repositories

Run the following from the MODA repository root. The snippet intentionally
creates both artifacts as private:

```python
from huggingface_hub import HfApi

api = HfApi()

model_repo = "HopitAI/moda-fashionsiglip-multiview-203m"
space_repo = "HopitAI/moda-fashionsiglip-multiview-demo"

api.create_repo(
    repo_id=model_repo,
    repo_type="model",
    private=True,
    exist_ok=True,
)
api.upload_folder(
    repo_id=model_repo,
    repo_type="model",
    folder_path="hf_repos/moda-fashionsiglip-multiview-203m",
    ignore_patterns=[
        "**/__pycache__/**",
        "**/.pytest_cache/**",
        "**/*.egg-info/**",
        "**/*.pyc",
    ],
    commit_message="Publish MODA multi-view retrieval system",
)
```

The model/system repository must be uploaded and made public before the Space
can build because the Space installs the package from that repository.

## Private-stage validation

In a clean environment:

```bash
pip install \
  "git+https://huggingface.co/HopitAI/moda-fashionsiglip-multiview-203m"

python -c \
  "from moda_fashionsiglip_multiview import load_recipe; print(load_recipe())"
```

Then check the model/system repository:

- the model card renders correctly;
- the repository contains no model weights or private data;
- the package installs from the Hub;
- local images can be indexed and searched.

## Publish the model repository

Only after the private model checks pass, make the model/system repository
public through its Hugging Face settings page.

## Stage the Space privately

After the model repository is public, create and upload the private Space:

```python
from huggingface_hub import HfApi

api = HfApi()
space_repo = "HopitAI/moda-fashionsiglip-multiview-demo"

api.create_repo(
    repo_id=space_repo,
    repo_type="space",
    space_sdk="gradio",
    private=True,
    exist_ok=True,
)
api.upload_folder(
    repo_id=space_repo,
    repo_type="space",
    folder_path="hf_spaces/moda-fashionsiglip-multiview-demo",
    ignore_patterns=[
        "**/__pycache__/**",
        "**/*.pyc",
    ],
    commit_message="Publish MODA multi-view retrieval demo",
)
```

Confirm that the Space builds successfully, uploaded images can be indexed and
searched, and CPU memory and first-load time are acceptable. Only then make
the Space public.

## Updating later

Edit and test the local GitHub-controlled folders first, then rerun the two
`upload_folder` calls. Hugging Face will create a new versioned commit. Keep
the frozen recipe and benchmark summary immutable unless a new result is
clearly versioned and documented.

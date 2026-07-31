---
tags:
- clip
- open-clip
- siglip
- fashion
- text-to-image-retrieval
- multimodal-retrieval
- embedding
- mps
library_name: open_clip
pipeline_tag: image-feature-extraction
license: mit
language:
- en
base_model: Marqo/marqo-fashionSigLIP
metrics:
- map
---

# MODA FashionSigLIP Multi-View 203M

A downloadable implementation of the zero-additional-parameter retrieval
architecture that produced four statistically significant full-corpus
text-to-image wins over official FashionSigLIP.

> This is a **retrieval system**, not a newly trained checkpoint. It downloads
> the unchanged Apache-2.0
> [`Marqo/marqo-fashionSigLIP`](https://huggingface.co/Marqo/marqo-fashionSigLIP)
> weights and applies a frozen query, gallery, and scoring recipe.

## Result

The same global recipe was evaluated on all six datasets using full-corpus
MAP@10 and 10,000 paired-bootstrap samples.

| Dataset | MODA MAP@10 | FashionSigLIP | Delta | 95% CI | Classification |
|---|---:|---:|---:|---:|---|
| KAGL | **0.29074** | 0.27687 | +0.01387 | [+0.00894, +0.01883] | significant win |
| Fashion200K | **0.19510** | 0.18577 | +0.00932 | [+0.00382, +0.01477] | significant win |
| DeepFashion In-Shop | **0.16371** | 0.15865 | +0.00507 | [+0.00304, +0.00710] | significant win |
| Polyvore | **0.37191** | 0.36645 | +0.00547 | [+0.00086, +0.01014] | significant win |
| Atlas | **0.18637** | 0.18264 | +0.00374 | [-0.00017, +0.00775] | inconclusive |
| DeepFashion Multimodal | **0.01504** | 0.01477 | +0.00028 | [-0.00121, +0.00192] | inconclusive |

Accurate summary: **4/6 significant wins, positive point estimates on 6/6,
and zero significant losses**.

## Install

```bash
pip install \
  "git+https://huggingface.co/HopitAI/moda-fashionsiglip-multiview-203m"
```

The first model load downloads the upstream FashionSigLIP checkpoint.

## Quick start

```python
from pathlib import Path

from moda_fashionsiglip_multiview import ModaFashionSigLIP

images = sorted(Path("my_catalog").glob("*.jpg"))

retriever = ModaFashionSigLIP.from_pretrained()  # CUDA → MPS → CPU
index = retriever.build_index(
    images,
    item_ids=[path.name for path in images],
)

results = retriever.search(
    "red floral summer dress",
    index,
    top_k=5,
)[0]

for result in results:
    print(result.rank, result.score, result.item_id)
```

Apple Silicon is supported explicitly:

```python
retriever = ModaFashionSigLIP.from_pretrained(device="mps")
```

The installation provides a complete folder-search command:

```bash
moda-fashion-search \
  --gallery ./my_catalog \
  --query "black leather ankle boots" \
  --top-k 10 \
  --device mps
```

## Standalone inference script

`inference.py` runs from a plain clone of this repository without installing
the package. It covers the three things people usually want first: ranking a
folder against a query, embedding images, and embedding a query.

```bash
# rank a folder of product images against a text query
python inference.py --gallery ./my_catalog --query "red floral summer dress"

# embed images (768-d parent vectors)
python inference.py --image img1.jpg img2.jpg

# embed a query
python inference.py --query "black leather ankle boots"

# build once, reuse the saved index later
python inference.py --gallery ./my_catalog --query "navy linen shirt" \
  --save-index ./catalog_index
python inference.py --load-index ./catalog_index --query "wool coat"
```

## Save and reload a gallery

The three gallery routes are stored with SafeTensors; no pickle loading is
required.

```python
index.save_pretrained("./catalog_index")

from moda_fashionsiglip_multiview import GalleryIndex

index = GalleryIndex.from_pretrained("./catalog_index")
results = retriever.search("navy linen shirt", index)[0]
```

## Architecture

### Query

```text
q_raw    = FashionSigLIP(query)
q_prompt = FashionSigLIP("a fashion product photo of " + query)
q        = normalize(q_raw + 0.25 × q_prompt)
```

### Gallery

```text
d_official = FashionSigLIP(original image)
d_pad      = FashionSigLIP(gray-128 square-padded image)
d_crop     = FashionSigLIP(center-square-cropped image)

d_parent = normalize(
    d_official + 0.25 × d_pad + 0.25 × d_crop
)
```

The index stores `d_parent`, `d_pad`, and `d_crop`.

### Score

```text
s_parent = cosine(q, d_parent)
s_pad    = cosine(q, d_pad)
s_crop   = cosine(q, d_crop)

score = 0.9 × s_parent
      + 0.1 × max(s_parent, s_pad, s_crop)
```

## Deployment profile

| Property | Value |
|---|---|
| Base checkpoint | `Marqo/marqo-fashionSigLIP` |
| Neural parameters | 203,155,970 |
| Additional learned parameters | **0** |
| Embedding dimension | 768 |
| Stored vectors per product | 3 |
| Query vectors used for search | 1 |
| Retrieval routes | 3 |
| Raw FP16 vector storage | approximately 4.5 KiB/product |
| Online cross-encoder | none |

`ModaFashionSigLIP.search()` performs exact, chunked scoring so the example is
easy to reproduce. At production scale, search three ANN indexes, union their
candidates, and apply the identical fusion formula to those candidates.
`GalleryIndex.save_pretrained()` preserves FP32 vectors by default for the
most conservative reproduction; production indexes may store FP16 vectors to
reach the storage figure above.

## Evaluation disclosure

- Task: text-to-image retrieval.
- Primary metric: full-corpus MAP@10.
- Confidence intervals: paired bootstrap with 10,000 samples.
- One global recipe was used across all six datasets.
- The exact `0.10` late-fusion blend was selected on external OpenVTON and
  leakage-audited GLAMI development data, not on target examples, images, or
  qrels.
- Previous target aggregate results were already known by benchmark iteration
  six.
- This is **not** a fresh independent blind SOTA evaluation.
- Atlas and DeepFashion Multimodal are numerical improvements, not significant
  wins.

The frozen result summary and exact public recipe are included in this
repository under `benchmark_results/` and `config.json`.

For a cloned repository, `python example_retrieval.py ...` invokes the same
installed command-line implementation.

## Intended use

- Fashion catalog text-to-image search.
- Reproducing the deterministic architecture on a private image collection.
- Studying multi-view preprocessing and conservative score fusion.

## Limitations

- Three image encodings are needed when building a gallery.
- The index uses three vectors and three retrieval routes per product.
- The query path performs two text encodings.
- Results can differ if the upstream checkpoint, OpenCLIP preprocessing, image
  decoding, corpus construction, or metric implementation changes.
- Fashion retrieval benchmarks do not establish performance for people
  recognition, biometric use, or unrelated visual domains.

## Licenses

The wrapper and retrieval code in this repository are MIT licensed. The
referenced Marqo FashionSigLIP checkpoint is distributed by Marqo under
Apache-2.0. No upstream weights are duplicated here.

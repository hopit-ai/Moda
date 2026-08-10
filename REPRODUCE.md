# Reproducing the MODA benchmarks

Every number on the [benchmark page](https://hopit-ai.github.io/Moda/) is
produced by the commands below. If you get a materially different figure,
please [open an issue](https://github.com/hopit-ai/Moda/issues) with your
environment and the command you ran — a number we cannot reproduce is a bug in
our reporting, not in your setup.

## What you need

| | |
|---|---|
| **Hardware** | One GPU with ≥16 GB (we used A10G). CPU works but the 202K-image Fashion200K gallery will take hours rather than minutes. |
| **Disk** | ~60 GB for the benchmark galleries and cached embeddings. |
| **Time** | ~25 min for the six text-to-image sets at 224px; ~2 h for a 384px model. LookBench adds ~15 min. |
| **Cost** | About $3–4 of on-demand A10G time for a full six-benchmark sweep. |
| **Accounts** | A Hugging Face account for the gated DeepFashion and LookBench datasets. |

## Setup

```bash
git clone https://github.com/hopit-ai/Moda.git && cd Moda
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login          # needed for gated benchmark datasets
```

## Text-to-image: the six academic benchmarks

**Full corpus, no subsampling.** `--max-corpus 0` is the rule from §3.1 of our
protocol: every query is scored against the complete gallery.

```bash
python3 -m benchmark.phase0_backbone_bench \
  --models "Marqo/marqo-fashionSigLIP" \
  --datasets "KAGL,polyvore,atlas,fashion200k,deepfashion_inshop,deepfashion_multimodal" \
  --max-corpus 0 \
  --out results/repro
```

This writes one JSON per model with MAP@10 per dataset. Expect
`Marqo/marqo-fashionSigLIP` to land within ±0.001 of:

| Dataset | Corpus | MAP@10 |
|---|---:|---:|
| KAGL | 44,434 | 0.2769 |
| Polyvore | 94,096 | 0.3665 |
| Atlas | 78,370 | 0.1826 |
| Fashion200K | 201,624 | 0.1858 |
| DeepFashion In-Shop | 52,591 | 0.1587 |
| DeepFashion Multimodal | 42,537 | 0.0148 |

**Reproduce a competitor before comparing to it** (§3.3). This is the step most
comparisons skip:

```bash
python3 -m benchmark.phase0_backbone_bench \
  --models "openclip:ViT-SO400M-14-SigLIP-384/webli,srpone/zooclaw-fashionsiglip2" \
  --datasets "KAGL,polyvore,atlas,fashion200k,deepfashion_inshop,deepfashion_multimodal" \
  --max-corpus 0 --out results/repro
```

## MODA: the zero-parameter recipe

`--fusion` enables the serving recipe: three image views per product
(official + aspect-pad + centre-crop, fused at α=0.25) and two query encodings
(raw + `"a fashion product photo of {query}"`). No weights change; the base
checkpoint is downloaded unmodified.

```bash
python3 -m benchmark.phase0_backbone_bench \
  --models "Marqo/marqo-fashionSigLIP" \
  --datasets "KAGL,polyvore,atlas,fashion200k,deepfashion_inshop,deepfashion_multimodal" \
  --max-corpus 0 --fusion \
  --save-emb results/repro/emb \
  --out results/repro
```

`--save-emb` retains the per-query and per-gallery vectors. Keep them: they are
what makes the significance testing below free.

## MODA Pro Lite

The open-weights trained encoder, run through the identical harness:

```bash
python3 -m benchmark.phase0_backbone_bench \
  --models "hf-hub:HopitAI/moda-pro-lite" \
  --datasets "KAGL,polyvore,atlas,fashion200k,deepfashion_inshop,deepfashion_multimodal" \
  --max-corpus 0 --fusion \
  --save-emb results/repro/emb \
  --out results/repro
```

## Significance testing

Paired bootstrap over the query axis, 10,000 resamples, using the embeddings
you just cached. CPU only, no GPU, seconds per dataset:

```bash
python3 -m benchmark.bootstrap_ci --emb-dir results/repro/emb
```

The pairing matters: the same resampled query indices are applied to both
systems, so query difficulty cancels and the interval is on the difference
itself. This is how we determined that MODA Pro Lite's advantage on Atlas is
**not** significant while its KAGL and Polyvore gains are.

## Image-to-image: LookBench

```bash
python3 -m benchmark.eval_lookbench_baseline \
  --models "hf-hub:HopitAI/moda-fashion-distilled,Marqo/marqo-fashionSigLIP" \
  --out results/repro/lookbench
```

Reported as Fine Recall@1 and nDCG@5, weighted across the four LookBench
subsets. Our FashionSigLIP re-run lands at 63.84 against a published 62.77; we
report both figures rather than picking one.

## What you cannot reproduce here

Two systems in the comparison are products, and we say so plainly:

- **MODA Pro** is a closed hosted system. Its results are checkable through the
  API, but its architecture and training are not published.
- **MODA Pro Lite's weights** are open and its evaluation is fully reproducible
  with the command above — but the **training pipeline and training data are
  proprietary**, so you cannot retrain it from scratch.

Everything else on the benchmark page — the protocol, the MODA recipe, the
distilled image model, and every competitor comparison — is reproducible from
this repository.

## Known sources of variance

| Source | Effect |
|---|---|
| Gallery alignment across separate runs | ≤1% on MAP@10. Per-model containers occasionally drop different images on network fetch; we intersect IDs before comparing. |
| fp16 vs fp32 accumulation | 4th decimal place. |
| Dataset revision drift | Benchmark hosts occasionally re-upload. Pin the dataset revision if you need bit-exactness. |
| Corpus size | **Not** variance — a real effect. Scores computed on a subsampled gallery are not comparable to full-corpus scores, in either direction. Never mix them in one table. |

## Getting help

Open an issue, or [book a call](https://calendly.com/arkid_/new-meeting?back=1)
if you want help evaluating on your own catalog.

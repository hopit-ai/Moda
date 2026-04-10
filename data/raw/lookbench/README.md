# LookBench (Tier 3)

Download evaluation code and dataset snapshots into this folder (large; not committed).

```bash
# From repo root (requires: datasets, huggingface_hub, git)
python scripts/download_lookbench.py
```

Produces:

- `datasets/<config>/` — Hugging Face `save_to_disk` trees for each subset (`real_studio_flat`, `aigen_studio`, `real_streetlook`, `aigen_streetlook`, `noise`).
- `look-bench/` — shallow clone of [SerendipityOneInc/look-bench](https://github.com/SerendipityOneInc/look-bench) (official metrics and runners).

After download, install their package from the clone (`pip install -e .`) and run evaluations per their `README.md`.

A machine-readable summary is written to `results/real/lookbench_download_manifest.json` after a successful run.

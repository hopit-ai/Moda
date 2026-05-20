"""
v2 — combine original 50K labeled pairs + synthetic prose-query pairs into a
training set, and augment the image index so synth pair_ids resolve to the
same image embedding rows as their originals.

Inputs:
  - pairs_50k_labeled.jsonl  (50K original labeled pairs)
  - pairs_v2_long_desc.jsonl (synthetic prose queries; pair_id = orig + '__synth_long')
  - student_image_index.json (50K original pair_id → row mapping)

Outputs:
  - pairs_v2_combined.jsonl  (original + synth, single training set)
  - v2_image_index.json      (original entries + synth entries pointing to same rows)
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "processed" / "v5_multifield"


def main():
    base_pairs = DATA / "pairs_50k_labeled.jsonl"
    synth_pairs = DATA / "pairs_v2_long_desc.jsonl"
    base_index = DATA / "student_image_index.json"

    out_pairs = DATA / "pairs_v2_combined.jsonl"
    out_index = DATA / "v2_image_index.json"

    if not synth_pairs.exists():
        raise SystemExit(f"missing {synth_pairs} — synth gen not done?")
    if not base_pairs.exists():
        # Fall back to unlabeled pairs if labeled missing
        base_pairs = DATA / "pairs_50k.jsonl"
    if not base_index.exists():
        raise SystemExit(f"missing {base_index}")

    image_index: dict[str, int] = json.loads(base_index.read_text())
    print(f"Base image_index: {len(image_index):,} entries")

    # Stream-merge: original first, then synth
    n_base = n_synth = n_skipped = 0
    with out_pairs.open("w") as out:
        with base_pairs.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    out.write(line + "\n")
                    n_base += 1
        with synth_pairs.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                pid = r["pair_id"]
                # synth pair_id should be `<orig>__synth_long`
                if "__synth_long" not in pid:
                    n_skipped += 1
                    continue
                orig_pid = pid.replace("__synth_long", "")
                if orig_pid not in image_index:
                    n_skipped += 1
                    continue
                # Map synth pair_id to same image row
                image_index[pid] = image_index[orig_pid]
                out.write(line + "\n")
                n_synth += 1

    out_index.write_text(json.dumps(image_index))
    print(f"Wrote {out_pairs}: {n_base:,} base + {n_synth:,} synth = {n_base + n_synth:,} total")
    print(f"Skipped {n_skipped} synth records (missing orig_pid in index)")
    print(f"Wrote augmented {out_index}: {len(image_index):,} entries")


if __name__ == "__main__":
    main()

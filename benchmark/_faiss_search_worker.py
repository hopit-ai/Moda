"""
Subprocess worker: loads FAISS index + numpy query embeddings, runs batch search,
saves results to disk. No PyTorch imported — avoids BLAS conflict.

Called by run_hnm_eval.py. Not meant to be used directly.

Args (positional):
  query_embeddings_path  .npy file with query embeddings
  faiss_index_path       path to FAISS index
  article_ids_path       JSON list of article ids
  results_path           output .json (list of lists of article_ids)
  top_k                  integer
"""

import json
import sys

import faiss
import numpy as np


def main():
    if len(sys.argv) != 6:
        print("Usage: python _faiss_search_worker.py <q_emb.npy> <index> <ids.json> <out.json> <top_k>")
        sys.exit(1)

    q_emb_path, index_path, ids_path, out_path, top_k_str = sys.argv[1:]
    top_k = int(top_k_str)

    q_emb = np.load(q_emb_path).astype(np.float32)
    index = faiss.read_index(index_path)
    with open(ids_path) as f:
        article_ids = json.load(f)

    print(f"Searching {len(q_emb)} queries against {index.ntotal} articles …", flush=True)

    batch_size = 256
    all_results = []
    for start in range(0, len(q_emb), batch_size):
        batch = q_emb[start: start + batch_size]
        _, indices = index.search(batch, top_k)
        for row in indices:
            all_results.append([article_ids[i] for i in row if i >= 0])

    with open(out_path, "w") as f:
        json.dump(all_results, f)

    print(f"Saved {len(all_results)} result lists to {out_path}", flush=True)


if __name__ == "__main__":
    main()

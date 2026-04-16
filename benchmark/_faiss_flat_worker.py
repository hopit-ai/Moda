"""
Subprocess worker: builds a flat inner-product FAISS index from numpy arrays
and runs batch search.  No PyTorch imported — avoids MPS/BLAS conflicts.

Args (positional):
  query_emb_path   .npy   query embeddings
  article_emb_path .npy   article embeddings
  ids_path         .json  article id list
  out_path         .json  output (list of lists)
  top_k            int
  query_chunk      int    (optional, default 512)
"""

import json
import sys

import faiss
import numpy as np


def main():
    if len(sys.argv) < 6:
        print("Usage: _faiss_flat_worker.py <q.npy> <art.npy> <ids.json> <out.json> <top_k> [chunk]")
        sys.exit(1)

    q_path, art_path, ids_path, out_path = sys.argv[1:5]
    top_k = int(sys.argv[5])
    chunk = int(sys.argv[6]) if len(sys.argv) > 6 else 512

    q_emb = np.load(q_path).astype(np.float32)
    art_emb = np.load(art_path).astype(np.float32)
    with open(ids_path) as f:
        article_ids = json.load(f)

    index = faiss.IndexFlatIP(art_emb.shape[1])
    index.add(art_emb)
    print(f"FAISS: {q_emb.shape[0]} queries x {index.ntotal} articles, top_k={top_k}", flush=True)

    results = []
    for start in range(0, q_emb.shape[0], chunk):
        batch = q_emb[start : start + chunk]
        _, I = index.search(batch, top_k)
        for row in I:
            results.append([article_ids[i] for i in row if i >= 0])

    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"Done — {len(results)} result lists", flush=True)


if __name__ == "__main__":
    main()

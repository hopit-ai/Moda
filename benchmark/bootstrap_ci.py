"""Paired bootstrap confidence intervals over cached full-corpus embeddings.

Per-query AP@10 is computed once per system, then resampled with replacement
10,000 times over the query axis. Pairing is essential: the same resampled
query indices are applied to both systems, so the CI is on the *difference*
and query difficulty cancels out.

CPU only, no GPU, no re-encoding. Usage:
    python3 -m benchmark.bootstrap_ci
"""
import glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "repos/marqo-FashionCLIP")
from benchmark.fuse_and_eval import reconstruct_corpus_full
from benchmark.phase1_surrogate_check import zrow

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--emb-dir", default="results/repro/emb")
CELL = _ap.parse_known_args()[0].emb_dir
LITE = "ft__cache_results_autoresearch_zcr2s_wise040.pt"
B = 10_000
SEED = 0
K = 10


def ap_at_k(S, order, queries, gt, k=K):
    """Per-query average precision@k. Returns a vector of length n_queries."""
    idx = {o: i for i, o in enumerate(order)}
    out = np.zeros(len(queries), dtype=np.float64)
    top = np.argpartition(-S, k, axis=1)[:, :k]
    for r, q in enumerate(queries):
        rel = gt.get(q, set())
        if not rel:
            continue
        cand = top[r][np.argsort(-S[r, top[r]])]
        hits, prec = 0, 0.0
        for rank, c in enumerate(cand, start=1):
            if order[c] in rel:
                hits += 1
                prec += hits / rank
        out[r] = prec / min(len(rel), k)
    return out


def rerank_scores(Sm, Sz, k=100):
    """MODA Pro: MODA retrieves top-k, both experts' z-scored sims are summed."""
    Zm, Zz = zrow(Sm), zrow(Sz)
    F = np.full_like(Sm, -1e9)
    topk = np.argpartition(-Sm, k, axis=1)[:, :k]
    rows = np.arange(Sm.shape[0])[:, None]
    F[rows, topk] = Zm[rows, topk] + Zz[rows, topk]
    return F


def paired_bootstrap(a, b, n=B, seed=SEED):
    """CI on mean(a) - mean(b), resampling query indices jointly."""
    rng = np.random.default_rng(seed)
    n_q = len(a)
    idx = rng.integers(0, n_q, size=(n, n_q))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # two-sided p on the sign of the difference
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return float(lo), float(hi), float(p)


def main():
    datasets = sorted(
        os.path.basename(f).split("__", 1)[1][:-4]
        for f in glob.glob(f"{CELL}/fsl__*.npz")
    )
    results = {}
    for ds in datasets:
        zm = np.load(f"{CELL}/fsl__{ds}.npz", allow_pickle=True)
        zz = np.load(f"{CELL}/{LITE}__{ds}.npz", allow_pickle=True)
        _, queries_gt, gt = reconstruct_corpus_full(ds)
        order = [str(x) for x in zm["order"]]
        qs = [str(x) for x in zm["queries"]]
        # gallery rows can differ in order between containers; align by id
        pos = {o: i for i, o in enumerate(str(x) for x in zz["order"])}
        sel = np.array([pos[o] for o in order])

        Sm = zm["q"].astype(np.float32) @ zm["gal"].astype(np.float32).T
        Sz = zz["q"].astype(np.float32) @ zz["gal"].astype(np.float32)[sel].T
        Sp = rerank_scores(Sm, Sz)

        ap = {
            "MODA": ap_at_k(Sm, order, qs, gt),
            "MODA Pro Lite": ap_at_k(Sz, order, qs, gt),
            "MODA Pro": ap_at_k(Sp, order, qs, gt),
        }
        row = {k: float(v.mean()) for k, v in ap.items()}
        row["n_queries"] = len(qs)
        row["ci"] = {}
        for x, y in [("MODA Pro", "MODA"), ("MODA Pro Lite", "MODA"), ("MODA Pro", "MODA Pro Lite")]:
            lo, hi, p = paired_bootstrap(ap[x], ap[y])
            row["ci"][f"{x} - {y}"] = {
                "delta": float(ap[x].mean() - ap[y].mean()),
                "ci95": [lo, hi], "p": p,
                "significant": bool(lo > 0 or hi < 0),
            }
        results[ds] = row
        print(f"{ds:<24} n={len(qs):>5}  MODA={row['MODA']:.5f}  "
              f"Lite={row['MODA Pro Lite']:.5f}  Pro={row['MODA Pro']:.5f}")

    out = "results/autoresearch/bootstrap_ci_full_corpus.json"
    json.dump(results, open(out, "w"), indent=1)
    print(f"\nwrote {out}")

    print(f"\n{'dataset':<24} {'comparison':<28} {'Δ abs':>9} {'95% CI':>22} {'verdict':>14}")
    for ds, row in results.items():
        for comp, c in row["ci"].items():
            v = "significant" if c["significant"] else "inconclusive"
            print(f"{ds:<24} {comp:<28} {c['delta']:>+9.5f} "
                  f"[{c['ci95'][0]:>+.5f}, {c['ci95'][1]:>+.5f}] {v:>14}")


if __name__ == "__main__":
    main()

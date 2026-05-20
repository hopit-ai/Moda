"""
Path 2 — Research-backed distillation trainer.

Differences vs Path 1 (`distill_fusion_to_student.py`):

  Path 1                                   |  Path 2
  -----------------------------------------|---------------------------------------------
  KL on row-softmax of [B,B] in-batch      |  MarginMSE + listwise KL on (1 pos + K=15
   teacher/student matrices                |   mined hard negatives) per query
  In-batch random negatives                |  K=15 hard negatives per query, mined offline
  No anchor                                |  λ·anchor loss to frozen-init student
  Single-dataset probe (f200k)             |  Multi-dataset probe (f200k + atlas + polyvore)
  Best-by f200k MAP@10                     |  Best-by mean MAP@10 across 3 probe datasets
  No R@100 tracking                        |  Probe also reports Recall@100 per dataset

See PATH2_PLAN.md and DIAGNOSTIC_PATH1.md for the rationale behind every change.

Loss:
    L = MarginMSE  +  0.5 · KL_listwise
                   +  0.1 · symmetric_InfoNCE
                   +  λ   · anchor_loss
where:
    MarginMSE = mean_{i,k} ((s_t(q_i,p+) - s_t(q_i,p_k)) - (s_s(q_i,p+) - s_s(q_i,p_k)))^2
    KL_listwise = mean_i KL(softmax(s_t/τ_t) || softmax(s_s/τ_s)) over the (K+1)-list
    anchor = 0.5 (1 - cos(img_init(p+), img_student(p+)))
           + 0.5 (1 - cos(txt_init(q),  txt_student(q)))
    λ = 0.05  (post-diagnostic; see DIAGNOSTIC_PATH1.md)

Data:
    data/processed/path2/hardnegs.jsonl  — 421 queries (39 zero-coverage skipped at load)
    data/processed/distillation_cache_fusion/{fashion_siglip, siglip2_b16_384}_embeddings.pt
        — L2-normed teacher embeddings for the 5K pool

Each training "row" is one query with:
    - 1 positive image (the strongest gold by score_linear), OR optionally
      multiple positives (we use the strongest one for MarginMSE and KL).
    - K=15 hard negative images (already mined).

We compute teacher scores from cached embeddings (no teacher forward passes
in the training loop). The student forwards 1 + K = 16 images per query.

Probe:
    benchmark/probe_fashion200k_10k.Fashion200kProbe — re-used for f200k,
    atlas, polyvore. Each probe call ~50s after corpus tensor warmup.

Usage:
    # Step-50 smoke (no probe abort, just verify it runs)
    .venv/bin/python benchmark/distill_path2.py \\
        --max-steps 50 --probe-steps "" --batch-size 8 \\
        --output-dir models/path2-smoke50

    # Step-200 smoke (probe at step 100 + 200, check trajectory)
    .venv/bin/python benchmark/distill_path2.py \\
        --max-steps 200 --probe-steps "100,200" --batch-size 16 \\
        --output-dir models/path2-smoke200

    # Full overnight run
    .venv/bin/python benchmark/distill_path2.py \\
        --max-steps 2000 --probe-steps "200,500,1000,1500,2000" \\
        --batch-size 16 \\
        --output-dir models/path2-full

Outputs:
    <out>/run_meta.json
    <out>/training_log.jsonl   — per-step loss components + per-probe metrics
    <out>/step_<N>/student_state_dict.pt
    <out>/best/student_state_dict.pt   — by mean MAP@10 over the 3 probe datasets
    <out>/summary.json
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmark"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("distill-p2")

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
HF_CACHE = REPO / "data" / "hf_cache"

DEFAULT_HARDNEGS = REPO / "data/processed/path2/hardnegs.jsonl"
DEFAULT_TEACHER_CACHE = REPO / "data/processed/distillation_cache_fusion"


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

def load_hardnegs(path: Path, K: int) -> list[dict]:
    """Load hardnegs.jsonl, drop queries with <K negatives, normalise to a
    single (q, p+, [p_1..p_K]) record per query.

    Picks p+ = the positive with the highest score_linear.
    """
    queries: list[dict] = []
    n_total = 0
    n_kept = 0
    with open(path) as f:
        for line in f:
            n_total += 1
            rec = json.loads(line)
            if not rec.get("positives") or len(rec.get("hard_negatives", [])) < K:
                continue
            best_pos = max(rec["positives"], key=lambda p: p["score_linear"])
            queries.append({
                "query": rec["query"],
                "pos_image_path": best_pos["image_path"],
                "pos_score_linear": best_pos["score_linear"],
                "neg_image_paths": [n["image_path"] for n in rec["hard_negatives"][:K]],
            })
            n_kept += 1
    log.info("[data] loaded %d queries, kept %d with K>=%d hard-negs (dropped %d)",
             n_total, n_kept, K, n_total - n_kept)
    return queries


def build_path_to_emb_idx(teacher_cache: dict) -> dict[str, int]:
    """Map image_path -> first row index in teacher cache. The cache rows mirror
    the original triplets file order; many rows share the same image_path.
    """
    out: dict[str, int] = {}
    for i, p in enumerate(teacher_cache["image_paths"]):
        if p not in out:
            out[p] = i
    return out


def build_query_to_emb_idx(teacher_cache: dict) -> dict[str, int]:
    out: dict[str, int] = {}
    for i, q in enumerate(teacher_cache["queries"]):
        if q not in out:
            out[q] = i
    return out


def load_teacher_caches(cache_dir: Path) -> dict:
    fsl = torch.load(cache_dir / "fashion_siglip_embeddings.pt",
                     map_location="cpu", weights_only=False)
    sl2 = torch.load(cache_dir / "siglip2_b16_384_embeddings.pt",
                     map_location="cpu", weights_only=False)
    assert fsl["queries"] == sl2["queries"], "FSL/SL2 caches have mismatched query order"
    assert fsl["image_paths"] == sl2["image_paths"], "FSL/SL2 caches have mismatched image order"
    log.info("[teacher] loaded FSL+SL2 caches, N=%d, dim=%d", len(fsl["queries"]), fsl["embed_dim"])

    img_path_to_idx = build_path_to_emb_idx(fsl)
    query_to_idx = build_query_to_emb_idx(fsl)
    log.info("[teacher] %d unique images, %d unique queries", len(img_path_to_idx), len(query_to_idx))
    return {
        "fsl": fsl, "sl2": sl2,
        "img_path_to_idx": img_path_to_idx,
        "query_to_idx": query_to_idx,
    }


def teacher_scores_for_batch(
    teachers: dict, batch: list[dict],
) -> torch.Tensor:
    """For B queries each with 1 pos + K negs, compute the fused teacher score
    s_t(q, p) for all (K+1) docs. Returns [B, K+1] on CPU.

    s_t = 0.5 * (cos_FSL(text, image) + cos_SL2(text, image))
    Both teachers' embeddings are L2-normed in cache, so cos = dot.
    """
    fsl_text = teachers["fsl"]["text"]
    fsl_img = teachers["fsl"]["image"]
    sl2_text = teachers["sl2"]["text"]
    sl2_img = teachers["sl2"]["image"]
    q2i = teachers["query_to_idx"]
    p2i = teachers["img_path_to_idx"]

    out = []
    for r in batch:
        qi = q2i[r["query"]]
        pos_i = p2i[r["pos_image_path"]]
        neg_is = [p2i[p] for p in r["neg_image_paths"]]
        doc_is = torch.tensor([pos_i] + neg_is, dtype=torch.long)

        s_fsl = (fsl_text[qi:qi + 1] @ fsl_img.index_select(0, doc_is).T).squeeze(0)
        s_sl2 = (sl2_text[qi:qi + 1] @ sl2_img.index_select(0, doc_is).T).squeeze(0)
        out.append(0.5 * (s_fsl + s_sl2))
    return torch.stack(out, dim=0)  # [B, K+1]


# -----------------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------------

def margin_mse(s_t: torch.Tensor, s_s: torch.Tensor) -> torch.Tensor:
    """Hofstätter 2020. Per (q, p+, p_k) triplet, match (s_t(q,p+)-s_t(q,p_k))
    to (s_s(q,p+)-s_s(q,p_k)).

    s_t, s_s: [B, K+1]. Position 0 is the positive.
    Output: scalar.
    """
    pos_t = s_t[:, 0:1]
    neg_t = s_t[:, 1:]
    pos_s = s_s[:, 0:1]
    neg_s = s_s[:, 1:]
    margin_t = pos_t - neg_t      # [B, K]
    margin_s = pos_s - neg_s
    return F.mse_loss(margin_s, margin_t)


def listwise_kl(s_t: torch.Tensor, s_s: torch.Tensor, tau_t: float, tau_s: float) -> torch.Tensor:
    """KL(softmax(s_t/τ_t) || softmax(s_s/τ_s)) over the (K+1)-list per query.

    Tamber 2025: τ_t=0.3 (smoother teacher), τ_s=0.05 (sharper student).
    """
    log_p = F.log_softmax(s_t / tau_t, dim=-1)
    log_q = F.log_softmax(s_s / tau_s, dim=-1)
    p = log_p.exp()
    return (p * (log_p - log_q)).sum(dim=-1).mean()


def symmetric_infonce_listwise(s_s: torch.Tensor, tau: float) -> torch.Tensor:
    """InfoNCE over the (K+1)-list per query with positive at index 0.

    This is NOT the in-batch InfoNCE; it's listwise within each query's
    (1 pos + K hard negs) candidate set. Position 0 is the positive.
    """
    labels = torch.zeros(s_s.shape[0], dtype=torch.long, device=s_s.device)
    return F.cross_entropy(s_s / tau, labels)


def anchor_loss(
    img_student: torch.Tensor, img_init: torch.Tensor,
    txt_student: torch.Tensor, txt_init: torch.Tensor,
) -> torch.Tensor:
    """1 - cos(student, init) averaged over batch, both modalities.

    All four tensors are L2-normed and shape [B, D]. Returns scalar in [0, 2].
    """
    img_anchor = 1.0 - (img_student * img_init).sum(dim=-1)
    txt_anchor = 1.0 - (txt_student * txt_init).sum(dim=-1)
    return 0.5 * img_anchor.mean() + 0.5 * txt_anchor.mean()


# -----------------------------------------------------------------------------
# Student forward (1 query, K+1 images)
# -----------------------------------------------------------------------------

def student_forward(
    student, preprocess, tokenizer,
    batch: list[dict], device: str,
    pos_only_for_anchor: bool,
):
    """Forward B queries, each with (1 pos + K negs) images.

    Returns:
        s_s: [B, K+1] cosine score matrix (text dotted with each image)
        img_pos_emb: [B, D] L2-normed student image embedding for positives only
                     (used by anchor loss, computed for free since pos is in batch)
        txt_emb:     [B, D] L2-normed student text embedding (used by anchor loss)
    """
    from PIL import Image
    queries = [r["query"] for r in batch]
    K = len(batch[0]["neg_image_paths"])
    B = len(batch)

    # Build a single image batch of shape [B*(K+1), 3, H, W] to amortise the
    # student image-tower forward.
    img_paths_flat = []
    for r in batch:
        img_paths_flat.append(r["pos_image_path"])
        img_paths_flat.extend(r["neg_image_paths"])

    images = [preprocess(Image.open(p).convert("RGB")) for p in img_paths_flat]
    img_tens = torch.stack(images).to(device)        # [B*(K+1), 3, H, W]
    tokens = tokenizer(queries).to(device)           # [B, T]

    img_feat_flat = student.encode_image(img_tens)   # [B*(K+1), D]
    txt_feat = student.encode_text(tokens)           # [B, D]

    img_feat_flat = img_feat_flat / img_feat_flat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    img_feat = img_feat_flat.view(B, K + 1, -1)      # [B, K+1, D]
    s_s = (txt_feat.unsqueeze(1) * img_feat).sum(dim=-1)  # [B, K+1]

    img_pos_emb = img_feat[:, 0, :] if pos_only_for_anchor else img_feat.reshape(-1, img_feat.shape[-1])
    return s_s, img_pos_emb, txt_feat


class InitAnchorCache:
    """Read-only on-disk cache of init-student embeddings, looked up by
    image_path / query string. Avoids holding a 375M-param frozen-init copy
    of the student in memory (which OOMed on MPS via paging).
    """
    def __init__(self, cache_path: Path):
        d = torch.load(cache_path, map_location="cpu", weights_only=False)
        self.img_emb: torch.Tensor = d["img"]      # [N_img, D] L2-normed fp32
        self.txt_emb: torch.Tensor = d["txt"]      # [N_txt, D] L2-normed fp32
        self.img_to_idx: dict[str, int] = {p: i for i, p in enumerate(d["image_paths"])}
        self.txt_to_idx: dict[str, int] = {q: i for i, q in enumerate(d["queries"])}
        log.info("[anchor cache] %d imgs, %d queries, dim=%d", self.img_emb.shape[0],
                 self.txt_emb.shape[0], self.img_emb.shape[1])

    def lookup(self, batch: list[dict], device: str):
        img_idx = torch.tensor([self.img_to_idx[r["pos_image_path"]] for r in batch], dtype=torch.long)
        txt_idx = torch.tensor([self.txt_to_idx[r["query"]] for r in batch], dtype=torch.long)
        return (
            self.img_emb.index_select(0, img_idx).to(device),
            self.txt_emb.index_select(0, txt_idx).to(device),
        )


# -----------------------------------------------------------------------------
# Multi-dataset probe wrapper
# -----------------------------------------------------------------------------

class MultiProbe:
    """Wrap N Fashion200kProbe instances (one per dataset) and report the
    per-dataset metrics + mean MAP@10 + mean Recall@100.
    """
    def __init__(self, datasets: list[str], corpus_size: int = 10000):
        from probe_fashion200k_10k import Fashion200kProbe
        self.datasets = datasets
        self.probes: dict[str, Fashion200kProbe] = {
            d: Fashion200kProbe(dataset=d, corpus_size=corpus_size, seed=42)
            for d in datasets
        }

    def run(self, model, preprocess, tokenizer, device: str = DEVICE, batch_size: int = 64) -> dict:
        out: dict = {"per_dataset": {}}
        per_map10 = []
        per_r100 = []
        for d in self.datasets:
            log.info("[probe] running %s ...", d)
            m = self.probes[d].run(model, preprocess, tokenizer, device=device, batch_size=batch_size)
            out["per_dataset"][d] = {
                "MAP@10": float(m.get("MAP@10", 0)),
                "Recall@10": float(m.get("Recall@10", 0)),
                "Recall@100": float(m.get("Recall@100", 0)),
                "NDCG@10": float(m.get("NDCG@10", 0)),
            }
            per_map10.append(out["per_dataset"][d]["MAP@10"])
            per_r100.append(out["per_dataset"][d]["Recall@100"])
            # IMPORTANT: free the 8.8GB corpus tensor between datasets, otherwise
            # we hold 3*8.8 = 26GB of preprocessed corpora across the 3 probes.
            self.probes[d]._preprocessed_cache.clear()
            self.probes[d]._pil_images = None
            gc.collect()
            if torch.backends.mps.is_available():
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass
        out["mean_MAP@10"] = sum(per_map10) / len(per_map10)
        out["mean_Recall@100"] = sum(per_r100) / len(per_r100)
        return out


# -----------------------------------------------------------------------------
# Image tower freeze helpers (same as Path 1)
# -----------------------------------------------------------------------------

def freeze_image_tower(student) -> int:
    n = 0
    for name, p in student.named_parameters():
        if name.startswith("visual."):
            p.requires_grad = False
            n += p.numel()
    return n


def unfreeze_image_tower(student) -> int:
    n = 0
    for name, p in student.named_parameters():
        if name.startswith("visual."):
            p.requires_grad = True
            n += p.numel()
    return n


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hardnegs", default=str(DEFAULT_HARDNEGS))
    p.add_argument("--teacher-cache-dir", default=str(DEFAULT_TEACHER_CACHE))
    p.add_argument("--anchor-cache",
                   default=str(REPO / "data/processed/path2/init_anchor_cache.pt"),
                   help="Pre-cached frozen-init student embeddings for the anchor loss.")
    p.add_argument("--student-model", default="ViT-B-16-SigLIP2-384")
    p.add_argument("--student-pretrained", default="webli")
    p.add_argument("--output-dir", required=True)

    p.add_argument("-K", "--K", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=16,
                   help="Number of queries per step. Each query forwards (1+K) images.")
    p.add_argument("--grad-accum", type=int, default=1)

    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--unfreeze-image-step", type=int, default=200,
                   help="Per PATH2_PLAN: 200 (was 500 in Path 1).")
    p.add_argument("--grad-checkpoint", action="store_true", default=False,
                   help="Enable gradient checkpointing on both towers — recompute activations in backward, "
                        "much smaller memory at ~30%% wall-time cost. REQUIRED for image-tower training on MPS.")

    # Loss weights — defaults from PATH2_PLAN.md (post-diagnostic):
    # L = MM + 0.5·KL + 0.1·InfoNCE + 0.05·anchor
    p.add_argument("--w-mm", type=float, default=1.0)
    p.add_argument("--w-kl", type=float, default=0.5)
    p.add_argument("--w-infonce", type=float, default=0.1)
    p.add_argument("--w-anchor", type=float, default=0.05)
    p.add_argument("--tau-teacher", type=float, default=0.3)
    p.add_argument("--tau-student", type=float, default=0.05)
    p.add_argument("--tau-infonce", type=float, default=0.05)

    p.add_argument("--probe-steps", default="200,500,1000,1500,2000")
    p.add_argument("--probe-datasets", default="fashion200k,atlas,polyvore")
    p.add_argument("--probe-corpus-size", type=int, default=10000,
                   help="Per-dataset corpus size for in-loop probes. 5000 halves probe wall-time.")
    p.add_argument("--probe-batch-size", type=int, default=64)
    # Abort criteria from PATH2_PLAN §6
    p.add_argument("--abort-step1", type=int, default=200)
    p.add_argument("--abort-step1-min-delta", type=float, default=-0.02,
                   help="At step1, abort if any dataset's MAP@10 - init_MAP@10 < this.")
    p.add_argument("--abort-step2", type=int, default=500)
    p.add_argument("--abort-step2-min-delta", type=float, default=0.0,
                   help="At step2, abort if any dataset's MAP@10 - init_MAP@10 < this.")
    p.add_argument("--abort-step2-min-r100-delta", type=float, default=-0.03,
                   help="At step2, also abort if any dataset's Recall@100 - init_Recall@100 < this.")
    p.add_argument("--abort-step2-min-mean-delta", type=float, default=-0.025,
                   help="At step2, abort if mean MAP@10 - init_mean_MAP@10 < this. Catches "
                        "uniform-mild-regression failure mode invisible to per-dataset thresholds.")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=10)

    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "training_log.jsonl"
    meta_path = out_dir / "run_meta.json"
    meta_path.write_text(json.dumps(vars(args), indent=2, default=str))

    log.info("=== Path 2 distillation run ===")
    log.info("output_dir=%s", out_dir)
    log.info("device=%s", DEVICE)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if DEVICE == "mps":
        torch.mps.manual_seed(args.seed)

    # 1. Data
    queries_data = load_hardnegs(Path(args.hardnegs), K=args.K)
    if not queries_data:
        raise RuntimeError("no usable training queries (all dropped). Check hardnegs file.")
    teachers = load_teacher_caches(Path(args.teacher_cache_dir))

    # Sanity: every query in queries_data must be in teacher cache
    missing_q = [q["query"] for q in queries_data if q["query"] not in teachers["query_to_idx"]]
    missing_p = []
    for q in queries_data:
        if q["pos_image_path"] not in teachers["img_path_to_idx"]:
            missing_p.append(q["pos_image_path"])
        for np_ in q["neg_image_paths"]:
            if np_ not in teachers["img_path_to_idx"]:
                missing_p.append(np_)
    if missing_q or missing_p:
        log.error("missing queries: %d  missing image paths: %d", len(missing_q), len(missing_p))
        if missing_q:
            log.error("first missing q: %r", missing_q[:3])
        if missing_p:
            log.error("first missing img: %r", missing_p[:3])
        raise RuntimeError("hardnegs references rows not present in the teacher cache")

    # 2. Student
    import open_clip
    log.info("[student] loading %s pretrained=%s", args.student_model, args.student_pretrained)
    student, _, preprocess = open_clip.create_model_and_transforms(
        args.student_model, pretrained=args.student_pretrained, cache_dir=str(HF_CACHE),
    )
    tokenizer = open_clip.get_tokenizer(args.student_model)
    student.to(DEVICE)
    student.train()

    if args.grad_checkpoint:
        log.info("[student] enabling gradient checkpointing on both towers")
        student.set_grad_checkpointing(True)

    log.info("[anchor] loading init-anchor cache from %s", args.anchor_cache)
    anchor_cache = InitAnchorCache(Path(args.anchor_cache))

    n_frozen = freeze_image_tower(student)
    n_total = sum(p.numel() for p in student.parameters())
    n_trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    log.info("[student] params: total=%d  frozen(image)=%d  trainable=%d",
             n_total, n_frozen, n_trainable)

    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95),
    )

    def lr_at_step(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return args.lr * 0.1 + (args.lr - args.lr * 0.1) * 0.5 * (1 + math.cos(math.pi * progress))

    # 3. Probe (lazy init; one warm-up call records init metrics for abort baselines)
    probe_steps: set[int] = set()
    if args.probe_steps.strip():
        probe_steps = {int(s.strip()) for s in args.probe_steps.split(",") if s.strip()}
    log.info("[probe] will probe at steps: %s", sorted(probe_steps))

    probe_datasets = [d.strip() for d in args.probe_datasets.split(",") if d.strip()]
    multi_probe = MultiProbe(probe_datasets, corpus_size=args.probe_corpus_size) if probe_steps else None

    init_metrics = None
    if multi_probe is not None:
        log.info("[probe] running INIT baseline (frozen student) ...")
        init_metrics = multi_probe.run(student, preprocess, tokenizer,
                                       device=DEVICE, batch_size=args.probe_batch_size)
        student.train()
        for d, m in init_metrics["per_dataset"].items():
            log.info("[probe-init] %s  MAP@10=%.4f  R@100=%.4f", d, m["MAP@10"], m["Recall@100"])
        log.info("[probe-init] mean MAP@10=%.4f  mean R@100=%.4f",
                 init_metrics["mean_MAP@10"], init_metrics["mean_Recall@100"])

    # 4. Training loop
    micro_bs = max(1, args.batch_size // args.grad_accum)
    log.info("[train] batch_size=%d  micro_bs=%d  grad_accum=%d  K=%d  (images per step = %d)",
             args.batch_size, micro_bs, args.grad_accum, args.K, args.batch_size * (1 + args.K))

    indices = list(range(len(queries_data)))
    random.shuffle(indices)
    cursor = 0

    def next_micro_batch() -> list[dict]:
        nonlocal cursor
        if cursor + micro_bs > len(indices):
            random.shuffle(indices)
            cursor = 0
        batch_idx = indices[cursor:cursor + micro_bs]
        cursor += micro_bs
        return [queries_data[i] for i in batch_idx]

    log_lines: list[dict] = []
    best_score = -1.0
    best_step = -1
    aborted = False

    t_start = time.time()
    for step in range(args.max_steps + 1):  # +1 so step==max_steps also probes
        cur_lr = lr_at_step(step)
        for g in optimizer.param_groups:
            g["lr"] = cur_lr

        # Unfreeze image tower
        if step == args.unfreeze_image_step and step > 0:
            n_un = unfreeze_image_tower(student)
            log.info("[train] step=%d UNFREEZING image tower (%d params)", step, n_un)
            optimizer = torch.optim.AdamW(
                [p for p in student.parameters() if p.requires_grad],
                lr=cur_lr, weight_decay=0.0, betas=(0.9, 0.95),
            )

        if step < args.max_steps:
            optimizer.zero_grad(set_to_none=True)
            mm_sum = kl_sum = nce_sum = anc_sum = 0.0

            for accum in range(args.grad_accum):
                batch = next_micro_batch()

                with torch.no_grad():
                    s_t = teacher_scores_for_batch(teachers, batch).to(DEVICE)

                s_s, img_pos_s, txt_s = student_forward(
                    student, preprocess, tokenizer, batch, DEVICE, pos_only_for_anchor=True,
                )
                img_pos_init, txt_init = anchor_cache.lookup(batch, DEVICE)

                loss_mm  = margin_mse(s_t, s_s)
                loss_kl  = listwise_kl(s_t, s_s, tau_t=args.tau_teacher, tau_s=args.tau_student)
                loss_nce = symmetric_infonce_listwise(s_s, tau=args.tau_infonce)
                loss_anc = anchor_loss(img_pos_s, img_pos_init, txt_s, txt_init)

                loss = (args.w_mm * loss_mm
                        + args.w_kl * loss_kl
                        + args.w_infonce * loss_nce
                        + args.w_anchor * loss_anc) / args.grad_accum
                loss.backward()

                mm_sum  += loss_mm.item()
                kl_sum  += loss_kl.item()
                nce_sum += loss_nce.item()
                anc_sum += loss_anc.item()

            torch.nn.utils.clip_grad_norm_(
                [p for p in student.parameters() if p.requires_grad], max_norm=1.0,
            )
            optimizer.step()

            mm_avg = mm_sum / args.grad_accum
            kl_avg = kl_sum / args.grad_accum
            nce_avg = nce_sum / args.grad_accum
            anc_avg = anc_sum / args.grad_accum
            total = (args.w_mm * mm_avg + args.w_kl * kl_avg
                     + args.w_infonce * nce_avg + args.w_anchor * anc_avg)

            if step % args.log_every == 0:
                elapsed = time.time() - t_start
                log.info("[train] step=%d/%d  lr=%.2e  loss=%.4f "
                         "(mm=%.4f, kl=%.4f, nce=%.4f, anc=%.4f)  elapsed=%.1fs",
                         step, args.max_steps, cur_lr, total,
                         mm_avg, kl_avg, nce_avg, anc_avg, elapsed)
            log_lines.append({
                "step": step, "lr": cur_lr,
                "mm": mm_avg, "kl": kl_avg, "nce": nce_avg, "anc": anc_avg,
                "total": total, "wall": time.time() - t_start,
            })

        # Probe
        if multi_probe is not None and step in probe_steps:
            log.info("[probe] running multi-dataset probe at step=%d ...", step)
            metrics = multi_probe.run(student, preprocess, tokenizer,
                                      device=DEVICE, batch_size=args.probe_batch_size)
            student.train()
            for d, m in metrics["per_dataset"].items():
                init_m = init_metrics["per_dataset"][d]
                log.info("[probe] step=%d  %s  MAP@10=%.4f (Δ%+.4f)  R@100=%.4f (Δ%+.4f)",
                         step, d, m["MAP@10"], m["MAP@10"] - init_m["MAP@10"],
                         m["Recall@100"], m["Recall@100"] - init_m["Recall@100"])
            log.info("[probe] step=%d  MEAN  MAP@10=%.4f (Δ%+.4f)  R@100=%.4f (Δ%+.4f)",
                     step, metrics["mean_MAP@10"],
                     metrics["mean_MAP@10"] - init_metrics["mean_MAP@10"],
                     metrics["mean_Recall@100"],
                     metrics["mean_Recall@100"] - init_metrics["mean_Recall@100"])

            log_lines.append({"step": step, "probe": True, "metrics": metrics})

            ckpt_dir = out_dir / f"step_{step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(student.state_dict(), ckpt_dir / "student_state_dict.pt")

            score = metrics["mean_MAP@10"]
            if score > best_score:
                best_score = score
                best_step = step
                best_dir = out_dir / "best"
                best_dir.mkdir(parents=True, exist_ok=True)
                torch.save(student.state_dict(), best_dir / "student_state_dict.pt")
                log.info("[probe] new best @ step=%d  mean MAP@10=%.4f", step, score)

            # Abort criteria
            for d, m in metrics["per_dataset"].items():
                init_m = init_metrics["per_dataset"][d]
                d_map = m["MAP@10"] - init_m["MAP@10"]
                d_r100 = m["Recall@100"] - init_m["Recall@100"]
                if step == args.abort_step1 and d_map < args.abort_step1_min_delta:
                    log.error("[abort] step=%d  %s  ΔMAP@10=%+.4f < threshold %+.4f",
                              step, d, d_map, args.abort_step1_min_delta)
                    aborted = True
                if step == args.abort_step2:
                    if d_map < args.abort_step2_min_delta:
                        log.error("[abort] step=%d  %s  ΔMAP@10=%+.4f < threshold %+.4f",
                                  step, d, d_map, args.abort_step2_min_delta)
                        aborted = True
                    if d_r100 < args.abort_step2_min_r100_delta:
                        log.error("[abort] step=%d  %s  ΔR@100=%+.4f < threshold %+.4f",
                                  step, d, d_r100, args.abort_step2_min_r100_delta)
                        aborted = True
            if step == args.abort_step2:
                d_mean = metrics["mean_MAP@10"] - init_metrics["mean_MAP@10"]
                if d_mean < args.abort_step2_min_mean_delta:
                    log.error("[abort] step=%d  MEAN ΔMAP@10=%+.4f < threshold %+.4f",
                              step, d_mean, args.abort_step2_min_mean_delta)
                    aborted = True

        with open(log_path, "w") as f:
            for line in log_lines:
                f.write(json.dumps(line) + "\n")

        if aborted:
            log.error("[abort] stopping run at step=%d", step)
            break

    summary = {
        "init_metrics": init_metrics,
        "max_steps_reached": step,
        "aborted": aborted,
        "best_step": best_step,
        "best_mean_map10": best_score,
        "wall_time_total_sec": time.time() - t_start,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log.info("=== summary ===")
    log.info("%s", json.dumps(summary, indent=2, default=str))

    gc.collect()
    if DEVICE == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    main()

"""Phase 9 — Gentle Ranking Refinement from Phase 4b checkpoint.

Phase 8 (SmoothAP + ListNet from scratch) collapsed the model.
Root cause: aggressive ranking losses moved embeddings too far.
FSL only moves embeddings 0.005-0.02 cosine distance from base.

Strategy:
  1. Start from Phase 4b dual-loss checkpoint (our best: within 3-7% of FSL)
  2. Mine hard negatives from model's own top-10 retrievals on training data
  3. Apply gentle triplet margin loss with tiny margin (0.02) and very low LR (5e-7)
  4. Early stop at 200 steps max
  5. Monitor embedding drift — if drift > 0.03 cosine, stop immediately

This targets the Phase 7 finding: "the gap is ranking quality, not recall".
We need to nudge the correct items 2-3 positions higher, not restructure embeddings.

Usage:
  python3 scripts/v3/phase9_ranking_refine.py --device mps
  python3 scripts/v3/phase9_ranking_refine.py --max-steps 100 --device mps  # quick test
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ranking-refine")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "processed" / "v3_stratified_multifield"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "v3_ranking_refine"

SCOPES = {
    "text-only": {
        "train_patterns": [
            "text.transformer.resblocks.8.",
            "text.transformer.resblocks.9.",
            "text.transformer.resblocks.10.",
            "text.transformer.resblocks.11.",
            "text.text_projection",
            "text.ln_final",
            "logit_scale",
            "logit_bias",
        ],
    },
}

L1_PATTERNS = [
    (re.compile(r"\b(jackets?|blazers?|coats?|parkas?|anorak|windbreaker|trench|overcoat|capes?|ponchos?|bombers?|vests?|waistcoats?|outerwear|sherwani)\b", re.I), "outerwear"),
    (re.compile(r"(dress|gowns?|rompers?|jumpsuits?|playsuits?|sarees?|kurtas?|kurtis?|kurta.?sets?|galabiyyas?|dhoti)", re.I), "dresses"),
    (re.compile(r"\b(tops?|blouses?|shirts?|tees?|t-?shirts?|tshirts?|tanks?|camisoles?|tunics?|polos?|henleys?|crop.?tops?|halters?|bustiers?|corsets?|hoodies?|sweatshirts?|sweaters?|pullovers?|cardigans?|knits?|chemises?)\b", re.I), "tops"),
    (re.compile(r"\b(pants?|trousers?|jeans?|denim|leggings?|chinos?|shorts?|skirts?|skorts?|culottes?|joggers?|sweatpants?|cargos?|capris?|churidars?|tracksuits?|tights?|hosiery)\b", re.I), "bottoms"),
    (re.compile(r"\b(shoes?|boots?|sneakers?|sandals?|heels?|pumps?|flats?|loafers?|slippers?|mules?|clogs?|espadrilles?|oxfords?|derby|brogues?|stilettos?|wedges?|platforms?|flip.?flops?|booties?|moccasins?)\b", re.I), "shoes"),
    (re.compile(r"\b(bags?|handbags?|purses?|clutches?|totes?|backpacks?|satchels?|crossbody|messenger|wallets?|pouches?|duffles?|weekenders?|rucksacks?|luggage|shoulder.?bags?)\b", re.I), "bags"),
    (re.compile(r"\b(accessor|jewelry|jewellery|necklaces?|bracelets?|bangles?|earrings?|rings?|watches?|sunglasses|eyeglasses|eyewear|scarves?|scarfs?|belts?|hats?|caps?|beanies?|gloves?|ties?|bow.?ties?|cufflinks?|brooches?|pins?|charms?|pendants?|stoles?|dupattas?|umbrellas?|socks?|earmuffs?)\b", re.I), "accessories"),
    (re.compile(r"\b(swimsuits?|bikinis?|swimwear|bathing|swim|one.?piece.?swim)\b", re.I), "swimwear"),
    (re.compile(r"\b(lingerie|bras?|underwear|panty|panties|briefs?|boxers?|nightgowns?|pajamas?|pyjamas?|robes?|sleepwear|lounge|intimates?|shapewear)\b", re.I), "intimates"),
    (re.compile(r"\b(activewear|sportswear|athletic|yoga|gym|workout|running|cycling|fitness|track)\b", re.I), "activewear"),
    (re.compile(r"\b(makeup|mascara|lipstick|foundation|concealer|eyeliner|eyeshadow|blush|fragrance|perfume|cologne|skincare|moisturizer|cleanser|serum|toner|sunscreen|lotion|shampoo|conditioner|nail.?polish|beauty|cosmetic|palette|highlighter)\b", re.I), "beauty"),
    (re.compile(r"\b(furniture|chairs?|tables?|lamps?|rugs?|pillows?|curtains?|beds?|sofas?|mirrors?|vases?|candles?|decor|lighting|shelves?|storage|kitchen|dining|drinkware|flatware|frames?|clocks?|ottoman|dresser|nightstand)\b", re.I), "home"),
]


def classify_l1(text: str) -> str:
    if not text:
        return "other"
    for pattern, label in L1_PATTERNS:
        if pattern.search(text):
            return label
    return "other"


def load_model_from_checkpoint(ckpt_path: str, device: torch.device):
    """Load base SigLIP + apply Phase 4b checkpoint."""
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16-SigLIP", pretrained="webli"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    model_state = model.state_dict()
    model_state.update(state)
    model.load_state_dict(model_state)
    model = model.to(device)
    log.info("Loaded model from %s (%d keys)", ckpt_path, len(state))
    return model, preprocess, tokenizer


@torch.no_grad()
def mine_hard_triplets(
    model, preprocess, tokenizer, device: torch.device,
    n_corpus: int = 5000, n_queries: int = 500, top_k: int = 20,
    seed: int = 42,
) -> list[dict]:
    """Mine hard negative triplets from training data.
    
    For each L1 category query:
      - Encode category label as query
      - Retrieve top-K images
      - Find hard negatives: wrong-category items ranked above correct ones
      - Create triplets: (query, positive, hard_negative)
    """
    model.eval()
    rng = random.Random(seed)

    log.info("Mining hard triplets from training data...")
    jsonl_path = DATA_DIR / "pairs.jsonl"
    all_pairs = []
    with open(jsonl_path) as f:
        for line in f:
            all_pairs.append(json.loads(line))

    rng.shuffle(all_pairs)
    pairs = all_pairs[:n_corpus]

    # Group by L1 category
    cat_to_pairs = defaultdict(list)
    for p in pairs:
        l1 = classify_l1(p.get("query", ""))
        cat_to_pairs[l1].append(p)

    log.info("  %d pairs across %d L1 categories", len(pairs), len(cat_to_pairs))

    # Encode all images
    images_pil = []
    valid_indices = []
    for i, p in enumerate(pairs):
        img_path = DATA_DIR / p["image_path"]
        try:
            img = Image.open(img_path).convert("RGB")
            images_pil.append(img)
            valid_indices.append(i)
        except Exception:
            continue

    log.info("  Encoding %d images...", len(images_pil))
    img_feats_list = []
    batch_size = 32
    for i in range(0, len(images_pil), batch_size):
        batch = images_pil[i:i + batch_size]
        tensors = torch.stack([preprocess(im) for im in batch]).to(device)
        feat = F.normalize(model.encode_image(tensors), dim=-1)
        img_feats_list.append(feat.cpu())
        del tensors, feat
    img_feats = torch.cat(img_feats_list, dim=0)  # (N, D)
    del images_pil, img_feats_list

    # Map valid_indices back to L1 categories
    idx_to_l1 = {}
    for vi, orig_i in enumerate(valid_indices):
        idx_to_l1[vi] = classify_l1(pairs[orig_i].get("query", ""))

    # Get unique L1 categories with enough items
    l1_to_feat_indices = defaultdict(list)
    for vi in range(len(valid_indices)):
        l1_to_feat_indices[idx_to_l1[vi]].append(vi)

    query_l1s = [l1 for l1, idxs in l1_to_feat_indices.items()
                 if len(idxs) >= 3 and l1 != "other"]
    log.info("  %d queryable L1 categories", len(query_l1s))

    # Encode L1 category labels as queries
    txt_feats = []
    for i in range(0, len(query_l1s), 64):
        batch_labels = query_l1s[i:i + 64]
        tokens = tokenizer(batch_labels).to(device)
        feat = F.normalize(model.encode_text(tokens), dim=-1)
        txt_feats.append(feat.cpu())
        del tokens, feat
    txt_feats = torch.cat(txt_feats, dim=0)

    # For each L1 query, find hard triplets
    sims = txt_feats @ img_feats.T  # (Q, N)
    triplets = []

    for qi, l1 in enumerate(query_l1s):
        ranked = torch.argsort(sims[qi], descending=True)[:top_k].numpy()
        pos_indices = set(l1_to_feat_indices[l1])

        # Find hard negatives: wrong-category items in top-K that rank above some positive
        negatives_in_topk = []
        positives_in_topk = []
        for rank, idx in enumerate(ranked):
            if idx in pos_indices:
                positives_in_topk.append((rank, idx))
            else:
                negatives_in_topk.append((rank, idx))

        if not positives_in_topk or not negatives_in_topk:
            continue

        # Create triplets: each (query, positive, negative) where negative ranks above positive
        for neg_rank, neg_idx in negatives_in_topk:
            for pos_rank, pos_idx in positives_in_topk:
                if neg_rank < pos_rank:  # negative ranked ABOVE positive — a ranking error
                    triplets.append({
                        "query_l1": l1,
                        "pos_feat_idx": int(pos_idx),
                        "neg_feat_idx": int(neg_idx),
                        "pos_rank": pos_rank,
                        "neg_rank": neg_rank,
                        "pos_sim": float(sims[qi, pos_idx]),
                        "neg_sim": float(sims[qi, neg_idx]),
                        "margin_needed": float(sims[qi, neg_idx] - sims[qi, pos_idx]),
                    })

    log.info("  Mined %d hard triplets across %d categories", len(triplets), len(query_l1s))

    # Also mine from the actual search queries (not just L1 labels)
    # Use per-query approach for more fine-grained triplets
    query_groups = defaultdict(list)
    for vi, orig_i in enumerate(valid_indices):
        q = pairs[orig_i]["query"]
        query_groups[q].append(vi)

    eligible_queries = [(q, idxs) for q, idxs in query_groups.items() if len(idxs) >= 2]
    rng.shuffle(eligible_queries)
    eligible_queries = eligible_queries[:n_queries]

    if eligible_queries:
        q_texts = [q for q, _ in eligible_queries]
        q_feats = []
        for i in range(0, len(q_texts), 64):
            tokens = tokenizer(q_texts[i:i + 64]).to(device)
            feat = F.normalize(model.encode_text(tokens), dim=-1)
            q_feats.append(feat.cpu())
            del tokens, feat
        q_feats = torch.cat(q_feats, dim=0)

        q_sims = q_feats @ img_feats.T

        for qi, (q_text, pos_indices_list) in enumerate(eligible_queries):
            pos_set = set(pos_indices_list)
            ranked = torch.argsort(q_sims[qi], descending=True)[:top_k].numpy()

            negs = [(r, idx) for r, idx in enumerate(ranked) if idx not in pos_set]
            poss = [(r, idx) for r, idx in enumerate(ranked) if idx in pos_set]

            if not poss or not negs:
                continue

            for nr, ni in negs:
                for pr, pi in poss:
                    if nr < pr:
                        triplets.append({
                            "query_l1": classify_l1(q_text),
                            "query_text": q_text,
                            "pos_feat_idx": int(pi),
                            "neg_feat_idx": int(ni),
                            "pos_rank": pr,
                            "neg_rank": nr,
                            "pos_sim": float(q_sims[qi, pi]),
                            "neg_sim": float(q_sims[qi, ni]),
                            "margin_needed": float(q_sims[qi, ni] - q_sims[qi, pi]),
                        })

    log.info("  Total triplets (L1 + query): %d", len(triplets))

    # Sort by margin_needed (smallest first = hardest) and deduplicate
    triplets.sort(key=lambda t: t["margin_needed"])

    # Store the image features and pair data for training
    return triplets, img_feats, pairs, valid_indices


def train_with_triplets(
    model, preprocess, tokenizer, device: torch.device,
    triplets: list[dict], img_feats: torch.Tensor,
    pairs: list[dict], valid_indices: list[int],
    args,
) -> dict:
    """Gentle triplet margin training on mined hard negatives."""
    scope = SCOPES["text-only"]

    # Freeze everything, unfreeze text scope
    for param in model.parameters():
        param.requires_grad = False
    trainable = 0
    for name, param in model.named_parameters():
        for pattern in scope["train_patterns"]:
            if pattern in name:
                param.requires_grad = True
                trainable += param.numel()
                break
    log.info("Trainable: %d (%.1fM)", trainable, trainable / 1e6)

    # Store initial text embeddings for drift monitoring
    sample_queries = list(set(t.get("query_text", t["query_l1"]) for t in triplets[:200]))[:50]
    with torch.no_grad():
        init_tok = tokenizer(sample_queries).to(device)
        init_text_embs = F.normalize(model.encode_text(init_tok), dim=-1).cpu()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98), eps=1e-6,
    )

    margin = args.margin
    max_steps = args.max_steps
    batch_size = args.batch_size
    drift_threshold = args.drift_threshold

    run_name = args.run_name or f"refine_v3_{int(time.time())}"
    run_dir = CHECKPOINT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("Run: %s", run_name)
    log.info("Triplets: %d, Max steps: %d, Batch: %d, Margin: %.3f, LR: %.2e",
             len(triplets), max_steps, batch_size, margin, args.lr)
    log.info("Drift threshold: %.4f cosine", drift_threshold)
    log.info("=" * 60)

    model.train()
    history = []
    best_loss = float("inf")
    rng = random.Random(42)

    # Pre-load all images for triplets (they're already encoded, but we need
    # the query text to be re-encoded during training since text tower is being trained)
    triplet_indices = list(range(len(triplets)))

    for step in range(1, max_steps + 1):
        rng.shuffle(triplet_indices)
        batch_idx = triplet_indices[:batch_size]
        batch_triplets = [triplets[i] for i in batch_idx]

        # Get query texts and image feature indices
        query_texts = [t.get("query_text", t["query_l1"]) for t in batch_triplets]
        pos_indices = [t["pos_feat_idx"] for t in batch_triplets]
        neg_indices = [t["neg_feat_idx"] for t in batch_triplets]

        # Re-encode queries (text tower is being trained)
        q_tok = tokenizer(query_texts).to(device)
        q_emb = F.normalize(model.encode_text(q_tok), dim=-1)

        # Get cached image features (image tower is frozen)
        pos_emb = img_feats[pos_indices].to(device)
        neg_emb = img_feats[neg_indices].to(device)

        # Triplet margin loss: d(q, pos) should be < d(q, neg) - margin
        # Using cosine: sim(q, pos) should be > sim(q, neg) + margin
        pos_sim = (q_emb * pos_emb).sum(dim=-1)
        neg_sim = (q_emb * neg_emb).sum(dim=-1)

        # Margin loss: max(0, margin - (pos_sim - neg_sim))
        loss = F.relu(margin - (pos_sim - neg_sim)).mean()

        if loss.item() > 0:  # Only update if there's a non-zero loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=0.5
            )
            optimizer.step()

        del q_tok, q_emb, pos_emb, neg_emb

        if step % 10 == 0 or step == 1:
            # Monitor embedding drift
            model.eval()
            with torch.no_grad():
                current_tok = tokenizer(sample_queries).to(device)
                current_text_embs = F.normalize(model.encode_text(current_tok), dim=-1).cpu()
                drift = 1.0 - F.cosine_similarity(init_text_embs, current_text_embs, dim=-1).mean().item()
            model.train()

            log.info("  Step %d/%d: loss=%.4f, drift=%.4f, avg_margin=%.4f",
                     step, max_steps, loss.item(), drift,
                     (pos_sim - neg_sim).mean().item() if loss.item() > 0 else 0.0)

            history.append({
                "step": step, "loss": loss.item(), "drift": drift,
            })

            # Drift safety: stop if embeddings moved too far
            if drift > drift_threshold:
                log.warning("DRIFT EXCEEDED %.4f > %.4f — stopping early!", drift, drift_threshold)
                break

            # Save if best
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save({
                    "step": step,
                    "model_state_dict": {
                        k: v for k, v in model.state_dict().items()
                        if any(p in k for p in scope["train_patterns"])
                    },
                    "loss": loss.item(),
                    "drift": drift,
                    "scope": "text-only",
                }, run_dir / "best.pt")

        if device.type == "mps":
            torch.mps.empty_cache()

    # Save final
    torch.save({
        "step": step,
        "model_state_dict": {
            k: v for k, v in model.state_dict().items()
            if any(p in k for p in scope["train_patterns"])
        },
        "history": history,
        "scope": "text-only",
    }, run_dir / "final.pt")

    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    log.info("Saved checkpoints to %s", run_dir)
    return {"history": history, "run_dir": str(run_dir)}


def parse_args():
    p = argparse.ArgumentParser(description="Phase 9: Gentle Ranking Refinement")
    p.add_argument("--checkpoint", type=str,
                   default="checkpoints/v3_gcl/phase4b_dual_loss/best.pt")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-7)
    p.add_argument("--margin", type=float, default=0.02)
    p.add_argument("--drift-threshold", type=float, default=0.03)
    p.add_argument("--mining-corpus", type=int, default=5000)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--run-name", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    log.info("Device: %s", device)

    ckpt_path = REPO_ROOT / args.checkpoint
    model, preprocess, tokenizer = load_model_from_checkpoint(str(ckpt_path), device)

    triplets, img_feats, pairs, valid_indices = mine_hard_triplets(
        model, preprocess, tokenizer, device,
        n_corpus=args.mining_corpus, seed=42,
    )

    if not triplets:
        log.warning("No hard triplets found — model may already be optimal!")
        return

    log.info("Triplet margin distribution: min=%.4f, median=%.4f, max=%.4f",
             triplets[0]["margin_needed"],
             triplets[len(triplets) // 2]["margin_needed"],
             triplets[-1]["margin_needed"])

    result = train_with_triplets(
        model, preprocess, tokenizer, device,
        triplets, img_feats, pairs, valid_indices, args,
    )

    elapsed = time.time() - t0
    log.info("Phase 9 complete in %.1f minutes", elapsed / 60)

    # Now run quick benchmark eval
    log.info("=" * 60)
    log.info("Running quick benchmark eval...")
    log.info("=" * 60)

    run_dir = Path(result["run_dir"])
    best_ckpt = run_dir / "best.pt"
    if best_ckpt.exists():
        # Reload model from best checkpoint for eval
        model_eval, preprocess_eval, tokenizer_eval = load_model_from_checkpoint(
            str(ckpt_path), device
        )
        ckpt = torch.load(str(best_ckpt), map_location="cpu", weights_only=False)
        model_state = model_eval.state_dict()
        model_state.update(ckpt["model_state_dict"])
        model_eval.load_state_dict(model_state)
        model_eval = model_eval.to(device).eval()

        # Quick eval on fashion200k (the key benchmark)
        from datasets import load_dataset

        benchmarks = ["fashion200k", "polyvore", "KAGL"]
        for bname in benchmarks:
            eval_one_benchmark(model_eval, preprocess_eval, tokenizer_eval,
                              bname, device, corpus_size=3000, seed=42)

    log.info("All done!")


@torch.no_grad()
def eval_one_benchmark(model, preprocess, tokenizer, benchmark_name: str,
                       device: torch.device, corpus_size: int = 3000, seed: int = 42):
    """Quick MAP@10 eval on one benchmark."""
    from datasets import load_dataset

    BENCHMARKS = {
        "fashion200k": {"hf": "Marqo/fashion200k", "split": "data", "query_col": "category3"},
        "atlas":       {"hf": "Marqo/atlas",       "split": "data", "query_col": "sub-category"},
        "polyvore":    {"hf": "Marqo/polyvore",    "split": "data", "query_col": "category"},
        "KAGL":        {"hf": "Marqo/KAGL",        "split": "data", "query_col": "category3"},
    }

    cfg = BENCHMARKS[benchmark_name]
    log.info("  Evaluating %s (corpus=%d)...", benchmark_name, corpus_size)

    ds = load_dataset(cfg["hf"], split=cfg["split"], streaming=True,
                      cache_dir=str(REPO_ROOT / "data" / "hf_cache"))

    categories, images_pil = [], []
    for row in ds:
        if len(categories) >= corpus_size:
            break
        query = (row.get(cfg["query_col"]) or "").strip()
        if not query or not row.get("image"):
            continue
        categories.append(query)
        images_pil.append(row["image"])

    rng = random.Random(seed)
    indices = list(range(len(categories)))
    rng.shuffle(indices)
    indices = indices[:corpus_size]
    categories = [categories[i] for i in indices]
    images_pil = [images_pil[i] for i in indices]

    cat_to_indices = defaultdict(list)
    for idx, cat in enumerate(categories):
        cat_to_indices[cat].append(idx)
    valid_cats = [c for c, idxs in cat_to_indices.items() if len(idxs) >= 2]
    rng2 = random.Random(seed)
    rng2.shuffle(valid_cats)
    query_cats = valid_cats[:min(500, len(valid_cats))]

    img_feats = []
    batch_size = 32
    for i in range(0, len(images_pil), batch_size):
        batch = images_pil[i:i + batch_size]
        tensors = torch.stack([preprocess(im.convert("RGB")) for im in batch]).to(device)
        feat = F.normalize(model.encode_image(tensors), dim=-1)
        img_feats.append(feat.cpu())
        del tensors, feat
    img_feats = torch.cat(img_feats, dim=0)

    txt_feats = []
    for i in range(0, len(query_cats), 64):
        tokens = tokenizer(query_cats[i:i + 64]).to(device)
        feat = F.normalize(model.encode_text(tokens), dim=-1)
        txt_feats.append(feat.cpu())
        del tokens, feat
    txt_feats = torch.cat(txt_feats, dim=0)

    scores = txt_feats @ img_feats.T
    aps = []
    for qi, cat in enumerate(query_cats):
        relevant = set(cat_to_indices[cat])
        ranked = np.argsort(-scores[qi].numpy())[:10]
        hits, psum = 0, 0.0
        for rank, idx in enumerate(ranked, 1):
            if idx in relevant:
                hits += 1
                psum += hits / rank
        n_rel = len(relevant)
        ap = psum / min(10, n_rel) if hits > 0 else 0.0
        aps.append(ap)

    map10 = float(np.mean(aps))
    log.info("  %s MAP@10 = %.4f (%d queries)", benchmark_name, map10, len(aps))
    return map10


if __name__ == "__main__":
    main()

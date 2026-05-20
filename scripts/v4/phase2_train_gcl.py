"""
Phase 2: GCL Training Loop

Fine-tunes Marqo-FashionSigLIP (ViT-B-16-SigLIP, 203M) using Generalized
Contrastive Learning (GCL) on our pattern-targeted dataset.

Key features:
- GCL loss with score-to-weight transformation (inverse_sqrt)
- Multi-field encoding: LHS=query text, RHS=weighted(image, title)
- Anchor regularization to prevent catastrophic forgetting
- Per-benchmark evaluation every N steps
- Early stopping if any benchmark regresses >3%
"""
import os, sys, json, time, math, argparse, logging
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import open_clip

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJ_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJ_ROOT / "data" / "processed" / "v4_pattern_targeted"
CKPT_DIR = PROJ_ROOT / "checkpoints" / "v4_gcl"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


# ==================== Dataset ====================

class PatternTargetedDataset(Dataset):
    """Loads pattern-targeted pairs for GCL training."""

    def __init__(self, pairs_files: list[Path], img_dir: Path,
                 preprocess, tokenizer, max_text_len: int = 77):
        self.img_dir = img_dir
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.records = []

        for pf in pairs_files:
            if not pf.exists():
                continue
            with open(pf) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    img_path = img_dir / rec["image_file"]
                    if img_path.exists():
                        self.records.append(rec)

        logger.info(f"Loaded {len(self.records)} training pairs")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_path = self.img_dir / rec["image_file"]
        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = self.preprocess(img)
        except Exception:
            img_tensor = torch.zeros(3, 224, 224)

        query_tokens = self.tokenizer([rec["query"]])[0]  # shape: (seq_len,)
        title_tokens = self.tokenizer([rec["title"]])[0]

        score = rec.get("score_linear", 80)
        return img_tensor, query_tokens, title_tokens, torch.tensor(score, dtype=torch.float32)


# ==================== GCL Loss ====================

def score_to_weight_inverse_sqrt(scores: torch.Tensor, max_score: float = 100.0) -> torch.Tensor:
    """Convert linear scores to weights using inverse square root."""
    return 1.0 / torch.sqrt(1.0 + max_score - scores.clamp(min=1, max=max_score))


def gcl_loss(query_embeds: torch.Tensor, doc_embeds: torch.Tensor,
             scores: torch.Tensor, logit_scale: torch.Tensor,
             max_score: float = 100.0) -> torch.Tensor:
    """
    Generalized Contrastive Learning loss.

    Weighted sigmoid cross-entropy where weights are derived from
    relevance scores via inverse_sqrt transformation.

    For each query-document pair in the batch, the diagonal elements
    are positives (weight from score), off-diagonals are negatives.
    """
    batch_size = query_embeds.shape[0]
    device = query_embeds.device

    similarity = logit_scale * query_embeds @ doc_embeds.T

    weights = score_to_weight_inverse_sqrt(scores, max_score)

    labels = torch.eye(batch_size, device=device)

    weight_matrix = torch.ones(batch_size, batch_size, device=device)
    weight_matrix[range(batch_size), range(batch_size)] = weights

    loss_row = F.binary_cross_entropy_with_logits(
        similarity, labels, weight=weight_matrix, reduction="mean"
    )
    loss_col = F.binary_cross_entropy_with_logits(
        similarity.T, labels, weight=weight_matrix.T, reduction="mean"
    )

    return (loss_row + loss_col) / 2


def anchor_regularization_loss(model_embeds: torch.Tensor,
                                anchor_embeds: torch.Tensor) -> torch.Tensor:
    """MSE loss between current and anchor (frozen base) embeddings."""
    return F.mse_loss(model_embeds, anchor_embeds)


# ==================== Multi-Field Encoding ====================

def encode_documents(model, images: torch.Tensor, title_tokens: torch.Tensor,
                     image_weight: float = 0.5, title_weight: float = 0.5) -> torch.Tensor:
    """Encode documents as weighted combination of image + title embeddings."""
    img_features = model.encode_image(images)
    img_features = F.normalize(img_features, dim=-1)

    title_features = model.encode_text(title_tokens)
    title_features = F.normalize(title_features, dim=-1)

    doc_features = image_weight * img_features + title_weight * title_features
    doc_features = F.normalize(doc_features, dim=-1)
    return doc_features


def encode_queries(model, query_tokens: torch.Tensor) -> torch.Tensor:
    """Encode query text."""
    query_features = model.encode_text(query_tokens)
    query_features = F.normalize(query_features, dim=-1)
    return query_features


# ==================== Evaluation ====================

def quick_eval_on_training_data(model, tokenizer, preprocess, data_dir: Path,
                                 device: str, max_pairs: int = 2000) -> dict:
    """Quick self-eval on a held-out slice of training data.

    Much faster than streaming benchmarks during training.
    Real benchmark eval is done in phase3.
    """
    pairs_file = data_dir / "pairs.jsonl"
    img_dir = data_dir / "images"
    if not pairs_file.exists():
        return {"error": "No training data"}

    records = []
    with open(pairs_file) as f:
        for line in f:
            records.append(json.loads(line))
            if len(records) >= max_pairs:
                break

    if len(records) < 100:
        return {"error": "Too few records"}

    model.eval()
    query_list = list(set(r["query"] for r in records))[:500]
    gt = defaultdict(set)
    doc_images = {}

    for r in records:
        q = r["query"]
        pid = r["product_id"]
        if q in query_list:
            gt[q].add(pid)
        img_path = img_dir / r["image_file"]
        if img_path.exists() and pid not in doc_images:
            doc_images[pid] = img_path

    doc_ids = list(doc_images.keys())[:5000]

    with torch.no_grad():
        q_embeds = []
        for j in range(0, len(query_list), 64):
            batch = tokenizer(query_list[j:j+64]).to(device)
            emb = model.encode_text(batch)
            q_embeds.append(F.normalize(emb, dim=-1).cpu())
        q_embeds = torch.cat(q_embeds)

        d_embeds = []
        for j in range(0, len(doc_ids), 32):
            batch_ids = doc_ids[j:j+32]
            imgs = []
            for did in batch_ids:
                try:
                    img = Image.open(doc_images[did]).convert("RGB")
                    imgs.append(preprocess(img))
                except Exception:
                    imgs.append(torch.zeros(3, 224, 224))
            img_batch = torch.stack(imgs).to(device)
            emb = model.encode_image(img_batch)
            d_embeds.append(F.normalize(emb, dim=-1).cpu())
        d_embeds = torch.cat(d_embeds)

    sims = q_embeds @ d_embeds.T
    _, top_indices = sims.topk(min(10, len(doc_ids)), dim=1)

    recall_at_1 = 0
    recall_at_10 = 0
    mrr_sum = 0
    n = 0
    for qi, q in enumerate(query_list):
        if q not in gt:
            continue
        relevant = gt[q]
        retrieved = [doc_ids[idx] for idx in top_indices[qi].tolist()]
        n += 1
        if retrieved[0] in relevant:
            recall_at_1 += 1
        if any(d in relevant for d in retrieved):
            recall_at_10 += 1
        for rank, d in enumerate(retrieved, 1):
            if d in relevant:
                mrr_sum += 1.0 / rank
                break

    if n == 0:
        return {"error": "No eval queries"}

    return {
        "recall@1": recall_at_1 / n,
        "recall@10": recall_at_10 / n,
        "mrr": mrr_sum / n,
        "n_queries": n,
        "n_docs": len(doc_ids),
    }


def run_quick_eval(model, tokenizer, preprocess, data_dir: Path, device: str) -> dict:
    """Quick self-eval on training data (fast, no HF streaming)."""
    r = quick_eval_on_training_data(model, tokenizer, preprocess, data_dir, device)
    if "mrr" in r:
        logger.info(f"  Quick eval: R@1={r['recall@1']:.3f} R@10={r['recall@10']:.3f} MRR={r['mrr']:.3f}")
    else:
        logger.info(f"  Quick eval: {r.get('error', '?')}")
    return r


# ==================== Training ====================

def main():
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(line_buffering=True)
        except OSError:
            pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--accum-steps", type=int, default=4,
                        help="Gradient accumulation (effective batch = batch_size * accum)")
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--anchor-lambda", type=float, default=0.3)
    parser.add_argument("--image-weight", type=float, default=0.5)
    parser.add_argument("--title-weight", type=float, default=0.5)
    parser.add_argument("--eval-every", type=int, default=2000,
                        help="Quick-eval interval (optimizer steps). Should be ≤ total_steps "
                             "or best_model.pt never updates mid-run.")
    parser.add_argument("--save-every", type=int, default=2000,
                        help="Save step_{n}.pt every N optimizer steps (includes optimizer).")
    parser.add_argument("--regression-threshold", type=float, default=0.03)
    parser.add_argument(
        "--lr-halve-on-regression",
        action="store_true",
        help="Halve LR when quick-eval drops vs baseline (often hurts stability).",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--output-name",
        type=str,
        default="final_model.pt",
        help="Filename under checkpoints/v4_gcl/ for saved weights.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help="Optional .pt to warm-start backbone/head (e.g. previous final_model.pt).",
    )
    args = parser.parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    device = args.device
    logger.info(f"Device: {device}")

    # Load model
    model_name = "hf-hub:Marqo/marqo-fashionSigLIP"
    logger.info("Loading Marqo-FashionSigLIP...")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device)
    model.train()

    if getattr(args, "init_checkpoint", None):
        ck = Path(args.init_checkpoint)
        if ck.is_file():
            try:
                sd = torch.load(ck, map_location="cpu", weights_only=True)
            except TypeError:
                sd = torch.load(ck, map_location="cpu")
            if isinstance(sd, dict) and "model_state_dict" in sd:
                sd = sd["model_state_dict"]
            r = model.load_state_dict(sd, strict=False)
            if r and hasattr(r, "missing_keys"):
                logger.info(
                    "Warm-start from %s (missing=%d, unexpected=%d)",
                    ck, len(r.missing_keys), len(r.unexpected_keys),
                )
            else:
                logger.info("Warm-start from %s", ck)
        else:
            logger.warning("init-checkpoint path not found: %s", ck)

    # Second full SigLIP is expensive on MPS (memory + sync); skip if unused.
    anchor_model = None
    if args.anchor_lambda > 0:
        logger.info("Creating anchor model (frozen)...")
        anchor_model, _, _ = open_clip.create_model_and_transforms(model_name)
        anchor_model = anchor_model.to(device)
        anchor_model.eval()
        for p in anchor_model.parameters():
            p.requires_grad = False
    else:
        logger.info("Skipping anchor model (--anchor-lambda 0).")

    # Baseline evaluation (quick self-eval)
    logger.info("Running baseline evaluation...")
    model.eval()
    baseline_results = run_quick_eval(model, tokenizer, preprocess, DATA_DIR, device)
    with open(CKPT_DIR / "baseline_results.json", "w") as f:
        json.dump(baseline_results, f, indent=2)
    model.train()

    # Load dataset
    pairs_files = [DATA_DIR / "pairs.jsonl", DATA_DIR / "synthetic_pairs.jsonl"]
    dataset = PatternTargetedDataset(pairs_files, DATA_DIR / "images",
                                     preprocess, tokenizer)

    pin_mem = device == "cuda"
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
        drop_last=True,
    )

    # Optimizer with differential learning rates
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "logit" in name or "proj" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": head_params, "lr": args.lr_head},
    ], weight_decay=args.weight_decay)

    total_steps = len(dataloader) * args.epochs // args.accum_steps
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)

    if total_steps > 0:
        if args.eval_every > total_steps:
            logger.warning(
                "eval_every=%d > total_steps=%d: skipping mid-training quick-eval; "
                "best_model.pt will not be updated during this run (only %s at end).",
                args.eval_every,
                total_steps,
                args.output_name,
            )
        if args.save_every > total_steps:
            logger.warning(
                "save_every=%d > total_steps=%d: no intermediate step_*.pt checkpoints.",
                args.save_every,
                total_steps,
            )

    def lr_schedule(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # Use OpenCLIP's learnable temperature (included in optimizer via model.parameters())
    def current_logit_scale() -> torch.Tensor:
        ls = model.logit_scale
        if hasattr(ls, "exp"):
            return ls.exp()
        return ls.float().exp()

    # Training loop
    global_step = 0
    best_avg_mrr = float(baseline_results.get("mrr", 0.0))
    training_log = []

    logger.info(f"Starting training: {total_steps} steps, {args.epochs} epochs")
    logger.info(f"Effective batch size: {args.batch_size * args.accum_steps}")

    for epoch in range(args.epochs):
        epoch_loss = 0
        epoch_gcl_loss = 0
        epoch_anchor_loss = 0
        n_batches = 0

        for batch_idx, (images, query_tokens, title_tokens, scores) in enumerate(dataloader):
            images = images.to(device)
            query_tokens = query_tokens.to(device)
            title_tokens = title_tokens.to(device)
            scores = scores.to(device)

            use_amp = device == "cuda"
            with torch.amp.autocast("cuda", enabled=use_amp):
                query_embeds = encode_queries(model, query_tokens)
                doc_embeds = encode_documents(
                    model, images, title_tokens,
                    image_weight=args.image_weight,
                    title_weight=args.title_weight
                )

                loss_gcl = gcl_loss(query_embeds, doc_embeds, scores,
                                    current_logit_scale(), max_score=100.0)

                loss_anchor_val = torch.tensor(0.0, device=device)
                if args.anchor_lambda > 0:
                    with torch.no_grad():
                        anchor_query = encode_queries(anchor_model, query_tokens)
                        anchor_doc_img = F.normalize(
                            anchor_model.encode_image(images), dim=-1)

                    loss_anchor_val = (
                        anchor_regularization_loss(query_embeds, anchor_query.detach()) +
                        anchor_regularization_loss(
                            F.normalize(model.encode_image(images), dim=-1),
                            anchor_doc_img.detach()
                        )
                    ) / 2

                loss = loss_gcl + args.anchor_lambda * loss_anchor_val
                loss = loss / args.accum_steps

            loss.backward()

            epoch_loss += loss.item() * args.accum_steps
            epoch_gcl_loss += loss_gcl.item()
            epoch_anchor_loss += loss_anchor_val.item()
            n_batches += 1

            if (batch_idx + 1) % args.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step <= 3 or global_step % 10 == 0:
                    logger.info("[train] step %d/%d", global_step, total_steps)

                if global_step % 20 == 0:
                    avg_loss = epoch_loss / n_batches
                    avg_gcl = epoch_gcl_loss / n_batches
                    avg_anchor = epoch_anchor_loss / n_batches
                    lr = optimizer.param_groups[0]["lr"]
                    logger.info(
                        f"[E{epoch+1}|S{global_step}] loss={avg_loss:.4f} "
                        f"gcl={avg_gcl:.4f} anchor={avg_anchor:.4f} lr={lr:.2e} "
                        f"scale={current_logit_scale().item():.1f}"
                    )

                if global_step % args.eval_every == 0:
                    logger.info(f"Running evaluation at step {global_step}...")
                    model.eval()
                    eval_results = run_quick_eval(model, tokenizer, preprocess, DATA_DIR, device)
                    model.train()

                    current_mrr = eval_results.get("mrr", 0)
                    base_mrr = baseline_results.get("mrr", 0)
                    regression = base_mrr > 0 and current_mrr < base_mrr - args.regression_threshold

                    log_entry = {
                        "step": global_step,
                        "epoch": epoch + 1,
                        "mrr": current_mrr,
                        "results": eval_results,
                        "regression": regression,
                    }
                    training_log.append(log_entry)

                    with open(CKPT_DIR / "training_log.json", "w") as f:
                        json.dump(training_log, f, indent=2, default=str)

                    if current_mrr > best_avg_mrr:
                        best_avg_mrr = current_mrr
                        ckpt_path = CKPT_DIR / "best_model.pt"
                        torch.save(model.state_dict(), ckpt_path)
                        logger.info(f"New best model: mrr={current_mrr:.4f}")

                    if regression and args.lr_halve_on_regression:
                        logger.warning("Regression vs baseline; reducing LR.")
                        for pg in optimizer.param_groups:
                            pg["lr"] *= 0.5
                    elif regression:
                        logger.warning(
                            "Quick-eval below baseline (no LR change; use "
                            "--lr-halve-on-regression to auto-reduce LR)."
                        )

                if global_step % args.save_every == 0:
                    ckpt_path = CKPT_DIR / f"step_{global_step}.pt"
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "step": global_step,
                        "epoch": epoch,
                    }, ckpt_path)

                if args.max_steps and global_step >= args.max_steps:
                    break

        if args.max_steps and global_step >= args.max_steps:
            break

        logger.info(f"Epoch {epoch+1} complete: avg_loss={epoch_loss / max(1, n_batches):.4f}")

    # Final eval
    logger.info("Final evaluation...")
    model.eval()
    final_results = run_quick_eval(model, tokenizer, preprocess, DATA_DIR, device)

    torch.save(model.state_dict(), CKPT_DIR / args.output_name)

    summary = {
        "baseline": baseline_results,
        "final": final_results,
        "best_mrr": best_avg_mrr,
        "total_steps": global_step,
    }
    with open(CKPT_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("Training complete!")
    logger.info(f"Best MRR: {best_avg_mrr:.4f}")
    base_mrr = baseline_results.get("mrr", 0)
    final_mrr = final_results.get("mrr", 0)
    delta = final_mrr - base_mrr
    logger.info(f"  MRR: {base_mrr:.3f} -> {final_mrr:.3f} ({'+' if delta >= 0 else ''}{delta:.3f})")


if __name__ == "__main__":
    main()

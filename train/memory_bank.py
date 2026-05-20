"""FIFO memory bank for small-batch contrastive learning on MPS (Appendix B)."""
import torch
import torch.nn.functional as F


class FIFOMemoryBank:
    def __init__(self, size: int, dim: int, device: str = "mps"):
        self.size = size
        self.dim = dim
        self.device = device
        self.img_bank = torch.zeros(size, dim, device=device, dtype=torch.float32)
        self.txt_bank = torch.zeros(size, dim, device=device, dtype=torch.float32)
        self.ptr = 0
        self.filled = 0

    @torch.no_grad()
    def update(self, img_emb: torch.Tensor, txt_emb: torch.Tensor):
        b = img_emb.size(0)
        img_n = F.normalize(img_emb.detach().float(), dim=-1)
        txt_n = F.normalize(txt_emb.detach().float(), dim=-1)
        if self.ptr + b <= self.size:
            self.img_bank[self.ptr : self.ptr + b] = img_n
            self.txt_bank[self.ptr : self.ptr + b] = txt_n
        else:
            first = self.size - self.ptr
            self.img_bank[self.ptr :] = img_n[:first]
            self.img_bank[: b - first] = img_n[first:]
            self.txt_bank[self.ptr :] = txt_n[:first]
            self.txt_bank[: b - first] = txt_n[first:]
        self.ptr = (self.ptr + b) % self.size
        self.filled = min(self.filled + b, self.size)

    def get(self, modality: str = "img") -> torch.Tensor:
        bank = self.img_bank if modality == "img" else self.txt_bank
        return bank[: self.filled]


def sigmoid_gcl_loss(
    img_emb: torch.Tensor,
    txt_emb: torch.Tensor,
    bank: "FIFOMemoryBank",
    logit_scale: torch.Tensor,
    logit_bias: torch.Tensor,
    gcl_weights: torch.Tensor | None = None,
    bank_weight: float = 0.5,
    gcl_piecewise: bool = False,
) -> tuple[torch.Tensor, dict]:
    """
    GCL weighted sigmoid loss with optional memory bank negatives.

    img_emb, txt_emb : (B, D) — raw (pre-norm) embeddings
    gcl_weights      : (B,) float32 in [1, 5] — per-pair GCL importance weight
    gcl_piecewise    : if True, apply 5× to top-10% pairs (Stage 3)
    """
    img_n = F.normalize(img_emb.float(), dim=-1)
    txt_n = F.normalize(txt_emb.float(), dim=-1)
    B = img_n.size(0)
    t = logit_scale.float().exp()

    # In-batch sigmoid matrix: (B, B)
    logits = t * (img_n @ txt_n.T) + logit_bias.float()
    labels = 2.0 * torch.eye(B, device=img_n.device) - 1.0  # +1 diag, -1 off
    loss_mat = -F.logsigmoid(labels * logits)  # (B, B)

    if gcl_weights is not None:
        w = gcl_weights.float().to(img_n.device)  # (B,)
        # Weight rows (image as query) and cols (text as query)
        loss_in = ((loss_mat * w.unsqueeze(1)).mean() + (loss_mat * w.unsqueeze(0)).mean()) / 2.0
    elif gcl_piecewise:
        # Stage 3: score = diag of logits, top-10% → 5×
        scores = logits.diagonal().detach()
        thresh = torch.quantile(scores, 0.90)
        w = torch.where(scores >= thresh, torch.tensor(5.0, device=scores.device), torch.tensor(1.0, device=scores.device))
        loss_in = ((loss_mat * w.unsqueeze(1)).mean() + (loss_mat * w.unsqueeze(0)).mean()) / 2.0
    else:
        loss_in = loss_mat.mean()

    # Bank negatives
    loss_bank = torch.tensor(0.0, device=img_n.device)
    if bank.filled > 0:
        # .clone() breaks the view link so bank.update()'s in-place slice assignment
        # doesn't bump the version counter of tensors saved by autograd.
        bank_txt = bank.get("txt").to(img_n.device).clone()  # (N, D)
        bank_img = bank.get("img").to(img_n.device).clone()

        logits_ib = t * (img_n @ bank_txt.T) + logit_bias.float()  # (B, N)
        loss_bank = loss_bank + (-F.logsigmoid(-logits_ib)).mean()

        logits_tb = t * (txt_n @ bank_img.T) + logit_bias.float()
        loss_bank = loss_bank + (-F.logsigmoid(-logits_tb)).mean()

    bank.update(img_emb, txt_emb)
    total = loss_in + bank_weight * loss_bank
    return total, {"loss_in": loss_in.item(), "loss_bank": loss_bank.item()}


def multi_positive_loss(
    img_emb: torch.Tensor,
    txt_emb: torch.Tensor,
    logit_scale: torch.Tensor,
    logit_bias: torch.Tensor,
    img_sim_thresh: float = 0.85,
    txt_sim_thresh: float = 0.90,
) -> torch.Tensor:
    """
    Add sigmoid loss for mined in-batch positives beyond the diagonal.
    img_sim_thresh / txt_sim_thresh: cosine similarity thresholds.
    """
    img_n = F.normalize(img_emb.float(), dim=-1)
    txt_n = F.normalize(txt_emb.float(), dim=-1)
    t = logit_scale.float().exp()

    # Mine additional positives
    img_img = img_n @ img_n.T  # (B, B)
    txt_txt = txt_n @ txt_n.T

    B = img_n.size(0)
    eye = torch.eye(B, device=img_n.device, dtype=torch.bool)

    extra_pos_img = (img_img > img_sim_thresh) & ~eye
    extra_pos_txt = (txt_txt > txt_sim_thresh) & ~eye

    extra_pos = extra_pos_img | extra_pos_txt
    if not extra_pos.any():
        return torch.tensor(0.0, device=img_n.device)

    logits = t * (img_n @ txt_n.T) + logit_bias.float()
    # Treat extra positives as positive pairs
    extra_loss = -F.logsigmoid(logits[extra_pos]).mean()
    return extra_loss

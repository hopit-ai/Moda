"""Fashion training dataset with title-rendering augmentation and multi-field support."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset


def render_title_on_image(img: Image.Image, title: str, font_size: int = 14) -> Image.Image:
    """Render product title in a white box at top-left corner of image."""
    img = img.copy()
    draw = ImageDraw.Draw(img)
    # White box background
    max_chars = 45
    text = title[:max_chars] + ("…" if len(title) > max_chars else "")
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    pad = 3
    box_w = bbox[2] - bbox[0] + 2 * pad
    box_h = bbox[3] - bbox[1] + 2 * pad
    draw.rectangle([0, 0, box_w, box_h], fill=(255, 255, 255))
    draw.text((pad, pad), text, fill=(0, 0, 0), font=font)
    return img


class FashionPairDataset(Dataset):
    """
    Loads (query_text, image) pairs for contrastive training.

    Each JSONL record must have:
      - query          : str  — text query (short/medium/long)
      - image_path     : str  — absolute or relative path to image
      - title          : str  — product title (for title-rendering aug)
      - score          : float (optional) — relevance score 1-100
      - long_description: str (optional)

    title_render_prob: fraction of batches where title is rendered onto image.
    """

    def __init__(
        self,
        jsonl_paths: list[str | Path],
        preprocess,
        title_render_prob: float = 0.5,
        max_pairs: int | None = None,
    ):
        self.preprocess = preprocess
        self.title_render_prob = title_render_prob
        self.records: list[dict] = []

        for jpath in jsonl_paths:
            jpath = Path(jpath)
            if not jpath.exists():
                print(f"  WARNING: {jpath} not found, skipping")
                continue
            for line in jpath.open():
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                img_path = Path(r["image_path"])
                if not img_path.is_absolute():
                    # Try relative to repo root
                    img_path = Path(__file__).resolve().parents[1] / img_path
                if img_path.exists():
                    r["_img_path"] = str(img_path)
                    self.records.append(r)

        if max_pairs is not None:
            random.shuffle(self.records)
            self.records = self.records[:max_pairs]

        print(f"  FashionPairDataset: {len(self.records):,} valid pairs")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        query = r["query"]
        title = r.get("title", query)
        score = float(r.get("score", r.get("score_linear", 50.0)))

        try:
            img = Image.open(r["_img_path"]).convert("RGB")
            if random.random() < self.title_render_prob and title:
                img = render_title_on_image(img, title)
            img_tensor = self.preprocess(img)
        except Exception:
            img_tensor = torch.zeros(3, 256, 256)

        long_desc = r.get("long_description", "")
        return query, img_tensor, title, long_desc, score


def fashion_collate(batch):
    queries, imgs, titles, descs, scores = zip(*batch)
    return (
        list(queries),
        torch.stack(imgs),
        list(titles),
        list(descs),
        torch.tensor(scores, dtype=torch.float32),
    )

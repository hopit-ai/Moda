"""
Download H&M product images from microsoft/hnm-search-data (HuggingFace).

Uses direct HTTP URLs + aiohttp for high-throughput async parallel downloads.
Resumes automatically (skips already-downloaded files).

Usage:
  python scripts/download_hnm_images.py
  python scripts/download_hnm_images.py --concurrency 64
"""

from __future__ import annotations

import asyncio
import csv
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_URL = (
    "https://huggingface.co/datasets/microsoft/hnm-search-data"
    "/resolve/main/data/raw/images"
)

_REPO_ROOT = Path(__file__).parent.parent
LOCAL_IMAGE_DIR = _REPO_ROOT / "data" / "raw" / "hnm_images"
ARTICLES_CSV = _REPO_ROOT / "data" / "raw" / "hnm_real" / "articles.csv"


def get_download_tasks() -> list[tuple[str, str, Path]]:
    """Return list of (article_id, url, local_path) for images not yet downloaded."""
    tasks = []
    with open(ARTICLES_CSV, newline="") as f:
        for row in csv.DictReader(f):
            aid = row.get("article_id", "").strip()
            if not aid:
                continue
            aid_padded = aid.zfill(10)
            prefix = aid_padded[:3]
            local_path = LOCAL_IMAGE_DIR / prefix / f"{aid_padded}.jpg"
            if local_path.exists() or local_path.is_symlink():
                continue
            url = f"{BASE_URL}/{prefix}/{aid_padded}.jpg"
            tasks.append((aid_padded, url, local_path))
    return tasks


async def download_batch(
    tasks: list[tuple[str, str, Path]],
    concurrency: int = 10,
    timeout_s: int = 60,
):
    import aiohttp

    sem = asyncio.Semaphore(concurrency)
    done = 0
    failed = 0
    failed_ids = []
    total = len(tasks)
    t_start = time.time()

    connector = aiohttp.TCPConnector(limit=concurrency, force_close=False)
    timeout = aiohttp.ClientTimeout(total=timeout_s)

    async with aiohttp.ClientSession(
        connector=connector, timeout=timeout,
        headers={"User-Agent": "MODA-downloader/1.0"},
    ) as session:
        async def fetch_one(aid: str, url: str, path: Path):
            nonlocal done, failed
            async with sem:
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        async with session.get(url, allow_redirects=True) as resp:
                            if resp.status == 200:
                                path.parent.mkdir(parents=True, exist_ok=True)
                                data = await resp.read()
                                path.write_bytes(data)
                                done += 1
                                break
                            elif resp.status == 429:
                                wait = 2 ** attempt * 5
                                await asyncio.sleep(wait)
                                continue
                            elif resp.status == 404:
                                failed += 1
                                if len(failed_ids) < 20:
                                    failed_ids.append(aid)
                                break
                            else:
                                if attempt == max_retries - 1:
                                    failed += 1
                                    if len(failed_ids) < 20:
                                        failed_ids.append(f"{aid}(HTTP{resp.status})")
                                else:
                                    await asyncio.sleep(2 ** attempt)
                    except Exception as e:
                        if attempt == max_retries - 1:
                            failed += 1
                            if len(failed_ids) < 20:
                                failed_ids.append(f"{aid}({type(e).__name__})")
                        else:
                            await asyncio.sleep(2 ** attempt)

                current = done + failed
                if current % 2000 == 0 or current == total:
                    elapsed = time.time() - t_start
                    rate = done / elapsed if elapsed > 0 else 0
                    log.info(
                        "Progress: %d/%d done (%.1f%%)  %.0f img/s  %.1f min  %d failed",
                        done, total, 100 * done / total,
                        rate, elapsed / 60, failed,
                    )

        await asyncio.gather(*(fetch_one(aid, url, path) for aid, url, path in tasks))

    elapsed = time.time() - t_start
    log.info("Download complete: %d ok, %d failed in %.1f min (%.0f img/s)",
             done, failed, elapsed / 60, (done + failed) / elapsed if elapsed > 0 else 0)
    if failed_ids:
        log.info("Sample failures: %s", failed_ids[:10])
    return done, failed


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--concurrency", type=int, default=50)
    args = p.parse_args()

    log.info("=" * 60)
    log.info("H&M Product Image Download (Phase 4A)")
    log.info("Target: %s", LOCAL_IMAGE_DIR)
    log.info("Concurrency: %d", args.concurrency)
    log.info("=" * 60)

    LOCAL_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    tasks = get_download_tasks()
    already = 105542 - len(tasks)
    log.info("Total articles: 105,542 | Already present: %d | To download: %d",
             already, len(tasks))

    if not tasks:
        log.info("All images already downloaded!")
    else:
        asyncio.run(download_batch(tasks, concurrency=args.concurrency))

    n_images = sum(1 for _ in LOCAL_IMAGE_DIR.rglob("*.jpg"))
    coverage = n_images / 105542 * 100

    print(f"\n{'=' * 60}")
    print("PHASE 4A -- Image Download Summary")
    print(f"{'=' * 60}")
    print(f"  Images present:  {n_images:,}")
    print(f"  Coverage:        {coverage:.1f}%")
    print(f"  Missing:         {105542 - n_images:,}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()

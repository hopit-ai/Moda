"""
Phase 0: Verify the full setup is ready.
Checks datasets, models, repos, and OpenSearch connectivity.
"""

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

EXPECTED_DATASETS = [
    "deepfashion_inshop",
    "deepfashion_multimodal",
    "fashion200k",
    "atlas",
    "polyvore",
    "hnm",
]

EXPECTED_MODELS = [
    "marqo-fashionSigLIP",
    "marqo-fashionCLIP",
    "clip-vit-base-patch32",
]

EXPECTED_REPOS = [
    "marqo-FashionCLIP",
    "GCL",
]


def check_datasets() -> dict:
    results = {}
    raw_dir = BASE_DIR / "data" / "raw"
    for name in EXPECTED_DATASETS:
        path = raw_dir / name
        exists = path.exists() and any(path.iterdir()) if path.exists() else False
        results[name] = {"exists": exists, "path": str(path)}
    return results


def check_models() -> dict:
    results = {}
    models_dir = BASE_DIR / "models"
    for name in EXPECTED_MODELS:
        path = models_dir / name
        exists = path.exists() and any(path.iterdir()) if path.exists() else False
        results[name] = {"exists": exists, "path": str(path)}
    return results


def check_repos() -> dict:
    results = {}
    repos_dir = BASE_DIR / "repos"
    for name in EXPECTED_REPOS:
        path = repos_dir / name
        exists = path.exists()
        results[name] = {"exists": exists, "path": str(path)}
    return results


def check_opensearch() -> dict:
    try:
        import requests
        resp = requests.get("http://localhost:9200", timeout=3)
        data = resp.json()
        return {"reachable": True, "version": data.get("version", {}).get("number", "unknown")}
    except Exception as e:
        return {"reachable": False, "error": str(e)}


def main():
    print("=" * 60)
    print("MODA Phase 0 - Setup Verification")
    print("=" * 60)

    all_ok = True

    print("\n[1/4] Datasets")
    datasets = check_datasets()
    for name, info in datasets.items():
        status = "✓" if info["exists"] else "✗"
        if not info["exists"]:
            all_ok = False
        print(f"  {status} {name}")

    print("\n[2/4] Models")
    models = check_models()
    for name, info in models.items():
        status = "✓" if info["exists"] else "✗"
        if not info["exists"]:
            all_ok = False
        print(f"  {status} {name}")

    print("\n[3/4] Repos")
    repos = check_repos()
    for name, info in repos.items():
        status = "✓" if info["exists"] else "✗"
        if not info["exists"]:
            all_ok = False
        print(f"  {status} {name}")

    print("\n[4/4] OpenSearch")
    os_check = check_opensearch()
    if os_check["reachable"]:
        print(f"  ✓ OpenSearch reachable (v{os_check['version']})")
    else:
        print(f"  ⚠ OpenSearch not reachable: {os_check.get('error', 'unknown')}")
        print("    (Not blocking — OpenSearch is needed for Phase 2)")

    print("\n" + "=" * 60)
    if all_ok:
        print("Phase 0 Complete ✓")
    else:
        print("Phase 0 Incomplete — some checks failed (see above)")
    print("=" * 60)

    report = {
        "datasets": datasets,
        "models": models,
        "repos": repos,
        "opensearch": os_check,
        "all_ready": all_ok,
    }
    report_path = BASE_DIR / "logs" / "verify_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to {report_path}")

    return all_ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)

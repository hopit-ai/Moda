#!/bin/bash
#
# Upload MODA models to HuggingFace Hub.
#
# Prerequisites:
#   1. pip install huggingface_hub
#   2. huggingface-cli login  (paste your HF token)
#
# Usage:
#   bash scripts/upload_to_hf.sh YOUR_HF_USERNAME
#   bash scripts/upload_to_hf.sh YOUR_HF_USERNAME distilled   # upload one model
#

set -euo pipefail

USERNAME="${1:?Usage: $0 <hf_username> [model_name]}"
MODEL="${2:-all}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
HF_DIR="$REPO_ROOT/hf_repos"

upload_model() {
    local name="$1"
    local dir="$HF_DIR/$name"

    if [ ! -d "$dir" ]; then
        echo "ERROR: $dir does not exist. Run: python scripts/package_for_hf.py"
        return 1
    fi

    echo ""
    echo "============================================"
    echo "  Uploading $name -> $USERNAME/$name"
    echo "============================================"

    huggingface-cli upload "$USERNAME/$name" "$dir" \
        --commit-message "Upload MODA model: $name"

    echo "  Done! https://huggingface.co/$USERNAME/$name"
}

MODELS=(
    moda-fashion-distilled
    moda-fashion-matryoshka
    moda-fashion-deepfashion2
    moda-fashion-distilled-512d
)

if [ "$MODEL" = "all" ]; then
    for m in "${MODELS[@]}"; do
        upload_model "$m"
    done
else
    # Accept short names like "distilled" -> "moda-fashion-distilled"
    FULL_NAME="moda-fashion-$MODEL"
    if [ -d "$HF_DIR/$FULL_NAME" ]; then
        upload_model "$FULL_NAME"
    elif [ -d "$HF_DIR/$MODEL" ]; then
        upload_model "$MODEL"
    else
        echo "ERROR: Unknown model '$MODEL'. Available:"
        for m in "${MODELS[@]}"; do echo "  $m"; done
        exit 1
    fi
fi

echo ""
echo "============================================"
echo "  All uploads complete!"
echo "============================================"
echo ""
echo "Your models are at:"
for m in "${MODELS[@]}"; do
    echo "  https://huggingface.co/$USERNAME/$m"
done

#!/bin/bash
# Download DeepFashion Category & Attribute Prediction Benchmark from CUHK
# Google Drive (best-effort; large dataset).
#
# Image archives are split across 7 zip parts (~2GB each), plus annotation
# directories and eval split files (small).
#
# Reference folder (CUHK official):
#   https://drive.google.com/drive/folders/0B7EVK8r0v71pQ2FuZ0k0QnhBQnc
#
# Usage:
#   ./scripts/download_deepfashion_ca.sh        # foreground
#   nohup ./scripts/download_deepfashion_ca.sh > logs/ca_download.log 2>&1 &

set -u
cd "$(dirname "$0")/.."
REPO="$(pwd)"
source .venv/bin/activate

OUT="$REPO/data/raw/deepfashion_ca"
mkdir -p "$OUT"
LOG="$REPO/logs/ca_download.log"
mkdir -p "$REPO/logs"

banner() {
    echo ""                                               | tee -a "$LOG"
    echo "==========================================="    | tee -a "$LOG"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')  $1"            | tee -a "$LOG"
    echo "==========================================="    | tee -a "$LOG"
}

banner "DeepFashion Category & Attribute download"
echo "  Output: $OUT"   | tee -a "$LOG"
echo "  Disk free before:" | tee -a "$LOG"
df -h "$OUT" | tail -2 | tee -a "$LOG"

# Strategy: download the whole CUHK folder via gdown --folder. This will
# fetch every file CUHK published in the C&A benchmark folder (Img/, Anno/,
# Eval/, list_*.txt) preserving structure.
banner "Attempting gdown --folder on CUHK C&A root"
gdown --folder \
    --output "$OUT" \
    --remaining-ok \
    "https://drive.google.com/drive/folders/0B7EVK8r0v71pQ2FuZ0k0QnhBQnc" \
    2>&1 | tee -a "$LOG"
RC=$?
banner "gdown rc=$RC"

if [ $RC -ne 0 ]; then
    echo "  gdown folder fetch FAILED. The CUHK folder may have moved or" | tee -a "$LOG"
    echo "  Google Drive may be rate-limiting. Manual options:"           | tee -a "$LOG"
    echo "    1. Visit https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html" | tee -a "$LOG"
    echo "    2. Open the Google Drive link, request access if needed,"   | tee -a "$LOG"
    echo "       and download Img.zip + Anno_coarse.zip + Anno_fine.zip"  | tee -a "$LOG"
    echo "       + Eval/ to $OUT manually."                               | tee -a "$LOG"
    exit $RC
fi

banner "Download complete. Listing contents:"
ls -lah "$OUT" | tee -a "$LOG"

# Try to unzip image part(s) if present
if ls "$OUT"/Img*.zip 1>/dev/null 2>&1; then
    banner "Unzipping Img*.zip"
    for z in "$OUT"/Img*.zip; do
        echo "  unzip $z" | tee -a "$LOG"
        unzip -q -n "$z" -d "$OUT/" 2>&1 | tee -a "$LOG"
    done
fi

if ls "$OUT"/Anno*.zip 1>/dev/null 2>&1; then
    banner "Unzipping Anno*.zip"
    for z in "$OUT"/Anno*.zip; do
        echo "  unzip $z" | tee -a "$LOG"
        unzip -q -n "$z" -d "$OUT/" 2>&1 | tee -a "$LOG"
    done
fi

banner "DONE"
echo "  Final layout:" | tee -a "$LOG"
find "$OUT" -maxdepth 2 -type d 2>/dev/null | sort | tee -a "$LOG"
echo "  Disk free after:" | tee -a "$LOG"
df -h "$OUT" | tail -2 | tee -a "$LOG"

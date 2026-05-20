# Vertex AI Workbench — Distillation Runbook

## Option A: Vertex AI Workbench (Managed Jupyter with GPU)

### Step 1: Create Workbench Instance

1. Go to: https://console.cloud.google.com/vertex-ai/workbench/instances
2. Click **"Create New"** → **"Advanced Options"**
3. Configure:
   - **Name:** `moda-distill`
   - **Region:** `europe-west4` (or your project's region)
   - **Machine type:** `a2-highgpu-1g` (1x A100 40GB, ~$4/hr)
     - Alternative: `g2-standard-8` (1x L4 24GB, ~$1/hr) — use `--batch-size 128`
   - **GPU:** Should auto-select A100 or L4 based on machine type
   - **Boot disk:** 100 GB SSD
   - **Framework:** PyTorch 2.x (CUDA 12.x)
   - **Idle shutdown:** 60 minutes (saves money if you forget)
4. Click **"Create"** — takes 2-3 minutes to provision

### Step 2: Open JupyterLab

1. Once status shows **"Active"**, click **"Open JupyterLab"**
2. You'll get a full Jupyter environment with GPU access

### Step 3: Upload and Run

Open a **Terminal** in JupyterLab, then:

```bash
# 1. Install dependencies
pip install open_clip_torch datasets Pillow numpy --quiet

# 2. Upload the script (or clone from git)
# Option A: Upload via JupyterLab file browser (drag & drop distill_l16_to_b16_gpu.py)
# Option B: Clone the repo
git clone <your-repo-url> && cd MODA

# 3. Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# 4. Run distillation
python scripts/distill_l16_to_b16_gpu.py \
  --output-dir ./distill_output \
  --batch-size 256 \
  --max-steps 5000 \
  --lr 2e-5 \
  --eval-steps "500,1000,2000,3000,4000,5000"

# For L4 (24GB) GPU, use smaller batch:
# python scripts/distill_l16_to_b16_gpu.py --batch-size 128 --max-steps 5000
```

### Step 4: Monitor

Training prints progress every 50 steps:
```
Step 50/5000 | loss=2.3421 (KL=0.8234 feat=0.4521 NCE=3.2100) | lr=5.00e-06 | 0.77 step/s | ETA 107.1 min
```

Evaluations run at steps 500, 1000, 2000, 3000, 4000, 5000.

### Step 5: Get Results

```bash
# Check best model
cat distill_output/best/metrics.json

# Check summary
cat distill_output/summary.json

# Copy to GCS (optional)
gcloud storage cp -r distill_output/* gs://<your-bucket>/moda/distill_output/
```

### Step 6: Download Best Model Locally

```bash
# On your Mac:
gcloud storage cp gs://<your-bucket>/moda/distill_output/best/student_state_dict.pt \
  ~/Desktop/Hobby/MODA/models/distill-b16-from-l16/best/student_state_dict.pt
```

### Step 7: STOP THE INSTANCE!

**CRITICAL** — A100 costs $4/hr. Once done:
1. Go back to Workbench instances page
2. Click **"Stop"** on your instance
3. Or delete it if you won't need it again

---

## Option B: Google Colab (Simpler, Cheaper)

If Vertex AI Workbench isn't available or you want something simpler:

1. Go to https://colab.research.google.com
2. Create new notebook
3. Runtime → Change runtime type → **T4** (free) or **A100** (Colab Pro)
4. Upload `distill_l16_to_b16_gpu.py` to Colab files
5. Run:

```python
!pip install open_clip_torch datasets Pillow numpy -q

# Upload the script
from google.colab import files
# files.upload()  # or just paste the script content

!python distill_l16_to_b16_gpu.py --batch-size 128 --max-steps 3000 --output-dir /content/distill_output

# Download results
files.download('/content/distill_output/best/student_state_dict.pt')
files.download('/content/distill_output/summary.json')
```

---

## Expected Results

| Setting | GPU | Batch Size | Steps | Time | Cost |
|---------|-----|-----------|-------|------|------|
| A100 40GB | a2-highgpu-1g | 256 | 5000 | ~2 hrs | ~$8 |
| L4 24GB | g2-standard-8 | 128 | 5000 | ~3 hrs | ~$3 |
| T4 16GB (Colab) | — | 64 | 3000 | ~4 hrs | Free/Pro |

Target: Student (B/16, 200M params, 768-d) beats FashionSigLIP (0.5369 on fashion200k).

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| CUDA OOM | Reduce `--batch-size` (try 128 or 64) |
| Dataset download slow | First run downloads ~5GB; subsequent runs use cache |
| Eval takes too long | Reduce `--eval-datasets "fashion200k"` for quick checks |
| No GPU quota | Try a different region or use Colab instead |
| Training diverges (loss NaN) | Reduce `--lr` to 1e-5 or 5e-6 |

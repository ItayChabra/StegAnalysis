#!/usr/bin/env bash
# =============================================================================
#  setup.sh — StegAnalysis Linux environment setup
#  Tested on Ubuntu 22.04 / 24.04 with CUDA 12.x and an A100 GPU.
#
#  Usage:
#    chmod +x setup.sh
#    ./setup.sh
# =============================================================================

set -euo pipefail

echo "======================================================================"
echo "  StegAnalysis — Linux Environment Setup"
echo "======================================================================"

# ── 1. System limits ──────────────────────────────────────────────────────────
# Raise the open-file limit for the current session.  The training loop opens
# thousands of images via ThreadPoolExecutor; the Linux default (1024) can be
# hit on large datasets.
echo "[SYS] Raising open-file limit to 65535 for this session..."
ulimit -n 65535

# Persist the limit across logins (requires sudo).
if command -v sudo &>/dev/null; then
    LIMIT_FILE="/etc/security/limits.d/steganalysis.conf"
    if [[ ! -f "$LIMIT_FILE" ]]; then
        echo "[SYS] Writing $LIMIT_FILE (requires sudo)..."
        sudo tee "$LIMIT_FILE" > /dev/null <<'EOF'
*   soft   nofile   65535
*   hard   nofile   65535
EOF
        echo "[SYS] Limit file written. Re-login for it to take effect globally."
    else
        echo "[SYS] $LIMIT_FILE already exists — skipping."
    fi
fi

# ── 2. Python virtual environment ─────────────────────────────────────────────
if [[ ! -d ".venv" ]]; then
    echo "[VENV] Creating virtual environment in .venv/ ..."
    python3 -m venv .venv
else
    echo "[VENV] .venv/ already exists — skipping creation."
fi

# shellcheck disable=SC1091
source .venv/bin/activate
echo "[VENV] Activated."

# ── 3. Pip upgrade ────────────────────────────────────────────────────────────
pip install --upgrade pip wheel setuptools --quiet

# ── 4. PyTorch with CUDA 12 ───────────────────────────────────────────────────
# Install the CUDA-12 wheel.  Change the index URL if your CUDA version differs:
#   CUDA 11.8 → https://download.pytorch.org/whl/cu118
#   CUDA 12.1 → https://download.pytorch.org/whl/cu121
#   CUDA 12.4 → https://download.pytorch.org/whl/cu124
echo "[PIP] Installing PyTorch (CUDA 12.4)..."
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124 \
    --quiet

# ── 5. Project dependencies ───────────────────────────────────────────────────
echo "[PIP] Installing project dependencies from requirements.txt..."
pip install -r requirements.txt --quiet

# ── 6. Verify GPU access ──────────────────────────────────────────────────────
echo ""
echo "[GPU] Checking CUDA availability..."
python3 - <<'PYEOF'
import torch, sys
if not torch.cuda.is_available():
    print("  WARNING: CUDA not available — training will run on CPU (very slow).")
    sys.exit(0)

print(f"  CUDA available: {torch.cuda.is_available()}")
print(f"  Device count:   {torch.cuda.device_count()}")
print(f"  Device name:    {torch.cuda.get_device_name(0)}")
print(f"  VRAM:           {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Verify TF32 (should be True on Ampere / A100)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
print(f"  TF32 matmul:    {torch.backends.cuda.matmul.allow_tf32}")
print(f"  TF32 cuDNN:     {torch.backends.cudnn.allow_tf32}")

# Verify torch.compile (needs Triton, available on Linux)
try:
    import torch._dynamo
    print(f"  torch.compile:  available (Triton backend)")
except Exception as e:
    print(f"  torch.compile:  NOT available — {e}")
PYEOF

# ── 7. Data folder structure ──────────────────────────────────────────────────
echo ""
echo "[DATA] Creating data folder structure..."
mkdir -p data/raw/flickr30k
mkdir -p "data/raw/BossBase and BOWS2"
mkdir -p data/external/clean
mkdir -p data/external/lsb_grayscale
mkdir -p data/external/lsb_rgb
mkdir -p training/evaluation_results

cat <<'DATAEOF'
[DATA] Folder structure ready.

  Place your images here:
    data/raw/flickr30k/             ← Flickr30k JPEGs  (lossy cover images)
    data/raw/BossBase and BOWS2/    ← BOSSbase PGMs/PNGs (lossless cover images)

  For StegoLSBDatasetTest.py, place external test sets here:
    data/external/clean/            ← clean reference images
    data/external/lsb_grayscale/    ← grayscale LSB stego images
    data/external/lsb_rgb/          ← RGB LSB stego images

DATAEOF

# ── 8. Quick smoke test ───────────────────────────────────────────────────────
echo "[TEST] Running model architecture smoke test..."
python3 - <<'PYEOF'
import torch
import sys
sys.path.insert(0, '.')
from models.srnet import SRNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = SRNet().to(device)

dummy = torch.randn(2, 1, 256, 256, device=device)
with torch.no_grad():
    out = model(dummy)

assert out.shape == (2, 2), f"Unexpected output shape: {out.shape}"
print(f"  SRNet forward pass OK — output shape {tuple(out.shape)} on {device}")

params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"  Parameters: {params:.2f}M")
PYEOF

echo ""
echo "======================================================================"
echo "  Setup complete.  Next steps:"
echo ""
echo "    source .venv/bin/activate"
echo "    python main.py                          # start training"
echo "    python training/evaluate.py             # post-training evaluation"
echo "    python class_demo.py --image <img.pgm>  # interactive demo"
echo "    python StegoLSBDatasetTest.py \\         # cross-dataset test"
echo "           --clean-dir data/external/clean \\"
echo "           --stego-gray-dir data/external/lsb_grayscale \\"
echo "           --stego-rgb-dir  data/external/lsb_rgb"
echo "======================================================================"
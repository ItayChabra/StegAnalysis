"""
FastAPI backend for the SRNet Steganalysis demo.

Endpoints
---------
GET  /health
POST /api/analyze
GET  /api/heatmap/{job_id}
GET  /api/noisemap/{job_id}
POST /api/embed
GET  /api/stego/{job_id}
"""

import io
import math
import os
import sys
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
from torchvision import transforms

# ── Project root on sys.path so imports work when uvicorn runs from api/ ──────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.srnet import SRNet                          # noqa: E402
from generators.unified_generator import UnifiedGenerator  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_SIZE   = 256
WINDOW_STRIDE = 64
THRESH_STEGO  = 0.75
THRESH_SUSP   = 0.50
# Identical to training/config.py LOG_FFT_SCALE — inlined to avoid importing
# the full training module chain.
LOG_FFT_SCALE = math.log1p(256 * 256)
CHECKPOINT    = ROOT / "srnet_finetuned_best.pth"

# Default payload for LSB embedding (demo only)
_DEFAULT_MESSAGE = "This is a secret message " * 4096

# ── Global singletons (loaded once in lifespan) ───────────────────────────────
_model:  Optional[SRNet]             = None
_gen:    Optional[UnifiedGenerator]  = None
_tmpdir: Optional[str]               = None


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _gen, _tmpdir
    _tmpdir = tempfile.mkdtemp(prefix="steganalysis_")
    _model  = SRNet().to(DEVICE)
    ckpt    = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    _model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    _model.eval()
    _gen = UnifiedGenerator()
    print(f"[server] Model loaded on {DEVICE}  |  artefacts → {_tmpdir}")
    yield


app = FastAPI(title="SRNet Steganalysis API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Inference helpers ─────────────────────────────────────────────────────────

def _compute_log_fft(spatial: torch.Tensor) -> torch.Tensor:
    """(1, H, W) spatial → (1, H, W) log-magnitude FFT, matching training."""
    fft_complex   = torch.fft.fft2(spatial)
    fft_shifted   = torch.fft.fftshift(fft_complex, dim=(-2, -1))
    log_magnitude = torch.log1p(torch.abs(fft_shifted))
    return log_magnitude / LOG_FFT_SCALE


_to_tensor = transforms.ToTensor()


def _patch_to_tensor(patch_arr: np.ndarray) -> torch.Tensor:
    """uint8 HxW array → (1, 2, H, W) model-ready tensor on DEVICE."""
    spatial = _to_tensor(Image.fromarray(patch_arr))    # (1, H, W)
    freq    = _compute_log_fft(spatial)                 # (1, H, W)
    return torch.cat([spatial, freq], dim=0).unsqueeze(0).to(DEVICE)


def _run_inference(image: Image.Image) -> dict:
    img_arr = np.array(image.convert("L"), dtype=np.uint8)
    h, w    = img_arr.shape

    # Pad images smaller than one window to 256×256
    pad_h = max(0, WINDOW_SIZE - h)
    pad_w = max(0, WINDOW_SIZE - w)
    if pad_h or pad_w:
        padded         = np.full((h + pad_h, w + pad_w), 128, dtype=np.uint8)
        padded[:h, :w] = img_arr
        img_arr        = padded
        h, w           = img_arr.shape

    rows   = list(range(0, h - WINDOW_SIZE + 1, WINDOW_STRIDE))
    cols   = list(range(0, w - WINDOW_SIZE + 1, WINDOW_STRIDE))
    scores = []

    _model.eval()
    with torch.no_grad():
        for top in rows:
            for left in cols:
                patch = img_arr[top : top + WINDOW_SIZE, left : left + WINDOW_SIZE]
                t     = _patch_to_tensor(patch)
                prob  = torch.softmax(_model(t), dim=1)[0, 1].item()
                scores.append(prob)

    if not scores:
        return {
            "scores": [], "max_score": 0.0, "mean_score": 0.0,
            "flagged": 0, "total": 0, "n_rows": 0, "n_cols": 0,
            "verdict": "CLEAN",
        }

    max_score  = max(scores)
    mean_score = sum(scores) / len(scores)
    flagged    = sum(1 for s in scores if s > THRESH_SUSP)

    if max_score > THRESH_STEGO:
        verdict = "STEGO_DETECTED"
    elif max_score > THRESH_SUSP:
        verdict = "SUSPICIOUS"
    else:
        verdict = "CLEAN"

    return {
        "scores":     scores,
        "max_score":  max_score,
        "mean_score": mean_score,
        "flagged":    flagged,
        "total":      len(scores),
        "n_rows":     len(rows),
        "n_cols":     len(cols),
        "verdict":    verdict,
    }


def _infer_method_hint(verdict: str, scores: list) -> Optional[str]:
    """Heuristic: spread between max and mean hints at LSB vs FFT embedding."""
    if verdict == "CLEAN" or len(scores) < 2:
        return None
    spread = max(scores) - (sum(scores) / len(scores))
    return "lsb_edge" if spread > 0.3 else "fft_mid"


# ── Artefact helpers ──────────────────────────────────────────────────────────

def _job_dir(job_id: str) -> str:
    d = os.path.join(_tmpdir, job_id)
    os.makedirs(d, exist_ok=True)
    return d


def _save_heatmap(
    job_id: str,
    scores: list,
    n_rows: int,
    n_cols: int,
    orig: Image.Image,
) -> None:
    out = os.path.join(_job_dir(job_id), "heatmap.png")
    if not scores or n_rows == 0 or n_cols == 0:
        orig.convert("RGB").save(out)
        return

    grid = np.array(scores, dtype=np.float32).reshape(n_rows, n_cols)
    fig, ax = plt.subplots(
        figsize=(orig.width / 100, orig.height / 100), dpi=100
    )
    ax.imshow(orig.convert("RGB"))
    ax.imshow(
        grid,
        cmap="jet",
        alpha=0.5,
        extent=[0, orig.width, orig.height, 0],
        vmin=0,
        vmax=1,
        aspect="auto",
        interpolation="nearest",
    )
    ax.axis("off")
    plt.tight_layout(pad=0)
    fig.savefig(out, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _save_original(job_id: str, image: Image.Image) -> None:
    out = os.path.join(_job_dir(job_id), "original.png")
    image.convert("RGB").save(out)


def _save_noisemap(job_id: str, width: int, height: int) -> None:
    out  = os.path.join(_job_dir(job_id), "noisemap.png")
    grey = Image.fromarray(np.full((height, width, 3), 128, dtype=np.uint8))
    grey.save(out)


def _save_stego(job_id: str, stego_arr: np.ndarray) -> None:
    out = os.path.join(_job_dir(job_id), "stego.png")
    Image.fromarray(stego_arr.astype(np.uint8)).save(out)


def _get_artefact(job_id: str, filename: str) -> str:
    path = os.path.join(_tmpdir, job_id, filename)
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404, detail={"error": f"Job {job_id!r} not found"}
        )
    return path


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "device": str(DEVICE)}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    data = await file.read()
    try:
        image = Image.open(io.BytesIO(data))
    except Exception:
        raise HTTPException(status_code=400, detail={"error": "Invalid image file"})

    result  = _run_inference(image)
    job_id  = uuid.uuid4().hex
    scores  = result["scores"]
    verdict = result["verdict"]
    orig    = image.convert("RGB")

    _save_original(job_id, image)
    _save_heatmap(job_id, scores, result["n_rows"], result["n_cols"], orig)
    _save_noisemap(job_id, orig.width, orig.height)

    return {
        "job_id":            job_id,
        "verdict":           verdict,
        "confidence":        round(result["max_score"], 4),
        "max_window_score":  round(result["max_score"], 4),
        "mean_window_score": round(result["mean_score"], 4),
        "flagged_windows":   result["flagged"],
        "total_windows":     result["total"],
        "window_rows":       result["n_rows"],
        "window_cols":       result["n_cols"],
        "psnr":              None,
        "method_hint":       _infer_method_hint(verdict, scores),
        "original_url":      f"/api/original/{job_id}",
        "heatmap_url":       f"/api/heatmap/{job_id}",
        "noise_map_url":     f"/api/noisemap/{job_id}",
        "window_scores":     [round(s, 4) for s in scores],
    }


@app.get("/api/heatmap/{job_id}")
async def get_heatmap(job_id: str):
    return FileResponse(_get_artefact(job_id, "heatmap.png"), media_type="image/png")


@app.get("/api/noisemap/{job_id}")
async def get_noisemap(job_id: str):
    return FileResponse(_get_artefact(job_id, "noisemap.png"), media_type="image/png")


_STRATEGY_CONFIGS = {
    "lsb_sequential": {
        "gen_type":       "lsb",
        "strategy":       "sequential",
        "bit_depth":      1,
        "step":           1,
        "edge_threshold": 0,
        "message":        _DEFAULT_MESSAGE,
    },
    "lsb_edge": {
        "gen_type":       "lsb",
        "strategy":       "edge",
        "edge_threshold": 9,
        "bit_depth":      1,
        "message":        _DEFAULT_MESSAGE,
    },
    "dct_mid": {
        "gen_type":         "dct",
        "coeff_selection":  "mid",
        "strength":         3.0,
    },
    "fft_mid": {
        "gen_type":   "fft",
        "freq_band":  "mid",
        "strength":   8.0,
    },
}


@app.post("/api/embed")
async def embed(
    file:     UploadFile = File(...),
    strategy: str        = Form(...),
    capacity: float      = Form(...),
):
    if strategy not in _STRATEGY_CONFIGS:
        raise HTTPException(
            status_code=400,
            detail={"error": f"Unknown strategy {strategy!r}. "
                             f"Valid: {list(_STRATEGY_CONFIGS)}"},
        )

    data = await file.read()
    try:
        image = Image.open(io.BytesIO(data)).convert("L")
    except Exception:
        raise HTTPException(status_code=400, detail={"error": "Invalid image file"})

    config = {**_STRATEGY_CONFIGS[strategy], "capacity_ratio": capacity}
    stego_arr, psnr = _gen.generate_stego(image, None, config)
    if stego_arr is None:
        raise HTTPException(status_code=500, detail={"error": "Embedding failed"})

    job_id          = uuid.uuid4().hex
    pixels_modified = int(image.width * image.height * capacity)
    _save_stego(job_id, stego_arr)

    return {
        "job_id":          job_id,
        "stego_url":       f"/api/stego/{job_id}",
        "psnr":            round(float(psnr), 2) if psnr else None,
        "pixels_modified": pixels_modified,
    }


@app.get("/api/original/{job_id}")
async def get_original(job_id: str):
    return FileResponse(_get_artefact(job_id, "original.png"), media_type="image/png")


@app.get("/api/stego/{job_id}")
async def get_stego(job_id: str):
    return FileResponse(_get_artefact(job_id, "stego.png"), media_type="image/png")
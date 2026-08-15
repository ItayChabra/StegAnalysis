"""
FastAPI backend for the SRNet Steganalysis demo.

Endpoints
---------
GET  /health
POST /api/analyze                 — sliding-window detection + bit-plane scores
POST /api/embed                   — embed a (optionally encrypted) message
POST /api/extract                 — recover a hidden message
POST /api/capacity                — max payload per method for an image
GET  /api/original/{job_id}
GET  /api/stego/{job_id}
GET  /api/heatmap/{job_id}        — suspicion overlay
GET  /api/noisemap/{job_id}       — SRM residual ("what the model sees")
GET  /api/diff/{job_id}           — cover↔stego pixel-diff heatmap (embed flow)
GET  /api/spectrum/{job_id}       — log-FFT magnitude + band rings
GET  /api/bitplane/{job_id}/{n}   — single bit-plane (0=LSB … 7=MSB)
GET  /api/sanitize/{job_id}       — steganography-stripped JPEG download
"""

import base64
import binascii
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
from scipy.signal import convolve2d
from torchvision import transforms

# ── Project root on sys.path so imports work when uvicorn runs from api/ ──────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.srnet import SRNet                               # noqa: E402
from generators.unified_generator import UnifiedGenerator    # noqa: E402
from payload import codec, crypto                            # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_SIZE   = 256
WINDOW_STRIDE = 64    # detection uses every 64-px shift; UI pools to a coarser display grid
THRESH_STEGO  = 0.80
THRESH_SUSP   = 0.50
# Identical to training/config.py LOG_FFT_SCALE — inlined to avoid importing
# the full training module chain.
LOG_FFT_SCALE = math.log1p(256 * 256)
CHECKPOINT    = ROOT / "srnet_finetuned_best.pth"

# SRM "KV" high-pass kernel — the canonical residual the detector keys off.
_KV_KERNEL = np.array([
    [-1,  2,  -2,  2, -1],
    [ 2, -6,   8, -6,  2],
    [-2,  8, -12,  8, -2],
    [ 2, -6,   8, -6,  2],
    [-1,  2,  -2,  2, -1],
], dtype=np.float32) / 12.0

# Per-method defaults for the recoverable embed/extract path.
_LSB_DEFAULTS = dict(strategy="sequential", step=1, bit_depth=1)
_DCT_DEFAULTS = dict(coeff_selection="mid", strength=3.0)
_FFT_DEFAULTS = dict(freq_band="mid", strength=32.0)
_RECOVERABLE  = ("lsb", "dct", "fft")

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


# ── Image helpers ─────────────────────────────────────────────────────────────

def _read_image(data: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(data))
    except Exception:
        raise HTTPException(status_code=400, detail={"error": "Invalid image file"})


def _gray_array(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("L"), dtype=np.uint8)


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
    img_arr = _gray_array(image)
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


# method_hint was a 2-line score-spread heuristic that mis-classified every test
# stego (the detector is binary — clean vs stego — and cannot infer method).
# Removed; do not reintroduce without a true multi-class classifier.


def _bitplane_scores(arr: np.ndarray) -> list:
    """Per-plane structure metrics (bit 0 = LSB). Used for the anomaly overlay."""
    out = []
    for n in range(8):
        plane = (arr >> n) & 1
        p = float(plane.mean())
        entropy = 0.0 if p in (0.0, 1.0) else \
            float(-p * math.log2(p) - (1 - p) * math.log2(1 - p))
        # Adjacent-pixel disagreement rate: ~0.5 = noise-like, lower = structured.
        randomness = float(np.mean(plane[:, 1:] != plane[:, :-1])) if arr.shape[1] > 1 else 0.0
        out.append({
            "plane": n,
            "entropy": round(entropy, 4),
            "randomness": round(randomness, 4),
        })
    return out


# ── Artefact (image) builders ─────────────────────────────────────────────────

def _job_dir(job_id: str) -> str:
    d = os.path.join(_tmpdir, job_id)
    os.makedirs(d, exist_ok=True)
    return d


def _save_gray(job_id: str, name: str, arr: np.ndarray,
               original_rgb: Optional[np.ndarray] = None) -> None:
    if original_rgb is not None:
        # Apply the per-pixel luminance delta to all three RGB channels so the
        # stego image keeps its original colours.
        gray_orig = np.array(
            Image.fromarray(original_rgb).convert("L"), dtype=np.int16)
        delta = arr.astype(np.int16) - gray_orig
        stego_rgb = np.clip(original_rgb.astype(np.int16) + delta[:, :, np.newaxis],
                            0, 255).astype(np.uint8)
        # The extractor reads convert("L"). Clipping a channel at 0/255 (or luma
        # rounding) can shift a pixel's recovered luma by 1 and flip an embedded
        # LSB — a single corrupted header bit breaks the whole reveal. Force any
        # pixel whose round-tripped luma drifts from the target to neutral gray
        # (R=G=B=arr), which luma-converts back exactly. These are sparse (only
        # modified + clipped pixels), so colour survives everywhere it safely can.
        target = arr.astype(np.uint8)
        recon  = np.array(Image.fromarray(stego_rgb, "RGB").convert("L"), dtype=np.uint8)
        bad    = recon != target
        if bad.any():
            stego_rgb[bad] = target[bad][:, np.newaxis]
        Image.fromarray(stego_rgb, "RGB").save(os.path.join(_job_dir(job_id), name))
    else:
        Image.fromarray(arr.astype(np.uint8)).save(os.path.join(_job_dir(job_id), name))


def _save_original(job_id: str, image: Image.Image) -> None:
    image.convert("RGB").save(os.path.join(_job_dir(job_id), "original.png"))


def _save_heatmap(job_id, scores, n_rows, n_cols, orig: Image.Image) -> None:
    out = os.path.join(_job_dir(job_id), "heatmap.png")
    if not scores or n_rows == 0 or n_cols == 0:
        orig.convert("RGB").save(out)
        return
    grid = np.array(scores, dtype=np.float32).reshape(n_rows, n_cols)
    fig, ax = plt.subplots(figsize=(orig.width / 100, orig.height / 100), dpi=100)
    ax.imshow(orig.convert("RGB"))
    ax.imshow(grid, cmap="jet", alpha=0.5,
              extent=[0, orig.width, orig.height, 0],
              vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    ax.axis("off")
    plt.tight_layout(pad=0)
    fig.savefig(out, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _build_srm_residual(arr: np.ndarray, out: str) -> None:
    """Amplified SRM (KV) residual — the high-pass view the detector relies on."""
    res = convolve2d(arr.astype(np.float32), _KV_KERNEL, mode="same", boundary="symm")
    amp = np.clip(np.abs(res) * 12.0, 0, 255).astype(np.uint8)
    Image.fromarray(amp).save(out)


def _build_diff(cover: np.ndarray, stego: np.ndarray, out: str) -> None:
    diff = np.abs(stego.astype(np.int16) - cover.astype(np.int16)).astype(np.float32)
    norm = diff / (diff.max() or 1.0)
    fig, ax = plt.subplots(figsize=(diff.shape[1] / 100, diff.shape[0] / 100), dpi=100)
    ax.imshow(norm, cmap="inferno", vmin=0, vmax=1)
    ax.axis("off")
    plt.tight_layout(pad=0)
    fig.savefig(out, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _build_spectrum(arr: np.ndarray, out: str) -> None:
    f   = np.fft.fftshift(np.fft.fft2(arr.astype(np.float32)))
    mag = np.log1p(np.abs(f))
    mag = mag / (mag.max() or 1.0)
    h, w   = arr.shape
    cy, cx = h // 2, w // 2
    max_r  = math.sqrt(cy ** 2 + cx ** 2)
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    ax.imshow(mag, cmap="magma")
    for frac, color in [(0.15, "#22c55e"), (0.45, "#f59e0b"), (0.75, "#ef4444")]:
        ax.add_patch(plt.Circle((cx, cy), frac * max_r,
                                fill=False, ec=color, lw=1.3, alpha=0.9))
    ax.axis("off")
    plt.tight_layout(pad=0)
    fig.savefig(out, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _get_artefact(job_id: str, filename: str) -> str:
    path = os.path.join(_tmpdir, job_id, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail={"error": f"Job {job_id!r} not found"})
    return path


def _load_gray_artefact(job_id: str, filename: str) -> np.ndarray:
    return np.array(Image.open(_get_artefact(job_id, filename)).convert("L"), dtype=np.uint8)


# ── Recoverable embed/extract plumbing ────────────────────────────────────────

def _recoverable_kwargs(method: str, strategy, step, bit_depth,
                        coeff_selection, strength, freq_band):
    """Build (generator, embed_kwargs, params_bytes, method_id) for a method."""
    if method == "lsb":
        kw = dict(
            strategy=strategy or _LSB_DEFAULTS["strategy"],
            step=int(step) if step is not None else _LSB_DEFAULTS["step"],
            bit_depth=int(bit_depth) if bit_depth is not None else _LSB_DEFAULTS["bit_depth"],
        )
        return _gen.generators["lsb"], kw, codec.pack_lsb_params(**kw), codec.METHOD_LSB
    if method == "dct":
        cs = coeff_selection if coeff_selection in ("mid", "low_mid") else _DCT_DEFAULTS["coeff_selection"]
        kw = dict(coeff_selection=cs,
                  strength=float(strength) if strength is not None else _DCT_DEFAULTS["strength"])
        return _gen.generators["dct"], kw, codec.pack_dct_params(**kw), codec.METHOD_DCT
    if method == "fft":
        kw = dict(freq_band=freq_band or _FFT_DEFAULTS["freq_band"],
                  strength=float(strength) if strength is not None else _FFT_DEFAULTS["strength"])
        return _gen.generators["fft"], kw, codec.pack_fft_params(**kw), codec.METHOD_FFT
    raise HTTPException(status_code=400, detail={"error": f"Method {method!r} is not recoverable"})


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "device": str(DEVICE)}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    data    = await file.read()
    image   = _read_image(data)
    arr     = _gray_array(image)

    result  = _run_inference(image)
    job_id  = uuid.uuid4().hex
    scores  = result["scores"]
    verdict = result["verdict"]
    orig    = image.convert("RGB")

    _save_original(job_id, image)
    _save_heatmap(job_id, scores, result["n_rows"], result["n_cols"], orig)
    _build_srm_residual(arr, os.path.join(_job_dir(job_id), "noisemap.png"))

    return {
        "job_id":            job_id,
        "filename":          file.filename,
        "verdict":           verdict,
        "confidence":        round(result["max_score"], 4),
        "max_window_score":  round(result["max_score"], 4),
        "mean_window_score": round(result["mean_score"], 4),
        "flagged_windows":   result["flagged"],
        "total_windows":     result["total"],
        "window_rows":       result["n_rows"],
        "window_cols":       result["n_cols"],
        "psnr":              None,
        "bitplane_scores":   _bitplane_scores(arr),
        "original_url":      f"/api/original/{job_id}",
        "heatmap_url":       f"/api/heatmap/{job_id}",
        "noise_map_url":     f"/api/noisemap/{job_id}",
        "spectrum_url":      f"/api/spectrum/{job_id}",
        "window_scores":     [round(s, 4) for s in scores],
    }


@app.post("/api/embed")
async def embed(
    file:            UploadFile      = File(...),
    method:          str             = Form("lsb"),
    message:         str             = Form(""),
    cipher:          str             = Form("none"),
    passphrase:      str             = Form(""),
    capacity:        Optional[float] = Form(None),
    # LSB params
    strategy:        Optional[str]   = Form(None),
    step:            Optional[int]   = Form(None),
    bit_depth:       Optional[int]   = Form(None),
    # DCT / FFT params
    coeff_selection: Optional[str]   = Form(None),
    strength:        Optional[float] = Form(None),
    freq_band:       Optional[str]   = Form(None),
):
    method = (method or "lsb").lower()
    data   = await file.read()
    image  = _read_image(data)
    cover  = _gray_array(image)
    # Keep the original RGB array so we can composite the stego back into colour.
    orig_rgb = np.array(image.convert("RGB"), dtype=np.uint8) if image.mode != "L" else None
    job_id = uuid.uuid4().hex

    # ── Adaptive (S-UNIWARD): cost-based noise, carries no recoverable message ──
    if method == "adaptive":
        config = {
            "gen_type": "adaptive", "adaptive_mode": "suniward",
            "capacity_ratio": float(capacity) if capacity is not None else 0.4,
            "canonical": str(coeff_selection).lower() in ("1", "true", "canonical"),
        }
        stego, psnr = _gen.generate_stego(cover, None, config)
        if stego is None:
            raise HTTPException(status_code=500, detail={"error": "Embedding failed"})
        _save_gray(job_id, "original.png", cover)
        _save_gray(job_id, "stego.png", stego)
        return {
            "job_id": job_id, "method": "adaptive", "recoverable": False,
            "cipher": "none", "psnr": round(float(psnr), 2) if psnr else None,
            "stego_url": f"/api/stego/{job_id}", "original_url": f"/api/original/{job_id}",
            "diff_url": f"/api/diff/{job_id}", "spectrum_url": f"/api/spectrum/{job_id}",
            "bitplane_scores": _bitplane_scores(stego),
            "pixels_modified": int(cover.size * config["capacity_ratio"]),
            "note": "Adaptive (S-UNIWARD) embeds statistical noise and carries no recoverable message.",
        }

    # ── SteganoGAN: GAN-learned embedding, carries no recoverable message ───────
    if method == "steganogan":
        stego, psnr = _gen.generate_stego(cover, None, {"gen_type": "steganogan"})
        if stego is None:
            raise HTTPException(status_code=500, detail={"error": "Embedding failed"})
        _save_gray(job_id, "original.png", cover)
        _save_gray(job_id, "stego.png", stego)
        return {
            "job_id": job_id, "method": "steganogan", "recoverable": False,
            "cipher": "none", "psnr": round(float(psnr), 2) if psnr else None,
            "stego_url": f"/api/stego/{job_id}", "original_url": f"/api/original/{job_id}",
            "diff_url": f"/api/diff/{job_id}", "spectrum_url": f"/api/spectrum/{job_id}",
            "bitplane_scores": _bitplane_scores(stego),
            "pixels_modified": int(np.sum(stego.astype(int) != cover.astype(int))),
            "note": "SteganoGAN uses a trained neural network to hide data and carries no recoverable message.",
        }

    # ── Recoverable methods (LSB / DCT / FFT) ──────────────────────────────────
    if method not in _RECOVERABLE:
        raise HTTPException(status_code=400,
                            detail={"error": f"Unknown method {method!r}"})

    try:
        cipher_id = crypto.cipher_id_from_name(cipher)
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"error": str(e)})

    gen, kw, params, method_id = _recoverable_kwargs(
        method, strategy, step, bit_depth, coeff_selection, strength, freq_band)

    try:
        ciphertext, salt, nonce = crypto.encrypt(message.encode("utf-8"), passphrase, cipher_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"error": str(e)})

    bits          = codec.frame(method_id, cipher_id, salt, nonce, params, ciphertext)
    capacity_bits = int(gen.payload_capacity_bits(cover.shape, **kw))
    if len(bits) > capacity_bits:
        max_plain = max(0, capacity_bits // 8 - codec.HEADER_SIZE)
        raise HTTPException(status_code=400, detail={
            "error": "Message too large for this image and settings.",
            "capacity_bytes": capacity_bits // 8,
            "needed_bytes": len(bits) // 8,
            "max_message_bytes": max_plain,
        })

    try:
        stego = gen.embed_payload(cover, bits, **kw)
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"error": str(e)})

    psnr = gen._calculate_psnr(cover, stego)
    _save_original(job_id, image)
    _save_gray(job_id, "stego.png", stego, original_rgb=orig_rgb)

    # Settings the extractor needs (carried through app state / shown in UI).
    extract_hint = {"method": method, "cipher": crypto.CIPHER_NAMES[cipher_id], **kw}

    return {
        "job_id":          job_id,
        "method":          method,
        "cipher":          crypto.CIPHER_NAMES[cipher_id],
        "recoverable":     True,
        "encrypted":       cipher_id != crypto.CIPHER_NONE,
        "psnr":            round(float(psnr), 2) if math.isfinite(psnr) else None,
        "capacity_bytes":  capacity_bits // 8,
        "used_bytes":      len(bits) // 8,
        "message_bytes":   len(ciphertext),
        "pixels_modified": len(bits),
        "extract_hint":    extract_hint,
        "stego_url":       f"/api/stego/{job_id}",
        "original_url":    f"/api/original/{job_id}",
        "diff_url":        f"/api/diff/{job_id}",
        "spectrum_url":    f"/api/spectrum/{job_id}",
        "bitplane_scores": _bitplane_scores(stego),
    }


@app.post("/api/extract")
async def extract(
    file:            UploadFile      = File(...),
    method:          Optional[str]   = Form(None),
    strategy:        Optional[str]   = Form(None),
    step:            Optional[int]   = Form(None),
    bit_depth:       Optional[int]   = Form(None),
    coeff_selection: Optional[str]   = Form(None),
    strength:        Optional[float] = Form(None),
    freq_band:       Optional[str]   = Form(None),
):
    """
    Reveal-only: locate the hidden payload and return the raw ciphertext.

    Decryption is a separate step (``/api/decrypt``) so the UI can show the
    encrypted bytes first and only ask for a passphrase if the data is actually
    encrypted.
    """
    data = await file.read()
    arr  = _gray_array(_read_image(data))

    candidates = [method.lower()] if method else list(_RECOVERABLE)

    for m in candidates:
        gen, kw, _params, _mid = _recoverable_kwargs(
            m, strategy, step, bit_depth, coeff_selection, strength, freq_band)
        try:
            header = codec.decode(lambda n: gen.extract_payload(arr, n, **kw))
        except codec.CodecError:
            continue  # wrong method / params, or no marker — try the next candidate

        cipher_id   = header["cipher_id"]
        ciphertext  = header["ciphertext"]
        encrypted   = cipher_id != crypto.CIPHER_NONE
        response = {
            "method":         codec.METHOD_NAMES.get(header["method_id"], m),
            "cipher":         crypto.CIPHER_NAMES.get(cipher_id, "none"),
            "cipher_id":      cipher_id,
            "encrypted":      encrypted,
            "bytes":          len(ciphertext),
            "ciphertext_b64": base64.b64encode(ciphertext).decode("ascii"),
            "salt_b64":       base64.b64encode(header["salt"]).decode("ascii"),
            "nonce_b64":      base64.b64encode(header["nonce"]).decode("ascii"),
        }
        if not encrypted:
            # No decryption needed — surface the plaintext directly.
            response["message"] = ciphertext.decode("utf-8", errors="replace")
        return response

    raise HTTPException(status_code=404, detail={
        "code": "no_payload",
        "error": "No recoverable message found. Check the method and settings used to embed.",
    })


@app.post("/api/decrypt")
async def decrypt_payload(
    ciphertext_b64: str = Form(...),
    cipher:         str = Form(...),
    passphrase:     str = Form(""),
    salt_b64:       str = Form(""),
    nonce_b64:      str = Form(""),
):
    """Decrypt a ciphertext previously returned by ``/api/extract``."""
    try:
        ciphertext = base64.b64decode(ciphertext_b64, validate=True)
        salt       = base64.b64decode(salt_b64, validate=True)  if salt_b64  else b""
        nonce      = base64.b64decode(nonce_b64, validate=True) if nonce_b64 else b""
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail={"error": "Malformed base64 payload."})

    try:
        cipher_id = crypto.cipher_id_from_name(cipher)
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"error": str(e)})

    try:
        plaintext = crypto.decrypt(ciphertext, passphrase, cipher_id, salt, nonce)
    except crypto.DecryptionError as e:
        raise HTTPException(status_code=422, detail={"code": "bad_key", "error": str(e)})

    return {
        "message":   plaintext.decode("utf-8", errors="replace"),
        "cipher":    crypto.CIPHER_NAMES.get(cipher_id, "none"),
        "encrypted": cipher_id != crypto.CIPHER_NONE,
        "bytes":     len(plaintext),
    }


@app.post("/api/capacity")
async def capacity(
    file:            UploadFile      = File(...),
    method:          str             = Form("lsb"),
    strategy:        Optional[str]   = Form(None),
    step:            Optional[int]   = Form(None),
    bit_depth:       Optional[int]   = Form(None),
    coeff_selection: Optional[str]   = Form(None),
    strength:        Optional[float] = Form(None),
    freq_band:       Optional[str]   = Form(None),
):
    method = (method or "lsb").lower()
    arr    = _gray_array(_read_image(await file.read()))

    if method not in _RECOVERABLE:
        # Adaptive uses entropy-budgeted bpp; report a nominal ceiling.
        bits = int(arr.size * (float(strength) if strength else 0.4))
        return {"method": method, "recoverable": False,
                "capacity_bits": bits, "capacity_bytes": bits // 8,
                "max_message_bytes": 0,
                "width": int(arr.shape[1]), "height": int(arr.shape[0])}

    gen, kw, _p, _m = _recoverable_kwargs(
        method, strategy, step, bit_depth, coeff_selection, strength, freq_band)
    bits = int(gen.payload_capacity_bits(arr.shape, **kw))
    return {
        "method":            method,
        "recoverable":       True,
        "capacity_bits":     bits,
        "capacity_bytes":    bits // 8,
        "header_bytes":      codec.HEADER_SIZE,
        "max_message_bytes": max(0, bits // 8 - codec.HEADER_SIZE),
        "width":             int(arr.shape[1]),
        "height":            int(arr.shape[0]),
    }


@app.get("/api/original/{job_id}")
async def get_original(job_id: str):
    return FileResponse(_get_artefact(job_id, "original.png"), media_type="image/png")


@app.get("/api/stego/{job_id}")
async def get_stego(job_id: str):
    return FileResponse(_get_artefact(job_id, "stego.png"), media_type="image/png")


@app.get("/api/heatmap/{job_id}")
async def get_heatmap(job_id: str):
    return FileResponse(_get_artefact(job_id, "heatmap.png"), media_type="image/png")


@app.get("/api/noisemap/{job_id}")
async def get_noisemap(job_id: str, source: str = "original"):
    """SRM residual. Precomputed for analyze; built on-demand for the embed flow."""
    if source == "original" and os.path.exists(os.path.join(_tmpdir, job_id, "noisemap.png")):
        return FileResponse(_get_artefact(job_id, "noisemap.png"), media_type="image/png")
    name = "stego.png" if source == "stego" else "original.png"
    out  = os.path.join(_job_dir(job_id), f"noisemap_{source}.png")
    if not os.path.exists(out):
        _build_srm_residual(_load_gray_artefact(job_id, name), out)
    return FileResponse(out, media_type="image/png")


@app.get("/api/diff/{job_id}")
async def get_diff(job_id: str):
    """Pixel-diff heatmap between cover and stego (embed flow only)."""
    out = os.path.join(_job_dir(job_id), "diff.png")
    if not os.path.exists(out):
        cover = _load_gray_artefact(job_id, "original.png")
        stego = _load_gray_artefact(job_id, "stego.png")
        _build_diff(cover, stego, out)
    return FileResponse(out, media_type="image/png")


@app.get("/api/spectrum/{job_id}")
async def get_spectrum(job_id: str, source: str = "original"):
    name = "stego.png" if source == "stego" else "original.png"
    out  = os.path.join(_job_dir(job_id), f"spectrum_{source}.png")
    if not os.path.exists(out):
        _build_spectrum(_load_gray_artefact(job_id, name), out)
    return FileResponse(out, media_type="image/png")



@app.get("/api/bitplane/{job_id}/{plane}")
async def get_bitplane(job_id: str, plane: int, source: str = "original"):
    if not 0 <= plane <= 7:
        raise HTTPException(status_code=400, detail={"error": "plane must be 0–7"})
    name = "stego.png" if source == "stego" else "original.png"
    out  = os.path.join(_job_dir(job_id), f"bitplane_{source}_{plane}.png")
    if not os.path.exists(out):
        arr = _load_gray_artefact(job_id, name)
        Image.fromarray((((arr >> plane) & 1) * 255).astype(np.uint8)).save(out)
    return FileResponse(out, media_type="image/png")


@app.get("/api/sanitize/{job_id}")
async def get_sanitize(job_id: str):
    """
    Return a steganography-stripped version of the original image as a
    downloadable PNG.

    Strategy: Gaussian blur (σ=0.7) then round, preserving the original
    colour mode (RGB or grayscale).

    Gaussian blur rather than LSB zeroing + random re-seed: the model's SRM
    high-pass filters detect *random* LSBs.  A clean image has LSBs correlated
    with local structure; the blur reintroduces that correlation rather than
    injecting independent noise.
    """
    from scipy.ndimage import gaussian_filter
    out = os.path.join(_job_dir(job_id), "sanitized.png")
    if not os.path.exists(out):
        src = os.path.join(_job_dir(job_id), "original.png")
        if not os.path.exists(src):
            raise HTTPException(status_code=404, detail={"error": "job not found", "code": "not_found"})
        img = Image.open(src)
        if img.mode == "L":
            arr = np.array(img, dtype=np.float32)
            blurred = gaussian_filter(arr, sigma=0.7)
            sanitized = np.clip(np.round(blurred), 0, 255).astype(np.uint8)
            Image.fromarray(sanitized, mode="L").save(out, format="PNG")
        else:
            arr = np.array(img.convert("RGB"), dtype=np.float32)
            blurred = np.stack(
                [gaussian_filter(arr[:, :, c], sigma=0.7) for c in range(3)], axis=2)
            sanitized = np.clip(np.round(blurred), 0, 255).astype(np.uint8)
            Image.fromarray(sanitized, mode="RGB").save(out, format="PNG")
    fname = f"sanitized_{job_id[:8]}.png"
    return FileResponse(
        out,
        media_type="image/png",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )

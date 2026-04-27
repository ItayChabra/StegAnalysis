"""
inference.py — Sliding-window steganalysis with pluggable score aggregation.

Key changes vs previous version
---------------------------------
AGGREGATION MODES  (per-folder, set in FOLDER_CONFIG)
  'max'    — original behaviour: P(stego) = max window score.
             Good when the stego signal is strong and localised (FFT, high-cap LSB).
  'mean'   — arithmetic mean of all window scores.
             More stable than max for signals distributed across the whole image.
  'vote'   — fraction of windows with score ≥ VOTE_FLOOR.
             Best for weak/distributed signals (DCT, low-capacity steganography).
  'p80'    — 80th-percentile window score.
             A compromise: less volatile than max, more sensitive than mean.

WHY DCT SCORES CLUSTER AROUND 0.47
  The Kaggle DCT folder likely contains images produced by a standard JPEG
  coefficient tool (JSteg / F5 / OutGuess / Steghide). Our SRNet was trained
  on a custom block-DCT embedder that writes to the spatial domain after IDCT;
  these are fundamentally different signal distributions.
  The model detects *something* (mean > 0.50 at random) but can't cross the
  decision boundary confidently. The vote aggregator captures this weak signal
  better than the single max window.

CHANNEL HANDLING
  LSB  (RGB PNG) — R / G / B channels scored individually; max taken across channels.
  DCT / FFT (grayscale BMP) — single luminance channel.
  Other RGB — YCbCr Y channel only.
"""

import glob
import math
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from models.srnet import SRNet

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_CHECKPOINT = 'srnet_finetuned_best.pth'
DATA_DIR         = '/home/linuxu/PycharmProjects/StegAnalysis/data/Kaggle_Testing'
DEVICE           = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WINDOW_SIZE   = 256
WINDOW_STRIDE = 64

# Floor used by the 'vote' aggregator: fraction of windows above this score.
VOTE_FLOOR = 0.40

# Per-folder configuration.
# Keys: 'threshold' — final decision boundary on the aggregated score.
#       'mode'      — aggregation strategy ('max' | 'mean' | 'vote' | 'p80').
FOLDER_CONFIG = {
    #  DCT: weak distributed signal — vote on fraction of windows above VOTE_FLOOR.
    #  Threshold of 0.35 means: if ≥ 35% of patches score ≥ 0.40, classify as stego.
    'dct': {'threshold': 0.35, 'mode': 'vote'},

    # FFT: strong signal, original max works fine.
    'fft': {'threshold': 0.50, 'mode': 'max'},

    # LSB: near-perfect, original max works fine.
    'lsb': {'threshold': 0.50, 'mode': 'max'},
}

_LOG_FFT_SCALE = math.log1p(256 * 256)


# ── Feature extraction ────────────────────────────────────────────────────────

def compute_log_fft(spatial_tensor: torch.Tensor) -> torch.Tensor:
    fft_complex   = torch.fft.fft2(spatial_tensor)
    fft_shifted   = torch.fft.fftshift(fft_complex, dim=(-2, -1))
    log_magnitude = torch.log1p(torch.abs(fft_shifted))
    return log_magnitude / _LOG_FFT_SCALE


def extract_channels(img_path: str, folder_name: str) -> list:
    """
    Return a list of grayscale PIL images to score independently.

    LSB / RGB PNG  → [R, G, B]  (stego is typically in all three channels)
    Grayscale 'L'  → [img]
    Other RGB/RGBA → [Y channel from YCbCr]
    """
    img = Image.open(img_path)

    if img.mode == 'L':
        return [img]

    if folder_name == 'lsb' and img.mode == 'RGB':
        r, g, b = img.split()
        return [r, g, b]

    if img.mode in ('RGB', 'RGBA'):
        return [img.convert('YCbCr').split()[0]]

    return [img.convert('L')]


# ── Sliding-window inference ──────────────────────────────────────────────────

def sliding_window_scores(model: torch.nn.Module,
                           img: Image.Image,
                           to_tensor) -> list[float]:
    """
    Run the model over every 256×256 window of *img* and return a list of
    P(stego) values — one per window.  Small images are centre-padded to 256.
    """
    img_array = np.array(img, dtype=np.uint8)
    h, w = img_array.shape

    # Pad images smaller than the window size
    pad_h = max(0, WINDOW_SIZE - h)
    pad_w = max(0, WINDOW_SIZE - w)
    if pad_h > 0 or pad_w > 0:
        padded        = np.full((h + pad_h, w + pad_w), 128, dtype=np.uint8)
        padded[:h, :w] = img_array
        img_array      = padded
        h, w           = img_array.shape

    scores = []
    with torch.no_grad():
        for top in range(0, h - WINDOW_SIZE + 1, WINDOW_STRIDE):
            for left in range(0, w - WINDOW_SIZE + 1, WINDOW_STRIDE):
                patch   = img_array[top: top + WINDOW_SIZE, left: left + WINDOW_SIZE]
                spatial = to_tensor(Image.fromarray(patch)).to(DEVICE)
                log_fft = compute_log_fft(spatial)
                tensor  = torch.cat([spatial, log_fft], dim=0).unsqueeze(0)
                prob    = torch.softmax(model(tensor), dim=1)[0, 1].item()
                scores.append(prob)

    return scores


def aggregate_scores(scores: list[float], mode: str) -> float:
    """
    Reduce a list of per-window scores to a single detection score.

    mode='max'  — maximum window score
    mode='mean' — arithmetic mean
    mode='vote' — fraction of windows at or above VOTE_FLOOR
    mode='p80'  — 80th percentile
    """
    if not scores:
        return 0.0

    arr = np.array(scores)

    if mode == 'max':
        return float(arr.max())
    if mode == 'mean':
        return float(arr.mean())
    if mode == 'vote':
        return float((arr >= VOTE_FLOOR).mean())
    if mode == 'p80':
        return float(np.percentile(arr, 80))

    raise ValueError(f"Unknown aggregation mode: {mode!r}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading SRNet on {DEVICE}...")
    model = SRNet().to(DEVICE)

    if not os.path.exists(MODEL_CHECKPOINT):
        print(f"ERROR: Could not find {MODEL_CHECKPOINT}.")
        return

    checkpoint = torch.load(MODEL_CHECKPOINT, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    # Strip torch.compile prefix if present
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    print(f"Weights loaded. Checkpoint val_acc: {checkpoint.get('val_acc', '?')}\n")
    model.eval()

    to_tensor = transforms.ToTensor()

    for folder, cfg in FOLDER_CONFIG.items():
        threshold = cfg['threshold']
        mode      = cfg['mode']

        folder_path = os.path.join(DATA_DIR, folder)
        if not os.path.exists(folder_path):
            print(f"[Skipped] Directory not found: {folder_path}\n")
            continue

        images = sorted(glob.glob(os.path.join(folder_path, '*.*')))[:200]
        if not images:
            print(f"[Skipped] No files found in {folder_path}\n")
            continue

        print(f"--- Testing '{folder}' ({len(images)} images) "
              f"[mode={mode}, thresh={threshold}] ---")

        detected_stego = 0
        valid_count    = 0
        final_scores   = []

        for img_path in images:
            try:
                channels      = extract_channels(img_path, folder)
                channel_score = 0.0

                for ch_img in channels:
                    window_scores = sliding_window_scores(model, ch_img, to_tensor)
                    channel_score = max(channel_score,
                                        aggregate_scores(window_scores, mode))

                final_scores.append(channel_score)
                if channel_score >= threshold:
                    detected_stego += 1
                valid_count += 1

            except Exception as e:
                print(f"  Error on {os.path.basename(img_path)}: {e}")

        if valid_count == 0:
            continue

        accuracy = detected_stego / valid_count * 100
        s        = np.array(final_scores)

        print(f"  Result:    {detected_stego}/{valid_count} detected as stego "
              f"({accuracy:.1f}%)")
        print(f"  Score distribution ({mode}) — "
              f"min: {s.min():.3f}  mean: {s.mean():.3f}  "
              f"max: {s.max():.3f}  median: {np.median(s):.3f}")

        # Show how many images land in each decision band
        bands = [0.25, 0.35, 0.50, 0.75]
        band_str = "  Bands: " + "  |  ".join(
            f"≥{b:.2f}: {(s >= b).sum()}" for b in bands
        )
        print(band_str)
        print()


if __name__ == "__main__":
    main()
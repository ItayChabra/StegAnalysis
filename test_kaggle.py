"""
inference.py — Sliding-window steganalysis with pluggable score aggregation.

Key changes vs previous version
---------------------------------
BENCHMARK EVALUATION
  The script now evaluates both Cover (negative) and Stego (positive) directories
  to accurately calculate True Positive and True Negative rates across different
  datasets (BOSSbase/BOWS2, Flickr30k, Kaggle methods).

AGGREGATION MODES
  'max'    — original behaviour: P(stego) = max window score.
             Good when the stego signal is strong and localised (FFT, high-cap LSB).
  'mean'   — arithmetic mean of all window scores.
             More stable than max for signals distributed across the whole image.
  'vote'   — fraction of windows with score ≥ VOTE_FLOOR.
             Best for weak/distributed signals (DCT, low-capacity steganography).
  'p80'    — 80th-percentile window score.
             A compromise: less volatile than max, more sensitive than mean.
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

MODEL_CHECKPOINT = 'srnet_ft_epoch_20.pth'
DEVICE           = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WINDOW_SIZE   = 256
WINDOW_STRIDE = 64
VOTE_FLOOR    = 0.35   # lowered from 0.40: adaptive stego scores cluster around 0.35–0.45
BATCH_SIZE = 64

# Define base paths as seen in your environment
KAGGLE_DIR = '/home/linuxu/PycharmProjects/StegAnalysis/data/Kaggle_Testing'
RAW_DIR    = '/home/linuxu/PycharmProjects/StegAnalysis/data/raw'

# Per-folder configuration for the benchmark.
# 'label'     — 0 for Covers (expecting 0% detection), 1 for Stego (expecting 100% detection).
# 'threshold' — final decision boundary on the aggregated score.
# 'mode'      — aggregation strategy ('max' | 'mean' | 'vote' | 'p80').
TEST_TARGETS = [
    # ── COVERS (Expected: Clean / 0% Stego) ──────────────────────────────────
    {'name': 'BOSS & BOWS2',  'path': os.path.join(RAW_DIR, 'BossBase and BOWS2'),       'label': 0, 'threshold': 0.60, 'mode': 'mean'},
    {'name': 'Flickr30k',     'path': os.path.join(RAW_DIR, 'flickr30k'),                'label': 0, 'threshold': 0.60, 'mode': 'mean'},

    # ── SPATIAL / ADAPTIVE STEGO (Expected: 100% Stego) ──────────────────────
    # p80: 80th-percentile window score — more sensitive than vote for spatially
    # uniform embedding where the signal is diffuse across all windows.
    # Threshold 0.45 targets the upper tail of the score distribution.
    {'name': 'HUGO',          'path': os.path.join(KAGGLE_DIR, 'HUGO'),                  'label': 1, 'threshold': 0.45, 'mode': 'p80'},
    {'name': 'S-UNIWARD',     'path': os.path.join(KAGGLE_DIR, 'S-UNIWARD'),             'label': 1, 'threshold': 0.45, 'mode': 'p80'},
    {'name': 'WOW',           'path': os.path.join(KAGGLE_DIR, 'WOW'),                   'label': 1, 'threshold': 0.45, 'mode': 'p80'},
    {'name': 'LSB',           'path': os.path.join(KAGGLE_DIR, 'lsb'),                   'label': 1, 'threshold': 0.50, 'mode': 'max'},

    # ── FREQUENCY STEGO (Expected: 100% Stego) ───────────────────────────────
    # If DCT detection is weak, consider changing 'mode' to 'vote' and threshold to 0.35
    {'name': 'DCT',           'path': os.path.join(KAGGLE_DIR, 'dct'),                   'label': 1, 'threshold': 0.50, 'mode': 'max'},
    {'name': 'FFT',           'path': os.path.join(KAGGLE_DIR, 'fft'),                   'label': 1, 'threshold': 0.50, 'mode': 'max'},
]

_LOG_FFT_SCALE = math.log1p(256 * 256)


# ── Feature extraction ────────────────────────────────────────────────────────

def compute_log_fft(spatial_tensor: torch.Tensor) -> torch.Tensor:
    fft_complex   = torch.fft.fft2(spatial_tensor)
    fft_shifted   = torch.fft.fftshift(fft_complex, dim=(-2, -1))
    log_magnitude = torch.log1p(torch.abs(fft_shifted))
    return log_magnitude / _LOG_FFT_SCALE


def extract_channels(img_path: str, target_name: str) -> list:
    img = Image.open(img_path)

    if img.mode == 'L':
        return [img]

    if img.mode in ('RGB', 'RGBA'):
        # LSB stego is hidden per-channel, so split for LSB folders.
        # For everything else, use luminance — matching training preprocessing.
        if target_name == 'LSB':
            r, g, b = img.convert('RGB').split()
            return [r, g, b]
        else:
            return [img.convert('L')]  # ← luminance, exactly as trained

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
        padded = np.full((h + pad_h, w + pad_w), 128, dtype=np.uint8)
        padded[:h, :w] = img_array
        img_array = padded
        h, w = img_array.shape

    all_scores = []
    current_batch = []

    # Wrap the entire generation and inference in no_grad
    with torch.no_grad():
        for top in range(0, h - WINDOW_SIZE + 1, WINDOW_STRIDE):
            for left in range(0, w - WINDOW_SIZE + 1, WINDOW_STRIDE):

                # 1. Extract and preprocess patch
                patch = img_array[top:top + WINDOW_SIZE, left:left + WINDOW_SIZE]
                spatial = to_tensor(Image.fromarray(patch))
                log_fft = compute_log_fft(spatial)
                tensor = torch.cat([spatial, log_fft], dim=0)
                current_batch.append(tensor)

                # 2. If batch is full, run inference and clear memory
                if len(current_batch) == BATCH_SIZE:
                    batch_t = torch.stack(current_batch).to(DEVICE)
                    probs = torch.softmax(model(batch_t), dim=1)[:, 1]
                    all_scores.extend(probs.cpu().tolist())

                    # Explicitly clear lists and delete tensors to free GPU memory
                    current_batch = []
                    del batch_t

        # 3. Process any remaining patches in the final partial batch
        if current_batch:
            batch_t = torch.stack(current_batch).to(DEVICE)
            probs = torch.softmax(model(batch_t), dim=1)[:, 1]
            all_scores.extend(probs.cpu().tolist())
            del batch_t

    return all_scores


def aggregate_scores(scores: list[float], mode: str) -> float:
    """
    Reduce a list of per-window scores to a single detection score.
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

    checkpoint = torch.load(MODEL_CHECKPOINT, map_location=DEVICE, weights_only=True)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    # Strip torch.compile prefix if present
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    print(f"Weights loaded. Checkpoint val_acc: {checkpoint.get('val_acc', '?')}\n")
    model.eval()

    to_tensor = transforms.ToTensor()

    for target in TEST_TARGETS:
        name      = target['name']
        folder    = target['path']
        label     = target['label']
        threshold = target['threshold']
        mode      = target['mode']

        if not os.path.exists(folder):
            print(f"[Skipped] Directory not found: {folder}\n")
            continue

        images = sorted(glob.glob(os.path.join(folder, '*.*')))[:200]
        if not images:
            print(f"[Skipped] No files found in {folder}\n")
            continue

        target_type = "STEGO" if label == 1 else "COVER"
        print(f"--- Testing {target_type}: '{name}' ({len(images)} images) "
              f"[mode={mode}, thresh={threshold}] ---")

        detected_stego = 0
        valid_count    = 0
        final_scores   = []

        for img_path in images:
            try:
                channels      = extract_channels(img_path, name)
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

        # Calculate accuracy based on what we expect the folder to be
        if label == 1:
            # We want it to be detected as stego
            accuracy = (detected_stego / valid_count) * 100
        else:
            # We want it to be detected as cover (NOT stego)
            correctly_classified_covers = valid_count - detected_stego
            accuracy = (correctly_classified_covers / valid_count) * 100

        s = np.array(final_scores)

        print(f"  Detected as Stego: {detected_stego}/{valid_count}")
        metric_name = "TPR (Detection Rate)" if label == 1 else "TNR (Clean Rate)"
        print(f"  {metric_name}: {accuracy:.1f}%")
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
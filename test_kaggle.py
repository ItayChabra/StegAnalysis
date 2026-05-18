"""
test_kaggle.py — Sliding-window steganalysis benchmark with mode comparison.

Every image is reduced to luminance and scored once with the sliding window.
The raw per-window scores are kept, then EVERY aggregation mode (max / mean /
vote / p80) is derived from that single inference pass and run through a
threshold sweep — so the deployable (mode, threshold) pair can be picked from
real data instead of guessed.

AGGREGATION MODES (aggregate_scores)
  'max'  — highest window score. Best for uniform stego (LSB/DCT/FFT); spikes
           false positives on textured covers.
  'mean' — arithmetic mean. Lowest cover scores; dilutes localised stego.
  'vote' — fraction of windows >= VOTE_FLOOR.
  'p80'  — 80th-percentile window score. Middle ground.
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

MODEL_CHECKPOINT = 'srnet_ft_epoch_15.pth'
DEVICE           = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# A deployable detector uses ONE aggregation mode and ONE threshold for every
# image. This script scores once, then compares all modes so the best pair can
# be read off the threshold sweep.
AGG_MODES = ['max', 'mean', 'vote', 'p80']

WINDOW_SIZE   = 256
WINDOW_STRIDE = 64
VOTE_FLOOR    = 0.35   # lowered from 0.40: adaptive stego scores cluster around 0.35–0.45
BATCH_SIZE    = 64

# Define base paths as seen in your environment
KAGGLE_DIR = '/home/linuxu/PycharmProjects/StegAnalysis/data/Kaggle_Testing'
RAW_DIR    = '/home/linuxu/PycharmProjects/StegAnalysis/data/raw'

# Per-folder configuration for the benchmark.
# 'label' — 0 for Covers (expecting 0% detection), 1 for Stego (expecting 100%).
# 'group' — 'cover' | 'basic' (LSB/DCT/FFT) | 'adaptive' (WOW/S-UNIWARD/HUGO);
#           used to break the threshold sweep down by difficulty class.
TEST_TARGETS = [
    # ── COVERS (Expected: Clean / 0% Stego) ──────────────────────────────────
    {'name': 'BOSS & BOWS2',  'path': os.path.join(RAW_DIR, 'BossBase and BOWS2'),  'label': 0, 'group': 'cover'},
    {'name': 'Flickr30k',     'path': os.path.join(RAW_DIR, 'flickr30k'),           'label': 0, 'group': 'cover'},

    # ── SPATIAL / ADAPTIVE STEGO (Expected: 100% Stego) ──────────────────────
    {'name': 'HUGO',          'path': os.path.join(KAGGLE_DIR, 'HUGO'),             'label': 1, 'group': 'adaptive'},
    {'name': 'S-UNIWARD',     'path': os.path.join(KAGGLE_DIR, 'S-UNIWARD'),        'label': 1, 'group': 'adaptive'},
    {'name': 'WOW',           'path': os.path.join(KAGGLE_DIR, 'WOW'),              'label': 1, 'group': 'adaptive'},

    # ── BASIC STEGO (Expected: 100% Stego) ───────────────────────────────────
    {'name': 'LSB',           'path': os.path.join(KAGGLE_DIR, 'lsb'),              'label': 1, 'group': 'basic'},
    {'name': 'DCT',           'path': os.path.join(KAGGLE_DIR, 'dct'),              'label': 1, 'group': 'basic'},
    {'name': 'FFT',           'path': os.path.join(KAGGLE_DIR, 'fft'),              'label': 1, 'group': 'basic'},
]

_LOG_FFT_SCALE = math.log1p(256 * 256)


# ── Feature extraction ────────────────────────────────────────────────────────

def compute_log_fft(spatial_tensor: torch.Tensor) -> torch.Tensor:
    fft_complex   = torch.fft.fft2(spatial_tensor)
    fft_shifted   = torch.fft.fftshift(fft_complex, dim=(-2, -1))
    log_magnitude = torch.log1p(torch.abs(fft_shifted))
    return log_magnitude / _LOG_FFT_SCALE


def load_luminance(img_path: str) -> Image.Image:
    """Return the image as luminance — the single channel SRNet is trained on.

    SRNet is trained exclusively on grayscale/luminance patches. Feeding raw
    R/G/B planes of a genuine colour image is out-of-distribution — the model
    reads demosaicing/chroma artifacts as embedding noise, which collapsed
    Flickr30k cover TNR to 3%. Every image, grayscale or colour, is reduced to
    luminance ('L' images convert as a no-op).
    """
    return Image.open(img_path).convert('L')


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


# ── Reporting ─────────────────────────────────────────────────────────────────

def aggregate_per_image(results: dict, mode: str) -> dict:
    """Collapse each image's raw window scores into one score for *mode*.

    results[name] = {'windows': [..per-image lists..], 'label', 'group'}
    -> {name: {'scores': [..per-image..], 'label', 'group'}}
    """
    return {
        name: {'scores': [aggregate_scores(w, mode) for w in r['windows']],
               'label': r['label'], 'group': r['group']}
        for name, r in results.items()
    }


def sweep_rows(scored: dict) -> list:
    """Sweep the decision threshold and return rows of
    (thresh, TNR, TPR_basic, TPR_adapt, TPR_all, balanced_acc, Youden_J)."""
    def pool(group):
        return np.array([s for r in scored.values()
                         if r['group'] == group for s in r['scores']])

    cover    = pool('cover')
    basic    = pool('basic')
    adaptive = pool('adaptive')
    parts    = [a for a in (basic, adaptive) if a.size]
    stego    = np.concatenate(parts) if parts else np.array([])

    if not cover.size or not stego.size:
        return []

    rows = []
    for t in np.arange(0.30, 0.8001, 0.05):
        t     = round(float(t), 3)
        tnr   = float((cover    <  t).mean() * 100)
        t_bas = float((basic    >= t).mean() * 100) if basic.size    else 0.0
        t_ada = float((adaptive >= t).mean() * 100) if adaptive.size else 0.0
        t_all = float((stego    >= t).mean() * 100)
        rows.append((t, tnr, t_bas, t_ada, t_all,
                     (tnr + t_all) / 2, tnr + t_all - 100))
    return rows


def report_mode(results: dict, mode: str):
    """Print per-target score distributions and the threshold sweep for *mode*.
    Returns (mode, best_row) where best_row maximises balanced accuracy."""
    scored = aggregate_per_image(results, mode)

    print("=" * 80)
    print(f"AGGREGATION MODE: {mode}")
    print("=" * 80)
    for name, r in scored.items():
        s   = np.array(r['scores'])
        tag = "STEGO" if r['label'] == 1 else "COVER"
        print(f"  {tag} {name:<14} n={len(s):>3}  "
              f"min {s.min():.3f}  median {np.median(s):.3f}  "
              f"mean {s.mean():.3f}  max {s.max():.3f}")

    rows = sweep_rows(scored)
    if not rows:
        print("  [sweep skipped — need both cover and stego results]")
        return mode, None

    best = max(rows, key=lambda r: r[5])
    print(f"\n  {'thresh':>7} | {'TNR':>7} | {'TPR basic':>10} | "
          f"{'TPR adapt':>10} | {'TPR all':>8} | {'bal-acc':>8} | {'Youden':>7}")
    print("  " + "-" * 74)
    for (t, tnr, t_bas, t_ada, t_all, bal, yj) in rows:
        mark = '  <-- best bal-acc' if t == best[0] else ''
        print(f"  {t:>7.2f} | {tnr:>6.1f}% | {t_bas:>9.1f}% | {t_ada:>9.1f}% | "
              f"{t_all:>7.1f}% | {bal:>7.1f}% | {yj:>6.1f}{mark}")
    print()
    return mode, best


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
    print(f"Weights loaded. Checkpoint val_acc: {checkpoint.get('val_acc', '?')}")
    print(f"Comparing aggregation modes: {', '.join(AGG_MODES)}\n")
    model.eval()

    to_tensor = transforms.ToTensor()
    results = {}

    # ── Single inference pass: store raw per-window scores for every image ───
    for target in TEST_TARGETS:
        name, folder = target['name'], target['path']
        label, group = target['label'], target['group']

        if not os.path.exists(folder):
            print(f"[Skipped] Directory not found: {folder}")
            continue

        images = sorted(glob.glob(os.path.join(folder, '*.*')))[:200]
        if not images:
            print(f"[Skipped] No files found in {folder}")
            continue

        target_type = "STEGO" if label == 1 else "COVER"
        print(f"Scoring {target_type:<5} '{name}' ({len(images)} images)...")

        per_image_windows = []
        for img_path in images:
            try:
                lum = load_luminance(img_path)
                per_image_windows.append(
                    sliding_window_scores(model, lum, to_tensor))
            except Exception as e:
                print(f"  Error on {os.path.basename(img_path)}: {e}")

        if per_image_windows:
            results[name] = {'windows': per_image_windows,
                             'label': label, 'group': group}

    if not results:
        print("\nNo results — nothing to report.")
        return

    # ── Compare every aggregation mode from the same inference pass ──────────
    print()
    summaries = [report_mode(results, mode) for mode in AGG_MODES]

    valid = [(m, b) for (m, b) in summaries if b is not None]
    if valid:
        print("#" * 80)
        print("BEST OPERATING POINT PER MODE  (ranked by balanced accuracy)")
        print("#" * 80)
        print(f"  {'mode':>6} | {'thresh':>7} | {'TNR':>7} | {'TPR basic':>10} | "
              f"{'TPR adapt':>10} | {'bal-acc':>8}")
        print("  " + "-" * 64)
        for (m, b) in sorted(valid, key=lambda mb: mb[1][5], reverse=True):
            print(f"  {m:>6} | {b[0]:>7.2f} | {b[1]:>6.1f}% | {b[2]:>9.1f}% | "
                  f"{b[3]:>9.1f}% | {b[5]:>7.1f}%")
        best_mode, best_row = max(valid, key=lambda mb: mb[1][5])
        print("  " + "-" * 64)
        print(f"  WINNER: mode={best_mode}  threshold={best_row[0]:.2f}  "
              f"bal-acc={best_row[5]:.1f}%")
        print("#" * 80)


if __name__ == "__main__":
    main()
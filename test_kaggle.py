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

import argparse
import glob
import math
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from models.srnet import SRNet

# ── Configuration ─────────────────────────────────────────────────────────────

# Defaults — both overridable on the command line (--checkpoint / --images).
MODEL_CHECKPOINT = 'srnet_best_val.pth'
N_IMAGES         = 200   # images scored per folder; lower = faster checkpoint sweep
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
# 'group' — 'cover' | 'basic' (LSB/DCT/FFT) | 'adaptive' (S-UNIWARD);
#           used to break the threshold sweep down by difficulty class.
NEW_SUNI_DIR = os.path.join(KAGGLE_DIR, 'New_S-UNIWARD')

TEST_TARGETS = [
    # ── COVERS (Expected: Clean / 0% Stego) ──────────────────────────────────
    {'name': 'BOSS & BOWS2',  'path': os.path.join(RAW_DIR, 'BossBase and BOWS2'),  'label': 0, 'group': 'cover'},
    {'name': 'Flickr30k',     'path': os.path.join(RAW_DIR, 'flickr30k'),           'label': 0, 'group': 'cover'},
    {'name': 'BOSSbase_256',  'path': os.path.join(NEW_SUNI_DIR, 'BOSSbase_256'),   'label': 0, 'group': 'cover'},

    # ── ADAPTIVE STEGO — held-out S-UNIWARD, BOSSbase 256×256, 0.2/0.4 bpp ────
    {'name': 'S-UNIWARD 0.2', 'path': os.path.join(NEW_SUNI_DIR, 'SUNI_02'),        'label': 1, 'group': 'adaptive'},
    {'name': 'S-UNIWARD 0.4', 'path': os.path.join(NEW_SUNI_DIR, 'SUNI_04'),        'label': 1, 'group': 'adaptive'},

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
    (thresh, TNR, TPR_basic, TPR_adapt, bal_acc, Youden_J, TPR_lsb, TPR_dct, TPR_fft).

    Balanced accuracy is basic-driven: (TNR + TPR_basic) / 2.
    Adaptive is shown as an informational column only.
    """
    def get_target_scores(target_name):
        if target_name in scored:
            return np.array(scored[target_name]['scores'])
        return np.array([])

    cover_pool = np.array([s for r in scored.values() if r['group'] == 'cover' for s in r['scores']])
    basic_pool = np.array([s for r in scored.values() if r['group'] == 'basic' for s in r['scores']])
    s_lsb  = get_target_scores('LSB')
    s_dct  = get_target_scores('DCT')
    s_fft  = get_target_scores('FFT')
    s_suni = np.concatenate([get_target_scores('S-UNIWARD 0.2'), get_target_scores('S-UNIWARD 0.4')])

    if not cover_pool.size or not basic_pool.size:
        return []

    rows = []
    for t in np.arange(0.30, 0.8001, 0.05):
        t     = round(float(t), 3)
        tnr   = float((cover_pool <  t).mean() * 100)
        t_bas = float((basic_pool >= t).mean() * 100)
        t_ada = float((s_suni     >= t).mean() * 100) if s_suni.size else 0.0
        t_lsb = float((s_lsb     >= t).mean() * 100) if s_lsb.size  else 0.0
        t_dct = float((s_dct     >= t).mean() * 100) if s_dct.size  else 0.0
        t_fft = float((s_fft     >= t).mean() * 100) if s_fft.size  else 0.0
        bal   = (tnr + t_bas) / 2
        yj    = tnr + t_bas - 100
        rows.append((t, tnr, t_bas, t_ada, bal, yj, t_lsb, t_dct, t_fft))
    return rows


def report_mode(results: dict, mode: str):
    """Print per-target score distributions and the threshold sweep for *mode*.
    Returns (mode, (thresh, tnr, t_bas, t_ada, bal)) where bal = (TNR+TPR_basic)/2."""
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

    best = max(rows, key=lambda r: r[4])
    print(f"\n  {'thresh':>7} | {'TNR':>7} | {'TPR bas':>8} | {'TPR ada':>8} | "
          f"{'bal-acc':>8} | {'LSB':>7} | {'DCT':>7} | {'FFT':>7}")
    print("  " + "-" * 74)
    for (t, tnr, t_bas, t_ada, bal, yj, t_lsb, t_dct, t_fft) in rows:
        mark = '  <-- best' if t == best[0] else ''
        print(f"  {t:>7.2f} | {tnr:>6.1f}% | {t_bas:>7.1f}% | {t_ada:>7.1f}% | "
              f"{bal:>7.1f}% | {t_lsb:>6.1f}% | {t_dct:>6.1f}% | {t_fft:>6.1f}%{mark}")
    print()
    return mode, (best[0], best[1], best[2], best[3], best[4])


# ── Main ──────────────────────────────────────────────────────────────────────

def main(checkpoint_path=MODEL_CHECKPOINT, n_images=N_IMAGES):
    print(f"Loading SRNet on {DEVICE}...")
    model = SRNet().to(DEVICE)

    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Could not find {checkpoint_path}.")
        return

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    # Strip torch.compile prefix if present
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    print(f"Checkpoint: {checkpoint_path}  (saved val_acc: {checkpoint.get('val_acc', '?')})")
    print(f"Images per folder: {n_images}  |  Aggregation modes: {', '.join(AGG_MODES)}\n")
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

        images = sorted(glob.glob(os.path.join(folder, '*.*')))[:n_images]
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
            # Per-target readout so each check reports as soon as it finishes.
            for mode in AGG_MODES:
                s = np.array([aggregate_scores(w, mode) for w in per_image_windows])
                print(f"  {mode:>5}: min {s.min():.3f}  median {np.median(s):.3f}  "
                      f"mean {s.mean():.3f}  max {s.max():.3f}")

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
        for (m, b) in sorted(valid, key=lambda mb: mb[1][4], reverse=True):
            print(f"  {m:>6} | {b[0]:>7.2f} | {b[1]:>6.1f}% | {b[2]:>9.1f}% | "
                  f"{b[3]:>9.1f}% | {b[4]:>7.1f}%")
        best_mode, best_row = max(valid, key=lambda mb: mb[1][4])
        print("  " + "-" * 64)
        print(f"  WINNER: mode={best_mode}  threshold={best_row[0]:.2f}  "
              f"bal-acc={best_row[4]:.1f}%")
        print("#" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sliding-window steganalysis benchmark (multi-mode sweep).")
    parser.add_argument('--checkpoint', default=MODEL_CHECKPOINT,
                        help=f"Model checkpoint to test (default: {MODEL_CHECKPOINT}).")
    parser.add_argument('--images', type=int, default=N_IMAGES,
                        help=f"Images scored per folder; lower = faster sweep "
                             f"(default: {N_IMAGES}).")
    args = parser.parse_args()
    main(checkpoint_path=args.checkpoint, n_images=args.images)
"""
validate.py — Held-out validation loop run at the end of every training epoch.
"""

import random

import torch
from PIL import Image
from torchvision import transforms

from generators.unified_generator import UnifiedGenerator
from training.config import (
    ALL_GEN_TYPES,
    DCT_COEFF_MODES,
    DEVICE,
    EVAL_SEED,
    FFT_FREQ_BANDS,
    GEN_TYPE_WEIGHTS,
    LSB_STRATEGIES,
    MAX_CAPACITY,
    MIN_CAPACITY,
)
from training.genome import compute_log_fft


def run_validation(
    model: torch.nn.Module,
    val_lossy: list,
    val_lossless: list,
    unified_gen: UnifiedGenerator,
    criterion: torch.nn.Module,
    epoch: int,
) -> tuple[float, float]:
    """
    Evaluate *model* on up to 750 randomly sampled validation images.

    For each image a random generator config is sampled uniformly across all
    gen_types and strategies so the validation score reflects overall coverage
    rather than any single operating point.

    Returns:
        (avg_loss, accuracy_pct)
    """
    model.eval()
    to_tensor = transforms.ToTensor()
    rng       = random.Random(EVAL_SEED + epoch)

    val_files = val_lossy + val_lossless
    rng.shuffle(val_files)
    val_files = val_files[:750]

    all_inputs, all_labels = [], []

    for path in val_files:
        config = _sample_val_config(rng)

        try:
            with Image.open(path) as img:
                crop_img = img.convert('L')

            w, h = crop_img.size
            if w < 256 or h < 256:
                continue

            left = (w - 256) // 2
            top  = (h - 256) // 2
            crop = crop_img.crop((left, top, left + 256, top + 256))

            stego_arr, _ = unified_gen.generate_stego(crop, None, config)
            if stego_arr is None:
                continue

            spatial_cover = to_tensor(crop)
            log_fft_cover = compute_log_fft(spatial_cover)
            all_inputs.append(torch.cat([spatial_cover, log_fft_cover], dim=0))
            all_labels.append(0)

            spatial_stego = to_tensor(Image.fromarray(stego_arr))
            log_fft_stego = compute_log_fft(spatial_stego)
            all_inputs.append(torch.cat([spatial_stego, log_fft_stego], dim=0))
            all_labels.append(1)

        except Exception:
            continue

    if not all_inputs:
        return 0.0, 0.0

    total_loss, correct_total, total_samples = 0.0, 0, 0
    VAL_BATCH = 64

    with torch.no_grad():
        for i in range(0, len(all_inputs), VAL_BATCH):
            inputs_t = torch.stack(all_inputs[i: i + VAL_BATCH]).to(DEVICE)
            labels_t = torch.tensor(
                all_labels[i: i + VAL_BATCH], dtype=torch.long).to(DEVICE)

            with torch.amp.autocast('cuda'):
                outputs  = model(inputs_t)
                per_loss = criterion(outputs, labels_t)
                loss     = per_loss.mean()

            _, preds = torch.max(outputs, 1)
            total_loss    += loss.item() * labels_t.size(0)
            correct_total += (preds == labels_t).sum().item()
            total_samples += labels_t.size(0)

    return total_loss / total_samples, 100.0 * correct_total / total_samples


# ── Internal ──────────────────────────────────────────────────────────────────

def _sample_val_config(rng: random.Random) -> dict:
    """Sample a random generator config for one validation image."""
    gen_type = rng.choices(ALL_GEN_TYPES, weights=GEN_TYPE_WEIGHTS)[0]

    if gen_type == 'lsb':
        return {
            'gen_type':       'lsb',
            'strategy':       rng.choices(LSB_STRATEGIES, weights=[2, 1, 1, 2])[0],
            'capacity_ratio': rng.uniform(MIN_CAPACITY, MAX_CAPACITY),
            'edge_threshold': rng.randint(0, 100),
            'step':           rng.randint(1, 15),
            'bit_depth':      1,
            'message':        None,
        }
    if gen_type == 'dct':
        return {
            'gen_type':        'dct',
            'coeff_selection': rng.choice(DCT_COEFF_MODES),
            'strength':        rng.uniform(1.0, 8.0),
            'capacity_ratio':  rng.uniform(MIN_CAPACITY, MAX_CAPACITY),
        }
    # fft
    return {
        'gen_type':       'fft',
        'freq_band':      rng.choice(FFT_FREQ_BANDS),
        'strength':       rng.uniform(2.0, 20.0),
        'capacity_ratio': rng.uniform(MIN_CAPACITY, MAX_CAPACITY),
    }
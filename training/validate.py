"""
validate.py — Held-out validation loop run at the end of every training epoch.
"""

import random

import torch
from PIL import Image
from torchvision import transforms

from generators.unified_generator import UnifiedGenerator
from training.config import (
    ADAPTIVE_MIN_CAPACITY,
    ADAPTIVE_MODES,
    ALL_GEN_TYPES,
    DCT_CAPACITY_RANGE,
    DCT_COEFF_MODES,
    DCT_STRENGTH_RANGE,
    DEVICE,
    EVAL_SEED,
    FFT_CAPACITY_RANGE,
    FFT_FREQ_BANDS,
    FFT_STRENGTH_RANGE,
    GEN_TYPE_WEIGHTS,
    LSB_CAPACITY_RANGE,
    LSB_STRATEGIES,
    MAX_CAPACITY,
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

    Prints per-gen_type accuracy so checkpoint selection is not blind to
    adaptive collapse (a model scoring 100% LSB + 0% adaptive looks the
    same in aggregate as a balanced model otherwise).

    Returns:
        (avg_loss, accuracy_pct)
    """
    model.eval()
    to_tensor = transforms.ToTensor()
    rng       = random.Random(EVAL_SEED + epoch)

    val_files = val_lossy + val_lossless
    rng.shuffle(val_files)
    val_files = val_files[:750]

    all_inputs, all_labels, all_gen_types = [], [], []

    for path in val_files:
        config = _sample_val_config(rng)

        try:
            with Image.open(path) as img:
                full_img = img.convert('L')

            w, h = full_img.size
            if w < 256 or h < 256:
                continue

            # Match training's resolution augmentation: half the images are the
            # whole image downsampled to 256×256, half a native-resolution crop.
            if rng.random() < 0.5:
                crop = full_img.resize((256, 256), Image.BILINEAR)
            else:
                left = (w - 256) // 2
                top  = (h - 256) // 2
                crop = full_img.crop((left, top, left + 256, top + 256))

            stego_arr, _ = unified_gen.generate_stego(crop, None, config)
            if stego_arr is None:
                continue

            gt = config['gen_type']

            spatial_cover = to_tensor(crop)
            log_fft_cover = compute_log_fft(spatial_cover)
            all_inputs.append(torch.cat([spatial_cover, log_fft_cover], dim=0))
            all_labels.append(0)
            all_gen_types.append(gt)

            spatial_stego = to_tensor(Image.fromarray(stego_arr))
            log_fft_stego = compute_log_fft(spatial_stego)
            all_inputs.append(torch.cat([spatial_stego, log_fft_stego], dim=0))
            all_labels.append(1)
            all_gen_types.append(gt)

        except Exception:
            continue

    if not all_inputs:
        return 0.0, 0.0

    total_loss, correct_total, total_samples = 0.0, 0, 0
    VAL_BATCH = 64

    all_preds = []
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
            all_preds.extend(preds.cpu().tolist())
            total_loss    += loss.item() * labels_t.size(0)
            correct_total += (preds == labels_t).sum().item()
            total_samples += labels_t.size(0)

    # Per-gen_type accuracy so adaptive collapse is visible in logs.
    type_correct: dict[str, int] = {}
    type_total:   dict[str, int] = {}
    for pred, label, gt in zip(all_preds, all_labels, all_gen_types):
        type_correct[gt] = type_correct.get(gt, 0) + int(pred == label)
        type_total[gt]   = type_total.get(gt, 0) + 1
    per_type_parts = [
        f"{gt}={100.0 * type_correct[gt] / type_total[gt]:.1f}%({type_total[gt]})"
        for gt in sorted(type_total)
    ]
    print(f"[VAL] Per-type: {' | '.join(per_type_parts)}")

    return total_loss / total_samples, 100.0 * correct_total / total_samples


# ── Internal ──────────────────────────────────────────────────────────────────

def _sample_val_config(rng: random.Random) -> dict:
    """Sample a random generator config for one validation image.

    capacity_ratio is TRUE bits-per-pixel; each method samples within its own
    physical range. Adaptive spans the full [0.20, 0.75] bpp target so per-type
    accuracy reflects real detection capability, not a single operating point.
    """
    gen_type = rng.choices(ALL_GEN_TYPES, weights=GEN_TYPE_WEIGHTS)[0]

    if gen_type == 'lsb':
        return {
            'gen_type':       'lsb',
            'strategy':       rng.choices(LSB_STRATEGIES, weights=[2, 1, 1])[0],
            'capacity_ratio': rng.uniform(*LSB_CAPACITY_RANGE),
            'step':           rng.randint(1, 8),
            'bit_depth':      1,
            'message':        None,
        }
    if gen_type == 'dct':
        return {
            'gen_type':        'dct',
            'coeff_selection': rng.choice(DCT_COEFF_MODES),
            'strength':        rng.uniform(*DCT_STRENGTH_RANGE),
            'capacity_ratio':  rng.uniform(*DCT_CAPACITY_RANGE),
        }
    if gen_type == 'adaptive':
        return {
            'gen_type':       'adaptive',
            'adaptive_mode':  rng.choice(ADAPTIVE_MODES),
            'capacity_ratio': rng.uniform(ADAPTIVE_MIN_CAPACITY, MAX_CAPACITY),
            'sigma_offset':   rng.uniform(0.5, 5.0),
            'use_diagonal':   rng.choice([True, False]),
            'cost_exponent':  rng.uniform(0.5, 2.0),
            'canonical':      True,
        }
    # fft
    return {
        'gen_type':       'fft',
        'freq_band':      rng.choice(FFT_FREQ_BANDS),
        'strength':       rng.uniform(*FFT_STRENGTH_RANGE),
        'capacity_ratio': rng.uniform(*FFT_CAPACITY_RANGE),
    }

"""
batch.py — Batch construction with seven-layer diversity guarantees and
           fixed-size padding for CUDA graph stability.
"""

import random
import torch

import training.config as config  # needed for getattr() on dynamic fine-tune overrides
from training.config import (
    ADAPTIVE_BATCH_FRACTION,
    FFT_COMBINED_BATCH_CAP,
    FIXED_BATCH_SIZE,
    LOW_CAPACITY_BATCH_FRACTION,
    NICHE_BATCH_CAP,
)
from training.genome import get_niche


def build_assigned_pairs(batch_files: list, evo_manager) -> tuple[list, int]:
    """
    Assign a steganography genome to each cover image path using layered
    diversity constraints.

    Layer 2 — Low-capacity floor (15 % of free slots, capacity below
        LOW_CAPACITY_THRESHOLD = 0.12 bpp).

    Layer 3 — Per-niche cap (40 % of free slots per niche).

    Layer 4 — FFT combined cap (25 % of free slots for all FFT sub-niches).

    Layer 5 — FFT-low low-strength floor (FFT_LOW_LOWSTRENGTH_FRACTION, default 0 %).
        Explicit floor for fft_low genomes with strength ≤ 7.5.
        Set in finetune.py to 10 % to target the weak spot.

    Layer 6 — DCT low-mid low-strength floor (DCT_LOWMID_LOWSTRENGTH_FRACTION, default 0 %).
        Explicit floor for dct_low_mid genomes with strength ≤ 3.5.
        Set in finetune.py to 10 % to target the weak spot.

    Layer 7 — Adaptive floor (ADAPTIVE_BATCH_FRACTION):
        Guarantees S-UNIWARD stego appears in every batch.

    Returns:
        (pairs, fallback_count)
    """
    n = len(batch_files)

    # ── Slot allocation ───────────────────────────────────────────────────────
    n_fft_lowstrength = max(1, int(n * getattr(config, 'FFT_LOW_LOWSTRENGTH_FRACTION',  0.0)))
    n_dct_lowmid      = max(1, int(n * getattr(config, 'DCT_LOWMID_LOWSTRENGTH_FRACTION', 0.0)))
    n_adaptive        = max(1, int(n * ADAPTIVE_BATCH_FRACTION))
    # Clamp so micro-batches (n < ~8) degrade gracefully — the floors above each
    # reserve >= 1 slot, which can otherwise underflow n_free / n_normal below 0
    # and produce nonsense slices. At BATCH_SIZE=64 these clamps never bind.
    n_free            = max(0, n - n_fft_lowstrength - n_dct_lowmid - n_adaptive)
    n_lowcap          = (max(1, int(n_free * LOW_CAPACITY_BATCH_FRACTION))
                         if n_free > 0 else 0)
    n_normal          = max(0, n_free - n_lowcap)

    # ── Path slicing ──────────────────────────────────────────────────────────
    shuffled = random.sample(batch_files, len(batch_files))
    cursor   = 0

    fft_low_paths    = shuffled[cursor : cursor + n_fft_lowstrength]; cursor += n_fft_lowstrength
    dct_lowmid_paths = shuffled[cursor : cursor + n_dct_lowmid];      cursor += n_dct_lowmid
    adaptive_paths   = shuffled[cursor : cursor + n_adaptive];        cursor += n_adaptive
    lowcap_paths     = shuffled[cursor : cursor + n_lowcap];          cursor += n_lowcap
    normal_paths     = shuffled[cursor:]

    pairs          = []
    fallback_count = 0

    # Layer 5 — FFT-low low-strength
    for path in fft_low_paths:
        pairs.append((path, evo_manager.get_lowstrength_fft_low_genome()))

    # Layer 6 — DCT low-mid low-strength
    for path in dct_lowmid_paths:
        pairs.append((path, evo_manager.get_lowstrength_dct_lowmid_genome()))

    # Layer 7 — Adaptive floor (S-UNIWARD)
    for path in adaptive_paths:
        pairs.append((path, evo_manager.get_adaptive_genome('suniward')))

    # Layer 2 — low-capacity
    for path in lowcap_paths:
        pairs.append((path, evo_manager.get_low_capacity_genome()))

    # Layers 3 + 4 — per-niche cap + FFT combined cap
    niche_cap  = int(n_normal * NICHE_BATCH_CAP)
    fft_cap    = int(n_normal * FFT_COMBINED_BATCH_CAP)
    niche_used: dict[str, int] = {}
    fft_used   = 0

    for path in normal_paths:
        placed = False
        for _ in range(15):
            g      = evo_manager.get_random_genome()
            niche  = get_niche(g)
            is_fft = g['gen_type'] == 'fft'

            if niche_used.get(niche, 0) >= niche_cap:
                continue
            if is_fft and fft_used >= fft_cap:
                continue

            niche_used[niche] = niche_used.get(niche, 0) + 1
            if is_fft:
                fft_used += 1
            pairs.append((path, g))
            placed = True
            break

        if not placed:
            pairs.append((path, evo_manager.get_random_genome()))
            fallback_count += 1

    random.shuffle(pairs)
    return pairs, fallback_count


def make_fixed_batch(
    inputs: list,
    labels: list,
    batch_genome_names: list,
) -> tuple:
    """
    Pad or truncate to exactly FIXED_BATCH_SIZE for CUDA graph stability.

    Padded slots use random noise tensors with weight=0.0 so they contribute
    nothing to the loss or accuracy computation.
    """
    if not inputs:
        return None, None, None, None

    current = len(inputs)
    if current > FIXED_BATCH_SIZE:
        inputs             = inputs[:FIXED_BATCH_SIZE]
        labels             = labels[:FIXED_BATCH_SIZE]
        batch_genome_names = batch_genome_names[:FIXED_BATCH_SIZE]
        current            = FIXED_BATCH_SIZE

    weights    = [1.0] * current
    pad_needed = FIXED_BATCH_SIZE - current

    if pad_needed > 0:
        noise_tensors      = [torch.randn_like(inputs[0]) for _ in range(pad_needed)]
        inputs             = inputs             + noise_tensors
        labels             = labels             + [0]    * pad_needed
        batch_genome_names = batch_genome_names + [None] * pad_needed
        weights            = weights            + [0.0]  * pad_needed

    return (
        torch.stack(inputs),
        torch.tensor(labels,  dtype=torch.long),
        torch.tensor(weights, dtype=torch.float32),
        batch_genome_names,
    )
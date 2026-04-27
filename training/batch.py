"""
batch.py — Batch construction with six-layer diversity guarantees and
           fixed-size padding for CUDA graph stability.
"""

import random
import torch

import training.config as config  # needed for getattr() on dynamic fine-tune overrides
from training.config import (
    ANY_EDGE_BATCH_FRACTION,
    FFT_COMBINED_BATCH_CAP,
    FIXED_BATCH_SIZE,
    HARD_EDGE_BATCH_FRACTION,
    LOW_CAPACITY_BATCH_FRACTION,
    NICHE_BATCH_CAP,
)
from training.genome import get_niche


def build_assigned_pairs(batch_files: list, evo_manager) -> tuple[list, int]:
    """
    Assign a steganography genome to each cover image path using six layers
    of diversity constraints.

    Layer 1a — Hard edge floor (HARD_EDGE_BATCH_FRACTION = 20 %):
        lsb_edge with threshold ≤ 9 AND capacity ≤ 0.25 every batch.

    Layer 1b — Any edge floor (ANY_EDGE_BATCH_FRACTION = 10 %):
        Any lsb_edge genome. Combined with 1a = 20 % edge total.

    Layer 2 — Low-capacity floor (15 % of free slots, capacity < 0.30).

    Layer 3 — Per-niche cap (40 % of free slots per niche, 8 niches total).

    Layer 4 — FFT combined cap (30 % of free slots for all FFT sub-niches).

    Layer 5 — FFT-low low-strength floor (FFT_LOW_LOWSTRENGTH_FRACTION, default 0 %).
        Explicit floor for fft_low genomes with strength ≤ 7.5.
        Set in finetune.py to 10 % to target the weak spot.

    Layer 6 — DCT low-mid low-strength floor (DCT_LOWMID_LOWSTRENGTH_FRACTION, default 0 %).
        Explicit floor for dct_low_mid genomes with strength ≤ 3.5.
        Set in finetune.py to 10 % to target the weak spot.

    Returns:
        (pairs, fallback_count)
    """
    n = len(batch_files)

    # ── Slot allocation ───────────────────────────────────────────────────────
    n_hard_edge       = max(1, int(n * HARD_EDGE_BATCH_FRACTION))
    n_any_edge        = max(1, int(n * ANY_EDGE_BATCH_FRACTION))
    n_edge_total      = n_hard_edge + n_any_edge
    n_fft_lowstrength = max(1, int(n * getattr(config, 'FFT_LOW_LOWSTRENGTH_FRACTION',  0.0)))
    n_dct_lowmid      = max(1, int(n * getattr(config, 'DCT_LOWMID_LOWSTRENGTH_FRACTION', 0.0)))
    n_free            = n - n_edge_total - n_fft_lowstrength - n_dct_lowmid
    n_lowcap          = max(1, int(n_free * LOW_CAPACITY_BATCH_FRACTION))
    n_normal          = n_free - n_lowcap

    # ── Path slicing ──────────────────────────────────────────────────────────
    shuffled = random.sample(batch_files, len(batch_files))
    cursor   = 0

    hard_paths     = shuffled[cursor : cursor + n_hard_edge];       cursor += n_hard_edge
    any_paths      = shuffled[cursor : cursor + n_any_edge];        cursor += n_any_edge
    fft_low_paths  = shuffled[cursor : cursor + n_fft_lowstrength]; cursor += n_fft_lowstrength
    dct_lowmid_paths = shuffled[cursor : cursor + n_dct_lowmid];    cursor += n_dct_lowmid
    lowcap_paths   = shuffled[cursor : cursor + n_lowcap];          cursor += n_lowcap
    normal_paths   = shuffled[cursor:]

    all_edge_genomes = [g for g in evo_manager.population if get_niche(g) == 'lsb_edge']
    if not all_edge_genomes:
        all_edge_genomes = [evo_manager._new_lsb("fallback_any_edge", 'edge')]

    pairs          = []
    fallback_count = 0

    # Layer 1a — hard edge
    for path in hard_paths:
        pairs.append((path, evo_manager.get_hard_edge_genome()))

    # Layer 1b — any edge
    for path in any_paths:
        pairs.append((path, random.choice(all_edge_genomes)))

    # Layer 5 — FFT-low low-strength
    for path in fft_low_paths:
        pairs.append((path, evo_manager.get_lowstrength_fft_low_genome()))

    # Layer 6 — DCT low-mid low-strength
    for path in dct_lowmid_paths:
        pairs.append((path, evo_manager.get_lowstrength_dct_lowmid_genome()))

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
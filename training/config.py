"""
config.py — Central configuration for all training hyperparameters and constants.

This is the only file you need to edit for tuning runs.
"""

import math
import multiprocessing
import torch

# ── Hardware ──────────────────────────────────────────────────────────────────
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 2)

# ── Training loop ─────────────────────────────────────────────────────────────
BATCH_SIZE                  = 64
GRADIENT_ACCUMULATION_STEPS = 2
EPOCHS                      = 80

# Fixed batch size for torch.compile CUDA graph stability.
# Covers + stegos interleaved, so 2× BATCH_SIZE.
FIXED_BATCH_SIZE = BATCH_SIZE * 2

# ── Learning rate schedule ────────────────────────────────────────────────────
INITIAL_LR = 0.0001
MIN_LR     = 5e-6
MAX_LR     = 0.0005

# ── Evolutionary algorithm ────────────────────────────────────────────────────
# 20 non-adaptive genomes (EA-bred: LSB/DCT/FFT) + 4 adaptive genomes
# (shape-evolved only — see evolution.py). evolve() keeps the two pools separate.
POPULATION_SIZE = 24

# ── Capacity semantics — TRUE bits-per-pixel for ALL generators ───────────────
# capacity_ratio is one uniform unit across every method: the payload in bits
# per pixel. Each generator translates bpp into its own quantity — pixels for
# LSB, 8×8 blocks for DCT, frequency components for FFT, λ for adaptive. Every
# method has a physical ceiling for a 256×256 image; generators cap silently,
# and the per-method ranges below keep the EA inside realistic, learnable bounds:
#   LSB : up to 1.0  bpp  (1 bit/pixel at bit_depth=1)
#   DCT : up to ~0.31 bpp (mid coeffs);  low_mid ~0.19;  random ~0.16
#   FFT : up to ~0.28 bpp (high band);   mid ~0.14;      low ~0.017 (hard cap)
LSB_CAPACITY_RANGE = (0.20, 0.75)
DCT_CAPACITY_RANGE = (0.12, 0.30)
FFT_CAPACITY_RANGE = (0.08, 0.20)

# Method-agnostic span (curriculum annealing, batch diversity layers).
MIN_CAPACITY = 0.05
MAX_CAPACITY = 0.75

# Adaptive (S-UNIWARD) payload is TRUE bpp but is set by the curriculum schedule
# (ADAPTIVE_CURRICULUM_SCHEDULE), NOT evolved by the EA. 0.20 bpp is the floor —
# it matches the hardest held-out test set (SUNI_02).
ADAPTIVE_MIN_CAPACITY = 0.20

# Strength floors — fence off near-invisible per-component modifications so the
# EA cannot drift into a low-strength corner (the old FFT Str=5.0 collapse).
DCT_STRENGTH_RANGE = (3.0, 8.0)    # DCT quantization step
FFT_STRENGTH_RANGE = (8.0, 20.0)   # FFT magnitude step (effective = ×sqrt(H·W))

# Generator taxonomy
ALL_GEN_TYPES   = ['lsb', 'dct', 'fft', 'adaptive']
LSB_STRATEGIES  = ['random', 'sequential', 'skip']
DCT_COEFF_MODES = ['mid', 'low_mid', 'random']
FFT_FREQ_BANDS  = ['low', 'mid', 'high']
ADAPTIVE_MODES  = ['suniward']

# Weights for [lsb, dct, fft, adaptive]. DCT and S-UNIWARD are the hardest to
# learn, so they get the largest share.
GEN_TYPE_WEIGHTS = [0.20, 0.30, 0.20, 0.30]

# FFT low-band seed diversity: 4 seeds varying strength. FFT-low's payload is
# physically capped at ~0.017 bpp, so its capacity seeds are nominal — every
# FFT-low genome embeds at that ceiling regardless of the requested bpp.
FFT_LOW_SEED_STRENGTHS  = [8.0, 11.0, 15.0, 20.0]
FFT_LOW_SEED_CAPACITIES = [0.05, 0.05, 0.05, 0.05]

SKIP_SEED_STEPS = [3, 7]

# Adaptive seed configs — S-UNIWARD only. The EA evolves ONLY the cost-model
# shape (sigma_offset, cost_exponent, use_diagonal); the payload is overridden
# at embed time by ADAPTIVE_CURRICULUM_SCHEDULE. The capacity_ratio field is a
# placeholder. Seeds span a range of cost-model shapes for EA diversity.
ADAPTIVE_SEED_CONFIGS = [
    # (mode, sigma_offset, capacity_ratio[placeholder — curriculum-overridden], cost_exponent)
    ('suniward', 0.5, 0.40, 1.2),
    ('suniward', 1.0, 0.40, 1.0),
    ('suniward', 2.0, 0.40, 0.8),
    ('suniward', 3.0, 0.40, 1.5),
]

# Adaptive payload curriculum. Every adaptive genome — Layer-7 floor and any
# EA-routed slot — has its capacity_ratio set from this schedule at embed time,
# so the EA can never collapse onto the lowest (hardest) payload. Each entry is
# (up_to_epoch, [payloads bpp]); the payload is sampled uniformly from the list
# for epochs below that bound. Accumulating: high payloads stay in the mix so
# the model keeps detecting strong stego while harder payloads phase in.
ADAPTIVE_CURRICULUM_SCHEDULE = [
    (15,  [0.75]),
    (30,  [0.75, 0.40]),
    (45,  [0.75, 0.40, 0.30]),
    (999, [0.75, 0.40, 0.30, 0.20]),
]

ALL_NICHES = [
    'lsb_random', 'lsb_sequential', 'lsb_skip',
    'dct',
    'fft_low', 'fft_mid', 'fft_high',
    'adaptive_suniward',
]

MIN_NICHE_SIZE = 2
# Capacity anti-collapse penalty (re-enabled after Run 20). The EA maximises
# fool rate, so with no counter-pressure it drives every genome to its
# lowest-capacity corner — collapsing the training distribution onto
# near-undetectable stego. Run 20 proved this: DCT/FFT/LSB all floor-hugging,
# and the model learned perturbation magnitude instead of stego structure.
#
# The penalty ramps linearly from 0 at the per-method threshold to
# CAPACITY_PENALTY_WEIGHT at capacity 0 (see evolution.py _penalised_fitness).
# A single absolute threshold can't be fair across methods whose true-bpp
# ceilings span ~0.017 (FFT-low) to 1.0 (LSB), so each method gets a threshold
# near the middle of its OWN range: floor-huggers are penalised, near-ceiling
# genomes are free. Adaptive uses its own floor (0.20) → never penalised, since
# adaptive capacity is curriculum-set (not EA-evolved); penalising it would only
# inject curriculum-sampling noise into cost-model shape selection.
CAPACITY_PENALTY_WEIGHT = 0.15
CAPACITY_PENALTY_THRESHOLDS = {
    'lsb':      0.45,                  # midpoint of (0.20, 0.75)
    'dct':      0.20,                  # mid-coeff (~0.31) escapes; low_mid/random floor-huggers nudged up
    'fft':      0.14,                  # fft_high (0.20) escapes; fft_mid at its cap; fft_low stays suppressed
    'adaptive': ADAPTIVE_MIN_CAPACITY, # 0.20 → effectively off (adaptive capacity >= 0.20 always)
}

# ── Batch diversity floors & caps ─────────────────────────────────────────────
# Layer 2 — Low-capacity floor (fraction of free slots)
LOW_CAPACITY_BATCH_FRACTION = 0.15
LOW_CAPACITY_THRESHOLD      = 0.12   # bpp — genomes below this count as low-payload
# Layer 3 — Per-niche cap (fraction of free slots)
NICHE_BATCH_CAP = 0.40
# Layer 4 — FFT combined cap (all three FFT sub-niches together)
FFT_COMBINED_BATCH_CAP = 0.25
# Layer 5 — DCT low-mid hard floor (0.0 → batch.py falls back to 1-slot minimum).
# Kept at 0.0: 0.12 caused BN variance collapse at epoch 1 by starving easy gradient
# signal during warmup. Experiment branch reached 78% val without this floor.
DCT_LOWMID_LOWSTRENGTH_FRACTION = 0.0
# Layer 6 — FFT low-band hard floor (0.0 → batch.py falls back to 1-slot minimum).
FFT_LOW_LOWSTRENGTH_FRACTION = 0.0
# Layer 7 — Adaptive floor: guarantee S-UNIWARD stego appears in every batch.
# Consumed by: training/batch.py build_assigned_pairs() Layer 7 loop via
# evo_manager.get_adaptive_genome('suniward').
ADAPTIVE_BATCH_FRACTION = 0.25

# Valid range for adaptive genome's cost_exponent field.
# Enforced (clamped) in evolution.py mutate() and validated at call-time in
# adaptive_gen.py embed() so out-of-range values raise immediately.
ADAPTIVE_COST_EXPONENT_BOUNDS = (0.5, 2.0)

# ── Curriculum ────────────────────────────────────────────────────────────────
CURRICULUM_END          = 8
CURRICULUM_BLEND_EPOCHS = 3

# ── Dataset split ─────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
SPLIT_FILE  = 'dataset_split.json'
SPLIT_SEED  = 42

# ── Validation / evaluation ───────────────────────────────────────────────────
EVAL_SEED = 99

# ── FFT normalisation ─────────────────────────────────────────────────────────
# Precomputed log(1 + 256²) ≈ 11.09 — safe upper bound for 256×256 log-FFT.
LOG_FFT_SCALE = math.log1p(256 * 256)
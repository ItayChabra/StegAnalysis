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
POPULATION_SIZE = 20

MIN_CAPACITY = 0.20
MAX_CAPACITY = 0.75
ADAPTIVE_MIN_CAPACITY = 0.05

# Generator taxonomy
ALL_GEN_TYPES   = ['lsb', 'dct', 'fft', 'adaptive']
LSB_STRATEGIES  = ['random', 'sequential', 'skip', 'edge']
DCT_COEFF_MODES = ['mid', 'low_mid', 'random']
FFT_FREQ_BANDS  = ['low', 'mid', 'high']
ADAPTIVE_MODES  = ['wow', 'suniward', 'hugo']

# Slightly reduce lsb/dct/fft weights to make room for adaptive
GEN_TYPE_WEIGHTS = [0.20, 0.33, 0.20, 0.27]

# FFT low-band seed diversity: 4 seeds bracketing the hard eval point.
FFT_LOW_SEED_STRENGTHS  = [5.0, 7.5, 10.0, 15.0]
FFT_LOW_SEED_CAPACITIES = [0.50, 0.40, 0.35, 0.25]

SKIP_SEED_STEPS = [3, 7]

# Adaptive seed configs — one per mode, varying sigma_offset difficulty.
# Low-capacity seeds (0.05–0.15) match Kaggle adaptive payload (~0.10 bpp).
ADAPTIVE_SEED_CONFIGS = [
    # (mode, sigma_offset, capacity_ratio, cost_exponent)
    # 0.40-capacity seeds restored: provide easy-to-detect stego during warmup epochs,
    # giving the model a gradient direction before harder seeds dominate.
    # Experiment branch proved these are required for training to bootstrap.
    ('wow',      1.0, 0.40, 1.0),   # warmup bootstrapping (easy, high payload)
    ('wow',      0.5, 0.10, 1.2),   # Kaggle-realistic low payload
    ('wow',      0.5, 0.07, 1.4),   # very low payload
    ('suniward', 1.0, 0.40, 1.0),   # warmup bootstrapping
    ('suniward', 0.5, 0.10, 1.2),   # Kaggle-realistic low payload
    ('suniward', 0.5, 0.07, 1.4),   # very low payload
    ('hugo',     1.0, 0.40, 1.0),   # warmup bootstrapping
    ('hugo',     1.0, 0.10, 1.0),   # Kaggle-realistic low payload
    ('hugo',     2.0, 0.07, 0.8),   # very low payload
]

ALL_NICHES = [
    'lsb_random', 'lsb_sequential', 'lsb_skip', 'lsb_edge',
    'dct',
    'fft_low', 'fft_mid', 'fft_high',
    'adaptive_wow', 'adaptive_suniward', 'adaptive_hugo',
]

MIN_NICHE_SIZE             = 2
CAPACITY_PENALTY_THRESHOLD          = MIN_CAPACITY + 0.10
ADAPTIVE_CAPACITY_PENALTY_THRESHOLD = ADAPTIVE_MIN_CAPACITY + 0.10
CAPACITY_PENALTY_WEIGHT    = 0.15

# ── Batch diversity floors & caps ─────────────────────────────────────────────
# Layer 1a — Hard edge floor: lsb_edge with threshold≤9 AND capacity≤0.25
HARD_EDGE_BATCH_FRACTION = 0.15   # reduced slightly to free slots for adaptive
# Layer 1b — Any edge floor
ANY_EDGE_BATCH_FRACTION  = 0.08
# Layer 2 — Low-capacity floor (fraction of free slots)
LOW_CAPACITY_BATCH_FRACTION = 0.15
LOW_CAPACITY_THRESHOLD      = 0.30
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
# Layer 7 — Adaptive floor: guarantee all three adaptive sub-niches appear each batch.
# Consumed by: training/batch.py build_assigned_pairs() Layer 7 loop.
# Slots assigned round-robin across ['wow', 'suniward', 'hugo'] via
# evo_manager.get_adaptive_genome(mode); at least 3 slots guaranteed per batch.
# Bumped from 0.15 to 0.18: 11 adaptive slots/batch vs 9 before.
ADAPTIVE_BATCH_FRACTION = 0.18

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
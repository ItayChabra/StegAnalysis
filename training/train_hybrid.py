import multiprocessing
import os
import sys
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import torch.optim as optim
from generators.unified_generator import UnifiedGenerator
from models.srnet import SRNet
import glob
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
import random
import copy
import json
import string
import numpy as np
from concurrent.futures import ThreadPoolExecutor

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True

# --- SETTINGS ---
BATCH_SIZE                  = 64
GRADIENT_ACCUMULATION_STEPS = 2
EPOCHS                      = 60
POPULATION_SIZE             = 20
NUM_WORKERS  = max(1, multiprocessing.cpu_count() - 2)
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cudnn.benchmark = True

INITIAL_LR   = 0.0001
# Run 6 fix: halved from 0.001. The volatile val swings (55% <-> 90%) in run 5
# were caused partly by overshooting at full LR during FFT-heavy training phases.
MIN_LR  = 5e-6
MAX_LR  = 0.0005
MIN_CAPACITY = 0.20
MAX_CAPACITY = 0.75

# Fixed batch size for torch.compile CUDA graph stability.
FIXED_BATCH_SIZE = BATCH_SIZE * 2   # covers + stegos interleaved

# ---- Generator taxonomy -------------------------------------------------------
ALL_GEN_TYPES   = ['lsb', 'dct', 'fft']
LSB_STRATEGIES  = ['random', 'sequential', 'skip', 'edge']
DCT_COEFF_MODES = ['mid', 'low_mid', 'random']
FFT_FREQ_BANDS  = ['low', 'mid', 'high']

FFT_LOW_SEED_STRENGTHS  = [5.0, 7.5, 10.0, 15.0]
FFT_LOW_SEED_CAPACITIES = [0.50, 0.40, 0.35, 0.25]

GEN_TYPE_WEIGHTS = [0.34, 0.33, 0.33]

# Run 6 fix: FFT niche split into three sub-niches.
# Previously all FFT genomes shared one 'fft' niche regardless of freq_band.
# 8 fft_low clones in a single niche → dampening by sqrt(8) ≈ 2.8x.
# With split: each sub-niche has 2-4 genomes → dampening by sqrt(2-4) ≈ 1.4-2x,
# AND the per-niche cap applies individually to fft_low / fft_mid / fft_high.
ALL_NICHES = [
    'lsb_random', 'lsb_sequential', 'lsb_skip', 'lsb_edge',
    'dct',
    'fft_low', 'fft_mid', 'fft_high',
]

MIN_NICHE_SIZE             = 2
CAPACITY_PENALTY_THRESHOLD = MIN_CAPACITY + 0.10
CAPACITY_PENALTY_WEIGHT    = 0.15

SKIP_SEED_STEPS = [3, 7]

# ---- Batch diversity guarantees -----------------------------------------------
# Layer 1a — Hard edge floor (20%): lsb_edge with threshold<=9 AND capacity<=0.25
HARD_EDGE_BATCH_FRACTION = 0.20
# Layer 1b — Any edge floor (10%): any lsb_edge genome
ANY_EDGE_BATCH_FRACTION  = 0.10
# Together 1a + 1b = 20% edge, same total as run 5's single 20% floor.
# The split ensures the hard eval config appears every batch, not just easy edges.

# Layer 2 — Low-capacity floor (15% of free slots): capacity < 0.30
LOW_CAPACITY_BATCH_FRACTION = 0.15
LOW_CAPACITY_THRESHOLD      = 0.30

# Layer 3 — Per-niche cap (40% of free slots)
NICHE_BATCH_CAP = 0.40

# Layer 4 — FFT combined cap (30% of free slots): NEW in run 6.
# Even with individual per-niche caps, fft_low + fft_mid + fft_high combined
# could consume up to 120% of free slots if all three hit their caps simultaneously.
# This ceiling limits all FFT variants together to 30% of free slots.
FFT_COMBINED_BATCH_CAP = 0.30

EVAL_SEED = 99

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
SPLIT_FILE  = 'dataset_split.json'
SPLIT_SEED  = 42

CURRICULUM_END          = 8
CURRICULUM_BLEND_EPOCHS = 3


# ==================== GENOME HELPERS ====================

# Precomputed scale factor for 256x256 images (~11.09)
# Safe upper bound for the DC component in log-magnitude FFT
_LOG_FFT_SCALE = math.log1p(256 * 256)


def compute_log_fft(spatial_tensor):
    """
    Convert a (1, H, W) spatial tensor to (1, H, W) log-magnitude FFT.
    Uses a precomputed fixed constant to preserve absolute payload differences.
    """
    # Native PyTorch FFT (keeps data on GPU, much faster than Numpy)
    fft_complex = torch.fft.fft2(spatial_tensor)

    # Shift the zero-frequency component to the center
    fft_shifted = torch.fft.fftshift(fft_complex, dim=(-2, -1))

    # Compute log magnitude ( log(1 + abs(x)) )
    log_magnitude = torch.log1p(torch.abs(fft_shifted))

    # Divide by the global constant
    return log_magnitude / _LOG_FFT_SCALE

def get_niche(genome):
    """
    Run 6 change: FFT genomes are now mapped to 'fft_low', 'fft_mid', or
    'fft_high' based on freq_band, instead of all returning 'fft'.
    This gives each sub-type its own independent niche budget, diversity
    dampening calculation, and per-niche batch cap.
    """
    gt = genome.get('gen_type', 'lsb')
    if gt == 'lsb':
        return f"lsb_{genome.get('strategy', 'random')}"
    if gt == 'fft':
        return f"fft_{genome.get('freq_band', 'mid')}"
    return gt   # 'dct'


def is_low_capacity(genome):
    return genome.get('capacity_ratio', 0.5) < LOW_CAPACITY_THRESHOLD


def is_hard_edge(genome):
    """True if this genome matches the hard eval config: threshold<=9, capacity<=0.25."""
    return (genome.get('gen_type') == 'lsb' and
            genome.get('strategy') == 'edge' and
            genome.get('edge_threshold', 100) <= 9 and
            genome.get('capacity_ratio', 1.0) <= 0.25)


# ==================== DATASET SPLIT ====================

def create_or_load_split(lossy_files, lossless_files):
    if os.path.exists(SPLIT_FILE):
        print(f"[SPLIT] Loading existing split from {SPLIT_FILE}")
        with open(SPLIT_FILE, 'r') as f:
            split = json.load(f)
        print(f"[SPLIT]   Lossy:    {len(split['lossy_train'])} train | "
              f"{len(split['lossy_val'])} val | {len(split['lossy_test'])} test")
        print(f"[SPLIT]   Lossless: {len(split['lossless_train'])} train | "
              f"{len(split['lossless_val'])} val | {len(split['lossless_test'])} test")
        return split

    print(f"[SPLIT] Creating new {TRAIN_RATIO:.0%}/{VAL_RATIO:.0%}/{TEST_RATIO:.0%} "
          f"split (seed={SPLIT_SEED})")
    rng = random.Random(SPLIT_SEED)

    def split_list(files):
        files    = [os.path.abspath(f) for f in files]
        shuffled = files.copy()
        rng.shuffle(shuffled)
        n       = len(shuffled)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)
        return (shuffled[:n_train],
                shuffled[n_train: n_train + n_val],
                shuffled[n_train + n_val:])

    lossy_train,    lossy_val,    lossy_test    = split_list(lossy_files)
    lossless_train, lossless_val, lossless_test = split_list(lossless_files)

    split = {
        'split_seed': SPLIT_SEED, 'train_ratio': TRAIN_RATIO,
        'val_ratio':  VAL_RATIO,  'test_ratio':  TEST_RATIO,
        'lossy_train':    lossy_train,    'lossy_val':    lossy_val,
        'lossy_test':     lossy_test,     'lossless_train': lossless_train,
        'lossless_val':   lossless_val,   'lossless_test':  lossless_test,
    }

    with open(SPLIT_FILE, 'w') as f:
        json.dump(split, f, indent=2)

    print(f"[SPLIT] Saved to {SPLIT_FILE}")
    print(f"[SPLIT]   Lossy:    {len(lossy_train)} train | "
          f"{len(lossy_val)} val | {len(lossy_test)} test")
    print(f"[SPLIT]   Lossless: {len(lossless_train)} train | "
          f"{len(lossless_val)} val | {len(lossless_test)} test")
    return split


# ==================== EVOLUTIONARY MANAGER ====================

class EvolutionaryManager:
    """
    Run 6 changes vs run 5
    ----------------------
    NICHE SYSTEM
      get_niche() returns 'fft_low', 'fft_mid', 'fft_high' instead of 'fft'.
      ALL_NICHES updated to 8 niches (was 6).
      evolve() injection handles the three FFT sub-niches individually.
      Diversity dampener now operates per-subtype rather than across all FFT.

    SEEDING
      Identical to run 5 (targeted edge seeds, low-cap sequential, fft_low x4,
      fft_mid/high x1 each, DCT x3, skip x2).
    """

    def __init__(self):
        self.population = []
        self.generation  = 0

        # LSB non-edge
        for strategy in ['random', 'sequential', 'skip']:
            self.population.append(self._new_lsb(f"Seed_lsb_{strategy}", strategy))

        # LSB sequential low-capacity
        g = self._new_lsb("Seed_lsb_sequential_lowcap", 'sequential')
        g['capacity_ratio'] = 0.25
        g['step']           = 1
        self.population.append(g)

        # LSB edge: generic threshold seeds
        for threshold in [5, 9, 15, 30]:
            g = self._new_lsb(f"Seed_lsb_edge_t{threshold}", 'edge')
            g['edge_threshold'] = threshold
            self.population.append(g)

        # LSB edge: hard-config seeds (pins both threshold AND capacity)
        for threshold, capacity in [(5, 0.21), (9, 0.21), (9, 0.25), (15, 0.25)]:
            g = self._new_lsb(
                f"Seed_lsb_edge_hard_t{threshold}_c{int(capacity * 100)}", 'edge')
            g['edge_threshold'] = threshold
            g['capacity_ratio'] = capacity
            self.population.append(g)

        # LSB skip: 2 seeds
        for s in SKIP_SEED_STEPS:
            g = self._new_lsb(f"Seed_skip_s{s}", 'skip')
            g['step'] = s
            self.population.append(g)

        # DCT
        for mode in DCT_COEFF_MODES:
            self.population.append(self._new_dct(f"Seed_dct_{mode}", mode))

        # FFT mid / high
        for band in ['mid', 'high']:
            self.population.append(self._new_fft(f"Seed_fft_{band}", band))

        # FFT low: 4 seeds
        for strength, capacity in zip(FFT_LOW_SEED_STRENGTHS, FFT_LOW_SEED_CAPACITIES):
            tag = str(strength).replace('.', 'p')
            g   = self._new_fft(f"Seed_fft_low_s{tag}", 'low')
            g['strength']       = strength
            g['capacity_ratio'] = capacity
            self.population.append(g)

        # Fill remainder
        while len(self.population) < POPULATION_SIZE:
            idx = len(self.population)
            self.population.append(self._generate_random_genome(f"Gen_{idx}"))

        self.stats = {g['name']: {'fooled': 0, 'attempts': 0} for g in self.population}

        niche_counts  = {n: sum(1 for g in self.population if get_niche(g) == n)
                         for n in ALL_NICHES}
        lowcap_counts = {n: sum(1 for g in self.population
                                if get_niche(g) == n and is_low_capacity(g))
                         for n in ALL_NICHES}
        print(f"[EVO INIT] Population: {len(self.population)}")
        print(f"[EVO INIT] Niches:  {niche_counts}")
        print(f"[EVO INIT] Low-cap: {lowcap_counts}")

    # ------------------------------------------------------------------ constructors

    def _new_lsb(self, name, strategy=None):
        return {
            'name':           name,
            'gen_type':       'lsb',
            'strategy':       strategy or random.choice(LSB_STRATEGIES),
            'step':           random.randint(1, 15),
            'bit_depth':      1,
            'edge_threshold': random.randint(0, 100),
            'capacity_ratio': random.triangular(MIN_CAPACITY, MAX_CAPACITY,
                                                MIN_CAPACITY + 0.15),
        }

    def _new_dct(self, name, coeff_selection=None):
        return {
            'name':            name,
            'gen_type':        'dct',
            'coeff_selection': coeff_selection or random.choice(DCT_COEFF_MODES),
            'strength':        round(random.uniform(2.0, 8.0), 2),
            'capacity_ratio':  random.triangular(MIN_CAPACITY, MAX_CAPACITY,
                                                 MIN_CAPACITY + 0.15),
        }

    def _new_fft(self, name, freq_band=None):
        return {
            'name':           name,
            'gen_type':       'fft',
            'freq_band':      freq_band or random.choice(FFT_FREQ_BANDS),
            'strength':       round(random.uniform(3.0, 20.0), 2),
            'capacity_ratio': random.triangular(MIN_CAPACITY, MAX_CAPACITY,
                                                MIN_CAPACITY + 0.15),
        }

    def _generate_random_genome(self, name):
        gen_type = random.choices(ALL_GEN_TYPES, weights=GEN_TYPE_WEIGHTS)[0]
        if gen_type == 'lsb':
            return self._new_lsb(name)
        if gen_type == 'dct':
            return self._new_dct(name)
        return self._new_fft(name)

    # ------------------------------------------------------------------ fitness

    def _penalised_fitness(self, genome, raw_fool_rate):
        capacity = genome.get('capacity_ratio', 0.5)
        if capacity < CAPACITY_PENALTY_THRESHOLD:
            shortfall = (CAPACITY_PENALTY_THRESHOLD - capacity) / CAPACITY_PENALTY_THRESHOLD
            penalty   = shortfall * CAPACITY_PENALTY_WEIGHT
        else:
            penalty = 0.0
        return max(0.0, raw_fool_rate - penalty)

    # ------------------------------------------------------------------ genetic operators

    def mutate(self, genome):
        g = copy.deepcopy(genome)
        g['name'] = f"{genome['name']}_m{self.generation}"
        n_mutations = 2 if random.random() < 0.4 else 1

        for _ in range(n_mutations):
            gt = g['gen_type']

            if gt == 'lsb':
                field = random.choice(['step', 'threshold', 'strategy', 'capacity', 'gen_type'])
                if field == 'step':
                    g['step'] = max(1, min(20, g['step'] + random.choice([-2, -1, 1, 2, 3])))
                elif field == 'threshold':
                    g['edge_threshold'] = max(0, min(100,
                        g['edge_threshold'] + random.randint(-20, 20)))
                elif field == 'strategy':
                    g['strategy'] = random.choice(LSB_STRATEGIES)
                elif field == 'capacity':
                    g['capacity_ratio'] = max(MIN_CAPACITY, min(MAX_CAPACITY,
                        g['capacity_ratio'] + random.uniform(-0.15, 0.15)))
                elif field == 'gen_type':
                    if random.random() < 0.15:
                        new_type = random.choice(['dct', 'fft'])
                        base = (self._new_dct(g['name']) if new_type == 'dct'
                                else self._new_fft(g['name']))
                        base['capacity_ratio'] = g['capacity_ratio']
                        g = base

            elif gt == 'dct':
                field = random.choice(['coeff', 'strength', 'capacity', 'gen_type'])
                if field == 'coeff':
                    g['coeff_selection'] = random.choice(DCT_COEFF_MODES)
                elif field == 'strength':
                    g['strength'] = max(2.0, min(10.0,
                        g['strength'] + random.uniform(-1.5, 1.5)))
                elif field == 'capacity':
                    g['capacity_ratio'] = max(MIN_CAPACITY, min(MAX_CAPACITY,
                        g['capacity_ratio'] + random.uniform(-0.15, 0.15)))
                elif field == 'gen_type':
                    if random.random() < 0.10:
                        base = self._new_fft(g['name'])
                        base['capacity_ratio'] = g['capacity_ratio']
                        g = base

            elif gt == 'fft':
                field = random.choice(['band', 'strength', 'capacity', 'gen_type'])
                if field == 'band':
                    other_bands = [b for b in FFT_FREQ_BANDS if b != g['freq_band']]
                    g['freq_band'] = random.choice(other_bands)
                elif field == 'strength':
                    g['strength'] = max(3.0, min(25.0,
                        g['strength'] + random.uniform(-3.0, 3.0)))
                elif field == 'capacity':
                    g['capacity_ratio'] = max(MIN_CAPACITY, min(MAX_CAPACITY,
                        g['capacity_ratio'] + random.uniform(-0.15, 0.15)))
                elif field == 'gen_type':
                    if random.random() < 0.10:
                        base = self._new_dct(g['name'])
                        base['capacity_ratio'] = g['capacity_ratio']
                        g = base

        return g

    def crossover(self, g1, g2):
        if get_niche(g1) == get_niche(g2) and random.random() < 0.7:
            g2 = self.mutate(g2)

        child = copy.deepcopy(g1)
        child['name'] = f"Cross_{self.generation}"

        if g1['gen_type'] != g2['gen_type']:
            if random.random() < 0.5:
                child['capacity_ratio'] = g2['capacity_ratio']
            return child

        gt = g1['gen_type']
        if gt == 'lsb':
            if random.random() < 0.5: child['step']           = g2['step']
            if random.random() < 0.5: child['edge_threshold'] = g2['edge_threshold']
            if random.random() < 0.5: child['strategy']       = g2['strategy']
        elif gt == 'dct':
            if random.random() < 0.5: child['coeff_selection'] = g2['coeff_selection']
            if random.random() < 0.5: child['strength']        = g2['strength']
        elif gt == 'fft':
            if random.random() < 0.5: child['freq_band'] = g2['freq_band']
            if random.random() < 0.5: child['strength']  = g2['strength']

        if random.random() < 0.5:
            child['capacity_ratio'] = g2['capacity_ratio']

        return child

    # ------------------------------------------------------------------ stats

    def update_batch_stats(self, names, is_fooled_list):
        for name, fooled in zip(names, is_fooled_list):
            if name in self.stats:
                self.stats[name]['attempts'] += 1
                if fooled:
                    self.stats[name]['fooled'] += 1

    # ------------------------------------------------------------------ evolution

    def _is_duplicate(self, genome, population):
        """Check if a genome's key parameters already exist in population."""
        for g in population:
            if (g['gen_type'] == genome['gen_type'] and
                    g.get('freq_band') == genome.get('freq_band') and
                    abs(g.get('strength', 0) - genome.get('strength', 0)) < 0.5 and
                    abs(g.get('capacity_ratio', 0) - genome.get('capacity_ratio', 0)) < 0.05):
                return True
        return False

    def evolve(self):
        self.generation += 1

        final_scores = {}
        for genome in self.population:
            data = self.stats[genome['name']]
            raw  = data['fooled'] / data['attempts'] if data['attempts'] > 0 else 0.0
            final_scores[genome['name']] = self._penalised_fitness(genome, raw)

        sorted_pop = sorted(self.population,
                            key=lambda g: final_scores.get(g['name'], 0.0),
                            reverse=True)

        print(f"\n[EVOLUTION] Generation {self.generation} — Top 3:")
        for i in range(min(3, len(sorted_pop))):
            g   = sorted_pop[i]
            sc  = final_scores.get(g['name'], 0.0) * 100
            d   = self.stats[g['name']]
            raw = (d['fooled'] / d['attempts'] * 100) if d['attempts'] > 0 else 0.0
            gt  = g['gen_type']
            if gt == 'lsb':
                detail = (f"Strat={g['strategy']} Step={g['step']} "
                          f"Edge={g['edge_threshold']} Cap={g['capacity_ratio']:.2f}")
            elif gt == 'dct':
                detail = (f"Coeff={g['coeff_selection']} Str={g['strength']:.1f} "
                          f"Cap={g['capacity_ratio']:.2f}")
            else:
                detail = (f"Band={g['freq_band']} Str={g['strength']:.1f} "
                          f"Cap={g['capacity_ratio']:.2f}")
            print(f"  #{i+1}: {g['name']} — {sc:.2f}% (raw {raw:.1f}%) | "
                  f"Niche={get_niche(g)} | {detail}")

        niche_counts = {n: sum(1 for g in self.population if get_niche(g) == n)
                        for n in ALL_NICHES}
        print(f"  Niches: {niche_counts}")

        # Niche preservation: MIN_NICHE_SIZE survivors per niche (all 8).
        niche_survivors = []
        for niche in ALL_NICHES:
            members = [g for g in sorted_pop if get_niche(g) == niche]
            for g in members[:MIN_NICHE_SIZE]:
                if g not in niche_survivors:
                    niche_survivors.append(g)

        elite = []
        for g in sorted_pop:
            if g not in elite:
                elite.append(g)
            if len(elite) == 3:
                break

        new_pop = list({id(g): g for g in elite + niche_survivors}.values())

        if len(sorted_pop) >= 2:
            child1 = self.crossover(sorted_pop[0], sorted_pop[1])
            if not self._is_duplicate(child1, new_pop):
                new_pop.append(child1)
            else:
                # Force diversity: if child is a clone, mutate the winner instead
                new_pop.append(self.mutate(sorted_pop[0]))

        if len(sorted_pop) >= 3:
            child2 = self.crossover(sorted_pop[0], sorted_pop[2])
            if not self._is_duplicate(child2, new_pop):
                new_pop.append(child2)
            else:
                new_pop.append(self.mutate(sorted_pop[0]))

        # Inject for two most under-represented niches.
        niche_counts_new = {n: sum(1 for g in new_pop if get_niche(g) == n)
                            for n in ALL_NICHES}
        underrepresented = sorted(niche_counts_new, key=niche_counts_new.get)
        for niche in underrepresented[:2]:
            if niche.startswith('lsb_'):
                strategy = niche.split('_', 1)[1]
                new_pop.append(self._new_lsb(f"Explore_{niche}_{self.generation}", strategy))
            elif niche == 'dct':
                new_pop.append(self._new_dct(f"Explore_dct_{self.generation}"))
            elif niche.startswith('fft_'):
                # Run 6: inject the correct FFT sub-band specifically.
                band = niche.split('_', 1)[1]   # 'low', 'mid', or 'high'
                new_pop.append(self._new_fft(f"Explore_{niche}_{self.generation}", band))

        while len(new_pop) < POPULATION_SIZE:
            parent_idx = min(random.randint(0, 2), len(sorted_pop) - 1)
            new_pop.append(self.mutate(sorted_pop[parent_idx]))

        self.population = new_pop[:POPULATION_SIZE]
        self.stats      = {g['name']: {'fooled': 0, 'attempts': 0} for g in self.population}
        return sorted_pop[0]

    # ------------------------------------------------------------------ sampling

    def get_random_genome(self):
        if self.generation == 0 or random.random() < 0.3:
            return random.choice(self.population)

        # Diversity dampener: weight / sqrt(niche_population_size).
        # With FFT split into sub-niches, fft_low clones are dampened against
        # the fft_low count only, not the entire FFT population.
        niche_counts = {}
        for g in self.population:
            n = get_niche(g)
            niche_counts[n] = niche_counts.get(n, 0) + 1

        weights = []
        for g in self.population:
            data = self.stats[g['name']]
            if data['attempts'] == 0:
                raw_weight = 0.15
            else:
                raw        = data['fooled'] / data['attempts']
                raw_weight = self._penalised_fitness(g, raw) + 0.05
            niche_size = niche_counts.get(get_niche(g), 1)
            weights.append(raw_weight / (niche_size ** 0.75))

        total   = sum(weights)
        weights = [w / total for w in weights]
        return random.choices(self.population, weights=weights, k=1)[0]

    def get_low_capacity_genome(self):
        """Return a genome with capacity_ratio < LOW_CAPACITY_THRESHOLD."""
        candidates = [g for g in self.population if is_low_capacity(g)]
        if candidates:
            return random.choice(candidates)
        choice = random.choice(['lsb_edge', 'lsb_sequential', 'fft_low'])
        if choice == 'lsb_edge':
            g = self._new_lsb("tmp_lowcap_edge", 'edge')
            g['capacity_ratio'] = 0.21
            g['edge_threshold'] = 9
        elif choice == 'lsb_sequential':
            g = self._new_lsb("tmp_lowcap_seq", 'sequential')
            g['capacity_ratio'] = 0.25
        else:
            g = self._new_fft("tmp_lowcap_fft_low", 'low')
            g['capacity_ratio'] = 0.25
            g['strength']       = 10.0
        return g

    def get_hard_edge_genome(self):
        """
        Return a lsb_edge genome with threshold<=9 AND capacity<=0.25.
        These are the exact parameters the model failed on in every run.
        Falls back to a constructed genome if none exist in the population.
        """
        candidates = [g for g in self.population if is_hard_edge(g)]
        if candidates:
            return random.choice(candidates)
        g = self._new_lsb("tmp_hard_edge", 'edge')
        g['edge_threshold'] = 9
        g['capacity_ratio'] = 0.21
        return g


# ==================== BATCH CONSTRUCTION ====================

def build_assigned_pairs(batch_files, evo_manager):
    """
    Four-layer batch diversity guarantee (run 6):

    Layer 1a — Hard edge floor (HARD_EDGE_BATCH_FRACTION = 20%):
        lsb_edge with threshold<=9 AND capacity<=0.25 every batch.
        Guarantees the model sees the specific failing eval config.

    Layer 1b — Any edge floor (ANY_EDGE_BATCH_FRACTION = 10%):
        Any lsb_edge genome. Together with 1a = 20% edge total.

    Layer 2 — Low-capacity floor (15% of free slots, capacity < 0.30).

    Layer 3 — Per-niche cap (40% of free slots per niche, 8 niches total).

    Layer 4 — FFT combined cap (30% of free slots for all FFT): NEW.
        Prevents fft_low + fft_mid + fft_high combined from dominating even
        when each is individually within its 40% per-niche cap.

    Returns (pairs, fallback_count).
    """
    n             = len(batch_files)
    n_hard_edge   = max(1, int(n * HARD_EDGE_BATCH_FRACTION))
    n_any_edge    = max(1, int(n * ANY_EDGE_BATCH_FRACTION))
    n_edge_total  = n_hard_edge + n_any_edge
    n_free        = n - n_edge_total
    n_lowcap      = max(1, int(n_free * LOW_CAPACITY_BATCH_FRACTION))
    n_normal      = n_free - n_lowcap

    all_edge_genomes = [g for g in evo_manager.population if get_niche(g) == 'lsb_edge']
    if not all_edge_genomes:
        fb = evo_manager._new_lsb("fallback_any_edge", 'edge')
        all_edge_genomes = [fb]

    pairs          = []
    fallback_count = 0

    shuffled_paths = random.sample(batch_files, len(batch_files))
    hard_paths     = shuffled_paths[:n_hard_edge]
    any_paths      = shuffled_paths[n_hard_edge: n_edge_total]
    lowcap_paths   = shuffled_paths[n_edge_total: n_edge_total + n_lowcap]
    normal_paths   = shuffled_paths[n_edge_total + n_lowcap:]

    for path in hard_paths:
        pairs.append((path, evo_manager.get_hard_edge_genome()))

    for path in any_paths:
        pairs.append((path, random.choice(all_edge_genomes)))

    for path in lowcap_paths:
        pairs.append((path, evo_manager.get_low_capacity_genome()))

    niche_used = {}
    fft_used   = 0
    niche_cap  = int(n_normal * NICHE_BATCH_CAP)
    fft_cap    = int(n_normal * FFT_COMBINED_BATCH_CAP)

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


# ==================== FIXED-SIZE BATCH ====================

def make_fixed_batch(inputs, labels, batch_genome_names):
    """
    Pad or truncate to exactly FIXED_BATCH_SIZE for CUDA graph stability.
    Padded slots use random noise tensors and have weight=0.0 so they
    contribute nothing to the loss or accuracy metrics.
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

    return (torch.stack(inputs),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(weights, dtype=torch.float32),
            batch_genome_names)


# ==================== HELPERS ====================

def generate_long_text_message(length=5000):
    chars = string.ascii_letters + string.digits + " " + ".,!?"
    return ''.join(random.choices(chars, k=length))


def load_balanced_dataset(raw_dir):
    lossy_dir    = os.path.join(raw_dir, 'flickr30k')
    lossless_dir = os.path.join(raw_dir, 'BossBase and BOWS2')

    print(f"[DATA] Scanning {raw_dir}...")
    lossy_files    = (glob.glob(os.path.join(lossy_dir, '*.jpg')) +
                      glob.glob(os.path.join(lossy_dir, '*.jpeg')))
    lossless_files = (glob.glob(os.path.join(lossless_dir, '*.pgm')) +
                      glob.glob(os.path.join(lossless_dir, '*.png')))

    print(f"[DATA] Found {len(lossy_files)} Lossy (Flickr) images.")
    print(f"[DATA] Found {len(lossless_files)} Lossless (BOSSbase) images.")
    if len(lossy_files) < BATCH_SIZE or len(lossless_files) < BATCH_SIZE:
        print("[WARN] Imbalance or missing files!")
    return lossy_files, lossless_files


def save_checkpoint(epoch, model, optimizer, best_genome, val_acc, filename="checkpoint.pth"):
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    torch.save({
        'epoch':                epoch,
        'model_state_dict':     raw_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_genome':          best_genome,
        'val_acc':              val_acc,
    }, filename)
    print(f"[CHECKPOINT] Saved to {filename}  (val_acc={val_acc:.2f}%)")


def adjust_learning_rate(optimizer, epoch):
    if epoch < 2:
        lr = INITIAL_LR + (MAX_LR - INITIAL_LR) * (epoch / 2)
    elif epoch < 12: # Hold MAX_LR during the hardest evolutionary phase
        lr = MAX_LR
    else:
        progress = (epoch - 12) / (EPOCHS - 12)
        lr = MIN_LR + 0.5 * (MAX_LR - MIN_LR) * (1 + math.cos(math.pi * progress))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def get_curriculum_blend_factor(epoch):
    blend_start = CURRICULUM_END - CURRICULUM_BLEND_EPOCHS
    if epoch < blend_start:
        return 0.0
    if epoch >= CURRICULUM_END:
        return 1.0
    return (epoch - blend_start) / CURRICULUM_BLEND_EPOCHS


# ==================== VALIDATION ====================

def run_validation(model, val_lossy, val_lossless, unified_gen, criterion, epoch):
    model.eval()
    to_tensor = transforms.ToTensor()
    rng       = random.Random(EVAL_SEED + epoch)

    val_files = val_lossy + val_lossless
    rng.shuffle(val_files)
    val_files = val_files[:750]

    all_inputs, all_labels = [], []

    for path in val_files:
        gen_type = rng.choices(ALL_GEN_TYPES, weights=GEN_TYPE_WEIGHTS)[0]

        if gen_type == 'lsb':
            config = {
                'gen_type':       'lsb',
                'strategy':       rng.choices(LSB_STRATEGIES, weights=[2, 1, 1, 2])[0],
                'capacity_ratio': rng.uniform(MIN_CAPACITY, MAX_CAPACITY),
                'edge_threshold': rng.randint(0, 100),
                'step':           rng.randint(1, 15),
                'bit_depth':      1,
                'message':        None,
            }
        elif gen_type == 'dct':
            config = {
                'gen_type':        'dct',
                'coeff_selection': rng.choice(DCT_COEFF_MODES),
                'strength':        rng.uniform(1.0, 8.0),
                'capacity_ratio':  rng.uniform(MIN_CAPACITY, MAX_CAPACITY),
            }
        else:
            config = {
                'gen_type':       'fft',
                'freq_band':      rng.choice(FFT_FREQ_BANDS),
                'strength':       rng.uniform(2.0, 20.0),
                'capacity_ratio': rng.uniform(MIN_CAPACITY, MAX_CAPACITY),
            }

        try:
            with Image.open(path) as img:
                crop_img = img.convert('L')
            w, h = crop_img.size
            if w < 256 or h < 256:
                continue
            left = (w - 256) // 2
            top = (h - 256) // 2
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
            labels_t = torch.tensor(all_labels[i: i + VAL_BATCH],
                                    dtype=torch.long).to(DEVICE)
            with torch.amp.autocast('cuda'):
                outputs  = model(inputs_t)
                per_loss = criterion(outputs, labels_t)
                loss     = per_loss.mean()
            _, preds   = torch.max(outputs, 1)
            total_loss    += loss.item() * labels_t.size(0)
            correct_total += (preds == labels_t).sum().item()
            total_samples += labels_t.size(0)

    return total_loss / total_samples, 100.0 * correct_total / total_samples


# ==================== TRAINING ====================

def run_training():
    print(f"Starting Hybrid Training Run 13 on {DEVICE}")
    print(f"Run 13 changes vs run 12:")
    print(f"  Architecture: Triple-branch frontend (SRM + spatial learnable + FFT learnable)")
    print(f"  Branch A: 11 frozen SRM filters (spatial only)")
    print(f"  Branch B: 21 learnable filters (spatial only — LSB/DCT specialization)")
    print(f"  Branch C: 32 learnable filters (FFT only — frequency ring specialization)")
    print(f"  Label smoothing: 0.1 (threshold calibration fix)")
    print(f"  CURRICULUM_END: {CURRICULUM_END}  CURRICULUM_BLEND_EPOCHS: {CURRICULUM_BLEND_EPOCHS}")

    lossy_files, lossless_files = load_balanced_dataset('data/raw')
    split = create_or_load_split(lossy_files, lossless_files)

    train_lossy    = split['lossy_train']
    train_lossless = split['lossless_train']
    val_lossy      = split['lossy_val']
    val_lossless   = split['lossless_val']

    print(f"\n[DATA] Train: {len(train_lossy)} lossy + {len(train_lossless)} lossless")
    print(f"[DATA] Val:   {len(val_lossy)} lossy + {len(val_lossless)} lossless")

    discriminator = SRNet().to(DEVICE)
    print("[INFO] Compiling model with torch.compile (reduce-overhead)...")
    discriminator = torch.compile(discriminator, mode='reduce-overhead')

    # Run 6: weight_decay doubled (1e-4 -> 2e-4) to reduce overconfidence.
    optimizer = optim.Adam(discriminator.parameters(), lr=INITIAL_LR, weight_decay=2e-4)

    # reduction='none' so padded slots can be zeroed out of the loss.
    criterion   = nn.CrossEntropyLoss(reduction='none', label_smoothing=0.1)
    scaler      = torch.amp.GradScaler('cuda')
    unified_gen = UnifiedGenerator()
    evo_manager = EvolutionaryManager()
    # to_tensor   = transforms.ToTensor()

    training_history = {
        'epochs': [], 'loss': [], 'model_acc': [], 'val_loss': [], 'val_acc': [],
        'gen_success': [], 'learning_rate': [],
        'fallback_rate': [], 'blend_factor': [], 'pad_rate': [],
    }

    min_dataset_size = min(len(train_lossy), len(train_lossless))
    steps_per_epoch  = max(1, min_dataset_size // (BATCH_SIZE // 2))
    best_val_acc     = 0.0
    best_val_epoch   = 0

    for epoch in range(EPOCHS):
        current_lr   = adjust_learning_rate(optimizer, epoch)
        blend_factor = get_curriculum_blend_factor(epoch)

        curriculum_active = blend_factor < 1.0
        track_evolution   = blend_factor > 0.0

        hard_min_capacity  = max(MIN_CAPACITY, 1.0 - (min(epoch, CURRICULUM_END - 1) * 0.08))
        min_capacity       = hard_min_capacity + blend_factor * (MIN_CAPACITY - hard_min_capacity)
        hard_max_edge      = min(70, epoch * 7)
        max_edge_threshold = int(hard_max_edge + blend_factor * (100 - hard_max_edge))

        print(f"\n{'=' * 65}")
        print(f"Epoch {epoch + 1}/{EPOCHS} | LR: {current_lr:.6f} | Blend: {blend_factor:.2f}")
        if curriculum_active:
            blend_note = (f" (blending {blend_factor*100:.0f}%)" if blend_factor > 0 else "")
            print(f"Curriculum: Cap [{min_capacity:.2f}-1.0] | "
                  f"Edge [0-{max_edge_threshold}]{blend_note}")
        else:
            print("Full evolution — no curriculum constraints.")
        print('=' * 65)

        random.shuffle(train_lossy)
        random.shuffle(train_lossless)

        total_loss      = 0.0
        correct_total   = 0
        total_samples   = 0
        epoch_fallbacks = 0
        epoch_pads      = 0

        discriminator.train()
        optimizer.zero_grad()

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            for step in range(steps_per_epoch):
                half_batch  = BATCH_SIZE // 2
                batch_files = (train_lossy[step * half_batch: (step + 1) * half_batch] +
                               train_lossless[step * half_batch: (step + 1) * half_batch])
                random.shuffle(batch_files)

                assigned_pairs, fallback_count = build_assigned_pairs(
                    batch_files, evo_manager)
                epoch_fallbacks += fallback_count

                def generate_pair(args):
                    path, genome = args
                    try:
                        with Image.open(path) as img:
                            cover_img = img.convert('L')

                        w, h = cover_img.size
                        if w < 256 or h < 256:
                            return None

                        i_c, j_c, h_c, w_c = transforms.RandomCrop.get_params(
                            cover_img, output_size=(256, 256))
                        cover_crop = TF.crop(cover_img, i_c, j_c, h_c, w_c)

                        genome_cfg = genome.copy()
                        gt         = genome_cfg['gen_type']

                        if curriculum_active:
                            genome_cfg['capacity_ratio'] = random.uniform(min_capacity, 1.0)
                            if gt == 'lsb':
                                genome_cfg['edge_threshold'] = random.randint(
                                    0, max_edge_threshold)

                        if gt == 'lsb':
                            genome_cfg['message'] = (
                                generate_long_text_message(5000)
                                if epoch >= 5 and random.random() < 0.5 else None
                            )
                            if 'capacity_ratio' not in genome_cfg:
                                genome_cfg['capacity_ratio'] = 0.5

                        stego_arr, _ = unified_gen.generate_stego(
                            cover_crop, None, genome_cfg)
                        if stego_arr is None:
                            return None

                        spatial_cover = TF.to_tensor(cover_crop)
                        log_fft_cover = compute_log_fft(spatial_cover)
                        cover_t = torch.cat([spatial_cover, log_fft_cover], dim=0)

                        spatial_stego = TF.to_tensor(Image.fromarray(stego_arr))
                        log_fft_stego = compute_log_fft(spatial_stego)
                        stego_t = torch.cat([spatial_stego, log_fft_stego], dim=0)

                        return (cover_t, stego_t, genome['name'])

                    except Exception as e:
                        print(f"\n[GEN ERROR] {genome.get('name', 'Unknown')}: {str(e)}")
                        return None

                inputs, labels, batch_genome_names = [], [], []

                for res in executor.map(generate_pair, assigned_pairs):
                    if res is None:
                        continue
                    cover_t, stego_t, g_name = res
                    inputs.extend([cover_t, stego_t])
                    labels.extend([0, 1])
                    batch_genome_names.extend([None, g_name])

                inputs_t, labels_t, weights_t, batch_genome_names = make_fixed_batch(
                    inputs, labels, batch_genome_names)

                if inputs_t is None:
                    print(f"\r[WARN] Step {step}: all generations failed — skipped.", end="")
                    continue

                n_real      = int(weights_t.sum().item())
                epoch_pads += FIXED_BATCH_SIZE - n_real

                inputs_t  = inputs_t.to(DEVICE, non_blocking=True)
                labels_t  = labels_t.to(DEVICE, non_blocking=True)
                weights_t = weights_t.to(DEVICE, non_blocking=True)

                if epoch == 0 and step == 0:
                    print("\n" + "=" * 65)
                    print("DIAGNOSTIC CHECK (Run 12: Spatial Only)")
                    print("=" * 65)

                    # FIX: Slice Channel 0 (Spatial) only.
                    # inputs_t[:n_real] has shape (B, 2, 256, 256)
                    # We take every other one (0::2 is covers, 1::2 is stegos)
                    real_data = inputs_t[:n_real]
                    covers = real_data[0::2, 0:1, :, :].cpu().numpy()
                    stegos = real_data[1::2, 0:1, :, :].cpu().numpy()

                    diff = np.abs(covers - stegos)
                    mod_rate = 100 * (diff > 0).sum() / diff.size
                    print(f"  Max Pixel Diff:    {diff.max():.6f}")  # Should now be 1/255 for 1-bit LSB
                    print(f"  Modification Rate: {mod_rate:.2f}%")
                    print("=" * 65 + "\n")

                perm             = torch.randperm(inputs_t.size(0))
                inputs_shuffled  = inputs_t[perm]
                labels_shuffled  = labels_t[perm]
                weights_shuffled = weights_t[perm]
                names_shuffled   = [batch_genome_names[i] for i in perm.tolist()]

                with torch.amp.autocast('cuda'):
                    outputs   = discriminator(inputs_shuffled)
                    per_loss  = criterion(outputs, labels_shuffled)
                    loss      = (per_loss * weights_shuffled).sum() / weights_shuffled.sum()
                    loss_accum = loss / GRADIENT_ACCUMULATION_STEPS

                scaler.scale(loss_accum).backward()

                if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    # Run 6: gradient clipping prevents spikes on distribution shifts.
                    # scaler.unscale_() converts from AMP scale to FP32 before clipping.
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        discriminator.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                _, preds = torch.max(outputs, 1)

                if track_evolution:
                    rel_names, fooled_results = [], []
                    for j, name in enumerate(names_shuffled):
                        if name is not None and weights_shuffled[j].item() > 0:
                            rel_names.append(name)
                            fooled_results.append(preds[j].item() == 0)
                    evo_manager.update_batch_stats(rel_names, fooled_results)

                real_mask      = weights_shuffled.bool()
                total_loss    += loss.item()
                correct_total += (preds[real_mask] == labels_shuffled[real_mask]).sum().item()
                total_samples += real_mask.sum().item()

                if step % 10 == 0:
                    print(f"\rStep {step}/{steps_per_epoch} | "
                          f"Loss: {loss.item():.4f} | "
                          f"Acc: {100 * correct_total / max(1, total_samples):.1f}%", end="")

        if steps_per_epoch % GRADIENT_ACCUMULATION_STEPS != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        torch.cuda.empty_cache()

        fallback_rate_pct = 100.0 * epoch_fallbacks / max(1, steps_per_epoch * BATCH_SIZE)
        pad_rate_pct      = 100.0 * epoch_pads / max(1, steps_per_epoch * FIXED_BATCH_SIZE)

        if fallback_rate_pct > 3.0:
            print(f"\n[WARN] Niche cap fallback: {fallback_rate_pct:.2f}% — "
                  "consider relaxing NICHE_BATCH_CAP or FFT_COMBINED_BATCH_CAP")
        else:
            print(f"\n[DIVERSITY] Fallback: {fallback_rate_pct:.2f}% OK  |  "
                  f"Pad rate: {pad_rate_pct:.2f}%")

        print("[VAL] Running validation...")
        val_loss, val_acc = run_validation(
            discriminator, val_lossy, val_lossless, unified_gen, criterion, epoch)
        print(f"[VAL] Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            best_val_epoch = epoch + 1
            save_checkpoint(epoch + 1, discriminator, optimizer,
                            evo_manager.population[0], val_acc, 'srnet_best_val.pth')
            print(f"[VAL] *** New best: {best_val_acc:.2f}% at epoch {best_val_epoch} ***")

        if total_samples > 0:
            avg_loss  = total_loss / steps_per_epoch
            acc_total = 100 * correct_total / total_samples

            if not track_evolution:
                print(f"[EPOCH] Loss: {avg_loss:.4f} | Train: {acc_total:.2f}% | "
                      f"Val: {val_acc:.2f}% | Curriculum")
                avg_gen_score = 0.0
            else:
                all_rates     = [d['fooled'] / d['attempts']
                                 for d in evo_manager.stats.values() if d['attempts'] > 0]
                avg_gen_score = (sum(all_rates) / len(all_rates)) if all_rates else 0.0
                blend_label   = f"blend={blend_factor*100:.0f}%" if curriculum_active else "full-evo"
                print(f"[EPOCH] Loss: {avg_loss:.4f} | Train: {acc_total:.2f}% | "
                      f"Val: {val_acc:.2f}% | GenFool: {avg_gen_score*100:.2f}% [{blend_label}]")

            training_history['epochs'].append(epoch + 1)
            training_history['loss'].append(avg_loss)
            training_history['model_acc'].append(acc_total)
            training_history['val_loss'].append(val_loss)
            training_history['val_acc'].append(val_acc)
            training_history['gen_success'].append(avg_gen_score * 100)
            training_history['learning_rate'].append(current_lr)
            training_history['fallback_rate'].append(round(fallback_rate_pct, 3))
            training_history['blend_factor'].append(round(blend_factor, 3))
            training_history['pad_rate'].append(round(pad_rate_pct, 3))

        if track_evolution:
            best_genome = evo_manager.evolve()
        else:
            evo_manager.generation += 1
            evo_manager.stats = {g['name']: {'fooled': 0, 'attempts': 0}
                                 for g in evo_manager.population}
            best_genome = evo_manager.population[0]

        if (epoch + 1) % 5 == 0:
            save_checkpoint(epoch + 1, discriminator, optimizer, best_genome, val_acc,
                            f"srnet_epoch_{epoch + 1}.pth")

    print(f"\n[INFO] Best val accuracy: {best_val_acc:.2f}% at epoch {best_val_epoch}")
    print("[INFO] Best model saved as: srnet_best_val.pth")
    print("[INFO] Use srnet_best_val.pth (not srnet_epoch_40.pth) as input to evaluate.py")

    with open('training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    print("[INFO] Training Complete.")


if __name__ == "__main__":
    run_training()
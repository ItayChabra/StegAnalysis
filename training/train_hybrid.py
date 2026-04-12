import multiprocessing
import os
import sys

# Fix the path so Python can find custom modules when run from any working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ── A100 memory allocator: eliminates fragmentation on long runs ──────────────
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

# ── A100 / Ampere: TF32 gives ~3× matmul throughput with negligible loss ──────
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True

# --- SETTINGS ---
BATCH_SIZE                  = 64
GRADIENT_ACCUMULATION_STEPS = 2    # Effective batch = 128
EPOCHS                      = 40
POPULATION_SIZE             = 20
NUM_WORKERS  = max(1, multiprocessing.cpu_count() - 2)
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cudnn.benchmark = True

INITIAL_LR   = 0.0001
MAX_LR       = 0.001
MIN_CAPACITY = 0.20
MAX_CAPACITY = 0.75

# ---- Generator taxonomy -------------------------------------------------------
ALL_GEN_TYPES   = ['lsb', 'dct', 'fft']
LSB_STRATEGIES  = ['random', 'sequential', 'skip', 'edge']
DCT_COEFF_MODES = ['mid', 'low_mid', 'random']
# 'low' removed — it dominated run 2 and isn't in the evaluation set.
# Evolution can still discover it via mutation, but we don't seed or protect it.
FFT_FREQ_BANDS  = ['mid', 'high']

GEN_TYPE_WEIGHTS = [0.50, 0.25, 0.25]

ALL_NICHES = ['lsb_random', 'lsb_sequential', 'lsb_skip', 'lsb_edge', 'dct', 'fft']

MIN_NICHE_SIZE             = 2
CAPACITY_PENALTY_THRESHOLD = MIN_CAPACITY + 0.10
CAPACITY_PENALTY_WEIGHT    = 0.15

SKIP_SEED_STEPS = [2, 3, 5, 7, 11]

# Guaranteed fraction of each batch reserved for the weakest niche (lsb_edge).
EDGE_BATCH_FRACTION = 0.20
# No single niche may consume more than this fraction of the free slots.
NICHE_BATCH_CAP     = 0.40

# Fixed seed for validation — same epoch always produces the same val set,
# so best-checkpoint tracking is meaningful across runs.
EVAL_SEED = 99

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
SPLIT_FILE  = 'dataset_split.json'
SPLIT_SEED  = 42


# ==================== GENOME HELPERS ====================

def get_niche(genome):
    """Map a genome to its niche label for niche preservation."""
    gt = genome.get('gen_type', 'lsb')
    if gt == 'lsb':
        return f"lsb_{genome.get('strategy', 'random')}"
    return gt   # 'dct' or 'fft'


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
    Manages a mixed population of LSB, DCT, and FFT generator genomes.

    Key design decisions vs run 2:
    - fft_low is NOT seeded — it dominated run 2's evolution and crowded out
      edge/high-frequency genomes. Evolution can still discover it via mutation.
    - 4 targeted lsb_edge seeds (thresholds 5, 9, 15, 30) replace the single
      generic edge seed, giving the model specific hard-to-detect patterns to
      learn from epoch 1.
    - get_random_genome() applies a diversity dampener (divide by sqrt of niche
      population size) so a monoculture cannot capture >40% of sampling weight.
    """

    def __init__(self):
        self.population = []
        self.generation  = 0

        # --- LSB: one seed per non-edge strategy ---
        for strategy in ['random', 'sequential', 'skip']:
            self.population.append(self._new_lsb(f"Seed_lsb_{strategy}", strategy))

        # --- LSB edge: 4 seeds at the thresholds the evaluator uses ---
        for threshold in [5, 9, 15, 30]:
            g = self._new_lsb(f"Seed_lsb_edge_t{threshold}", 'edge')
            g['edge_threshold'] = threshold
            self.population.append(g)

        # --- LSB skip: diverse step sizes ---
        for s in SKIP_SEED_STEPS:
            g = self._new_lsb(f"Seed_skip_s{s}", 'skip')
            g['step'] = s
            self.population.append(g)

        # --- DCT across coefficient modes ---
        for mode in DCT_COEFF_MODES:
            self.population.append(self._new_dct(f"Seed_dct_{mode}", mode))

        # --- FFT: mid and high only (low excluded from seeding) ---
        for band in FFT_FREQ_BANDS:
            self.population.append(self._new_fft(f"Seed_fft_{band}", band))

        # Fill remainder with random genomes.
        while len(self.population) < POPULATION_SIZE:
            idx = len(self.population)
            self.population.append(self._generate_random_genome(f"Gen_{idx}"))

        self.stats = {g['name']: {'fooled': 0, 'attempts': 0} for g in self.population}

    # ------------------------------------------------------------------ constructors

    def _new_lsb(self, name, strategy=None):
        return {
            'name':           name,
            'gen_type':       'lsb',
            'strategy':       strategy or random.choice(LSB_STRATEGIES),
            'step':           random.randint(1, 15),
            'bit_depth':      1,
            'edge_threshold': random.randint(0, 100),
            'capacity_ratio': random.uniform(MIN_CAPACITY, MAX_CAPACITY),
        }

    def _new_dct(self, name, coeff_selection=None):
        return {
            'name':            name,
            'gen_type':        'dct',
            'coeff_selection': coeff_selection or random.choice(DCT_COEFF_MODES),
            'strength':        round(random.uniform(1.0, 8.0), 2),
            'capacity_ratio':  random.uniform(MIN_CAPACITY, MAX_CAPACITY),
        }

    def _new_fft(self, name, freq_band=None):
        return {
            'name':           name,
            'gen_type':       'fft',
            # Default draw from the seeded bands (mid/high); mutation can introduce low.
            'freq_band':      freq_band or random.choice(FFT_FREQ_BANDS),
            'strength':       round(random.uniform(2.0, 20.0), 2),
            'capacity_ratio': random.uniform(MIN_CAPACITY, MAX_CAPACITY),
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
        n_mutations = 2 if random.random() < 0.2 else 1

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
                        base     = self._new_dct(g['name']) if new_type == 'dct' else self._new_fft(g['name'])
                        base['capacity_ratio'] = g['capacity_ratio']
                        g = base

            elif gt == 'dct':
                field = random.choice(['coeff', 'strength', 'capacity', 'gen_type'])
                if field == 'coeff':
                    g['coeff_selection'] = random.choice(DCT_COEFF_MODES)
                elif field == 'strength':
                    g['strength'] = max(0.5, min(10.0,
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
                    # All three bands available via mutation — low can re-enter this way.
                    g['freq_band'] = random.choice(['low', 'mid', 'high'])
                elif field == 'strength':
                    g['strength'] = max(1.0, min(25.0,
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
        """Crossover only mixes genomes of the same gen_type."""
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
                detail = f"Strat: {g['strategy']} | Step: {g['step']} | Edge: {g['edge_threshold']}"
            elif gt == 'dct':
                detail = f"Coeff: {g['coeff_selection']} | Strength: {g['strength']:.1f}"
            else:
                detail = f"Band: {g['freq_band']} | Strength: {g['strength']:.1f}"
            print(f"  #{i+1}: {g['name']} — {sc:.2f}% (raw {raw:.1f}%) | "
                  f"Type: {gt} | Cap: {g['capacity_ratio']:.2f} | {detail}")

        # Print niche coverage summary so monoculture is immediately visible
        niche_counts = {n: sum(1 for g in self.population if get_niche(g) == n)
                        for n in ALL_NICHES}
        print(f"  Niche coverage: { {k: v for k, v in niche_counts.items()} }")

        # --- Niche preservation ---
        niche_survivors = []
        for niche in ALL_NICHES:
            members = [g for g in sorted_pop if get_niche(g) == niche]
            for g in members[:MIN_NICHE_SIZE]:
                if g not in niche_survivors:
                    niche_survivors.append(g)

        # Elite: top-3 overall.
        elite = []
        for g in sorted_pop:
            if g not in elite:
                elite.append(g)
            if len(elite) == 3:
                break

        new_pop = list({id(g): g for g in elite + niche_survivors}.values())

        # Crossover from top parents.
        if len(sorted_pop) >= 2:
            new_pop.append(self.crossover(sorted_pop[0], sorted_pop[1]))
        if len(sorted_pop) >= 3:
            new_pop.append(self.crossover(sorted_pop[0], sorted_pop[2]))

        # Inject fresh genomes for the two most under-represented niches.
        niche_counts_new = {n: sum(1 for g in new_pop if get_niche(g) == n)
                            for n in ALL_NICHES}
        underrepresented = sorted(niche_counts_new, key=niche_counts_new.get)
        for niche in underrepresented[:2]:
            if niche.startswith('lsb_'):
                strategy = niche.split('_', 1)[1]
                new_pop.append(self._new_lsb(f"Explore_{niche}_{self.generation}", strategy))
            elif niche == 'dct':
                new_pop.append(self._new_dct(f"Explore_dct_{self.generation}"))
            else:
                new_pop.append(self._new_fft(f"Explore_fft_{self.generation}"))

        # Fill remainder with mutations biased toward fit parents.
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

        # Count population per niche for diversity dampening.
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

            # Diversity dampener: divide by sqrt(niche population size).
            # A niche with 14 clones gets weight / 3.7.
            # A niche with 2 genomes gets weight / 1.4.
            # Prevents any single generator from monopolising batch sampling.
            niche_size = niche_counts.get(get_niche(g), 1)
            weights.append(raw_weight / (niche_size ** 0.5))

        total   = sum(weights)
        weights = [w / total for w in weights]
        return random.choices(self.population, weights=weights, k=1)[0]


# ==================== BATCH CONSTRUCTION ====================

def build_assigned_pairs(batch_files, evo_manager):
    """
    Builds (image_path, genome) pairs with guaranteed diversity.

    - EDGE_BATCH_FRACTION of slots are always filled with lsb_edge genomes,
      ensuring the model sees edge steganography every single batch regardless
      of evolutionary fitness scores.
    - The remaining free slots are filled from standard weighted sampling, but
      no single niche may exceed NICHE_BATCH_CAP of those free slots.

    This directly prevents the fft_low monoculture seen in run 2, where one
    genome captured ~80% of batch composition after epoch 11.
    """
    n        = len(batch_files)
    n_edge   = max(2, int(n * EDGE_BATCH_FRACTION))
    n_free   = n - n_edge

    edge_genomes = [g for g in evo_manager.population if get_niche(g) == 'lsb_edge']
    if not edge_genomes:
        # Fallback: create a temporary edge genome if evolution displaced all of them.
        fallback = evo_manager._new_lsb("fallback_edge", 'edge')
        fallback['edge_threshold'] = 9
        edge_genomes = [fallback]

    pairs = []

    # Guaranteed edge slots (shuffled paths so they're not all from one source).
    edge_paths = random.sample(batch_files, n_edge)
    for path in edge_paths:
        pairs.append((path, random.choice(edge_genomes)))

    # Free slots with per-niche cap.
    free_paths  = [p for p in batch_files if p not in edge_paths]
    niche_used  = {}
    niche_cap   = int(n_free * NICHE_BATCH_CAP)

    for path in free_paths:
        placed = False
        for _ in range(15):   # up to 15 attempts to find a diverse genome
            g     = evo_manager.get_random_genome()
            niche = get_niche(g)
            if niche_used.get(niche, 0) < niche_cap:
                niche_used[niche] = niche_used.get(niche, 0) + 1
                pairs.append((path, g))
                placed = True
                break
        if not placed:
            # All niches are capped — just pick randomly (rare).
            pairs.append((path, evo_manager.get_random_genome()))

    random.shuffle(pairs)
    return pairs


# ==================== HELPERS ====================

def generate_long_text_message(length=5000):
    chars = string.ascii_letters + string.digits + " " + ".,!?"
    return ''.join(random.choices(chars, k=length))


def load_balanced_dataset(raw_dir):
    lossy_dir    = os.path.join(raw_dir, 'flickr30k')
    lossless_dir = os.path.join(raw_dir, 'BossBase and BOWS2')

    print(f"[DATA] Scanning specific folders in {raw_dir}...")
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
    # Unwrap compiled model before saving so the checkpoint is portable.
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
    elif epoch < 20:
        lr = MAX_LR
    else:
        lr = MAX_LR * 0.97 ** (epoch - 20)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


# ==================== VALIDATION ====================

def run_validation(model, val_lossy, val_lossless, unified_gen, criterion, epoch):
    model.eval()
    to_tensor = transforms.ToTensor()

    # Fixed seed: EVAL_SEED + epoch ensures the same epoch always produces the
    # same validation set, making best-checkpoint tracking meaningful.
    # Previous runs used random.Random(epoch) alone, causing wild swings
    # (e.g. 57% → 90% → 57% in consecutive epochs) from strategy sampling variance.
    rng = random.Random(EVAL_SEED + epoch)

    val_files = val_lossy + val_lossless
    rng.shuffle(val_files)
    val_files = val_files[:750]   # increased from 500 to reduce per-epoch variance

    all_inputs = []
    all_labels = []

    for path in val_files:
        gen_type = rng.choices(ALL_GEN_TYPES, weights=GEN_TYPE_WEIGHTS)[0]

        if gen_type == 'lsb':
            # lsb_edge and lsb_random weighted 2x to reflect their importance.
            LSB_STRATEGY_WEIGHTS = [2, 1, 1, 2]  # [random, sequential, skip, edge]
            config = {
                'gen_type':       'lsb',
                'strategy':       rng.choices(LSB_STRATEGIES, weights=LSB_STRATEGY_WEIGHTS)[0],
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
                'freq_band':      rng.choice(['low', 'mid', 'high']),  # all bands in val
                'strength':       rng.uniform(2.0, 20.0),
                'capacity_ratio': rng.uniform(MIN_CAPACITY, MAX_CAPACITY),
            }

        try:
            img  = Image.open(path).convert('L')
            w, h = img.size
            if w < 256 or h < 256:
                continue
            left = (w - 256) // 2
            top  = (h - 256) // 2
            crop = img.crop((left, top, left + 256, top + 256))

            stego_arr, _ = unified_gen.generate_stego(crop, None, config)
            if stego_arr is None:
                continue

            all_inputs.append(to_tensor(crop))
            all_labels.append(0)
            all_inputs.append(to_tensor(Image.fromarray(stego_arr)))
            all_labels.append(1)
        except Exception:
            continue

    if not all_inputs:
        return 0.0, 0.0

    total_loss    = 0.0
    correct_total = 0
    total_samples = 0
    VAL_BATCH     = 64

    with torch.no_grad():
        for i in range(0, len(all_inputs), VAL_BATCH):
            inputs_t = torch.stack(all_inputs[i: i + VAL_BATCH]).pin_memory().to(DEVICE, non_blocking=True)
            labels_t = torch.tensor(all_labels[i: i + VAL_BATCH],
                                    dtype=torch.long).pin_memory().to(DEVICE, non_blocking=True)
            with torch.amp.autocast('cuda'):
                outputs = model(inputs_t)
                loss    = criterion(outputs, labels_t)
            _, preds   = torch.max(outputs, 1)
            total_loss    += loss.item() * labels_t.size(0)
            correct_total += (preds == labels_t).sum().item()
            total_samples += labels_t.size(0)

    return total_loss / total_samples, 100.0 * correct_total / total_samples


# ==================== TRAINING ====================

def run_training():
    print(f"Starting Hybrid Training on {DEVICE}")
    print(f"Generator types: {ALL_GEN_TYPES}")
    print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS} "
          f"({BATCH_SIZE} x {GRADIENT_ACCUMULATION_STEPS} accumulation steps)")
    print(f"Worker threads: {NUM_WORKERS}  |  TF32: enabled  |  torch.compile: enabled")
    print(f"Edge batch floor: {EDGE_BATCH_FRACTION*100:.0f}%  |  "
          f"Niche batch cap: {NICHE_BATCH_CAP*100:.0f}%  |  "
          f"Val images: 750  |  Val seed: EVAL_SEED+epoch")

    lossy_files, lossless_files = load_balanced_dataset('data/raw')
    split = create_or_load_split(lossy_files, lossless_files)

    train_lossy    = split['lossy_train']
    train_lossless = split['lossless_train']
    val_lossy      = split['lossy_val']
    val_lossless   = split['lossless_val']

    print(f"\n[DATA] Training on   {len(train_lossy)} lossy + {len(train_lossless)} lossless")
    print(f"[DATA] Validating on {len(val_lossy)} lossy + {len(val_lossless)} lossless")

    discriminator = SRNet().to(DEVICE)

    print("[INFO] Compiling model with torch.compile (reduce-overhead)…")
    discriminator = torch.compile(discriminator, mode='reduce-overhead')

    optimizer   = optim.Adam(discriminator.parameters(), lr=INITIAL_LR, weight_decay=1e-4)
    criterion   = nn.CrossEntropyLoss()
    scaler      = torch.amp.GradScaler('cuda')
    unified_gen = UnifiedGenerator()
    evo_manager = EvolutionaryManager()
    to_tensor   = transforms.ToTensor()

    training_history = {
        'epochs': [], 'loss': [], 'model_acc': [],
        'val_loss': [], 'val_acc': [],
        'gen_success': [], 'learning_rate': []
    }

    min_dataset_size = min(len(train_lossy), len(train_lossless))
    steps_per_epoch  = max(1, min_dataset_size // (BATCH_SIZE // 2))
    best_val_acc     = 0.0
    best_val_epoch   = 0

    for epoch in range(EPOCHS):
        current_lr = adjust_learning_rate(optimizer, epoch)

        if epoch < 10:
            min_capacity       = max(MIN_CAPACITY, 1.0 - (epoch * 0.08))
            max_edge_threshold = min(70, epoch * 7)
            curriculum_active  = True
        else:
            min_capacity       = MIN_CAPACITY
            max_edge_threshold = 100
            curriculum_active  = False

        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{EPOCHS} | LR: {current_lr:.6f}")
        if curriculum_active:
            print(f"Curriculum: Cap [{min_capacity:.2f}-1.0] | LSB Edge [0-{max_edge_threshold}]")
        else:
            print('EVOLUTION ACTIVATED — Full competitive training begins!'
                  if epoch == 10 else 'Evolution Active')
        print('=' * 60)

        random.shuffle(train_lossy)
        random.shuffle(train_lossless)
        total_loss    = 0
        correct_total = 0
        total_samples = 0
        discriminator.train()
        optimizer.zero_grad()

        for step in range(steps_per_epoch):
            half_batch  = BATCH_SIZE // 2
            batch_files = (train_lossy[step * half_batch: (step + 1) * half_batch] +
                           train_lossless[step * half_batch: (step + 1) * half_batch])
            random.shuffle(batch_files)

            # Diversity-guaranteed batch: 20% edge floor, 40% per-niche cap.
            assigned_pairs = build_assigned_pairs(batch_files, evo_manager)

            def generate_pair(args):
                path, genome = args
                try:
                    cover_img = Image.open(path).convert('L')
                    w, h = cover_img.size
                    if w < 256 or h < 256:
                        return None

                    i_crop, j_crop, h_crop, w_crop = transforms.RandomCrop.get_params(
                        cover_img, output_size=(256, 256))
                    cover_crop = TF.crop(cover_img, i_crop, j_crop, h_crop, w_crop)

                    genome_cfg = genome.copy()
                    gt         = genome_cfg['gen_type']

                    if curriculum_active:
                        genome_cfg['capacity_ratio'] = random.uniform(min_capacity, 1.0)
                        if gt == 'lsb':
                            genome_cfg['edge_threshold'] = random.randint(0, max_edge_threshold)
                        # DCT/FFT: capacity curriculum only, no sub-strategy override.

                    if gt == 'lsb':
                        genome_cfg['message'] = (
                            generate_long_text_message(5000)
                            if epoch >= 5 and random.random() < 0.5 else None
                        )
                        if 'capacity_ratio' not in genome_cfg:
                            genome_cfg['capacity_ratio'] = 0.5

                    stego_arr, _ = unified_gen.generate_stego(cover_crop, None, genome_cfg)
                    if stego_arr is None:
                        return None

                    return (to_tensor(cover_crop),
                            to_tensor(Image.fromarray(stego_arr)),
                            genome['name'])
                except Exception:
                    return None

            inputs             = []
            labels             = []
            batch_genome_names = []

            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                for res in executor.map(generate_pair, assigned_pairs):
                    if res is None:
                        continue
                    cover_t, stego_t, g_name = res
                    inputs.extend([cover_t, stego_t])
                    labels.extend([0, 1])
                    batch_genome_names.extend([None, g_name])

            if not inputs:
                continue

            inputs_t = torch.stack(inputs).pin_memory().to(DEVICE, non_blocking=True)
            labels_t = torch.tensor(labels, dtype=torch.long).pin_memory().to(DEVICE, non_blocking=True)

            if epoch == 0 and step == 0:
                print("\n" + "=" * 60)
                print("DIAGNOSTIC CHECK")
                print("=" * 60)
                covers   = inputs_t[0::2].cpu().numpy()
                stegos   = inputs_t[1::2].cpu().numpy()
                diff     = np.abs(covers - stegos)
                mod_rate = 100 * (diff > 0).sum() / diff.size
                print(f"  Max Pixel Diff:      {diff.max():.6f}")
                print(f"  Mean Pixel Diff:     {diff.mean():.6f}")
                print(f"  Pixels Modified:     {(diff > 0).sum():,} / {diff.size:,}")
                print(f"  Modification Rate:   {mod_rate:.2f}%")
                print(f"  Batch:               {labels.count(0)} covers, {labels.count(1)} stegos")
                print("=" * 60 + "\n")

            perm                  = torch.randperm(inputs_t.size(0))
            inputs_shuffled       = inputs_t[perm]
            labels_shuffled       = labels_t[perm]
            shuffled_genome_names = [batch_genome_names[idx] for idx in perm.tolist()]

            with torch.amp.autocast('cuda'):
                outputs = discriminator(inputs_shuffled)
                loss    = criterion(outputs, labels_shuffled) / GRADIENT_ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            _, preds = torch.max(outputs, 1)

            if not curriculum_active:
                relevant_names, fooled_results = [], []
                for j, name in enumerate(shuffled_genome_names):
                    if name is not None:
                        relevant_names.append(name)
                        fooled_results.append(preds[j].item() == 0)
                evo_manager.update_batch_stats(relevant_names, fooled_results)

            total_loss    += loss.item() * GRADIENT_ACCUMULATION_STEPS
            correct_total += (preds == labels_shuffled).sum().item()
            total_samples += labels_shuffled.size(0)

            if step % 10 == 0:
                print(f"\rStep {step}/{steps_per_epoch} | "
                      f"Loss: {loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f} | "
                      f"Acc: {100 * correct_total / total_samples:.1f}%", end="")

        if steps_per_epoch % GRADIENT_ACCUMULATION_STEPS != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        torch.cuda.empty_cache()

        print(f"\n[VAL] Running validation...")
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

            if curriculum_active:
                print(f"[EPOCH SUMMARY] Loss: {avg_loss:.4f} | "
                      f"Train: {acc_total:.2f}% | Val: {val_acc:.2f}% | Curriculum Active")
                avg_gen_score = 0.0
            else:
                all_rates     = [d['fooled'] / d['attempts']
                                 for d in evo_manager.stats.values() if d['attempts'] > 0]
                avg_gen_score = (sum(all_rates) / len(all_rates)) if all_rates else 0.0
                print(f"[EPOCH SUMMARY] Loss: {avg_loss:.4f} | "
                      f"Train: {acc_total:.2f}% | Val: {val_acc:.2f}% | "
                      f"Gen Fool: {avg_gen_score * 100:.2f}%")

            training_history['epochs'].append(epoch + 1)
            training_history['loss'].append(avg_loss)
            training_history['model_acc'].append(acc_total)
            training_history['val_loss'].append(val_loss)
            training_history['val_acc'].append(val_acc)
            training_history['gen_success'].append(avg_gen_score * 100)
            training_history['learning_rate'].append(current_lr)

        if not curriculum_active:
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
    print(f"[INFO] Best model saved as: srnet_best_val.pth")
    print(f"[INFO] Use srnet_best_val.pth (not srnet_epoch_40.pth) as input to evaluate.py")

    with open('training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    print("[INFO] Training Complete.")


if __name__ == "__main__":
    run_training()
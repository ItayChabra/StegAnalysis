import multiprocessing
import os

# Fix CUDA memory fragmentation BEFORE importing torch.
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

# --- SETTINGS ---
BATCH_SIZE = 64
GRADIENT_ACCUMULATION_STEPS = 2   # Effective batch = 128, zero OOM risk
EPOCHS = 30
POPULATION_SIZE = 20
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 2)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cudnn.benchmark = True
INITIAL_LR = 0.0001
MAX_LR = 0.001
MIN_CAPACITY = 0.20
MAX_CAPACITY = 0.75

ALL_STRATEGIES = ['random', 'sequential', 'skip', 'edge']
MIN_NICHE_SIZE = 2
CAPACITY_PENALTY_THRESHOLD = MIN_CAPACITY + 0.10
CAPACITY_PENALTY_WEIGHT = 0.15

# Diverse skip seeds so the evolution has real signal on step geometry
# from the very first competitive epoch.
SKIP_SEED_STEPS = [2, 3, 5, 7, 11]

# Dataset split ratios — must sum to 1.0
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# Where the split is saved so evaluate.py can load it.
SPLIT_FILE  = 'dataset_split.json'

# Fix this seed permanently after the first run — changing it would leak
# test images into training on a subsequent run.
SPLIT_SEED  = 42


# ==================== DATASET SPLIT ====================

def create_or_load_split(lossy_files, lossless_files):
    """
    Create a deterministic 70/15/15 train/val/test split and save it to
    SPLIT_FILE, or load an existing split if one already exists.

    Saves absolute paths so evaluate.py resolves files correctly regardless
    of working directory.

    IMPORTANT: Once the split file exists it is NEVER regenerated, even if
    the dataset grows. This guarantees the test set is genuinely held out
    and was never seen during training.
    """
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
        'split_seed':       SPLIT_SEED,
        'train_ratio':      TRAIN_RATIO,
        'val_ratio':        VAL_RATIO,
        'test_ratio':       TEST_RATIO,
        'lossy_train':      lossy_train,
        'lossy_val':        lossy_val,
        'lossy_test':       lossy_test,
        'lossless_train':   lossless_train,
        'lossless_val':     lossless_val,
        'lossless_test':    lossless_test,
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
    def __init__(self):
        self.population = []

        for strategy in ['random', 'sequential', 'edge']:
            self.population.append(self._generate_random_genome(f"Seed_{strategy}", strategy))

        for s in SKIP_SEED_STEPS:
            g = self._generate_random_genome(f"Seed_skip_s{s}", 'skip')
            g['step'] = s
            self.population.append(g)

        while len(self.population) < POPULATION_SIZE:
            idx = len(self.population)
            self.population.append(self._generate_random_genome(f"Gen_{idx}"))

        self.stats = {g['name']: {'fooled': 0, 'attempts': 0} for g in self.population}
        self.generation = 0

    def _generate_random_genome(self, name, strategy=None):
        genome = {'name': name, 'gen_type': 'lsb'}
        genome.update({
            'strategy':       strategy if strategy else random.choice(ALL_STRATEGIES),
            'step':           random.randint(1, 15),
            'bit_depth':      1,
            'edge_threshold': random.randint(0, 100),
            'capacity_ratio': random.uniform(MIN_CAPACITY, MAX_CAPACITY),
        })
        return genome

    def _penalised_fitness(self, genome, raw_fool_rate):
        capacity = genome.get('capacity_ratio', 0.5)
        if capacity < CAPACITY_PENALTY_THRESHOLD:
            shortfall = (CAPACITY_PENALTY_THRESHOLD - capacity) / CAPACITY_PENALTY_THRESHOLD
            penalty   = shortfall * CAPACITY_PENALTY_WEIGHT
        else:
            penalty = 0.0
        return max(0.0, raw_fool_rate - penalty)

    def mutate(self, genome):
        new_genome = copy.deepcopy(genome)
        new_genome['name'] = f"{genome['name']}_m{self.generation}"
        num_mutations = 2 if random.random() < 0.2 else 1
        for _ in range(num_mutations):
            mutation = random.choice(['step', 'threshold', 'strategy', 'capacity'])
            if mutation == 'step':
                new_genome['step'] = max(1, min(20, new_genome['step'] + random.choice([-2, -1, 1, 2, 3])))
            elif mutation == 'threshold':
                new_genome['edge_threshold'] = max(0, min(100,
                    new_genome['edge_threshold'] + random.randint(-20, 20)))
            elif mutation == 'strategy':
                new_genome['strategy'] = random.choice(ALL_STRATEGIES)
            elif mutation == 'capacity':
                new_genome['capacity_ratio'] = max(MIN_CAPACITY, min(MAX_CAPACITY,
                    new_genome['capacity_ratio'] + random.uniform(-0.15, 0.15)))
        return new_genome

    def crossover(self, genome1, genome2):
        new_genome = copy.deepcopy(genome1)
        new_genome['name'] = f"Cross_{self.generation}"
        if random.random() < 0.5: new_genome['step']           = genome2['step']
        if random.random() < 0.5: new_genome['edge_threshold'] = genome2['edge_threshold']
        if random.random() < 0.5: new_genome['strategy']       = genome2['strategy']
        if random.random() < 0.5: new_genome['capacity_ratio'] = genome2['capacity_ratio']
        return new_genome

    def update_batch_stats(self, names, is_fooled_list):
        for name, fooled in zip(names, is_fooled_list):
            if name in self.stats:
                self.stats[name]['attempts'] += 1
                if fooled:
                    self.stats[name]['fooled'] += 1

    def evolve(self):
        self.generation += 1
        final_scores = {}
        for genome in self.population:
            name = genome['name']
            data = self.stats[name]
            raw  = data['fooled'] / data['attempts'] if data['attempts'] > 0 else 0.0
            final_scores[name] = self._penalised_fitness(genome, raw)

        sorted_pop = sorted(self.population,
                            key=lambda g: final_scores.get(g['name'], 0.0),
                            reverse=True)

        print(f"\n[EVOLUTION] Generation {self.generation} - Top 3 (penalised fitness):")
        for i in range(min(3, len(sorted_pop))):
            g        = sorted_pop[i]
            score    = final_scores.get(g['name'], 0.0) * 100
            raw_data = self.stats[g['name']]
            raw_rate = (raw_data['fooled'] / raw_data['attempts'] * 100) if raw_data['attempts'] > 0 else 0.0
            print(f"  #{i+1}: {g['name']} - {score:.2f}% (raw {raw_rate:.1f}%) | "
                  f"Strat: {g['strategy']} | Cap: {g['capacity_ratio']:.2f} | "
                  f"Step: {g['step']} | Edge: {g['edge_threshold']}")

        niche_survivors = []
        for strategy in ALL_STRATEGIES:
            strategy_members = [g for g in sorted_pop if g.get('strategy') == strategy]
            for g in strategy_members[:MIN_NICHE_SIZE]:
                if g not in niche_survivors:
                    niche_survivors.append(g)

        elite = []
        for g in sorted_pop:
            if g not in elite:
                elite.append(g)
            if len(elite) == 3:
                break

        new_pop = list({id(g): g for g in elite + niche_survivors}.values())
        if len(sorted_pop) >= 2: new_pop.append(self.crossover(sorted_pop[0], sorted_pop[1]))
        if len(sorted_pop) >= 3: new_pop.append(self.crossover(sorted_pop[0], sorted_pop[2]))

        strategy_counts  = {s: sum(1 for g in new_pop if g.get('strategy') == s) for s in ALL_STRATEGIES}
        underrepresented = sorted(strategy_counts, key=strategy_counts.get)
        for strategy in underrepresented[:2]:
            new_pop.append(self._generate_random_genome(f"Explore_{strategy}_{self.generation}", strategy))

        while len(new_pop) < POPULATION_SIZE:
            parent_idx = min(random.randint(0, 2), len(sorted_pop) - 1)
            new_pop.append(self.mutate(sorted_pop[parent_idx]))

        self.population = new_pop[:POPULATION_SIZE]
        self.stats = {g['name']: {'fooled': 0, 'attempts': 0} for g in self.population}
        return sorted_pop[0]

    def get_random_genome(self):
        if self.generation == 0 or random.random() < 0.3:
            return random.choice(self.population)

        weights = []
        for g in self.population:
            data = self.stats[g['name']]
            if data['attempts'] == 0:
                weights.append(0.15)
            else:
                raw   = data['fooled'] / data['attempts']
                score = self._penalised_fitness(g, raw)
                weights.append(score + 0.05)

        total   = sum(weights)
        weights = [w / total for w in weights]
        return random.choices(self.population, weights=weights, k=1)[0]


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
    torch.save({
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
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
    """
    Run one pass over the validation split.

    Uses random strategy selection with a fixed per-epoch seed so val
    scores are directly comparable across epochs. The val set is never
    touched by the evolutionary manager.

    Returns (val_loss, val_acc).
    """
    model.eval()
    to_tensor = transforms.ToTensor()
    rng       = random.Random(epoch)  # reproducible per epoch

    val_files = val_lossy + val_lossless
    rng.shuffle(val_files)
    val_files = val_files[:500]       # cap for speed — ~1-2 min per epoch

    all_inputs = []
    all_labels = []

    for path in val_files:
        strategy = rng.choice(ALL_STRATEGIES)
        config   = {
            'gen_type':       'lsb',
            'strategy':       strategy,
            'capacity_ratio': rng.uniform(MIN_CAPACITY, MAX_CAPACITY),
            'edge_threshold': rng.randint(0, 100),
            'step':           rng.randint(1, 15),
            'bit_depth':      1,
            'message':        None,
        }
        try:
            img  = Image.open(path).convert('L')
            w, h = img.size
            if w < 256 or h < 256:
                continue

            # Center crop — keeps validation deterministic (no random crop).
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
            batch_inputs = all_inputs[i: i + VAL_BATCH]
            batch_labels = all_labels[i: i + VAL_BATCH]

            inputs_t = torch.stack(batch_inputs).to(DEVICE, non_blocking=True)
            labels_t = torch.tensor(batch_labels, dtype=torch.long).to(DEVICE, non_blocking=True)

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
    print(f"Starting Hybrid 50/50 Training on {DEVICE}")
    print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS} "
          f"({BATCH_SIZE} x {GRADIENT_ACCUMULATION_STEPS} accumulation steps)")

    # torch.compile disabled on Windows — requires Triton (Linux/WSL2 only).
    # discriminator = torch.compile(discriminator)

    lossy_files, lossless_files = load_balanced_dataset('data/raw')

    # Create/load split BEFORE any training — test set is never touched here.
    split = create_or_load_split(lossy_files, lossless_files)

    train_lossy    = split['lossy_train']
    train_lossless = split['lossless_train']
    val_lossy      = split['lossy_val']
    val_lossless   = split['lossless_val']
    # test sets belong exclusively to evaluate.py — do not load them here.

    print(f"\n[DATA] Training on   {len(train_lossy)} lossy + {len(train_lossless)} lossless")
    print(f"[DATA] Validating on {len(val_lossy)} lossy + {len(val_lossless)} lossless")

    discriminator = SRNet().to(DEVICE)
    optimizer     = optim.Adam(discriminator.parameters(), lr=INITIAL_LR, weight_decay=1e-4)
    criterion     = nn.CrossEntropyLoss()
    scaler        = torch.amp.GradScaler('cuda')
    unified_gen   = UnifiedGenerator()
    evo_manager   = EvolutionaryManager()
    to_tensor     = transforms.ToTensor()

    training_history = {
        'epochs': [], 'loss': [], 'model_acc': [],
        'val_loss': [], 'val_acc': [],
        'gen_success': [], 'learning_rate': []
    }

    min_dataset_size = min(len(train_lossy), len(train_lossless))
    steps_per_epoch  = max(1, min_dataset_size // (BATCH_SIZE // 2))

    best_val_acc   = 0.0
    best_val_epoch = 0

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
            print(f"Curriculum: Cap [{min_capacity:.2f}-1.0] | Edge [0-{max_edge_threshold}]")
        else:
            print('EVOLUTION ACTIVATED - Full competitive training begins!'
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
            half_batch     = BATCH_SIZE // 2
            batch_lossy    = train_lossy[step * half_batch: (step + 1) * half_batch]
            batch_lossless = train_lossless[step * half_batch: (step + 1) * half_batch]
            batch_files    = batch_lossy + batch_lossless
            random.shuffle(batch_files)

            # Genomes assigned on main thread — never inside workers.
            assigned_pairs = [(path, evo_manager.get_random_genome()) for path in batch_files]

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

                    genome_with_capacity = genome.copy()
                    if curriculum_active:
                        genome_with_capacity['capacity_ratio'] = random.uniform(min_capacity, 1.0)
                        genome_with_capacity['edge_threshold']  = random.randint(0, max_edge_threshold)
                        if epoch < 5:
                            genome_with_capacity['strategy'] = 'random'

                    genome_with_capacity['message'] = (
                        generate_long_text_message(length=5000)
                        if epoch >= 5 and random.random() < 0.5
                        else None
                    )
                    if 'capacity_ratio' not in genome_with_capacity:
                        genome_with_capacity['capacity_ratio'] = 0.5

                    stego_arr, _ = unified_gen.generate_stego(cover_crop, None, genome_with_capacity)
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

            inputs_t = torch.stack(inputs).to(DEVICE, non_blocking=True)
            labels_t = torch.tensor(labels, dtype=torch.long).to(DEVICE, non_blocking=True)

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
                relevant_names = []
                fooled_results = []
                for j, name in enumerate(shuffled_genome_names):
                    if name is not None:
                        relevant_names.append(name)
                        fooled_results.append(preds[j].item() == 0)
                evo_manager.update_batch_stats(relevant_names, fooled_results)

            total_loss    += loss.item() * GRADIENT_ACCUMULATION_STEPS
            correct_total += (preds == labels_shuffled).sum().item()
            total_samples += labels_shuffled.size(0)

            if step % 10 == 0:
                acc_current = 100 * correct_total / total_samples
                print(f"\rStep {step}/{steps_per_epoch} | "
                      f"Loss: {loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f} | "
                      f"Acc: {acc_current:.1f}%", end="")

        # Flush leftover accumulated gradients at epoch end.
        if steps_per_epoch % GRADIENT_ACCUMULATION_STEPS != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        torch.cuda.empty_cache()

        # --- VALIDATION ---
        print(f"\n[VAL] Running validation...")
        val_loss, val_acc = run_validation(
            discriminator, val_lossy, val_lossless, unified_gen, criterion, epoch)
        print(f"[VAL] Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            best_val_epoch = epoch + 1
            save_checkpoint(epoch + 1, discriminator, optimizer,
                            evo_manager.population[0], val_acc,
                            'srnet_best_val.pth')
            print(f"[VAL] *** New best: {best_val_acc:.2f}% at epoch {best_val_epoch} ***")

        if total_samples > 0:
            avg_loss  = total_loss / steps_per_epoch
            acc_total = 100 * correct_total / total_samples

            if curriculum_active:
                print(f"[EPOCH SUMMARY] Loss: {avg_loss:.4f} | "
                      f"Train Acc: {acc_total:.2f}% | Val Acc: {val_acc:.2f}% | "
                      f"Curriculum Active")
                avg_gen_score = 0.0
            else:
                all_rates     = [d['fooled'] / d['attempts']
                                 for d in evo_manager.stats.values() if d['attempts'] > 0]
                avg_gen_score = (sum(all_rates) / len(all_rates)) if all_rates else 0.0
                print(f"[EPOCH SUMMARY] Loss: {avg_loss:.4f} | "
                      f"Train Acc: {acc_total:.2f}% | Val Acc: {val_acc:.2f}% | "
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
            evo_manager.stats  = {g['name']: {'fooled': 0, 'attempts': 0}
                                  for g in evo_manager.population}
            best_genome = evo_manager.population[0]

        if (epoch + 1) % 5 == 0:
            save_checkpoint(epoch + 1, discriminator, optimizer, best_genome, val_acc,
                            f"srnet_epoch_{epoch + 1}.pth")

    print(f"\n[INFO] Best validation accuracy: {best_val_acc:.2f}% at epoch {best_val_epoch}")
    print(f"[INFO] Best model saved as: srnet_best_val.pth")
    print(f"[INFO] Use srnet_best_val.pth (not srnet_epoch_30.pth) as input to evaluate.py")

    with open('training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    print("[INFO] Training Complete.")


if __name__ == "__main__":
    run_training()
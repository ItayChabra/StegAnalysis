import multiprocessing
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import torch.optim as optim
from generators.unified_generator import UnifiedGenerator
from models.srnet import SRNet
import os
import glob
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
import random
import copy
import json
import uuid
import string
import numpy as np

# --- SETTINGS ---
BATCH_SIZE = 64
EPOCHS = 30
POPULATION_SIZE = 20
NUM_WORKERS = min(8, multiprocessing.cpu_count())
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cudnn.benchmark = True
INITIAL_LR = 0.0001
MAX_LR = 0.001  # Restored from 0.0003
MIN_CAPACITY = 0.20  # Prevent generators from going too subtle
MAX_CAPACITY = 0.75


class EvolutionaryManager:
    def __init__(self):
        self.population = []
        for i in range(POPULATION_SIZE):
            self.population.append(self._generate_random_genome(f"Gen_{i}"))
        self.stats = {g['name']: {'fooled': 0, 'attempts': 0} for g in self.population}
        self.generation = 0

    def _generate_random_genome(self, name):
        genome = {'name': name, 'gen_type': 'lsb'}
        genome.update({
            'strategy': random.choice(['random', 'sequential', 'skip']),
            'step': random.randint(1, 15),
            'bit_depth': 1,
            'edge_threshold': random.randint(0, 100),
            'capacity_ratio': random.uniform(MIN_CAPACITY, MAX_CAPACITY),  # Constrained
        })
        return genome

    def mutate(self, genome):
        new_genome = copy.deepcopy(genome)
        new_genome['name'] = f"{genome['name']}_m{self.generation}"
        num_mutations = 2 if random.random() < 0.2 else 1
        for _ in range(num_mutations):
            mutation = random.choice(['step', 'threshold', 'strategy', 'capacity'])
            if mutation == 'step':
                change = random.choice([-2, -1, 1, 2, 3])
                new_genome['step'] = max(1, min(20, new_genome['step'] + change))
            elif mutation == 'threshold':
                change = random.randint(-20, 20)
                new_genome['edge_threshold'] = max(0, min(100, new_genome['edge_threshold'] + change))
            elif mutation == 'strategy':
                new_genome['strategy'] = random.choice(['random', 'sequential', 'skip'])
            elif mutation == 'capacity':
                change = random.uniform(-0.15, 0.15)
                new_genome['capacity_ratio'] = max(MIN_CAPACITY,
                                                   min(MAX_CAPACITY, new_genome['capacity_ratio'] + change))
        return new_genome

    def crossover(self, genome1, genome2):
        new_genome = copy.deepcopy(genome1)
        new_genome['name'] = f"Cross_{self.generation}"
        if random.random() < 0.5: new_genome['step'] = genome2['step']
        if random.random() < 0.5: new_genome['edge_threshold'] = genome2['edge_threshold']
        if random.random() < 0.5: new_genome['strategy'] = genome2['strategy']
        if random.random() < 0.5: new_genome['capacity_ratio'] = genome2['capacity_ratio']
        return new_genome

    def update_batch_stats(self, names, is_fooled_list):
        for name, fooled in zip(names, is_fooled_list):
            if name in self.stats:
                self.stats[name]['attempts'] += 1
                if fooled: self.stats[name]['fooled'] += 1

    def evolve(self):
        self.generation += 1
        final_scores = {}
        for name, data in self.stats.items():
            if data['attempts'] > 0:
                final_scores[name] = data['fooled'] / data['attempts']
            else:
                final_scores[name] = 0.0
        sorted_pop = sorted(self.population, key=lambda g: final_scores.get(g['name'], 0), reverse=True)

        print(f"\n[EVOLUTION] Generation {self.generation} - Top 3:")
        for i in range(min(3, len(sorted_pop))):
            g = sorted_pop[i]
            score = final_scores.get(g['name'], 0) * 100
            print(
                f"  #{i + 1}: {g['name']} - {score:.2f}% | Strat: {g['strategy']} | Cap: {g['capacity_ratio']:.2f} | Edge: {g['edge_threshold']}")

        new_pop = sorted_pop[:3]
        if len(sorted_pop) >= 2: new_pop.append(self.crossover(sorted_pop[0], sorted_pop[1]))
        if len(sorted_pop) >= 3: new_pop.append(self.crossover(sorted_pop[0], sorted_pop[2]))
        for i in range(2): new_pop.append(self._generate_random_genome(f"Gen_Rnd_{self.generation}_{i}"))
        while len(new_pop) < POPULATION_SIZE:
            parent_idx = min(random.randint(0, 2), len(sorted_pop) - 1)
            new_pop.append(self.mutate(sorted_pop[parent_idx]))
        self.population = new_pop
        self.stats = {g['name']: {'fooled': 0, 'attempts': 0} for g in self.population}
        return sorted_pop[0]

    def get_random_genome(self):
        if self.generation == 0 or random.random() < 0.3:
            return random.choice(self.population)
        weights = []
        for g in self.population:
            data = self.stats[g['name']]
            score = (data['fooled'] / data['attempts']) if data['attempts'] > 0 else 0.1
            weights.append(score + 0.05)
        total = sum(weights)
        weights = [w / total for w in weights]
        return random.choices(self.population, weights=weights, k=1)[0]


def generate_long_text_message(length=5000):
    chars = string.ascii_letters + string.digits + " " + ".,!?"
    return ''.join(random.choices(chars, k=length))


def load_balanced_dataset(raw_dir):
    lossy_dir = os.path.join(raw_dir, 'flickr30k')
    lossless_dir = os.path.join(raw_dir, 'BossBase and BOWS2')

    print(f"[DATA] Scanning specific folders in {raw_dir}...")
    lossy_files = glob.glob(os.path.join(lossy_dir, '*.jpg')) + \
                  glob.glob(os.path.join(lossy_dir, '*.jpeg'))
    lossless_files = glob.glob(os.path.join(lossless_dir, '*.pgm')) + \
                     glob.glob(os.path.join(lossless_dir, '*.png'))

    print(f"[DATA] Found {len(lossy_files)} Lossy (Flickr) images.")
    print(f"[DATA] Found {len(lossless_files)} Lossless (BOSSbase) images.")

    if len(lossy_files) < BATCH_SIZE or len(lossless_files) < BATCH_SIZE:
        print("[WARN] Imbalance or missing files!")
    return lossy_files, lossless_files


def save_checkpoint(epoch, model, optimizer, best_genome, filename="checkpoint.pth"):
    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'best_genome': best_genome}, filename)
    print(f"[CHECKPOINT] Saved to {filename}")


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


def run_training():
    print(f"🚀 Starting Hybrid 50/50 Training on {DEVICE}")
    lossy_files, lossless_files = load_balanced_dataset('data/raw')

    discriminator = SRNet().to(DEVICE)
    optimizer = optim.Adam(discriminator.parameters(), lr=INITIAL_LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')
    unified_gen = UnifiedGenerator()
    evo_manager = EvolutionaryManager()
    to_tensor = transforms.ToTensor()

    training_history = {'epochs': [], 'loss': [], 'model_acc': [], 'gen_success': [], 'learning_rate': []}
    min_dataset_size = min(len(lossy_files), len(lossless_files))
    steps_per_epoch = max(1, min_dataset_size // (BATCH_SIZE // 2))

    for epoch in range(EPOCHS):
        current_lr = adjust_learning_rate(optimizer, epoch)

        # Calculate curriculum parameters for this epoch
        if epoch < 10:  # Shortened from 12
            min_capacity = max(MIN_CAPACITY, 1.0 - (epoch * 0.08))  # Respects MIN_CAPACITY floor
            max_edge_threshold = min(70, epoch * 7)  # 0 → 70 over 10 epochs
            curriculum_active = True
        else:
            min_capacity = MIN_CAPACITY
            max_edge_threshold = 100
            curriculum_active = False

        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{EPOCHS} | LR: {current_lr:.6f}")
        if curriculum_active:
            print(f"📚 Curriculum: Cap [{min_capacity:.2f}-1.0] | Edge [0-{max_edge_threshold}]")
        else:
            if epoch == 10:  # Changed from 12
                print(f"🧬 EVOLUTION ACTIVATED - Full competitive training begins!")
            else:
                print(f"🧬 Evolution Active")
        print('=' * 60)

        random.shuffle(lossy_files)
        random.shuffle(lossless_files)
        total_loss = 0
        correct_total = 0
        total_samples = 0
        discriminator.train()

        for step in range(steps_per_epoch):
            half_batch = BATCH_SIZE // 2
            batch_lossy = lossy_files[step * half_batch: (step + 1) * half_batch]
            batch_lossless = lossless_files[step * half_batch: (step + 1) * half_batch]
            batch_files = batch_lossy + batch_lossless
            random.shuffle(batch_files)

            inputs = []
            labels = []
            batch_genome_names = []

            for path in batch_files:
                genome = evo_manager.get_random_genome()
                temp_id = str(uuid.uuid4())
                temp_cover_path = f"temp_{temp_id}.png"

                try:
                    cover_img = Image.open(path).convert('L')
                    w, h = cover_img.size
                    if w < 256 or h < 256: continue

                    i_crop, j_crop, h_crop, w_crop = transforms.RandomCrop.get_params(
                        cover_img, output_size=(256, 256))
                    cover_crop = TF.crop(cover_img, i_crop, j_crop, h_crop, w_crop)
                    cover_crop.save(temp_cover_path)

                    genome_with_capacity = genome.copy()

                    # --- GRADUAL CURRICULUM LEARNING ---
                    if curriculum_active:
                        # Override genome parameters with curriculum constraints
                        genome_with_capacity['capacity_ratio'] = random.uniform(min_capacity, 1.0)
                        genome_with_capacity['edge_threshold'] = random.randint(0, max_edge_threshold)
                        # Early epochs: prefer random strategy for better coverage
                        if epoch < 5:
                            genome_with_capacity['strategy'] = 'random'

                    # Message strategy
                    if epoch < 5:
                        # Early: Always random bits for max entropy
                        genome_with_capacity['message'] = None
                    else:
                        # Later: Mix text and random
                        if random.random() < 0.5:
                            genome_with_capacity['message'] = generate_long_text_message(length=5000)
                        else:
                            genome_with_capacity['message'] = None

                    if 'capacity_ratio' not in genome_with_capacity:
                        genome_with_capacity['capacity_ratio'] = 0.5

                    # Generate stego
                    stego_arr, _ = unified_gen.generate_stego(temp_cover_path, None, genome_with_capacity)
                    if stego_arr is None: continue
                    stego_img = Image.fromarray(stego_arr)

                    # Add to batch
                    inputs.append(to_tensor(cover_crop))
                    labels.append(0)
                    batch_genome_names.append(None)

                    inputs.append(to_tensor(stego_img))
                    labels.append(1)
                    batch_genome_names.append(genome['name'])

                except Exception:
                    continue
                finally:
                    if os.path.exists(temp_cover_path):
                        os.remove(temp_cover_path)

            if not inputs: continue

            inputs_t = torch.stack(inputs).to(DEVICE)
            labels_t = torch.tensor(labels).to(DEVICE)

            # --- DIAGNOSTIC CHECK (First batch of first epoch) ---
            if epoch == 0 and step == 0:
                print("\n" + "=" * 60)
                print("🔍 DIAGNOSTIC CHECK")
                print("=" * 60)
                covers = inputs_t[0::2].cpu().numpy()
                stegos = inputs_t[1::2].cpu().numpy()
                diff = np.abs(covers - stegos)
                print(f"  Max Pixel Diff: {diff.max():.6f}")
                print(f"  Mean Pixel Diff: {diff.mean():.6f}")
                print(f"  Pixels Modified: {(diff > 0).sum():,} / {diff.size:,}")
                mod_rate = 100 * (diff > 0).sum() / diff.size
                print(f"  Modification Rate: {mod_rate:.2f}%")
                print(f"  Batch: {labels.count(0)} covers, {labels.count(1)} stegos")
                print("=" * 60 + "\n")

            # Shuffle batch
            perm = torch.randperm(inputs_t.size(0))
            inputs_shuffled = inputs_t[perm]
            labels_shuffled = labels_t[perm]
            shuffled_genome_names = [batch_genome_names[idx] for idx in perm.tolist()]

            # Training step
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = discriminator(inputs_shuffled)
                loss = criterion(outputs, labels_shuffled)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            _, preds = torch.max(outputs, 1)

            # Update evolution stats (only when curriculum is inactive)
            if not curriculum_active:
                relevant_names = []
                fooled_results = []
                for j, name in enumerate(shuffled_genome_names):
                    if name is not None:
                        is_fooled = (preds[j].item() == 0)
                        relevant_names.append(name)
                        fooled_results.append(is_fooled)
                evo_manager.update_batch_stats(relevant_names, fooled_results)

            total_loss += loss.item()
            correct_total += (preds == labels_shuffled).sum().item()
            total_samples += labels_shuffled.size(0)

            if step % 10 == 0:
                acc_current = 100 * correct_total / total_samples
                print(f"\rStep {step}/{steps_per_epoch} | Loss: {loss.item():.4f} | Acc: {acc_current:.1f}%", end="")

        # End of Epoch Stats
        if total_samples > 0:
            avg_loss = total_loss / steps_per_epoch
            acc_total = 100 * correct_total / total_samples

            if curriculum_active:
                # During curriculum, evolution is paused
                print(
                    f"\n[EPOCH SUMMARY] Loss: {avg_loss:.4f} | Acc: {acc_total:.2f}% | 📚 Curriculum Active - Evolution Paused")
                avg_gen_score = 0.0  # Not tracking during curriculum
            else:
                # Normal evolution tracking
                all_rates = [d['fooled'] / d['attempts'] for d in evo_manager.stats.values() if d['attempts'] > 0]
                avg_gen_score = (sum(all_rates) / len(all_rates)) if all_rates else 0.0
                print(
                    f"\n[EPOCH SUMMARY] Loss: {avg_loss:.4f} | Acc: {acc_total:.2f}% | Gen Fool: {avg_gen_score * 100:.2f}%")

            training_history['epochs'].append(epoch + 1)
            training_history['loss'].append(avg_loss)
            training_history['model_acc'].append(acc_total)
            training_history['gen_success'].append(avg_gen_score * 100)
            training_history['learning_rate'].append(current_lr)

        # Only evolve when curriculum is inactive
        if not curriculum_active:
            best_genome = evo_manager.evolve()
        else:
            # During curriculum, just reset stats without evolution
            evo_manager.generation += 1
            evo_manager.stats = {g['name']: {'fooled': 0, 'attempts': 0} for g in evo_manager.population}
            best_genome = evo_manager.population[0]  # Dummy for checkpoint

        if (epoch + 1) % 5 == 0:
            save_checkpoint(epoch + 1, discriminator, optimizer, best_genome, f"srnet_epoch_{epoch + 1}.pth")

    with open('training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    print("\n[INFO] Training Complete.")


if __name__ == "__main__":
    run_training()
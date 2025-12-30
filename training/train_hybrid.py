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
import time
import json

# --- IMPROVED SETTINGS ---
BATCH_SIZE = 64
EPOCHS = 30
WARMUP_EPOCHS = 0
POPULATION_SIZE = 20  # Increase population diversity
NUM_WORKERS = min(8, multiprocessing.cpu_count())
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Enable A100 Tensor Core optimizations
cudnn.benchmark = True

# Learning rate schedule - start MUCH slower
INITIAL_LR = 0.00001  # 10x slower than before
MAX_LR = 0.0001


class EvolutionaryManager:
    def __init__(self):
        self.available_types = ['lsb']
        self.population = []
        for i in range(POPULATION_SIZE):
            self.population.append(self._generate_random_genome(f"Gen_{i}"))
        self.stats = {g['name']: {'fooled': 0, 'attempts': 0} for g in self.population}
        self.generation = 0

    def _generate_random_genome(self, name):
        """Generate more diverse genomes with wider parameter ranges"""
        genome = {'name': name, 'gen_type': 'lsb'}
        genome.update({
            'strategy': random.choice(['random', 'sequential', 'skip']),
            'step': random.randint(1, 15),  # Wider range
            'bit_depth': 1,
            'edge_threshold': random.randint(0, 100),  # Wider range
            # NEW: Add capacity control
            'capacity_ratio': random.uniform(0.1, 0.7),  # Use 10-70% of pixels
        })
        return genome

    def mutate(self, genome):
        """More aggressive mutations"""
        new_genome = copy.deepcopy(genome)
        new_genome['name'] = f"{genome['name']}_m{self.generation}"

        # Multiple simultaneous mutations (20% chance)
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
                new_genome['capacity_ratio'] = max(0.05, min(0.8, new_genome['capacity_ratio'] + change))

        return new_genome

    def crossover(self, genome1, genome2):
        """Combine traits from two genomes"""
        new_genome = copy.deepcopy(genome1)
        new_genome['name'] = f"Cross_{self.generation}"

        # Randomly inherit each trait
        if random.random() < 0.5:
            new_genome['step'] = genome2['step']
        if random.random() < 0.5:
            new_genome['edge_threshold'] = genome2['edge_threshold']
        if random.random() < 0.5:
            new_genome['strategy'] = genome2['strategy']
        if random.random() < 0.5:
            new_genome['capacity_ratio'] = genome2['capacity_ratio']

        return new_genome

    def update_batch_stats(self, names, is_fooled_list):
        for name, fooled in zip(names, is_fooled_list):
            if name in self.stats:
                self.stats[name]['attempts'] += 1
                if fooled:
                    self.stats[name]['fooled'] += 1

    def evolve(self):
        """Improved evolution with elitism, crossover, and diversity preservation"""
        self.generation += 1

        final_scores = {}
        for name, data in self.stats.items():
            if data['attempts'] > 0:
                final_scores[name] = data['fooled'] / data['attempts']
            else:
                final_scores[name] = 0.0

        sorted_pop = sorted(self.population, key=lambda g: final_scores.get(g['name'], 0), reverse=True)

        # Show top 3 performers
        print(f"\n[EVOLUTION] Generation {self.generation} - Top 3 Genomes:")
        for i in range(min(3, len(sorted_pop))):
            g = sorted_pop[i]
            score = final_scores.get(g['name'], 0) * 100
            print(f"  #{i + 1}: {g['name']} - {score:.2f}% fooling rate")
            print(
                f"       Strategy={g['strategy']}, Step={g['step']}, Thresh={g['edge_threshold']}, Capacity={g.get('capacity_ratio', 0.5):.2f}")

        # NEW EVOLUTION STRATEGY
        new_pop = []

        # 1. Keep top 3 (elitism)
        new_pop.extend(sorted_pop[:3])

        # 2. Add 2 crossovers from top performers
        if len(sorted_pop) >= 2:
            new_pop.append(self.crossover(sorted_pop[0], sorted_pop[1]))
            if len(sorted_pop) >= 3:
                new_pop.append(self.crossover(sorted_pop[0], sorted_pop[2]))

        # 3. Add 2 completely random (diversity)
        for i in range(2):
            new_pop.append(self._generate_random_genome(f"Gen_Rnd_{self.generation}_{i}"))

        # 4. Fill rest with mutations of best genomes
        while len(new_pop) < POPULATION_SIZE:
            parent_idx = min(random.randint(0, 2), len(sorted_pop) - 1)
            new_pop.append(self.mutate(sorted_pop[parent_idx]))

        self.population = new_pop
        self.stats = {g['name']: {'fooled': 0, 'attempts': 0} for g in self.population}
        return sorted_pop[0]

    def get_random_genome(self):
        """Weighted selection - slightly favor better performers"""
        if self.generation == 0 or random.random() < 0.3:
            return random.choice(self.population)

        # Calculate weights based on recent performance
        weights = []
        for g in self.population:
            data = self.stats[g['name']]
            if data['attempts'] > 0:
                score = data['fooled'] / data['attempts']
            else:
                score = 0.1  # Unknown genomes get some chance
            weights.append(score + 0.05)  # Add baseline to ensure all have some chance

        total = sum(weights)
        weights = [w / total for w in weights]

        return random.choices(self.population, weights=weights, k=1)[0]


def load_images(raw_dir):
    files = []
    for ext in ('*.png', '*.jpg', '*.jpeg', '*.pgm'):
        files.extend(glob.glob(os.path.join(raw_dir, ext)))
    return files


def save_checkpoint(epoch, model, optimizer, best_genome, filename="checkpoint.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_genome': best_genome
    }, filename)
    print(f"[CHECKPOINT] Saved to {filename}")


def adjust_learning_rate(optimizer, epoch):
    """Gradual learning rate warmup"""
    if epoch < 5:
        # Linear warmup over first 5 epochs
        lr = INITIAL_LR + (MAX_LR - INITIAL_LR) * (epoch / 5)
    elif epoch < 15:
        # Stay at max
        lr = MAX_LR
    else:
        # Slight decay
        lr = MAX_LR * 0.95 ** (epoch - 15)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def run_training():
    print(f"🚀 Starting Improved Evolutionary Training on {DEVICE}")
    print(f"   Key Changes:")
    print(f"   - No warmup period (immediate evolution)")
    print(f"   - Slower initial learning rate (10x reduction)")
    print(f"   - Larger, more diverse population")
    print(f"   - Enhanced mutation & crossover")

    raw_images = load_images('data/raw')
    if len(raw_images) == 0:
        print("[ERROR] No images found in data/raw")
        return
    print(f"[INFO] Loaded {len(raw_images)} images.")

    discriminator = SRNet().to(DEVICE)
    optimizer = optim.Adam(discriminator.parameters(), lr=INITIAL_LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler('cuda')

    unified_gen = UnifiedGenerator()
    evo_manager = EvolutionaryManager()
    to_tensor = transforms.ToTensor()

    training_history = {
        'epochs': [],
        'loss': [],
        'model_acc': [],
        'gen_success': [],
        'learning_rate': []
    }

    for epoch in range(EPOCHS):
        current_lr = adjust_learning_rate(optimizer, epoch)
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{EPOCHS} | LR: {current_lr:.6f}")
        print(f"{'=' * 60}")

        random.shuffle(raw_images)

        total_loss = 0
        correct_total = 0
        total_samples = 0

        discriminator.train()

        total_batches = len(raw_images) // BATCH_SIZE

        for i in range(0, len(raw_images), BATCH_SIZE):
            batch_files = raw_images[i: i + BATCH_SIZE]
            if len(batch_files) < 4: break

            inputs = []
            labels = []
            batch_genome_names = []

            for path in batch_files:
                genome = evo_manager.get_random_genome()

                try:
                    # Pass capacity_ratio to generator
                    genome_with_capacity = genome.copy()
                    if 'capacity_ratio' not in genome_with_capacity:
                        genome_with_capacity['capacity_ratio'] = 0.5

                    stego_arr, _ = unified_gen.generate_stego(path, None, genome_with_capacity)
                    if stego_arr is None: continue

                    cover_img = Image.open(path).convert('L')
                    stego_img = Image.fromarray(stego_arr)

                    w, h = cover_img.size
                    if w < 256 or h < 256: continue

                    i_crop, j_crop, h_crop, w_crop = transforms.RandomCrop.get_params(
                        cover_img, output_size=(256, 256))

                    cover_crop = TF.crop(cover_img, i_crop, j_crop, h_crop, w_crop)
                    stego_crop = TF.crop(stego_img, i_crop, j_crop, h_crop, w_crop)

                    inputs.append(to_tensor(cover_crop))
                    labels.append(0)
                    batch_genome_names.append(None)

                    inputs.append(to_tensor(stego_crop))
                    labels.append(1)
                    batch_genome_names.append(genome['name'])

                except Exception as e:
                    continue

            if not inputs: continue

            inputs_t = torch.stack(inputs).to(DEVICE)
            labels_t = torch.tensor(labels).to(DEVICE)

            perm = torch.randperm(inputs_t.size(0))
            inputs_shuffled = inputs_t[perm]
            labels_shuffled = labels_t[perm]
            shuffled_genome_names = [batch_genome_names[idx] for idx in perm.tolist()]

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                outputs = discriminator(inputs_shuffled)
                loss = criterion(outputs, labels_shuffled)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            _, preds = torch.max(outputs, 1)

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

            batch_idx = i // BATCH_SIZE
            if batch_idx % 10 == 0:
                acc_current = 100 * correct_total / total_samples
                print(f"\rBatch {batch_idx}/{total_batches} | Loss: {loss.item():.4f} | Acc: {acc_current:.1f}%",
                      end="")

        print()

        if total_samples > 0:
            avg_loss = total_loss / (total_samples / BATCH_SIZE)
            acc_total = 100 * correct_total / total_samples
            all_rates = [d['fooled'] / d['attempts'] for d in evo_manager.stats.values() if d['attempts'] > 0]
            avg_gen_score = (sum(all_rates) / len(all_rates)) if all_rates else 0.0

            print(f"\n[EPOCH SUMMARY]")
            print(f"  Loss: {avg_loss:.4f}")
            print(f"  Model Accuracy: {acc_total:.2f}%")
            print(f"  Generator Fooling Rate: {avg_gen_score * 100:.2f}%")

            training_history['epochs'].append(epoch + 1)
            training_history['loss'].append(avg_loss)
            training_history['model_acc'].append(acc_total)
            training_history['gen_success'].append(avg_gen_score * 100)
            training_history['learning_rate'].append(current_lr)

        # EVOLVE EVERY EPOCH (no warmup)
        best_genome = evo_manager.evolve()

        if (epoch + 1) % 5 == 0:
            save_checkpoint(epoch + 1, discriminator, optimizer, best_genome, f"srnet_epoch_{epoch + 1}.pth")

    with open('training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    print("\n[INFO] Training Complete.")


if __name__ == "__main__":
    run_training()
import torch
import torch.nn as nn
import torch.optim as optim
from generators.unified_generator import UnifiedGenerator
from models.srnet import SRNet
import os
import glob
from PIL import Image
from torchvision import transforms
import random
import copy
import time
import json

# --- Settings ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# GPU Settings (Use these for your GTX 970)
BATCH_SIZE = 16
EPOCHS = 30
POPULATION_SIZE = 5


class EvolutionaryManager:
    """
    Manages the population and mutations for ALL generator types.
    """

    def __init__(self):
        self.available_types = ['lsb']
        self.population = []
        for i in range(POPULATION_SIZE):
            self.population.append(self._generate_random_genome(f"Gen_{i}"))
        self.scores = {g['name']: 0.0 for g in self.population}

    def _generate_random_genome(self, name):
        """Randomly selects a generator type and creates valid parameters."""
        gen_type = random.choice(self.available_types)
        genome = {'name': name, 'gen_type': gen_type}

        if gen_type == 'lsb':
            genome.update({
                'strategy': random.choice(['random', 'sequential', 'skip']),
                'step': random.randint(1, 8),
                'bit_depth': 1,
                'edge_threshold': random.randint(0, 50)
            })
        return genome

    def mutate(self, genome):
        new_genome = copy.deepcopy(genome)
        new_genome['name'] = f"{genome['name']}_v"

        if new_genome['gen_type'] == 'lsb':
            mutation = random.choice(['step', 'threshold', 'strategy', 'depth'])
            if mutation == 'step':
                change = random.choice([-1, 1, 2])
                new_genome['step'] = max(1, min(15, new_genome['step'] + change))
            elif mutation == 'threshold':
                change = random.randint(-10, 10)
                new_genome['edge_threshold'] = max(0, min(100, new_genome['edge_threshold'] + change))
            elif mutation == 'strategy':
                new_genome['strategy'] = random.choice(['random', 'sequential', 'skip'])
            elif mutation == 'depth':
                new_genome['bit_depth'] = 2 if new_genome['bit_depth'] == 1 else 1

        return new_genome

    def evolve(self):
        # Sort by score (descending)
        sorted_pop = sorted(self.population, key=lambda g: self.scores.get(g['name'], 0), reverse=True)
        best_genome = sorted_pop[0]

        print(
            f"\n[EVOLUTION] Best Genome: {best_genome['name']} (Score: {self.scores.get(best_genome['name'], 0):.2f})")
        if best_genome['gen_type'] == 'lsb':
            print(
                f"   Traits: Strategy={best_genome['strategy']}, Step={best_genome['step']}, Threshold={best_genome['edge_threshold']}")

        # LOGGING: Print full population scores
        print("   Population Scores:")
        for g in sorted_pop:
            score = self.scores.get(g['name'], 0)
            print(f"      {g['name']}: {score:.2f}")

        # Keep the best one
        new_pop = [best_genome]

        # Add a fresh random one (Diversity Injection)
        new_pop.append(self._generate_random_genome(f"Gen_Random_{int(time.time())}"))

        # Fill the rest with mutations of the best
        while len(new_pop) < POPULATION_SIZE:
            new_pop.append(self.mutate(best_genome))

        self.population = new_pop
        self.scores = {g['name']: 0.0 for g in self.population}

        return best_genome  # Return best genome for checkpoint saving

    def get_random_genome(self):
        return random.choice(self.population)

    def update_score(self, name, points):
        if name in self.scores:
            self.scores[name] += points


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


def evolutionary_training():
    print(f"[INFO] Starting Evolutionary Training on {DEVICE}...")

    discriminator = SRNet().to(DEVICE)
    optimizer = optim.Adam(discriminator.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    unified_gen = UnifiedGenerator()
    evo_manager = EvolutionaryManager()

    raw_images = load_images('data/raw')
    if not raw_images:
        print("[ERROR] No images found in data/raw!")
        return

    # If running on CPU, limit data. If GPU, use all.
    if DEVICE.type == 'cpu':
        print(f"[WARN] CPU MODE: Limiting to 100 images.")
        raw_images = raw_images[:100]
    else:
        print(f"[INFO] GPU MODE: Using full dataset ({len(raw_images)} images).")

    transform = transforms.Compose([transforms.ToTensor()])

    # --- Data Collection for Graphs ---
    training_history = {
        'epochs': [],
        'loss': [],
        'total_acc': [],
        'cover_acc': [],
        'stego_acc': [],
        'gen_success': []
    }

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
        random.shuffle(raw_images)

        total_loss = 0
        correct_total = 0
        correct_cover = 0
        correct_stego = 0
        total_samples = 0

        for i in range(0, len(raw_images), BATCH_SIZE):
            batch_files = raw_images[i: i + BATCH_SIZE]
            if not batch_files: break

            genome = evo_manager.get_random_genome()

            inputs = []
            labels = []

            for path in batch_files:
                try:
                    # 1. Try to generate Stego FIRST
                    stego_arr, _ = unified_gen.generate_stego(path, None, genome)
                    if stego_arr is None:
                        continue

                        # 2. Load Cover only if Stego succeeded
                    img = Image.open(path).convert('L').resize((256, 256))

                    # Add Pair to Batch
                    inputs.append(transform(img))  # Cover
                    labels.append(0)

                    inputs.append(transform(Image.fromarray(stego_arr)))  # Stego
                    labels.append(1)

                except Exception:
                    continue

            if not inputs: continue

            inputs_t = torch.stack(inputs).to(DEVICE)
            labels_t = torch.tensor(labels).to(DEVICE)

            # --- Train Step ---
            optimizer.zero_grad()
            outputs = discriminator(inputs_t)
            loss = criterion(outputs, labels_t)
            loss.backward()
            optimizer.step()

            # --- Metrics & Scoring ---
            _, preds = torch.max(outputs, 1)

            cover_preds = preds[0::2]
            stego_preds = preds[1::2]

            # Update Generator Score
            gen_successes = (stego_preds == 0).sum().item()
            if len(stego_preds) > 0:
                evo_manager.update_score(genome['name'], gen_successes / len(stego_preds))

            # Update Discriminator Metrics
            total_loss += loss.item()
            correct_total += (preds == labels_t).sum().item()
            correct_cover += (cover_preds == 0).sum().item()
            correct_stego += (stego_preds == 1).sum().item()
            total_samples += labels_t.size(0)

        # --- Epoch Summary & Stats Storage ---
        avg_gen_score = 0
        if total_samples > 0:
            avg_loss = total_loss / (total_samples / BATCH_SIZE)
            acc_total = 100 * correct_total / total_samples
            acc_cover = 100 * correct_cover / (total_samples / 2)
            acc_stego = 100 * correct_stego / (total_samples / 2)

            print(f"[STATS] Loss={avg_loss:.4f} | Total Acc={acc_total:.2f}%")
            print(f"   Cover Acc: {acc_cover:.2f}% | Stego Acc: {acc_stego:.2f}%")

            # Calculate Avg Generator Score
            current_scores = list(evo_manager.scores.values())
            if current_scores:
                avg_gen_score = sum(current_scores) / len(current_scores)
                print(f"   Avg Generator Success Rate: {avg_gen_score:.2%}")

            # Store Data
            training_history['epochs'].append(epoch + 1)
            training_history['loss'].append(avg_loss)
            training_history['total_acc'].append(acc_total)
            training_history['cover_acc'].append(acc_cover)
            training_history['stego_acc'].append(acc_stego)
            training_history['gen_success'].append(avg_gen_score * 100)

        # Evolve and get best genome
        best_genome_of_epoch = evo_manager.evolve()

        # --- Checkpoint & Examples (Every 5 Epochs) ---
        if (epoch + 1) % 5 == 0:
            # 1. Save Checkpoint
            save_checkpoint(epoch + 1, discriminator, optimizer, best_genome_of_epoch, f"srnet_epoch_{epoch + 1}.pth")

            # 2. Save Visual Example
            os.makedirs('examples', exist_ok=True)
            # Pick a random image from this batch's raw files for testing
            test_img_path = raw_images[0]
            example_stego, psnr = unified_gen.generate_stego(test_img_path, None, best_genome_of_epoch)

            if example_stego is not None:
                orig = Image.open(test_img_path).convert('L').resize((256, 256))
                orig.save(f"examples/epoch_{epoch + 1}_original.png")
                Image.fromarray(example_stego).save(f"examples/epoch_{epoch + 1}_stego.png")
                print(f"[INFO] Example image saved to examples/ | PSNR: {psnr:.2f} dB")

    # --- End of Training: Save History ---
    with open('training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    print("\n[INFO] Training history saved to training_history.json")


if __name__ == "__main__":
    evolutionary_training()
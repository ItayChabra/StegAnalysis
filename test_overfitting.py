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
import random
import copy
import time
import json

# --- SANITY CHECK SETTINGS ---
# We use a tiny batch and a tiny dataset to force the model to memorize (overfit).
BATCH_SIZE = 8  # Small batch for stability
OVERFIT_SIZE = 8  # Only use 8 unique images!
EPOCHS = 20
POPULATION_SIZE = 3  # Small population to reduce noise

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cudnn.benchmark = True


class SanityEvolutionaryManager:
    """
    Simplified Manager for Sanity Testing.
    Forces 'easier' genomes initially to guarantee the model has something to find.
    """

    def __init__(self):
        self.population = []
        for i in range(POPULATION_SIZE):
            self.population.append(self._generate_easy_genome(f"Gen_{i}"))
        self.scores = {g['name']: 0.0 for g in self.population}

    def _generate_easy_genome(self, name):
        """Creates an 'easy' genome (high modification) so the model CAN learn."""
        return {
            'name': name,
            'gen_type': 'lsb',
            'strategy': random.choice(['sequential', 'random']),  # Skip 'skip' strategy as it's harder
            'step': random.randint(3, 5),  # Higher step = more visible noise
            'bit_depth': 1,
            'edge_threshold': 0  # 0 threshold = modify everywhere (easier to detect)
        }

    def mutate(self, genome):
        new_genome = copy.deepcopy(genome)
        new_genome['name'] = f"{genome['name']}_v"
        # minimal mutation for sanity check
        new_genome['step'] = max(1, min(10, new_genome['step'] + random.choice([-1, 1])))
        return new_genome

    def evolve(self):
        sorted_pop = sorted(self.population, key=lambda g: self.scores.get(g['name'], 0), reverse=True)
        best_genome = sorted_pop[0]

        print(
            f"\n[EVOLUTION] Best Genome: {best_genome['name']} (Score: {self.scores.get(best_genome['name'], 0):.2f})")

        new_pop = [best_genome]
        while len(new_pop) < POPULATION_SIZE:
            new_pop.append(self.mutate(best_genome))

        self.population = new_pop
        self.scores = {g['name']: 0.0 for g in self.population}
        return best_genome

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


def run_sanity_check():
    print(f"🚀 Starting SANITY CHECK (Overfitting Test) on {DEVICE}...")

    # 1. Setup Data
    raw_images = load_images('data/raw')
    if len(raw_images) == 0:
        print("[ERROR] No images found in data/raw")
        return

    # CRITICAL: Pick only OVERFIT_SIZE images and repeat them to simulate a dataset
    # This forces the model to memorize these specific images.
    target_images = raw_images[:OVERFIT_SIZE]
    print(f"[SANITY] Selected {len(target_images)} unique images.")

    # Inflate dataset so one epoch performs multiple updates
    training_data = target_images * 50
    print(f"[SANITY] Inflated dataset size: {len(training_data)} samples.")

    # 2. Setup Components
    discriminator = SRNet().to(DEVICE)
    optimizer = optim.Adam(discriminator.parameters(), lr=0.0001)  # Standard LR
    criterion = nn.CrossEntropyLoss()

    unified_gen = UnifiedGenerator()
    evo_manager = SanityEvolutionaryManager()
    transform = transforms.Compose([transforms.ToTensor()])

    print(f"[SANITY] Goal: Accuracy should reach >95% quickly.\n")

    for epoch in range(EPOCHS):
        print(f"--- Epoch {epoch + 1}/{EPOCHS} ---")
        random.shuffle(training_data)

        total_loss = 0
        correct = 0
        total_samples = 0

        discriminator.train()

        # Manual Batch Loop
        for i in range(0, len(training_data), BATCH_SIZE):
            batch_files = training_data[i: i + BATCH_SIZE]
            if len(batch_files) < BATCH_SIZE: continue

            genome = evo_manager.get_random_genome()

            inputs = []
            labels = []

            for path in batch_files:
                try:
                    # Load Cover
                    img = Image.open(path).convert('L').resize((256, 256))

                    # Generate Stego
                    stego_arr, _ = unified_gen.generate_stego(path, None, genome)

                    if stego_arr is None:
                        # If generation fails, SKIP this pair.
                        # We don't want to train on Cover vs Cover (that's impossible to learn)
                        continue

                    # Prepare Tensors
                    cover_tensor = transform(img)
                    stego_tensor = transform(Image.fromarray(stego_arr))

                    inputs.append(cover_tensor)
                    labels.append(0)  # Cover

                    inputs.append(stego_tensor)
                    labels.append(1)  # Stego

                except Exception as e:
                    print(f"Err: {e}")
                    continue

            if not inputs:
                continue

            inputs_t = torch.stack(inputs).to(DEVICE)
            labels_t = torch.tensor(labels).to(DEVICE)

            # Shuffle batch to prevent [0,1,0,1] pattern memorization
            perm = torch.randperm(inputs_t.size(0))
            inputs_t = inputs_t[perm]
            labels_t = labels_t[perm]

            # Optimization
            optimizer.zero_grad()
            outputs = discriminator(inputs_t)
            loss = criterion(outputs, labels_t)
            loss.backward()
            optimizer.step()

            # Scoring
            _, preds = torch.max(outputs, 1)
            total_loss += loss.item()
            correct += (preds == labels_t).sum().item()
            total_samples += labels_t.size(0)

            # Update Evolution (simple average of current batch)
            # (In sanity check, we just want to ensure non-zero feedback)
            acc = (preds == labels_t).float().mean().item()
            evo_manager.update_score(genome['name'], 1.0 - acc)  # Score = how well it fooled model

        if total_samples > 0:
            avg_loss = total_loss / (total_samples / BATCH_SIZE)
            accuracy = 100 * correct / total_samples

            print(f"Batch Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

            if accuracy > 98.0:
                print("\n✅ SANITY CHECK PASSED! Model has successfully overfitted.")
                print(f"Stopped early at Epoch {epoch + 1}")
                return

        evo_manager.evolve()


if __name__ == "__main__":
    run_sanity_check()
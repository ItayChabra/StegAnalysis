import os
import torch
import torch.nn as nn
import torch.optim as optim
from generators.unified_generator import UnifiedGenerator
from models.srnet import SRNet
import glob
from PIL import Image
from torchvision import transforms
import random
import copy

# --- Test Settings ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 4
EPOCHS = 5
POPULATION_SIZE = 3


# --- Modified Manager (FORCED TO BE DUMB) ---
class EvolutionaryManager:
    """
    SPECIAL TEST VERSION: Forces 'sequential' AND 'threshold=0'.
    This is the easiest possible scenario for the discriminator.
    """

    def __init__(self):
        self.available_types = ['lsb']
        self.population = []
        for i in range(POPULATION_SIZE):
            self.population.append(self._generate_random_genome(f"Gen_{i}"))
        self.scores = {g['name']: 0.0 for g in self.population}

    def _generate_random_genome(self, name):
        """Forces Dumb Strategy."""
        genome = {'name': name, 'gen_type': 'lsb'}

        # FORCE EVERYTHING TO BE VISIBLE
        genome.update({
            'strategy': 'sequential',  # כותב ברצף (קל לזיהוי)
            'step': 1,
            'bit_depth': 1,
            'edge_threshold': 0  # <--- השינוי הגדול: כותב גם על 'חלק', לא מחפש קצוות!
        })
        return genome

    def mutate(self, genome):
        """Mutates only 'step' (Keeps strategy and threshold fixed)."""
        new_genome = copy.deepcopy(genome)
        new_genome['name'] = f"{genome['name']}_v"

        # ביטלנו את האפשרות לשנות Threshold או Strategy
        # הגנרטור יכול רק לשנות את ה-'Step' (שזה לא יעזור לו הרבה)
        mutation = 'step'

        if mutation == 'step':
            change = random.choice([-1, 1, 2])
            new_genome['step'] = max(1, min(15, new_genome['step'] + change))

        return new_genome

    def evolve(self):
        sorted_pop = sorted(self.population, key=lambda g: self.scores.get(g['name'], 0), reverse=True)
        best_genome = sorted_pop[0]

        print(
            f"\nEVOLUTION: Best Genome was {best_genome['name']} (Score: {self.scores.get(best_genome['name'], 0):.2f})")
        print(f"   Traits: Strategy={best_genome['strategy']}, Threshold={best_genome['edge_threshold']} (Fixed)")

        new_pop = sorted_pop[:2]
        for _ in range(POPULATION_SIZE - 2):
            new_pop.append(self.mutate(best_genome))

        self.population = new_pop
        self.scores = {g['name']: 0.0 for g in self.population}

    def get_random_genome(self):
        return random.choice(self.population)

    def update_score(self, name, points):
        if name in self.scores:
            self.scores[name] += points


# --- Helper Functions ---
def load_images(raw_dir):
    files = []
    for ext in ('*.png', '*.jpg', '*.jpeg', '*.pgm'):
        files.extend(glob.glob(os.path.join(raw_dir, ext)))
    return files


def check_data_setup():
    data_path = os.path.join("data", "raw")
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        print(f"[WARN] Created missing folder: {data_path}")
        return False

    valid_exts = ('.jpg', '.jpeg', '.png', '.pgm')
    files = [f for f in os.listdir(data_path) if f.lower().endswith(valid_exts)]

    if len(files) < 20:
        print(f"[ERROR] Found only {len(files)} images in '{data_path}'.")
        return False

    print(f"[OK] Found {len(files)} images ready for test.")
    return True


# --- Main Training Loop ---
def run_overfitting_test():
    print(f"Starting 'Dumb Generator' Test on {DEVICE}...")
    print("GOAL: Accuracy should SKYROCKET to >90% because Threshold is 0.")

    discriminator = SRNet().to(DEVICE)
    optimizer = optim.Adam(discriminator.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    unified_gen = UnifiedGenerator()
    evo_manager = EvolutionaryManager()

    raw_images = load_images('data/raw')
    if not raw_images:
        print("Error: No images found!")
        return

    # Limit images for speed
    if DEVICE.type == 'cpu':
        print(f"CPU MODE: Using only 100 images out of {len(raw_images)}")
        raw_images = raw_images[:100]

    transform = transforms.Compose([transforms.ToTensor()])

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
        random.shuffle(raw_images)

        total_loss = 0
        correct = 0
        total_samples = 0

        for i in range(0, len(raw_images), BATCH_SIZE):
            batch_files = raw_images[i: i + BATCH_SIZE]
            if not batch_files: break

            genome = evo_manager.get_random_genome()

            inputs = []
            labels = []

            for path in batch_files:
                try:
                    img = Image.open(path).convert('L').resize((256, 256))
                    inputs.append(transform(img))
                    labels.append(0)

                    stego_arr, _ = unified_gen.generate_stego(path, None, genome)
                    if stego_arr is not None:
                        inputs.append(transform(Image.fromarray(stego_arr)))
                        labels.append(1)
                except Exception:
                    continue

            if not inputs: continue

            inputs_t = torch.stack(inputs).to(DEVICE)
            labels_t = torch.tensor(labels).to(DEVICE)

            optimizer.zero_grad()
            outputs = discriminator(inputs_t)
            loss = criterion(outputs, labels_t)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)

            # Score logic
            stego_preds = preds[1::2]
            if len(stego_preds) > 0:
                successes = (stego_preds == 0).sum().item()
                success_rate = successes / len(stego_preds)
                evo_manager.update_score(genome['name'], success_rate)

            total_loss += loss.item()
            correct += (preds == labels_t).sum().item()
            total_samples += labels_t.size(0)

        avg_loss = total_loss / (total_samples / BATCH_SIZE) if total_samples > 0 else 0
        accuracy = 100 * correct / total_samples if total_samples > 0 else 0

        print(f"Avg Loss: {avg_loss:.4f}, Model Acc: {accuracy:.2f}%")
        evo_manager.evolve()


if __name__ == "__main__":
    if check_data_setup():
        run_overfitting_test()
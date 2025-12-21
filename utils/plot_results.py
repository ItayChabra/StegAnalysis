import json
import matplotlib.pyplot as plt
import numpy as np

# Load training data
with open('training_history.json', 'r') as f:
    data = json.load(f)

# Create figure with 2 rows, 2 columns
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Evolutionary Adversarial Training Results', fontsize=16, fontweight='bold')

# --- Plot 1: Accuracies Over Time ---
ax1 = axes[0, 0]
ax1.plot(data['epochs'], data['total_acc'], 'b-', linewidth=2, label='Total Accuracy')
ax1.plot(data['epochs'], data['cover_acc'], 'g--', linewidth=1.5, label='Cover Accuracy')
ax1.plot(data['epochs'], data['stego_acc'], 'r--', linewidth=1.5, label='Stego Accuracy')
ax1.axhline(y=50, color='gray', linestyle=':', alpha=0.5, label='Random Baseline')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_title('Discriminator Performance', fontsize=13, fontweight='bold')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

# --- Plot 2: Generator Success Rate ---
ax2 = axes[0, 1]
ax2.plot(data['epochs'], data['gen_success'], 'orange', linewidth=2, marker='o', markersize=4)
ax2.axhline(y=50, color='gray', linestyle=':', alpha=0.5, label='Random Baseline')
ax2.fill_between(data['epochs'], data['gen_success'], alpha=0.3, color='orange')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Success Rate (%)', fontsize=12)
ax2.set_title('Generator Fooling Rate', fontsize=13, fontweight='bold')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)

# --- Plot 3: Loss Curve ---
ax3 = axes[1, 0]
ax3.plot(data['epochs'], data['loss'], 'purple', linewidth=2)
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('Cross-Entropy Loss', fontsize=12)
ax3.set_title('Discriminator Loss', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)

# --- Plot 4: Arms Race Analysis ---
ax4 = axes[1, 1]
# Calculate discriminator advantage (stego acc - gen success)
disc_advantage = np.array(data['stego_acc']) - np.array(data['gen_success'])
colors = ['green' if x > 0 else 'red' for x in disc_advantage]
ax4.bar(data['epochs'], disc_advantage, color=colors, alpha=0.6)
ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax4.set_xlabel('Epoch', fontsize=12)
ax4.set_ylabel('Discriminator Advantage (%)', fontsize=12)
ax4.set_title('Arms Race Balance\n(Positive = Disc Winning, Negative = Gen Winning)',
              fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
print("Saved training_results.png")
plt.show()

# --- Print Summary Statistics ---
print("\n" + "=" * 50)
print("TRAINING SUMMARY")
print("=" * 50)

final_epoch = data['epochs'][-1]
print(f"Total Epochs: {final_epoch}")
print(f"\nFinal Metrics (Epoch {final_epoch}):")
print(f"  • Total Accuracy: {data['total_acc'][-1]:.2f}%")
print(f"  • Cover Accuracy: {data['cover_acc'][-1]:.2f}%")
print(f"  • Stego Accuracy: {data['stego_acc'][-1]:.2f}%")
print(f"  • Generator Success: {data['gen_success'][-1]:.2f}%")
print(f"  • Final Loss: {data['loss'][-1]:.4f}")

# Peak performance
max_acc_idx = np.argmax(data['total_acc'])
print(f"\nPeak Discriminator Performance:")
print(f"  • Epoch {data['epochs'][max_acc_idx]}: {data['total_acc'][max_acc_idx]:.2f}%")

# Best generator performance
max_gen_idx = np.argmax(data['gen_success'])
print(f"\nBest Generator Performance:")
print(f"  • Epoch {data['epochs'][max_gen_idx]}: {data['gen_success'][max_gen_idx]:.2f}%")

# Evolution indicator
if len(data['epochs']) >= 10:
    early_gen = np.mean(data['gen_success'][:5])
    late_gen = np.mean(data['gen_success'][-5:])
    improvement = late_gen - early_gen
    print(f"\nEvolutionary Progress:")
    print(f"  • Early Avg (Epochs 1-5): {early_gen:.2f}%")
    print(f"  • Late Avg (Last 5): {late_gen:.2f}%")
    print(f"  • Improvement: {improvement:+.2f}%")

    if improvement > 10:
        print("Strong evolutionary adaptation!")
    elif improvement > 0:
        print("Modest improvement - consider longer training")
    else:
        print("No improvement - check hyperparameters")

print("=" * 50)
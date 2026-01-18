import json
import matplotlib.pyplot as plt
import numpy as np

# Load training data
with open('../training_history.json', 'r') as f:
    data = json.load(f)

# Create figure with 2 rows, 2 columns
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Hybrid Steganalysis Training Results', fontsize=16, fontweight='bold')

# --- Plot 1: Accuracy Over Time ---
ax1 = axes[0, 0]
# CHANGE: Use 'model_acc' instead of 'total_acc'
ax1.plot(data['epochs'], data['model_acc'], 'b-', linewidth=2, label='Total Accuracy')
# REMOVED: cover_acc and stego_acc (not in your JSON)
ax1.axhline(y=50, color='gray', linestyle=':', alpha=0.5, label='Random Baseline')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_title('Discriminator Performance', fontsize=13, fontweight='bold')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

# --- Plot 2: Generator Success Rate ---
ax2 = axes[0, 1]
ax2.plot(data['epochs'], data['gen_success'], 'orange', linewidth=2, marker='o', markersize=4, label='Gen Fooling Rate')
ax2.axhline(y=50, color='gray', linestyle=':', alpha=0.5, label='Random Baseline')
ax2.fill_between(data['epochs'], data['gen_success'], alpha=0.3, color='orange')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Success Rate (%)', fontsize=12)
ax2.set_title('Generator "Fooling" Rate', fontsize=13, fontweight='bold')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)

# --- Plot 3: Loss Curve ---
ax3 = axes[1, 0]
ax3.plot(data['epochs'], data['loss'], 'purple', linewidth=2, label='Loss')
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('Cross-Entropy Loss', fontsize=12)
ax3.set_title('Discriminator Loss', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)

# --- Plot 4: Learning Rate (New) ---
# Since we don't have stego_acc for the Arms Race plot, visualizing LR is more useful
# to see the Curriculum/Sniper phases.
ax4 = axes[1, 1]
ax4.plot(data['epochs'], data['learning_rate'], 'g-', linewidth=2, label='LR')
ax4.set_xlabel('Epoch', fontsize=12)
ax4.set_ylabel('Learning Rate', fontsize=12)
ax4.set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.ticklabel_format(axis='y', style='sci', scilimits=(0,0)) # Scientific notation for small LR

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
# CHANGE: Updated key names to match JSON
print(f"  • Total Accuracy: {data['model_acc'][-1]:.2f}%")
print(f"  • Generator Success: {data['gen_success'][-1]:.2f}%")
print(f"  • Final Loss: {data['loss'][-1]:.4f}")

# Peak performance
max_acc_idx = np.argmax(data['model_acc'])
print(f"\nPeak Discriminator Performance:")
print(f"  • Epoch {data['epochs'][max_acc_idx]}: {data['model_acc'][max_acc_idx]:.2f}%")

# Best generator performance
max_gen_idx = np.argmax(data['gen_success'])
print(f"\nBest Generator Performance:")
print(f"  • Epoch {data['epochs'][max_gen_idx]}: {data['gen_success'][max_gen_idx]:.2f}%")

print("=" * 50)
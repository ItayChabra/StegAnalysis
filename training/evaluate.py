"""
evaluate.py — Post-training evaluation for the Adversarial Steganalysis model.

Run ONCE after training is complete, using the best checkpoint:
    python training/evaluate.py
    python training/evaluate.py --model srnet_best_val.pth --split dataset_split.json

Reads dataset_split.json (written by train_hybrid.py) to load the held-out
test images that were never seen during training or validation.

Outputs per-strategy:
  - Accuracy, TPR, FPR, AUC
  - Optimal decision threshold (Youden's J)
  - ROC curve plot saved to evaluation_results/roc_curves.png
  - Full metrics saved to evaluation_results/metrics.json
"""

import os
import sys
import json
import random
import argparse
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Allow running from project root or from training/ subfolder.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.srnet import SRNet
from generators.unified_generator import UnifiedGenerator

# ==================== CLI / CONFIGURATION ====================

def parse_args():
    parser = argparse.ArgumentParser(description="Steganalysis model evaluation")
    parser.add_argument(
        '--model', default='srnet_best_val.pth',
        help='Path to model checkpoint (default: srnet_best_val.pth)')
    parser.add_argument(
        '--split', default='dataset_split.json',
        help='Path to dataset split JSON (default: dataset_split.json)')
    parser.add_argument(
        '--output-dir', default='training/evaluation_results',
        help='Directory for output files (default: training/evaluation_results)')
    parser.add_argument(
        '--images-per-strategy', type=int, default=500,
        help='Images to evaluate per strategy (default: 500)')
    return parser.parse_args()


OUTPUT_DIR  = 'training/evaluation_results'
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGES_PER_STRATEGY = 500
EVAL_SEED = 99

STRATEGY_CONFIGS = {
    'sequential': {
        'gen_type':       'lsb',
        'strategy':       'sequential',
        'capacity_ratio': 0.50,
        'edge_threshold': 0,
        'bit_depth':      1,
        'step':           1,
        'message':        None,
    },
    'random': {
        'gen_type':       'lsb',
        'strategy':       'random',
        'capacity_ratio': 0.50,
        'edge_threshold': 0,
        'bit_depth':      1,
        'step':           1,
        'message':        None,
    },
    'skip': {
        'gen_type':       'lsb',
        'strategy':       'skip',
        'capacity_ratio': 0.56,
        'edge_threshold': 95,
        'bit_depth':      1,
        'step':           3,
        'message':        None,
    },
    'edge': {
        'gen_type':       'lsb',
        'strategy':       'edge',
        'capacity_ratio': 0.21,
        'edge_threshold': 9,
        'bit_depth':      1,
        'step':           1,
        'message':        None,
    },
}


# ==================== HELPERS ====================

def load_model(model_path):
    print(f"[SETUP] Loading model: {model_path}  |  Device: {DEVICE}")
    model      = SRNet().to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    model.eval()
    val_acc = checkpoint.get('val_acc', None)
    if val_acc is not None:
        print(f"[SETUP] Checkpoint val_acc at save time: {val_acc:.2f}%")
    return model


def load_test_files(split_file):
    if not os.path.exists(split_file):
        raise FileNotFoundError(
            f"{split_file} not found. Run train_hybrid.py first to generate the split.")
    with open(split_file, 'r') as f:
        split = json.load(f)
    test_files = split['lossy_test'] + split['lossless_test']
    print(f"[SPLIT] Loaded {len(test_files)} test images "
          f"({len(split['lossy_test'])} lossy + {len(split['lossless_test'])} lossless)")
    return test_files


def center_crop_256(image_path):
    """Load and center-crop an image to 256×256. Returns None if too small."""
    img  = Image.open(image_path).convert('L')
    w, h = img.size
    if w < 256 or h < 256:
        return None
    left = (w - 256) // 2
    top  = (h - 256) // 2
    return img.crop((left, top, left + 256, top + 256))


def get_score(model, image: Image.Image, to_tensor) -> float:
    """Run the model on a single PIL image and return P(stego)."""
    tensor = to_tensor(image).unsqueeze(0).pin_memory().to(DEVICE, non_blocking=True)
    with torch.no_grad():
        output = model(tensor)
        prob   = torch.softmax(output, dim=1)[0, 1].item()
    return prob


# ==================== METRICS ====================

def compute_roc(labels, scores):
    labels = np.array(labels)
    scores = np.array(scores)

    thresholds = np.linspace(0.0, 1.0, 201)
    tpr_list   = []
    fpr_list   = []

    n_pos = labels.sum()
    n_neg = len(labels) - n_pos

    for thresh in thresholds:
        preds = (scores >= thresh).astype(int)
        tp    = ((preds == 1) & (labels == 1)).sum()
        fp    = ((preds == 1) & (labels == 0)).sum()
        tpr_list.append(tp / n_pos if n_pos > 0 else 0.0)
        fpr_list.append(fp / n_neg if n_neg > 0 else 0.0)

    fpr_arr = np.array(fpr_list[::-1])
    tpr_arr = np.array(tpr_list[::-1])
    auc     = float(np.trapezoid(tpr_arr, fpr_arr))

    return fpr_list, tpr_list, thresholds.tolist(), auc


def youden_threshold(fpr_list, tpr_list, thresholds):
    best_j      = -1.0
    best_thresh = 0.5
    for fpr, tpr, t in zip(fpr_list, tpr_list, thresholds):
        j = tpr - fpr
        if j > best_j:
            best_j      = j
            best_thresh = t
    return best_thresh, best_j


def compute_accuracy_at_threshold(labels, scores, threshold):
    labels = np.array(labels)
    scores = np.array(scores)
    preds  = (scores >= threshold).astype(int)
    acc    = (preds == labels).mean() * 100.0
    n_pos  = labels.sum()
    n_neg  = len(labels) - n_pos
    tp     = ((preds == 1) & (labels == 1)).sum()
    fp     = ((preds == 1) & (labels == 0)).sum()
    tpr    = tp / n_pos if n_pos > 0 else 0.0
    fpr    = fp / n_neg if n_neg > 0 else 0.0
    return acc, tpr, fpr


def eer(fpr_list, tpr_list, thresholds):
    best_diff   = float('inf')
    best_eer    = 1.0
    best_thresh = 0.5
    for fpr, tpr, t in zip(fpr_list, tpr_list, thresholds):
        fnr  = 1.0 - tpr
        diff = abs(fpr - fnr)
        if diff < best_diff:
            best_diff   = diff
            best_eer    = (fpr + fnr) / 2.0
            best_thresh = t
    return best_eer, best_thresh


# ==================== ROC PLOT ====================

def save_roc_plot(all_roc_data, output_path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        colors  = {'sequential': '#e74c3c', 'random': '#3498db',
                   'skip': '#2ecc71',       'edge': '#f39c12'}

        for strategy, roc in all_roc_data.items():
            fpr = roc['fpr'][::-1]
            tpr = roc['tpr'][::-1]
            auc = roc['auc']
            ax.plot(fpr, tpr,
                    label=f"{strategy}  (AUC={auc:.3f})",
                    color=colors.get(strategy, '#888'),
                    linewidth=2)

        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC=0.500)')
        ax.set_xlabel('False Positive Rate',  fontsize=12)
        ax.set_ylabel('True Positive Rate',   fontsize=12)
        ax.set_title('ROC Curves — Per Strategy\n(test set, never seen during training)',
                     fontsize=13)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"[PLOT] ROC curves saved to {output_path}")

    except ImportError:
        print("[PLOT] matplotlib not installed — skipping ROC plot.")


# ==================== MAIN EVALUATION ====================

def run_evaluation(model_path, split_file, output_dir, images_per_strategy):
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("           STEGANALYSIS MODEL EVALUATION")
    print("=" * 70)

    model        = load_model(model_path)
    test_files   = load_test_files(split_file)
    unified_gen  = UnifiedGenerator()
    to_tensor    = transforms.ToTensor()

    rng = random.Random(EVAL_SEED)
    rng.shuffle(test_files)

    all_metrics  = {}
    all_roc_data = {}

    for strategy_name, config in STRATEGY_CONFIGS.items():
        print(f"\n{'=' * 70}")
        print(f"[STRATEGY: {strategy_name.upper()}]")
        print(f"  Capacity: {config['capacity_ratio']:.2f}  |  "
              f"Edge threshold: {config['edge_threshold']}  |  "
              f"Step: {config.get('step', 1)}")
        print("=" * 70)

        labels = []
        scores = []
        failed = 0

        sampled = rng.sample(test_files, min(images_per_strategy, len(test_files)))

        for i, path in enumerate(sampled):
            crop = center_crop_256(path)
            if crop is None:
                failed += 1
                continue

            clean_score = get_score(model, crop, to_tensor)
            labels.append(0)
            scores.append(clean_score)

            stego_arr, _ = unified_gen.generate_stego(crop, None, config)
            if stego_arr is None:
                failed += 1
                labels.pop()
                scores.pop()
                continue

            stego_img   = Image.fromarray(stego_arr.astype(np.uint8))
            stego_score = get_score(model, stego_img, to_tensor)
            labels.append(1)
            scores.append(stego_score)

            if (i + 1) % 100 == 0:
                print(f"  Progress: {i + 1}/{len(sampled)} images processed...", end='\r')

        print(f"  Processed: {len(sampled) - failed} pairs  "
              f"({failed} skipped — too small or embedding failed)")

        if not labels:
            print("  ERROR: No valid image pairs — skipping strategy.")
            continue

        fpr_list, tpr_list, thresholds, auc = compute_roc(labels, scores)
        opt_thresh, youden_j = youden_threshold(fpr_list, tpr_list, thresholds)
        acc, tpr, fpr = compute_accuracy_at_threshold(labels, scores, opt_thresh)
        eer_val, eer_thresh = eer(fpr_list, tpr_list, thresholds)
        acc_05, tpr_05, fpr_05 = compute_accuracy_at_threshold(labels, scores, 0.50)

        print(f"\n  --- Results ---")
        print(f"  AUC:                        {auc:.4f}")
        print(f"  EER:                        {eer_val * 100:.2f}%  (threshold={eer_thresh:.3f})")
        print(f"\n  At OPTIMAL threshold ({opt_thresh:.3f}, Youden J={youden_j:.3f}):")
        print(f"    Accuracy:  {acc:.2f}%")
        print(f"    TPR:       {tpr * 100:.2f}%  (detection rate)")
        print(f"    FPR:       {fpr * 100:.2f}%  (false positive rate)")
        print(f"\n  At STATIC threshold (0.500):")
        print(f"    Accuracy:  {acc_05:.2f}%")
        print(f"    TPR:       {tpr_05 * 100:.2f}%")
        print(f"    FPR:       {fpr_05 * 100:.2f}%")

        all_metrics[strategy_name] = {
            'n_pairs':           (len(labels) // 2),
            'auc':               round(auc, 4),
            'eer':               round(eer_val, 4),
            'eer_threshold':     round(eer_thresh, 4),
            'optimal_threshold': round(opt_thresh, 4),
            'youden_j':          round(youden_j, 4),
            'acc_at_optimal':    round(acc, 2),
            'tpr_at_optimal':    round(tpr, 4),
            'fpr_at_optimal':    round(fpr, 4),
            'acc_at_0.5':        round(acc_05, 2),
            'tpr_at_0.5':        round(tpr_05, 4),
            'fpr_at_0.5':        round(fpr_05, 4),
        }
        all_roc_data[strategy_name] = {
            'fpr': fpr_list,
            'tpr': tpr_list,
            'auc': auc,
        }

    # --- FINAL SUMMARY TABLE ---
    print(f"\n{'=' * 70}")
    print("                    FINAL EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  {'Strategy':<12}  {'AUC':>6}  {'EER':>6}  "
          f"{'Opt Thresh':>10}  {'TPR@Opt':>9}  {'FPR@Opt':>9}  {'Acc@Opt':>9}")
    print("  " + "-" * 66)

    for name, m in all_metrics.items():
        print(f"  {name:<12}  {m['auc']:>6.4f}  {m['eer'] * 100:>5.1f}%  "
              f"{m['optimal_threshold']:>10.3f}  "
              f"{m['tpr_at_optimal'] * 100:>8.1f}%  "
              f"{m['fpr_at_optimal'] * 100:>8.1f}%  "
              f"{m['acc_at_optimal']:>8.1f}%")

    print("\n  Interpretation guide:")
    print("    AUC > 0.95  → Strong detection")
    print("    AUC 0.80-0.95 → Good detection")
    print("    AUC 0.60-0.80 → Weak detection — more training needed")
    print("    AUC < 0.60  → Near-random — architectural change likely needed")

    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[SAVE] Metrics saved to {metrics_path}")

    roc_path = os.path.join(output_dir, 'roc_curves.png')
    save_roc_plot(all_roc_data, roc_path)

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(
        model_path          = args.model,
        split_file          = args.split,
        output_dir          = args.output_dir,
        images_per_strategy = args.images_per_strategy,
    )
"""
evaluate.py — Post-training evaluation for the Adversarial Steganalysis model.

Run ONCE after training is complete, using the best checkpoint:
    python training/evaluate.py
    python training/evaluate.py --model srnet_best_val.pth --split dataset_split.json

Reads dataset_split.json (written by train_hybrid.py) to load the held-out
test images that were never seen during training or validation.

Evaluates 8 scenarios: 4 LSB + 2 DCT + 2 FFT.

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
    parser = argparse.ArgumentParser(description="Steganalysis model evaluation (LSB + DCT + FFT)")
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


DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EVAL_SEED = 99

STRATEGY_CONFIGS = {
    # ── LSB — spatial domain ──────────────────────────────────────────────────
    'lsb_sequential': {
        'gen_type': 'lsb', 'strategy': 'sequential',
        'capacity_ratio': 0.50, 'edge_threshold': 0,
        'bit_depth': 1, 'step': 1, 'message': None,
    },
    'lsb_random': {
        'gen_type': 'lsb', 'strategy': 'random',
        'capacity_ratio': 0.50, 'edge_threshold': 0,
        'bit_depth': 1, 'step': 1, 'message': None,
    },
    'lsb_skip': {
        'gen_type': 'lsb', 'strategy': 'skip',
        'capacity_ratio': 0.56, 'edge_threshold': 95,
        'bit_depth': 1, 'step': 3, 'message': None,
    },
    'lsb_edge': {
        'gen_type': 'lsb', 'strategy': 'edge',
        'capacity_ratio': 0.21, 'edge_threshold': 9,
        'bit_depth': 1, 'step': 1, 'message': None,
    },
    # ── DCT — block frequency domain ─────────────────────────────────────────
    'dct_mid': {
        'gen_type': 'dct', 'coeff_selection': 'mid',
        'strength': 3.0, 'capacity_ratio': 0.50,
    },
    'dct_low_mid': {
        'gen_type': 'dct', 'coeff_selection': 'low_mid',
        'strength': 2.0, 'capacity_ratio': 0.40,
    },
    # ── FFT — global frequency domain ────────────────────────────────────────
    'fft_low': {
        'gen_type': 'fft', 'freq_band': 'low',
        'strength': 10.0, 'capacity_ratio': 0.35,
    },
    'fft_mid': {
        'gen_type': 'fft', 'freq_band': 'mid',
        'strength': 8.0, 'capacity_ratio': 0.30,
    },
    'fft_high': {
        'gen_type': 'fft', 'freq_band': 'high',
        'strength': 6.0, 'capacity_ratio': 0.25,
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
    labels     = np.array(labels)
    scores     = np.array(scores)
    thresholds = np.linspace(0.0, 1.0, 201)
    tpr_list, fpr_list = [], []
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
    best_j, best_thresh = -1.0, 0.5
    for fpr, tpr, t in zip(fpr_list, tpr_list, thresholds):
        j = tpr - fpr
        if j > best_j:
            best_j, best_thresh = j, t
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


def eer_metric(fpr_list, tpr_list, thresholds):
    best_diff, best_eer, best_thresh = float('inf'), 1.0, 0.5
    for fpr, tpr, t in zip(fpr_list, tpr_list, thresholds):
        fnr  = 1.0 - tpr
        diff = abs(fpr - fnr)
        if diff < best_diff:
            best_diff  = diff
            best_eer   = (fpr + fnr) / 2.0
            best_thresh = t
    return best_eer, best_thresh


# ==================== ROC PLOT ====================

def save_roc_plot(all_roc_data, output_path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('ROC Curves — Per Generator Type (held-out test set)', fontsize=14)
        palette = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12',
                   '#9b59b6', '#1abc9c', '#e67e22', '#34495e']
        groups  = {
            'LSB Strategies': [k for k in all_roc_data if k.startswith('lsb_')],
            'DCT Variants':   [k for k in all_roc_data if k.startswith('dct_')],
            'FFT Variants':   [k for k in all_roc_data if k.startswith('fft_')],
        }
        for ax, (title, keys) in zip(axes, groups.items()):
            for i, key in enumerate(keys):
                if key not in all_roc_data:
                    continue
                r = all_roc_data[key]
                ax.plot(r['fpr'][::-1], r['tpr'][::-1],
                        label=f"{key}  AUC={r['auc']:.3f}",
                        color=palette[i % len(palette)], linewidth=2)
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.4)
            ax.set(title=title, xlabel='FPR', ylabel='TPR', xlim=[0, 1], ylim=[0, 1])
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"[PLOT] ROC curves saved to {output_path}")

    except ImportError:
        print("[PLOT] matplotlib not installed — pip install matplotlib")


# ==================== MAIN EVALUATION ====================

def run_evaluation(model_path, split_file, output_dir, images_per_strategy):
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("           STEGANALYSIS MODEL EVALUATION  (LSB + DCT + FFT)")
    print("=" * 70)

    model        = load_model(model_path)
    test_files   = load_test_files(split_file)
    unified_gen  = UnifiedGenerator()
    to_tensor    = transforms.ToTensor()

    rng = random.Random(EVAL_SEED)
    rng.shuffle(test_files)

    all_metrics  = {}
    all_roc_data = {}

    for name, config in STRATEGY_CONFIGS.items():
        gt = config['gen_type']
        print(f"\n{'=' * 70}")
        print(f"[{name.upper()}]  gen_type={gt}")
        print("=" * 70)

        labels, scores, failed = [], [], 0
        sampled = rng.sample(test_files, min(images_per_strategy, len(test_files)))

        for i, path in enumerate(sampled):
            crop = center_crop_256(path)
            if crop is None:
                failed += 1
                continue

            labels.append(0)
            scores.append(get_score(model, crop, to_tensor))

            stego_arr, _ = unified_gen.generate_stego(crop, None, config)
            if stego_arr is None:
                failed += 1
                labels.pop()
                scores.pop()
                continue

            labels.append(1)
            scores.append(get_score(model, Image.fromarray(stego_arr.astype(np.uint8)), to_tensor))

            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(sampled)}...", end='\r')

        print(f"  Pairs: {len(sampled) - failed}  ({failed} skipped)")
        if not labels:
            print("  ERROR: no valid pairs — skipping.")
            continue

        fpr_list, tpr_list, thresholds, auc = compute_roc(labels, scores)
        opt_t, yj         = youden_threshold(fpr_list, tpr_list, thresholds)
        acc,  tpr,  fpr   = compute_accuracy_at_threshold(labels, scores, opt_t)
        eer_val, eer_t    = eer_metric(fpr_list, tpr_list, thresholds)
        acc5, tpr5, fpr5  = compute_accuracy_at_threshold(labels, scores, 0.50)

        print(f"\n  --- Results ---")
        print(f"  AUC: {auc:.4f}  |  EER: {eer_val * 100:.2f}%  (threshold={eer_t:.3f})")
        print(f"\n  At OPTIMAL threshold ({opt_t:.3f}, Youden J={yj:.3f}):")
        print(f"    Accuracy: {acc:.2f}%  |  TPR: {tpr * 100:.1f}%  |  FPR: {fpr * 100:.1f}%")
        print(f"\n  At STATIC threshold (0.500):")
        print(f"    Accuracy: {acc5:.2f}%  |  TPR: {tpr5 * 100:.1f}%  |  FPR: {fpr5 * 100:.1f}%")

        all_metrics[name] = {
            'gen_type':          gt,
            'n_pairs':           len(labels) // 2,
            'auc':               round(auc, 4),
            'eer':               round(eer_val, 4),
            'eer_threshold':     round(eer_t, 4),
            'optimal_threshold': round(opt_t, 4),
            'youden_j':          round(yj, 4),
            'acc_at_optimal':    round(acc, 2),
            'tpr_at_optimal':    round(tpr, 4),
            'fpr_at_optimal':    round(fpr, 4),
            'acc_at_0.5':        round(acc5, 2),
            'tpr_at_0.5':        round(tpr5, 4),
            'fpr_at_0.5':        round(fpr5, 4),
        }
        all_roc_data[name] = {'fpr': fpr_list, 'tpr': tpr_list, 'auc': auc}

    # --- FINAL SUMMARY TABLE ---
    print(f"\n{'=' * 70}")
    print("                    FINAL EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  {'Strategy':<18}  {'Type':>4}  {'AUC':>6}  {'EER':>6}  "
          f"{'OptT':>5}  {'TPR':>6}  {'FPR':>6}  {'Acc':>6}")
    print("  " + "-" * 65)
    for n, m in all_metrics.items():
        print(f"  {n:<18}  {m['gen_type']:>4}  {m['auc']:>6.4f}  "
              f"{m['eer'] * 100:>5.1f}%  {m['optimal_threshold']:>5.3f}  "
              f"{m['tpr_at_optimal'] * 100:>5.1f}%  {m['fpr_at_optimal'] * 100:>5.1f}%  "
              f"{m['acc_at_optimal']:>5.1f}%")

    print("\n  AUC: >0.95 strong | 0.80-0.95 good | 0.60-0.80 weak | <0.60 near-random")

    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[SAVE] Metrics saved to {metrics_path}")

    save_roc_plot(all_roc_data, os.path.join(output_dir, 'roc_curves.png'))
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(
        model_path          = args.model,
        split_file          = args.split,
        output_dir          = args.output_dir,
        images_per_strategy = args.images_per_strategy,
    )
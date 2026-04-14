"""
evaluate.py — Post-training evaluation for the Adversarial Steganalysis model.

Run ONCE after training is complete, using the best checkpoint:
    python training/evaluate.py
    python training/evaluate.py --model srnet_best_val.pth --split dataset_split.json

Reads dataset_split.json (written by train_hybrid.py) to load the held-out
test images that were never seen during training or validation.

KEY CHANGE vs previous version:
  Each strategy is now evaluated across MULTIPLE parameter configurations
  (low / mid / high capacity and strength), not just a single fixed config.
  Per-strategy results report MEAN and MIN AUC across all configs.
  For a security application, MIN AUC is the operative number — it tells you
  the worst-case operating point a real attacker could exploit.

Evaluates strategies: 4 LSB + 2 DCT + 3 FFT (including fft_low variants).

Outputs:
  - Per-strategy mean/min AUC table printed to console
  - Full per-config metrics saved to evaluation_results/metrics.json
  - ROC curves (one per strategy group) saved to evaluation_results/roc_curves.png
  - Min-AUC summary saved to evaluation_results/min_auc_summary.json
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.srnet import SRNet
from generators.unified_generator import UnifiedGenerator


# ==================== CLI / CONFIGURATION ====================

def parse_args():
    parser = argparse.ArgumentParser(description="Steganalysis model evaluation (multi-config)")
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
        '--images-per-config', type=int, default=300,
        help='Images per strategy config variant (default: 300). '
             'Total images = images-per-config × num_configs_per_strategy.')
    return parser.parse_args()


DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EVAL_SEED = 99

# ── Strategy configs ────────────────────────────────────────────────────────
# Each strategy now has a LIST of configs representing low / mid / high
# difficulty operating points.
#
# Design principle: the single-config approach from the previous evaluator
# gives you AUC at one point in parameter space. If that point happens to
# sit in a region the model saw frequently during training, the result is
# optimistic. Testing across a range gives an honest picture.
#
# For LSB: capacity_ratio drives difficulty (lower = fewer pixels modified = harder).
# For DCT/FFT: strength drives visibility (lower strength = subtler = harder).
#
# The "reference" config (middle entry) matches what the previous evaluator used,
# so results are directly comparable.

STRATEGY_CONFIGS = {

    # ── LSB Spatial ──────────────────────────────────────────────────────────

    'lsb_sequential': [
        {'label': 'low_cap',  'gen_type': 'lsb', 'strategy': 'sequential', 'capacity_ratio': 0.25, 'edge_threshold': 0, 'bit_depth': 1, 'step': 1, 'message': None},
        {'label': 'mid_cap',  'gen_type': 'lsb', 'strategy': 'sequential', 'capacity_ratio': 0.50, 'edge_threshold': 0, 'bit_depth': 1, 'step': 1, 'message': None},  # reference
        {'label': 'high_cap', 'gen_type': 'lsb', 'strategy': 'sequential', 'capacity_ratio': 0.75, 'edge_threshold': 0, 'bit_depth': 1, 'step': 1, 'message': None},
    ],

    'lsb_random': [
        {'label': 'low_cap',  'gen_type': 'lsb', 'strategy': 'random', 'capacity_ratio': 0.25, 'edge_threshold': 0, 'bit_depth': 1, 'step': 1, 'message': None},
        {'label': 'mid_cap',  'gen_type': 'lsb', 'strategy': 'random', 'capacity_ratio': 0.50, 'edge_threshold': 0, 'bit_depth': 1, 'step': 1, 'message': None},  # reference
        {'label': 'high_cap', 'gen_type': 'lsb', 'strategy': 'random', 'capacity_ratio': 0.75, 'edge_threshold': 0, 'bit_depth': 1, 'step': 1, 'message': None},
    ],

    'lsb_skip': [
        {'label': 'tight_step',  'gen_type': 'lsb', 'strategy': 'skip', 'capacity_ratio': 0.40, 'edge_threshold': 95, 'bit_depth': 1, 'step': 2,  'message': None},
        {'label': 'mid_step',    'gen_type': 'lsb', 'strategy': 'skip', 'capacity_ratio': 0.56, 'edge_threshold': 95, 'bit_depth': 1, 'step': 3,  'message': None},  # reference
        {'label': 'large_step',  'gen_type': 'lsb', 'strategy': 'skip', 'capacity_ratio': 0.70, 'edge_threshold': 95, 'bit_depth': 1, 'step': 7,  'message': None},
    ],

    'lsb_edge': [
        # Edge embedding is hardest at low capacity + low threshold (few edge pixels used).
        # The reference config is the hardest point; we also test slightly easier variants.
        {'label': 'hard',   'gen_type': 'lsb', 'strategy': 'edge', 'capacity_ratio': 0.21, 'edge_threshold': 9,  'bit_depth': 1, 'step': 1, 'message': None},  # reference
        {'label': 'medium', 'gen_type': 'lsb', 'strategy': 'edge', 'capacity_ratio': 0.35, 'edge_threshold': 15, 'bit_depth': 1, 'step': 1, 'message': None},
        {'label': 'easy',   'gen_type': 'lsb', 'strategy': 'edge', 'capacity_ratio': 0.50, 'edge_threshold': 30, 'bit_depth': 1, 'step': 1, 'message': None},
    ],

    # ── DCT Block Frequency ───────────────────────────────────────────────────

    'dct_mid': [
        {'label': 'low_strength',  'gen_type': 'dct', 'coeff_selection': 'mid', 'strength': 1.5, 'capacity_ratio': 0.50},
        {'label': 'mid_strength',  'gen_type': 'dct', 'coeff_selection': 'mid', 'strength': 3.0, 'capacity_ratio': 0.50},  # reference
        {'label': 'high_strength', 'gen_type': 'dct', 'coeff_selection': 'mid', 'strength': 6.0, 'capacity_ratio': 0.50},
    ],

    'dct_low_mid': [
        {'label': 'low_strength',  'gen_type': 'dct', 'coeff_selection': 'low_mid', 'strength': 1.0, 'capacity_ratio': 0.40},
        {'label': 'mid_strength',  'gen_type': 'dct', 'coeff_selection': 'low_mid', 'strength': 2.0, 'capacity_ratio': 0.40},  # reference
        {'label': 'high_strength', 'gen_type': 'dct', 'coeff_selection': 'low_mid', 'strength': 4.0, 'capacity_ratio': 0.40},
    ],

    # ── FFT Global Frequency ──────────────────────────────────────────────────

    # fft_low: the run-3 weak spot. Three configs bracket the reference point
    # (strength=10, capacity=0.35) so we see whether the fix produces robust
    # coverage or just memorizes the reference.
    'fft_low': [
        {'label': 'low_strength',  'gen_type': 'fft', 'freq_band': 'low', 'strength': 5.0,  'capacity_ratio': 0.50},
        {'label': 'mid_strength',  'gen_type': 'fft', 'freq_band': 'low', 'strength': 10.0, 'capacity_ratio': 0.35},  # reference (run-3 eval point)
        {'label': 'high_strength', 'gen_type': 'fft', 'freq_band': 'low', 'strength': 15.0, 'capacity_ratio': 0.25},
    ],

    'fft_mid': [
        {'label': 'low_strength',  'gen_type': 'fft', 'freq_band': 'mid', 'strength': 4.0, 'capacity_ratio': 0.30},
        {'label': 'mid_strength',  'gen_type': 'fft', 'freq_band': 'mid', 'strength': 8.0, 'capacity_ratio': 0.30},  # reference
        {'label': 'high_strength', 'gen_type': 'fft', 'freq_band': 'mid', 'strength': 14.0,'capacity_ratio': 0.30},
    ],

    'fft_high': [
        {'label': 'low_strength',  'gen_type': 'fft', 'freq_band': 'high', 'strength': 3.0, 'capacity_ratio': 0.25},
        {'label': 'mid_strength',  'gen_type': 'fft', 'freq_band': 'high', 'strength': 6.0, 'capacity_ratio': 0.25},  # reference
        {'label': 'high_strength', 'gen_type': 'fft', 'freq_band': 'high', 'strength': 10.0,'capacity_ratio': 0.25},
    ],
}

# AUC quality bands for the summary table.
AUC_BANDS = [
    (0.95, 'STRONG  ✓✓'),
    (0.90, 'good    ✓ '),
    (0.80, 'weak    ~ '),
    (0.60, 'poor    ✗ '),
    (0.00, 'RANDOM  ✗✗'),
]

def auc_label(auc):
    for threshold, label in AUC_BANDS:
        if auc >= threshold:
            return label
    return 'RANDOM  ✗✗'


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
            best_diff   = diff
            best_eer    = (fpr + fnr) / 2.0
            best_thresh = t
    return best_eer, best_thresh


def evaluate_single_config(model, test_files, unified_gen, to_tensor, config, n_images, rng):
    """
    Run model against one specific generator config on n_images test images.
    Returns (labels, scores, n_failed).
    """
    sampled = rng.sample(test_files, min(n_images, len(test_files)))
    labels, scores, failed = [], [], 0

    # Strip the 'label' key before passing to the generator.
    gen_config = {k: v for k, v in config.items() if k != 'label'}

    for path in sampled:
        crop = center_crop_256(path)
        if crop is None:
            failed += 1
            continue

        labels.append(0)
        scores.append(get_score(model, crop, to_tensor))

        stego_arr, _ = unified_gen.generate_stego(crop, None, gen_config)
        if stego_arr is None:
            failed += 1
            labels.pop()
            scores.pop()
            continue

        labels.append(1)
        scores.append(get_score(model, Image.fromarray(stego_arr.astype(np.uint8)), to_tensor))

    return labels, scores, failed


# ==================== ROC PLOT ====================

def save_roc_plot(all_strategy_roc, output_path):
    """
    Plots one ROC curve per strategy (using the reference/mid config).
    Strategies are grouped into LSB / DCT / FFT subplots.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(21, 7))
        fig.suptitle('ROC Curves — Per Strategy, Reference Config (held-out test set)',
                     fontsize=14)
        palette = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12',
                   '#9b59b6', '#1abc9c', '#e67e22', '#34495e', '#c0392b']
        groups  = {
            'LSB Strategies': [k for k in all_strategy_roc if k.startswith('lsb_')],
            'DCT Variants':   [k for k in all_strategy_roc if k.startswith('dct_')],
            'FFT Variants':   [k for k in all_strategy_roc if k.startswith('fft_')],
        }
        for ax, (title, keys) in zip(axes, groups.items()):
            for i, key in enumerate(keys):
                if key not in all_strategy_roc:
                    continue
                r    = all_strategy_roc[key]
                mean = r['mean_auc']
                min_ = r['min_auc']
                ax.plot(r['ref_fpr'][::-1], r['ref_tpr'][::-1],
                        label=f"{key}  mean={mean:.3f}  min={min_:.3f}",
                        color=palette[i % len(palette)], linewidth=2)
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.4)
            ax.set(title=title, xlabel='FPR', ylabel='TPR', xlim=[0, 1], ylim=[0, 1])
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"[PLOT] ROC curves saved to {output_path}")

    except ImportError:
        print("[PLOT] matplotlib not installed — pip install matplotlib")


# ==================== MAIN EVALUATION ====================

def run_evaluation(model_path, split_file, output_dir, images_per_config):
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 75)
    print("        STEGANALYSIS EVALUATION  (multi-config, LSB + DCT + FFT)")
    print("=" * 75)
    print(f"  Images per config variant: {images_per_config}")
    print(f"  Configs per strategy:      3  (low / mid[reference] / high)")
    print(f"  Total test pairs per strat: up to {images_per_config * 3}")
    print()

    model        = load_model(model_path)
    test_files   = load_test_files(split_file)
    unified_gen  = UnifiedGenerator()
    to_tensor    = transforms.ToTensor()

    rng = random.Random(EVAL_SEED)

    all_metrics       = {}   # strategy → list of per-config dicts
    all_strategy_roc  = {}   # strategy → {mean_auc, min_auc, ref_fpr, ref_tpr}
    min_auc_summary   = {}   # strategy → min AUC (the operative security number)

    for strategy_name, config_list in STRATEGY_CONFIGS.items():
        gt = config_list[0]['gen_type']
        print(f"\n{'=' * 75}")
        print(f"[{strategy_name.upper()}]  gen_type={gt}  "
              f"({len(config_list)} configs × {images_per_config} images)")
        print("=" * 75)

        config_results = []
        ref_fpr = ref_tpr = None

        for cfg in config_list:
            label = cfg.get('label', '?')
            print(f"\n  ── Config: {label} ──")

            labels, scores, failed = evaluate_single_config(
                model, test_files, unified_gen, to_tensor, cfg, images_per_config, rng)

            n_pairs = len(labels) // 2
            print(f"  Pairs: {n_pairs}  ({failed} skipped)")

            if not labels:
                print("  ERROR: no valid pairs — skipping config.")
                continue

            fpr_list, tpr_list, thresholds, auc = compute_roc(labels, scores)
            opt_t, yj       = youden_threshold(fpr_list, tpr_list, thresholds)
            acc, tpr, fpr   = compute_accuracy_at_threshold(labels, scores, opt_t)
            eer_val, eer_t  = eer_metric(fpr_list, tpr_list, thresholds)
            acc5, tpr5, fpr5 = compute_accuracy_at_threshold(labels, scores, 0.50)

            print(f"  AUC: {auc:.4f}  [{auc_label(auc)}]  |  EER: {eer_val*100:.2f}%")
            print(f"  At optimal threshold ({opt_t:.3f}):  "
                  f"Acc {acc:.2f}%  TPR {tpr*100:.1f}%  FPR {fpr*100:.1f}%")
            print(f"  At static threshold  (0.500):  "
                  f"Acc {acc5:.2f}%  TPR {tpr5*100:.1f}%  FPR {fpr5*100:.1f}%")

            result = {
                'config_label':      label,
                'gen_type':          gt,
                'config':            {k: v for k, v in cfg.items() if k != 'label'},
                'n_pairs':           n_pairs,
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
            config_results.append(result)

            # Use the middle (reference) config for the ROC plot curve.
            if label in ('mid_cap', 'mid_strength', 'mid_step', 'hard', 'mid_strength'):
                ref_fpr = fpr_list
                ref_tpr = tpr_list
            # Fallback: always keep the last one in case none matched.
            if ref_fpr is None:
                ref_fpr = fpr_list
                ref_tpr = tpr_list

        if not config_results:
            print(f"  [WARN] No valid configs for {strategy_name} — skipping.")
            continue

        auc_values = [r['auc'] for r in config_results]
        mean_auc   = float(np.mean(auc_values))
        min_auc    = float(np.min(auc_values))
        max_auc    = float(np.max(auc_values))

        print(f"\n  ── {strategy_name} SUMMARY ──")
        print(f"  AUC:  mean={mean_auc:.4f}  min={min_auc:.4f}  max={max_auc:.4f}")
        print(f"  Operative security AUC (min): {min_auc:.4f}  [{auc_label(min_auc)}]")

        all_metrics[strategy_name] = {
            'gen_type':    gt,
            'mean_auc':    round(mean_auc, 4),
            'min_auc':     round(min_auc, 4),
            'max_auc':     round(max_auc, 4),
            'configs':     config_results,
        }
        all_strategy_roc[strategy_name] = {
            'mean_auc': mean_auc,
            'min_auc':  min_auc,
            'ref_fpr':  ref_fpr,
            'ref_tpr':  ref_tpr,
        }
        min_auc_summary[strategy_name] = round(min_auc, 4)

    # ── FINAL SUMMARY TABLE ──────────────────────────────────────────────────
    print(f"\n{'=' * 75}")
    print("                     FINAL EVALUATION SUMMARY")
    print("  Operative security number = MIN AUC across all config variants.")
    print("  An attacker will choose the parameter point where your model is weakest.")
    print("=" * 75)
    print(f"  {'Strategy':<20}  {'Type':>4}  {'Mean AUC':>8}  {'Min AUC':>8}  "
          f"{'Max AUC':>8}  {'Min Rating':>12}")
    print("  " + "-" * 68)

    for name, m in all_metrics.items():
        rating = auc_label(m['min_auc'])
        flag   = '  ⚠' if m['min_auc'] < 0.90 else ''
        print(f"  {name:<20}  {m['gen_type']:>4}  "
              f"{m['mean_auc']:>8.4f}  {m['min_auc']:>8.4f}  {m['max_auc']:>8.4f}  "
              f"{rating}{flag}")

    print()
    print("  AUC bands: ≥0.95 STRONG ✓✓ | 0.90–0.95 good ✓ | "
          "0.80–0.90 weak ~ | 0.60–0.80 poor ✗ | <0.60 RANDOM ✗✗")
    print("  ⚠  = min AUC below 0.90 — address before next training phase")

    # Identify any strategies needing attention.
    weak = [(n, m['min_auc']) for n, m in all_metrics.items() if m['min_auc'] < 0.90]
    if weak:
        print(f"\n  [ACTION REQUIRED] {len(weak)} strategy(ies) below 0.90 min AUC:")
        for name, auc in sorted(weak, key=lambda x: x[1]):
            print(f"    {name:<20}  min AUC={auc:.4f}")
    else:
        print("\n  [PASS] All strategies ≥ 0.90 min AUC — suite is ready for next phase.")

    # ── SAVE OUTPUTS ─────────────────────────────────────────────────────────
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[SAVE] Full metrics saved to {metrics_path}")

    min_auc_path = os.path.join(output_dir, 'min_auc_summary.json')
    with open(min_auc_path, 'w') as f:
        json.dump(min_auc_summary, f, indent=2)
    print(f"[SAVE] Min-AUC summary saved to {min_auc_path}")

    save_roc_plot(all_strategy_roc, os.path.join(output_dir, 'roc_curves.png'))
    print("\n" + "=" * 75 + "\n")


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(
        model_path       = args.model,
        split_file       = args.split,
        output_dir       = args.output_dir,
        images_per_config = args.images_per_config,
    )
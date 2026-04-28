"""
finetune.py — Targeted fine-tuning from a pre-trained checkpoint.

Key design decisions vs the EA-based approach
----------------------------------------------
NO EVOLUTIONARY ALGORITHM
    The EA is actively harmful during fine-tuning: whichever generator has the
    highest fool rate (fft_low Str=2.0 in your case) floods every batch, causing
    the model to forget everything else. Val accuracy collapses from 83% → 63%.

FIXED WEIGHTED SAMPLER
    Each batch is assembled from a fixed menu of strategy configs. The sampling
    weight of each strategy is inversely proportional to its min-AUC from the
    last evaluation — weak spots get more batches, strong ones get fewer but
    are never starved.

FROZEN BACKBONE OPTION (first N epochs)
    Optionally freeze layers 2–7 for the first few epochs so only the
    classification head adapts to the new examples. Prevents the most
    aggressive forgetting.

LEARNING RATE — TWO-PHASE COSINE
    Phase 1 (backbone frozen): cosine from FT_MAX_LR → FT_MIN_LR over
    FREEZE_BACKBONE_EPOCHS epochs. Head-only updates can tolerate higher LR.

    Phase 2 (backbone unfrozen): fresh cosine cycle starting from FT_MAX_LR_FULL
    (lower ceiling) → FT_MIN_LR over the remaining epochs. Opening the backbone
    into a high LR causes the val-acc dip seen when using a single shared cycle —
    the model only stabilises around epoch 9 when the LR has decayed far enough.
    Two independent cycles eliminate this wasted window.

Usage
-----
    python finetune.py
    python finetune.py --checkpoint srnet_best_val.pth --epochs 20
"""

import argparse
import json
import math
import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from generators.unified_generator import UnifiedGenerator
from models.srnet import SRNet
from training.dataset import create_or_load_split, load_balanced_dataset
from training.genome import compute_log_fft
from training.utils import save_checkpoint
from training.validate import run_validation
import training.config as cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
cudnn.benchmark = True

# ── Fine-tune hyperparameters ─────────────────────────────────────────────────

FT_EPOCHS = 20
FT_MAX_LR = 1e-5  # Phase 1 (head-only): high LR is safe, backbone frozen
FT_MAX_LR_FULL = 3e-6  # Phase 2 (full model): lower ceiling protects backbone
FT_MIN_LR = 1e-7  # Floor for both phases
FT_BATCH_SIZE = 64
FT_ACCUM_STEPS = 2
FT_WORKERS = max(1, cfg.NUM_WORKERS)

# Freeze the residual backbone for this many epochs, adapting only the head.
# Set to 0 to fine-tune all layers from the start.
FREEZE_BACKBONE_EPOCHS = 4

DEFAULT_CHECKPOINT = 'srnet_best_val.pth'

# ── Weighted strategy menu ────────────────────────────────────────────────────
# Each entry is a (weight, config_dict) pair.
# Weight = 1 - min_AUC from the last evaluation run.
# Strategies that scored ≥ 0.98 get a floor weight of 0.02 so they never
# vanish from the batch and the model doesn't forget them.
#
# To update weights: paste your latest min_auc_summary.json values into
# MIN_AUC_FROM_EVAL below and the menu rebuilds automatically.

MIN_AUC_FROM_EVAL = {
    "lsb_sequential": 0.958,
    "lsb_random": 0.9942,
    "lsb_skip": 0.9896,
    "lsb_edge": 0.9117,
    "dct_mid": 0.9822,
    "dct_low_mid": 0.9761,
    "fft_low": 0.9142,
    "fft_mid": 0.9829,
    "fft_high": 0.9864
}

# Reference configs — one representative per strategy (the hard / low-strength one).
_STRATEGY_CONFIGS = [
    # ── LSB ──────────────────────────────────────────────────────────────────
    ('lsb_sequential', {'gen_type': 'lsb', 'strategy': 'sequential',
                        'capacity_ratio': 0.25, 'bit_depth': 1, 'step': 1}),
    ('lsb_random', {'gen_type': 'lsb', 'strategy': 'random',
                    'capacity_ratio': 0.25, 'bit_depth': 1}),
    ('lsb_skip', {'gen_type': 'lsb', 'strategy': 'skip',
                  'capacity_ratio': 0.40, 'bit_depth': 1, 'step': 7,
                  'edge_threshold': 95}),
    ('lsb_edge', {'gen_type': 'lsb', 'strategy': 'edge',
                  'capacity_ratio': 0.21, 'bit_depth': 1,
                  'edge_threshold': 9}),

    # ── DCT ──────────────────────────────────────────────────────────────────
    ('dct_mid', {'gen_type': 'dct', 'coeff_selection': 'mid',
                 'strength': 1.5, 'capacity_ratio': 0.50}),
    ('dct_low_mid', {'gen_type': 'dct', 'coeff_selection': 'low_mid',
                     'strength': 2.0, 'capacity_ratio': 0.40}),

    # ── FFT ──────────────────────────────────────────────────────────────────
    ('fft_low', {'gen_type': 'fft', 'freq_band': 'low',
                 'strength': 5.0, 'capacity_ratio': 0.50}),
    ('fft_mid', {'gen_type': 'fft', 'freq_band': 'mid',
                 'strength': 4.0, 'capacity_ratio': 0.30}),
    ('fft_high', {'gen_type': 'fft', 'freq_band': 'high',
                  'strength': 3.0, 'capacity_ratio': 0.25}),
]


def _build_sampler():
    """
    Returns (strategy_names, configs, weights) with weights normalised to sum=1.
    Weak strategies (low AUC) get higher weight; strong ones get a floor of 0.02.
    """
    names, configs, weights = [], [], []
    for name, config in _STRATEGY_CONFIGS:
        min_auc = MIN_AUC_FROM_EVAL.get(name, 0.90)
        w = max(0.02, 1.0 - min_auc)
        names.append(name)
        configs.append(config)
        weights.append(w)

    total = sum(weights)
    weights = [w / total for w in weights]

    print("[SAMPLER] Strategy weights:")
    for n, w in zip(names, weights):
        bar = '█' * int(w * 40)
        print(f"  {n:<20} {w:.3f}  {bar}")

    return names, configs, weights


# ── LR schedule ───────────────────────────────────────────────────────────────

def _cosine_lr(optimizer, epoch, total_epochs):
    UNFREEZE_WARMUP = 2

    if epoch < FREEZE_BACKBONE_EPOCHS:
        phase_len = max(1, FREEZE_BACKBONE_EPOCHS)
        progress = epoch / max(1, phase_len - 1) if phase_len > 1 else 1.0
        lr = FT_MIN_LR + 0.5 * (FT_MAX_LR - FT_MIN_LR) * (1 + math.cos(math.pi * progress))

    else:
        phase_epoch = epoch - FREEZE_BACKBONE_EPOCHS
        phase_len = max(1, total_epochs - FREEZE_BACKBONE_EPOCHS)

        if phase_epoch < UNFREEZE_WARMUP:
            ramp_progress = (phase_epoch + 1) / UNFREEZE_WARMUP
            lr = FT_MIN_LR + ramp_progress * (FT_MAX_LR_FULL - FT_MIN_LR)
        else:
            cosine_epoch = phase_epoch - UNFREEZE_WARMUP
            cosine_len = max(1, phase_len - UNFREEZE_WARMUP)
            progress = cosine_epoch / max(1, cosine_len - 1) if cosine_len > 1 else 1.0
            lr = FT_MIN_LR + 0.5 * (FT_MAX_LR_FULL - FT_MIN_LR) * (1 + math.cos(math.pi * progress))

    for pg in optimizer.param_groups:
        pg['lr'] = lr
    return lr


# ── Backbone freeze / unfreeze ────────────────────────────────────────────────

def _set_backbone_frozen(model, frozen: bool):
    """
    Freeze or unfreeze layers 2–7 (the shallow residual blocks).
    Branches and layers 8–11 (the deep pooling blocks) + FC always train.
    """
    raw = model._orig_mod if hasattr(model, '_orig_mod') else model
    freeze_prefixes = ('layer2', 'layer3', 'layer4', 'layer5', 'layer6', 'layer7')
    for name, param in raw.named_parameters():
        if any(name.startswith(p) for p in freeze_prefixes):
            param.requires_grad = not frozen

    state = "FROZEN" if frozen else "UNFROZEN"
    print(f"[FREEZE] Backbone layers 2–7: {state}")


# ── Single pair generation ────────────────────────────────────────────────────

def _generate_pair(args):
    path, config, unified_gen, to_tensor = args
    try:
        with Image.open(path) as img:
            cover_img = img.convert('L')
        w, h = cover_img.size
        if w < 256 or h < 256:
            return None

        i_c, j_c, h_c, w_c = transforms.RandomCrop.get_params(
            cover_img, output_size=(256, 256))
        cover_crop = TF.crop(cover_img, i_c, j_c, h_c, w_c)

        # Randomise capacity within ±0.10 of the reference config to keep variety
        cfg_used = config.copy()
        base_cap = config.get('capacity_ratio', 0.5)
        cfg_used['capacity_ratio'] = float(np.clip(
            base_cap + random.uniform(-0.10, 0.10),
            cfg.MIN_CAPACITY, cfg.MAX_CAPACITY))

        stego_arr, _ = unified_gen.generate_stego(cover_crop, None, cfg_used)
        if stego_arr is None:
            return None

        spatial_cover = TF.to_tensor(cover_crop)
        cover_t = torch.cat([spatial_cover, compute_log_fft(spatial_cover)], dim=0)

        spatial_stego = TF.to_tensor(Image.fromarray(stego_arr))
        stego_t = torch.cat([spatial_stego, compute_log_fft(spatial_stego)], dim=0)

        return cover_t, stego_t

    except Exception as e:
        print(f"\n[GEN ERROR] {config.get('gen_type', '?')}: {e}")
        return None

# ── Main fine-tuning loop ─────────────────────────────────────────────────────

def run_finetune(checkpoint_path: str, epochs: int):
    phase2_lr_str = f"{FT_MAX_LR_FULL:.0e} → {FT_MIN_LR:.0e}"
    print(f"\n{'=' * 65}")
    print("  FINE-TUNING RUN (fixed weighted sampler — no EA)")
    print(f"  Base checkpoint : {checkpoint_path}")
    print(f"  LR phase 1      : {FT_MAX_LR:.0e} → {FT_MIN_LR:.0e}  (head only, {FREEZE_BACKBONE_EPOCHS} epochs)")
    print(f"  LR phase 2      : {phase2_lr_str}  (full model, {epochs - FREEZE_BACKBONE_EPOCHS} epochs)")
    print(f"  Freeze backbone : first {FREEZE_BACKBONE_EPOCHS} epochs")
    print(f"  Epochs          : {epochs}")
    print('=' * 65)

    # ── Data ──────────────────────────────────────────────────────────────────
    lossy_files, lossless_files = load_balanced_dataset('data/raw')
    split = create_or_load_split(lossy_files, lossless_files)
    train_files = split['lossy_train'] + split['lossless_train']
    val_lossy = split['lossy_val']
    val_lossless = split['lossless_val']

    print(f"\n[DATA] Train pool: {len(train_files)} images")
    print(f"[DATA] Val pool:   {len(val_lossy) + len(val_lossless)} images")

    strategy_names, strategy_configs, strategy_weights = _build_sampler()

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SRNet().to(cfg.DEVICE)
    model = torch.compile(model, mode='reduce-overhead')

    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    ckpt = torch.load(checkpoint_path, map_location=cfg.DEVICE, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    raw_model.load_state_dict(state_dict)

    base_val_acc = ckpt.get('val_acc', 0.0)
    print(f"[RESUME] Loaded checkpoint — base val_acc: {base_val_acc:.2f}%")

    optimizer = optim.Adam(model.parameters(), lr=FT_MAX_LR, weight_decay=2e-4)
    criterion = nn.CrossEntropyLoss(reduction='none', label_smoothing=0.1)
    scaler = torch.amp.GradScaler('cuda')

    unified_gen = UnifiedGenerator()
    to_tensor = transforms.ToTensor()

    steps_per_epoch = max(1, len(train_files) // FT_BATCH_SIZE)
    best_val_acc = base_val_acc
    best_val_epoch = 0

    history = {'epoch': [], 'train_acc': [], 'val_acc': [], 'lr': [], 'strategy_counts': []}

    for epoch in range(epochs):
        lr = _cosine_lr(optimizer, epoch, epochs)

        # Freeze or unfreeze backbone at phase boundary
        if epoch == 0 and FREEZE_BACKBONE_EPOCHS > 0:
            _set_backbone_frozen(model, frozen=True)
        elif epoch == FREEZE_BACKBONE_EPOCHS:
            _set_backbone_frozen(model, frozen=False)

        print(f"\n{'=' * 65}")
        print(f"Epoch {epoch + 1}/{epochs} | LR: {lr:.2e}")
        print('=' * 65)

        random.shuffle(train_files)

        total_loss = 0.0
        correct = 0
        total_samples = 0
        strategy_counts = {n: 0 for n in strategy_names}

        model.train()
        optimizer.zero_grad()

        with ThreadPoolExecutor(max_workers=FT_WORKERS) as executor:
            for step in range(steps_per_epoch):
                batch_paths = train_files[step * FT_BATCH_SIZE:
                                          (step + 1) * FT_BATCH_SIZE]
                chosen_indices = random.choices(
                    range(len(strategy_configs)),
                    weights=strategy_weights,
                    k=len(batch_paths))

                args_list = [
                    (path, strategy_configs[idx], unified_gen, to_tensor)
                    for path, idx in zip(batch_paths, chosen_indices)
                ]

                for idx in chosen_indices:
                    strategy_counts[strategy_names[idx]] += 1

                inputs, labels = [], []
                for res in executor.map(_generate_pair, args_list):
                    if res is None:
                        continue
                    cover_t, stego_t = res
                    inputs.extend([cover_t, stego_t])
                    labels.extend([0, 1])

                if not inputs:
                    continue

                inputs_t = torch.stack(inputs).to(cfg.DEVICE, non_blocking=True)
                labels_t = torch.tensor(labels, dtype=torch.long).to(cfg.DEVICE)

                perm = torch.randperm(inputs_t.size(0))
                inputs_t = inputs_t[perm]
                labels_t = labels_t[perm]

                with torch.amp.autocast('cuda'):
                    outputs = model(inputs_t)
                    per_loss = criterion(outputs, labels_t)
                    loss = per_loss.mean() / FT_ACCUM_STEPS

                scaler.scale(loss).backward()

                if (step + 1) % FT_ACCUM_STEPS == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                _, preds = torch.max(outputs, 1)
                total_loss += loss.item() * FT_ACCUM_STEPS
                correct += (preds == labels_t).sum().item()
                total_samples += labels_t.size(0)

                if step % 20 == 0:
                    print(f"\rStep {step}/{steps_per_epoch} | "
                          f"Loss: {loss.item() * FT_ACCUM_STEPS:.4f} | "
                          f"Acc: {100 * correct / max(1, total_samples):.1f}%", end="")

        # Final gradient flush
        if steps_per_epoch % FT_ACCUM_STEPS != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        torch.cuda.empty_cache()

        train_acc = 100 * correct / max(1, total_samples)
        total_used = sum(strategy_counts.values())

        print(f"\n[SAMPLER] Batch usage this epoch (total={total_used}):")
        for n, c in sorted(strategy_counts.items(), key=lambda x: -x[1]):
            pct = 100 * c / max(1, total_used)
            print(f"  {n:<20} {c:>5}  ({pct:.1f}%)")

        print("[VAL] Running validation...")
        val_loss, val_acc = run_validation(
            model, val_lossy, val_lossless, unified_gen, criterion, epoch)
        print(f"[VAL] Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%  "
              f"(base: {base_val_acc:.2f}%  delta: {val_acc - base_val_acc:+.2f}%)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch + 1
            save_checkpoint(epoch + 1, model, optimizer,
                            {}, val_acc, 'srnet_finetuned_best.pth')
            print(f"[VAL] *** New best: {best_val_acc:.2f}% ***")

        print(f"[EPOCH] Train: {train_acc:.2f}%  |  Val: {val_acc:.2f}%  |  "
              f"LR: {lr:.2e}")

        history['epoch'].append(epoch + 1)
        history['train_acc'].append(round(train_acc, 2))
        history['val_acc'].append(round(val_acc, 2))
        history['lr'].append(lr)
        history['strategy_counts'].append(strategy_counts)

        if (epoch + 1) % 5 == 0:
            save_checkpoint(epoch + 1, model, optimizer, {},
                            val_acc, f'srnet_ft_epoch_{epoch + 1}.pth')

    print(f"\n[DONE] Best val accuracy: {best_val_acc:.2f}% at epoch {best_val_epoch}")
    print("[DONE] Best model saved as: srnet_finetuned_best.pth")

    with open('finetune_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print("[DONE] History saved to finetune_history.json")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Fine-tune SRNet on weak-spot strategies")
    p.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT,
                   help=f'Checkpoint to resume from (default: {DEFAULT_CHECKPOINT})')
    p.add_argument('--epochs', type=int, default=FT_EPOCHS,
                   help=f'Number of fine-tune epochs (default: {FT_EPOCHS})')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    run_finetune(checkpoint_path=args.checkpoint, epochs=args.epochs)

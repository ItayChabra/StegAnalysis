"""
train_hybrid.py — Top-level training orchestration.

This file contains only the run_training() function and __main__ entry point.
All hyperparameters live in config.py; all logic lives in the specialist modules.

Run with:
    python training/train_hybrid.py
    python main.py
"""

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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from generators.unified_generator import UnifiedGenerator
from models.srnet import SRNet

from training.batch    import build_assigned_pairs, make_fixed_batch
from training.config   import (
    BATCH_SIZE,
    CURRICULUM_END,
    DEVICE,
    EPOCHS,
    FIXED_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    INITIAL_LR,
    MIN_CAPACITY,
    NUM_WORKERS,
)
from training.dataset  import create_or_load_split, load_balanced_dataset
from training.evolution import EvolutionaryManager
from training.genome   import compute_log_fft, get_niche
from training.utils    import (
    adjust_learning_rate,
    generate_long_text_message,
    get_curriculum_blend_factor,
    save_checkpoint,
)
from training.validate import run_validation

# PyTorch performance flags
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
cudnn.benchmark                       = True


def run_training(checkpoint_path='srnet_best_val.pth'):
    print(f"Starting Hybrid Training Run 13 on {DEVICE}")
    print("Run 13 changes vs run 12:")
    print("  Architecture: Triple-branch frontend (SRM + spatial learnable + FFT learnable)")
    print("  Branch A: 11 frozen SRM filters (spatial only)")
    print("  Branch B: 53 learnable filters (spatial only — LSB/DCT specialization)")
    print("  Branch C: 21 learnable filters (FFT only — frequency ring specialization)")
    print("  Label smoothing: 0.1 (threshold calibration fix)")
    print(f"  CURRICULUM_END: {CURRICULUM_END}  "
          f"CURRICULUM_BLEND_EPOCHS: {get_curriculum_blend_factor.__module__}")

    # ── Data ──────────────────────────────────────────────────────────────────
    lossy_files, lossless_files = load_balanced_dataset('data/raw')
    split = create_or_load_split(lossy_files, lossless_files)

    train_lossy    = split['lossy_train']
    train_lossless = split['lossless_train']
    val_lossy      = split['lossy_val']
    val_lossless   = split['lossless_val']

    print(f"\n[DATA] Train: {len(train_lossy)} lossy + {len(train_lossless)} lossless")
    print(f"[DATA] Val:   {len(val_lossy)} lossy + {len(val_lossless)} lossless")

    # ── Model ─────────────────────────────────────────────────────────────────
    discriminator = SRNet().to(DEVICE)
    print("[INFO] Compiling model with torch.compile (reduce-overhead)...")
    discriminator = torch.compile(discriminator, mode='reduce-overhead')

    optimizer = optim.Adam(discriminator.parameters(), lr=INITIAL_LR, weight_decay=2e-4)
    criterion = nn.CrossEntropyLoss(reduction='none', label_smoothing=0.1)
    scaler = torch.amp.GradScaler('cuda')

    # ── Checkpoint resume ─────────────────────────────────────────────────────
    best_val_acc   = 0.0
    best_val_epoch = 0

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[RESUME] Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt)
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        raw_model = discriminator._orig_mod if hasattr(discriminator, '_orig_mod') else discriminator
        raw_model.load_state_dict(state_dict)

        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        best_val_acc   = ckpt.get('val_acc', 0.0)
        best_val_epoch = ckpt.get('epoch',   0)
        print(f"[RESUME] Restored val_acc={best_val_acc:.2f}%  epoch={best_val_epoch}")
    else:
        print("[RESUME] No checkpoint found — training from scratch.")

    # ── Managers ──────────────────────────────────────────────────────────────
    unified_gen = UnifiedGenerator()
    evo_manager = EvolutionaryManager()

    # ── History ───────────────────────────────────────────────────────────────
    training_history = {
        'epochs': [], 'loss': [], 'model_acc': [], 'val_loss': [], 'val_acc': [],
        'gen_success': [], 'learning_rate': [],
        'fallback_rate': [], 'blend_factor': [], 'pad_rate': [],
    }

    min_dataset_size = min(len(train_lossy), len(train_lossless))
    steps_per_epoch  = max(1, min_dataset_size // (BATCH_SIZE // 2))

    # ── Epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(EPOCHS):
        current_lr   = adjust_learning_rate(optimizer, epoch)
        blend_factor = get_curriculum_blend_factor(epoch)

        curriculum_active = blend_factor < 1.0
        track_evolution   = blend_factor > 0.0

        # Curriculum parameter annealing
        hard_min_capacity  = max(MIN_CAPACITY, 1.0 - (min(epoch, CURRICULUM_END - 1) * 0.08))
        min_capacity       = hard_min_capacity + blend_factor * (MIN_CAPACITY - hard_min_capacity)
        hard_max_edge      = min(70, epoch * 7)
        max_edge_threshold = int(hard_max_edge + blend_factor * (100 - hard_max_edge))

        _print_epoch_header(epoch, current_lr, blend_factor, curriculum_active,
                            min_capacity, max_edge_threshold)

        random.shuffle(train_lossy)
        random.shuffle(train_lossless)

        total_loss      = 0.0
        correct_total   = 0
        total_samples   = 0
        epoch_fallbacks = 0
        epoch_pads      = 0

        discriminator.train()
        optimizer.zero_grad()

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            for step in range(steps_per_epoch):
                half_batch  = BATCH_SIZE // 2
                batch_files = (
                    train_lossy[step * half_batch: (step + 1) * half_batch] +
                    train_lossless[step * half_batch: (step + 1) * half_batch]
                )
                random.shuffle(batch_files)

                assigned_pairs, fallback_count = build_assigned_pairs(
                    batch_files, evo_manager)
                epoch_fallbacks += fallback_count

                # ── Per-pair generation closure ────────────────────────────
                def generate_pair(args, _epoch=epoch,
                                  _curriculum=curriculum_active,
                                  _min_cap=min_capacity,
                                  _max_edge=max_edge_threshold):
                    path, genome = args
                    try:
                        with Image.open(path) as img:
                            cover_img = img.convert('L')

                        w, h = cover_img.size
                        if w < 256 or h < 256:
                            return None

                        i_c, j_c, h_c, w_c = transforms.RandomCrop.get_params(
                            cover_img, output_size=(256, 256))
                        cover_crop = TF.crop(cover_img, i_c, j_c, h_c, w_c)

                        genome_cfg = genome.copy()
                        gt         = genome_cfg['gen_type']

                        if _curriculum:
                            genome_cfg['capacity_ratio'] = random.uniform(_min_cap, 1.0)
                            if gt == 'lsb':
                                genome_cfg['edge_threshold'] = random.randint(0, _max_edge)

                        if gt == 'lsb':
                            genome_cfg['message'] = (
                                generate_long_text_message(5000)
                                if _epoch >= 5 and random.random() < 0.5 else None
                            )
                            genome_cfg.setdefault('capacity_ratio', 0.5)

                        stego_arr, _ = unified_gen.generate_stego(
                            cover_crop, None, genome_cfg)
                        if stego_arr is None:
                            return None

                        spatial_cover = TF.to_tensor(cover_crop)
                        cover_t = torch.cat([spatial_cover,
                                             compute_log_fft(spatial_cover)], dim=0)

                        spatial_stego = TF.to_tensor(Image.fromarray(stego_arr))
                        stego_t = torch.cat([spatial_stego,
                                             compute_log_fft(spatial_stego)], dim=0)

                        return cover_t, stego_t, genome['name']

                    except Exception as e:
                        print(f"\n[GEN ERROR] {genome.get('name', 'Unknown')}: {e}")
                        return None

                # ── Collect batch ──────────────────────────────────────────
                inputs, labels, batch_genome_names = [], [], []

                for res in executor.map(generate_pair, assigned_pairs):
                    if res is None:
                        continue
                    cover_t, stego_t, g_name = res
                    inputs.extend([cover_t, stego_t])
                    labels.extend([0, 1])
                    batch_genome_names.extend([None, g_name])

                inputs_t, labels_t, weights_t, batch_genome_names = make_fixed_batch(
                    inputs, labels, batch_genome_names)

                if inputs_t is None:
                    print(f"\r[WARN] Step {step}: all generations failed — skipped.", end="")
                    continue

                n_real      = int(weights_t.sum().item())
                epoch_pads += FIXED_BATCH_SIZE - n_real

                inputs_t  = inputs_t.to(DEVICE, non_blocking=True)
                labels_t  = labels_t.to(DEVICE, non_blocking=True)
                weights_t = weights_t.to(DEVICE, non_blocking=True)

                # ── Diagnostic on step 0 of epoch 0 ───────────────────────
                if epoch == 0 and step == 0:
                    _run_diagnostic(inputs_t, n_real)

                # ── Forward / backward ────────────────────────────────────
                perm             = torch.randperm(inputs_t.size(0))
                inputs_shuffled  = inputs_t[perm]
                labels_shuffled  = labels_t[perm]
                weights_shuffled = weights_t[perm]
                names_shuffled   = [batch_genome_names[i] for i in perm.tolist()]

                with torch.amp.autocast('cuda'):
                    outputs   = discriminator(inputs_shuffled)
                    per_loss  = criterion(outputs, labels_shuffled)
                    loss      = (per_loss * weights_shuffled).sum() / weights_shuffled.sum()
                    loss_accum = loss / GRADIENT_ACCUMULATION_STEPS

                scaler.scale(loss_accum).backward()

                if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                # ── Evolutionary stats & accuracy ─────────────────────────
                _, preds = torch.max(outputs, 1)

                if track_evolution:
                    rel_names, fooled_results = [], []
                    for j, name in enumerate(names_shuffled):
                        if name is not None and weights_shuffled[j].item() > 0:
                            rel_names.append(name)
                            fooled_results.append(preds[j].item() == 0)
                    evo_manager.update_batch_stats(rel_names, fooled_results)

                real_mask      = weights_shuffled.bool()
                total_loss    += loss.item()
                correct_total += (preds[real_mask] == labels_shuffled[real_mask]).sum().item()
                total_samples += real_mask.sum().item()

                if step % 10 == 0:
                    print(f"\rStep {step}/{steps_per_epoch} | "
                          f"Loss: {loss.item():.4f} | "
                          f"Acc: {100 * correct_total / max(1, total_samples):.1f}%", end="")

        # Final gradient flush
        if steps_per_epoch % GRADIENT_ACCUMULATION_STEPS != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        torch.cuda.empty_cache()

        # ── End-of-epoch reporting ─────────────────────────────────────────
        fallback_rate_pct = 100.0 * epoch_fallbacks / max(1, steps_per_epoch * BATCH_SIZE)
        pad_rate_pct      = 100.0 * epoch_pads / max(1, steps_per_epoch * FIXED_BATCH_SIZE)

        if fallback_rate_pct > 3.0:
            print(f"\n[WARN] Niche cap fallback: {fallback_rate_pct:.2f}% — "
                  "consider relaxing NICHE_BATCH_CAP or FFT_COMBINED_BATCH_CAP")
        else:
            print(f"\n[DIVERSITY] Fallback: {fallback_rate_pct:.2f}% OK  |  "
                  f"Pad rate: {pad_rate_pct:.2f}%")

        print("[VAL] Running validation...")
        val_loss, val_acc = run_validation(
            discriminator, val_lossy, val_lossless, unified_gen, criterion, epoch)
        print(f"[VAL] Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            best_val_epoch = epoch + 1
            save_checkpoint(epoch + 1, discriminator, optimizer,
                            evo_manager.population[0], val_acc, 'srnet_best_val.pth')
            print(f"[VAL] *** New best: {best_val_acc:.2f}% at epoch {best_val_epoch} ***")

        if total_samples > 0:
            avg_loss  = total_loss / steps_per_epoch
            acc_total = 100 * correct_total / total_samples

            if not track_evolution:
                print(f"[EPOCH] Loss: {avg_loss:.4f} | Train: {acc_total:.2f}% | "
                      f"Val: {val_acc:.2f}% | Curriculum")
                avg_gen_score = 0.0
            else:
                all_rates = [d['fooled'] / d['attempts']
                             for d in evo_manager.stats.values() if d['attempts'] > 0]
                avg_gen_score = sum(all_rates) / len(all_rates) if all_rates else 0.0
                blend_label   = (f"blend={blend_factor*100:.0f}%"
                                 if curriculum_active else "full-evo")
                print(f"[EPOCH] Loss: {avg_loss:.4f} | Train: {acc_total:.2f}% | "
                      f"Val: {val_acc:.2f}% | GenFool: {avg_gen_score*100:.2f}% "
                      f"[{blend_label}]")

            training_history['epochs'].append(epoch + 1)
            training_history['loss'].append(avg_loss)
            training_history['model_acc'].append(acc_total)
            training_history['val_loss'].append(val_loss)
            training_history['val_acc'].append(val_acc)
            training_history['gen_success'].append(avg_gen_score * 100)
            training_history['learning_rate'].append(current_lr)
            training_history['fallback_rate'].append(round(fallback_rate_pct, 3))
            training_history['blend_factor'].append(round(blend_factor, 3))
            training_history['pad_rate'].append(round(pad_rate_pct, 3))

        # ── Evolution step ─────────────────────────────────────────────────
        if track_evolution:
            best_genome = evo_manager.evolve()
        else:
            evo_manager.generation += 1
            evo_manager.stats = {g['name']: {'fooled': 0, 'attempts': 0}
                                 for g in evo_manager.population}
            best_genome = evo_manager.population[0]

        if (epoch + 1) % 5 == 0:
            save_checkpoint(epoch + 1, discriminator, optimizer, best_genome, val_acc,
                            f"srnet_epoch_{epoch + 1}.pth")

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n[INFO] Best val accuracy: {best_val_acc:.2f}% at epoch {best_val_epoch}")
    print("[INFO] Best model saved as: srnet_best_val.pth")
    print("[INFO] Use srnet_best_val.pth (not a periodic checkpoint) as input to evaluate.py")

    with open('training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    print("[INFO] Training Complete.")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _print_epoch_header(epoch, lr, blend_factor, curriculum_active,
                        min_capacity, max_edge_threshold):
    print(f"\n{'=' * 65}")
    print(f"Epoch {epoch + 1}/{EPOCHS} | LR: {lr:.6f} | Blend: {blend_factor:.2f}")
    if curriculum_active:
        blend_note = (f" (blending {blend_factor*100:.0f}%)" if blend_factor > 0 else "")
        print(f"Curriculum: Cap [{min_capacity:.2f}-1.0] | "
              f"Edge [0-{max_edge_threshold}]{blend_note}")
    else:
        print("Full evolution — no curriculum constraints.")
    print('=' * 65)


def _run_diagnostic(inputs_t, n_real):
    """Quick sanity-check printed once at the very start of training."""
    print("\n" + "=" * 65)
    print("DIAGNOSTIC CHECK (Run 13: Triple-Branch)")
    print("=" * 65)
    real_data = inputs_t[:n_real]
    covers    = real_data[0::2, 0:1, :, :].cpu().numpy()
    stegos    = real_data[1::2, 0:1, :, :].cpu().numpy()
    diff      = np.abs(covers - stegos)
    mod_rate  = 100 * (diff > 0).sum() / diff.size
    print(f"  Max Pixel Diff:    {diff.max():.6f}")
    print(f"  Modification Rate: {mod_rate:.2f}%")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_training()
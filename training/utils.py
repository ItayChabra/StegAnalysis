"""
utils.py — Miscellaneous utilities: LR schedule, checkpointing, text generation.
"""

import math
import random
import string

import torch

from training.config import (
    CURRICULUM_BLEND_EPOCHS,
    CURRICULUM_END,
    EPOCHS,
    INITIAL_LR,
    MAX_LR,
    MIN_LR,
)


# ── Learning rate schedule ────────────────────────────────────────────────────

def adjust_learning_rate(optimizer: torch.optim.Optimizer, epoch: int) -> float:
    """
    Three-phase LR schedule:
        0-1   — linear warm-up from INITIAL_LR to MAX_LR
        2-11  — hold at MAX_LR (hardest evolutionary phase)
        12+   — cosine decay from MAX_LR down to MIN_LR
    """
    if epoch < 2:
        lr = INITIAL_LR + (MAX_LR - INITIAL_LR) * (epoch / 2)
    elif epoch < 12:
        lr = MAX_LR
    else:
        progress = (epoch - 12) / (EPOCHS - 12)
        lr = MIN_LR + 0.5 * (MAX_LR - MIN_LR) * (1 + math.cos(math.pi * progress))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def get_curriculum_blend_factor(epoch: int) -> float:
    """
    Returns a value in [0, 1] that drives the curriculum → full-evolution
    transition over CURRICULUM_BLEND_EPOCHS epochs ending at CURRICULUM_END.
    """
    blend_start = CURRICULUM_END - CURRICULUM_BLEND_EPOCHS
    if epoch < blend_start:
        return 0.0
    if epoch >= CURRICULUM_END:
        return 1.0
    return (epoch - blend_start) / CURRICULUM_BLEND_EPOCHS


# ── Checkpointing ─────────────────────────────────────────────────────────────

def save_checkpoint(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    best_genome: dict,
    val_acc: float,
    filename: str = "checkpoint.pth",
) -> None:
    """Save model weights, optimizer state, and metadata to *filename*."""
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    torch.save({
        'epoch':                epoch,
        'model_state_dict':     raw_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_genome':          best_genome,
        'val_acc':              val_acc,
    }, filename)
    print(f"[CHECKPOINT] Saved to {filename}  (val_acc={val_acc:.2f}%)")


# ── Data helpers ──────────────────────────────────────────────────────────────

def generate_long_text_message(length: int = 5000) -> str:
    """Generate a random ASCII string of *length* characters for LSB embedding."""
    chars = string.ascii_letters + string.digits + " " + ".,!?"
    return ''.join(random.choices(chars, k=length))
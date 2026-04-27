"""
dataset.py — Image discovery and reproducible train / val / test splitting.
"""

import glob
import json
import os
import random

from training.config import (
    BATCH_SIZE,
    SPLIT_FILE,
    SPLIT_SEED,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
)


def load_balanced_dataset(raw_dir: str):
    """
    Scan *raw_dir* for lossy (Flickr30k JPEGs) and lossless (BOSSbase PGMs/PNGs).

    Returns:
        (lossy_files, lossless_files) — lists of absolute file paths.
    """
    lossy_dir    = os.path.join(raw_dir, 'flickr30k')
    lossless_dir = os.path.join(raw_dir, 'BossBase and BOWS2')

    print(f"[DATA] Scanning {raw_dir}...")
    lossy_files = (
        glob.glob(os.path.join(lossy_dir, '*.jpg')) +
        glob.glob(os.path.join(lossy_dir, '*.jpeg'))
    )
    lossless_files = (
        glob.glob(os.path.join(lossless_dir, '*.pgm')) +
        glob.glob(os.path.join(lossless_dir, '*.png'))
    )

    print(f"[DATA] Found {len(lossy_files)} Lossy (Flickr) images.")
    print(f"[DATA] Found {len(lossless_files)} Lossless (BOSSbase) images.")

    if len(lossy_files) < BATCH_SIZE or len(lossless_files) < BATCH_SIZE:
        print("[WARN] Imbalance or missing files — check your data directories.")

    return lossy_files, lossless_files


def create_or_load_split(lossy_files: list, lossless_files: list) -> dict:
    """
    Load an existing dataset split from *SPLIT_FILE* or create a new one.

    The split is stratified: lossy and lossless files are shuffled and divided
    independently so each subset maintains the same source balance.

    Returns a dict with keys:
        lossy_train, lossy_val, lossy_test,
        lossless_train, lossless_val, lossless_test
    """
    if os.path.exists(SPLIT_FILE):
        print(f"[SPLIT] Loading existing split from {SPLIT_FILE}")
        with open(SPLIT_FILE, 'r') as f:
            split = json.load(f)
        _print_split_sizes(split)
        return split

    print(
        f"[SPLIT] Creating new {TRAIN_RATIO:.0%}/{VAL_RATIO:.0%}/{TEST_RATIO:.0%} "
        f"split (seed={SPLIT_SEED})"
    )
    rng = random.Random(SPLIT_SEED)

    def _split_list(files):
        files    = [os.path.abspath(f) for f in files]
        shuffled = files.copy()
        rng.shuffle(shuffled)
        n_train = int(len(shuffled) * TRAIN_RATIO)
        n_val   = int(len(shuffled) * VAL_RATIO)
        return (
            shuffled[:n_train],
            shuffled[n_train: n_train + n_val],
            shuffled[n_train + n_val:],
        )

    lossy_train,    lossy_val,    lossy_test    = _split_list(lossy_files)
    lossless_train, lossless_val, lossless_test = _split_list(lossless_files)

    split = {
        'split_seed':     SPLIT_SEED,
        'train_ratio':    TRAIN_RATIO,
        'val_ratio':      VAL_RATIO,
        'test_ratio':     TEST_RATIO,
        'lossy_train':    lossy_train,
        'lossy_val':      lossy_val,
        'lossy_test':     lossy_test,
        'lossless_train': lossless_train,
        'lossless_val':   lossless_val,
        'lossless_test':  lossless_test,
    }

    with open(SPLIT_FILE, 'w') as f:
        json.dump(split, f, indent=2)

    print(f"[SPLIT] Saved to {SPLIT_FILE}")
    _print_split_sizes(split)
    return split


# ── Internal ──────────────────────────────────────────────────────────────────

def _print_split_sizes(split: dict) -> None:
    print(
        f"[SPLIT]   Lossy:    {len(split['lossy_train'])} train | "
        f"{len(split['lossy_val'])} val | {len(split['lossy_test'])} test"
    )
    print(
        f"[SPLIT]   Lossless: {len(split['lossless_train'])} train | "
        f"{len(split['lossless_val'])} val | {len(split['lossless_test'])} test"
    )
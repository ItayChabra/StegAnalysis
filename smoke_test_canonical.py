"""
smoke_test_canonical.py — Quick sanity check for the canonical S-UNIWARD implementation.

Scores three groups against srnet_finetuned_best.pth:
  A) BOSSbase_256 covers  (expect: low scores, ~0.08 median)
  B) Canonical-embedded   (canonical=True @ 0.2 bpp from BOSSbase_256 covers)
  C) Actual SUNI_02       (held-out canonical S-UNIWARD @ 0.2 bpp)

Groups B and C should look similar — both are canonical SUNI @ 0.2 bpp on BOSSbase.
If B ≈ C ≈ A (all near-cover), the model has no signal for canonical SUNI yet,
confirming the training gap that Run 21 is designed to fix.
If B >> A, the generator is introducing detectable artifacts (would be a bug).
"""

import glob
import math
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(__file__))
from models.srnet import SRNet
from generators.unified_generator import UnifiedGenerator

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT   = 'srnet_finetuned_best.pth'
N_IMAGES     = 50
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_LOG_FFT_SCALE = math.log1p(256 * 256)

BOSS_DIR  = 'data/Kaggle_Testing/New_S-UNIWARD/BOSSbase_256'
SUNI02_DIR = 'data/Kaggle_Testing/New_S-UNIWARD/SUNI_02'

EMBED_CONFIG = {
    'gen_type':       'adaptive',
    'adaptive_mode':  'suniward',
    'capacity_ratio': 0.20,
    'sigma_offset':   1.0,
    'cost_exponent':  1.0,
    'use_diagonal':   True,
    'canonical':      True,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_log_fft(t: torch.Tensor) -> torch.Tensor:
    return torch.log1p(torch.abs(torch.fft.fftshift(
        torch.fft.fft2(t), dim=(-2, -1)))) / _LOG_FFT_SCALE


def score_image(model, img: Image.Image, to_tensor) -> float:
    t = to_tensor(img).to(DEVICE)
    inp = torch.cat([t, compute_log_fft(t)], dim=0).unsqueeze(0)
    with torch.no_grad():
        return torch.softmax(model(inp), dim=1)[0, 1].item()


def load_and_crop(path: str) -> Image.Image | None:
    img = Image.open(path).convert('L')
    w, h = img.size
    if w < 256 or h < 256:
        return None
    left, top = (w - 256) // 2, (h - 256) // 2
    return img.crop((left, top, left + 256, top + 256))


def stats(arr: np.ndarray) -> str:
    return (f"n={len(arr):>3}  min {arr.min():.3f}  "
            f"median {np.median(arr):.3f}  mean {arr.mean():.3f}  max {arr.max():.3f}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT}\n")

    model = SRNet().to(DEVICE)
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
    sd = {k.replace('_orig_mod.', ''): v for k, v in ckpt.get('model_state_dict', ckpt).items()}
    model.load_state_dict(sd)
    model.eval()
    print(f"Loaded  (val_acc at save: {ckpt.get('val_acc', '?')})\n")

    gen       = UnifiedGenerator()
    to_tensor = transforms.ToTensor()

    boss_files  = sorted(glob.glob(os.path.join(BOSS_DIR,  '*.*')))[:N_IMAGES]
    suni02_files = sorted(glob.glob(os.path.join(SUNI02_DIR, '*.*')))[:N_IMAGES]

    if not boss_files:
        print(f"ERROR: no files in {BOSS_DIR}"); return
    if not suni02_files:
        print(f"ERROR: no files in {SUNI02_DIR}"); return

    cover_scores, canonical_scores, suni02_scores = [], [], []
    embed_failures = 0

    print(f"Scoring {N_IMAGES} images per group...")
    for path in boss_files:
        crop = load_and_crop(path)
        if crop is None:
            continue

        # A) Cover score
        cover_scores.append(score_image(model, crop, to_tensor))

        # B) Canonical-embed then score
        stego_arr, _ = gen.generate_stego(crop, None, EMBED_CONFIG)
        if stego_arr is None:
            embed_failures += 1
            cover_scores.pop()
            continue
        stego_img = Image.fromarray(stego_arr)
        canonical_scores.append(score_image(model, stego_img, to_tensor))

    for path in suni02_files:
        crop = load_and_crop(path)
        if crop is None:
            continue
        suni02_scores.append(score_image(model, crop, to_tensor))

    if embed_failures:
        print(f"  ({embed_failures} embed failures skipped)\n")

    cover_arr    = np.array(cover_scores)
    canonical_arr = np.array(canonical_scores)
    suni02_arr   = np.array(suni02_scores)

    print("=" * 70)
    print("SMOKE TEST RESULTS  (P(stego) scores, single-patch inference)")
    print("=" * 70)
    print(f"  A) BOSSbase_256 covers  : {stats(cover_arr)}")
    print(f"  B) Canonical-embedded   : {stats(canonical_arr)}  [canonical=True @ 0.2 bpp]")
    print(f"  C) Actual SUNI_02       : {stats(suni02_arr)}  [held-out canonical SUNI @ 0.2 bpp]")
    print()

    bc_overlap = np.mean(np.abs(canonical_arr[:len(suni02_arr)] - suni02_arr[:len(canonical_arr)]))
    ac_gap     = canonical_arr.mean() - cover_arr.mean()
    print(f"  B−A mean gap  (canonical lift over cover)  : {ac_gap:+.4f}")
    print(f"  |B−C| mean    (generator vs held-out SUNI) : {bc_overlap:.4f}")
    print()
    print("INTERPRETATION")
    if canonical_arr.mean() < cover_arr.mean() + 0.02:
        print("  Model sees canonical stego as cover-level — training gap confirmed.")
        print("  Run 21 with canonical=True should teach the model to detect this.")
    elif canonical_arr.mean() > cover_arr.mean() + 0.10:
        print("  WARNING: canonical stego scores much higher than covers.")
        print("  Either the generator is adding detectable artifacts, or the model")
        print("  has incidentally learned some canonical SUNI signal already.")
    else:
        print("  Modest lift — borderline. Check the distributions manually.")
    print("=" * 70)


if __name__ == '__main__':
    main()
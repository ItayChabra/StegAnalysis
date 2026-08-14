"""
verify_readme_numbers.py — re-measure the two figures the README reports that
test_kaggle.py's sweep table cannot supply directly.

Why this exists:
  1. SteganoGAN is in the 'gan' group, which test_kaggle.py deliberately EXCLUDES
     from the threshold sweep. The logs therefore give SteganoGAN as a score
     distribution only, with no detection rate — so the README had to mix units
     (percentages for LSB/DCT/FFT, raw medians for SteganoGAN). This computes a
     real TPR at the operating threshold.

  2. The adaptive (S-UNIWARD) numbers are sampled from a *prefix*:
     test_kaggle.py takes sorted(glob)[:n_images], and the first 200 of the
     10,000 BOSSbase-derived files are systematically lower-scoring than the
     folder as a whole. This re-scores a much larger sample, and compares
     S-UNIWARD against its MATCHED cover (BOSSbase_256) rather than against the
     unrelated BOSS&BOWS2 set.

Read-only: loads a checkpoint and scores images. Trains nothing, writes no weights.

Usage:
    python scripts/verify_readme_numbers.py --checkpoint srnet_steganogan_best.pth
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.srnet import SRNet                                    # noqa: E402
from test_kaggle import (KAGGLE_DIR, aggregate_scores,            # noqa: E402
                         load_luminance, sliding_window_scores)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Matched-pair groups: each stego folder is compared against the cover folder it
# was actually derived from, not against an unrelated cover dataset.
ADAPTIVE_SET = {
    'cover': ('BOSSbase_256', os.path.join(KAGGLE_DIR, 'New_S-UNIWARD', 'BOSSbase_256')),
    'stego': [('S-UNIWARD 0.2', os.path.join(KAGGLE_DIR, 'New_S-UNIWARD', 'SUNI_02')),
              ('S-UNIWARD 0.4', os.path.join(KAGGLE_DIR, 'New_S-UNIWARD', 'SUNI_04'))],
}

GAN_SET = {
    'cover': ('SGAN cover', os.path.join(KAGGLE_DIR, 'Steganogan', 'cover')),
    'stego': [('SGAN dense',    os.path.join(KAGGLE_DIR, 'Steganogan', 'dense')),
              ('SGAN basic',    os.path.join(KAGGLE_DIR, 'Steganogan', 'basic')),
              ('SGAN residual', os.path.join(KAGGLE_DIR, 'Steganogan', 'residual'))],
}

THRESHOLDS = [0.30, 0.50, 0.65, 0.80]


def load_model(path):
    model = SRNet().to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
    sd = ckpt.get('model_state_dict', ckpt)
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    print(f"Checkpoint: {path}  (saved val_acc: {ckpt.get('val_acc', '?')})")
    return model


def score_folder(model, to_tensor, folder, n, mode='max'):
    files = sorted(glob.glob(os.path.join(folder, '*.*')))[:n]
    if not files:
        print(f"  !! no images in {folder}")
        return np.array([])
    out = []
    for i, p in enumerate(files):
        out.append(aggregate_scores(sliding_window_scores(model, load_luminance(p), to_tensor), mode))
        if (i + 1) % 250 == 0:
            print(f"    ...{i + 1}/{len(files)}", flush=True)
    return np.array(out)


def report(title, group, model, to_tensor, n):
    print(f"\n{'=' * 78}\n{title}  (n={n}/folder, mode=max)\n{'=' * 78}")
    cname, cpath = group['cover']
    cov = score_folder(model, to_tensor, cpath, n)
    print(f"\n  MATCHED COVER  {cname:<16} n={len(cov):<5} "
          f"median {np.median(cov):.3f}  mean {cov.mean():.3f}")

    stegos = {}
    for sname, spath in group['stego']:
        s = score_folder(model, to_tensor, spath, n)
        stegos[sname] = s
        sep = np.median(s) - np.median(cov)
        print(f"  STEGO          {sname:<16} n={len(s):<5} "
              f"median {np.median(s):.3f}  mean {s.mean():.3f}   "
              f"Δmedian vs matched cover {sep:+.3f}")

    print(f"\n  {'threshold':>10} | {'TNR':>7} | " +
          " | ".join(f"{k:>14}" for k in stegos))
    print("  " + "-" * (22 + 17 * len(stegos)))
    for t in THRESHOLDS:
        tnr = (cov < t).mean() * 100
        tprs = " | ".join(f"{(s >= t).mean() * 100:>13.1f}%" for s in stegos.values())
        print(f"  {t:>10.2f} | {tnr:>6.1f}% | {tprs}")
    return cov, stegos


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='srnet_steganogan_best.pth')
    ap.add_argument('--gan-images', type=int, default=200)
    ap.add_argument('--adaptive-images', type=int, default=2000)
    a = ap.parse_args()

    m = load_model(a.checkpoint)
    tt = transforms.ToTensor()
    report("SteganoGAN — detection rate vs matched cover", GAN_SET, m, tt, a.gan_images)
    report("S-UNIWARD — larger sample vs MATCHED cover", ADAPTIVE_SET, m, tt, a.adaptive_images)
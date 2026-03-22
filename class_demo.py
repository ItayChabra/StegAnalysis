import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import os

from models.srnet import SRNet
from generators.unified_generator import UnifiedGenerator

# ==================== CONFIGURATION ====================
MODEL_PATH        = "srnet_epoch_30.pth"
DEVICE            = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLEAN_IMAGE_PATH  = r"E:\PycharmProjects\data\raw\BossBase and BOWS2\2.pgm"

# Sliding window settings
WINDOW_SIZE       = 256     # must match the training crop size
WINDOW_STRIDE     = 64      # 75% overlap — fine-grained coverage
DETECTION_THRESH  = 0.50    # probability above which a window is "flagged"

SECRET_MESSAGE    = "This is a secret message " * 4096

# ==================== STRATEGY CONFIGS ====================
# Parameters derived from the evolutionary training run:
#   - edge / skip dominated the later generations.
#   - Representative configs are taken from the best-evolved genomes in
#     the training log so the test reflects real adversarial conditions.
STRATEGY_CONFIGS = {
    'sequential': {
        'gen_type':       'lsb',
        'strategy':       'sequential',
        'capacity_ratio': 0.50,
        'edge_threshold': 0,
        'bit_depth':      1,
        'step':           1,
        'message':        SECRET_MESSAGE,
    },
    'random': {
        'gen_type':       'lsb',
        'strategy':       'random',
        'capacity_ratio': 0.50,
        'edge_threshold': 0,
        'bit_depth':      1,
        'step':           1,
        'message':        SECRET_MESSAGE,
    },
    'skip': {
        # Best evolved skip genome: Cap 0.56, Edge threshold 95-96
        'gen_type':       'lsb',
        'strategy':       'skip',
        'capacity_ratio': 0.56,
        'edge_threshold': 95,
        'bit_depth':      1,
        'step':           3,
        'message':        SECRET_MESSAGE,
    },
    'edge': {
        # Best evolved edge genome: Cap 0.21, Edge threshold 9
        'gen_type':       'lsb',
        'strategy':       'edge',
        'capacity_ratio': 0.21,
        'edge_threshold': 9,
        'bit_depth':      1,
        'step':           1,
        'message':        SECRET_MESSAGE,
    },
}


# ==================== SLIDING WINDOW DETECTION ====================
def sliding_window_detect(ai_model, image: Image.Image) -> dict:
    """
    Slide a WINDOW_SIZE x WINDOW_SIZE window across `image` with WINDOW_STRIDE
    and return per-window stego probability scores.

    Returns a dict with:
        scores      – list of all per-window probabilities
        max_score   – highest single-window probability
        mean_score  – average probability across all windows
        flagged     – number of windows that exceed DETECTION_THRESH
        total       – total number of windows evaluated
        verdict     – human-readable verdict string
    """
    to_tensor = transforms.ToTensor()
    img_array = np.array(image.convert('L'), dtype=np.uint8)
    h, w      = img_array.shape

    # Pad so at least one full window fits even on small images.
    pad_h = max(0, WINDOW_SIZE - h)
    pad_w = max(0, WINDOW_SIZE - w)
    if pad_h > 0 or pad_w > 0:
        padded = np.full((h + pad_h, w + pad_w), 128, dtype=np.uint8)
        padded[:h, :w] = img_array
        img_array = padded
        h, w = img_array.shape

    scores = []
    ai_model.eval()

    with torch.no_grad():
        for top in range(0, h - WINDOW_SIZE + 1, WINDOW_STRIDE):
            for left in range(0, w - WINDOW_SIZE + 1, WINDOW_STRIDE):
                patch = img_array[top: top + WINDOW_SIZE, left: left + WINDOW_SIZE]
                patch_img    = Image.fromarray(patch)
                patch_tensor = to_tensor(patch_img).unsqueeze(0).to(DEVICE)

                output        = ai_model(patch_tensor)
                probabilities = torch.softmax(output, dim=1)
                stego_prob    = probabilities[0, 1].item()
                scores.append(stego_prob)

    if not scores:
        return {'scores': [], 'max_score': 0.0, 'mean_score': 0.0,
                'flagged': 0, 'total': 0, 'verdict': 'NO WINDOWS'}

    max_score  = max(scores)
    mean_score = sum(scores) / len(scores)
    flagged    = sum(1 for s in scores if s > DETECTION_THRESH)

    # Verdict uses max_score: a single highly suspicious region is enough.
    if max_score > 0.75:
        verdict = 'STEGO DETECTED   (High confidence)'
    elif max_score > DETECTION_THRESH:
        verdict = 'SUSPICIOUS       (Flagged for review)'
    else:
        verdict = 'CLEAN            (Detection failed)'

    return {
        'scores':     scores,
        'max_score':  max_score,
        'mean_score': mean_score,
        'flagged':    flagged,
        'total':      len(scores),
        'verdict':    verdict,
    }


# ==================== HELPERS ====================
def load_and_prepare_image(path: str) -> Image.Image:
    """Load a grayscale image and center-crop to the largest multiple of
    WINDOW_STRIDE that still fits, giving the slider clean coverage."""
    img  = Image.open(path).convert('L')
    w, h = img.size
    if w < WINDOW_SIZE or h < WINDOW_SIZE:
        # Upscale only if truly too small; keep aspect ratio intact.
        scale = max(WINDOW_SIZE / w, WINDOW_SIZE / h)
        img   = img.resize((int(w * scale) + 1, int(h * scale) + 1),
                           Image.Resampling.LANCZOS)
        w, h = img.size
    # Snap dimensions down to the nearest stride multiple for clean tiling.
    new_w = ((w - WINDOW_SIZE) // WINDOW_STRIDE) * WINDOW_STRIDE + WINDOW_SIZE
    new_h = ((h - WINDOW_SIZE) // WINDOW_STRIDE) * WINDOW_STRIDE + WINDOW_SIZE
    left  = (w - new_w) // 2
    top   = (h - new_h) // 2
    return img.crop((left, top, left + new_w, top + new_h))


def print_window_result(label: str, result: dict):
    flagged_pct = 100.0 * result['flagged'] / result['total'] if result['total'] > 0 else 0.0
    print(f"  Max window score:   {result['max_score']  * 100:>6.1f}%")
    print(f"  Mean window score:  {result['mean_score'] * 100:>6.1f}%")
    print(f"  Flagged windows:    {result['flagged']:>4d} / {result['total']} "
          f"({flagged_pct:.1f}%)")
    print(f"  Verdict:            {result['verdict']}")


# ==================== MAIN DEMO ====================
def run_strategy_evaluation():
    print("\n" + "=" * 70)
    print("     STRATEGY EVALUATION — SLIDING WINDOW STEGANALYSIS")
    print(f"     Window: {WINDOW_SIZE}×{WINDOW_SIZE}  |  Stride: {WINDOW_STRIDE}  |  "
          f"Flag threshold: {DETECTION_THRESH * 100:.0f}%")
    print("=" * 70)

    # --- SETUP ---
    print(f"\n[SETUP]  Loading model: {MODEL_PATH}  |  Device: {DEVICE}")
    ai_detector = SRNet().to(DEVICE)
    checkpoint  = torch.load(MODEL_PATH, map_location=DEVICE)
    ai_detector.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    ai_detector.eval()

    stego_gen = UnifiedGenerator()

    # Prepare cover image once; every strategy embeds into the same source.
    cover_img = load_and_prepare_image(CLEAN_IMAGE_PATH)
    w, h      = cover_img.size
    windows_h = (h - WINDOW_SIZE) // WINDOW_STRIDE + 1
    windows_w = (w - WINDOW_SIZE) // WINDOW_STRIDE + 1
    total_windows = windows_h * windows_w

    print(f"         Image:  {os.path.basename(CLEAN_IMAGE_PATH)}  "
          f"({w}×{h} after prep)")
    print(f"         Windows expected per image: "
          f"{windows_h} rows × {windows_w} cols = {total_windows}")
    print()

    # --- BASELINE: scan clean image ---
    print("=" * 70)
    print("[BASELINE]  Clean image (no embedding)")
    print("=" * 70)
    baseline = sliding_window_detect(ai_detector, cover_img)
    print_window_result("Clean", baseline)

    # --- PER-STRATEGY EVALUATION ---
    results_summary = {}

    for strategy_name, config in STRATEGY_CONFIGS.items():
        print()
        print("=" * 70)
        print(f"[STRATEGY: {strategy_name.upper()}]")
        print(f"  Capacity: {config['capacity_ratio']:.2f}  |  "
              f"Edge threshold: {config['edge_threshold']}  |  "
              f"Step: {config.get('step', 1)}")
        print("=" * 70)

        # Embed — pass PIL.Image directly (no temp file)
        stego_arr, psnr = stego_gen.generate_stego(cover_img, None, config)

        if stego_arr is None:
            print("  ERROR: embedding failed — skipping strategy.")
            continue

        stego_img = Image.fromarray(stego_arr.astype(np.uint8))

        pixels_total    = cover_img.size[0] * cover_img.size[1]
        pixels_modified = int(pixels_total * config['capacity_ratio'])
        print(f"  Embedding complete")
        print(f"  Pixels modified: {pixels_modified:,} / {pixels_total:,} "
              f"({config['capacity_ratio'] * 100:.0f}%)")
        print(f"  PSNR: {psnr:.2f} dB  (>40 dB = imperceptible)")

        # Save stego for visual inspection
        out_path = f"demo_stego_{strategy_name}.png"
        stego_img.save(out_path)
        print(f"  Saved: {out_path}")

        # Sliding window detection
        print(f"\n  Running sliding window detection...")
        result = sliding_window_detect(ai_detector, stego_img)
        print_window_result(strategy_name, result)

        # Delta vs baseline
        delta_max  = (result['max_score']  - baseline['max_score'])  * 100
        delta_mean = (result['mean_score'] - baseline['mean_score']) * 100
        print(f"\n  Δ vs baseline  →  Max: {delta_max:+.1f}%   Mean: {delta_mean:+.1f}%")

        results_summary[strategy_name] = {
            'psnr':       psnr,
            'max_score':  result['max_score'],
            'mean_score': result['mean_score'],
            'flagged':    result['flagged'],
            'total':      result['total'],
            'verdict':    result['verdict'],
            'delta_max':  delta_max,
            'delta_mean': delta_mean,
        }

    # --- FINAL COMPARISON TABLE ---
    print()
    print("=" * 70)
    print("                    FINAL COMPARISON TABLE")
    print("=" * 70)
    header = f"  {'Strategy':<12}  {'PSNR':>7}  {'Max%':>7}  {'Mean%':>7}  " \
             f"{'Flagged':>10}  {'Δ Max':>7}  Verdict"
    print(header)
    print("  " + "-" * 66)

    # Baseline row
    flagged_pct_bl = 100.0 * baseline['flagged'] / baseline['total'] if baseline['total'] > 0 else 0
    print(f"  {'clean':<12}  {'—':>7}  "
          f"{baseline['max_score'] * 100:>6.1f}%  "
          f"{baseline['mean_score'] * 100:>6.1f}%  "
          f"{baseline['flagged']:>4d}/{baseline['total']:<4d} ({flagged_pct_bl:4.1f}%)  "
          f"{'—':>7}  {baseline['verdict']}")

    for name, r in results_summary.items():
        flagged_pct = 100.0 * r['flagged'] / r['total'] if r['total'] > 0 else 0
        print(f"  {name:<12}  {r['psnr']:>6.1f}dB  "
              f"{r['max_score'] * 100:>6.1f}%  "
              f"{r['mean_score'] * 100:>6.1f}%  "
              f"{r['flagged']:>4d}/{r['total']:<4d} ({flagged_pct:4.1f}%)  "
              f"{r['delta_max']:>+6.1f}%  {r['verdict']}")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    run_strategy_evaluation()
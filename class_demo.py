import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import os

from models.srnet import SRNet
from generators.unified_generator import UnifiedGenerator

# ==================== CLI ====================

def parse_args():
    parser = argparse.ArgumentParser(description="Sliding-window steganalysis class demo")
    parser.add_argument(
        '--image',
        default=os.path.join('data', 'raw', 'BossBase and BOWS2', '2.pgm'),
        help='Path to cover image (default: data/raw/BossBase and BOWS2/2.pgm)')
    parser.add_argument(
        '--model', default='srnet_best_val.pth',
        help='Path to model checkpoint (default: srnet_best_val.pth)')
    return parser.parse_args()


# ==================== CONFIGURATION ====================

DEVICE           = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WINDOW_SIZE      = 256
WINDOW_STRIDE    = 64
DETECTION_THRESH = 0.50
SECRET_MESSAGE   = "This is a secret message " * 4096

# ==================== STRATEGY CONFIGS ====================
STRATEGY_CONFIGS = {
    # ── LSB — spatial domain ──────────────────────────────────────────────────
    'lsb_sequential': {
        'gen_type': 'lsb', 'strategy': 'sequential',
        'capacity_ratio': 0.50, 'edge_threshold': 0,
        'bit_depth': 1, 'step': 1, 'message': SECRET_MESSAGE,
    },
    'lsb_random': {
        'gen_type': 'lsb', 'strategy': 'random',
        'capacity_ratio': 0.50, 'edge_threshold': 0,
        'bit_depth': 1, 'step': 1, 'message': SECRET_MESSAGE,
    },
    'lsb_skip': {
        # Best evolved skip genome: Cap 0.56, Edge threshold 95-96
        'gen_type': 'lsb', 'strategy': 'skip',
        'capacity_ratio': 0.56, 'edge_threshold': 95,
        'bit_depth': 1, 'step': 3, 'message': SECRET_MESSAGE,
    },
    'lsb_edge': {
        # Best evolved edge genome: Cap 0.21, Edge threshold 9
        'gen_type': 'lsb', 'strategy': 'edge',
        'capacity_ratio': 0.21, 'edge_threshold': 9,
        'bit_depth': 1, 'step': 1, 'message': SECRET_MESSAGE,
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
    'fft_mid': {
        'gen_type': 'fft', 'freq_band': 'mid',
        'strength': 8.0, 'capacity_ratio': 0.30,
    },
    'fft_high': {
        'gen_type': 'fft', 'freq_band': 'high',
        'strength': 6.0, 'capacity_ratio': 0.25,
    },
}


# ==================== SLIDING WINDOW DETECTION ====================

def sliding_window_detect(ai_model, image: Image.Image) -> dict:
    to_tensor = transforms.ToTensor()
    img_array = np.array(image.convert('L'), dtype=np.uint8)
    h, w      = img_array.shape

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
                patch        = img_array[top: top + WINDOW_SIZE, left: left + WINDOW_SIZE]
                patch_tensor = to_tensor(Image.fromarray(patch)).unsqueeze(0).to(DEVICE)
                prob         = torch.softmax(ai_model(patch_tensor), dim=1)[0, 1].item()
                scores.append(prob)

    if not scores:
        return {'scores': [], 'max_score': 0.0, 'mean_score': 0.0,
                'flagged': 0, 'total': 0, 'verdict': 'NO WINDOWS'}

    max_score  = max(scores)
    mean_score = sum(scores) / len(scores)
    flagged    = sum(1 for s in scores if s > DETECTION_THRESH)

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
    img  = Image.open(path).convert('L')
    w, h = img.size
    if w < WINDOW_SIZE or h < WINDOW_SIZE:
        scale = max(WINDOW_SIZE / w, WINDOW_SIZE / h)
        img   = img.resize((int(w * scale) + 1, int(h * scale) + 1),
                           Image.Resampling.LANCZOS)
        w, h = img.size
    new_w = ((w - WINDOW_SIZE) // WINDOW_STRIDE) * WINDOW_STRIDE + WINDOW_SIZE
    new_h = ((h - WINDOW_SIZE) // WINDOW_STRIDE) * WINDOW_STRIDE + WINDOW_SIZE
    left  = (w - new_w) // 2
    top   = (h - new_h) // 2
    return img.crop((left, top, left + new_w, top + new_h))


def print_window_result(result: dict):
    flagged_pct = 100.0 * result['flagged'] / result['total'] if result['total'] > 0 else 0.0
    print(f"  Max window score:   {result['max_score']  * 100:>6.1f}%")
    print(f"  Mean window score:  {result['mean_score'] * 100:>6.1f}%")
    print(f"  Flagged windows:    {result['flagged']:>4d} / {result['total']} "
          f"({flagged_pct:.1f}%)")
    print(f"  Verdict:            {result['verdict']}")


# ==================== MAIN DEMO ====================

def run_strategy_evaluation(model_path: str, image_path: str):
    print("\n" + "=" * 70)
    print("     STRATEGY EVALUATION — SLIDING WINDOW STEGANALYSIS")
    print(f"     Window: {WINDOW_SIZE}×{WINDOW_SIZE}  |  Stride: {WINDOW_STRIDE}  |  "
          f"Flag threshold: {DETECTION_THRESH * 100:.0f}%")
    print(f"     Generators: LSB (4 strategies) + DCT (2 variants) + FFT (2 variants)")
    print("=" * 70)

    print(f"\n[SETUP]  Loading model: {model_path}  |  Device: {DEVICE}")
    ai_detector = SRNet().to(DEVICE)
    checkpoint  = torch.load(model_path, map_location=DEVICE)
    ai_detector.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    ai_detector.eval()

    stego_gen = UnifiedGenerator()

    cover_img = load_and_prepare_image(image_path)
    w, h      = cover_img.size
    windows_h = (h - WINDOW_SIZE) // WINDOW_STRIDE + 1
    windows_w = (w - WINDOW_SIZE) // WINDOW_STRIDE + 1
    total_windows = windows_h * windows_w

    print(f"         Image:   {os.path.basename(image_path)}  ({w}×{h} after prep)")
    print(f"         Windows: {windows_h} rows × {windows_w} cols = {total_windows}\n")

    # Baseline
    print("=" * 70)
    print("[BASELINE]  Clean image (no embedding)")
    print("=" * 70)
    baseline = sliding_window_detect(ai_detector, cover_img)
    print_window_result(baseline)

    results_summary = {}

    for strategy_name, config in STRATEGY_CONFIGS.items():
        gt = config['gen_type']
        print()
        print("=" * 70)
        print(f"[STRATEGY: {strategy_name.upper()}]  ({gt.upper()})")
        if gt == 'lsb':
            print(f"  Capacity: {config['capacity_ratio']:.2f}  |  "
                  f"Edge threshold: {config['edge_threshold']}  |  Step: {config.get('step', 1)}")
        elif gt == 'dct':
            print(f"  Capacity: {config['capacity_ratio']:.2f}  |  "
                  f"Coeff: {config['coeff_selection']}  |  Strength: {config['strength']:.1f}")
        else:
            print(f"  Capacity: {config['capacity_ratio']:.2f}  |  "
                  f"Band: {config['freq_band']}  |  Strength: {config['strength']:.1f}")
        print("=" * 70)

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

        out_path = f"demo_stego_{strategy_name}.png"
        stego_img.save(out_path)
        print(f"  Saved: {out_path}")

        print(f"\n  Running sliding window detection...")
        result = sliding_window_detect(ai_detector, stego_img)
        print_window_result(result)

        delta_max  = (result['max_score']  - baseline['max_score'])  * 100
        delta_mean = (result['mean_score'] - baseline['mean_score']) * 100
        print(f"\n  Δ vs baseline  →  Max: {delta_max:+.1f}%   Mean: {delta_mean:+.1f}%")

        results_summary[strategy_name] = {
            'gen_type':   gt,
            'psnr':       psnr,
            'max_score':  result['max_score'],
            'mean_score': result['mean_score'],
            'flagged':    result['flagged'],
            'total':      result['total'],
            'verdict':    result['verdict'],
            'delta_max':  delta_max,
            'delta_mean': delta_mean,
        }

    # Final table
    print()
    print("=" * 70)
    print("                    FINAL COMPARISON TABLE")
    print("=" * 70)
    print(f"  {'Strategy':<18}  {'Type':>4}  {'PSNR':>7}  {'Max%':>7}  "
          f"{'Mean%':>7}  {'Flagged':>10}  {'ΔMax':>6}  Verdict")
    print("  " + "-" * 70)

    flagged_pct_bl = 100.0 * baseline['flagged'] / baseline['total'] if baseline['total'] > 0 else 0
    print(f"  {'clean':<18}  {'—':>4}  {'—':>7}  "
          f"{baseline['max_score'] * 100:>6.1f}%  "
          f"{baseline['mean_score'] * 100:>6.1f}%  "
          f"{baseline['flagged']:>4d}/{baseline['total']:<4d} ({flagged_pct_bl:4.1f}%)  "
          f"{'—':>6}  {baseline['verdict']}")

    for name, r in results_summary.items():
        flagged_pct = 100.0 * r['flagged'] / r['total'] if r['total'] > 0 else 0
        print(f"  {name:<18}  {r['gen_type']:>4}  {r['psnr']:>6.1f}dB  "
              f"{r['max_score'] * 100:>6.1f}%  "
              f"{r['mean_score'] * 100:>6.1f}%  "
              f"{r['flagged']:>4d}/{r['total']:<4d} ({flagged_pct:4.1f}%)  "
              f"{r['delta_max']:>+5.1f}%  {r['verdict']}")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    args = parse_args()
    run_strategy_evaluation(model_path=args.model, image_path=args.image)
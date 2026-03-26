import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import pandas as pd
from pathlib import Path

from models.srnet import SRNet

# --- CONFIGURATION ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WINDOW_SIZE = 256
STRIDE = 128
NUM_TEST_IMAGES = 100


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate SRNet on clean vs LSB stego images (grayscale & RGB)")
    parser.add_argument(
        '--model', default='srnet_best_val.pth',
        help='Path to model checkpoint (default: srnet_best_val.pth)')
    parser.add_argument(
        '--clean-dir',
        default='data/external/clean',
        help='Directory of clean test images')
    parser.add_argument(
        '--stego-gray-dir',
        default='data/external/lsb_grayscale',
        help='Directory of grayscale LSB stego images')
    parser.add_argument(
        '--stego-rgb-dir',
        default='data/external/lsb_rgb',
        help='Directory of RGB LSB stego images')
    parser.add_argument(
        '--num-images', type=int, default=NUM_TEST_IMAGES,
        help=f'Max images per category (default: {NUM_TEST_IMAGES})')
    return parser.parse_args()


def load_model(model_path: str):
    print(f"[INFO] Loading model from {model_path} on {DEVICE}...")
    model = SRNet().to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("[INFO] Model loaded successfully")
    return model


def evaluate_single_channel(model, img_grayscale, transform):
    """
    Evaluate a single grayscale channel using sliding windows.
    Returns max score and mean score across all tiles.
    """
    w, h = img_grayscale.size

    if w < WINDOW_SIZE or h < WINDOW_SIZE:
        img_grayscale = img_grayscale.resize((WINDOW_SIZE, WINDOW_SIZE), Image.Resampling.LANCZOS)
        w, h = WINDOW_SIZE, WINDOW_SIZE

    tiles = []
    for y in range(0, h - WINDOW_SIZE + 1, STRIDE):
        for x in range(0, w - WINDOW_SIZE + 1, STRIDE):
            tile = img_grayscale.crop((x, y, x + WINDOW_SIZE, y + WINDOW_SIZE))
            tiles.append(transform(tile))

    if not tiles:
        tiles = [transform(img_grayscale)]

    batch_tensors = torch.stack(tiles).to(DEVICE)

    with torch.no_grad():
        outputs = model(batch_tensors)
        probs = F.softmax(outputs, dim=1)
        stego_probs = probs[:, 1].cpu().numpy()

    return stego_probs.max(), stego_probs.mean(), len(tiles)


def evaluate_image(model, img_path, transform):
    """
    Evaluate an image intelligently:
    - RGB:       test each channel separately (stego might be in R, G, or B)
    - Grayscale: test directly
    - RGBA:      test RGB channels, ignore alpha
    """
    try:
        img = Image.open(img_path)

        if img.mode == 'RGB':
            img_array = np.array(img)

            r_channel = Image.fromarray(img_array[:, :, 0], mode='L')
            g_channel = Image.fromarray(img_array[:, :, 1], mode='L')
            b_channel = Image.fromarray(img_array[:, :, 2], mode='L')

            max_r, mean_r, tiles_r = evaluate_single_channel(model, r_channel, transform)
            max_g, mean_g, tiles_g = evaluate_single_channel(model, g_channel, transform)
            max_b, mean_b, tiles_b = evaluate_single_channel(model, b_channel, transform)

            final_max  = max(max_r, max_g, max_b)
            final_mean = max(mean_r, mean_g, mean_b)

            channel_scores = {'R': max_r, 'G': max_g, 'B': max_b}
            best_channel   = max(channel_scores, key=channel_scores.get)

            return final_max, final_mean, tiles_r, best_channel

        elif img.mode == 'L':
            max_score, mean_score, num_tiles = evaluate_single_channel(model, img, transform)
            return max_score, mean_score, num_tiles, 'L'

        elif img.mode == 'RGBA':
            img_array = np.array(img)

            r_channel = Image.fromarray(img_array[:, :, 0], mode='L')
            g_channel = Image.fromarray(img_array[:, :, 1], mode='L')
            b_channel = Image.fromarray(img_array[:, :, 2], mode='L')

            max_r, mean_r, tiles_r = evaluate_single_channel(model, r_channel, transform)
            max_g, mean_g, tiles_g = evaluate_single_channel(model, g_channel, transform)
            max_b, mean_b, tiles_b = evaluate_single_channel(model, b_channel, transform)

            final_max  = max(max_r, max_g, max_b)
            final_mean = max(mean_r, mean_g, mean_b)

            channel_scores = {'R': max_r, 'G': max_g, 'B': max_b}
            best_channel   = max(channel_scores, key=channel_scores.get)

            return final_max, final_mean, tiles_r, best_channel

        else:
            print(f"[WARN] Unsupported format '{img.mode}' for {img_path.name}")
            return 0.0, 0.0, 0, 'Unknown'

    except Exception as e:
        print(f"[ERROR] Failed to process {img_path.name}: {e}")
        return 0.0, 0.0, 0, 'Error'


def run_evaluation(model_path, clean_dir, stego_gray_dir, stego_rgb_dir, num_images):
    print("🔍 Starting Comprehensive LSB Evaluation: Grayscale vs RGB")
    print("=" * 70)
    print("Testing model on:")
    print("  • Clean RGB images")
    print("  • Stego Grayscale LSB")
    print("  • Stego RGB LSB")
    print("=" * 70)

    clean_dir      = Path(clean_dir)
    stego_gray_dir = Path(stego_gray_dir)
    stego_rgb_dir  = Path(stego_rgb_dir)

    for label, directory in [('Clean', clean_dir),
                              ('Stego Gray', stego_gray_dir),
                              ('Stego RGB', stego_rgb_dir)]:
        if not directory.exists():
            print(f"[ERROR] {label} directory not found: {directory}")
            return

    model     = load_model(model_path)
    transform = transforms.ToTensor()
    results   = []

    # ===== CLEAN IMAGES =====
    clean_files = list(clean_dir.glob("*.*"))[:num_images]
    print(f"\n📂 Evaluating {len(clean_files)} CLEAN images (RGB)...")

    for img_path in tqdm(clean_files, desc="Clean"):
        max_score, mean_score, num_tiles, channel = evaluate_image(model, img_path, transform)
        results.append({
            "Type": "Clean",
            "Subtype": "Clean_RGB",
            "Filename": img_path.name,
            "Max_Score": max_score,
            "Mean_Score": mean_score,
            "Num_Tiles": num_tiles,
            "Best_Channel": channel,
            "Prediction": "Stego" if max_score > 0.5 else "Clean",
            "Correct": max_score < 0.5
        })

    # ===== GRAYSCALE STEGO =====
    stego_gray_files = list(stego_gray_dir.glob("*.*"))[:num_images]
    print(f"\n📂 Evaluating {len(stego_gray_files)} STEGO GRAYSCALE images...")

    for img_path in tqdm(stego_gray_files, desc="Stego Gray"):
        max_score, mean_score, num_tiles, channel = evaluate_image(model, img_path, transform)
        results.append({
            "Type": "Stego",
            "Subtype": "Stego_Grayscale",
            "Filename": img_path.name,
            "Max_Score": max_score,
            "Mean_Score": mean_score,
            "Num_Tiles": num_tiles,
            "Best_Channel": channel,
            "Prediction": "Stego" if max_score > 0.5 else "Clean",
            "Correct": max_score > 0.5
        })

    # ===== RGB STEGO =====
    stego_rgb_files = list(stego_rgb_dir.glob("*.*"))[:num_images]
    print(f"\n📂 Evaluating {len(stego_rgb_files)} STEGO RGB images...")

    for img_path in tqdm(stego_rgb_files, desc="Stego RGB"):
        max_score, mean_score, num_tiles, channel = evaluate_image(model, img_path, transform)
        results.append({
            "Type": "Stego",
            "Subtype": "Stego_RGB",
            "Filename": img_path.name,
            "Max_Score": max_score,
            "Mean_Score": mean_score,
            "Num_Tiles": num_tiles,
            "Best_Channel": channel,
            "Prediction": "Stego" if max_score > 0.5 else "Clean",
            "Correct": max_score > 0.5
        })

    # ===== ANALYSIS =====
    df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("                    COMPREHENSIVE RESULTS")
    print("=" * 70)

    print("\n📊 Accuracy by Image Type:")
    print("-" * 70)
    for subtype in ['Clean_RGB', 'Stego_Grayscale', 'Stego_RGB']:
        subset = df[df['Subtype'] == subtype]
        if len(subset) > 0:
            acc      = subset['Correct'].mean() * 100
            avg_max  = subset['Max_Score'].mean()
            avg_mean = subset['Mean_Score'].mean()
            print(f"{subtype:20s}: {acc:6.2f}% accuracy | "
                  f"Avg Max Score: {avg_max:.4f} | Avg Mean Score: {avg_mean:.4f}")

    print("\n" + "-" * 70)
    overall_acc = df['Correct'].mean() * 100
    clean_acc   = df[df['Type'] == 'Clean']['Correct'].mean() * 100
    stego_acc   = df[df['Type'] == 'Stego']['Correct'].mean() * 100

    print(f"\n🎯 OVERALL ACCURACY:     {overall_acc:.2f}%")
    print(f"   Clean Detection:      {clean_acc:.2f}%  (True Negative Rate)")
    print(f"   Stego Detection:      {stego_acc:.2f}%  (True Positive Rate)")

    stego_gray_acc = df[df['Subtype'] == 'Stego_Grayscale']['Correct'].mean() * 100
    stego_rgb_acc  = df[df['Subtype'] == 'Stego_RGB']['Correct'].mean() * 100
    print(f"\n   ├─ Grayscale LSB:     {stego_gray_acc:.2f}%")
    print(f"   └─ RGB LSB:           {stego_rgb_acc:.2f}%")

    print("\n📊 Channel Detection Distribution:")
    print("-" * 70)
    channel_breakdown = df.groupby(['Subtype', 'Best_Channel']).size().unstack(fill_value=0)
    print(channel_breakdown)

    print("\n" + "=" * 70)

    tp = len(df[(df['Type'] == 'Stego') & (df['Correct'] == True)])
    fn = len(df[(df['Type'] == 'Stego') & (df['Correct'] == False)])
    tn = len(df[(df['Type'] == 'Clean') & (df['Correct'] == True)])
    fp = len(df[(df['Type'] == 'Clean') & (df['Correct'] == False)])

    print("\nConfusion Matrix:")
    print(f"              Predicted Clean  |  Predicted Stego")
    print(f"Actual Clean:      {tn:3d}         |      {fp:3d}")
    print(f"Actual Stego:      {fn:3d}         |      {tp:3d}")

    stem        = Path(model_path).stem
    output_file = f'evaluation_{stem}_grayscale_vs_rgb.csv'
    df.to_csv(output_file, index=False)
    print(f"\n💾 Detailed results saved to: {output_file}")

    print("\n📊 Sample Misclassifications:")
    print("\n❌ False Positives (Clean flagged as Stego):")
    fps = df[(df['Type'] == 'Clean') & (df['Correct'] == False)].head(3)
    print(fps[['Filename', 'Max_Score', 'Best_Channel']] if len(fps) > 0 else "  None! 🎉")

    print("\n❌ False Negatives - Grayscale Stego (flagged as Clean):")
    fns_gray = df[(df['Subtype'] == 'Stego_Grayscale') & (df['Correct'] == False)].head(3)
    print(fns_gray[['Filename', 'Max_Score', 'Best_Channel']] if len(fns_gray) > 0 else "  None! 🎉")

    print("\n❌ False Negatives - RGB Stego (flagged as Clean):")
    fns_rgb = df[(df['Subtype'] == 'Stego_RGB') & (df['Correct'] == False)].head(3)
    print(fns_rgb[['Filename', 'Max_Score', 'Best_Channel']] if len(fns_rgb) > 0 else "  None! 🎉")

    print("\n" + "=" * 70)
    print("🔍 KEY INSIGHTS:")
    print("=" * 70)

    if stego_gray_acc > stego_rgb_acc + 5:
        print(f"✅ Model performs BETTER on grayscale LSB ({stego_gray_acc:.1f}%) vs RGB LSB ({stego_rgb_acc:.1f}%)")
        print("   → This matches your training (grayscale-focused)")
    elif stego_rgb_acc > stego_gray_acc + 5:
        print(f"⚠️  Model performs BETTER on RGB LSB ({stego_rgb_acc:.1f}%) vs grayscale LSB ({stego_gray_acc:.1f}%)")
        print("   → Unexpected! May indicate channel extraction is helping")
    else:
        print(f"✅ Model performs SIMILARLY on both grayscale ({stego_gray_acc:.1f}%) and RGB LSB ({stego_rgb_acc:.1f}%)")
        print("   → Good generalization across color formats!")

    print("=" * 70)


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(
        model_path     = args.model,
        clean_dir      = args.clean_dir,
        stego_gray_dir = args.stego_gray_dir,
        stego_rgb_dir  = args.stego_rgb_dir,
        num_images     = args.num_images,
    )
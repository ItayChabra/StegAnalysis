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
MODEL_PATH = "srnet_epoch_30.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WINDOW_SIZE = 256
STRIDE = 128
NUM_TEST_IMAGES = 100


def load_model():
    print(f"[INFO] Loading model from {MODEL_PATH} on {DEVICE}...")
    model = SRNet().to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
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

    # Resize if too small
    if w < WINDOW_SIZE or h < WINDOW_SIZE:
        img_grayscale = img_grayscale.resize((WINDOW_SIZE, WINDOW_SIZE), Image.Resampling.LANCZOS)
        w, h = WINDOW_SIZE, WINDOW_SIZE

    # Extract tiles
    tiles = []
    for y in range(0, h - WINDOW_SIZE + 1, STRIDE):
        for x in range(0, w - WINDOW_SIZE + 1, STRIDE):
            tile = img_grayscale.crop((x, y, x + WINDOW_SIZE, y + WINDOW_SIZE))
            tiles.append(transform(tile))

    if not tiles:
        tiles = [transform(img_grayscale)]

    # Batch inference
    batch_tensors = torch.stack(tiles).to(DEVICE)

    with torch.no_grad():
        outputs = model(batch_tensors)
        probs = F.softmax(outputs, dim=1)
        stego_probs = probs[:, 1].cpu().numpy()

    return stego_probs.max(), stego_probs.mean(), len(tiles)


def evaluate_image(model, img_path, transform):
    """
    Evaluate an image intelligently:
    - If RGB: Test each channel separately (stego might be in R, G, or B)
    - If grayscale: Test directly

    This handles the common case where LSB is embedded in one RGB channel.
    """
    try:
        img = Image.open(img_path)

        # Case 1: RGB image - test each channel separately
        if img.mode == 'RGB':
            img_array = np.array(img)

            # Extract and test each channel as grayscale
            r_channel = Image.fromarray(img_array[:, :, 0], mode='L')
            g_channel = Image.fromarray(img_array[:, :, 1], mode='L')
            b_channel = Image.fromarray(img_array[:, :, 2], mode='L')

            max_r, mean_r, tiles_r = evaluate_single_channel(model, r_channel, transform)
            max_g, mean_g, tiles_g = evaluate_single_channel(model, g_channel, transform)
            max_b, mean_b, tiles_b = evaluate_single_channel(model, b_channel, transform)

            # Take maximum across all channels
            # (If stego is in any channel, we'll detect it)
            final_max = max(max_r, max_g, max_b)
            final_mean = max(mean_r, mean_g, mean_b)

            # Identify which channel had highest score
            channel_scores = {'R': max_r, 'G': max_g, 'B': max_b}
            best_channel = max(channel_scores, key=channel_scores.get)

            return final_max, final_mean, tiles_r, best_channel

        # Case 2: Grayscale image - test directly
        elif img.mode == 'L':
            max_score, mean_score, num_tiles = evaluate_single_channel(model, img, transform)
            return max_score, mean_score, num_tiles, 'L'

        # Case 3: RGBA - test RGB channels, ignore alpha
        elif img.mode == 'RGBA':
            img_array = np.array(img)

            r_channel = Image.fromarray(img_array[:, :, 0], mode='L')
            g_channel = Image.fromarray(img_array[:, :, 1], mode='L')
            b_channel = Image.fromarray(img_array[:, :, 2], mode='L')

            max_r, mean_r, tiles_r = evaluate_single_channel(model, r_channel, transform)
            max_g, mean_g, tiles_g = evaluate_single_channel(model, g_channel, transform)
            max_b, mean_b, tiles_b = evaluate_single_channel(model, b_channel, transform)

            final_max = max(max_r, max_g, max_b)
            final_mean = max(mean_r, mean_g, mean_b)

            channel_scores = {'R': max_r, 'G': max_g, 'B': max_b}
            best_channel = max(channel_scores, key=channel_scores.get)

            return final_max, final_mean, tiles_r, best_channel

        # Unsupported format
        else:
            print(f"[WARN] Unsupported format '{img.mode}' for {img_path.name}")
            return 0.0, 0.0, 0, 'Unknown'

    except Exception as e:
        print(f"[ERROR] Failed to process {img_path.name}: {e}")
        return 0.0, 0.0, 0, 'Error'


def run_evaluation():
    print("🔍 Starting Comprehensive LSB Evaluation: Grayscale vs RGB")
    print("=" * 70)
    print("Testing model on:")
    print("  • Clean RGB images (First Kaggle)")
    print("  • Stego Grayscale LSB (Diego Zanchett)")
    print("  • Stego RGB LSB (Diego Zanchett)")
    print("=" * 70)

    # Dataset paths
    clean_dir = Path(r"E:\Stego-Images-Dataset-LSB\test\test\clean")
    stego_gray_dir = Path(r"E:\Digital Steganography\lsb_grayscale")
    stego_rgb_dir = Path(r"E:\Digital Steganography\lsb")

    # Verify paths
    if not clean_dir.exists():
        print(f"[ERROR] Clean directory not found: {clean_dir}")
        return
    if not stego_gray_dir.exists():
        print(f"[ERROR] Grayscale stego directory not found: {stego_gray_dir}")
        return
    if not stego_rgb_dir.exists():
        print(f"[ERROR] RGB stego directory not found: {stego_rgb_dir}")
        return

    # Load model
    model = load_model()
    transform = transforms.ToTensor()
    results = []

    # ===== EVALUATE CLEAN IMAGES =====
    clean_files = list(clean_dir.glob("*.*"))[:NUM_TEST_IMAGES]
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

    # ===== EVALUATE GRAYSCALE STEGO =====
    stego_gray_files = list(stego_gray_dir.glob("*.*"))[:NUM_TEST_IMAGES]
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

    # ===== EVALUATE RGB STEGO =====
    stego_rgb_files = list(stego_rgb_dir.glob("*.*"))[:NUM_TEST_IMAGES]
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

    # Detailed accuracy by subtype
    print("\n📊 Accuracy by Image Type:")
    print("-" * 70)
    for subtype in ['Clean_RGB', 'Stego_Grayscale', 'Stego_RGB']:
        subset = df[df['Subtype'] == subtype]
        if len(subset) > 0:
            acc = subset['Correct'].mean() * 100
            avg_max = subset['Max_Score'].mean()
            avg_mean = subset['Mean_Score'].mean()
            print(f"{subtype:20s}: {acc:6.2f}% accuracy | "
                  f"Avg Max Score: {avg_max:.4f} | Avg Mean Score: {avg_mean:.4f}")

    # Overall metrics
    print("\n" + "-" * 70)
    overall_acc = df['Correct'].mean() * 100
    clean_acc = df[df['Type'] == 'Clean']['Correct'].mean() * 100
    stego_acc = df[df['Type'] == 'Stego']['Correct'].mean() * 100

    print(f"\n🎯 OVERALL ACCURACY:     {overall_acc:.2f}%")
    print(f"   Clean Detection:      {clean_acc:.2f}%  (True Negative Rate)")
    print(f"   Stego Detection:      {stego_acc:.2f}%  (True Positive Rate)")

    # Separate stego accuracies
    stego_gray_acc = df[df['Subtype'] == 'Stego_Grayscale']['Correct'].mean() * 100
    stego_rgb_acc = df[df['Subtype'] == 'Stego_RGB']['Correct'].mean() * 100
    print(f"\n   ├─ Grayscale LSB:     {stego_gray_acc:.2f}%")
    print(f"   └─ RGB LSB:           {stego_rgb_acc:.2f}%")

    # Channel analysis
    print("\n📊 Channel Detection Distribution:")
    print("-" * 70)
    channel_breakdown = df.groupby(['Subtype', 'Best_Channel']).size().unstack(fill_value=0)
    print(channel_breakdown)

    print("\n" + "=" * 70)

    # Confusion matrix
    tp = len(df[(df['Type'] == 'Stego') & (df['Correct'] == True)])
    fn = len(df[(df['Type'] == 'Stego') & (df['Correct'] == False)])
    tn = len(df[(df['Type'] == 'Clean') & (df['Correct'] == True)])
    fp = len(df[(df['Type'] == 'Clean') & (df['Correct'] == False)])

    print("\nConfusion Matrix:")
    print(f"              Predicted Clean  |  Predicted Stego")
    print(f"Actual Clean:      {tn:3d}         |      {fp:3d}")
    print(f"Actual Stego:      {fn:3d}         |      {tp:3d}")

    # Save results
    output_file = f'evaluation_{MODEL_PATH.replace(".pth", "")}_grayscale_vs_rgb.csv'
    df.to_csv(output_file, index=False)
    print(f"\n💾 Detailed results saved to: {output_file}")

    # Show examples
    print("\n📊 Sample Misclassifications:")

    print("\n❌ False Positives (Clean flagged as Stego):")
    fps = df[(df['Type'] == 'Clean') & (df['Correct'] == False)].head(3)
    if len(fps) > 0:
        print(fps[['Filename', 'Max_Score', 'Best_Channel']])
    else:
        print("  None! 🎉")

    print("\n❌ False Negatives - Grayscale Stego (flagged as Clean):")
    fns_gray = df[(df['Subtype'] == 'Stego_Grayscale') & (df['Correct'] == False)].head(3)
    if len(fns_gray) > 0:
        print(fns_gray[['Filename', 'Max_Score', 'Best_Channel']])
    else:
        print("  None! 🎉")

    print("\n❌ False Negatives - RGB Stego (flagged as Clean):")
    fns_rgb = df[(df['Subtype'] == 'Stego_RGB') & (df['Correct'] == False)].head(3)
    if len(fns_rgb) > 0:
        print(fns_rgb[['Filename', 'Max_Score', 'Best_Channel']])
    else:
        print("  None! 🎉")

    # Key insights
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
    run_evaluation()
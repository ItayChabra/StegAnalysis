import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import os

from models.srnet import SRNet
from generators.unified_generator import UnifiedGenerator

# ==================== CONFIGURATION ====================
MODEL_PATH = "srnet_epoch_30.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLEAN_IMAGE_PATH = r"E:\PycharmProjects\data\raw\BossBase and BOWS2\1.pgm"
OUTPUT_CLEAN_CROP = "demo_clean.png"
OUTPUT_STEGO_PATH = "demo_stego.png"

SECRET_MESSAGE = "This is a secret message " * 4096


# ==================== DETECTION FUNCTION ====================
def detect_steganography(ai_model, image_path):
    """Use AI to detect if an image contains hidden data."""
    img = Image.open(image_path).convert('L')
    width, height = img.size

    # Ensure minimum size
    if width < 256 or height < 256:
        new_size = (max(256, width), max(256, height))
        padded_img = Image.new('L', new_size, color=128)
        padded_img.paste(img, ((new_size[0] - width) // 2, (new_size[1] - height) // 2))
        img = padded_img
        width, height = img.size

    # Center crop to 256x256
    left = (width - 256) // 2
    top = (height - 256) // 2
    cropped_img = img.crop((left, top, left + 256, top + 256))

    # Convert to tensor and run through model
    img_tensor = transforms.ToTensor()(cropped_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = ai_model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        return probabilities[0, 1].item()


# ==================== MAIN DEMO ====================
def run_classroom_demo():
    """Classroom demonstration: Hide message and detect with AI"""

    print("\n" + "=" * 70)
    print("       STEGANOGRAPHY & AI DETECTION - CLASSROOM DEMO")
    print("=" * 70)

    # PHASE 1: SETUP
    print("\n[PHASE 1: SETUP]")
    print(f"  Loading AI detection model: {MODEL_PATH}")

    ai_detector = SRNet().to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    ai_detector.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    ai_detector.eval()

    stego_generator = UnifiedGenerator()
    print("  System initialized\n")

    # PHASE 2: ANALYZE CLEAN IMAGE
    print("=" * 70)
    print("[PHASE 2: ANALYZE CLEAN IMAGE - Baseline Test]")
    print("=" * 70)

    clean_image = Image.open(CLEAN_IMAGE_PATH).convert('L')
    width, height = clean_image.size
    print(f"  Original image: {os.path.basename(CLEAN_IMAGE_PATH)}")
    print(f"  Size: {width} x {height} pixels")

    # Crop to 256x256
    if width >= 256 and height >= 256:
        left = (width - 256) // 2
        top = (height - 256) // 2
        clean_cropped = clean_image.crop((left, top, left + 256, top + 256))
    else:
        clean_cropped = clean_image.resize((256, 256), Image.Resampling.LANCZOS)

    clean_cropped.save(OUTPUT_CLEAN_CROP)
    print(f"  Saved: {OUTPUT_CLEAN_CROP}")

    print(f"\n  Running AI detection on clean image...")
    clean_score = detect_steganography(ai_detector, OUTPUT_CLEAN_CROP)

    print(f"\n  RESULT: {clean_score * 100:.1f}% confidence of hidden data")
    print(f"  Verdict: {'CLEAN' if clean_score < 0.5 else 'SUSPICIOUS'}")

    # PHASE 3: HIDE SECRET MESSAGE
    print("\n" + "=" * 70)
    print("[PHASE 3: HIDE SECRET MESSAGE - LSB Steganography]")
    print("=" * 70)

    print(f"  Secret Message: \"{SECRET_MESSAGE[:50]}...\"")
    print(f"  Message Length: {len(SECRET_MESSAGE):,} characters")

    embedding_config = {
        'gen_type': 'lsb',
        'strategy': 'sequential',
        'message': SECRET_MESSAGE,
        'capacity_ratio': 0.50,
        'edge_threshold': 0,
        'bit_depth': 1
    }

    total_pixels = 256 * 256
    pixels_used = int(total_pixels * 0.50)

    print(f"  Embedding Strategy: Sequential LSB")
    print(f"  Capacity Used: 50% ({pixels_used:,} pixels)")
    print(f"\n  Hiding message in image...")

    stego_array, image_quality = stego_generator.generate_stego(
        OUTPUT_CLEAN_CROP, None, embedding_config
    )

    if stego_array is None:
        print("  ERROR: Failed to create stego image!")
        return

    Image.fromarray(stego_array.astype(np.uint8)).save(OUTPUT_STEGO_PATH)

    print(f"  Message successfully hidden!")
    print(f"  Stego image saved: {OUTPUT_STEGO_PATH}")
    print(f"  Image quality: {image_quality:.2f} dB PSNR (>40 dB = imperceptible)")

    # PHASE 4: AI DETECTION
    print("\n" + "=" * 70)
    print("[PHASE 4: AI DETECTION - Can AI Find The Hidden Data?]")
    print("=" * 70)

    print(f"  Scanning stego image with AI detector...")
    print(f"  Model: SRNet (Spatial Rich Network)")

    stego_score = detect_steganography(ai_detector, OUTPUT_STEGO_PATH)

    print(f"\n  RESULT: {stego_score * 100:.1f}% confidence of hidden data")

    if stego_score > 0.75:
        print(f"  Verdict: STEGO DETECTED (High confidence)")
    elif stego_score > 0.50:
        print(f"  Verdict: SUSPICIOUS (Flagged for review)")
    else:
        print(f"  Verdict: CLEAN (Detection failed)")

    # FINAL RESULTS
    print("\n" + "=" * 70)
    print("                    FINAL RESULTS")
    print("=" * 70)

    print(f"\n  Detection Scores:")
    print(f"     Clean Image:  {clean_score * 100:>6.1f}%")
    print(f"     Stego Image:  {stego_score * 100:>6.1f}%")
    print(f"     Difference:   {(stego_score - clean_score) * 100:>+6.1f}%")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    run_classroom_demo()

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def visualize_lsb_plane(image_path, output_path="lsb_plane.png"):
    """
    Extract and visualize the LSB plane.
    Hidden data will appear as patterns in this image!
    """
    # Load image
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)

    # Extract LSB (least significant bit) from each pixel
    lsb_plane = (img_array & 1) * 255  # Multiply by 255 to make visible

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # LSB plane (where the secret is!)
    axes[1].imshow(lsb_plane, cmap='gray')
    axes[1].set_title('LSB Plane (Hidden Data Layer)', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # Difference visualization (enhanced contrast)
    axes[2].imshow(lsb_plane, cmap='hot')
    axes[2].set_title('LSB Plane (Enhanced - Hot Colors)', fontsize=14, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"✅ LSB plane visualization saved to: {output_path}")
    return lsb_plane


def compare_clean_vs_stego_lsb(clean_path, stego_path):
    """
    Side-by-side comparison showing the hidden data
    """
    clean = np.array(Image.open(clean_path).convert('L'))
    stego = np.array(Image.open(stego_path).convert('L'))

    clean_lsb = (clean & 1) * 255
    stego_lsb = (stego & 1) * 255

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Clean image
    axes[0, 0].imshow(clean, cmap='gray')
    axes[0, 0].set_title('Clean Image (Original)', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    # Clean LSB (random noise)
    axes[0, 1].imshow(clean_lsb, cmap='hot')
    axes[0, 1].set_title('Clean LSB Plane (Natural Noise)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    # Stego image (looks identical)
    axes[1, 0].imshow(stego, cmap='gray')
    axes[1, 0].set_title('Stego Image (Looks Identical!)', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')

    # Stego LSB (HIDDEN DATA VISIBLE!)
    axes[1, 1].imshow(stego_lsb, cmap='hot')
    axes[1, 1].set_title('Stego LSB Plane (HIDDEN DATA!)', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig('lsb_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("✅ Comparison saved to: lsb_comparison.png")


# Usage in your demo:
compare_clean_vs_stego_lsb("demo_clean.png", "demo_stego.png")

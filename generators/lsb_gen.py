import numpy as np
from PIL import Image
import os


class LSBGenerator:
    """
    LSB (Least Significant Bit) Generator optimized for Deep Learning Training.
    Features:
    - Auto-resize to target dimensions (e.g., 256x256).
    - Grayscale conversion.
    - Vectorized embedding (Fast).
    - PSNR calculation.
    """

    def __init__(self, payload_size=0.4, target_size=(256, 256)):
        """
        Args:
            payload_size: Ratio of pixels to hide data in (0.0 - 1.0).
            target_size: Tuple (width, height) for SRNet input.
        """
        self.payload_size = payload_size
        self.target_size = target_size

    def embed(self, cover_image_path, output_path=None, message=None):
        """
        Embeds bits into an image.

        Args:
            cover_image_path: Path to source image.
            output_path: Path to save stego image (MUST be .png).
            message: String message (optional). If None, random bits are used.

        Returns:
            stego_array: The resulting image as numpy array.
            psnr: Image quality metric.
        """
        # 1. Load Image
        try:
            img = Image.open(cover_image_path).convert('L')  # Must be Grayscale for SRNet
        except Exception as e:
            print(f"Error opening {cover_image_path}: {e}")
            return None, 0

        # 2. Resize BEFORE embedding (Critical!)
        # אם נקטין את התמונה אחרי ההטמנה, המידע ייהרס בגלל אינטרפולציה
        if img.size != self.target_size:
            img = img.resize(self.target_size, Image.Resampling.LANCZOS)

        img_array = np.array(img, dtype=np.uint8)

        # 3. Calculate Capacity
        total_pixels = img_array.size
        num_pixels_to_embed = int(total_pixels * self.payload_size)

        # 4. Generate Bits (Random or Text)
        if message is None:
            # Fast random bits generation
            message_bits = np.random.randint(0, 2, num_pixels_to_embed, dtype=np.uint8)
        else:
            message_bits = self._text_to_bits(message)
            if len(message_bits) > num_pixels_to_embed:
                # Truncate if too long
                message_bits = message_bits[:num_pixels_to_embed]
            elif len(message_bits) < num_pixels_to_embed:
                # Pad with zeros if too short
                padding = np.zeros(num_pixels_to_embed - len(message_bits), dtype=np.uint8)
                message_bits = np.concatenate([message_bits, padding])

        # 5. Embed Bits (Vectorized - Fast)
        stego_array = img_array.flatten()

        # Choose random pixels to modify
        indices = np.random.choice(total_pixels, num_pixels_to_embed, replace=False)

        # Reset LSB to 0 using Bitwise AND (mask 11111110 -> 0xFE)
        stego_array[indices] &= 0xFE
        # Set LSB to message bit using Bitwise OR
        stego_array[indices] |= message_bits

        # Reshape back to image dimensions
        stego_array = stego_array.reshape(img_array.shape)

        # 6. Save Image
        if output_path:
            # Enforce PNG to prevent compression artifacts
            if not output_path.lower().endswith('.png'):
                output_path = os.path.splitext(output_path)[0] + '.png'

            Image.fromarray(stego_array).save(output_path)

        # 7. Calculate PSNR
        psnr = self._calculate_psnr(img_array, stego_array)

        return stego_array, psnr

    def _text_to_bits(self, text):
        """Converts text string to numpy array of bits."""
        bits = []
        for char in text:
            byte = ord(char)
            for i in range(8):
                bits.append((byte >> (7 - i)) & 1)  # Big Endian is standard
        return np.array(bits, dtype=np.uint8)

    def _calculate_psnr(self, original, stego):
        """Calculates Peak Signal-to-Noise Ratio."""
        mse = np.mean((original.astype(float) - stego.astype(float)) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr


def generate_dataset(cover_dir, output_dir, num_images=1000, payload=0.4):
    """
    Batch processing function to create training dataset.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Initialize Generator
    generator = LSBGenerator(payload_size=payload)

    # Get image files
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.pgm')
    cover_images = [f for f in os.listdir(cover_dir) if f.lower().endswith(valid_exts)]

    # Limit number of images
    cover_images = cover_images[:num_images]

    stats = {'success': 0, 'failed': 0, 'avg_psnr': []}

    print(f"🚀 Starting generation of {len(cover_images)} images...")

    for i, img_name in enumerate(cover_images):
        try:
            cover_path = os.path.join(cover_dir, img_name)
            # Save as PNG
            stego_name = f"stego_{os.path.splitext(img_name)[0]}.png"
            stego_path = os.path.join(output_dir, stego_name)

            _, psnr = generator.embed(cover_path, stego_path)

            if psnr > 0:
                stats['success'] += 1
                stats['avg_psnr'].append(psnr)
            else:
                stats['failed'] += 1

            if (i + 1) % 100 == 0:
                print(f"   Processed {i + 1}/{len(cover_images)}...")

        except Exception as e:
            print(f"Failed on {img_name}: {e}")
            stats['failed'] += 1

    print("\n=== Generation Summary ===")
    print(f"Success: {stats['success']}")
    print(f"Failed: {stats['failed']}")
    if stats['avg_psnr']:
        print(f"Avg PSNR: {np.mean(stats['avg_psnr']):.2f} dB")

    return stats


# --- Main Test Block ---
if __name__ == "__main__":
    print("Testing LSB Generator...")

    dummy = np.random.randint(0, 255, (300, 300), dtype=np.uint8)
    Image.fromarray(dummy).save("temp_test.jpg")

    gen = LSBGenerator(payload_size=0.4)
    _, psnr = gen.embed("temp_test.jpg", "temp_stego.png", message="Secret!")

    print(f"PSNR: {psnr:.2f}")

    if os.path.exists("temp_test.jpg"): os.remove("temp_test.jpg")
    if os.path.exists("temp_stego.png"): os.remove("temp_stego.png")
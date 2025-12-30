from generators.base_generator import BaseGenerator
import numpy as np
from PIL import Image
import os


class LSBGenerator(BaseGenerator):
    def __init__(self, target_size=(256, 256)):
        self.target_size = target_size

    def _get_complex_areas(self, img_array, threshold):
        """Returns indices of pixels that are on edges/textures."""
        img_f = img_array.astype(float)
        dx = np.diff(img_f, axis=1, append=0)
        dy = np.diff(img_f, axis=0, append=0)
        magnitude = np.sqrt(dx ** 2 + dy ** 2)
        return np.where(magnitude.flatten() > threshold)[0]

    def _text_to_bits(self, text):
        """Helper: Converts string to numpy array of bits."""
        b = text.encode('utf-8')
        arr = np.frombuffer(b, dtype=np.uint8)
        bits = np.unpackbits(arr)
        return bits

    def run(self, cover_path, output_path, **params):
        """
        Implementation of the BaseGenerator interface.
        """
        strategy = params.get('strategy', 'random')
        step = params.get('step', 1)
        bit_depth = params.get('bit_depth', 1)
        edge_threshold = params.get('edge_threshold', 0)
        message = params.get('message', None)
        capacity_ratio = params.get('capacity_ratio', 0.5)  # NEW parameter

        return self.embed(cover_path, output_path,
                          message=message,
                          strategy=strategy,
                          step=step,
                          bit_depth=bit_depth,
                          edge_threshold=edge_threshold,
                          capacity_ratio=capacity_ratio)

    def embed(self, cover_path, output_path, message=None,
              strategy='random', step=1,
              bit_depth=1, edge_threshold=0,
              capacity_ratio=0.5):

        # 1. Load & Resize
        try:
            img = Image.open(cover_path).convert('L')
        except Exception:
            return None, 0

        if img.size != self.target_size:
            img = img.resize(self.target_size, Image.Resampling.LANCZOS)

        img_array = np.array(img, dtype=np.uint8)
        flat_img = img_array.flatten()
        total_pixels = flat_img.size

        # 2. Determine Available Pixels
        if edge_threshold > 0:
            available_indices = self._get_complex_areas(img_array, edge_threshold)
            if len(available_indices) == 0:
                available_indices = np.arange(total_pixels)
        else:
            available_indices = np.arange(total_pixels)

        # 3. Strategy Selection with Capacity Control

        # Calculate target number of pixels based on capacity_ratio
        target_pixels = int(len(available_indices) * capacity_ratio)
        target_pixels = max(1, target_pixels)  # At least 1 pixel

        if strategy == 'random':
            # Randomly select target_pixels from available
            chosen_indices = np.random.choice(
                available_indices,
                min(target_pixels, len(available_indices)),
                replace=False
            )

        elif strategy == 'skip':
            # Use skip pattern, but limit to target_pixels
            skipped = available_indices[::step]
            if len(skipped) > target_pixels:
                # Randomly sample from skipped to reach target
                chosen_indices = np.random.choice(skipped, target_pixels, replace=False)
            else:
                chosen_indices = skipped

        elif strategy == 'sequential':
            # Take first N available pixels
            chosen_indices = available_indices[:target_pixels]

        else:  # Fallback
            chosen_indices = available_indices[:target_pixels]

        # 4. Generate or Prepare Bits
        exact_bits_needed = len(chosen_indices) * bit_depth

        if message:
            bits = self._text_to_bits(message)
            if len(bits) > exact_bits_needed:
                bits = bits[:exact_bits_needed]
            elif len(bits) < exact_bits_needed:
                # Pad with random bits
                padding = np.random.randint(0, 2, exact_bits_needed - len(bits), dtype=np.uint8)
                bits = np.concatenate([bits, padding])
        else:
            # Random payload
            bits = np.random.randint(0, 2, exact_bits_needed, dtype=np.uint8)

        # 5. Vectorized Embedding
        bits_reshaped = bits.reshape((len(chosen_indices), bit_depth))
        pixels = flat_img[chosen_indices].copy()

        for b in range(bit_depth):
            mask = 255 - (1 << b)
            secret_bit_col = bits_reshaped[:, b]
            pixels &= mask
            pixels |= (secret_bit_col << b)

        flat_img[chosen_indices] = pixels

        # 6. Finalize
        stego_array = flat_img.reshape(img_array.shape)
        psnr = self._calculate_psnr(img_array, stego_array)

        return stego_array, psnr

    def _calculate_psnr(self, original, stego):
        mse = np.mean((original.astype(float) - stego.astype(float)) ** 2)
        if mse == 0: return float('inf')
        return 20 * np.log10(255.0 / np.sqrt(mse))
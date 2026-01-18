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
        capacity_ratio = params.get('capacity_ratio', 0.5)

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

        # 1. Load
        try:
            img = Image.open(cover_path).convert('L')
        except Exception:
            return None, 0

        img_array = np.array(img, dtype=np.uint8)
        flat_img = img_array.flatten()
        total_pixels = flat_img.size

        # FIX 1: Calculate target based on TOTAL pixels (not available pixels)
        target_pixels = int(total_pixels * capacity_ratio)
        target_pixels = max(1, target_pixels)

        # 2. Determine Available Pixels with Fallback
        if edge_threshold > 0:
            available_indices = self._get_complex_areas(img_array, edge_threshold)

            # FIX 2: If not enough edge pixels, expand to all pixels (guaranteed capacity)
            if len(available_indices) < target_pixels:
                # Uncomment for debugging:
                # print(f"[LSB WARNING] Edge threshold {edge_threshold} too high. "
                #       f"Only {len(available_indices)} edge pixels, need {target_pixels}. "
                #       f"Expanding to all pixels.")
                available_indices = np.arange(total_pixels)

            if len(available_indices) == 0:
                available_indices = np.arange(total_pixels)
        else:
            available_indices = np.arange(total_pixels)

        # 3. Strategy Selection
        if strategy == 'random':
            # Randomly select target_pixels from available
            chosen_indices = np.random.choice(
                available_indices,
                target_pixels,
                replace=False
            )

        elif strategy == 'skip':
            # FIX 3: Maintain spatial regularity of skip pattern (no random sampling)
            skipped = available_indices[::step]

            if len(skipped) < target_pixels:
                # Not enough pixels with this step size
                # Uncomment for debugging:
                # print(f"[LSB WARNING] Skip step {step} too large. "
                #       f"Only {len(skipped)} pixels available, need {target_pixels}. "
                #       f"Using all skipped pixels.")
                chosen_indices = skipped
            else:
                # Take first target_pixels from skip pattern (maintains regularity)
                chosen_indices = skipped[:target_pixels]

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

        # FIX 4: Diagnostic output (uncomment to verify capacity is correct)
        # actual_mod_rate = len(chosen_indices) / total_pixels
        # print(f"[LSB] Target: {capacity_ratio:.2%} | Actual: {actual_mod_rate:.2%} | "
        #       f"Edge: {edge_threshold} | Strategy: {strategy} | Step: {step}")

        return stego_array, psnr

    def _calculate_psnr(self, original, stego):
        mse = np.mean((original.astype(float) - stego.astype(float)) ** 2)
        if mse == 0: return float('inf')
        return 20 * np.log10(255.0 / np.sqrt(mse))

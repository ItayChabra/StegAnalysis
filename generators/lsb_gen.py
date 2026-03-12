from generators.base_generator import BaseGenerator
import numpy as np
from PIL import Image


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

        target_pixels = int(total_pixels * capacity_ratio)
        target_pixels = max(1, target_pixels)

        # 2. Strategy Selection
        # FIX: 'edge' is now a standalone first-class strategy rather than a
        # pre-filter that silently narrowed the pixel pool for every other strategy.
        # Previously, any edge_threshold > 0 would filter available_indices before
        # the strategy block ran, so 'random' and 'skip' were both secretly operating
        # on edge pixels only — making edge_threshold an invisible override, not an
        # independent strategy. Each branch below is fully self-contained.

        if strategy == 'edge':
            # Embed only in high-gradient (textured/edge) regions.
            # Falls back to the full image if the threshold is too tight.
            candidate_indices = (
                self._get_complex_areas(img_array, edge_threshold)
                if edge_threshold > 0
                else np.arange(total_pixels)
            )
            if len(candidate_indices) < target_pixels:
                # Not enough edge pixels — widen to full image so capacity is honoured.
                candidate_indices = np.arange(total_pixels)

            chosen_indices = np.random.choice(candidate_indices, target_pixels, replace=False)

        elif strategy == 'random':
            # Uniform random selection across the entire image (no spatial bias).
            chosen_indices = np.random.choice(total_pixels, target_pixels, replace=False)

        elif strategy == 'skip':
            # Spatially regular sub-sampling with a fixed step size.
            skipped = np.arange(0, total_pixels, step)

            if len(skipped) < target_pixels:
                # Step too large — use every skipped pixel, accept lower capacity.
                chosen_indices = skipped
            else:
                chosen_indices = skipped[:target_pixels]

        elif strategy == 'sequential':
            # Simplest baseline: first N pixels in raster order.
            chosen_indices = np.arange(min(target_pixels, total_pixels))

        else:
            # Fallback for unknown strategies.
            chosen_indices = np.random.choice(total_pixels, target_pixels, replace=False)

        # 3. Generate or Prepare Bits
        # FIX: derive exact_bits_needed from the *actual* len(chosen_indices), not
        # the pre-computed target_pixels. The 'skip' strategy can yield fewer pixels
        # than target_pixels when the step is large; using target_pixels here caused
        # a silent reshape crash in step 4.
        exact_bits_needed = len(chosen_indices) * bit_depth

        if message:
            bits = self._text_to_bits(message)
            if len(bits) > exact_bits_needed:
                bits = bits[:exact_bits_needed]
            elif len(bits) < exact_bits_needed:
                padding = np.random.randint(0, 2, exact_bits_needed - len(bits), dtype=np.uint8)
                bits = np.concatenate([bits, padding])
        else:
            bits = np.random.randint(0, 2, exact_bits_needed, dtype=np.uint8)

        # 4. Vectorized Embedding
        bits_reshaped = bits.reshape((len(chosen_indices), bit_depth))
        pixels = flat_img[chosen_indices].copy()

        for b in range(bit_depth):
            mask = 255 - (1 << b)
            secret_bit_col = bits_reshaped[:, b]
            pixels &= mask
            pixels |= (secret_bit_col << b)

        flat_img[chosen_indices] = pixels

        # 5. Finalize
        stego_array = flat_img.reshape(img_array.shape)
        psnr = self._calculate_psnr(img_array, stego_array)

        if output_path:
            Image.fromarray(stego_array).save(output_path)

        return stego_array, psnr

    def _calculate_psnr(self, original, stego):
        mse = np.mean((original.astype(float) - stego.astype(float)) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(255.0 / np.sqrt(mse))
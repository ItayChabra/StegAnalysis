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

    def _load_image_array(self, cover_input):
        """
        Accepts a file path (str), a PIL.Image, or a numpy ndarray.
        Always returns a uint8 numpy array in grayscale.
        Raises ValueError on unsupported types.
        """
        if isinstance(cover_input, np.ndarray):
            # Already an array — ensure correct dtype and 2-D (grayscale).
            arr = cover_input.astype(np.uint8)
            if arr.ndim == 3:
                # Convert RGB/RGBA to grayscale by taking the first channel,
                # matching the convert('L') behaviour used for file paths.
                arr = arr[:, :, 0]
            return arr

        if isinstance(cover_input, Image.Image):
            img = cover_input.convert('L') if cover_input.mode != 'L' else cover_input
            return np.array(img, dtype=np.uint8)

        if isinstance(cover_input, str):
            img = Image.open(cover_input).convert('L')
            return np.array(img, dtype=np.uint8)

        raise ValueError(
            f"cover_input must be a file path (str), PIL.Image, or np.ndarray. "
            f"Got: {type(cover_input)}"
        )

    def run(self, cover_input, output_path, **params):
        """
        Implementation of the BaseGenerator interface.

        cover_input: str (file path), PIL.Image, or np.ndarray.
        """
        strategy       = params.get('strategy', 'random')
        step           = params.get('step', 1)
        bit_depth      = params.get('bit_depth', 1)
        edge_threshold = params.get('edge_threshold', 0)
        message        = params.get('message', None)
        capacity_ratio = params.get('capacity_ratio', 0.5)

        return self.embed(cover_input, output_path,
                          message=message,
                          strategy=strategy,
                          step=step,
                          bit_depth=bit_depth,
                          edge_threshold=edge_threshold,
                          capacity_ratio=capacity_ratio)

    def embed(self, cover_input, output_path, message=None,
              strategy='random', step=1,
              bit_depth=1, edge_threshold=0,
              capacity_ratio=0.5):
        """
        cover_input: str (file path), PIL.Image, or np.ndarray.
                     Accepts all three so callers never need to write a temp
                     file just to hand an image back to this method.
        """
        # 1. Load — accept path, PIL.Image, or ndarray.
        try:
            img_array = self._load_image_array(cover_input)
        except Exception:
            return None, 0

        flat_img     = img_array.flatten()
        total_pixels = flat_img.size

        target_pixels = max(1, int(total_pixels * capacity_ratio))

        # 2. Strategy Selection — each branch is fully self-contained.
        if strategy == 'edge':
            candidate_indices = (
                self._get_complex_areas(img_array, edge_threshold)
                if edge_threshold > 0
                else np.arange(total_pixels)
            )
            if len(candidate_indices) < target_pixels:
                candidate_indices = np.arange(total_pixels)
            chosen_indices = np.random.choice(candidate_indices, target_pixels, replace=False)

        elif strategy == 'random':
            chosen_indices = np.random.choice(total_pixels, target_pixels, replace=False)

        elif strategy == 'skip':
            skipped = np.arange(0, total_pixels, step)
            if len(skipped) < target_pixels:
                chosen_indices = skipped
            else:
                chosen_indices = skipped[:target_pixels]

        elif strategy == 'sequential':
            chosen_indices = np.arange(min(target_pixels, total_pixels))

        else:
            chosen_indices = np.random.choice(total_pixels, target_pixels, replace=False)

        # 3. Generate or Prepare Bits
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
            mask           = 255 - (1 << b)
            secret_bit_col = bits_reshaped[:, b]
            pixels        &= mask
            pixels        |= (secret_bit_col << b)

        flat_img[chosen_indices] = pixels

        # 5. Finalize
        stego_array = flat_img.reshape(img_array.shape)
        psnr        = self._calculate_psnr(img_array, stego_array)

        if output_path:
            Image.fromarray(stego_array).save(output_path)

        return stego_array, psnr

    def _calculate_psnr(self, original, stego):
        mse = np.mean((original.astype(float) - stego.astype(float)) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(255.0 / np.sqrt(mse))
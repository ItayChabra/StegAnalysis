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
        # Encode to bytes (UTF-8)
        b = text.encode('utf-8')
        # Convert to numpy array of uint8
        arr = np.frombuffer(b, dtype=np.uint8)
        # Unpack bits (creates a long array of 0s and 1s)
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
        message = params.get('message', None)  # <--- הוספנו תמיכה בפרמטר message

        return self.embed(cover_path, output_path,
                          message=message,
                          strategy=strategy,
                          step=step,
                          bit_depth=bit_depth,
                          edge_threshold=edge_threshold)

    def embed(self, cover_path, output_path, message=None,
              strategy='random', step=1,
              bit_depth=1, edge_threshold=0):

        # 1. Load & Resize
        try:
            img = Image.open(cover_path).convert('L')
        except Exception as e:
            return None, 0

        if img.size != self.target_size:
            img = img.resize(self.target_size, Image.Resampling.LANCZOS)

        img_array = np.array(img, dtype=np.uint8)
        total_pixels = img_array.size

        # 2. Determine Available Pixels
        if edge_threshold > 0:
            available_indices = self._get_complex_areas(img_array, edge_threshold)
            if len(available_indices) == 0:
                available_indices = np.arange(total_pixels)
        else:
            available_indices = np.arange(total_pixels)

        # 3. Handle Message (Random vs Real Text)
        max_capacity_bits = len(available_indices) * bit_depth

        if message:
            # === Real Message Logic ===
            bits = self._text_to_bits(message)

            # Check if message fits
            if len(bits) > max_capacity_bits:
                print(f"[WARN] Message too long! Truncating from {len(bits)} bits to {max_capacity_bits} bits.")
                bits = bits[:max_capacity_bits]

            # Note: For real stego, we usually store the length in the first 32 bits.
            # Here we just embed raw data for simplicity.
        else:
            # === Random Training Noise Logic ===
            target_bits = int(max_capacity_bits * 0.5)
            bits = np.random.randint(0, 2, target_bits, dtype=np.uint8)

        # 4. Strategy Selection
        num_bits_needed = len(bits)
        pixels_needed = int(np.ceil(num_bits_needed / bit_depth))

        if strategy == 'random':
            if pixels_needed > len(available_indices): pixels_needed = len(available_indices)
            chosen_indices = np.random.choice(available_indices, pixels_needed, replace=False)
        elif strategy == 'sequential':
            chosen_indices = available_indices[:pixels_needed]
        elif strategy == 'skip':
            chosen_indices = available_indices[::step][:pixels_needed]
        else:
            chosen_indices = available_indices[:pixels_needed]

        # 5. Vectorized Embedding (With fix)
        stego_array = img_array.flatten()
        bit_pointer = 0

        for idx in chosen_indices:
            if bit_pointer >= len(bits): break

            pixel_val = int(stego_array[idx])

            for b in range(bit_depth):
                if bit_pointer >= len(bits): break
                secret_bit = int(bits[bit_pointer])

                # Fixed Mask Logic
                mask = 255 - (1 << b)
                pixel_val &= mask
                pixel_val |= (secret_bit << b)
                bit_pointer += 1

            stego_array[idx] = np.uint8(pixel_val)

        # 6. Finalize
        stego_array = stego_array.reshape(img_array.shape)
        psnr = self._calculate_psnr(img_array, stego_array)

        return stego_array, psnr

    def _calculate_psnr(self, original, stego):
        mse = np.mean((original.astype(float) - stego.astype(float)) ** 2)
        if mse == 0: return float('inf')
        return 20 * np.log10(255.0 / np.sqrt(mse))
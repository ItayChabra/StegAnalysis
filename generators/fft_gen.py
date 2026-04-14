"""
FFT Steganography Generator — fully vectorized.

The magnitude quantization loop that previously iterated over every chosen
frequency component in Python is replaced by NumPy fancy-indexing and
array arithmetic, giving the same result in a fraction of the time.

Genome parameters:
    gen_type:       'fft'
    capacity_ratio: float  — fraction of eligible frequency components to modify
    freq_band:      'low' | 'mid' | 'high'
    strength:       float 2.0–20.0 — quantization step in magnitude domain
"""

import numpy as np
from PIL import Image

from generators.base_generator import BaseGenerator


class FFTGenerator(BaseGenerator):
    """
    Global frequency-domain steganography via 2D FFT magnitude embedding.

    Fully vectorized: frequency-band mask construction, component selection,
    and magnitude quantization all use NumPy array operations.

    Frequency bands (fraction of max radius from DC centre):
        'low'  :  0 – 15 %
        'mid'  : 15 – 45 %   (default)
        'high' : 45 – 75 %
    """

    _BAND_LIMITS = {
        'low':  (0.00, 0.15),
        'mid':  (0.15, 0.45),
        'high': (0.45, 0.75),
    }

    # ------------------------------------------------------------------ utils

    def _load_image_array(self, cover_input):
        if isinstance(cover_input, np.ndarray):
            arr = cover_input.astype(np.uint8)
            return arr[:, :, 0] if arr.ndim == 3 else arr
        if isinstance(cover_input, Image.Image):
            img = cover_input if cover_input.mode == 'L' else cover_input.convert('L')
            return np.array(img, dtype=np.uint8)
        if isinstance(cover_input, str):
            return np.array(Image.open(cover_input).convert('L'), dtype=np.uint8)
        raise ValueError(f"Unsupported cover_input type: {type(cover_input)}")

    def _text_to_bits(self, text):
        return np.unpackbits(np.frombuffer(text.encode('utf-8'), dtype=np.uint8))

    def _calculate_psnr(self, original, stego):
        mse = np.mean((original.astype(float) - stego.astype(float)) ** 2)
        return float('inf') if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))

    def _build_band_mask(self, h, w, freq_band):
        lo, hi     = self._BAND_LIMITS.get(freq_band, self._BAND_LIMITS['mid'])
        cy, cx     = h // 2, w // 2
        max_radius = np.sqrt(cy ** 2 + cx ** 2)
        y, x  = np.mgrid[0:h, 0:w]
        dist  = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)

        mask = (dist >= lo * max_radius) & (dist < hi * max_radius)
        # Exclude the DC component (center) to avoid global brightness shifts
        mask[cy, cx] = False
        return mask

    def _shifted_conjugate_partner(self, rows, cols, h, w):
        """Calculates the conjugate symmetric partner for coordinates in a shifted FFT."""
        un_rows = (rows + h // 2) % h
        un_cols = (cols + w // 2) % w
        partner_un_rows = (-un_rows) % h
        partner_un_cols = (-un_cols) % w
        partner_rows = (partner_un_rows + h // 2) % h
        partner_cols = (partner_un_cols + w // 2) % w
        return partner_rows, partner_cols

    # ------------------------------------------------------------------ interface

    def run(self, cover_input, output_path, **params):
        return self.embed(
            cover_input, output_path,
            capacity_ratio = params.get('capacity_ratio', 0.3),
            freq_band      = params.get('freq_band',      'mid'),
            strength       = params.get('strength',       8.0),
            message        = params.get('message',        None),
        )

    def embed(self, cover_input, output_path,
              capacity_ratio=0.3, freq_band='mid',
              strength=8.0, message=None):

        try:
            img_array = self._load_image_array(cover_input)
        except Exception:
            return None, 0

        h, w = img_array.shape

        # ---- Forward FFT ------------------------------------------------
        fft_raw     = np.fft.fft2(img_array.astype(float))
        fft_shifted = np.fft.fftshift(fft_raw)
        magnitude   = np.abs(fft_shifted)
        phase       = np.angle(fft_shifted)

        # ---- Frequency-band mask ----------------------------------------
        mask = self._build_band_mask(h, w, freq_band)
        eligible_rows, eligible_cols = np.where(mask)

        # Calculate conjugate partners to ensure symmetry
        partner_rows, partner_cols = self._shifted_conjugate_partner(
            eligible_rows, eligible_cols, h, w)

        # Keep only one half of the symmetric pairs to avoid double-modifying
        flat = eligible_rows * w + eligible_cols
        partner_flat = partner_rows * w + partner_cols
        keep = flat <= partner_flat

        eligible_rows = eligible_rows[keep]
        eligible_cols = eligible_cols[keep]
        partner_rows = partner_rows[keep]
        partner_cols = partner_cols[keep]

        n_unique = len(eligible_rows)
        if n_unique == 0:
            return None, 0

        target_count = max(1, int(n_unique * capacity_ratio))
        total_bits   = target_count

        # ---- Prepare bits -----------------------------------------------
        if message:
            bits = self._text_to_bits(message)
            if len(bits) > total_bits:
                bits = bits[:total_bits]
            elif len(bits) < total_bits:
                bits = np.concatenate(
                    [bits, np.random.randint(0, 2, total_bits - len(bits), dtype=np.uint8)])
        else:
            bits = np.random.randint(0, 2, total_bits, dtype=np.uint8)

        # ---- Sub-sample within band — vectorized ------------------------
        chosen_idx  = np.random.choice(n_unique, target_count, replace=False)
        rows        = eligible_rows[chosen_idx]
        cols        = eligible_cols[chosen_idx]
        pair_rows   = partner_rows[chosen_idx]
        pair_cols   = partner_cols[chosen_idx]

        # ---- Vectorized magnitude quantization --------------------------
        # Scale strength so it survives the 1/N scaling of the inverse FFT
        # np.sqrt(h * w) provides a dynamic scale factor (e.g., 256 for a 256x256 image)
        effective_strength = strength * np.sqrt(h * w)

        # No Python loop — all target_count components are processed at once.
        mags = magnitude[rows, cols].copy()                         # (target_count,)

        q    = np.round(mags / effective_strength).astype(np.int64)

        # Enforce parity: bit=1 → q odd; bit=0 → q even.
        # wrong_parity is True wherever the current parity doesn't match the bit.
        current_even   = (q % 2 == 0)
        want_odd       = bits.astype(bool)                          # bit=1 wants odd
        wrong_parity   = current_even == want_odd                   # XOR logic
        q[wrong_parity] += 1

        modified_mag = magnitude.copy()
        new_mag = np.maximum(0.0, q * effective_strength)
        modified_mag[rows, cols] = new_mag

        # Apply exact same modification to conjugate partners
        non_self = (rows != pair_rows) | (cols != pair_cols)
        modified_mag[pair_rows[non_self], pair_cols[non_self]] = new_mag[non_self]

        # ---- Inverse FFT ------------------------------------------------
        modified_fft_shifted = modified_mag * np.exp(1j * phase)
        modified_fft         = np.fft.ifftshift(modified_fft_shifted)
        stego_float          = np.real(np.fft.ifft2(modified_fft))

        # Proper mathematical rounding before casting to uint8
        stego_array          = np.clip(np.rint(stego_float), 0, 255).astype(np.uint8)

        psnr = self._calculate_psnr(img_array, stego_array)

        if output_path:
            Image.fromarray(stego_array).save(output_path)

        return stego_array, psnr
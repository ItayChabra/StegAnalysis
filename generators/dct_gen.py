"""
DCT Steganography Generator — fully vectorized.

All 8×8 blocks are transformed, quantized, and inverse-transformed in a
single batch of NumPy/SciPy calls with no Python loops over pixels or blocks.

Genome parameters:
    gen_type:         'dct'
    capacity_ratio:   float  — fraction of 8×8 blocks to modify
    coeff_selection:  'mid' | 'low_mid' | 'random'
    strength:         float 1.0–8.0 — quantization step size
"""

import random
import numpy as np
from PIL import Image

from generators.base_generator import BaseGenerator


class DCTGenerator(BaseGenerator):
    """
    Frequency-domain steganography using 8×8 block DCT.

    Vectorized implementation: the image is reshaped into all blocks at once,
    a single batched 2D DCT is applied across every block simultaneously, bits
    are embedded via NumPy array operations (no Python loops), and the inverse
    DCT reconstructs the full image in one pass.

    Bit embedding rule:
        bit = 1  →  quantized coefficient is ODD  multiple of `strength`
        bit = 0  →  quantized coefficient is EVEN multiple of `strength`
    """

    # Standard JPEG zigzag scan order — index 0 is the DC component (always skipped).
    ZIGZAG = [
        (0,0),(0,1),(1,0),(2,0),(1,1),(0,2),(0,3),(1,2),(2,1),(3,0),
        (4,0),(3,1),(2,2),(1,3),(0,4),(0,5),(1,4),(2,3),(3,2),(4,1),
        (5,0),(6,0),(5,1),(4,2),(3,3),(2,4),(1,5),(0,6),(0,7),(1,6),
        (2,5),(3,4),(4,3),(5,2),(6,1),(7,0),(7,1),(6,2),(5,3),(4,4),
        (3,5),(2,6),(1,7),(2,7),(3,6),(4,5),(5,4),(6,3),(7,2),(7,3),
        (6,4),(5,5),(4,6),(3,7),(4,7),(5,6),(6,5),(7,4),(7,5),(6,6),
        (5,7),(6,7),(7,6),(7,7),
    ]

    _COEFF_RANGES = {
        'mid':     list(range(5,  25)),
        'low_mid': list(range(2,  14)),
        'random':  list(range(5,  30)),
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

    def _get_coeff_positions(self, coeff_selection):
        indices = self._COEFF_RANGES.get(coeff_selection, self._COEFF_RANGES['mid'])
        if coeff_selection == 'random':
            indices = random.sample(indices, min(10, len(indices)))
        return [self.ZIGZAG[i] for i in indices if i < len(self.ZIGZAG)]

    # ------------------------------------------------------------------ interface

    def run(self, cover_input, output_path, **params):
        return self.embed(
            cover_input, output_path,
            capacity_ratio  = params.get('capacity_ratio',  0.5),
            coeff_selection = params.get('coeff_selection', 'mid'),
            strength        = params.get('strength',        3.0),
            message         = params.get('message',         None),
        )

    def embed(self, cover_input, output_path,
              capacity_ratio=0.5, coeff_selection='mid',
              strength=3.0, message=None):

        try:
            img_array = self._load_image_array(cover_input)
        except Exception:
            return None, 0

        h, w = img_array.shape

        # Pad to a multiple of 8 so every block is complete.
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        padded = np.pad(img_array, ((0, pad_h), (0, pad_w)), mode='edge').astype(np.float32)
        ph, pw   = padded.shape
        blocks_h = ph // 8
        blocks_w = pw // 8
        n_blocks = blocks_h * blocks_w

        # ---- Batch 2D DCT via SciPy dctn --------------------------------
        # Reshape to (blocks_h, blocks_w, 8, 8) — all blocks at once.
        # axis ordering: row-of-blocks, col-of-blocks, block-row, block-col
        try:
            from scipy.fft import dctn, idctn
        except ImportError:
            from scipy.fftpack import dctn, idctn

        blocks    = padded.reshape(blocks_h, 8, blocks_w, 8).transpose(0, 2, 1, 3)
        # shape: (blocks_h, blocks_w, 8, 8)
        dct_blocks = dctn(blocks, axes=(-2, -1), norm='ortho')
        # dct_blocks shape: (blocks_h, blocks_w, 8, 8)

        # ---- Select which blocks and which coefficients to embed in ----
        coeff_positions = self._get_coeff_positions(coeff_selection)  # list of (r,c) tuples
        bits_per_block  = len(coeff_positions)
        target_blocks   = max(1, int(n_blocks * capacity_ratio))
        total_bits      = target_blocks * bits_per_block

        # Generate / prepare bits.
        if message:
            bits = self._text_to_bits(message)
            if len(bits) > total_bits:
                bits = bits[:total_bits]
            elif len(bits) < total_bits:
                bits = np.concatenate(
                    [bits, np.random.randint(0, 2, total_bits - len(bits), dtype=np.uint8)])
        else:
            bits = np.random.randint(0, 2, total_bits, dtype=np.uint8)

        # Randomly select which blocks receive payload — vectorized.
        block_indices = np.random.choice(n_blocks, target_blocks, replace=False)
        # Convert flat block indices to (row, col) block coordinates.
        block_rows = block_indices // blocks_w
        block_cols = block_indices % blocks_w

        # ---- Vectorized quantization embedding --------------------------
        # Process each coefficient position across ALL selected blocks at once.
        bits_2d = bits.reshape(target_blocks, bits_per_block)  # (n_selected, n_coeffs)

        for coeff_idx, (cr, cc) in enumerate(coeff_positions):
            # Extract this coefficient from every selected block in one shot.
            coeffs = dct_blocks[block_rows, block_cols, cr, cc]   # shape: (target_blocks,)
            b      = bits_2d[:, coeff_idx].astype(np.float32)

            # Quantize: q = round(coeff / strength)
            q = np.round(coeffs / strength).astype(np.int32)

            # Enforce parity: bit=1 → q must be odd; bit=0 → q must be even.
            # If q is even and bit=1, or q is odd and bit=0, increment q.
            wrong_parity = (q % 2 == 0).astype(np.int32) * b.astype(np.int32) + \
                           (q % 2 != 0).astype(np.int32) * (1 - b.astype(np.int32))
            q = q + wrong_parity.astype(np.int32)

            dct_blocks[block_rows, block_cols, cr, cc] = q * strength

        # ---- Batch inverse DCT ------------------------------------------
        idct_blocks = idctn(dct_blocks, axes=(-2, -1), norm='ortho')

        # Reconstruct the padded image with proper mathematical rounding
        stego_padded = idct_blocks.transpose(0, 2, 1, 3).reshape(ph, pw)
        stego_array = np.clip(np.rint(stego_padded[:h, :w]), 0, 255).astype(np.uint8)

        psnr = self._calculate_psnr(img_array, stego_array)

        if output_path:
            Image.fromarray(stego_array).save(output_path)

        return stego_array, psnr
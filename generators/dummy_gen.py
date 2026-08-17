"""
dummy_gen.py — Minimal generator that exists only to prove the plug-in
interface for Chapter 10.2's supportability/extensibility NFR test
(tests/nonfunctional/test_extensibility.py). Not used by training or the demo.
"""

import numpy as np
from PIL import Image

from generators.base_generator import BaseGenerator


class DummyGen(BaseGenerator):
    """Minimal BaseGenerator implementation: flips the LSB of every pixel."""

    def run(self, cover_input, output_path, **params):
        if isinstance(cover_input, np.ndarray):
            arr = cover_input.astype(np.uint8)
        elif isinstance(cover_input, Image.Image):
            arr = np.array(cover_input.convert('L'), dtype=np.uint8)
        else:
            arr = np.array(Image.open(cover_input).convert('L'), dtype=np.uint8)

        stego = (arr ^ 1).astype(np.uint8)
        if output_path:
            Image.fromarray(stego, mode='L').save(output_path)
        return stego, 0.0

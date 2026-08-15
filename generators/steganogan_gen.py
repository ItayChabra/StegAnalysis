"""
steganogan_gen.py — GAN-based steganography generator (SteganoGAN).

Wraps the vendored DAI-Lab SteganoGAN encoder (generators/steganogan_src) as a
BaseGenerator so the EA/training pipeline and the demo can treat learned,
GAN-embedded stego like any other method.

How it differs from the classical generators:
  - The perturbation is produced by a trained convolutional encoder, not a
    hand-designed rule. Given a cover and a random bit payload of shape
    (1, data_depth, H, W), the encoder outputs a full-resolution residual image
    that hides the payload while staying close to the cover.
  - Like adaptive_gen (S-UNIWARD), this is a NON-RECOVERABLE adversarial
    generator: run() embeds random bits to produce statistically realistic
    stego for training the detector. It exposes no embed_payload/extract_payload
    recoverable codec (SteganoGAN's own decode is probabilistic).

Pipeline conventions (matched to the other generators):
  - Accepts a file path (str), PIL.Image, or np.ndarray as cover_input.
  - Operates on a single-channel (grayscale) cover — the cover is replicated to
    RGB for the encoder, then the RGB stego is folded back to luminance so the
    returned array is 2-D uint8, exactly like lsb/dct/fft/adaptive.
  - Runs at the cover's own resolution, exactly like lsb/dct/fft/adaptive — the
    encoder is fully convolutional (Conv2d/BatchNorm2d/LeakyReLU only), so no
    fixed input size is required. Every training caller (train_hybrid.py,
    validate.py, evaluate.py, finetune.py) already crops covers to 256x256
    before calling any generator, so this never mattered for training; a
    forced resize here only broke the live embed API, where the cover keeps
    its uploaded size and the caller expects stego.shape == cover.shape.
  - Returns (stego_2d_uint8, psnr), same shape as the input.

Weights:
  - Loads a plain state_dict checkpoint (default: repo-root steganogan_dense.pth)
    produced by scripts/convert_steganogan_weights.py from the pretrained
    dense.steg. data_depth / hidden_size are read from the checkpoint; the
    pretrained dense model is data_depth=8, hidden_size=32.
  - data_depth is fixed by the pretrained conv shapes and cannot be changed
    without retraining, so 'capacity' is informational here.
"""

import threading
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from generators.base_generator import BaseGenerator
from generators.steganogan_src import BasicEncoder, ResidualEncoder, DenseEncoder

# Luminance weights (ITU-R 601), matching PIL's "L" conversion.
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)

_ENCODER_CLASSES = {
    'BasicEncoder': BasicEncoder,
    'ResidualEncoder': ResidualEncoder,
    'DenseEncoder': DenseEncoder,
}

_DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / 'steganogan_dense.pth'


class SteganoGANGenerator(BaseGenerator):
    def __init__(self, weights_path=None, device=None):
        self.weights_path = Path(weights_path) if weights_path else _DEFAULT_WEIGHTS
        self._device = device
        # Lazily built on first use so importing the module (and constructing the
        # UnifiedGenerator dispatch table) never touches disk or CUDA.
        self._encoder = None
        self._data_depth = None
        # The training/finetune loops build pairs across worker threads, so the
        # one-time load must be race-free.
        self._load_lock = threading.Lock()

    # ------------------------------------------------------------------ setup
    def _ensure_loaded(self):
        if self._encoder is not None:
            return
        with self._load_lock:
            if self._encoder is not None:      # another thread won the race
                return
            if self._device is None:
                self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

            ckpt = torch.load(self.weights_path, map_location='cpu',
                              weights_only=False)
            enc_cls = _ENCODER_CLASSES.get(ckpt.get('encoder_class', 'DenseEncoder'),
                                           DenseEncoder)
            data_depth = int(ckpt['data_depth'])
            encoder = enc_cls(data_depth=data_depth,
                              hidden_size=int(ckpt['hidden_size']))
            encoder.load_state_dict(ckpt['encoder'])
            encoder.eval().to(self._device)
            self._data_depth = data_depth
            self._encoder = encoder             # set last: publishes readiness

    # ---------------------------------------------------------------- loading
    def _load_gray(self, cover_input):
        """Return a 2-D uint8 grayscale array at the cover's own resolution
        (accepts path/PIL/ndarray). No resize — see the module docstring."""
        if isinstance(cover_input, np.ndarray):
            arr = cover_input.astype(np.uint8)
            img = Image.fromarray(arr[:, :, 0] if arr.ndim == 3 else arr, mode='L')
        elif isinstance(cover_input, Image.Image):
            img = cover_input if cover_input.mode == 'L' else cover_input.convert('L')
        elif isinstance(cover_input, str):
            img = Image.open(cover_input).convert('L')
        else:
            raise ValueError(
                f"cover_input must be a file path (str), PIL.Image, or np.ndarray. "
                f"Got: {type(cover_input)}"
            )
        return np.array(img, dtype=np.uint8)

    # -------------------------------------------------------------- interface
    def run(self, cover_input, output_path, **params):
        """
        Params:
            capacity : informational only — data_depth is fixed by the weights.
        """
        try:
            gray = self._load_gray(cover_input)
        except Exception:
            return None, 0

        try:
            self._ensure_loaded()
        except Exception:
            return None, 0

        # Grayscale cover → RGB in [-1, 1], (1, 3, H, W).
        cover = torch.from_numpy(gray.astype(np.float32) / 127.5 - 1.0)
        cover = cover.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(self._device)

        H, W = gray.shape
        payload = torch.zeros((1, self._data_depth, H, W),
                              device=self._device).random_(0, 2)

        with torch.no_grad():
            stego = self._encoder(cover, payload).clamp(-1.0, 1.0)

        # RGB stego → uint8 → luminance grayscale, matching the cover's "L" space.
        rgb = ((stego[0].permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5)
        rgb = np.clip(rgb, 0, 255)
        stego_gray = np.clip(rgb @ _LUMA, 0, 255).astype(np.uint8)

        psnr = self._calculate_psnr(gray, stego_gray)

        if output_path:
            Image.fromarray(stego_gray).save(output_path)

        return stego_gray, psnr

    @staticmethod
    def _calculate_psnr(original, stego):
        mse = np.mean((original.astype(float) - stego.astype(float)) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(255.0 / np.sqrt(mse))
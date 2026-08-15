"""
Vendored SteganoGAN network definitions (encoder / decoder / critic).

Source: DAI-Lab/SteganoGAN — https://github.com/DAI-Lab/SteganoGAN
Licence: MIT (see the upstream LICENSE file). Copyright (c) 2019 MIT Data To AI Lab.

Why vendored instead of `pip install steganogan`:
    The upstream package pins `torch>=1.0.0,<2.0.0` (plus numpy<1.16, scipy<1.2,
    Pillow<8, reedsolo==0.3). Installing it would downgrade this project's
    torch 2.5.x environment. The networks themselves use only standard layers
    (Conv2d / BatchNorm2d / LeakyReLU / Tanh / cat) and run unmodified on
    modern PyTorch, so we vendor just the three network modules and load the
    pretrained weights as plain state_dicts (see scripts/convert_steganogan_weights.py).

Only the encoder is used by generators/steganogan_gen.py. The decoder and
critic are vendored for weight conversion and possible future fine-tuning.
"""

from generators.steganogan_src.encoders import (
    BasicEncoder, ResidualEncoder, DenseEncoder,
)
from generators.steganogan_src.decoders import BasicDecoder, DenseDecoder
from generators.steganogan_src.critics import BasicCritic

__all__ = [
    'BasicEncoder', 'ResidualEncoder', 'DenseEncoder',
    'BasicDecoder', 'DenseDecoder', 'BasicCritic',
]
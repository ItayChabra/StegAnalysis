"""
Convert an upstream SteganoGAN ``.steg`` pickle into a plain state_dict checkpoint.

The upstream ``.steg`` files are full pickled ``SteganoGAN`` objects
(``torch.save(self)``) whose class graph lives under the ``steganogan.*`` module
path and whose package pins torch<2.0. We never want that coupling at runtime,
so this one-off script:

    1. Registers the vendored networks (generators/steganogan_src) into
       ``sys.modules`` under the ``steganogan.*`` names the pickle references,
       plus a bare ``SteganoGAN`` shim and stubs for optional deps
       (reedsolo / tqdm) that the real package imports but we don't need.
    2. ``torch.load``s the pickle once.
    3. Extracts ``data_depth`` / ``hidden_size`` and the encoder + decoder
       ``state_dict``s.
    4. Re-saves them as a plain dict that the generator loads by constructing a
       fresh vendored network and calling ``load_state_dict`` — no pickle, no
       version pin.

Usage:
    python scripts/convert_steganogan_weights.py path/to/dense.steg \
        --arch dense --out steganogan_dense.pth
"""

import argparse
import sys
import types
from pathlib import Path

import torch

# Make the project root importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generators.steganogan_src import encoders, decoders, critics  # noqa: E402


def _install_import_aliases():
    """Point the pickle's ``steganogan.*`` references at our vendored code."""
    # Stub optional deps the upstream package imports at module load but that we
    # don't need for weight extraction.
    for name in ('reedsolo', 'tqdm', 'imageio'):
        if name not in sys.modules:
            stub = types.ModuleType(name)
            if name == 'reedsolo':
                stub.RSCodec = object          # only referenced, never called here
            if name == 'tqdm':
                stub.tqdm = lambda x, *a, **k: x
            sys.modules[name] = stub

    pkg = types.ModuleType('steganogan')
    pkg.__path__ = []                          # mark as a package
    sys.modules['steganogan'] = pkg
    sys.modules['steganogan.encoders'] = encoders
    sys.modules['steganogan.decoders'] = decoders
    sys.modules['steganogan.critics'] = critics

    # The top-level SteganoGAN object only needs to unpickle its __dict__; a bare
    # class with the right qualified name is enough (its __init__ is never run).
    models = types.ModuleType('steganogan.models')

    class SteganoGAN(object):
        pass

    models.SteganoGAN = SteganoGAN
    sys.modules['steganogan.models'] = models

    # The pretrained pickle carries trained Adam optimizer state whose legacy
    # layout modern torch's Optimizer.__setstate__ rejects. We discard the
    # optimizers anyway, so make their unpickle a tolerant dict-restore.
    from torch.optim.optimizer import Optimizer
    Optimizer.__setstate__ = lambda self, state: self.__dict__.update(state)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('steg_path', help='Path to the upstream .steg pickle')
    ap.add_argument('--arch', default='dense',
                    help='Architecture label to record (basic/residual/dense)')
    ap.add_argument('--out', default='steganogan_dense.pth',
                    help='Destination .pth for the plain checkpoint')
    args = ap.parse_args()

    _install_import_aliases()

    print(f'Loading {args.steg_path} ...')
    obj = torch.load(args.steg_path, map_location='cpu', weights_only=False)

    data_depth = int(getattr(obj, 'data_depth', getattr(obj.encoder, 'data_depth')))
    hidden_size = int(getattr(obj.encoder, 'hidden_size'))
    enc_cls = type(obj.encoder).__name__
    dec_cls = type(obj.decoder).__name__
    print(f'  encoder={enc_cls} decoder={dec_cls} '
          f'data_depth={data_depth} hidden_size={hidden_size}')

    ckpt = {
        'arch': args.arch,
        'encoder_class': enc_cls,
        'decoder_class': dec_cls,
        'data_depth': data_depth,
        'hidden_size': hidden_size,
        'encoder': obj.encoder.state_dict(),
        'decoder': obj.decoder.state_dict(),
        'source': 'DAI-Lab/SteganoGAN pretrained (MIT); converted to plain state_dict',
    }
    torch.save(ckpt, args.out)
    n_enc = sum(v.numel() for v in ckpt['encoder'].values())
    print(f'Saved {args.out}  (encoder params: {n_enc:,})')


if __name__ == '__main__':
    main()
from generators.lsb_gen import LSBGenerator
from generators.dct_gen import DCTGenerator
from generators.fft_gen import FFTGenerator


class UnifiedGenerator:
    """
    Central hub for all steganography generators.

    Supported gen_type values:
        'lsb'  — Spatial LSB embedding  (LSBGenerator)
        'dct'  — Block DCT embedding     (DCTGenerator)
        'fft'  — Global FFT embedding    (FFTGenerator)

    generate_stego() accepts a file path (str), PIL.Image, or np.ndarray as
    cover_input, so callers never need to write a temporary file to disk.
    """

    def __init__(self):
        self.generators = {
            'lsb': LSBGenerator(),
            'dct': DCTGenerator(),
            'fft': FFTGenerator(),
        }

    def generate_stego(self, cover_input, output_path, config):
        """
        Args:
            cover_input:  str (file path), PIL.Image, or np.ndarray.
                          All three are forwarded directly to the generator,
                          eliminating the temp-file round-trip in the training loop.
            output_path:  Destination path for the stego image, or None to skip
                          saving to disk (recommended during training).
            config:       Dictionary containing 'gen_type' and generator parameters.

        Returns:
            (stego_array, metric) — numpy array + PSNR float, or (None, 0) on failure.
        """
        gen_type = config.get('gen_type', 'lsb')

        if gen_type in self.generators:
            return self.generators[gen_type].run(cover_input, output_path, **config)

        print(f"[UnifiedGenerator] Unknown generator type: '{gen_type}'. "
              f"Available: {list(self.generators.keys())}")
        return None, 0
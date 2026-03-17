import random
from generators.lsb_gen import LSBGenerator
# from generators.dct_gen import DCTGenerator


class UnifiedGenerator:
    """
    Central hub for all generators.

    generate_stego() now accepts a file path (str), PIL.Image, or np.ndarray
    as cover_input, so callers no longer need to write a temporary file to disk
    before calling this method.
    """

    def __init__(self):
        self.generators = {
            'lsb': LSBGenerator(),
            # 'dct': DCTGenerator(),
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
        """
        gen_type = config.get('gen_type', 'lsb')

        if gen_type in self.generators:
            return self.generators[gen_type].run(cover_input, output_path, **config)

        print(f"Unknown generator type: {gen_type}")
        return None, 0
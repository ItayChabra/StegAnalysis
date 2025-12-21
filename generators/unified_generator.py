import random
from generators.lsb_gen import LSBGenerator


# from generators.dct_gen import DCTGenerator

class UnifiedGenerator:
    """
    Central hub for all generators.
    """

    def __init__(self):
        self.generators = {
            'lsb': LSBGenerator(),
            # 'dct': DCTGenerator(),
        }

    def generate_stego(self, cover_path, output_path, config):
        """
        Args:
            config: Dictionary containing 'gen_type' and parameters.
        """
        gen_type = config.get('gen_type', 'lsb')

        if gen_type in self.generators:
            return self.generators[gen_type].run(cover_path, output_path, **config)
        else:
            print(f"Unknown generator type: {gen_type}")
            return None, 0
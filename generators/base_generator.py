from abc import ABC, abstractmethod


class BaseGenerator(ABC):
    """
    Abstract Interface for ALL generators.
    Ensures every generator (LSB, DCT, GAN) has a standard 'run' method.
    """

    @abstractmethod
    def run(self, cover_path, output_path, **params):
        """
        Must be implemented by every generator.
        Args:
            cover_path: Path to the clean image.
            output_path: Where to save the stego image (can be None).
            **params: Dictionary of specific hyperparameters (e.g. step, threshold).
        Returns:
            stego_array: Numpy array of the result.
            metrics: Tuple or dict of metrics (e.g. psnr).
        """
        pass
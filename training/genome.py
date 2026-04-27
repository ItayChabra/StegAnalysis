"""
genome.py — Genome utility functions shared by the evolutionary manager,
            batch builder, and training loop.
"""

import torch
from training.config import (
    LOG_FFT_SCALE,
    LOW_CAPACITY_THRESHOLD,
)


# ── FFT feature extraction ────────────────────────────────────────────────────

def compute_log_fft(spatial_tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert a (1, H, W) spatial tensor to a (1, H, W) log-magnitude FFT map.

    Uses a fixed global scale constant so absolute payload differences between
    cover and stego images are preserved across all training batches.
    """
    fft_complex  = torch.fft.fft2(spatial_tensor)
    fft_shifted  = torch.fft.fftshift(fft_complex, dim=(-2, -1))
    log_magnitude = torch.log1p(torch.abs(fft_shifted))
    return log_magnitude / LOG_FFT_SCALE


# ── Genome classification helpers ─────────────────────────────────────────────

def get_niche(genome: dict) -> str:
    """
    Map a genome to its niche string.

    LSB genomes → 'lsb_<strategy>'   (e.g. 'lsb_edge')
    FFT genomes → 'fft_<freq_band>'  (e.g. 'fft_low')
    DCT genomes → 'dct'

    Run 6 change: FFT genomes now map to three sub-niches ('fft_low',
    'fft_mid', 'fft_high') instead of a single 'fft' niche so each sub-type
    gets its own independent diversity-dampening budget and batch cap.
    """
    gt = genome.get('gen_type', 'lsb')
    if gt == 'lsb':
        return f"lsb_{genome.get('strategy', 'random')}"
    if gt == 'fft':
        return f"fft_{genome.get('freq_band', 'mid')}"
    return 'dct'


def is_low_capacity(genome: dict) -> bool:
    """True if this genome's capacity_ratio is below the low-capacity threshold."""
    return genome.get('capacity_ratio', 0.5) < LOW_CAPACITY_THRESHOLD


def is_hard_edge(genome: dict) -> bool:
    """
    True if this genome matches the hard eval config:
        gen_type='lsb', strategy='edge', threshold≤9, capacity≤0.25.

    These are the exact parameters the model consistently struggled with
    in earlier runs, so we guarantee their presence in every batch.
    """
    return (
        genome.get('gen_type') == 'lsb'
        and genome.get('strategy') == 'edge'
        and genome.get('edge_threshold', 100) <= 9
        and genome.get('capacity_ratio', 1.0) <= 0.25
    )
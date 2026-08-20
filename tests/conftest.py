"""
Shared pytest fixtures for the backend QA suite.

Design notes:
  - No fixture depends on data/ (gitignored, not present in a fresh clone) or
    any other external dataset. Cover images are synthetic, generated with a
    fixed seed so every run is deterministic and reproducible.
  - The FastAPI TestClient is session-scoped: it loads the real SRNet
    checkpoint once (srnet_steganogan_best.pth, ~1-2s on CPU) and reuses it
    across every API test, rather than reloading per test.
"""

import io

import numpy as np
import pytest
from PIL import Image


# ── Synthetic cover images ──────────────────────────────────────────────────
# 256x256 matches the app's actual sliding-window size. Random noise (not a
# flat/gradient image) gives every generator non-trivial DCT/FFT coefficients
# and plenty of embedding capacity for short test payloads.

@pytest.fixture
def cover_array():
    """Deterministic 256x256 grayscale uint8 cover, as a numpy array."""
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, 256, size=(256, 256), dtype=np.uint8)


@pytest.fixture
def cover_rgb_array():
    """Deterministic 256x256 RGB uint8 cover — for colour-preservation tests."""
    rng = np.random.default_rng(seed=7)
    return rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)


@pytest.fixture
def cover_png_bytes(cover_array):
    """cover_array encoded as PNG bytes, ready for a multipart upload."""
    buf = io.BytesIO()
    Image.fromarray(cover_array, mode="L").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def cover_rgb_png_bytes(cover_rgb_array):
    """cover_rgb_array encoded as PNG bytes, ready for a multipart upload."""
    buf = io.BytesIO()
    Image.fromarray(cover_rgb_array, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def png_bytes_of(arr: np.ndarray) -> bytes:
    """Encode an arbitrary uint8 array (2-D gray or 3-D RGB) as PNG bytes."""
    mode = "L" if arr.ndim == 2 else "RGB"
    buf = io.BytesIO()
    Image.fromarray(arr, mode=mode).save(buf, format="PNG")
    return buf.getvalue()


# ── FastAPI test client ─────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def api_client():
    """
    A TestClient with the app's lifespan (model load) run exactly once for the
    whole test session. CPU-only is fine — DEVICE falls back automatically.
    """
    from fastapi.testclient import TestClient
    from api.server import app

    with TestClient(app) as client:
        yield client

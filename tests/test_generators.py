"""
Tests for the steganography generators — both the recoverable codec path
(embed_payload/extract_payload, used by /api/embed + /api/extract) and the
training-facing run() path (used by /api/embed for adaptive/steganogan, and
by the whole training pipeline).

Recoverable-path correctness is the thing that matters most here: if
embed_payload/extract_payload aren't bit-exact, a user's hidden message comes
back corrupted with no warning (the CRC in payload/codec.py is the only
safety net, and it just makes corruption loud instead of silent).
"""

import numpy as np
import pytest

from generators.dct_gen import DCTGenerator
from generators.fft_gen import FFTGenerator
from generators.lsb_gen import LSBGenerator
from generators.unified_generator import UnifiedGenerator
from payload import codec, crypto

RECOVERABLE_GENERATORS = {
    "lsb": (LSBGenerator(), dict(strategy="sequential", step=1, bit_depth=1)),
    "dct": (DCTGenerator(), dict(coeff_selection="mid", strength=3.0)),
    "fft": (FFTGenerator(), dict(freq_band="mid", strength=32.0)),
}

_PACK_PARAMS = {
    "lsb": codec.pack_lsb_params,
    "dct": codec.pack_dct_params,
    "fft": codec.pack_fft_params,
}


# ── recoverable codec: raw bit round trip ───────────────────────────────────

@pytest.mark.parametrize("name", RECOVERABLE_GENERATORS.keys())
def test_embed_extract_payload_bit_exact_round_trip(name, cover_array):
    gen, kw = RECOVERABLE_GENERATORS[name]
    rng = np.random.default_rng(123)
    bits = rng.integers(0, 2, size=200, dtype=np.uint8)

    stego = gen.embed_payload(cover_array, bits, **kw)
    assert stego.shape == cover_array.shape
    assert stego.dtype == cover_array.dtype

    recovered = gen.extract_payload(stego, len(bits), **kw)
    assert np.array_equal(recovered, bits), f"{name}: bit mismatch after round trip"


@pytest.mark.parametrize("name", RECOVERABLE_GENERATORS.keys())
def test_embed_payload_does_not_mutate_input_array(name, cover_array):
    # api/server.py reuses `cover` for original.png after embedding — a
    # generator that mutates in place would corrupt the "before" image.
    gen, kw = RECOVERABLE_GENERATORS[name]
    original = cover_array.copy()
    bits = np.ones(64, dtype=np.uint8)
    gen.embed_payload(cover_array, bits, **kw)
    assert np.array_equal(cover_array, original)


# ── recoverable codec: full framed round trip (mirrors /api/embed + /api/extract) ──
#
# IMPORTANT: DCT (and, less often, FFT) round-trip through embed_payload isn't
# always bit-perfect. Both generators embed by quantizing a float-domain
# coefficient, then IDCT/IFFT back to a uint8 spatial image (rounded/clipped),
# which extraction re-transforms and re-quantizes -- for coefficients that land
# close to a quantization boundary, that spatial round trip can flip which side
# of the boundary they land on. Measured empirically on 30 random 900-bit
# messages at the exact defaults api/server.py ships with: DCT mid/strength=3.0
# flips >=1 bit in ~10% of trials (up to 5 bits); FFT mid/strength=32.0 saw 0
# flips in the same sample. This is exactly why payload/codec.py's CRC-32
# exists: the real guarantee the system provides is "correct message, or a
# clean, detected failure" -- never a wrong message accepted as valid.

@pytest.mark.parametrize("name", RECOVERABLE_GENERATORS.keys())
@pytest.mark.parametrize("cipher_id", [crypto.CIPHER_NONE, crypto.CIPHER_AES256GCM])
def test_full_framed_round_trip_correct_or_cleanly_detected(name, cipher_id, cover_array):
    gen, kw = RECOVERABLE_GENERATORS[name]
    method_id = codec.METHOD_IDS[name]
    message = b"This is a real end-to-end test of the hide/reveal pipeline."
    passphrase = "correct horse battery staple"

    ciphertext, salt, nonce = crypto.encrypt(message, passphrase, cipher_id)
    params = _PACK_PARAMS[name](**kw)
    bits = codec.frame(method_id, cipher_id, salt, nonce, params, ciphertext)

    stego = gen.embed_payload(cover_array, bits, **kw)

    try:
        header = codec.decode(lambda n: gen.extract_payload(stego, n, **kw))
    except codec.CodecError:
        if name == "lsb":
            raise AssertionError("LSB is integer-domain and must always round-trip exactly")
        return  # DCT/FFT: a clean, detected failure is acceptable — see note above

    assert header["method_id"] == method_id
    recovered = crypto.decrypt(header["ciphertext"], passphrase, cipher_id,
                               header["salt"], header["nonce"])
    assert recovered == message


def test_dct_bit_corruption_is_caught_by_crc_not_silently_accepted(cover_array):
    """
    Reproduces a known-corrupting case (cover seed 42 / this exact message,
    DCT mid/strength=3.0 — see the note above) and asserts codec.decode()
    raises CodecError rather than returning a wrong-but-accepted ciphertext.
    """
    gen = DCTGenerator()
    kw = dict(coeff_selection="mid", strength=3.0)
    message = b"This is a real end-to-end test of the hide/reveal pipeline."
    ciphertext, salt, nonce = crypto.encrypt(message, "pw", crypto.CIPHER_NONE)
    params = codec.pack_dct_params(**kw)
    bits = codec.frame(codec.METHOD_DCT, crypto.CIPHER_NONE, salt, nonce, params, ciphertext)

    stego = gen.embed_payload(cover_array, bits, **kw)
    recovered_bits = gen.extract_payload(stego, len(bits), **kw)

    assert not np.array_equal(bits, recovered_bits), (
        "This case is expected to corrupt 1 bit -- if it now round-trips "
        "cleanly, the underlying DCT precision behaviour changed; update or "
        "remove this test rather than leaving it silently non-exercising."
    )
    with pytest.raises(codec.CodecError, match="corrupted"):
        codec.decode(lambda n: recovered_bits[:n])


@pytest.mark.parametrize("name", RECOVERABLE_GENERATORS.keys())
def test_wrong_passphrase_fails_after_real_embed(name, cover_array):
    # End-to-end version of the crypto-layer test: the message really was
    # embedded and extracted correctly: only decryption should fail.
    #
    # crypto.encrypt() draws a fresh random salt/nonce each run, so for
    # DCT/FFT (see the round-trip-precision note above) this occasionally
    # lands on a coefficient pattern that flips a bit in transit — a real,
    # independent failure mode from what this test targets. Treat a clean,
    # detected CodecError the same as the earlier round-trip test: acceptable
    # for DCT/FFT, a hard failure for LSB.
    gen, kw = RECOVERABLE_GENERATORS[name]
    method_id = codec.METHOD_IDS[name]
    cipher_id = crypto.CIPHER_AES256GCM
    ciphertext, salt, nonce = crypto.encrypt(b"top secret", "right-pw", cipher_id)
    params = _PACK_PARAMS[name](**kw)
    bits = codec.frame(method_id, cipher_id, salt, nonce, params, ciphertext)
    stego = gen.embed_payload(cover_array, bits, **kw)

    try:
        header = codec.decode(lambda n: gen.extract_payload(stego, n, **kw))
    except codec.CodecError:
        if name == "lsb":
            raise AssertionError("LSB is integer-domain and must always round-trip exactly")
        return

    with pytest.raises(crypto.DecryptionError):
        crypto.decrypt(header["ciphertext"], "wrong-pw", cipher_id, header["salt"], header["nonce"])


# ── capacity ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", RECOVERABLE_GENERATORS.keys())
def test_payload_capacity_bits_positive(name):
    gen, kw = RECOVERABLE_GENERATORS[name]
    bits = gen.payload_capacity_bits((256, 256), **kw)
    assert bits > 0


@pytest.mark.parametrize("name", RECOVERABLE_GENERATORS.keys())
def test_payload_capacity_scales_with_image_size(name):
    gen, kw = RECOVERABLE_GENERATORS[name]
    small = gen.payload_capacity_bits((64, 64), **kw)
    large = gen.payload_capacity_bits((256, 256), **kw)
    assert large > small


@pytest.mark.parametrize("name", RECOVERABLE_GENERATORS.keys())
def test_embed_payload_raises_when_message_exceeds_capacity(name, cover_array):
    gen, kw = RECOVERABLE_GENERATORS[name]
    capacity = gen.payload_capacity_bits(cover_array.shape, **kw)
    too_many_bits = np.zeros(capacity + 1000, dtype=np.uint8)
    with pytest.raises(ValueError):
        gen.embed_payload(cover_array, too_many_bits, **kw)


# ── run() sanity (training-facing / non-recoverable path) ──────────────────

ALL_GEN_TYPES = ["lsb", "dct", "fft", "adaptive", "steganogan"]


@pytest.fixture(scope="module")
def unified_gen():
    return UnifiedGenerator()


@pytest.mark.parametrize("gen_type", ALL_GEN_TYPES)
def test_run_produces_valid_stego(gen_type, unified_gen, cover_array):
    stego, psnr = unified_gen.generate_stego(cover_array, None, {"gen_type": gen_type})
    assert stego is not None, f"{gen_type}: generate_stego returned None"
    assert stego.dtype == np.uint8
    assert stego.shape == cover_array.shape
    assert np.isfinite(psnr) or psnr == float("inf")
    assert psnr > 0


@pytest.mark.parametrize("gen_type", ALL_GEN_TYPES)
def test_run_preserves_non_square_non_256_shape(gen_type, unified_gen):
    """
    Regression test for the steganogan shape-mismatch bug (fixed in
    generators/steganogan_gen.py): a generator that force-resizes its input
    breaks api/server.py's assumption that stego.shape == cover.shape for
    ANY uploaded image size, not just 256x256 squares.
    """
    rng = np.random.default_rng(99)
    odd_cover = rng.integers(0, 256, size=(180, 300), dtype=np.uint8)  # non-square, non-256
    stego, _ = unified_gen.generate_stego(odd_cover, None, {"gen_type": gen_type})
    assert stego is not None
    assert stego.shape == odd_cover.shape, (
        f"{gen_type}: output shape {stego.shape} != input shape {odd_cover.shape}"
    )


@pytest.mark.parametrize("gen_type", ["adaptive", "steganogan"])
def test_non_recoverable_generators_change_the_image(gen_type, unified_gen, cover_array):
    # Sanity check that these methods actually embed something rather than
    # silently returning the cover unchanged.
    stego, _ = unified_gen.generate_stego(cover_array, None, {"gen_type": gen_type})
    assert not np.array_equal(stego, cover_array)


def test_unknown_gen_type_returns_none(unified_gen, cover_array):
    stego, psnr = unified_gen.generate_stego(cover_array, None, {"gen_type": "not-a-real-method"})
    assert stego is None
    assert psnr == 0

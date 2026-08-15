"""
Unit tests for payload/codec.py — the self-describing frame format that lets
/api/extract identify a payload's method, cipher, and length purely from the
embedded bits, with no side channel.
"""

import numpy as np
import pytest

from payload import codec


# ── frame / decode round trip ───────────────────────────────────────────────

@pytest.mark.parametrize("method_id", [codec.METHOD_LSB, codec.METHOD_DCT, codec.METHOD_FFT])
@pytest.mark.parametrize("cipher_id", [0, 1, 2, 3])
def test_frame_decode_round_trip(method_id, cipher_id):
    ciphertext = b"hello world, this is a test payload!"
    salt = b"S" * 16
    nonce = b"N" * 12
    params = b"P" * 8

    bits = codec.frame(method_id, cipher_id, salt, nonce, params, ciphertext)

    # decode() drives extraction via a read_bits(n) callback — simulate a
    # generator's extract_payload by slicing the same bit array it embedded.
    header = codec.decode(lambda n: bits[:n])

    assert header["method_id"] == method_id
    assert header["cipher_id"] == cipher_id
    assert header["salt"] == salt
    assert header["nonce"] == nonce
    assert header["params"] == params
    assert header["ciphertext"] == ciphertext


def test_frame_decode_empty_ciphertext():
    bits = codec.frame(codec.METHOD_LSB, 0, b"", b"", b"", b"")
    header = codec.decode(lambda n: bits[:n])
    assert header["ciphertext"] == b""
    assert header["payload_len"] == 0


def test_frame_salt_nonce_shorter_than_field_are_padded():
    # Cipher=none legitimately has empty salt/nonce; Fernet has empty nonce.
    bits = codec.frame(codec.METHOD_LSB, 0, salt=b"", nonce=b"", params=b"", ciphertext=b"x")
    header = codec.decode(lambda n: bits[:n])
    assert header["salt"] == b"\x00" * 16
    assert header["nonce"] == b"\x00" * 12


def test_frame_salt_nonce_longer_than_field_are_truncated():
    bits = codec.frame(codec.METHOD_LSB, 0, salt=b"X" * 100, nonce=b"Y" * 100,
                       params=b"Z" * 100, ciphertext=b"x")
    header = codec.decode(lambda n: bits[:n])
    assert header["salt"] == b"X" * 16
    assert header["nonce"] == b"Y" * 12
    assert header["params"] == b"Z" * 8


# ── corruption / integrity checks ───────────────────────────────────────────

def test_decode_rejects_missing_magic():
    garbage = np.random.default_rng(0).integers(0, 2, size=codec.HEADER_BITS, dtype=np.uint8)
    with pytest.raises(codec.CodecError):
        codec.decode(lambda n: garbage[:n])


def test_decode_rejects_short_data():
    too_short = np.zeros(codec.HEADER_BITS - 8, dtype=np.uint8)
    with pytest.raises(codec.CodecError):
        codec.decode(lambda n: too_short[:n])


def test_decode_rejects_crc_mismatch():
    bits = codec.frame(codec.METHOD_LSB, 0, b"", b"", b"", b"authentic message")
    corrupted = bits.copy()
    # Flip a bit inside the ciphertext region (after the header).
    flip_at = codec.HEADER_BITS + 3
    corrupted[flip_at] ^= 1

    with pytest.raises(codec.CodecError, match="corrupted"):
        codec.decode(lambda n: corrupted[:n])


def test_decode_rejects_unsupported_version():
    bits = codec.frame(codec.METHOD_LSB, 0, b"", b"", b"", b"x")
    corrupted = bits.copy()
    # version is the byte right after the 4-byte magic (bit offset 32..40).
    version_byte = codec.bits_to_bytes(corrupted[32:40])
    bumped = bytes([version_byte[0] + 1])
    corrupted[32:40] = codec.bytes_to_bits(bumped)

    with pytest.raises(codec.CodecError, match="version"):
        codec.decode(lambda n: corrupted[:n])


# ── method-specific param packing ───────────────────────────────────────────

@pytest.mark.parametrize("strategy,step,bit_depth", [
    ("sequential", 1, 1),
    ("skip", 7, 2),
    ("skip", 65535, 4),  # max representable step (packed as H, 16-bit)
])
def test_lsb_params_round_trip(strategy, step, bit_depth):
    packed = codec.pack_lsb_params(strategy=strategy, step=step, bit_depth=bit_depth)
    assert len(packed) == 8
    out = codec.unpack_lsb_params(packed)
    assert out == {"strategy": strategy, "step": step, "bit_depth": bit_depth}


@pytest.mark.parametrize("coeff_selection,strength", [("mid", 3.0), ("low_mid", 7.5)])
def test_dct_params_round_trip(coeff_selection, strength):
    packed = codec.pack_dct_params(coeff_selection=coeff_selection, strength=strength)
    assert len(packed) == 8
    out = codec.unpack_dct_params(packed)
    assert out["coeff_selection"] == coeff_selection
    assert out["strength"] == pytest.approx(strength, rel=1e-5)


@pytest.mark.parametrize("freq_band,strength", [("low", 8.0), ("mid", 32.0), ("high", 64.0)])
def test_fft_params_round_trip(freq_band, strength):
    packed = codec.pack_fft_params(freq_band=freq_band, strength=strength)
    assert len(packed) == 8
    out = codec.unpack_fft_params(packed)
    assert out["freq_band"] == freq_band
    assert out["strength"] == pytest.approx(strength, rel=1e-5)


def test_unknown_lsb_strategy_id_falls_back_to_sequential():
    # unpack must never raise on an out-of-range id (forward-compat / corruption).
    out = codec.unpack_lsb_params(codec._PARAMS_LEN * b"\xff")
    assert out["strategy"] == "sequential"


# ── bit/byte helpers ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("data", [b"", b"A", b"hello", bytes(range(256))])
def test_bytes_bits_round_trip(data):
    bits = codec.bytes_to_bits(data)
    assert bits.dtype == np.uint8
    assert set(np.unique(bits)).issubset({0, 1})
    assert codec.bits_to_bytes(bits) == data


def test_bytes_to_bits_is_msb_first():
    # 0b10000000 = 0x80 -> [1,0,0,0,0,0,0,0]
    bits = codec.bytes_to_bits(b"\x80")
    assert list(bits) == [1, 0, 0, 0, 0, 0, 0, 0]


def test_total_bits_for():
    assert codec.total_bits_for(0) == codec.HEADER_BITS
    assert codec.total_bits_for(10) == codec.HEADER_BITS + 80


def test_header_size_is_52_bytes():
    # Locks in the documented frame layout (README/CLAUDE.md cite 52 bytes).
    assert codec.HEADER_SIZE == 52

"""
Self-describing payload frame for recoverable steganography.

A real embed→extract round-trip needs the extractor to know, from the image
alone, *how* the bits were laid down: which method, how the message was
encrypted, how long it is, and the method-specific parameters needed to rebuild
the exact bit positions. This module packs all of that into a fixed-size header
that is embedded ahead of the (optionally encrypted) message.

Layout (big-endian, 52-byte fixed header, then the ciphertext)::

    magic        4s   b"STG1"
    version      B    = 1
    method_id    B    0=lsb 1=dct 2=fft
    cipher_id    B    see crypto_utils (0=none 1=aes 2=chacha 3=fernet)
    flags        B    reserved (0)
    payload_len  I    ciphertext length in bytes
    salt         16s  KDF salt (crypto_utils.SALT_LEN)
    nonce        12s  AEAD nonce (crypto_utils.NONCE_LEN; zero-padded)
    params       8s   method-specific (see pack_*_params below)
    crc32        I    CRC-32 of the ciphertext (corruption / wrong-decode guard)

The whole frame (header + ciphertext) is converted to a bit array with
``np.unpackbits`` and handed to a generator's ``embed_payload``. Extraction is
the mirror: read the header bits first to learn ``payload_len``, then read the
ciphertext bits, verify the CRC, and decrypt.
"""

import struct
import zlib

import numpy as np

from generators import crypto_utils

MAGIC = b"STG1"
VERSION = 1

# Method identifiers (must match the frontend + server routing).
METHOD_LSB = 0
METHOD_DCT = 1
METHOD_FFT = 2

METHOD_NAMES = {METHOD_LSB: "lsb", METHOD_DCT: "dct", METHOD_FFT: "fft"}
METHOD_IDS = {v: k for k, v in METHOD_NAMES.items()}

# Deterministic-strategy sub-codes stored inside the 8-byte params block.
LSB_STRATEGY_IDS = {"sequential": 0, "skip": 1}
LSB_STRATEGY_NAMES = {v: k for k, v in LSB_STRATEGY_IDS.items()}

DCT_COEFF_IDS = {"mid": 0, "low_mid": 1}          # 'random' is not recoverable
DCT_COEFF_NAMES = {v: k for k, v in DCT_COEFF_IDS.items()}

FFT_BAND_IDS = {"low": 0, "mid": 1, "high": 2}
FFT_BAND_NAMES = {v: k for k, v in FFT_BAND_IDS.items()}

_HEADER_FMT = ">4sBBBBI16s12s8sI"
HEADER_SIZE = struct.calcsize(_HEADER_FMT)        # 52 bytes
HEADER_BITS = HEADER_SIZE * 8

_PARAMS_LEN = 8


class CodecError(Exception):
    """Raised when a frame cannot be parsed — no magic, bad version, or CRC fail."""


# ── method params (fixed 8-byte block) ────────────────────────────────────────

def pack_lsb_params(strategy="sequential", step=1, bit_depth=1):
    sid = LSB_STRATEGY_IDS.get(strategy, 0)
    return struct.pack(">BHB4x", sid, int(step) & 0xFFFF, int(bit_depth) & 0xFF)


def unpack_lsb_params(params):
    sid, step, bit_depth = struct.unpack(">BHB4x", params)
    return {
        "strategy": LSB_STRATEGY_NAMES.get(sid, "sequential"),
        "step": step,
        "bit_depth": bit_depth,
    }


def pack_dct_params(coeff_selection="mid", strength=3.0):
    cid = DCT_COEFF_IDS.get(coeff_selection, 0)
    return struct.pack(">Bf3x", cid, float(strength))


def unpack_dct_params(params):
    cid, strength = struct.unpack(">Bf3x", params)
    return {"coeff_selection": DCT_COEFF_NAMES.get(cid, "mid"), "strength": strength}


def pack_fft_params(freq_band="mid", strength=8.0):
    bid = FFT_BAND_IDS.get(freq_band, 1)
    return struct.pack(">Bf3x", bid, float(strength))


def unpack_fft_params(params):
    bid, strength = struct.unpack(">Bf3x", params)
    return {"freq_band": FFT_BAND_NAMES.get(bid, "mid"), "strength": strength}


# ── bit <-> byte helpers ──────────────────────────────────────────────────────

def bytes_to_bits(data):
    """bytes → uint8 ndarray of 0/1 bits (MSB-first per byte)."""
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def bits_to_bytes(bits):
    """uint8 ndarray of 0/1 bits → bytes (length must be a multiple of 8)."""
    return np.packbits(np.asarray(bits, dtype=np.uint8)).tobytes()


# ── frame / parse ─────────────────────────────────────────────────────────────

def frame(method_id, cipher_id, salt, nonce, params, ciphertext):
    """
    Build the full payload bit array (header + ciphertext).

    ``salt`` / ``nonce`` may be shorter than their fields (e.g. empty for
    cipher=none, or empty nonce for Fernet); they are right-padded with zeros.
    """
    salt = (salt or b"").ljust(crypto_utils.SALT_LEN, b"\x00")[: crypto_utils.SALT_LEN]
    nonce = (nonce or b"").ljust(crypto_utils.NONCE_LEN, b"\x00")[: crypto_utils.NONCE_LEN]
    params = (params or b"").ljust(_PARAMS_LEN, b"\x00")[:_PARAMS_LEN]
    crc = zlib.crc32(ciphertext) & 0xFFFFFFFF

    header = struct.pack(
        _HEADER_FMT,
        MAGIC, VERSION, method_id, cipher_id, 0,
        len(ciphertext), salt, nonce, params, crc,
    )
    return bytes_to_bits(header + ciphertext)


def total_bits_for(ciphertext_len):
    """Number of bits a frame occupies for a ciphertext of ``ciphertext_len`` bytes."""
    return (HEADER_SIZE + ciphertext_len) * 8


def parse_header(header_bits):
    """Parse the fixed header from its first :data:`HEADER_BITS` bits."""
    if len(header_bits) < HEADER_BITS:
        raise CodecError("Not enough data for a payload header.")
    header = bits_to_bytes(header_bits[:HEADER_BITS])
    magic, version, method_id, cipher_id, flags, payload_len, salt, nonce, params, crc = \
        struct.unpack(_HEADER_FMT, header)
    if magic != MAGIC:
        raise CodecError("No recoverable payload found (missing marker).")
    if version != VERSION:
        raise CodecError(f"Unsupported payload version {version}.")
    return {
        "method_id": method_id,
        "cipher_id": cipher_id,
        "flags": flags,
        "payload_len": payload_len,
        "salt": salt,
        "nonce": nonce,
        "params": params,
        "crc": crc,
    }


def decode(read_bits):
    """
    Drive extraction given a ``read_bits(n) -> uint8 bit ndarray`` callable.

    ``read_bits`` must return the first ``n`` payload bits in the same
    deterministic order they were embedded. Returns the parsed header dict with
    an added ``"ciphertext"`` (bytes). Raises :class:`CodecError` on a missing
    marker or CRC mismatch.
    """
    header = parse_header(read_bits(HEADER_BITS))
    total = total_bits_for(header["payload_len"])
    all_bits = read_bits(total)
    ct_bits = all_bits[HEADER_BITS:total]
    ciphertext = bits_to_bytes(ct_bits)
    if (zlib.crc32(ciphertext) & 0xFFFFFFFF) != header["crc"]:
        raise CodecError("Payload is corrupted (checksum mismatch).")
    header["ciphertext"] = ciphertext
    return header

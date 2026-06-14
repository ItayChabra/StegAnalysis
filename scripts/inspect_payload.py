"""
Inspect a stego image's recoverable payload step-by-step.

Mirrors what /api/extract + /api/decrypt do internally, but prints every
intermediate value so the encryption chain can be sanity-checked against any
external AES-GCM / ChaCha20-Poly1305 / Fernet tool.

Usage
-----
    python scripts/inspect_payload.py path/to/stego.png --passphrase secret
    python scripts/inspect_payload.py img.png -p secret --method lsb --strength 3.0

What it prints (in order):
    1. The 52-byte codec header (magic, version, method, cipher, lengths, CRC)
    2. Salt, nonce, ciphertext (hex + base64)
    3. AEAD tag split off from the ciphertext when applicable
    4. The PBKDF2-derived 32-byte cipher key (hex) — what an online tool wants
    5. The decrypted plaintext
"""

import argparse
import base64
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Project root on sys.path so the script runs without installation.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from generators.unified_generator import UnifiedGenerator
from payload import codec, crypto

# Mirrors the defaults used by api/server.py — keep in sync if those change.
LSB_DEFAULTS = dict(strategy="sequential", step=1, bit_depth=1)
DCT_DEFAULTS = dict(coeff_selection="mid", strength=3.0)
FFT_DEFAULTS = dict(freq_band="mid", strength=32.0)
METHODS = ("lsb", "dct", "fft")


def hexline(label, data, limit=64):
    """Compact hex dump: shows the first `limit` bytes, then a … if truncated."""
    h = data.hex()
    shown = h[: limit * 2]
    tail = "…" if len(data) > limit else ""
    print(f"  {label:<14} ({len(data):>4} B)  {shown}{tail}")


def build_kwargs(method, args):
    """Per-method kwargs identical to api/server.py:_recoverable_kwargs."""
    if method == "lsb":
        return dict(
            strategy=args.strategy or LSB_DEFAULTS["strategy"],
            step=args.step if args.step is not None else LSB_DEFAULTS["step"],
            bit_depth=args.bit_depth if args.bit_depth is not None else LSB_DEFAULTS["bit_depth"],
        )
    if method == "dct":
        return dict(
            coeff_selection=args.coeff_selection or DCT_DEFAULTS["coeff_selection"],
            strength=args.strength if args.strength is not None else DCT_DEFAULTS["strength"],
        )
    if method == "fft":
        return dict(
            freq_band=args.freq_band or FFT_DEFAULTS["freq_band"],
            strength=args.strength if args.strength is not None else FFT_DEFAULTS["strength"],
        )
    raise SystemExit(f"unknown method {method!r}")


def find_payload(arr, args, ug):
    """Try the given method (or every candidate) and return (method, kw, header)."""
    candidates = [args.method] if args.method else list(METHODS)
    for m in candidates:
        kw = build_kwargs(m, args)
        gen = ug.generators[m]
        try:
            header = codec.decode(lambda n: gen.extract_payload(arr, n, **kw))
        except codec.CodecError:
            continue
        return m, kw, header
    raise SystemExit(
        "No recoverable payload found. Pass --method/--strength/--step to match the embed settings."
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("image", type=Path, help="Path to the stego image (PNG, etc.)")
    ap.add_argument("-p", "--passphrase", default="", help="Passphrase used at embed (default: empty)")
    ap.add_argument("--method", choices=METHODS, help="Override auto-detect")
    # LSB params
    ap.add_argument("--strategy", choices=("sequential", "skip"))
    ap.add_argument("--step", type=int)
    ap.add_argument("--bit-depth", type=int)
    # DCT params
    ap.add_argument("--coeff-selection", choices=("mid", "low_mid"))
    # DCT / FFT params
    ap.add_argument("--strength", type=float)
    ap.add_argument("--freq-band", choices=("low", "mid", "high"))
    args = ap.parse_args()

    if not args.image.exists():
        raise SystemExit(f"image not found: {args.image}")

    arr = np.array(Image.open(args.image).convert("L"), dtype=np.uint8)
    print(f"\n[image] {args.image}  ({arr.shape[1]}×{arr.shape[0]} px)\n")

    ug = UnifiedGenerator()
    method, kw, header = find_payload(arr, args, ug)

    cipher_id   = header["cipher_id"]
    cipher_name = crypto.CIPHER_NAMES.get(cipher_id, f"id={cipher_id}")
    method_name = codec.METHOD_NAMES.get(header["method_id"], method)
    ciphertext  = header["ciphertext"]

    print("[codec header]")
    print(f"  method         {method_name}  (extract kwargs: {kw})")
    print(f"  cipher         {cipher_name}  (id={cipher_id})")
    print(f"  payload_len    {header['payload_len']} bytes")
    print(f"  crc32          {header['crc']:08x}\n")

    # AEAD ciphers (GCM / ChaCha20-Poly1305) append a 16-byte tag to the
    # ciphertext. Online tools almost always want the tag fed separately, so
    # we display them as two distinct fields rather than one blob.
    is_aead = cipher_id in (crypto.CIPHER_AES256GCM, crypto.CIPHER_CHACHA20)
    if is_aead:
        body, tag = ciphertext[:-16], ciphertext[-16:]
    else:
        body, tag = ciphertext, b""

    print("[raw bytes]")
    hexline("salt",       header["salt"])
    hexline("nonce",      header["nonce"])
    hexline("ciphertext", body)
    if is_aead:
        hexline("auth tag", tag)
    print(f"  ciphertext (b64)        {base64.b64encode(body).decode()}")
    if is_aead:
        print(f"  auth tag   (b64)        {base64.b64encode(tag).decode()}")
        print(f"  full (ct ‖ tag, b64)    {base64.b64encode(ciphertext).decode()}")
    print()

    if cipher_id == crypto.CIPHER_NONE:
        print("[plaintext]  (cipher=none — ciphertext IS the plaintext)")
        print(f"  {ciphertext.decode('utf-8', errors='replace')!r}")
        return

    if cipher_id == crypto.CIPHER_FERNET:
        print("[fernet]  token is base64url(version || ts || iv || ct || hmac) — not a raw cipher input\n")

    # PBKDF2 — what makes the passphrase usable as a cipher key.
    print(f"[KDF]  PBKDF2-HMAC-SHA256, iterations=200000, key_len=32")
    if not args.passphrase:
        print("  (no --passphrase given; skipping key derivation + decryption)\n")
        return
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    salt_used = header["salt"][: crypto.SALT_LEN]
    key = PBKDF2HMAC(hashes.SHA256(), 32, salt_used, 200_000).derive(args.passphrase.encode("utf-8"))
    hexline("derived key", key)
    print()

    # Run the same decrypt the API does — proves the printed inputs round-trip.
    try:
        plaintext = crypto.decrypt(
            ciphertext, args.passphrase, cipher_id, header["salt"], header["nonce"]
        )
    except crypto.DecryptionError as e:
        print(f"[decrypt]  FAILED — {e}")
        return

    print("[plaintext]")
    print(f"  bytes: {len(plaintext)}")
    print(f"  utf-8: {plaintext.decode('utf-8', errors='replace')!r}")


if __name__ == "__main__":
    main()

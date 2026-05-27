"""
Passphrase-based authenticated encryption for the steganography demo.

Used by the recoverable embed/extract path (see ``payload_codec.py`` and the
``/api/embed`` + ``/api/extract`` endpoints). A message is optionally encrypted
before it is framed and embedded; on extraction it is decrypted with the same
passphrase.

All supported ciphers are *authenticated* — supplying the wrong passphrase (or a
tampered ciphertext) raises :class:`DecryptionError` instead of returning
garbage bytes. This is what lets the UI show a clean "wrong key" message.

Cipher IDs (stored in the payload header)::

    0 = none                  (no encryption — plaintext is embedded directly)
    1 = AES-256-GCM
    2 = ChaCha20-Poly1305
    3 = Fernet                (AES-128-CBC + HMAC, manages its own nonce/IV)

Key derivation: PBKDF2-HMAC-SHA256, 200k iterations, random 16-byte salt.
The salt and nonce are stored in the header so extraction is self-describing.
"""

import base64
import os

from cryptography.exceptions import InvalidTag
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# ── Cipher identifiers (must match payload_codec / frontend) ──────────────────
CIPHER_NONE = 0
CIPHER_AES256GCM = 1
CIPHER_CHACHA20 = 2
CIPHER_FERNET = 3

CIPHER_NAMES = {
    CIPHER_NONE: "none",
    CIPHER_AES256GCM: "aes256gcm",
    CIPHER_CHACHA20: "chacha20poly1305",
    CIPHER_FERNET: "fernet",
}
CIPHER_IDS = {v: k for k, v in CIPHER_NAMES.items()}

SALT_LEN = 16
NONCE_LEN = 12          # AES-GCM / ChaCha20-Poly1305 nonce size
_PBKDF2_ITERATIONS = 200_000
_KEY_LEN = 32           # 256-bit key (AES-256 / ChaCha20); Fernet uses first 32 bytes


class DecryptionError(Exception):
    """Raised when decryption fails — wrong passphrase or tampered ciphertext."""


def cipher_id_from_name(name):
    """Map a cipher name (or id) coming from the API to a numeric cipher id."""
    if name is None or name == "":
        return CIPHER_NONE
    if isinstance(name, int):
        return name
    key = str(name).strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "none": CIPHER_NONE,
        "aes": CIPHER_AES256GCM,
        "aes256": CIPHER_AES256GCM,
        "aes256gcm": CIPHER_AES256GCM,
        "chacha20": CIPHER_CHACHA20,
        "chacha20poly1305": CIPHER_CHACHA20,
        "fernet": CIPHER_FERNET,
    }
    if key not in aliases:
        raise ValueError(f"Unknown cipher {name!r}. Valid: {sorted(set(aliases))}")
    return aliases[key]


def _derive_key(passphrase, salt):
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=_KEY_LEN,
        salt=salt,
        iterations=_PBKDF2_ITERATIONS,
    )
    return kdf.derive(passphrase.encode("utf-8"))


def encrypt(plaintext, passphrase, cipher_id):
    """
    Encrypt ``plaintext`` (bytes) with ``passphrase`` using ``cipher_id``.

    Returns ``(ciphertext, salt, nonce)`` — all bytes. ``salt`` / ``nonce`` are
    empty for ``CIPHER_NONE``. The auth tag is appended to ``ciphertext`` by the
    AEAD ciphers; Fernet packs everything into its own token.
    """
    if cipher_id == CIPHER_NONE:
        return plaintext, b"", b""

    if not passphrase:
        raise ValueError("A passphrase is required when encryption is enabled.")

    salt = os.urandom(SALT_LEN)
    key = _derive_key(passphrase, salt)

    if cipher_id == CIPHER_AES256GCM:
        nonce = os.urandom(NONCE_LEN)
        ct = AESGCM(key).encrypt(nonce, plaintext, None)
        return ct, salt, nonce

    if cipher_id == CIPHER_CHACHA20:
        nonce = os.urandom(NONCE_LEN)
        ct = ChaCha20Poly1305(key).encrypt(nonce, plaintext, None)
        return ct, salt, nonce

    if cipher_id == CIPHER_FERNET:
        token = Fernet(base64.urlsafe_b64encode(key)).encrypt(plaintext)
        return token, salt, b""      # Fernet embeds its own IV in the token

    raise ValueError(f"Unknown cipher_id {cipher_id!r}")


def decrypt(ciphertext, passphrase, cipher_id, salt, nonce):
    """
    Reverse :func:`encrypt`. Returns the recovered plaintext bytes.

    Raises :class:`DecryptionError` if the passphrase is wrong or the ciphertext
    has been altered (authenticated ciphers detect this).
    """
    if cipher_id == CIPHER_NONE:
        return ciphertext

    if not passphrase:
        raise DecryptionError("A passphrase is required to decrypt this message.")

    key = _derive_key(passphrase, salt)

    try:
        if cipher_id == CIPHER_AES256GCM:
            return AESGCM(key).decrypt(nonce, ciphertext, None)
        if cipher_id == CIPHER_CHACHA20:
            return ChaCha20Poly1305(key).decrypt(nonce, ciphertext, None)
        if cipher_id == CIPHER_FERNET:
            return Fernet(base64.urlsafe_b64encode(key)).decrypt(ciphertext)
    except (InvalidTag, InvalidToken):
        raise DecryptionError("Wrong passphrase or the data has been altered.")

    raise ValueError(f"Unknown cipher_id {cipher_id!r}")

"""
Unit tests for payload/crypto.py — passphrase-based authenticated encryption.

The core guarantee under test: every cipher must reject a wrong passphrase (or
tampered ciphertext) with DecryptionError, never silently return garbage. This
is what lets the UI show a clean "wrong key" message instead of mangled text.
"""

import pytest

from payload import crypto

ALL_CIPHERS = [crypto.CIPHER_NONE, crypto.CIPHER_AES256GCM,
              crypto.CIPHER_CHACHA20, crypto.CIPHER_FERNET]
AUTHENTICATED_CIPHERS = [crypto.CIPHER_AES256GCM, crypto.CIPHER_CHACHA20, crypto.CIPHER_FERNET]


@pytest.mark.parametrize("cipher_id", ALL_CIPHERS)
def test_encrypt_decrypt_round_trip(cipher_id):
    plaintext = b"the quick brown fox jumps over the lazy dog"
    ciphertext, salt, nonce = crypto.encrypt(plaintext, "correct horse battery staple", cipher_id)
    recovered = crypto.decrypt(ciphertext, "correct horse battery staple", cipher_id, salt, nonce)
    assert recovered == plaintext


@pytest.mark.parametrize("cipher_id", ALL_CIPHERS)
def test_round_trip_empty_plaintext(cipher_id):
    ciphertext, salt, nonce = crypto.encrypt(b"", "pw", cipher_id)
    assert crypto.decrypt(ciphertext, "pw", cipher_id, salt, nonce) == b""


@pytest.mark.parametrize("cipher_id", ALL_CIPHERS)
def test_round_trip_unicode_message(cipher_id):
    # UTF-8 encoded before this layer, matching how server.py calls it
    # (message.encode("utf-8")).
    plaintext = "héllo wörld — 日本語 — emoji 🔒".encode("utf-8")
    ciphertext, salt, nonce = crypto.encrypt(plaintext, "pw", cipher_id)
    assert crypto.decrypt(ciphertext, "pw", cipher_id, salt, nonce) == plaintext


@pytest.mark.parametrize("cipher_id", AUTHENTICATED_CIPHERS)
def test_wrong_passphrase_raises_decryption_error(cipher_id):
    ciphertext, salt, nonce = crypto.encrypt(b"secret message", "right-password", cipher_id)
    with pytest.raises(crypto.DecryptionError):
        crypto.decrypt(ciphertext, "wrong-password", cipher_id, salt, nonce)


@pytest.mark.parametrize("cipher_id", AUTHENTICATED_CIPHERS)
def test_tampered_ciphertext_raises_decryption_error(cipher_id):
    ciphertext, salt, nonce = crypto.encrypt(b"secret message", "pw", cipher_id)
    tampered = bytes([ciphertext[0] ^ 0xFF]) + ciphertext[1:]
    with pytest.raises(crypto.DecryptionError):
        crypto.decrypt(tampered, "pw", cipher_id, salt, nonce)


@pytest.mark.parametrize("cipher_id", [crypto.CIPHER_AES256GCM, crypto.CIPHER_CHACHA20])
def test_aead_ciphertext_differs_from_plaintext(cipher_id):
    # Sanity: these are not no-ops — the ciphertext must not just be the
    # plaintext with a tag appended in a way that's recoverable without a key.
    plaintext = b"not encrypted would be visible here" * 3
    ciphertext, _, _ = crypto.encrypt(plaintext, "pw", cipher_id)
    assert plaintext not in ciphertext


def test_cipher_none_is_passthrough_no_encryption():
    plaintext = b"plaintext stays plaintext"
    ciphertext, salt, nonce = crypto.encrypt(plaintext, "", crypto.CIPHER_NONE)
    assert ciphertext == plaintext
    assert salt == b""
    assert nonce == b""
    assert crypto.decrypt(ciphertext, "", crypto.CIPHER_NONE, salt, nonce) == plaintext


@pytest.mark.parametrize("cipher_id", AUTHENTICATED_CIPHERS)
def test_empty_passphrase_rejected_on_encrypt(cipher_id):
    with pytest.raises(ValueError):
        crypto.encrypt(b"secret", "", cipher_id)


@pytest.mark.parametrize("cipher_id", AUTHENTICATED_CIPHERS)
def test_empty_passphrase_rejected_on_decrypt(cipher_id):
    ciphertext, salt, nonce = crypto.encrypt(b"secret", "pw", cipher_id)
    with pytest.raises(crypto.DecryptionError):
        crypto.decrypt(ciphertext, "", cipher_id, salt, nonce)


@pytest.mark.parametrize("cipher_id", AUTHENTICATED_CIPHERS)
def test_different_salts_produce_different_ciphertext(cipher_id):
    # Encrypting the same message twice must not be deterministic (random
    # salt/nonce each time) — otherwise identical messages would be
    # distinguishable by ciphertext alone.
    ct1, salt1, _ = crypto.encrypt(b"same message", "pw", cipher_id)
    ct2, salt2, _ = crypto.encrypt(b"same message", "pw", cipher_id)
    assert salt1 != salt2
    assert ct1 != ct2


# ── cipher_id_from_name ─────────────────────────────────────────────────────

@pytest.mark.parametrize("name,expected", [
    (None, crypto.CIPHER_NONE),
    ("", crypto.CIPHER_NONE),
    ("none", crypto.CIPHER_NONE),
    ("aes256gcm", crypto.CIPHER_AES256GCM),
    ("AES-256-GCM", crypto.CIPHER_AES256GCM),
    ("aes", crypto.CIPHER_AES256GCM),
    ("chacha20poly1305", crypto.CIPHER_CHACHA20),
    ("ChaCha20-Poly1305", crypto.CIPHER_CHACHA20),
    ("fernet", crypto.CIPHER_FERNET),
])
def test_cipher_id_from_name_aliases(name, expected):
    assert crypto.cipher_id_from_name(name) == expected


def test_cipher_id_from_name_passthrough_int():
    assert crypto.cipher_id_from_name(crypto.CIPHER_AES256GCM) == crypto.CIPHER_AES256GCM


def test_cipher_id_from_name_unknown_raises():
    with pytest.raises(ValueError):
        crypto.cipher_id_from_name("not-a-real-cipher")


def test_cipher_names_and_ids_are_consistent_inverses():
    for cid, name in crypto.CIPHER_NAMES.items():
        assert crypto.CIPHER_IDS[name] == cid

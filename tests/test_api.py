"""
Integration tests for api/server.py via FastAPI's TestClient — exercises the
real SRNet model, real generators, and real crypto through actual HTTP calls,
the same way the frontend does.

Runs on CPU (DEVICE falls back automatically); the session-scoped api_client
fixture loads the model once for the whole file.
"""

import numpy as np
import pytest

from payload import codec
from tests.conftest import png_bytes_of


def _upload(client, path, filename_bytes, data=None, filename="cover.png"):
    files = {"file": (filename, filename_bytes, "image/png")}
    return client.post(path, files=files, data=data or {})


# ── /health ──────────────────────────────────────────────────────────────────

def test_health(api_client):
    r = api_client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ── /api/analyze ─────────────────────────────────────────────────────────────

def test_analyze_returns_valid_response(api_client, cover_png_bytes):
    r = _upload(api_client, "/api/analyze", cover_png_bytes)
    assert r.status_code == 200
    body = r.json()
    assert body["verdict"] in ("CLEAN", "SUSPICIOUS", "STEGO_DETECTED")
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["total_windows"] >= 1
    assert len(body["bitplane_scores"]) == 8
    assert body["original_url"].startswith("/api/original/")
    assert body["heatmap_url"].startswith("/api/heatmap/")


def test_analyze_rejects_invalid_image(api_client):
    r = _upload(api_client, "/api/analyze", b"this is not a valid image file")
    assert r.status_code == 400
    assert "error" in r.json()["detail"]


def test_analyze_handles_small_image(api_client):
    # Smaller than one 256x256 window — exercises the pad-to-window path.
    small = png_bytes_of(np.random.default_rng(1).integers(0, 256, size=(40, 40), dtype=np.uint8))
    r = _upload(api_client, "/api/analyze", small)
    assert r.status_code == 200
    assert r.json()["total_windows"] >= 1


# ── /api/embed + /api/extract + /api/decrypt: recoverable round trip ────────

def test_lsb_embed_extract_decrypt_full_round_trip(api_client, cover_png_bytes):
    message = "The quick brown fox jumps over the lazy dog."
    passphrase = "correct horse battery staple"

    r = _upload(api_client, "/api/embed", cover_png_bytes, data={
        "method": "lsb", "message": message,
        "cipher": "aes256gcm", "passphrase": passphrase,
    })
    assert r.status_code == 200
    embed_body = r.json()
    assert embed_body["recoverable"] is True
    assert embed_body["method"] == "lsb"
    assert embed_body["encrypted"] is True

    stego_resp = api_client.get(embed_body["stego_url"])
    assert stego_resp.status_code == 200
    assert stego_resp.headers["content-type"] == "image/png"

    r = _upload(api_client, "/api/extract", stego_resp.content, data={"method": "lsb"})
    assert r.status_code == 200
    extract_body = r.json()
    assert extract_body["encrypted"] is True
    assert "message" not in extract_body  # encrypted: server must NOT leak ciphertext-as-message

    r = api_client.post("/api/decrypt", data={
        "ciphertext_b64": extract_body["ciphertext_b64"],
        "cipher": extract_body["cipher"],
        "passphrase": passphrase,
        "salt_b64": extract_body["salt_b64"],
        "nonce_b64": extract_body["nonce_b64"],
    })
    assert r.status_code == 200
    assert r.json()["message"] == message


def test_lsb_embed_extract_no_encryption_message_direct(api_client, cover_png_bytes):
    message = "no encryption needed here"
    r = _upload(api_client, "/api/embed", cover_png_bytes, data={
        "method": "lsb", "message": message, "cipher": "none",
    })
    assert r.status_code == 200
    stego = api_client.get(r.json()["stego_url"]).content

    r = _upload(api_client, "/api/extract", stego, data={"method": "lsb"})
    assert r.status_code == 200
    body = r.json()
    assert body["encrypted"] is False
    assert body["message"] == message


def test_wrong_passphrase_returns_422_bad_key(api_client, cover_png_bytes):
    r = _upload(api_client, "/api/embed", cover_png_bytes, data={
        "method": "lsb", "message": "secret", "cipher": "aes256gcm", "passphrase": "right-pw",
    })
    stego = api_client.get(r.json()["stego_url"]).content
    extract_body = _upload(api_client, "/api/extract", stego, data={"method": "lsb"}).json()

    r = api_client.post("/api/decrypt", data={
        "ciphertext_b64": extract_body["ciphertext_b64"],
        "cipher": extract_body["cipher"],
        "passphrase": "wrong-pw",
        "salt_b64": extract_body["salt_b64"],
        "nonce_b64": extract_body["nonce_b64"],
    })
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "bad_key"


def test_extract_with_no_payload_returns_404(api_client, cover_png_bytes):
    # cover_png_bytes was never embedded into — nothing recoverable in it.
    r = _upload(api_client, "/api/extract", cover_png_bytes, data={"method": "lsb"})
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "no_payload"


def test_extract_auto_detects_method_when_not_specified(api_client, cover_png_bytes):
    r = _upload(api_client, "/api/embed", cover_png_bytes, data={
        "method": "lsb", "message": "auto-detect me", "cipher": "none",
    })
    stego = api_client.get(r.json()["stego_url"]).content

    # No 'method' field at all — server tries lsb/dct/fft in turn.
    r = _upload(api_client, "/api/extract", stego)
    assert r.status_code == 200
    assert r.json()["message"] == "auto-detect me"


def test_fft_embed_extract_round_trip(api_client, cover_png_bytes):
    message = "fft round trip"
    r = _upload(api_client, "/api/embed", cover_png_bytes, data={
        "method": "fft", "message": message, "cipher": "none",
    })
    assert r.status_code == 200
    stego = api_client.get(r.json()["stego_url"]).content
    r = _upload(api_client, "/api/extract", stego, data={"method": "fft"})
    assert r.status_code == 200
    assert r.json()["message"] == message


def test_dct_embed_extract_correct_or_cleanly_rejected(api_client, cover_png_bytes):
    # See tests/test_generators.py for why DCT's quantization round trip isn't
    # always bit-perfect (~10% of trials at the server's default strength).
    # Through the API this must never surface as a 500 or garbled message —
    # only a correct extract, or a clean 404 (decode() raising CodecError).
    message = "dct round trip"
    r = _upload(api_client, "/api/embed", cover_png_bytes, data={
        "method": "dct", "message": message, "cipher": "none",
    })
    assert r.status_code == 200
    stego = api_client.get(r.json()["stego_url"]).content
    r = _upload(api_client, "/api/extract", stego, data={"method": "dct"})
    assert r.status_code in (200, 404)
    if r.status_code == 200:
        assert r.json()["message"] == message
    else:
        assert r.json()["detail"]["code"] == "no_payload"


def test_embed_message_too_large_returns_400(api_client, cover_png_bytes):
    r = _upload(api_client, "/api/embed", cover_png_bytes, data={
        "method": "lsb", "message": "x" * 100_000, "cipher": "none",
    })
    assert r.status_code == 400
    assert "capacity_bytes" in r.json()["detail"]


def test_embed_unknown_method_returns_400(api_client, cover_png_bytes):
    r = _upload(api_client, "/api/embed", cover_png_bytes, data={"method": "not-a-real-method"})
    assert r.status_code == 400


# ── /api/embed: non-recoverable methods (adaptive / steganogan) ─────────────

@pytest.mark.parametrize("method", ["adaptive", "steganogan"])
def test_non_recoverable_embed_reports_correctly(api_client, cover_png_bytes, method):
    r = _upload(api_client, "/api/embed", cover_png_bytes, data={"method": method})
    assert r.status_code == 200
    body = r.json()
    assert body["recoverable"] is False
    assert body["method"] == method
    assert "note" in body
    assert api_client.get(body["stego_url"]).status_code == 200


@pytest.mark.parametrize("method", ["adaptive", "steganogan"])
def test_non_recoverable_embed_preserves_colour(api_client, cover_rgb_png_bytes, method):
    # Regression test: both methods used to flatten colour uploads to
    # grayscale on both Original and Modified (fixed in commit 86fddd6).
    r = _upload(api_client, "/api/embed", cover_rgb_png_bytes, data={"method": method})
    assert r.status_code == 200
    body = r.json()

    orig = api_client.get(body["original_url"])
    stego = api_client.get(body["stego_url"])
    assert orig.status_code == 200 and stego.status_code == 200

    from PIL import Image
    import io
    orig_img = Image.open(io.BytesIO(orig.content))
    stego_img = Image.open(io.BytesIO(stego.content))
    assert orig_img.mode == "RGB"
    assert stego_img.mode == "RGB"
    # Real colour variance, not a flattened/greyscale-looking RGB image.
    assert np.array(orig_img).std(axis=(0, 1)).min() > 5
    assert np.array(stego_img).std(axis=(0, 1)).min() > 5


def test_non_recoverable_embed_handles_non_256_shape(api_client):
    # Regression test for the steganogan force-resize bug (commit 057f780):
    # output must match the uploaded image's own shape, not a fixed 256x256.
    odd = png_bytes_of(np.random.default_rng(5).integers(0, 256, size=(180, 300), dtype=np.uint8))
    r = _upload(api_client, "/api/embed", odd, data={"method": "steganogan"})
    assert r.status_code == 200
    stego = api_client.get(r.json()["stego_url"])
    from PIL import Image
    import io
    assert Image.open(io.BytesIO(stego.content)).size == (300, 180)  # PIL is (W, H)


def test_extract_rejects_non_recoverable_method(api_client, cover_png_bytes):
    r = _upload(api_client, "/api/extract", cover_png_bytes, data={"method": "adaptive"})
    assert r.status_code == 400


# ── /api/capacity ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("method", ["lsb", "dct", "fft"])
def test_capacity_recoverable_methods(api_client, cover_png_bytes, method):
    r = _upload(api_client, "/api/capacity", cover_png_bytes, data={"method": method})
    assert r.status_code == 200
    body = r.json()
    assert body["recoverable"] is True
    assert body["capacity_bytes"] > 0
    assert body["max_message_bytes"] >= 0
    assert body["max_message_bytes"] == body["capacity_bytes"] - codec.HEADER_SIZE \
        or body["max_message_bytes"] == 0


@pytest.mark.parametrize("method", ["adaptive", "steganogan"])
def test_capacity_non_recoverable_methods(api_client, cover_png_bytes, method):
    r = _upload(api_client, "/api/capacity", cover_png_bytes, data={"method": method})
    assert r.status_code == 200
    body = r.json()
    assert body["recoverable"] is False
    assert body["max_message_bytes"] == 0


# ── job-scoped image endpoints ───────────────────────────────────────────────

def test_unknown_job_id_returns_404(api_client):
    r = api_client.get("/api/original/not-a-real-job-id")
    assert r.status_code == 404


def test_stego_and_diff_endpoints_after_embed(api_client, cover_png_bytes):
    r = _upload(api_client, "/api/embed", cover_png_bytes, data={
        "method": "lsb", "message": "hi", "cipher": "none",
    })
    body = r.json()
    for url in (body["original_url"], body["stego_url"], body["diff_url"], body["spectrum_url"]):
        resp = api_client.get(url)
        assert resp.status_code == 200, f"{url} -> {resp.status_code}"
        assert resp.headers["content-type"] == "image/png"

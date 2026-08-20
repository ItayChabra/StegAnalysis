"""
Non-functional test — Chapter 10.2 row 5 (Security / Penetration Test).

Builds a plain-text file carrying arbitrary/malicious content (a shell
injection string + the industry-standard EICAR antivirus test signature —
safe, not live malware, but the string every AV engine is built to flag),
renames it to .jpg, and uploads it through the real FastAPI TestClient
against the actual /api/analyze route (api/server.py::analyze -> _read_image)
— not a mocked path. The functional suite (Table 16 row 4) already covers a
plain mismatched-extension text file; this variant carries a hostile payload
to confirm the same format-sniffing gate holds for it too.

Runs fast, needs no external data — part of the default `pytest` run.
"""

EICAR = br"X5O!P%@AP[4\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*"


def _malicious_payload():
    """Arbitrary/malicious text content, not a valid image in any format."""
    return (
        b"#!/bin/sh\nrm -rf / --no-preserve-root\n"
        b"<script>alert(document.cookie)</script>\n"
        + EICAR + b"\n"
        b"'; DROP TABLE users; --\n"
    )


def test_malicious_payload_disguised_as_jpg_is_rejected(api_client):
    payload = _malicious_payload()
    r = api_client.post(
        "/api/analyze",
        files={"file": ("cover.jpg", payload, "image/jpeg")},
    )
    assert r.status_code == 400, (
        f"expected 400 Invalid Format, got {r.status_code}: {r.text}"
    )
    body = r.json()
    assert "detail" in body
    assert "error" in body["detail"]
    assert "invalid" in body["detail"]["error"].lower()
    # No job_id / artefact URLs should ever appear for a rejected upload —
    # confirms the pipeline stopped at format-sniffing and never ran inference.
    assert "job_id" not in body


def test_malicious_payload_disguised_as_jpg_via_embed_endpoint_is_rejected(api_client):
    """Same polyglot-style attack against /api/embed, which shares _read_image."""
    payload = _malicious_payload()
    r = api_client.post(
        "/api/embed",
        files={"file": ("cover.jpg", payload, "image/jpeg")},
        data={"method": "lsb", "message": "hi", "cipher": "none"},
    )
    assert r.status_code == 400, (
        f"expected 400 Invalid Format, got {r.status_code}: {r.text}"
    )
    assert "invalid" in r.json()["detail"]["error"].lower()


def test_control_real_jpg_bytes_are_accepted(api_client, cover_png_bytes):
    """Sanity control: a genuinely valid image (PNG bytes, .jpg extension —
    PIL sniffs content not extension) is still processed normally, proving
    the 400s above are a real content check, not an extension blanket-reject."""
    r = api_client.post(
        "/api/analyze",
        files={"file": ("cover.jpg", cover_png_bytes, "image/jpeg")},
    )
    assert r.status_code == 200
    assert "job_id" in r.json()

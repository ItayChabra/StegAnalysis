"""
edge_case_coverage.py — Chapter 10.2 edge-case coverage matrix.

This is a source cross-reference, not a runtime measurement: every cell below
was verified by reading the cited test function and confirming what it
actually asserts (file:line), not by name-matching alone. Three levels:

  full    — a real test asserts the exact behaviour for this case
  partial — the behaviour is asserted at ONE layer (e.g. codec unit test)
            but not exercised end-to-end through the real API/pipeline, or
            only a subset of the relevant combinations is covered
  gap     — no test found asserting this case at all

Re-verify against the current test suite before trusting this for the book —
it is a snapshot as of this run, and will silently go stale if tests are
added/removed/renamed without updating this file.

Outputs (scripts/qa_reports/output/):
  - edge_case_coverage.json
  - edge_case_coverage.csv
  - edge_case_coverage.png

Run:
    python -m scripts.qa_reports.edge_case_coverage
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parent / "output"

# (case, status, evidence)
ROWS = [
    # ── input equivalence classes (Chapter 10) ──────────────────────────
    ("Valid images", "full",
     "tests/test_api.py:test_analyze_returns_valid_response; "
     "tests/test_generators.py:test_run_produces_valid_stego[5 methods]"),
    ("Invalid formats", "full",
     "tests/test_api.py:test_analyze_rejects_invalid_image (400, Invalid image file)"),
    ("Corrupted / malicious content", "full",
     "tests/nonfunctional/test_security_malformed_upload.py: "
     "shell+XSS+SQLi+EICAR payload disguised as .jpg, rejected 400 on /analyze and /embed"),
    ("Clean ground truth (labelled)", "full",
     "tests/nonfunctional/test_reliability_witness_set.py: 500 held-out-split clean images"),
    ("Stego ground truth (labelled)", "full",
     "tests/nonfunctional/test_reliability_witness_set.py: 500 stego, 100/method x 5 methods"),

    # ── boundary / edge cases ────────────────────────────────────────────
    ("Non-square images", "full",
     "tests/test_generators.py:test_run_preserves_non_square_non_256_shape[5 methods] (180x300); "
     "tests/test_api.py:test_non_recoverable_embed_handles_non_256_shape (steganogan, via API)"),
    ("Very small images", "full",
     "tests/test_api.py:test_analyze_handles_small_image (40x40, below one 256 window, pad path)"),
    ("Very large images", "gap",
     "No test uses an image above ordinary photo size (largest in-suite input is "
     "~500x375 flickr30k); no boundary test for e.g. multi-megapixel uploads."),
    ("Borderline confidence (~50%)", "gap",
     "test_analyze_returns_valid_response only asserts verdict is one of the 3 enum "
     "values and confidence in [0,1] -- no test forces a score near the 0.50 "
     "SUSPICIOUS/CLEAN boundary to check classification at the edge."),
    ("Wrong password", "full",
     "tests/test_crypto.py:test_wrong_passphrase_raises_decryption_error[3 auth ciphers]; "
     "tests/test_generators.py:test_wrong_passphrase_fails_after_real_embed[3 methods]; "
     "tests/test_api.py:test_wrong_passphrase_returns_422_bad_key"),
    ("Tampered ciphertext", "partial",
     "tests/test_crypto.py:test_tampered_ciphertext_raises_decryption_error[3 auth ciphers] "
     "-- crypto-unit only; no test tampers bytes of a real embedded stego image and "
     "extracts via /api/extract"),
    ("Truncated payload headers", "partial",
     "tests/test_codec.py:test_decode_rejects_short_data -- codec-unit only; no "
     "/api/extract test against a genuinely truncated/short stego upload"),
    ("CRC mismatches", "full",
     "tests/test_codec.py:test_decode_rejects_crc_mismatch (synthetic bit flip); "
     "tests/test_generators.py:test_dct_bit_corruption_is_caught_by_crc_not_silently_accepted "
     "(real DCT precision-loss case, not synthetic)"),
    ("Malformed magic bytes", "partial",
     "tests/test_codec.py:test_decode_rejects_missing_magic -- codec-unit only; no "
     "API-level equivalent (though /api/extract shares the same codec.decode() path)"),
    ("Oversized message vs. capacity", "full",
     "tests/test_generators.py:test_embed_payload_raises_when_message_exceeds_capacity[3 methods]; "
     "tests/test_api.py:test_embed_message_too_large_returns_400"),
    ("5 methods x codec/crypto combinations", "partial",
     "Full pixel-level embed->encrypt->extract->decrypt integration "
     "(test_full_framed_round_trip_correct_or_cleanly_detected) covers only "
     "{lsb,dct,fft} x {none,aes256gcm} = 6/12 recoverable combos; chacha20poly1305 "
     "and fernet are untested at this exact integration layer (each cipher IS "
     "covered standalone: crypto-unit tests all 4 generically, codec-framing test "
     "covers all 4 x {lsb,dct,fft}=12/12 at the frame/decode layer, just not through "
     "a real pixel round trip). adaptive/steganogan correctly have no cipher combos "
     "(architecturally non-recoverable; verified by "
     "test_non_recoverable_embed_reports_correctly)."),
]

STATUS_SCORE = {"full": 1.0, "partial": 0.5, "gap": 0.0}
STATUS_COLOR = {"full": "#2e7d32", "partial": "#e0a72e", "gap": "#c0392b"}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUT_DIR / "edge_case_coverage.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "status", "evidence"])
        w.writerows(ROWS)

    counts = {"full": 0, "partial": 0, "gap": 0}
    for _, status, _ in ROWS:
        counts[status] += 1
    summary = {
        "n_cases": len(ROWS),
        "counts": counts,
        "cases": [{"case": c, "status": s, "evidence": e} for c, s, e in ROWS],
    }
    (OUT_DIR / "edge_case_coverage.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"n_cases": len(ROWS), "counts": counts}, indent=2))

    # ── chart: horizontal status strip ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(ROWS) + 1.5))
    labels = [c for c, _, _ in ROWS]
    scores = [STATUS_SCORE[s] for _, s, _ in ROWS]
    colors = [STATUS_COLOR[s] for _, s, _ in ROWS]
    y = range(len(ROWS))
    ax.barh(y, [1] * len(ROWS), color=colors, alpha=0.85, height=0.7)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    for i, (_, status, _) in enumerate(ROWS):
        ax.text(0.5, i, status.upper(), ha="center", va="center",
                 fontsize=8, color="white", fontweight="bold")
    ax.set_title(
        f"Edge-case coverage: {counts['full']} full / {counts['partial']} partial / "
        f"{counts['gap']} gap (n={len(ROWS)})"
    )
    from matplotlib.patches import Patch
    handles = [Patch(color=STATUS_COLOR[s], label=s) for s in ["full", "partial", "gap"]]
    ax.legend(handles=handles, loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "edge_case_coverage.png", dpi=150)
    print(f"chart written to {OUT_DIR / 'edge_case_coverage.png'}")


if __name__ == "__main__":
    main()

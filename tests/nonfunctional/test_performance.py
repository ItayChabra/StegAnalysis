"""
Non-functional test — Chapter 10.2 row 1 (Performance / Stress Test).

Measures wall-clock time through the REAL /api/analyze endpoint (the exact
code path the frontend and CLI hit — multipart parse, sliding-window SRNet
inference, heatmap/SRM-residual artefact generation, JSON response), not a
mocked or trimmed-down inference call.

Requires `data/raw/flickr30k` (gitignored — not present in a fresh clone), so
it is skipped unless both the data directory exists AND RUN_NFR_TESTS=1 is
set. This keeps the default `pytest` run (141 backend / 56 frontend, see
README) untouched.

Run explicitly:
    RUN_NFR_TESTS=1 pytest tests/nonfunctional/test_performance.py -s -m slow
"""

import json
import os
import platform
import random
import time
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.slow

RUN_NFR      = os.environ.get("RUN_NFR_TESTS") == "1"
DATA_DIR     = Path(__file__).resolve().parents[2] / "data" / "raw" / "flickr30k"
RESULTS_DIR  = Path(__file__).resolve().parent / "results"
BATCH_N      = 100
SINGLE_LIMIT = 0.6   # seconds — criterion for a single image
BATCH_LIMIT  = 0.3   # seconds/image — criterion for batch average

_SKIP_REASON = (
    "non-functional stress test — run explicitly with "
    "RUN_NFR_TESTS=1 pytest tests/nonfunctional/test_performance.py -s -m slow "
    "(needs data/raw/flickr30k, which is gitignored and not in a fresh clone)"
)


def _sample_real_images(n, seed=42):
    files = sorted(DATA_DIR.glob("*.jpg"))
    if len(files) < n:
        pytest.skip(f"need {n} images under {DATA_DIR}, found {len(files)}")
    return random.Random(seed).sample(files, n)


@pytest.mark.skipif(not RUN_NFR or not DATA_DIR.exists(), reason=_SKIP_REASON)
def test_performance_stress(api_client):
    from api import server as srv

    images = _sample_real_images(1 + BATCH_N)
    single_path, batch_paths = images[0], images[1:]

    def _post(path):
        data = path.read_bytes()
        return api_client.post(
            "/api/analyze",
            files={"file": (path.name, data, "image/jpeg")},
        )

    # ── single image ────────────────────────────────────────────────────
    t0 = time.perf_counter()
    r = _post(single_path)
    single_time = time.perf_counter() - t0
    assert r.status_code == 200, r.text

    # ── batch of BATCH_N ────────────────────────────────────────────────
    t0 = time.perf_counter()
    for p in batch_paths:
        r = _post(p)
        assert r.status_code == 200, r.text
    batch_total = time.perf_counter() - t0
    batch_avg = batch_total / len(batch_paths)

    single_pass = single_time < SINGLE_LIMIT
    batch_pass  = batch_avg < BATCH_LIMIT

    report = {
        "device":               str(srv.DEVICE),
        "gpu_name":             torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cpu":                  platform.processor() or platform.machine(),
        "checkpoint":           str(srv.CHECKPOINT),
        "single_image_seconds": round(single_time, 4),
        "single_image_file":    single_path.name,
        "batch_size":           len(batch_paths),
        "batch_total_seconds":  round(batch_total, 4),
        "batch_avg_seconds":    round(batch_avg, 4),
        "single_limit":         SINGLE_LIMIT,
        "batch_limit":          BATCH_LIMIT,
        "single_pass":          single_pass,
        "batch_pass":           batch_pass,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "performance.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    assert single_pass, f"single image {single_time:.4f}s >= {SINGLE_LIMIT}s limit"
    assert batch_pass, f"batch average {batch_avg:.4f}s/image >= {BATCH_LIMIT}s/image limit"

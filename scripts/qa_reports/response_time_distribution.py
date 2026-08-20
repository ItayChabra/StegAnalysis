"""
response_time_distribution.py — Chapter 10.2 row 1 (Performance), repeated-
measurement follow-up to the earlier single-trial stress test.

Hits the real POST /api/analyze route (full pipeline: multipart parse ->
sliding-window SRNet inference -> heatmap PNG -> SRM residual -> JSON) via
FastAPI's in-process TestClient — same code path as api/server.py, no
network layer, no mocking.

Collects:
  - 30 single-image requests, each a distinct real flickr30k photo
  - 5 independent runs of a 100-image batch (500 more distinct photos, no
    overlap with the single-image sample or between runs)

Outputs (scripts/qa_reports/output/):
  - response_time_raw.csv       one row per HTTP request
  - response_time_summary.json  mean/median/stdev/p50/p95/p99 per group
  - response_time_distribution.png

Run:
    python scripts/qa_reports/response_time_distribution.py
"""

import csv
import json
import random
import statistics
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT       = Path(__file__).resolve().parents[2]
DATA_DIR   = ROOT / "data" / "raw" / "flickr30k"
OUT_DIR    = Path(__file__).resolve().parent / "output"

N_SINGLE     = 30
N_BATCH_RUNS = 5
BATCH_SIZE   = 100
SINGLE_LIMIT = 0.6
BATCH_LIMIT  = 0.3
SEED         = 20260817


def percentile(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * (p / 100)
    f, c = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def summarize(vals):
    s = sorted(vals)
    return {
        "n":      len(vals),
        "mean":   round(statistics.fmean(vals), 4),
        "median": round(statistics.median(vals), 4),
        "stdev":  round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0,
        "min":    round(min(vals), 4),
        "max":    round(max(vals), 4),
        "p50":    round(percentile(s, 50), 4),
        "p95":    round(percentile(s, 95), 4),
        "p99":    round(percentile(s, 99), 4),
    }


def main():
    from fastapi.testclient import TestClient
    from api.server import app

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(DATA_DIR.glob("*.jpg"))
    need = N_SINGLE + N_BATCH_RUNS * BATCH_SIZE
    assert len(files) >= need, f"need {need} images, found {len(files)}"
    rng = random.Random(SEED)
    sample = rng.sample(files, need)
    single_files = sample[:N_SINGLE]
    batch_files = [sample[N_SINGLE + i * BATCH_SIZE: N_SINGLE + (i + 1) * BATCH_SIZE]
                   for i in range(N_BATCH_RUNS)]

    rows = []  # request_type, run_id, index, filename, latency_seconds, status_code

    with TestClient(app) as client:
        def _post(path):
            data = path.read_bytes()
            t0 = time.perf_counter()
            r = client.post("/api/analyze", files={"file": (path.name, data, "image/jpeg")})
            dt = time.perf_counter() - t0
            return r.status_code, dt

        # warm-up call (excluded from stats — model already loaded by lifespan,
        # this just primes CUDA kernels/allocator so run 1 isn't penalised
        # relative to runs 2-5).
        _post(single_files[0])

        print(f"[single] running {N_SINGLE} requests...")
        for i, p in enumerate(single_files):
            status, dt = _post(p)
            rows.append(("single", 0, i, p.name, dt, status))
            assert status == 200

        for run_id in range(N_BATCH_RUNS):
            print(f"[batch run {run_id + 1}/{N_BATCH_RUNS}] running {BATCH_SIZE} requests...")
            for i, p in enumerate(batch_files[run_id]):
                status, dt = _post(p)
                rows.append(("batch", run_id + 1, i, p.name, dt, status))
                assert status == 200

    with open(OUT_DIR / "response_time_raw.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["request_type", "run_id", "index", "filename", "latency_seconds", "status_code"])
        w.writerows(rows)

    single_lat = [r[4] for r in rows if r[0] == "single"]
    batch_lat  = [r[4] for r in rows if r[0] == "batch"]
    batch_by_run = {
        run_id: [r[4] for r in rows if r[0] == "batch" and r[1] == run_id]
        for run_id in range(1, N_BATCH_RUNS + 1)
    }

    single_summary = summarize(single_lat)
    batch_summary  = summarize(batch_lat)
    batch_run_summaries = {k: summarize(v) for k, v in batch_by_run.items()}

    single_pass_count = sum(1 for v in single_lat if v < SINGLE_LIMIT)
    batch_run_avg_pass = {k: (statistics.fmean(v) < BATCH_LIMIT) for k, v in batch_by_run.items()}

    summary = {
        "n_single_requests": len(single_lat),
        "n_batch_requests": len(batch_lat),
        "single_limit_seconds": SINGLE_LIMIT,
        "batch_limit_seconds": BATCH_LIMIT,
        "single_stats": single_summary,
        "batch_per_image_stats": batch_summary,
        "batch_run_stats": batch_run_summaries,
        "single_requests_under_threshold": single_pass_count,
        "single_requests_total": len(single_lat),
        "single_pass_rate": round(single_pass_count / len(single_lat), 4),
        "batch_run_avg_pass": batch_run_avg_pass,
        "batch_runs_passing_avg_threshold": sum(batch_run_avg_pass.values()),
        "batch_runs_total": N_BATCH_RUNS,
    }
    (OUT_DIR / "response_time_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # ── chart ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.hist(single_lat, bins=15, alpha=0.75, color="#3b7dd8", label="single-image (n=30)")
    ax.hist(batch_lat, bins=30, alpha=0.55, color="#e07b39", label="per-image in batch (n=500)")
    ax.axvline(SINGLE_LIMIT, color="#3b7dd8", linestyle="--", linewidth=1.5,
               label=f"single threshold {SINGLE_LIMIT}s")
    ax.axvline(BATCH_LIMIT, color="#e07b39", linestyle="--", linewidth=1.5,
               label=f"batch threshold {BATCH_LIMIT}s")
    ax.set_xlabel("Latency (seconds)")
    ax.set_ylabel("Request count")
    ax.set_title("POST /api/analyze latency distribution")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    bp = ax2.boxplot([single_lat, batch_lat], labels=["single\n(n=30)", "batch per-image\n(n=500)"],
                      showmeans=True, patch_artist=True)
    for patch, color in zip(bp["boxes"], ["#3b7dd8", "#e07b39"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax2.axhline(SINGLE_LIMIT, color="#3b7dd8", linestyle="--", linewidth=1.2)
    ax2.axhline(BATCH_LIMIT, color="#e07b39", linestyle="--", linewidth=1.2)
    ax2.set_ylabel("Latency (seconds)")
    ax2.set_title("Single vs. batch latency spread")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "response_time_distribution.png", dpi=150)
    print(f"chart written to {OUT_DIR / 'response_time_distribution.png'}")


if __name__ == "__main__":
    main()

"""
availability_load_test.py — Chapter 10.2 service-availability NFR.

Starts the REAL uvicorn server (api.server:app, single worker — the exact
command documented in CLAUDE.md/README, no --workers flag) as a subprocess,
then drives it over real HTTP (not TestClient — genuine sockets, so
connection-level failures are actually observable) with a fixed concurrency
for a fixed wall-clock window.

Parameters chosen for this hardware (4-core host, one shared GPU partition —
see tests/nonfunctional/results/performance.json for the exact GPU) and for a
demo-scale app, not a stress-to-destruction test:
  - concurrency = 4   (matches host core count; the server is single-worker,
                        so this also tests how the async route behaves when
                        more requests arrive than one process can truly run
                        in parallel — see interpretation in the summary)
  - duration    = 180s (long enough to accumulate a statistically meaningful
                        request count without turning a demo box into a
                        space heater)
  - per-request timeout = 15s (>>25x the ~0.3-0.6s measured typical latency;
                        generous enough that a timeout means something is
                        actually wrong, not just queuing behind concurrency)

Reports the metric it actually measures: request SUCCESS RATE over the test
window (successful HTTP 200s / total attempted requests), plus a breakdown of
non-2xx / timeout / connection-error outcomes and whether the server process
was still alive and answering /health at the end. This is explicitly NOT
"uptime" (no continuous external monitoring, no SLA window) — see summary
JSON field names.

Outputs (scripts/qa_reports/output/):
  - availability_raw.csv
  - availability_summary.json
  - availability_timeline.png

Run:
    python -m scripts.qa_reports.availability_load_test
"""

import csv
import json
import random
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT      = Path(__file__).resolve().parents[2]
DATA_DIR  = ROOT / "data" / "raw" / "flickr30k"
OUT_DIR   = Path(__file__).resolve().parent / "output"

PORT           = 8011
BASE_URL       = f"http://127.0.0.1:{PORT}"
CONCURRENCY    = 4
DURATION_S     = 180
REQUEST_TIMEOUT_S = 15.0
STARTUP_TIMEOUT_S = 60.0


def _wait_for_server(proc, timeout_s):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server process exited early with code {proc.returncode}")
        try:
            with httpx.Client(timeout=2.0) as c:
                r = c.get(f"{BASE_URL}/health")
                if r.status_code == 200:
                    return
        except (httpx.ConnectError, httpx.ReadTimeout, socket.error):
            pass
        time.sleep(0.5)
    raise RuntimeError("server did not become healthy in time")


def _worker(worker_id, files, stop_at, rows, rows_lock, rng_seed):
    rng = random.Random(rng_seed)
    with httpx.Client(timeout=REQUEST_TIMEOUT_S) as client:
        while time.time() < stop_at:
            path = rng.choice(files)
            data = path.read_bytes()
            t0 = time.perf_counter()
            wall_t0 = time.time()
            outcome, status = "ok", None
            try:
                r = client.post(f"{BASE_URL}/api/analyze",
                                 files={"file": (path.name, data, "image/jpeg")})
                status = r.status_code
                outcome = "success" if r.status_code == 200 else "http_error"
            except httpx.TimeoutException:
                outcome = "timeout"
            except httpx.ConnectError:
                outcome = "connection_error"
            except httpx.HTTPError:
                outcome = "http_client_error"
            dt = time.perf_counter() - t0
            with rows_lock:
                rows.append((worker_id, wall_t0, dt, outcome, status, path.name))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(DATA_DIR.glob("*.jpg"))[:2000]
    assert files, f"no images found under {DATA_DIR}"

    print(f"[server] starting uvicorn on port {PORT} ...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.server:app",
         "--port", str(PORT), "--host", "127.0.0.1", "--log-level", "warning"],
        cwd=str(ROOT),
    )
    try:
        _wait_for_server(proc, STARTUP_TIMEOUT_S)
        print("[server] healthy, starting load test "
              f"({CONCURRENCY} workers x {DURATION_S}s, {REQUEST_TIMEOUT_S}s timeout)")

        rows = []
        rows_lock = threading.Lock()
        start = time.time()
        stop_at = start + DURATION_S
        threads = [
            threading.Thread(target=_worker, args=(i, files, stop_at, rows, rows_lock, 1000 + i))
            for i in range(CONCURRENCY)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall_elapsed = time.time() - start

        # crash / restart check: process still running + still healthy right after load
        server_alive = proc.poll() is None
        try:
            with httpx.Client(timeout=5.0) as c:
                post_test_health = c.get(f"{BASE_URL}/health").status_code == 200
        except httpx.HTTPError:
            post_test_health = False

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    with open(OUT_DIR / "availability_raw.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["worker_id", "wall_time", "latency_seconds", "outcome", "status_code", "filename"])
        w.writerows(rows)

    n_total = len(rows)
    outcome_counts = {}
    for r in rows:
        outcome_counts[r[3]] = outcome_counts.get(r[3], 0) + 1
    n_success = outcome_counts.get("success", 0)
    success_rate = n_success / n_total if n_total else 0.0

    summary = {
        "methodology": (
            "success_rate = successful HTTP 200 responses / total requests attempted "
            "during a fixed wall-clock load window. This is a request-success-rate "
            "measurement, NOT continuous-uptime/SLA availability -- no external "
            "monitor, no probing outside the load window."
        ),
        "port": PORT,
        "concurrency": CONCURRENCY,
        "duration_target_s": DURATION_S,
        "duration_actual_s": round(wall_elapsed, 2),
        "request_timeout_s": REQUEST_TIMEOUT_S,
        "n_requests_total": n_total,
        "outcome_counts": outcome_counts,
        "success_rate": round(success_rate, 6),
        "success_rate_percent": round(success_rate * 100, 4),
        "throughput_req_per_s": round(n_total / wall_elapsed, 3) if wall_elapsed else 0,
        "server_process_alive_after_test": server_alive,
        "health_check_after_test_ok": post_test_health,
        "server_crashed_or_needed_restart": not server_alive,
    }
    (OUT_DIR / "availability_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # ── chart ────────────────────────────────────────────────────────────
    colors = {"success": "#3b7dd8", "http_error": "#e07b39",
              "timeout": "#c0392b", "connection_error": "#7f1d1d",
              "http_client_error": "#7f1d1d"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    for outcome in sorted(outcome_counts):
        xs = [r[1] - start for r in rows if r[3] == outcome]
        ys = [r[2] for r in rows if r[3] == outcome]
        ax.scatter(xs, ys, s=10, alpha=0.6, label=f"{outcome} (n={len(xs)})",
                   color=colors.get(outcome, "gray"))
    ax.set_xlabel("Elapsed time in load window (seconds)")
    ax.set_ylabel("Request latency (seconds)")
    ax.set_title(f"Requests over {DURATION_S}s window, {CONCURRENCY} concurrent workers")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    labels = list(outcome_counts.keys())
    counts = [outcome_counts[k] for k in labels]
    bar_colors = [colors.get(k, "gray") for k in labels]
    ax2.bar(labels, counts, color=bar_colors, alpha=0.85)
    for i, c in enumerate(counts):
        ax2.text(i, c, str(c), ha="center", va="bottom", fontsize=9)
    ax2.set_ylabel("Request count")
    ax2.set_title(f"Outcome breakdown (n={n_total}, success rate {success_rate*100:.2f}%)")
    plt.setp(ax2.get_xticklabels(), rotation=20, ha="right")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "availability_timeline.png", dpi=150)
    print(f"chart written to {OUT_DIR / 'availability_timeline.png'}")


if __name__ == "__main__":
    main()

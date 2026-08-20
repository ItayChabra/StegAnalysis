"""
Non-functional test — Chapter 10.2 row 2 (Reliability / Accuracy — Blind
Witness Set).

Assembles a 1000-image labelled set (500 clean + 500 stego, 100 per generator
across lsb/dct/fft/adaptive/steganogan) from the project's own held-out TEST
split (dataset_split.json — never seen in training/validation), runs the
deployed checkpoint (srnet_steganogan_best.pth, the same weights api/server.py
loads) through the exact sliding-window inference function the API uses
(api.server._run_inference), and reports accuracy/recall/precision/F1/
confusion matrix against the criteria (Accuracy >= 90%, Recall >= 85%).

"Positive" (predicted stego) = verdict == 'STEGO_DETECTED' (max window score
> 0.80), matching the only state the frontend renders as "HIDDEN DATA FOUND".
A secondary "loose" reading that also counts 'SUSPICIOUS' as positive is
reported alongside for transparency, since it is not the primary criterion.

Requires dataset_split.json's source images on disk (data/raw/, gitignored —
not present in a fresh clone), so it is skipped unless RUN_NFR_TESTS=1 is set
and the split file's images resolve. This keeps the default `pytest` run
(141 backend / 56 frontend, see README) untouched.

Run explicitly:
    RUN_NFR_TESTS=1 pytest tests/nonfunctional/test_reliability_witness_set.py -s -m slow
"""

import json
import os
import random
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

pytestmark = pytest.mark.slow

RUN_NFR     = os.environ.get("RUN_NFR_TESTS") == "1"
ROOT        = Path(__file__).resolve().parents[2]
SPLIT_FILE  = ROOT / "dataset_split.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

N_CLEAN    = 500
N_STEGO    = 500
METHODS    = ["lsb", "dct", "fft", "adaptive", "steganogan"]
PER_METHOD = N_STEGO // len(METHODS)
SEED       = 1234

ACC_THRESHOLD = 0.90
REC_THRESHOLD = 0.85

_SKIP_REASON = (
    "non-functional reliability test — run explicitly with "
    "RUN_NFR_TESTS=1 pytest tests/nonfunctional/test_reliability_witness_set.py -s -m slow "
    "(needs dataset_split.json's source images under data/, gitignored / not in a fresh clone)"
)


def _test_split_pool():
    d = json.loads(SPLIT_FILE.read_text())
    paths = d["lossless_test"] + d["lossy_test"]
    return [p for p in paths if Path(p).exists()]


def _method_config(method):
    if method == "adaptive":
        # canonical=True matches production wiring (evolution.py/validate.py/
        # finetune.py/evaluate.py all pass this; run()'s own default is False).
        return dict(gen_type="adaptive", canonical=True)
    return dict(gen_type=method)


@pytest.mark.skipif(
    not RUN_NFR or not SPLIT_FILE.exists(),
    reason=_SKIP_REASON,
)
def test_reliability_witness_set(api_client):
    # api_client fixture triggers the app lifespan, loading the real
    # checkpoint into api.server._model before we call _run_inference directly.
    from api import server as srv
    from generators.unified_generator import UnifiedGenerator

    pool = _test_split_pool()
    if len(pool) < N_CLEAN + N_STEGO:
        pytest.skip(f"need {N_CLEAN + N_STEGO} test-split images on disk, found {len(pool)}")

    rng = random.Random(SEED)
    rng.shuffle(pool)
    clean_paths        = pool[:N_CLEAN]
    stego_source_paths = pool[N_CLEAN:N_CLEAN + N_STEGO]
    method_assignment  = [m for m in METHODS for _ in range(PER_METHOD)]

    gen = UnifiedGenerator()

    y_true, y_pred_strict, y_pred_loose, max_scores = [], [], [], []
    per_method = {m: {"n": 0, "detected_strict": 0, "detected_loose": 0} for m in METHODS}
    gen_failures = []

    def _classify(image):
        result = srv._run_inference(image)
        strict = 1 if result["verdict"] == "STEGO_DETECTED" else 0
        loose  = 1 if result["verdict"] in ("STEGO_DETECTED", "SUSPICIOUS") else 0
        return strict, loose, result["max_score"]

    # ── clean bucket ────────────────────────────────────────────────────────
    clean_source_counter = Counter()
    for p in clean_paths:
        clean_source_counter[Path(p).parent.name] += 1
        strict, loose, score = _classify(Image.open(p))
        y_true.append(0)
        y_pred_strict.append(strict)
        y_pred_loose.append(loose)
        max_scores.append(score)

    # ── stego bucket ────────────────────────────────────────────────────────
    for path, method in zip(stego_source_paths, method_assignment):
        cfg = _method_config(method)
        arr, psnr = gen.generate_stego(path, None, cfg)
        if arr is None:
            gen_failures.append({"path": path, "method": method})
            continue
        strict, loose, score = _classify(Image.fromarray(arr, mode="L"))
        y_true.append(1)
        y_pred_strict.append(strict)
        y_pred_loose.append(loose)
        max_scores.append(score)
        per_method[method]["n"] += 1
        per_method[method]["detected_strict"] += strict
        per_method[method]["detected_loose"] += loose

    def _metrics(y_pred):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        n  = len(y_true)
        accuracy  = (tp + tn) / n if n else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        return {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy":  round(accuracy, 4),
            "recall":    round(recall, 4),
            "precision": round(precision, 4),
            "f1":        round(f1, 4),
        }

    strict_metrics = _metrics(y_pred_strict)
    loose_metrics  = _metrics(y_pred_loose)

    acc_pass = strict_metrics["accuracy"] >= ACC_THRESHOLD
    rec_pass = strict_metrics["recall"] >= REC_THRESHOLD

    report = {
        "checkpoint": str(srv.CHECKPOINT),
        "device": str(srv.DEVICE),
        "n_clean": len(clean_paths),
        "n_stego_requested": N_STEGO,
        "n_stego_generated": sum(v["n"] for v in per_method.values()),
        "gen_failures": gen_failures,
        "clean_source_mix": dict(clean_source_counter),
        "stego_method_mix": {m: PER_METHOD for m in METHODS},
        "primary_definition": "positive = verdict == STEGO_DETECTED (max_score > 0.80)",
        "primary_metrics": strict_metrics,
        "secondary_definition": "positive = verdict in {STEGO_DETECTED, SUSPICIOUS} (max_score > 0.50)",
        "secondary_metrics": loose_metrics,
        "per_method_detection_strict": {
            m: {"n": v["n"], "detected": v["detected_strict"],
                "rate": round(v["detected_strict"] / v["n"], 4) if v["n"] else None}
            for m, v in per_method.items()
        },
        "acc_threshold": ACC_THRESHOLD,
        "rec_threshold": REC_THRESHOLD,
        "acc_pass": acc_pass,
        "rec_pass": rec_pass,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "reliability_witness_set.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    assert acc_pass, f"accuracy {strict_metrics['accuracy']:.4f} < {ACC_THRESHOLD} threshold"
    assert rec_pass, f"recall {strict_metrics['recall']:.4f} < {REC_THRESHOLD} threshold"

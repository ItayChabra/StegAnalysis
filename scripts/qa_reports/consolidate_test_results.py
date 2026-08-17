"""
consolidate_test_results.py — Chapter 10.2 success/failure rate breakdown.

Parses REAL, freshly-generated JUnit XML (backend pytest, both the default
fast run and the RUN_NFR_TESTS=1 slow run) and vitest's JSON reporter output
(frontend) — no hand-typed counts — into one consolidated dataset, then
charts pass/fail by category (backend / frontend / non-functional) and by
individual test file.

Expects these to already exist (produced by run_and_consolidate() or by
running the commands printed below yourself):
  scripts/qa_reports/output/junit_backend_fast.xml
  scripts/qa_reports/output/junit_backend_slow.xml
  scripts/qa_reports/output/vitest_results.json

Outputs (scripts/qa_reports/output/):
  - consolidated_results.csv         one row per test (category, file, name, status)
  - consolidated_summary.json        counts by category and by file
  - test_results_breakdown.png

Run:
    python -m scripts.qa_reports.consolidate_test_results
"""

import csv
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT    = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "output"

JUNIT_FAST = OUT_DIR / "junit_backend_fast.xml"
JUNIT_SLOW = OUT_DIR / "junit_backend_slow.xml"
VITEST_JSON = OUT_DIR / "vitest_results.json"


def run_pytest_and_vitest():
    """Actually execute the suites fresh so the XML/JSON below is real, not stale."""
    import os
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[pytest] running default (fast) backend suite ...")
    subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q",
         "--ignore=tests/nonfunctional",
         f"--junitxml={JUNIT_FAST}"],
        cwd=str(ROOT), check=False,
    )
    print("[pytest] running the 4 fast tests/nonfunctional files (extensibility/security) ...")
    subprocess.run(
        [sys.executable, "-m", "pytest",
         "tests/nonfunctional/test_extensibility.py",
         "tests/nonfunctional/test_security_malformed_upload.py",
         "-q", f"--junitxml={OUT_DIR / 'junit_nonfunctional_fast.xml'}"],
        cwd=str(ROOT), check=False,
    )
    print("[pytest] running the 2 slow (RUN_NFR_TESTS=1) rows — performance + reliability ...")
    env = dict(**os.environ, RUN_NFR_TESTS="1")
    subprocess.run(
        [sys.executable, "-m", "pytest",
         "tests/nonfunctional/test_performance.py",
         "tests/nonfunctional/test_reliability_witness_set.py",
         "-q", "-m", "slow", f"--junitxml={JUNIT_SLOW}"],
        cwd=str(ROOT), env=env, check=False,
    )

    print("[vitest] running frontend suite ...")
    subprocess.run(
        ["npm", "test", "--", "--reporter=json", f"--outputFile={VITEST_JSON}"],
        cwd=str(ROOT / "frontend"), check=False,
    )


def _category_for_backend_file(filepath: str) -> str:
    return "non-functional" if "nonfunctional" in filepath else "backend"


def _parse_junit(path: Path):
    if not path.exists():
        return []
    tree = ET.parse(path)
    root = tree.getroot()
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    out = []
    for suite in suites:
        for case in suite.findall("testcase"):
            classname = case.get("classname", "")
            name = case.get("name", "")
            filepath = classname.replace(".", "/") + ".py"
            if case.find("failure") is not None or case.find("error") is not None:
                status = "fail"
            elif case.find("skipped") is not None:
                status = "skip"
            else:
                status = "pass"
            out.append({
                "category": _category_for_backend_file(filepath),
                "file": filepath,
                "test": name,
                "status": status,
            })
    return out


def _parse_vitest(path: Path):
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    out = []
    for result in data.get("testResults", []):
        filepath = result.get("name", "unknown")
        rel = filepath.split("frontend/")[-1] if "frontend/" in filepath else filepath
        for assertion in result.get("assertionResults", []):
            status = {"passed": "pass", "failed": "fail", "pending": "skip",
                      "skipped": "skip"}.get(assertion.get("status"), "unknown")
            out.append({
                "category": "frontend",
                "file": rel,
                "test": assertion.get("fullName", assertion.get("title", "")),
                "status": status,
            })
    return out


def main():
    run_pytest_and_vitest()

    rows = (
        _parse_junit(JUNIT_FAST)
        + _parse_junit(OUT_DIR / "junit_nonfunctional_fast.xml")
        + _parse_junit(JUNIT_SLOW)
        + _parse_vitest(VITEST_JSON)
    )
    assert rows, "no test results parsed — check the suites actually ran"

    with open(OUT_DIR / "consolidated_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["category", "file", "test", "status"])
        w.writeheader()
        w.writerows(rows)

    by_category = {}
    by_file = {}
    for r in rows:
        c = by_category.setdefault(r["category"], {"pass": 0, "fail": 0, "skip": 0})
        c[r["status"]] = c.get(r["status"], 0) + 1
        f_ = by_file.setdefault(r["file"], {"category": r["category"], "pass": 0, "fail": 0, "skip": 0})
        f_[r["status"]] = f_.get(r["status"], 0) + 1

    summary = {
        "n_total_tests": len(rows),
        "by_category": by_category,
        "by_file": by_file,
    }
    (OUT_DIR / "consolidated_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # ── charts ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    cats = list(by_category.keys())
    passes = [by_category[c].get("pass", 0) for c in cats]
    fails = [by_category[c].get("fail", 0) for c in cats]
    skips = [by_category[c].get("skip", 0) for c in cats]
    ax = axes[0]
    ax.bar(cats, passes, label="pass", color="#3b7dd8")
    ax.bar(cats, fails, bottom=passes, label="fail", color="#c0392b")
    ax.bar(cats, skips, bottom=[p + fl for p, fl in zip(passes, fails)],
           label="skip", color="#bdbdbd")
    for i, c in enumerate(cats):
        total = passes[i] + fails[i] + skips[i]
        ax.text(i, total + 1, str(total), ha="center", fontsize=9)
    ax.set_ylabel("Test count")
    ax.set_title("Pass/fail by category")
    ax.legend(fontsize=8)

    files_sorted = sorted(by_file.items(), key=lambda kv: (-kv[1]["fail"], kv[0]))
    labels = [Path(f).name for f, _ in files_sorted]
    fp = [v["pass"] for _, v in files_sorted]
    ff = [v["fail"] for _, v in files_sorted]
    fs = [v["skip"] for _, v in files_sorted]
    ax2 = axes[1]
    y = range(len(labels))
    ax2.barh(y, fp, label="pass", color="#3b7dd8")
    ax2.barh(y, ff, left=fp, label="fail", color="#c0392b")
    ax2.barh(y, fs, left=[a + b for a, b in zip(fp, ff)], label="skip", color="#bdbdbd")
    ax2.set_yticks(list(y))
    ax2.set_yticklabels(labels, fontsize=7)
    ax2.invert_yaxis()
    ax2.set_xlabel("Test count")
    ax2.set_title("Pass/fail by test file")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "test_results_breakdown.png", dpi=150)
    print(f"chart written to {OUT_DIR / 'test_results_breakdown.png'}")


if __name__ == "__main__":
    main()

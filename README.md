# StegAnalysis — AI-Powered Steganography Detection

A full-stack system that detects hidden data in images. A custom triple-branch
CNN (**SRNet**) is trained adversarially against an evolutionary algorithm that
breeds steganography generators to fool it, and the result is served through a
FastAPI backend and a React frontend built for a live, non-technical demo.

```
  cover image ──► [ generator population ]──► stego image
                          ▲                        │
                          │  fool rate             ▼
                    [ evolutionary  ]◄────── [ SRNet detector ]
                    [   algorithm   ]              │
                                                   ▼
                                     P(hidden data) + suspicion map
```

---

## Table of contents

- [What it does](#what-it-does)
- [Quick start](#quick-start)
- [Repository layout](#repository-layout)
- [Steganography methods](#steganography-methods)
- [The detector](#the-detector)
- [Training pipeline](#training-pipeline)
- [Results](#results)
- [Benchmarking](#benchmarking)
- [API reference](#api-reference)
- [Frontend](#frontend)
- [Model checkpoints](#model-checkpoints)
- [Credits & licence](#credits--licence)

---

## What it does

The system has three user-facing flows, all backed by the same detector:

| Flow | What the user does | What happens |
|---|---|---|
| **Analyze** | Upload any image | SRNet scans it with a 256×256 sliding window and returns a suspicion score, a heatmap, a residual "what the model sees" view, and an FFT spectrum |
| **Embed** | Upload an image + a message | The message is (optionally) encrypted and embedded via LSB / DCT / FFT / S-UNIWARD; returns the stego image, PSNR and capacity |
| **Extract** | Upload a stego image + passphrase | Auto-detects the embedding method, decodes the framed payload and decrypts it |

Everything the audience sees is expressed in plain English — the internal ML
terminology is deliberately kept out of the UI (see the glossary in
[`CLAUDE.md`](CLAUDE.md#3-domain-glossary)).

---

## Quick start

### Prerequisites

- Python 3.10+
- Node 18+
- An NVIDIA GPU is recommended for training; inference runs fine on CPU

### 1. Environment

```bash
./setup.sh            # venv + CUDA torch + deps + folder structure + smoke test
source .venv/bin/activate
```

Or manually:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Backend

```bash
uvicorn api.server:app --reload --port 8000
```

### 3. Frontend

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173
```

### 4. CLI demo (no web UI)

```bash
python class_demo.py --image path/to/image.png
```

---

## Repository layout

```
├── main.py                        # training entry point → training/train_hybrid.py
├── test_kaggle.py                 # sliding-window benchmark + threshold/mode sweep
├── class_demo.py                  # CLI detection demo
├── setup.sh                       # one-shot environment bootstrap
│
├── models/
│   └── srnet.py                   # SRNet architecture (frozen — do not modify)
│
├── generators/                    # steganography embedders
│   ├── base_generator.py          # abstract base class
│   ├── lsb_gen.py                 # spatial LSB (random / sequential / skip)
│   ├── dct_gen.py                 # block-DCT coefficient embedding
│   ├── fft_gen.py                 # global FFT ring embedding
│   ├── adaptive_gen.py            # S-UNIWARD (simplified + canonical db8 path)
│   ├── steganogan_gen.py          # GAN-learned embedding (SteganoGAN)
│   ├── steganogan_src/            # vendored DAI-Lab networks (MIT)
│   └── unified_generator.py       # gen_type → generator dispatcher
│
├── training/
│   ├── config.py                  # ALL hyperparameters — tune here only
│   ├── train_hybrid.py            # main adversarial training loop
│   ├── evolution.py               # EA: population, mutation, fitness, niches
│   ├── genome.py                  # genome dataclass + seeding
│   ├── batch.py                   # 8-layer diversity batch construction
│   ├── validate.py                # per-epoch validation (per-generator accuracy)
│   ├── evaluate.py                # post-run evaluation, per-generator AUC
│   ├── finetune.py                # focused fine-tuning (adaptive / SteganoGAN)
│   └── dataset.py, utils.py
│
├── payload/                       # recoverable message codec
│   ├── codec.py                   # 52-byte self-describing header + CRC-32
│   └── crypto.py                  # AES-256-GCM / ChaCha20-Poly1305 / Fernet
│
├── api/
│   └── server.py                  # FastAPI application
│
├── frontend/                      # React + Vite (CSS Modules, no UI libraries)
│   └── src/{pages,components,hooks,api,context}
│
└── scripts/
    ├── convert_steganogan_weights.py   # .steg pickle → plain state_dict
    ├── verify_readme_numbers.py        # matched-cover detection rates (SGAN + adaptive)
    └── inspect_payload.py
```

---

## Steganography methods

Five embedding families are implemented. `capacity_ratio` is **true
bits-per-pixel (bpp)** across all of them — each generator converts bpp into its
own internal quantity.

| Method | Domain | Physical ceiling (256×256) | Recoverable | Evolved by EA |
|---|---|---|---|---|
| **LSB** | Spatial least-significant bits | 1.0 bpp at `bit_depth=1` | ✅ | ✅ |
| **DCT** | 8×8 block frequency coefficients | ~0.31 mid · ~0.19 low_mid · ~0.16 random | ✅ | ✅ |
| **FFT** | Global frequency rings | ~0.28 high · ~0.14 mid · ~0.017 low (hard cap) | ✅ | ✅ |
| **S-UNIWARD** | Adaptive spatial (cost-model driven) | curriculum-set, floor 0.20 bpp | ❌ | shape only |
| **SteganoGAN** | Learned convolutional encoder | fixed by weights | ❌ | ❌ |

Generators cap silently at their ceiling — FFT-low in particular ignores the
requested capacity entirely and always embeds at ~0.017 bpp, so its capacity
genes are nominal.

**Recoverable vs. adversarial.** LSB/DCT/FFT expose `embed_payload` /
`extract_payload` with deterministic positions, so a real message round-trips.
S-UNIWARD and SteganoGAN embed random bits — they exist to produce statistically
realistic stego for training the detector, and are surfaced in the UI as
non-recoverable.

**S-UNIWARD.** `adaptive_gen.py` supports `canonical=True`, which activates the
Daubechies-8 back-convolution cost map matching the reference implementation.
λ is solved against the ternary entropy `Σ H(p_i) = bpp·N`, so capacity is
genuine bpp — verified against reference files (0.20 bpp → ~3.2% pixels changed,
~63 dB PSNR; 0.40 bpp → ~7.1%, ~60 dB).

**SteganoGAN.** The DAI-Lab networks are vendored rather than pip-installed
(upstream pins `torch<2.0`). The pretrained `dense.steg` pickle is converted once
into a plain state_dict via `scripts/convert_steganogan_weights.py`, severing the
pickle/torch-version coupling. The encoder is `data_depth=8, hidden_size=32`;
grayscale covers are replicated to RGB, encoded, and folded back to luminance so
the output matches every other generator's 2-D uint8 contract.

---

## The detector

**SRNet** is a triple-branch residual network. Each 256×256 patch is fed in as a
**2-channel tensor** — channel 0 is the spatial luminance, channel 1 is its
log-FFT magnitude — and the three parallel front-end branches read those channels
separately before merging:

| Branch | Filters | Input | Role |
|---|---|---|---|
| A | 11 × 3×3, **frozen** SRM | spatial | fixed high-pass residual kernels |
| B | 53 × 3×3, learnable | spatial | learned spatial residuals |
| C | 21 × 3×3, learnable (abs) | log-FFT | frequency-domain artifacts |

The three outputs concatenate into an 85-channel map, pass through ten residual
stages, then global average pooling and a 2-way classifier.

> **Important:** SRNet is trained on **luminance only**. Feeding raw R/G/B planes
> is out-of-distribution — the model reads demosaicing artifacts as signal. All
> inference paths (including `api/server.py`) must `convert('L')` first.

Full images are scanned with a **sliding window** (256×256, stride 64) and the
per-patch probabilities are aggregated. Four aggregation modes are swept by the
benchmark (`max`, `mean`, `vote`, `p80`); **`max` is the operating mode** — `mean`
collapses DCT/FFT into the cover distribution and `vote` saturates on covers.

---

## Training pipeline

Training is a co-evolutionary loop: the detector learns from stego images
produced by a population of generator genomes, and the population is bred to
maximise fool rate against the *current* detector.

```bash
python main.py                              # full training run
python training/evaluate.py                 # per-generator AUC breakdown
python training/finetune.py --steganogan-focus   # focused fine-tune
```

**All hyperparameters live in [`training/config.py`](training/config.py).** Edit
only that file for tuning runs.

### Batch construction (8 diversity layers)

`training/batch.py` guarantees coverage rather than trusting random sampling:

| Layer | Constraint | Default |
|---|---|---|
| 1 | Baseline — remaining slots drawn from the EA population at random | rest of batch |
| 2 | Low-capacity floor — hard, near-invisible payloads below 0.12 bpp | 15% of free slots |
| 3 | Per-niche cap — no single niche dominates a batch | 40% of free slots |
| 4 | FFT combined cap — all FFT sub-niches together | 25% of free slots |
| 5 | `fft_low` low-strength floor (strength ≤ 7.5) | 0%, set to 10% in finetune |
| 6 | `dct_low_mid` low-strength floor (strength ≤ 3.5) | 0%, set to 10% in finetune |
| 7 | **Adaptive floor** — S-UNIWARD in every batch | 25% |
| 8 | **SteganoGAN floor** — GAN stego in every batch | 12% |

Layers 7 and 8 inject directly rather than through the EA population: adaptive
evolves only its cost-model *shape* (its payload is set by
`ADAPTIVE_CURRICULUM_SCHEDULE` at embed time), and SteganoGAN has no evolvable
parameters at all (frozen encoder, baked-in payload depth). Both are therefore
kept out of `ALL_GEN_TYPES` / `GEN_TYPE_WEIGHTS` / `ALL_NICHES`.

### Evolutionary algorithm

- **Genomes** encode `gen_type` plus per-method parameters (strategy, capacity,
  strength, frequency band, cost exponent, …).
- **Fitness** = fool rate against the current detector, minus a capacity penalty
  (`CAPACITY_PENALTY_WEIGHT = 0.15`) that stops the EA collapsing every genome
  into its lowest-capacity corner.
- **Niches** keep each method family alive independently, so a single dominant
  strategy cannot wipe out the diversity the detector needs.

### Fine-tuning

`training/finetune.py` runs a frozen-backbone → gentle-full-model schedule with a
strategy sampler weighted by each method's current weakness. Two focus modes:

```bash
python training/finetune.py --adaptive-focus      # → srnet_adaptive_best.pth
python training/finetune.py --steganogan-focus    # → srnet_steganogan_best.pth
```

`--steganogan-focus` dedicates ~45% of each batch to SteganoGAN while
**rehearsing every prior method at a 0.05 weight floor**, which is what keeps the
fine-tune from catastrophically forgetting LSB/DCT/FFT.

---

## Results

All figures below are for **`srnet_steganogan_best.pth`**, sliding window
256×256 / stride 64, aggregation mode **`max`**. Every number is a **detection
rate (%)** — the fraction of images scoring at or above the threshold — so the
columns are directly comparable.

### Headline — `max` @ 0.80

| Target | Detection rate | Source |
|---|---|---|
| Covers (TNR — BOSS&BOWS2, Flickr30k, BOSSbase-256, SGAN) | **96.4%** | `test_kaggle.py`, n=200 |
| LSB | **100.0%** | `test_kaggle.py`, n=200 |
| DCT | **97.5%** | `test_kaggle.py`, n=200 |
| FFT | **97.5%** | `test_kaggle.py`, n=200 |
| SteganoGAN — dense | **97.5%** | `verify_readme_numbers.py`, n=200 |
| SteganoGAN — basic | **93.5%** | `verify_readme_numbers.py`, n=200 |
| SteganoGAN — residual | **88.0%** | `verify_readme_numbers.py`, n=200 |
| S-UNIWARD @ 0.4 bpp | **18.6%** ⚠️ | `verify_readme_numbers.py`, n=2000 |
| S-UNIWARD @ 0.2 bpp | **6.7%** ⚠️ | `verify_readme_numbers.py`, n=2000 |
| **Balanced accuracy** (basic-driven, excludes adaptive and GAN) | **97.4%** | `test_kaggle.py`, n=200 |

`test_kaggle.py` puts SteganoGAN in an informational `gan` group that is excluded
from the threshold sweep, so it reports GAN score *distributions* but no
detection rate. `scripts/verify_readme_numbers.py` re-scores those folders (and
adaptive, at a 10× larger sample) to produce the percentages above.

Two caveats on the pooled TNR: it mixes four cover sets whose difficulty differs
(measured separately at 0.80: `BOSSbase_256` 97.7%, `SGAN cover` 91.5%), and it
includes `BOSSbase_256` at n=200, which is the prefix-sensitive folder described
below. Read it as an aggregate, not as a per-dataset guarantee.

### Threshold sweep — the false-positive trade-off

| Threshold | TNR | TPR basic | LSB | DCT | FFT | bal-acc |
|---|---|---|---|---|---|---|
| 0.50 | 83.8% | 99.8% | 100% | 100% | 99.5% | 91.8% |
| 0.65 | 91.6% | 99.3% | 100% | 99.0% | 99.0% | 95.5% |
| 0.75 | 94.7% | 98.7% | 100% | 97.5% | 98.5% | 96.7% |
| **0.80** | **96.4%** | **98.3%** | **100%** | **97.5%** | **97.5%** | **97.4%** |

### Known limitation: low-payload S-UNIWARD

Adaptive is the weak method, but it is **weak, not inverted** — measured against
its **matched cover** (`BOSSbase_256`, the exact source the SUNI files were
derived from), the signal is correctly ordered and monotonic in payload:

| Set (n=2000) | Median score | Δ vs matched cover |
|---|---|---|
| `BOSSbase_256` (matched cover) | 0.194 | — |
| S-UNIWARD @ 0.2 bpp | 0.237 | **+0.043** |
| S-UNIWARD @ 0.4 bpp | 0.410 | **+0.216** |

The separation is real but small, so detection depends heavily on where the
threshold sits — and the deployed 0.80 threshold is tuned for cover TNR across
*all* datasets, far above where adaptive separates:

| Threshold | TNR (matched cover) | S-UNIWARD 0.2 | S-UNIWARD 0.4 |
|---|---|---|---|
| 0.30 | 66.6% | 42.3% | **59.2%** |
| 0.50 | 82.9% | 26.3% | 44.0% |
| 0.65 | 91.0% | 17.0% | 33.5% |
| 0.80 | 97.7% | 6.7% | 18.6% |

So adaptive detection ranges from **~59% at a permissive threshold to ~19% at the
deployed one** for 0.4 bpp — the cost of that sensitivity is cover TNR collapsing
from 97.7% to 66.6%. There is no single threshold that serves both adaptive and
the low-false-positive requirement, which is exactly why the demo operating point
sacrifices adaptive.

> **Two measurement traps this project hit, recorded so they aren't repeated.**
> **(1) Wrong baseline.** Comparing S-UNIWARD against `BOSS & BOWS2` (a
> *different* cover dataset, median 0.237) instead of its matched `BOSSbase_256`
> cover makes the signal look inverted. It is not — that comparison is invalid.
> **(2) Prefix sampling.** `test_kaggle.py` takes `sorted(glob)[:n]`, and the
> first 200 of the 10,000 BOSSbase-derived files score systematically lower than
> the folder as a whole (cover median 0.105 at n=200 vs 0.194 at n=2000). Small
> `--images` values distort every BOSSbase-derived number, adaptive included; the
> Flickr30k / BOSS&BOWS2 / LSB / DCT / FFT folders are stable to ±0.008.

Closing the adaptive gap is the known research frontier for content-adaptive
embedding at low payload, not a tuning bug. It requires wiring `canonical=True`
S-UNIWARD into the *training* path and retraining — not another weighted
fine-tune. See [`kaggle_bench_conclusion.md`](kaggle_bench_conclusion.md) for the
SteganoGAN fine-tune before/after — noting that its "inverted signal" section
uses the wrong cover baseline and is superseded by the matched-cover table above.

The headline 97.4% balanced accuracy is **basic-driven** — it covers
LSB/DCT/FFT/covers and excludes both adaptive and SteganoGAN by construction.

---

## Benchmarking

```bash
# Full sweep: 4 aggregation modes x 11 thresholds x every target folder
python test_kaggle.py --checkpoint srnet_steganogan_best.pth --images 200

# Matched-cover detection rates for SteganoGAN + a large adaptive sample
python scripts/verify_readme_numbers.py --checkpoint srnet_steganogan_best.pth
```

`test_kaggle.py` covers covers/LSB/DCT/FFT/S-UNIWARD plus three SteganoGAN
encoder variants in two datasets, and prints the best operating point per mode.
`scripts/verify_readme_numbers.py` fills its two gaps: SteganoGAN detection rates
(the `gan` group is excluded from the sweep) and adaptive at a sample large
enough not to be distorted by prefix selection.

Three things worth knowing before trusting a number:

> **Validation accuracy is misleading** during fine-tuning when the batch is
> biased toward hard cases. `training/evaluate.py` also generates its stego
> on-the-fly from our own generators, so it measures in-distribution performance.
> `test_kaggle.py` runs against genuine third-party reference files — judge a
> checkpoint by it, not by val_acc.

> **`--images` is a prefix, not a sample.** Folders are read as
> `sorted(glob)[:n]`, so a low `--images` scores the alphabetically first files.
> For the 10,000-file BOSSbase-derived folders that prefix is not representative;
> use a few thousand when adaptive numbers matter.

> **Compare stego against its matched cover.** Each stego folder has a cover
> folder it was derived from. Scoring it against an unrelated cover set produces
> meaningless deltas — including apparent sign inversions.

Benchmark artifacts in the repo:

- `kaggle_bench_conclusion.md` — SteganoGAN fine-tune before/after (⚠️ its
  adaptive "inverted signal" section uses the wrong cover baseline; see Results)
- `kaggle_bench_finetuned_best.log` / `kaggle_bench_steganogan_best.log` — raw sweeps
- `mega_test_epoch15.log` / `mega_test_best_epoch16.log` — 1500-image runs of the
  pre-SteganoGAN checkpoint
- `steganogan_finetune.log`, `finetune_steganogan_history.json` — training traces

---

## API reference

FastAPI server on `http://localhost:8000`. Strict HTTP REST — no WebSockets or
streaming.

### `POST /api/analyze`
`multipart/form-data` with `file`. Runs synchronous SRNet inference and returns
confidence, metrics, per-window scores, per-plane `bitplane_scores`, a heuristic
`method_hint`, and URLs for the heatmap / noisemap / spectrum.

### `POST /api/embed`

```
file:        [image]
method:      "lsb" | "dct" | "fft" | "adaptive"
message:     text to hide
cipher:      "none" | "aes256gcm" | "chacha20poly1305" | "fernet"
passphrase:  required when cipher != none
# optional per-method params:
strategy / step / bit_depth   (lsb)
coeff_selection / strength    (dct)
freq_band / strength          (fft)
capacity                      (adaptive)
```

Returns `job_id`, `stego_url`, `psnr`, `capacity_bytes`, `used_bytes`,
`recoverable`, and `extract_hint` (the settings needed to reveal it).

### `POST /api/extract`
Recovers a hidden message, auto-detecting LSB/DCT/FFT for default settings.
Non-default `strength`/`step` must be supplied — the embed response's
`extract_hint` carries them. Returns `{message, method, cipher, bytes}`, or a
typed error: `404 no_payload`, `422 bad_key`.

### `POST /api/decrypt`
Second step of the two-stage reveal — decrypts an already-extracted payload
package with a passphrase, so extraction and decryption can fail independently.

### `POST /api/capacity`
Given `file` + `method` (+ params), returns `max_message_bytes` for that image.

### `GET /health`
Liveness probe.

### Image endpoints (per `job_id`, all return PNG)
`/api/original`, `/api/stego`, `/api/heatmap`, `/api/noisemap` (SRM residual),
`/api/spectrum` (log-FFT + band rings), `/api/diff` (cover↔stego pixel diff),
`/api/bitplane/{n}` (bit plane 0–7), `/api/sanitize` (payload-stripped image).
Several accept `?source=stego|cover`.

### Payload codec

`payload/codec.py` frames a **52-byte self-describing header** — magic, method,
cipher, length, salt, nonce, params, CRC-32 — ahead of the ciphertext;
`payload/crypto.py` performs the AEAD/Fernet crypto. This recoverable path is
kept strictly **separate** from the training `run()` / `embed()` paths.

---

## Frontend

React 18 + Vite, routed with `react-router-dom` across `/analyze`, `/embed`,
`/extract`, `/learn`. Dark, clinical, high-contrast signals-intelligence
aesthetic.

Conventions:

- All `fetch()` calls live in `src/api/client.js`; components consume them via
  per-flow state-machine hooks (`useAnalysis`, `useBatchAnalysis`, `useEmbed`,
  `useExtract`, `useCapacity`, `useHistory`) driven by a status enum. Shared
  session state — the history drawer — lives in `context/AppContext.jsx`.
- **CSS Modules only** — no Tailwind, MUI, shadcn or any component library.
  Everything is built from scratch on native CSS custom properties.
- Complex interactions (e.g. the before/after comparison slider) are hand-built
  from raw DOM events plus `requestAnimationFrame`.
- Generated API images are cache-busted with `?t=${Date.now()}`.

---

## Model checkpoints

| File | What it is |
|---|---|
| `srnet_steganogan_best.pth` | **Current best** — SteganoGAN-focus fine-tune (val 88.24%, bal-acc 97.4%) |
| `srnet_finetuned_best.pth` | Previous deployed weights (val 87.21%) — the base the above started from |
| `srnet_best_val.pth` | Best validation checkpoint from the last full training run |
| `srnet_epoch_*.pth` | Full-run epoch checkpoints |
| `srnet_ft_epoch_*.pth` | Fine-tune epoch checkpoints |
| `steganogan_dense.pth` | Converted SteganoGAN dense encoder/decoder state_dict |

Training histories: `training_history.json`, `finetune_history.json`,
`finetune_steganogan_history.json`.

> Checkpoints are committed intentionally — they are part of the project
> deliverable. `models/srnet.py` is a frozen architecture; changing it
> invalidates every checkpoint above.

---

## Credits & licence

- **SteganoGAN** — networks under `generators/steganogan_src/` are vendored from
  [DAI-Lab/SteganoGAN](https://github.com/DAI-Lab/SteganoGAN), MIT licence,
  © 2019 MIT Data To AI Lab.
- **S-UNIWARD** — cost model after Holub, Fridrich & Denemark, *Universal
  distortion function for steganography in an arbitrary domain* (2014).
- **SRNet** — architecture after Boroumand, Chen & Fridrich, *Deep residual
  network for steganalysis of digital images* (2019).
- **Datasets** — BOSSbase 1.01, BOWS2, Flickr30k, plus Kaggle steganography test
  sets. Image data is **not** committed (see `.gitignore`).

Project context and working conventions for contributors live in
[`CLAUDE.md`](CLAUDE.md).
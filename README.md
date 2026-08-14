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
    └── inspect_payload.py
```

---

## Steganography methods

Five embedding families are implemented. `capacity_ratio` is **true
bits-per-pixel (bpp)** across all of them — each generator converts bpp into its
own internal quantity.

| Method | Domain | Physical ceiling | Recoverable | Evolved by EA |
|---|---|---|---|---|
| **LSB** | Spatial least-significant bits | ≤ 1.0 bpp | ✅ | ✅ |
| **DCT** | 8×8 block frequency coefficients | ≈ 0.31 bpp | ✅ | ✅ |
| **FFT** | Global frequency rings | ≈ 0.28 bpp (high) / 0.017 (low) | ✅ | ✅ |
| **S-UNIWARD** | Adaptive spatial (cost-model driven) | curriculum-set | ❌ | shape only |
| **SteganoGAN** | Learned convolutional encoder | fixed by weights | ❌ | ❌ |

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

**SRNet** is a triple-branch convolutional network operating on single-channel
256×256 luminance patches.

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

| Layer | Purpose |
|---|---|
| 1 | Niche cap — no single generator niche dominates a batch |
| 2 | Low-capacity floor — hard, near-invisible payloads always present |
| 3–6 | Per-method weak-spot floors (e.g. `fft_low` low-strength, `dct_lowmid`) |
| 7 | **Adaptive floor** — S-UNIWARD in every batch (`ADAPTIVE_BATCH_FRACTION = 0.25`) |
| 8 | **SteganoGAN floor** — GAN stego in every batch (`STEGANOGAN_BATCH_FRACTION = 0.12`) |

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

Benchmark: `test_kaggle.py`, 200 images per folder, sliding window 256×256 /
stride 64, checkpoint **`srnet_steganogan_best.pth`**, aggregation **`max` @ 0.80**.

| Target | Detection rate |
|---|---|
| Covers (TNR — BOSS&BOWS2, Flickr30k, BOSSbase-256) | **96.4%** |
| LSB | **100.0%** |
| DCT | **97.5%** |
| FFT | **97.5%** |
| SteganoGAN (dense / basic / residual, median score) | **0.95–0.96** |
| S-UNIWARD @ 0.2 / 0.4 bpp | **3.2%** ⚠️ |
| **Balanced accuracy** (basic-driven, excludes adaptive) | **97.4%** |

Threshold sweep (`max` mode) — the accuracy/false-positive trade-off:

| Threshold | TNR | TPR basic | LSB | DCT | FFT | bal-acc |
|---|---|---|---|---|---|---|
| 0.50 | 83.8% | 99.8% | 100% | 100% | 99.5% | 91.8% |
| 0.65 | 91.6% | 99.3% | 100% | 99.0% | 99.0% | 95.5% |
| 0.75 | 94.7% | 98.7% | 100% | 97.5% | 98.5% | 96.7% |
| **0.80** | **96.4%** | **98.3%** | **100%** | **97.5%** | **97.5%** | **97.4%** |

### Known limitation: low-payload S-UNIWARD

**Adaptive detection is the open problem, and this README does not claim
otherwise.** At the tested payloads, S-UNIWARD stego scores sit *below* the cover
distribution (median 0.121 @ 0.2 bpp and 0.181 @ 0.4 bpp, vs. 0.237 for covers) —
an inverted signal with no separating threshold. Detection never exceeds ~25% at
any usable operating point, in this checkpoint or its predecessor.

This is the known research frontier for content-adaptive embedding at low
payload, not a tuning bug. Closing it requires wiring `canonical=True` S-UNIWARD
into the *training* path and retraining — not another weighted fine-tune. See
[`kaggle_bench_conclusion.md`](kaggle_bench_conclusion.md) for the full
before/after analysis.

The headline 97.4% balanced accuracy is **basic-driven** — it covers
LSB/DCT/FFT/covers and excludes adaptive by construction.

---

## Benchmarking

```bash
python test_kaggle.py --checkpoint srnet_steganogan_best.pth --images 200
```

Sweeps four aggregation modes × eleven thresholds across every configured target
folder (covers, LSB, DCT, FFT, S-UNIWARD @ 0.2/0.4 bpp, and three SteganoGAN
encoder variants in two datasets), then prints the best operating point per mode.

> Validation accuracy during fine-tuning is **misleading** when the batch is
> biased toward hard cases, and `training/evaluate.py` generates its stego
> on-the-fly from our own generators (in-distribution). `test_kaggle.py` runs
> against genuine third-party reference files and is the honest metric — always
> judge a checkpoint by it.

Benchmark artifacts in the repo:

- `kaggle_bench_conclusion.md` — before/after analysis of the SteganoGAN fine-tune
- `kaggle_bench_finetuned_best.log` / `kaggle_bench_steganogan_best.log` — raw sweeps
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

### `POST /api/capacity`
Given `file` + `method` (+ params), returns `max_message_bytes` for that image.

### Image endpoints (per `job_id`, all return PNG)
`/api/original`, `/api/stego`, `/api/heatmap`, `/api/noisemap` (SRM residual),
`/api/spectrum` (log-FFT + band rings), `/api/diff` (cover↔stego pixel diff),
`/api/bitplane/{n}` (bit plane 0–7). Several accept `?source=stego|cover`.

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
  `useExtract`) driven by a status enum.
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
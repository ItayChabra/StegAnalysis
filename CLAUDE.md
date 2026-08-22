# CLAUDE.md — AI Steganalysis System

> This file gives Claude Code the context it needs to work effectively on this project.
> Read it fully before writing any code, creating any file, or proposing any architecture.

---

## 1. Project Overview

This is a full-stack **AI-powered steganography detection system** built for a live demonstration. It detects hidden data embedded by three classical techniques (LSB, DCT, FFT), one content-adaptive algorithm (S-UNIWARD), and one GAN-learned embedder (SteganoGAN).

The project consists of:
1. **Backend (Python/PyTorch):** A custom triple-branch convolutional network called **SRNet**.
2. **API (FastAPI):** A REST interface connecting the ML inference to the client.
3. **Generators (Python):** Scripts to embed steganographic payloads for training and demo. `adaptive_gen.py` implements S-UNIWARD with a simplified path and a canonical Daubechies-8 back-convolution path (`canonical=True`); `steganogan_gen.py` wraps a vendored pretrained SteganoGAN encoder.
4. **Training pipeline (Python):** An evolutionary algorithm (EA) breeds generator genomes to maximise fool rate against the current model. See `training/` for all components.
5. **Frontend (React/Vite):** A non-technical, highly visual UI designed to make the model's reasoning intuitive for a general audience.

---

## 2. Repository Structure

```
/ (root)
├── CLAUDE.md                  ← you are here
├── main.py                    ← training entry point (calls training/train_hybrid.py)
├── test_kaggle.py             ← sliding-window benchmark; compares aggregation modes
├── srnet_steganogan_best.pth  ← current best / deployed weights (val_acc ≈ 88.2%)
├── srnet_finetuned_best.pth   ← previous weights; base of the above fine-tune
├── dataset_split.json         ← train/val/test split (seed 42, 70/15/15)
├── training_history.json      ← per-epoch metrics from the last full training run
├── finetune_history.json      ← per-epoch metrics from the last finetune run
├── models/
│   └── srnet.py               ← SRNet architecture (triple-branch CNN) — DO NOT MODIFY
├── generators/
│   ├── base_generator.py      ← abstract base class shared by all generators
│   ├── lsb_gen.py             ← LSB generator (random / sequential / skip strategies)
│   ├── dct_gen.py             ← DCT generator (mid / low_mid / random coeff modes)
│   ├── fft_gen.py             ← FFT generator (low / mid / high freq bands)
│   ├── adaptive_gen.py        ← S-UNIWARD generator; canonical=True enables the
│   │                             Daubechies-8 back-convolution cost map
│   ├── steganogan_gen.py      ← GAN-learned embedding (frozen pretrained encoder)
│   ├── steganogan_src/        ← vendored DAI-Lab networks (MIT)
│   ├── dummy_gen.py           ← minimal reference generator used by the
│   │                             extensibility test (tests/nonfunctional/)
│   └── unified_generator.py   ← dispatcher: routes gen_type to the right generator
├── payload/
│   ├── codec.py                ← self-describing header framing (magic, method,
│   │                              cipher, length, salt, nonce, params, CRC-32)
│   └── crypto.py                ← AEAD/Fernet encrypt-decrypt
├── training/
│   ├── config.py              ← ALL hyperparameters and constants — edit here for tuning
│   ├── train_hybrid.py        ← main training loop (called by main.py)
│   ├── evolution.py           ← EA: genome population, mutation, fitness, niches
│   ├── genome.py              ← genome dataclass and seeding logic
│   ├── batch.py               ← batch construction with diversity layers (8 layers)
│   ├── validate.py            ← per-epoch validation loop
│   ├── evaluate.py            ← post-run evaluation; per-generator AUC breakdown
│   ├── finetune.py            ← focused fine-tuning (--adaptive-focus / --steganogan-focus)
│   ├── dataset.py             ← dataset loading and train/val/test splitting
│   ├── utils.py               ← shared training utilities
│   └── evaluation_results/    ← JSON metrics written by evaluate.py
├── frontend/                  ← React/Vite web application
│   ├── src/
│   │   ├── components/        ← co-located *.test.jsx (Vitest)
│   │   ├── pages/
│   │   ├── hooks/              ← co-located *.test.js (Vitest)
│   │   ├── api/
│   │   └── main.jsx
│   ├── index.html
│   └── package.json
├── api/
│   └── server.py              ← FastAPI server entry point
├── tests/                     ← backend pytest suite (see §9 Testing)
│   ├── conftest.py
│   ├── test_api.py / test_codec.py / test_crypto.py / test_generators.py
│   └── nonfunctional/         ← Chapter 10.2 QA suite (perf, reliability,
│                                 security, extensibility)
├── scripts/qa_reports/        ← QA report generation (charts/CSVs built from
│                                 real pytest + Vitest output, not hand-typed)
├── pytest.ini
└── docs/screenshots/          ← README images
```

---

## 3. Domain Glossary

**Critical Rule for Frontend UI:** Never surface raw ML terminology to the audience. When modifying the UI, everything must pass through this translation table. Internal code and backend logic should use the standard technical terms.

| Internal term | Plain-English meaning | Show in UI as |
|---------------|-----------------------|---------------|
| Cover image | The original, unmodified image | "Original" |
| Stego image | Image with hidden data embedded | "Modified" |
| LSB | Hides data in the least-significant bit of each pixel | "Pixel-level hiding" |
| DCT | Hides data in JPEG frequency coefficients | "JPEG frequency hiding" |
| FFT | Hides data in global frequency rings | "Frequency-domain hiding" |
| S-UNIWARD | Adaptive algorithm that hides data in noisy/textured areas | "Adaptive spatial hiding" |
| SteganoGAN | Trained neural network (GAN) that learns where to hide data | "AI-learned hiding" |
| PSNR | Signal quality metric; >40 dB = visually identical | "Quality score" |
| Sliding window | Backend ML scan technique (256×256 patches) | **Do not mention in UI** |
| P(stego) | Model's probability that a patch contains hidden data | "Suspicion score" |
| Verdict | Final binary decision: clean or stego | "CLEAN" / "HIDDEN DATA FOUND" |
| Noise map | Amplified pixel residual (stego − cover) × 10 | "What the model sees" |
| Heatmap | Jet-colourmap of per-patch suspicion scores | "Suspicion map" |

---

## 4. Training Pipeline

All hyperparameters live in `training/config.py`. Edit only that file for tuning runs.

**Capacity semantics:** `capacity_ratio` is TRUE bits-per-pixel (bpp) across all generators. Each generator translates bpp into its own quantity internally (pixels for LSB, 8×8 blocks for DCT, frequency components for FFT, λ for adaptive). Physical ceilings: LSB ≤ 1.0 bpp, DCT ≤ ~0.31 bpp, FFT-high ≤ ~0.28 bpp, FFT-low ≤ ~0.017 bpp.

**EA capacity penalty:** Re-enabled at `CAPACITY_PENALTY_WEIGHT = 0.15` with per-method thresholds in `CAPACITY_PENALTY_THRESHOLDS`. Prevents floor-collapse (EA maximising fool rate by driving every genome to its lowest-capacity corner).

**Adaptive curriculum:** Adaptive capacity is NOT evolved by the EA — it is set at embed time by `ADAPTIVE_CURRICULUM_SCHEDULE`. The EA only evolves the S-UNIWARD cost-model shape (sigma_offset, cost_exponent).

**Canonical S-UNIWARD:** `adaptive_gen.py` supports `canonical=True`, which activates the Daubechies-8 back-convolution cost map matching the reference implementation. The generator's own default is `canonical=False` (simplified path), but **the training pipeline already passes `canonical=True` everywhere** — `evolution.py` (both adaptive genome constructors), `validate.py`, `finetune.py` and `evaluate.py`. `train_hybrid.py` prints this at startup. Do not treat canonical wiring as outstanding work.

**SteganoGAN:** frozen pretrained encoder with no evolvable parameters, so it is kept out of `ALL_GEN_TYPES` / `GEN_TYPE_WEIGHTS` / `ALL_NICHES` and injected via a batch floor (`STEGANOGAN_BATCH_FRACTION`) plus a fixed validation share. Same pattern as the adaptive floor.

**Judging a checkpoint:** validation accuracy is unreliable when the batch is weighted toward hard cases, and `evaluate.py` generates its stego from our own generators (in-distribution). Judge by `test_kaggle.py`, which uses third-party reference files. When comparing a stego set against covers, use the *matched* cover folder — the set the stego images were derived from — not an unrelated cover dataset.

---

## 5. API Contract

The FastAPI server runs on `http://localhost:8000`. Communication is strict HTTP REST (no WebSockets/streaming).

### POST `/api/analyze`
Uploads an image (`multipart/form-data`) and runs synchronous SRNet inference. Returns JSON with confidence, metrics, window scores, per-plane `bitplane_scores`, a heuristic `method_hint`, and URLs for heatmap / noisemap / spectrum.

### POST `/api/embed`
Embeds an (optionally encrypted) message. Recoverable methods are `lsb`/`dct`/`fft`; `adaptive` (S-UNIWARD) embeds statistical noise only (`recoverable: false`).

**Request (`multipart/form-data`):**
```
file:            [image file]
method:          "lsb" | "dct" | "fft" | "adaptive"
message:         text to hide (recoverable methods)
cipher:          "none" | "aes256gcm" | "chacha20poly1305" | "fernet"
passphrase:      required when cipher != none
# per-method params (optional, defaulted):
strategy/step/bit_depth      (lsb)   coeff_selection/strength (dct)
freq_band/strength           (fft)   capacity (adaptive)
```
Response includes `job_id`, `stego_url`, `psnr`, `capacity_bytes`, `used_bytes`, `recoverable`, and `extract_hint` (the settings needed to reveal it).

### POST `/api/extract`
Reveal-only: locates the hidden payload and returns it. Auto-detects the method (LSB/DCT/FFT) for default settings; non-default `strength`/`step` must be supplied (the embed `extract_hint` carries these). If the payload is unencrypted, the response includes the plaintext `message` directly; otherwise decryption is a separate step so the UI can show the ciphertext before asking for a passphrase. Returns `{method, cipher, cipher_id, encrypted, bytes, ciphertext_b64, salt_b64, nonce_b64, message?}`, or `404 no_payload`.

### POST `/api/decrypt`
Second step of the reveal flow: decrypts a `ciphertext_b64` previously returned by `/api/extract`, given `cipher`, `passphrase`, `salt_b64`, `nonce_b64`. Returns `{message, cipher, encrypted, bytes}`, or a typed error: `400` malformed base64/cipher name, `422 bad_key`.

### POST `/api/capacity`
Given `file` + `method` (+ params), returns the max recoverable payload (`max_message_bytes`) for that image.

### Recoverable codec (backend)
`payload/codec.py` frames a 52-byte self-describing header (magic, method, cipher, length, salt, nonce, params, CRC-32) ahead of the ciphertext; `payload/crypto.py` does the AEAD/Fernet crypto. Each generator exposes `embed_payload`/`extract_payload` (deterministic positions) **separate from** the training `run()`/`embed()` paths, which are unchanged.

### Image GET endpoints (per `job_id`)
`/api/original`, `/api/stego`, `/api/heatmap`, `/api/noisemap` (SRM residual; `?source=stego` in embed flow), `/api/spectrum` (log-FFT + band rings; `?source=`), `/api/diff` (cover↔stego pixel diff, embed flow), `/api/bitplane/{n}` (bit plane 0–7; `?source=`), `/api/sanitize` (Gaussian-blur-based steg-stripped copy of the original). All return PNG.

### GET `/health`
Liveness probe, no auth or job state. Used by `scripts/qa_reports/availability_load_test.py` to confirm the server is up before driving load against it.

---

## 6. Working Conventions

### General Context

- **Target Audience:** The UI and demo are for non-technical users. Visual clarity and straightforward copy are paramount.
- **Theme:** Dark, clinical, high-contrast (radar/signals-intelligence aesthetic).

### Python / Backend (FastAPI, PyTorch)

- **Dependencies:** fastapi, uvicorn, python-multipart, torch, torchvision, Pillow, numpy, scipy, cryptography.
- Maintain strict separation between ML logic (`models/`), dataset generation (`generators/`), and the API layer (`api/`).
- Ensure CORS is configured correctly in `server.py` for local Vite development.

### JavaScript / Frontend (React, Vite)

- **API Calls:** All `fetch()` calls must live in `frontend/src/api/client.js`. Components use custom hooks (e.g., `useAnalysis`) to access these. Throw descriptive errors on non-2xx responses.
- **Styling:** Use standard CSS Modules (`.module.css`). No Tailwind, MUI, Shadcn, or UI component libraries. Build elements from scratch using native CSS custom properties.
- **Caching:** Cache-bust dynamically generated API images by appending timestamps: `?t=${Date.now()}`.
- **Routing:** `react-router-dom` with pages `/analyze`, `/embed`, `/extract`, `/learn` (`frontend/src/pages/`). Shared session state (history drawer) lives in `context/AppContext.jsx`.
- **State Management:** Per-flow state-machine hooks (`useAnalysis`, `useBatchAnalysis`, `useEmbed`, `useExtract`) using a status enum (IDLE, UPLOADING/PROCESSING, COMPLETE, ERROR).
- **Complex UI Elements:** Custom interactive elements (like the Before/After Image Comparison Slider) must be built from scratch using raw DOM events (`onMouseDown`, `onMouseMove`) and `requestAnimationFrame`.

---

## 7. Running the Stack

### Backend (Terminal 1):
```bash
pip install fastapi uvicorn python-multipart torch torchvision pillow numpy scipy cryptography
uvicorn api.server:app --reload --port 8000
```

### Frontend (Terminal 2):
```bash
cd frontend
npm install
npm run dev
# Opens at http://localhost:5173
```

### Benchmark:
```bash
python test_kaggle.py --checkpoint srnet_steganogan_best.pth --images 200
```

---

## 8. Testing

```bash
# Backend — payload codec, crypto, all 5 generators, live API endpoints
pytest                       # 141 tests, CPU-only, no external dataset

# Frontend — config, hooks, components
cd frontend && npm test      # Vitest, 56 tests
```

`tests/conftest.py` generates cover images synthetically (fixed-seed noise) rather than reading from `data/`, so the default suite runs in a fresh clone with no dataset present. The session-scoped `api_client` fixture loads the real `srnet_steganogan_best.pth` checkpoint once and reuses it across every API test.

**`tests/nonfunctional/`** is the Chapter 10.2 non-functional QA suite — one file per row: performance, reliability (blind witness set drawn from the held-out TEST split in `dataset_split.json`), security (malformed/malicious upload), extensibility (adds `generators/dummy_gen.py` as a new generator without touching the dispatcher). Only the fast, no-external-data tests (extensibility, security) run in the default `pytest` invocation; tests needing `data/raw/flickr30k` are skipped unless both the directory exists and `RUN_NFR_TESTS=1` is set:
```bash
RUN_NFR_TESTS=1 pytest tests/nonfunctional/test_performance.py -s -m slow
```

**`scripts/qa_reports/`** turns real, freshly-generated test output (JUnit XML from pytest, Vitest's JSON reporter, live HTTP timing against a real `uvicorn` subprocess) into the charts/CSVs/summary JSON in `scripts/qa_reports/output/` — nothing here is hand-typed. Re-run a script rather than hand-editing its output when the underlying test suite changes.

---

## 9. Critical Protected Files

Unless explicitly instructed otherwise by the user, do not modify the following files:

- `models/srnet.py` — frozen architecture
- `srnet_steganogan_best.pth` — current best / deployed weights
- `srnet_finetuned_best.pth` — previous deployed weights
- `srnet_best_val.pth` — current best after training weights
- any `.pth` file — all checkpoint files are training artifacts
# CLAUDE.md — AI Steganalysis System

> This file gives Claude Code the context it needs to work effectively on this project.
> Read it fully before writing any code, creating any file, or proposing any architecture.

---

## 1. Project Overview

This is a full-stack **AI-powered steganography detection system** built for a live demonstration. The system detects hidden data embedded in images using three basic techniques (LSB, DCT, FFT) and one adaptive algorithm (S-UNIWARD).

The project consists of:
1. **Backend (Python/PyTorch):** A custom triple-branch convolutional network called **SRNet**.
2. **API (FastAPI):** A REST interface connecting the ML inference to the client.
3. **Generators (Python):** Scripts to embed steganographic payloads for training and demo. The adaptive generator (`adaptive_gen.py`) implements S-UNIWARD with both a simplified path (default) and a canonical Daubechies-8 back-convolution path (`canonical=True`).
4. **Training pipeline (Python):** An evolutionary algorithm (EA) breeds generator genomes to maximise fool rate against the current model. See `training/` for all components.
5. **Frontend (React/Vite):** A non-technical, highly visual UI designed to make the model's reasoning intuitive for a general audience.

---

## 2. Repository Structure

```
/ (root)
├── CLAUDE.md                  ← you are here
├── main.py                    ← training entry point (calls training/train_hybrid.py)
├── test_kaggle.py             ← sliding-window benchmark; compares aggregation modes
├── class_demo.py              ← CLI demo (sliding-window detection on a single image)
├── srnet_finetuned_best.pth   ← current best model weights (val_acc ≈ 87.6%)
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
│   └── unified_generator.py   ← dispatcher: routes gen_type to the right generator
├── training/
│   ├── config.py              ← ALL hyperparameters and constants — edit here for tuning
│   ├── train_hybrid.py        ← main training loop (called by main.py)
│   ├── evolution.py           ← EA: genome population, mutation, fitness, niches
│   ├── genome.py              ← genome dataclass and seeding logic
│   ├── batch.py               ← batch construction with diversity layers (7 layers)
│   ├── validate.py            ← per-epoch validation loop
│   ├── evaluate.py            ← post-run evaluation; per-generator AUC breakdown
│   ├── finetune.py            ← head-only fine-tuning on a frozen backbone
│   ├── dataset.py             ← dataset loading and train/val/test splitting
│   ├── utils.py               ← shared training utilities
│   └── evaluation_results/    ← JSON metrics written by evaluate.py
├── frontend/                  ← React/Vite web application
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── hooks/
│   │   ├── api/
│   │   └── main.jsx
│   ├── index.html
│   └── package.json
└── api/
    └── server.py              ← FastAPI server entry point
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

**Canonical S-UNIWARD:** `adaptive_gen.py` supports `canonical=True`, which activates the Daubechies-8 back-convolution cost map that matches the reference implementation. Default is `canonical=False` (simplified path). Wiring `canonical=True` into the training pipeline is required before the model can learn to detect canonical S-UNIWARD images.

---

## 5. API Contract

The FastAPI server runs on `http://localhost:8000`. Communication is strict HTTP REST (no WebSockets/streaming).

### POST `/api/analyze`
Uploads an image (`multipart/form-data`) and runs synchronous SRNet inference. Returns JSON with confidence, metrics, window scores, and URLs for the generated heatmap and noisemap.

### GET `/api/heatmap/{job_id}`
Returns a PNG heatmap of suspicion scores overlaid on the image.

### GET `/api/noisemap/{job_id}`
Returns a PNG of the amplified pixel residual.

### POST `/api/embed`
Demo tool that embeds a payload using one of the generators. Returns the stego image URL and metrics.

**Request:**
```
Content-Type: multipart/form-data
file:       [image file]
strategy:   "lsb_sequential" | "dct_mid" | "fft_mid"
capacity:   0.5   (float, sent as a form field, range 0.1–0.75)
```

**Response:**
```json
{
  "job_id": "def456",
  "stego_url": "/api/stego/def456",
  "psnr": 44.2,
  "pixels_modified": 32768
}
```

### GET `/api/stego/{job_id}`
Returns the stego PNG produced by `/api/embed`.

---

## 6. Working Conventions

### General Context

- **Target Audience:** The UI and demo are for non-technical users. Visual clarity and straightforward copy are paramount.
- **Theme:** Dark, clinical, high-contrast (radar/signals-intelligence aesthetic).

### Python / Backend (FastAPI, PyTorch)

- **Dependencies:** fastapi, uvicorn, python-multipart, torch, torchvision, Pillow, numpy, scipy.
- Maintain strict separation between ML logic (`models/`), dataset generation (`generators/`), and the API layer (`api/`).
- Ensure CORS is configured correctly in `server.py` for local Vite development.

### JavaScript / Frontend (React, Vite)

- **API Calls:** All `fetch()` calls must live in `frontend/src/api/client.js`. Components use custom hooks (e.g., `useAnalysis`) to access these. Throw descriptive errors on non-2xx responses.
- **Styling:** Use standard CSS Modules (`.module.css`). No Tailwind, MUI, Shadcn, or UI component libraries. Build elements from scratch using native CSS custom properties.
- **Caching:** Cache-bust dynamically generated API images by appending timestamps: `?t=${Date.now()}`.
- **State Management:** Managed via a single state machine enum (IDLE, UPLOADING, ANALYZING, COMPLETE, ERROR).
- **Complex UI Elements:** Custom interactive elements (like the Before/After Image Comparison Slider) must be built from scratch using raw DOM events (`onMouseDown`, `onMouseMove`) and `requestAnimationFrame`.

---

## 7. Running the Stack

### Backend (Terminal 1):
```bash
pip install fastapi uvicorn python-multipart torch torchvision pillow numpy scipy
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
python test_kaggle.py --checkpoint srnet_finetuned_best.pth --images 200
```

---

## 8. Critical Protected Files

Unless explicitly instructed otherwise by the user, do not modify the following files:

- `models/srnet.py` — frozen architecture
- `srnet_finetuned_best.pth` — current best after finetune weights
- `srnet_best_val.pth` — current best after training weights
- any `.pth` file — all checkpoint files are training artifacts
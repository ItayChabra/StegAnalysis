# CLAUDE.md — AI Steganalysis System

> This file gives Claude Code the context it needs to work effectively on this project.
> Read it fully before writing any code, creating any file, or proposing any architecture.

---

## 1. Project Overview

This is a full-stack **AI-powered steganography detection system** built for a live demonstration. The system detects hidden data embedded in images using multiple methods, including basic techniques like LSB/DCT/FFT and advanced adaptive algorithms (WOW, S-UNIWARD, HUGO).

The project consists of:
1.  **Backend (Python/PyTorch):** A custom triple-branch convolutional network called **SRNet**.
2.  **API (FastAPI):** A REST interface connecting the ML inference to the client.
3.  **Generators (Python):** Scripts to embed steganographic payloads for testing/demoing, including advanced cost-based adaptive generators.
4.  **Frontend (React/Vite):** A non-technical, highly visual UI designed to make the model's reasoning intuitive for a general audience.

---

## 2. Repository Structure

```
/ (root)
├── CLAUDE.md                  ← you are here
├── main.py                    ← training entry point
├── class_demo.py              ← CLI demo (sliding-window detection)
├── srnet_finetuned_best.pth   ← trained model weights
├── models/
│   └── srnet.py               ← SRNet architecture (triple-branch CNN)
├── generators/
│   ├── lsb_gen.py             ← LSB steganography generator
│   ├── dct_gen.py             ← DCT steganography generator
│   ├── fft_gen.py             ← FFT steganography generator
│   ├── adaptive_gen.py        ← WOW / S-UNIWARD / HUGO steganography generator
│   └── unified_generator.py   ← dispatcher for all generators
├── training/                  ← model training and validation logic
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
| WOW / S-UNIWARD / HUGO | Advanced algorithms that hide data in noisy/textured areas | "Adaptive spatial hiding" |
| PSNR | Signal quality metric; >40 dB = visually identical | "Quality score" |
| Sliding window| Backend ML scan technique (256×256 patches) | **Do not mention in UI** |
| P(stego) | Model's probability that a patch contains hidden data| "Suspicion score" |
| Verdict | Final binary decision: clean or stego | "CLEAN" / "HIDDEN DATA FOUND" |
| Noise map | Amplified pixel residual (stego − cover) × 10 | "What the model sees" |
| Heatmap | Jet-colourmap of per-patch suspicion scores | "Suspicion map" |

---

## 4. API Contract

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
strategy:   "lsb_sequential" | "lsb_edge" | "dct_mid" | "fft_mid" | "wow" | "suniward" | "hugo"
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

## 5. Working Conventions

Depending on which part of the stack you are modifying, adhere to the following rules:

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

## 6. Running the Stack

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

---

## 7. Critical Protected Files

Unless explicitly instructed otherwise by the user, do not modify the following core ML files, as they represent frozen, verified architectures and weights:

- `models/srnet.py`
- `srnet_best_val.pth`
- `srnet_finetuned_best.pth`
- `finetune_history.json`
- any .pth file
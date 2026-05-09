# CLAUDE.md — Steganalysis Frontend (feat/srm-frontend)

> This file gives Claude Code the context it needs to work effectively on this project.
> Read it fully before writing any code, creating any file, or proposing any architecture.

---

## 1. Project Overview

This is the frontend for an **AI-powered steganography detection system** built for a
live university class demonstration. The Python/PyTorch backend runs a custom model
called **SRNet** — a triple-branch convolutional network that detects hidden data
embedded in images using three methods: LSB (pixel-level bit flipping), DCT (JPEG
frequency domain), and FFT (global frequency domain).

**The audience is non-technical.** The frontend must make the model's reasoning
*visible and intuitive*, not just show a number. Every UI decision should ask:
"Would a student with no ML background understand what just happened?"

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
│   └── unified_generator.py   ← dispatcher for all generators
├── training/                  ← all training code (do not modify)
├── frontend/                  ← THIS IS WHERE WE WORK
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── hooks/
│   │   ├── api/               ← all fetch calls live here
│   │   └── main.jsx
│   ├── public/
│   ├── index.html
│   └── package.json
└── api/
    └── server.py              ← FastAPI server (see Section 4)
```

---

## 3. Domain Glossary (use these terms in UI copy)

**The single most important rule:** never surface raw ML terminology to the audience.
Everything passes through this translation table before it appears in the UI.

| Internal term | Plain-English meaning | Show in UI as |
|---------------|-----------------------|---------------|
| Cover image | The original, unmodified image | "Original" |
| Stego image | Image with hidden data embedded | "Modified" |
| LSB | Hides data in the least-significant bit of each pixel | "Pixel-level hiding" |
| DCT | Hides data in JPEG frequency coefficients | "JPEG frequency hiding" |
| FFT | Hides data in global frequency rings | "Frequency-domain hiding" |
| PSNR | Signal quality metric; >40 dB = visually identical | "Quality score" |
| Sliding window | Backend ML scan technique (256×256 patches) | **Do not mention in UI** |
| P(stego) | Model's probability that a patch contains hidden data | "Suspicion score" |
| Verdict | Final binary decision: clean or stego | "CLEAN" / "HIDDEN DATA FOUND" |
| Noise map | Amplified pixel residual (stego − cover) × 10 | "What the model sees" |
| Heatmap | Jet-colourmap of per-patch suspicion scores | "Suspicion map" |

> **Critical naming note:** "Sliding window" is an internal ML term for the backend's
> 256×256 patch scanning process. It must **never** appear in the UI. The UI feature
> that lets users drag a divider to compare two images is called the
> **Image Comparison Slider** — a standard before/after UI pattern that is completely
> separate from the ML concept. Do not conflate these two things.

---

## 4. Backend API Contract

The FastAPI server lives at `http://localhost:8000`.
**All communication is standard HTTP REST — no WebSockets, no polling, no streaming.**
Image uploads use `multipart/form-data`. All other responses are JSON or PNG files.

Add CORS middleware on startup so the Vite dev server can reach the API:

```python
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### POST `/api/analyze`
Upload an image. The server runs the full SRNet inference pipeline synchronously
and returns the complete result in one response. No polling required.

**Request:**
```
Content-Type: multipart/form-data
file: <image binary>
```

**Response:**
```json
{
  "job_id": "abc123",
  "verdict": "CLEAN" | "SUSPICIOUS" | "STEGO_DETECTED",
  "confidence": 0.87,
  "max_window_score": 0.91,
  "mean_window_score": 0.54,
  "flagged_windows": 12,
  "total_windows": 49,
  "window_rows": 7,
  "window_cols": 7,
  "psnr": null,
  "method_hint": "fft_mid" | "lsb_edge" | null,
  "heatmap_url": "/api/heatmap/abc123",
  "noise_map_url": "/api/noisemap/abc123",
  "window_scores": [0.12, 0.45, 0.91, ...]
}
```

`window_scores` is a flat array in row-major order with length
`window_rows × window_cols`. The frontend uses this array to colour the heat grid.

### GET `/api/heatmap/{job_id}`
Returns a PNG — a jet-colourmap heatmap of per-patch suspicion scores overlaid
on the original image. The right half of the Image Comparison Slider shows this
when the "Suspicion map" toggle is active.

### GET `/api/noisemap/{job_id}`
Returns a PNG — the amplified pixel residual (stego − cover) × 10, normalised
and contrast-stretched for visibility. The right half of the Image Comparison
Slider shows this when the "What the model sees" toggle is active. For a clean
cover image with no known stego pair, this endpoint returns a near-grey PNG.

### POST `/api/embed` (demo mode only)
Embeds a secret payload into an uploaded image using one of the project's
existing generators. Returns the stego image URL and quality metadata.

**Request:**
```
Content-Type: multipart/form-data
file:       <image binary>
strategy:   "lsb_sequential" | "lsb_edge" | "dct_mid" | "fft_mid"
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

After a successful embed the frontend fetches the stego image from `stego_url`
and immediately calls `POST /api/analyze` with it, showing the full
embed → detect flow without any manual steps.

### GET `/api/stego/{job_id}`
Returns the stego PNG produced by `/api/embed`. Used both to display the image
in the UI and to re-upload it for analysis.

---

## 5. Frontend Features — Spec

### 5.1 Drop Zone
- Accept a single image: PNG, JPG, or PGM only; max 20 MB
- Validate file type and size on drop; show an inline error for rejections
- Show a thumbnail preview immediately using `URL.createObjectURL`
- Show a large "Analyze →" CTA button below the preview
- While state is `UPLOADING` or `ANALYZING`: disable the button and show a
  spinner; do not allow new drops

### 5.2 Analysis Dashboard
Rendered only when `state === 'COMPLETE'`. Contains five sub-components:

**VerdictBanner**
Large colour-coded chip with icon and verdict text:
- `CLEAN` → `--verdict-clean` (green), icon ✓
- `SUSPICIOUS` → `--verdict-warn` (amber), icon ⚠
- `HIDDEN DATA FOUND` → `--verdict-alert` (red), icon 🔍

Animate in on mount: `opacity 0→1` + `translateY(8px→0)` over 400 ms.

**ConfidenceMeter**
Horizontal bar labelled "Model Confidence". Fills left-to-right on mount over
600 ms via a CSS `width` transition triggered by adding a class after mount.
Bar colour: blue below 50 %, amber 50–75 %, red above 75 %.

**WindowHeatGrid**
CSS grid sized to `window_rows × window_cols` from the API response.
Each cell is coloured by its value in `window_scores`, interpolated in HSL
between `--heatmap-low` (blue, score = 0) and `--heatmap-high` (red, score = 1).
- Cells stagger fade-in: 20 ms delay × cell index
- Hovering a cell shows a tooltip with the exact score as a percentage
- Clicking a cell highlights the corresponding spatial region in the left
  panel of the Image Comparison Slider: draw an absolutely-positioned glowing
  dashed `<div>` overlay at the correct 256×256 patch position

**StatsRow**
Three metric cards in a row:
1. "Flagged Patches" — `flagged_windows / total_windows`
2. "Peak Suspicion" — `max_window_score` as a percentage
3. "Quality Score" — `psnr` value in dB, or "N/A" if null

**MethodBadge** (conditional)
If `method_hint` is non-null, show a small tag below StatsRow:
"Pattern matches: [translated method name]" — use the glossary in Section 3
to translate (e.g. `"fft_mid"` → `"Frequency-domain hiding"`).

### 5.3 Image Comparison Slider
A full-width before/after comparison panel that lets the audience see the
original image and the model's output side by side. This is a standard
drag-to-reveal UI pattern — it has nothing to do with the backend's ML
patch-scanning logic.

**Left half:** the original uploaded image, labelled "What you see"
**Right half:** either the noisemap or heatmap PNG fetched from the API,
  labelled "What the model sees" (noisemap) or "Suspicion map" (heatmap)

Implementation requirements:
- Implement the draggable divider entirely from scratch using `onMouseDown` /
  `onMouseMove` / `onMouseUp` on the container. Support touch events too.
  **No third-party slider or comparison library.**
- Apply `clip-path: inset(0 X% 0 0)` to the right image, where X is derived
  from the divider's position as a percentage of container width.
- Wrap the clip-path update inside a `requestAnimationFrame` callback to
  prevent jank on fast drags.
- The divider handle is a thin vertical line (1 px, `--accent`) with a ↔
  icon centred on it. Add a subtle circular background behind the icon.
- A toggle button above the panel switches the right image between:
  - `/api/noisemap/{job_id}` — label: "What the model sees"
  - `/api/heatmap/{job_id}` — label: "Suspicion map"
  All image URLs must include a `?t=<timestamp>` cache-busting param.
- When a WindowHeatGrid cell is clicked, draw a dashed glowing rectangle
  on the left panel at the corresponding 256×256 patch position. Use an
  absolutely-positioned `<div>` overlay (not canvas) for simplicity.
- Height: 480 px on desktop, 280 px on mobile (≤ 768 px viewport width).

---

## 6. State Machine

A single `useState` enum in `App.jsx` controls which components render.
**There are no WebSockets and no polling.** The `ANALYZING` state covers
the synchronous wait for the `POST /api/analyze` response.

```
IDLE → UPLOADING → ANALYZING → COMPLETE
                                   │
                              (reset button)
                                   │
                                 IDLE

Any state → ERROR → IDLE  (after user dismisses the error)
```

| State | Trigger | What renders |
|-------|---------|--------------|
| `IDLE` | Initial load or reset | DropZone only |
| `UPLOADING` | User clicks "Analyze →" | DropZone (disabled) + spinner |
| `ANALYZING` | File uploaded; waiting for inference response | Full-screen "Analyzing…" indicator |
| `COMPLETE` | API returns result JSON | Dashboard + Image Comparison Slider |
| `ERROR` | Any fetch error or non-2xx response | ErrorBanner + "Try again" button |

During `ANALYZING`, show a pulsing indicator and the copy
**"Model is scanning the image…"** — this is the one place it is acceptable to
use the word "scanning" as it describes what the user is waiting for, not the
ML technique. Do not say "sliding window".

---

## 7. Design Direction

**Aesthetic:** Dark, clinical, high-contrast. Think a radar or signals-intelligence
terminal — not a consumer app. The audience should feel like they are watching
something real and serious.

**Color palette — define all of these as CSS custom properties in `index.css`:**
```css
:root {
  --bg-base:        #0a0c10;
  --bg-surface:     #12151c;
  --bg-elevated:    #1a1e28;
  --border:         #2a2f3d;
  --text-primary:   #e8eaf0;
  --text-secondary: #6b7280;
  --accent:         #3b82f6;
  --verdict-clean:  #22c55e;
  --verdict-warn:   #f59e0b;
  --verdict-alert:  #ef4444;
  --heatmap-low:    #1d4ed8;
  --heatmap-high:   #dc2626;
}
```

**Typography — load from Google Fonts in `index.html`:**
```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=DM+Sans:wght@400;500&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
```
- Headings and verdict text: `'IBM Plex Mono'`
- Body text and labels: `'DM Sans'`
- All numbers, scores, and percentages: `'JetBrains Mono'`

Set these on the root in `index.css`:
```css
body {
  font-family: 'DM Sans', sans-serif;
  background: var(--bg-base);
  color: var(--text-primary);
}
```

**Motion — keep it purposeful:**
- Verdict reveal: `opacity 0→1` + `translateY(8px→0)`, 400 ms ease-out
- Confidence bar fill: CSS `width` transition, 600 ms ease-out on mount
- Heat grid cells: stagger `opacity 0→1`, 20 ms delay × cell index
- Comparison slider divider: update via `requestAnimationFrame`, no CSS
  transition while dragging

---

## 8. Component Hierarchy and File Layout

```
<App>                            ← owns state machine enum
  <Header />                    ← logo, "SRNet Steganalysis", Demo Mode toggle
  <DropZone />                  ← always mounted; disabled when state ≠ IDLE
  <AnalyzingIndicator />        ← only when state === 'ANALYZING'
  <Dashboard>                   ← only when state === 'COMPLETE'
    <VerdictBanner />
    <ConfidenceMeter />
    <WindowHeatGrid />
    <StatsRow />
    <MethodBadge />             ← conditional on method_hint
  </Dashboard>
  <ImageComparisonSlider />     ← only when state === 'COMPLETE'
  <ErrorBanner />               ← only when state === 'ERROR'
</App>
```

File layout inside `frontend/src/`:
```
api/
  client.js                    ← every fetch() call lives here, nowhere else
hooks/
  useAnalysis.js               ← state machine + calls into client.js
pages/
  Home.jsx                     ← assembles the full single page
components/
  Header/
    Header.jsx + Header.module.css
  DropZone/
    DropZone.jsx + DropZone.module.css
  AnalyzingIndicator/
    AnalyzingIndicator.jsx + AnalyzingIndicator.module.css
  Dashboard/
    Dashboard.jsx + Dashboard.module.css
    VerdictBanner.jsx + VerdictBanner.module.css
    ConfidenceMeter.jsx + ConfidenceMeter.module.css
    WindowHeatGrid.jsx + WindowHeatGrid.module.css
    StatsRow.jsx + StatsRow.module.css
    MethodBadge.jsx + MethodBadge.module.css
  ImageComparisonSlider/
    ImageComparisonSlider.jsx + ImageComparisonSlider.module.css
  ErrorBanner/
    ErrorBanner.jsx + ErrorBanner.module.css
```

---

## 9. Working Conventions for Claude Code

- **All `fetch()` calls live in `api/client.js` only.** Components call
  `useAnalysis`; the hook calls `client.js`. No `fetch()` anywhere else.
- **CSS Modules only.** No Tailwind. No inline styles except for dynamic
  values (e.g. `style={{ width: `${pct}%` }}`). Use the CSS custom properties
  from Section 7 inside every `.module.css` file.
- **No UI component libraries.** No MUI, Shadcn, Radix, Headless UI, etc.
  Build every element from scratch so the aesthetic is fully controlled.
- **No WebSockets.** Do not add WebSocket logic anywhere. The API is pure REST.
- **No routing.** Single page, single URL, no React Router.
- **No authentication.** No login, sessions, or API keys.
- **Cache-bust all image URLs** from the API to prevent stale previews:
  ```js
  `${BASE_URL}/api/heatmap/${jobId}?t=${Date.now()}`
  ```
- **Image Comparison Slider must be built from scratch** — mouse/touch events +
  `requestAnimationFrame` + CSS `clip-path`. No libraries.
- **Every `client.js` function must throw a descriptive `Error`** on non-2xx
  responses so `useAnalysis` can catch it and set state to `'ERROR'`.
- **Target layout:** 1280 × 800 px (laptop + projector). Test at this resolution.

---

## 10. Running the Stack

```bash
# Terminal 1 — Backend
# Run from the repo root
pip install fastapi uvicorn python-multipart torch torchvision pillow numpy scipy
uvicorn api.server:app --reload --port 8000

# Terminal 2 — Frontend
cd frontend
npm install
npm run dev
# Opens at http://localhost:5173
```

---

## 11. Files Claude Code Must NOT Modify

- `models/srnet.py`
- `generators/` (any file inside)
- `training/` (any file inside)
- `main.py`
- `class_demo.py`
- `srnet_finetuned_best.pth`
- `finetune_history.json`
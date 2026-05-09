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

| Term | Plain-English meaning | Show in UI as |
|------|-----------------------|---------------|
| Cover image | The original, unmodified image | "Original" |
| Stego image | Image with hidden data embedded | "Modified" |
| LSB | Hides data in the least-significant bit of each pixel | "Pixel-level hiding" |
| DCT | Hides data in JPEG frequency coefficients | "JPEG frequency hiding" |
| FFT | Hides data in global frequency rings | "Frequency-domain hiding" |
| PSNR | Signal quality metric; >40 dB = visually identical | "Quality score" |
| Sliding window | Model scans image in 256×256 patches | Show as animated grid overlay |
| P(stego) | Model's probability that a window contains hidden data | "Suspicion score" |
| Verdict | Final binary decision: clean or stego | "CLEAN" / "HIDDEN DATA FOUND" |

**Never show raw model internals** (logits, softmax, weight names) to the audience.
Translate everything through the glossary above.

---

## 4. Backend API Contract

The FastAPI server lives at `http://localhost:8000`.
All image data is sent as multipart/form-data. All responses are JSON.

### POST `/api/analyze`
Upload an image for full analysis.

**Request:**
```
Content-Type: multipart/form-data
file: <image binary>
```

**Response:**
```json
{
  "verdict": "CLEAN" | "SUSPICIOUS" | "STEGO_DETECTED",
  "confidence": 0.87,          // 0–1, model's certainty
  "max_window_score": 0.91,    // highest patch score
  "mean_window_score": 0.54,   // average patch score
  "flagged_windows": 12,       // windows above threshold
  "total_windows": 49,
  "psnr": 42.3,                // null if cover image
  "method_hint": "fft_mid" | "lsb_edge" | null,  // detected method if known
  "heatmap_url": "/api/heatmap/{job_id}",         // URL to fetch heatmap PNG
  "noise_map_url": "/api/noisemap/{job_id}",      // amplified residual PNG
  "window_scores": [0.12, 0.45, 0.91, ...]        // flat array, row-major
}
```

### GET `/api/heatmap/{job_id}`
Returns a PNG heatmap (jet colormap) of per-window suspicion scores overlaid
on the original image. Used in the Analysis Dashboard.

### GET `/api/noisemap/{job_id}`
Returns a PNG of the amplified pixel residual (stego − cover) × 10. Used in
the Sliding Window viewer to show what the model "sees."

### POST `/api/embed` (demo mode only)
Embed a secret payload into an uploaded image so the audience can see the
before/after.

**Request:**
```json
{
  "strategy": "lsb_sequential" | "lsb_edge" | "dct_mid" | "fft_mid",
  "capacity": 0.5
}
```
Plus `file: <image binary>` in the same multipart form.

**Response:**
```json
{
  "job_id": "abc123",
  "stego_url": "/api/stego/{job_id}",
  "psnr": 44.2,
  "pixels_modified": 32768
}
```

### WebSocket `/ws/analyze/{job_id}`
Streams sliding-window progress events so the UI can animate the scan in real time.

**Event format:**
```json
{ "type": "window_scored", "row": 2, "col": 3, "score": 0.74, "progress": 0.45 }
{ "type": "complete", "verdict": "STEGO_DETECTED" }
```

---

## 5. Frontend Features — Spec

### 5.1 Drop Zone
- Accept single image OR folder (multiple images)
- Show thumbnail preview immediately on drop
- Validate: only PNG, JPG, PGM accepted; max 20 MB
- After drop, show a single "Analyze" CTA button
- For folders: show a count badge ("12 images queued")

### 5.2 Analysis Dashboard
Shown after analysis completes. Contains:

**Header band:** Verdict chip — large, colour-coded
- `CLEAN` → green, with a ✓ icon
- `SUSPICIOUS` → amber, with a ⚠ icon
- `HIDDEN DATA FOUND` → red, with a 🔍 icon

**Confidence meter:** Horizontal bar, 0–100%, labeled "Model Confidence"

**Window grid:** Visual 7×7 grid of coloured cells representing each 256×256 patch.
  Cells colour from cool (blue = clean) to hot (red = suspicious).
  Hovering a cell shows its exact score.

**Stats row (3 cards):**
  - "Flagged Patches" : N / total
  - "Peak Suspicion" : max_window_score as %
  - "Quality Score" : PSNR value (show "N/A" if cover)

**Method tag (optional):** If `method_hint` is returned, show a small badge like
  "Pattern matches: Pixel-level hiding (LSB)"

### 5.3 Sliding Window Viewer
A full-width before/after comparison panel. Left half = original image.
Right half = noise map (amplified residuals) OR heatmap.

- Draggable vertical divider — mouse drag or touch
- Toggle button above: "Noise Map" / "Suspicion Heatmap"
- Optionally animate the scan: as the WebSocket streams window scores, draw
  a glowing 256×256 rectangle marching across the image in scan order.
- Label the left side "What you see" and right side "What the model sees"

### 5.4 Live Scan Animation (bonus, high impact for demo)
While the WebSocket streams events:
- Draw each 256×256 window as a glowing rectangle that briefly flashes its
  score colour before the next window activates.
- A progress bar at the bottom shows scan completion %.
- The verdict chip is revealed with a short animation once `type: "complete"` fires.

---

## 6. Design Direction

**Aesthetic:** Dark, clinical, high-contrast. Think a radar or signals-intelligence
terminal — not a consumer app. The audience should feel like they're watching
something real and serious.

**Color palette (CSS variables):**
```css
--bg-base:       #0a0c10;   /* near-black background */
--bg-surface:    #12151c;   /* card/panel background */
--bg-elevated:   #1a1e28;   /* hover states */
--border:        #2a2f3d;   /* subtle borders */
--text-primary:  #e8eaf0;   /* main text */
--text-secondary:#6b7280;   /* labels, metadata */
--accent:        #3b82f6;   /* blue — scanning / active */
--verdict-clean: #22c55e;   /* green */
--verdict-warn:  #f59e0b;   /* amber */
--verdict-alert: #ef4444;   /* red */
--heatmap-low:   #1d4ed8;   /* cold patches */
--heatmap-high:  #dc2626;   /* hot patches */
```

**Typography:**
- Display / headings: `IBM Plex Mono` (monospace, technical, trustworthy)
- Body: `Inter` or `DM Sans`
- Numbers and scores: `JetBrains Mono` (crisp, readable at small sizes)

**Motion:**
- Scanning animation: use a CSS `box-shadow` pulse on the active window rect
- Verdict reveal: `opacity 0→1` + `translateY(8px→0)` over 400ms
- Heatmap cells: stagger fade-in, 20ms delay per cell
- Slider divider: smooth CSS `transition` on the clip

---

## 7. Component Hierarchy

```
<App>
  <Header />                    ← logo, "SRNet Steganalysis"
  <DropZone />                  ← drag-drop, file picker, thumbnail
  <AnalysisPipeline>            ← orchestrates state machine
    <ScanProgress />            ← WebSocket progress bar + window animation
    <Dashboard>
      <VerdictBanner />
      <ConfidenceMeter />
      <WindowHeatGrid />
      <StatsRow />
      <MethodBadge />
    </Dashboard>
    <SliderViewer />            ← before/after image comparison
  </AnalysisPipeline>
</App>
```

**State machine (use a simple `useState` enum):**
```
IDLE → UPLOADING → SCANNING → COMPLETE → ERROR
```

---

## 8. Working Conventions for Claude Code

- All API calls live in `frontend/src/api/client.js` — no fetch() calls in components.
- Use CSS Modules or plain CSS with the variables above — no Tailwind (class verbosity
  hurts readability in a demo setting).
- The slider viewer must work with just mouse events (no third-party slider library
  needed) — implement it from scratch for reliability.
- Do not add authentication, user accounts, or routing. This is a single-page demo.
- Keep bundle size minimal: no UI component libraries (MUI, Shadcn, etc.).
  Build everything from scratch so the aesthetic is fully controlled.
- All images served from the API use cache-busting query strings to avoid stale previews.
- WebSocket connection must gracefully degrade to polling if WS is unavailable.

---

## 9. Running the Stack

```bash
# Backend
pip install fastapi uvicorn python-multipart torch torchvision pillow numpy scipy
cd api && uvicorn server:app --reload --port 8000

# Frontend
cd frontend && npm install && npm run dev
# → http://localhost:5173
```

---

## 10. Files Claude Code Should NOT Modify

- `models/srnet.py`
- `generators/`
- `training/`
- `main.py`
- `class_demo.py`
- `finetune_history.json`

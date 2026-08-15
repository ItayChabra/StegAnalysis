# StegAnalysis — AI-Powered Steganography Detection

A full-stack system that detects data hidden inside images. A convolutional
neural network (**SRNet**) is trained adversarially against an evolutionary
algorithm that continuously breeds new steganography techniques to fool it, and
the trained detector is served through a FastAPI backend and a React web app.

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

## Contents

- [What it does](#what-it-does)
- [Quick start](#quick-start)
- [Repository layout](#repository-layout)
- [Steganography methods](#steganography-methods)
- [The detector](#the-detector)
- [How it was trained](#how-it-was-trained)
- [Results](#results)
- [API reference](#api-reference)
- [Frontend](#frontend)
- [Model checkpoints](#model-checkpoints)
- [Credits & licence](#credits--licence)

---

## What it does

Three user-facing flows, all backed by the same detector:

| Flow | The user provides | The system returns |
|---|---|---|
| **Analyze** | Any image | A suspicion score, a heatmap showing *where* the suspicion is, a residual view of what the model reacts to, and a frequency spectrum |
| **Embed** | An image + a message | The message encrypted and hidden inside the image, with image-quality (PSNR) and capacity figures |
| **Extract** | A stego image + passphrase | The recovered message — the hiding method is detected automatically |

The interface is written for a non-technical audience: no machine-learning
jargon is shown to the user.

---

## Quick start

**Prerequisites:** Python 3.10+, Node 18+. A GPU is recommended for training;
detection runs fine on CPU.

```bash
# 1. Environment (venv + CUDA PyTorch + dependencies + smoke test)
./setup.sh
source .venv/bin/activate

# 2. Backend
uvicorn api.server:app --reload --port 8000

# 3. Frontend (second terminal)
cd frontend && npm install && npm run dev     # http://localhost:5173
```

---

## Repository layout

```
├── main.py               # training entry point
├── test_kaggle.py        # evaluation against third-party test sets
│
├── models/srnet.py       # the detector architecture
│
├── generators/           # the five steganography methods
│   ├── lsb_gen.py            # spatial least-significant-bit
│   ├── dct_gen.py            # block-DCT (JPEG-style frequency)
│   ├── fft_gen.py            # global FFT frequency bands
│   ├── adaptive_gen.py       # S-UNIWARD content-adaptive
│   ├── steganogan_gen.py     # GAN-learned embedding
│   └── unified_generator.py  # dispatcher
│
├── training/
│   ├── config.py         # all hyperparameters live here
│   ├── train_hybrid.py   # adversarial training loop
│   ├── evolution.py      # the evolutionary algorithm
│   ├── batch.py          # batch construction with diversity guarantees
│   └── finetune.py       # focused fine-tuning
│
├── payload/              # encrypted message codec (header + AES/ChaCha20)
├── api/server.py         # FastAPI application
└── frontend/             # React + Vite web application
```

---

## Steganography methods

Five embedding families are implemented. Capacity is expressed as true
**bits per pixel (bpp)** across all of them.

| Method | Where it hides data | Message recoverable |
|---|---|---|
| **LSB** | The least-significant bit of each pixel | ✅ |
| **DCT** | JPEG-style 8×8 block frequency coefficients | ✅ |
| **FFT** | Global frequency bands of the whole image | ✅ |
| **S-UNIWARD** | Adaptively, in noisy and textured regions | ❌ |
| **SteganoGAN** | Wherever a trained neural encoder chooses | ❌ |

The first three carry a real, recoverable message. S-UNIWARD and SteganoGAN
embed random data — they exist to generate realistic training material for the
detector and are presented in the UI as non-recoverable.

**S-UNIWARD** follows the reference formulation (Daubechies-8 back-convolution
cost model), with the embedding strength solved against ternary entropy so that
the requested bpp is genuine. Calibration was verified against reference files:
0.20 bpp changes ~3.2% of pixels at ~63 dB PSNR.

**SteganoGAN** uses the pretrained DAI-Lab dense encoder, vendored into the
repository and converted to a plain PyTorch state dict so it does not depend on
the upstream package's older PyTorch pin.

---

## The detector

**SRNet** is a triple-branch residual network. Each 256×256 patch enters as a
two-channel tensor — the spatial image plus its frequency-domain (log-FFT)
representation — and three parallel branches read them before merging:

| Branch | Filters | Reads | Purpose |
|---|---|---|---|
| A | 11, **frozen** SRM kernels | spatial | fixed high-pass noise-residual filters |
| B | 53, learnable | spatial | learned spatial residuals |
| C | 21, learnable | log-FFT | frequency-domain artifacts |

Their outputs concatenate into an 85-channel map, pass through ten residual
stages, and end in global average pooling and a binary classifier.

Whole images are scanned with a **256×256 sliding window** and the per-window
probabilities are reduced to a single score by taking the maximum — a hidden
message may occupy only part of an image, so the most suspicious region governs
the verdict.

> The detector is trained on image **luminance** only. All inference paths
> convert to grayscale first; feeding raw colour channels is out-of-distribution
> and degrades accuracy badly.

---

## How it was trained

Training is a co-evolutionary loop. In each generation, a population of
generator "genomes" — each a steganography method plus its parameters — embeds
data into cover images; the detector trains on the result; and the genomes are
then bred and mutated to maximise their success at fooling the *current*
detector. The detector and its adversaries therefore improve together.

Two mechanisms keep this stable:

- **Diversity guarantees.** Batches are constructed so that no single technique
  can dominate, with reserved quotas for the hardest cases — low-payload
  embeddings, weak-strength frequency embeddings, and a guaranteed share for
  S-UNIWARD and SteganoGAN in every batch.
- **A capacity penalty** in the fitness function, which prevents the algorithm
  from "winning" trivially by driving every technique to its lowest, least
  detectable payload.

```bash
python main.py                                   # full training run
python training/evaluate.py                      # per-method evaluation
python training/finetune.py --steganogan-focus   # focused fine-tune
```

All hyperparameters live in [`training/config.py`](training/config.py).

---

## Results

Evaluated on **third-party test images the model never saw during training**
(BOSSbase, BOWS2, Flickr30k and public Kaggle steganography sets), using the
deployed operating point.

| Target | Detection rate |
|---|---|
| Clean images correctly cleared | **96.4%** |
| LSB | **100.0%** |
| DCT | **97.5%** |
| FFT | **97.5%** |
| SteganoGAN | **88.0 – 97.5%** |
| S-UNIWARD @ 0.4 bpp | 14.3% |
| S-UNIWARD @ 0.2 bpp | 1.9% |
| **Balanced accuracy** | **97.4%** |

Detection is essentially saturated on the three classical methods and strong on
GAN-generated stego. Because the test images come from independent sources and
third-party embedding tools, these figures demonstrate generalisation rather
than memorisation of our own generators.

### Known limitation: content-adaptive embedding

S-UNIWARD is the one method the detector does not handle well, and the results
above report that honestly. The signal is present but faint: stego images do
score higher than their matched cover images, and consistently so as the payload
grows, but the margin is small enough that the deployed threshold — chosen to
keep false alarms on clean images low — sits above it. Loosening the threshold
raises 0.4 bpp detection to roughly 47%, at the cost of dropping clean-image
accuracy to 84%.

Low-payload content-adaptive steganography is a recognised open problem in the
steganalysis literature, not an implementation defect. The headline balanced
accuracy above is computed over clean images and the three classical methods,
and **excludes S-UNIWARD** — it is reported separately rather than averaged away.

---

## API reference

FastAPI server on `http://localhost:8000`. Plain HTTP REST.

| Endpoint | Purpose |
|---|---|
| `POST /api/analyze` | Detection. Returns confidence, per-window scores, and URLs for the visualisations |
| `POST /api/embed` | Hide a message. Returns the stego image, PSNR, capacity, and the settings needed to extract |
| `POST /api/extract` | Recover a hidden message, auto-detecting the method |
| `POST /api/decrypt` | Decrypt an already-extracted payload (two-step reveal) |
| `POST /api/capacity` | Maximum message size for a given image and method |
| `GET /health` | Liveness probe |

**`POST /api/embed`** accepts `file`, `method` (`lsb` / `dct` / `fft` /
`adaptive` / `steganogan`), `message`, `cipher` (`none` / `aes256gcm` /
`chacha20poly1305` / `fernet`) and `passphrase`, plus optional per-method
parameters. `adaptive` and `steganogan` are non-recoverable — see
[Steganography methods](#steganography-methods).

**Image endpoints**, all returning PNG for a given `job_id`: `/api/original`,
`/api/stego`, `/api/heatmap`, `/api/noisemap`, `/api/spectrum`, `/api/diff`,
`/api/bitplane/{n}`, `/api/sanitize`.

**Payload format.** `payload/codec.py` prefixes the ciphertext with a 52-byte
self-describing header (magic number, method, cipher, length, salt, nonce,
parameters and a CRC-32 checksum), so an extractor can identify and validate a
payload without being told how it was made. Encryption is authenticated
(AES-256-GCM, ChaCha20-Poly1305 or Fernet), so a wrong passphrase is reported as
such rather than returning garbage.

---

## Frontend

React 18 + Vite, with routes for `/analyze`, `/embed`, `/extract` and `/learn`.
Dark, high-contrast interface.

All network calls are centralised in `src/api/client.js` and consumed through
per-flow state-machine hooks. Styling is plain CSS Modules with no UI component
library, and interactive elements such as the before/after comparison slider are
built directly from DOM events.

---

## Model checkpoints

| File | What it is |
|---|---|
| `srnet_steganogan_best.pth` | **Current best** — the deployed detector |
| `srnet_finetuned_best.pth` | Previous checkpoint, the base the above was fine-tuned from |
| `srnet_best_val.pth` | Best validation checkpoint from the last full training run |
| `steganogan_dense.pth` | Converted SteganoGAN encoder weights |

Per-epoch checkpoints and JSON training histories are also committed, so any
reported figure can be reproduced.

> `models/srnet.py` is treated as frozen — changing the architecture invalidates
> every checkpoint above.

---

## Credits & licence

- **SteganoGAN** — networks under `generators/steganogan_src/` are vendored from
  [DAI-Lab/SteganoGAN](https://github.com/DAI-Lab/SteganoGAN), MIT licence,
  © 2019 MIT Data To AI Lab.
- **S-UNIWARD** — cost model after Holub, Fridrich & Denemark, *Universal
  distortion function for steganography in an arbitrary domain* (2014).
- **SRNet** — architecture after Boroumand, Chen & Fridrich, *Deep residual
  network for steganalysis of digital images* (2019).
- **Datasets** — BOSSbase 1.01, BOWS2, Flickr30k, and public Kaggle
  steganography test sets. Image data is not committed to the repository.

Development conventions and internal notes are in [`CLAUDE.md`](CLAUDE.md).
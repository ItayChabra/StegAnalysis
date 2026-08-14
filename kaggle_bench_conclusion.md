# SteganoGAN Finetune — Benchmark Conclusion

**Benchmark:** `test_kaggle.py`, 200 images/folder, sliding-window (256×256, stride 64).
**Compared:**
- `srnet_finetuned_best.pth` (val_acc 87.21%) — **BEFORE** the SteganoGAN finetune (base checkpoint it started from). → `kaggle_bench_finetuned_best.log`
- `srnet_steganogan_best.pth` (val_acc 88.24%) — **AFTER**, output of the SteganoGAN-focus finetune (45.5% steganogan sampler weight, epoch 8 best). → `kaggle_bench_steganogan_best.log`

Balanced accuracy in the sweep is **basic-driven** `(TNR + TPR_basic)/2` and **excludes adaptive** — adaptive is an informational column only.

---

## 1. Head-to-head at `max @0.80` (the reported "winner" operating point)

| Metric                | BEFORE (finetuned_best) | AFTER (steganogan_best) | Δ        |
|-----------------------|-------------------------|-------------------------|----------|
| TNR (covers)          | 95.3%                   | 96.4%                   | **+1.1** |
| TPR basic (LSB/DCT/FFT)| 98.3%                  | 98.3%                   | 0.0      |
| **TPR adaptive**      | **3.0%**                | **3.2%**                | +0.2     |
| bal-acc               | 96.8%                   | 97.4%                   | **+0.6** |

At the loose end (`max @0.30`) adaptive is 24.8% (before) vs 19.0% (after) — i.e. the finetune **slightly reduced** adaptive sensitivity.

## 2. SteganoGAN detection — the target of the finetune (max-mode median score)

| Variant          | BEFORE | AFTER  | Δ        |
|------------------|--------|--------|----------|
| SGAN dense       | 0.942  | 0.963  | +0.021   |
| **SGAN basic**   | 0.912  | 0.952  | **+0.040** |
| SGAN residual    | 0.955  | 0.950  | −0.005   |
| v3 dense         | 0.936  | 0.965  | +0.029   |
| **v3 basic**     | 0.893  | 0.950  | **+0.057** |
| v3 residual      | 0.958  | 0.952  | −0.006   |

The base model **already detected SteganoGAN well** (~0.89–0.96 median) because GAN residuals overlap learned stego signatures. The finetune's real, consistent gain is on the **weakest variant, `basic`** (+0.04–0.06), and it firmed up `dense`; `residual` was already saturated and moved within noise.

## 3. Covers / TNR

The finetune **lowered cover suspicion scores** (max median): BOSS&BOWS2 0.295→0.237, Flickr30k 0.302→0.218, SGAN cover 0.301→0.225. This is the source of the +1.1 TNR gain — a small, welcome side effect.

## 4. Adaptive (S-UNIWARD) — unchanged and broken in BOTH models

Stego scores are **below the covers'** in both checkpoints (inverted signal, no separating threshold):

| Source              | max median (before) | max median (after) |
|---------------------|---------------------|--------------------|
| BOSS&BOWS2 (cover)  | 0.295               | 0.237              |
| S-UNIWARD 0.2       | 0.164               | 0.121              |
| S-UNIWARD 0.4       | 0.232               | 0.181              |

Adaptive TPR never exceeds ~25% at any usable threshold in either model. The SteganoGAN finetune neither addressed nor could address this.

---

## Verdict

**Keep `srnet_steganogan_best.pth` — it is a mild net improvement over the base:**
- **+ SteganoGAN** solidified, mainly the `basic` encoder variant (~0.90 → 0.95 median).
- **+ TNR** up ~1 point (covers score lower); bal-acc 96.8% → 97.4%.
- **= Basic** (LSB/DCT/FFT) unchanged at ~98–100%.
- **− Adaptive** slightly worse (already near-zero; now marginally lower).

**Caveats for the demo:**
- The finetune's SteganoGAN gain is **incremental**, not transformative — the base already caught GAN stego, so don't overstate the before/after.
- The headline "97.4% bal-acc" is **3 of 4 methods**; it excludes adaptive by construction. At that operating point adaptive detection is 3%.
- `SGAN cover` and `SGANv3 cover` produce byte-identical stats — the two datasets share the same cover source, so the v3 covers are not an independent TNR sample.

**Real open problem (unchanged by this run):** S-UNIWARD is an inverted, undetectable signal in both models. No threshold or aggregation mode rescues it — it needs the canonical S-UNIWARD (`canonical=True`) wiring into training + a retrain, not another SteganoGAN-weighted finetune.

"""
adaptive_gen.py — WOW / S-UNIWARD / HUGO steganography generator.

Three selectable embedding modes (set via genome parameter `adaptive_mode`):

  'wow'       WOW (Wavelet Obtained Weights)
              Cost = sum_k 1 / (|W_k * cover| + sigma_offset)
              Daubechies-8 directional residuals (HL, LH, HH sub-bands).
              Smooth regions → cost → 1/sigma_offset (high, penalised).
              Textured/edge regions → cost → 0 (low, preferred for embedding).

  'suniward'  S-UNIWARD (Spatial UNIWARD)
              Cost = sum_k 1 / (|W_k(cover)| + sigma_offset)
              Note: This is the standard simplified implementation used in practice,
              approximating the relative distortion defined in Holub & Fridrich (2013).

  'hugo'      HUGO-lite (Highly Undetectable steGO)
              Cost = 1 / (RMS of 1st + 2nd order directional differences + sigma_offset)
              Approximate Haralick-feature distance in a 5×5 window.

All three modes share:
  - Daubechies-8 (length-16) directional filter bank (HL, LH, HH sub-bands)
  - Gibbs-optimal ±1 embedding via numerically-stable λ binary search (expit)
  - Spatial wet border: outermost 8 pixels unembeddable (db8 padding artefact zone)
  - Value-saturation guard: pixels at 0 / 255 are unembeddable (WET_COST)
  - Cost floor 1e-3 to prevent lambda-search underflow backdoor on smooth images

Evolvable genome parameters:
  adaptive_mode   : str 'wow' | 'suniward' | 'hugo'
  capacity_ratio  : float 0.20–0.75  (expected fraction of pixels changed)
  sigma_offset    : float 0.5–5.0    (cost floor; controls smooth-region penalty)
  use_diagonal    : bool              (include HH sub-band; WOW/S-UNIWARD only)
  cost_exponent   : float 0.5–2.0    (sharpen/flatten cost distribution)

Note: No message bits are encoded — output is statistically realistic stego
for adversarial training, not bit-recoverable steganography.
"""

import numpy as np
from PIL import Image
from scipy.signal import convolve2d
from scipy.ndimage import uniform_filter
from scipy.special import expit

from generators.base_generator import BaseGenerator

# Daubechies-8 decomposition filters (length 16).
# Hardcoded from PyWavelets db8: dec_hi[n] = (-1)^(n+1) * dec_lo[15-n]
_DB8_DEC_LO = np.array([
    -1.1747678400228192e-04,  6.7544940599858819e-04, -3.9174037299597711e-04,
    -4.8703529930106628e-03,  8.7460940474057759e-03,  1.3981027917015516e-02,
    -4.4088253931064719e-02, -1.7369301002022108e-02,  1.2874742662018601e-01,
     4.7248457399797254e-04, -2.8401554296242810e-01, -1.5829105256023893e-02,
     5.8535468365486910e-01,  6.7563073629801278e-01,  3.1287159091446594e-01,
     5.4415842243081609e-02,
])
_DB8_DEC_HI = np.array([
    -5.4415842243081609e-02,  3.1287159091446594e-01, -6.7563073629801278e-01,
     5.8535468365486910e-01,  1.5829105256023893e-02, -2.8401554296242810e-01,
    -4.7248457399797254e-04,  1.2874742662018601e-01,  1.7369301002022108e-02,
    -4.4088253931064719e-02, -1.3981027917015516e-02,  8.7460940474057759e-03,
     4.8703529930106628e-03, -3.9174037299597711e-04, -6.7544940599858819e-04,
    -1.1747678400228192e-04,
])

_WET_COST    = 1e10               # cost for unembeddable pixels
_WET_BORDER  = 8                  # spatial border width (pixels) — db8 filter radius
_COST_FLOOR  = 1e-3               # minimum cost; prevents lambda-search underflow backdoor
_VALID_MODES = frozenset({'wow', 'suniward', 'hugo'})


class AdaptiveGenerator(BaseGenerator):

    # ------------------------------------------------------------------ utils

    def _load_image_array(self, cover_input):
        if isinstance(cover_input, np.ndarray):
            arr = cover_input.astype(np.uint8)
            return arr[:, :, 0] if arr.ndim == 3 else arr
        if isinstance(cover_input, Image.Image):
            img = cover_input if cover_input.mode == 'L' else cover_input.convert('L')
            return np.array(img, dtype=np.uint8)
        if isinstance(cover_input, str):
            return np.array(Image.open(cover_input).convert('L'), dtype=np.uint8)
        raise ValueError(f"Unsupported cover_input type: {type(cover_input)}")

    def _calculate_psnr(self, original, stego):
        mse = np.mean((original.astype(float) - stego.astype(float)) ** 2)
        return float('inf') if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))

    @staticmethod
    def _conv1d_rows(x, k):
        return convolve2d(x, k[np.newaxis, :], mode='same', boundary='symm')

    @staticmethod
    def _conv1d_cols(x, k):
        return convolve2d(x, k[:, np.newaxis], mode='same', boundary='symm')

    def _db8_subbands(self, img_f, use_diagonal):
        """Return db8 directional sub-band residuals: [HL, LH] or [HL, LH, HH]."""
        HL = self._conv1d_cols(self._conv1d_rows(img_f, _DB8_DEC_HI), _DB8_DEC_LO)
        LH = self._conv1d_cols(self._conv1d_rows(img_f, _DB8_DEC_LO), _DB8_DEC_HI)
        bands = [HL, LH]
        if use_diagonal:
            HH = self._conv1d_cols(self._conv1d_rows(img_f, _DB8_DEC_HI), _DB8_DEC_HI)
            bands.append(HH)
        return bands

    # ------------------------------------------------------------------ WOW cost

    def _compute_wow_cost(self, img_f, use_diagonal=True,
                          sigma_offset=1.0, cost_exponent=1.0):
        """
        WOW: cost_i = sum_k  1 / (|r_k(i)| + sigma_offset)

        Reciprocal weighting over db8 directional sub-bands.
        Smooth regions: residuals ≈ 0 → cost → 1/sigma_offset (high).
        Textured regions: large residuals → cost → 0 (preferred for embedding).
        """
        cost = np.zeros_like(img_f, dtype=np.float64)
        for band in self._db8_subbands(img_f, use_diagonal):
            cost += 1.0 / (np.abs(band) + sigma_offset)

        if cost_exponent != 1.0:
            cost = cost ** cost_exponent

        cost += _COST_FLOOR
        if not np.all(cost > 0):
            raise RuntimeError(f"cost has non-positive entries (min={cost.min():.3g})")
        return cost

    # ------------------------------------------------------------------ S-UNIWARD cost

    def _compute_suniward_cost(self, img_f, use_diagonal=True,
                               sigma_offset=1.0, cost_exponent=1.0):
        weights = [1.0, 1.0, 0.7] if use_diagonal else [1.0, 1.0]
        cost = np.zeros_like(img_f, dtype=np.float64)
        for w, wb in zip(weights, self._db8_subbands(img_f, use_diagonal)):
            cost += w / (np.abs(wb) + sigma_offset)
        if cost_exponent != 1.0:
            cost = cost ** cost_exponent
        cost += _COST_FLOOR
        if not np.all(cost > 0):
            raise RuntimeError(f"suniward cost invalid (min={cost.min():.3g})")
        return cost

    # ------------------------------------------------------------------ HUGO cost

    def _compute_hugo_cost(self, img_f, sigma_offset=1.0, cost_exponent=1.0):
        """
        HUGO-lite: cost based on combined 1st + 2nd order directional difference norms.

        1st-order: 8-connected pixel differences (local gradient)
        2nd-order: Laplacian-like differences in H, V, and 2 diagonal directions
                   d2[i,j] = x[i,j+1] - 2*x[i,j] + x[i,j-1]  (and rotations)

        Together these approximate SPAM feature sensitivity — the key statistical
        property that makes HUGO distinct from WOW/S-UNIWARD. Full HUGO uses
        co-occurrence histograms; this captures the same local complexity signal
        per-pixel without histogram computation.
        """
        h, w = img_f.shape
        pad1 = np.pad(img_f, 1, mode='symmetric')  # (h+2, w+2)
        pad2 = np.pad(img_f, 2, mode='symmetric')  # (h+4, w+4)

        # ---- First-order: 8-directional pixel differences ---------------------
        diffs_1st = np.stack([
            img_f - pad1[0:h, 1:w + 1],  # N
            img_f - pad1[2:h + 2, 1:w + 1],  # S
            img_f - pad1[1:h + 1, 0:w],  # W
            img_f - pad1[1:h + 1, 2:w + 2],  # E
            img_f - pad1[0:h, 0:w],  # NW
            img_f - pad1[0:h, 2:w + 2],  # NE
            img_f - pad1[2:h + 2, 0:w],  # SW
            img_f - pad1[2:h + 2, 2:w + 2],  # SE
        ], axis=0)  # (8, H, W)

        # ---- Second-order: finite 2nd derivatives in 4 directions -------------
        # Indexing: img_f == pad2[2:h+2, 2:w+2]
        d2_h = pad2[2:h + 2, 3:w + 3] - 2 * img_f + pad2[2:h + 2, 1:w + 1]  # horizontal
        d2_v = pad2[3:h + 3, 2:w + 2] - 2 * img_f + pad2[1:h + 1, 2:w + 2]  # vertical
        d2_d1 = pad2[3:h + 3, 3:w + 3] - 2 * img_f + pad2[1:h + 1, 1:w + 1]  # NW-SE diagonal
        d2_d2 = pad2[1:h + 1, 3:w + 3] - 2 * img_f + pad2[3:h + 3, 1:w + 1]  # NE-SW diagonal
        diffs_2nd = np.stack([d2_h, d2_v, d2_d1, d2_d2], axis=0)  # (4, H, W)

        # ---- Combined L2 feature magnitude (12 components) -------------------
        all_diffs = np.concatenate([diffs_1st, diffs_2nd], axis=0)  # (12, H, W)
        feature_mag = np.sqrt((all_diffs ** 2).mean(axis=0))  # (H, W)

        # Low feature_mag → homogeneous → high cost (avoid embedding there)
        cost = uniform_filter(1.0 / (feature_mag + sigma_offset), size=3)

        if cost_exponent != 1.0:
            cost = cost ** cost_exponent

        cost += _COST_FLOOR

        if not np.all(cost > 0):
            raise RuntimeError(f"hugo cost invalid (min={cost.min():.3g})")

        return cost

    # ------------------------------------------------------------------ lambda search

    def _find_lambda(self, cost, payload):
        """
        Binary search for λ s.t.  Σ_i expit(-λ·ρ_i) == payload * N.

        Uses scipy.special.expit throughout — no raw exp() calls, no overflow risk.
        Called on feasible pixels only (wet pixels excluded before this call).
        Returns λ=0 if payload ≥ 0.5 (trivially achievable without any push).
        """
        n = cost.size
        target = payload * n

        if n * 0.5 <= target:
            return 0.0

        lo, hi = 0.0, 50.0
        with np.errstate(over='ignore', under='ignore'):
            for _ in range(14):
                if expit(-hi * cost).sum() <= target:
                    break
                hi *= 2.0

            for _ in range(24):
                lam = (lo + hi) / 2.0
                if expit(-lam * cost).sum() > target:
                    lo = lam
                else:
                    hi = lam

        return (lo + hi) / 2.0

    # ------------------------------------------------------------------ embedding

    def embed(self, cover_input, output_path=None, seed=None, **params):
        """
        Embed a payload using WOW / S-UNIWARD / HUGO cost model.

        Params (genome keys):
            adaptive_mode   : 'wow' | 'suniward' | 'hugo'   (default 'wow')
            capacity_ratio  : float  (default 0.2)
            sigma_offset    : float  (default 1.0)
            use_diagonal    : bool   (default True)
            cost_exponent   : float  (default 1.0)
        seed (int | None): fixes the stochastic embedding for reproducibility.

        Returns:
            (stego_array: np.ndarray, psnr: float)  or  (None, 0) on failure.
        """
        try:
            img = self._load_image_array(cover_input)
        except Exception:
            return None, 0

        img_f   = img.astype(np.float64)
        mode    = params.get('adaptive_mode', 'wow')
        sig_off = params.get('sigma_offset',  1.0)
        diag    = params.get('use_diagonal',  True)
        exp_    = params.get('cost_exponent', 1.0)

        if mode not in _VALID_MODES:
            raise ValueError(
                f"adaptive_mode must be one of {sorted(_VALID_MODES)}, got {mode!r}")
        if not (0.5 <= exp_ <= 2.0):
            raise ValueError(f"cost_exponent must be in [0.5, 2.0], got {exp_!r}")

        # ---- Compute 2-D cost map ----------------------------------------
        if mode == 'suniward':
            cost_2d = self._compute_suniward_cost(
                img_f, use_diagonal=diag, sigma_offset=sig_off, cost_exponent=exp_)
        elif mode == 'hugo':
            cost_2d = self._compute_hugo_cost(
                img_f, sigma_offset=sig_off, cost_exponent=exp_)
        else:
            cost_2d = self._compute_wow_cost(
                img_f, use_diagonal=diag, sigma_offset=sig_off, cost_exponent=exp_)

        # ---- Wet border: spatial edges (db8 padding artefacts) + saturation --
        h, w = img_f.shape
        b = max(1, min(_WET_BORDER, min(h, w) // 16))
        cost_2d[:b, :]  = _WET_COST
        cost_2d[-b:, :] = _WET_COST
        cost_2d[:, :b]  = _WET_COST
        cost_2d[:, -b:] = _WET_COST

        flat_img  = img.flatten().astype(np.int32)
        flat_cost = cost_2d.flatten().copy()

        flat_cost[(flat_img == 0) | (flat_img == 255)] = _WET_COST

        # ---- Find λ and compute per-pixel change probabilities --------------
        payload = params.get('capacity_ratio', 0.2)
        feasible_mask = flat_cost < (_WET_COST * 0.5)
        if feasible_mask.sum() == 0:
            return img.copy(), float('inf')

        lam = self._find_lambda(flat_cost[feasible_mask], payload)
        with np.errstate(over='ignore', under='ignore'):
            p_change = np.clip(expit(-lam * flat_cost), 1e-8, 1 - 1e-8)

        # ---- Stochastic ±1 embedding ----------------------------------------
        rng         = np.random.default_rng(seed)
        change_mask = rng.random(flat_img.size) < p_change
        delta       = rng.choice(np.array([-1, 1]), size=flat_img.size)

        delta[flat_img == 0]   =  1
        delta[flat_img == 255] = -1

        flat_img[change_mask] = np.clip(
            flat_img[change_mask] + delta[change_mask], 0, 255)

        stego = flat_img.reshape(img.shape).astype(np.uint8)
        psnr  = self._calculate_psnr(img, stego)

        if output_path:
            Image.fromarray(stego).save(output_path)

        return stego, psnr

    # ------------------------------------------------------------------ interface

    def run(self, cover_input, output_path, **params):
        return self.embed(cover_input, output_path, **params)
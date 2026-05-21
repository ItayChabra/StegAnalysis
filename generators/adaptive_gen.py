"""
adaptive_gen.py — S-UNIWARD adaptive steganography generator.

S-UNIWARD (Spatial UNIWARD) embedding — two selectable cost models:
  canonical=True  : reference additive distortion (Holub & Fridrich 2014)
      rho = sum_k conv2( 1 / (|conv2(X, F_k)| + 1), rot180(|F_k|) )
    The directional BACK-convolution is the defining UNIWARD step: it spreads
    each sub-band coefficient's reciprocal suppressor over the full filter
    support, summing the relative distortion of every wavelet coefficient a
    pixel flip perturbs. Fixed σ=1, all three sub-bands weight 1.0, no
    cost_exponent. Use this for stego that must match canonical (BOSSbase)
    S-UNIWARD detectors.
  canonical=False : simplified, evolvable approximation (DEFAULT; A/B baseline)
      cost = sum_k w_k / (|W_k(cover)| + sigma_offset)
    Reciprocal of the local Daubechies-8 residual only — NO back-convolution.
    Smooth regions → high cost; textured/edge regions → low cost. sigma_offset,
    cost_exponent and the HH weight (0.7) are live EA knobs here.

  WOW and HUGO were dropped from scope — S-UNIWARD is the single representative
  adaptive method, trained and demoed at high payload.

Mechanics:
  - Daubechies-8 (length-16) directional filter bank (HL, LH, HH sub-bands)
  - Gibbs-optimal ±1 embedding via numerically-stable λ binary search (expit)
  - Spatial wet border: outermost 8 pixels unembeddable (db8 padding artefact zone)
  - Value-saturation guard: pixels at 0 / 255 are unembeddable (WET_COST)
  - Cost floor 1e-3 to prevent lambda-search underflow backdoor on smooth images

Evolvable genome parameters:
  adaptive_mode   : str 'suniward'   (only supported mode)
  capacity_ratio  : float 0.20–0.75  (TRUE payload in bits-per-pixel — see below)
  canonical       : bool  False      (True → reference cost; ignores the knobs below)
  sigma_offset    : float 0.5–5.0    (cost floor; controls smooth-region penalty)
  use_diagonal    : bool             (include HH sub-band)
  cost_exponent   : float 0.5–2.0    (sharpen/flatten cost distribution)

Payload semantics:
  capacity_ratio is a TRUE bits-per-pixel rate. The ±1 embedder is a
  payload-limited sender: λ is solved so the total ternary entropy
  Σ H(p_i) equals capacity_ratio·N (see _find_lambda / _ternary_entropy).
  This matches reference S-UNIWARD and the BOSSbase 0.2/0.4-bpp test sets.
  Ternary ±1 embedding caps at 1.5 bpp; requests above that clamp to λ=0.
  (Earlier revisions treated capacity_ratio as the fraction of pixels
  changed — ~6-15× more distortion than the same number in bpp.)

Note: No message bits are encoded — output is statistically realistic stego
for adversarial training, not bit-recoverable steganography.
"""

import numpy as np
from PIL import Image
from scipy.signal import convolve2d
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
_CANON_SIGMA = 1.0                # canonical S-UNIWARD stabilising constant (fixed)
_VALID_MODES = frozenset({'suniward'})


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

    @staticmethod
    def _db8_directional_filters_2d():
        """The three db8 directional 2-D filters as outer products: LH, HL, HH.

        Same sub-band geometry as _db8_subbands, but materialised as full 2-D
        kernels so the canonical back-convolution can use |F_k| over the whole
        16×16 support.
        """
        lo, hi = _DB8_DEC_LO, _DB8_DEC_HI
        return [np.outer(lo, hi), np.outer(hi, lo), np.outer(hi, hi)]

    # ------------------------------------------------------------------ S-UNIWARD cost

    def _compute_suniward_cost_canonical(self, img_f):
        """Canonical spatial S-UNIWARD additive distortion (Holub & Fridrich 2014).

            rho = sum_k  conv2( 1 / (|conv2(X, F_k)| + σ),  rot180(|F_k|) ),   σ = 1

        The second (back-)convolution is the defining UNIWARD step the simplified
        cost omits: it spreads each sub-band coefficient's reciprocal suppressor
        over the full filter support, summing the relative distortion of every
        wavelet coefficient a pixel flip perturbs.

        Reference-faithful fingerprint: symmetric padding by the filter length,
        zero-boundary 'same' convolutions, the even-length circshift correction
        (db8 = 16 taps → shift by 1 in each axis), then crop to original size.
        All three sub-bands weighted 1.0; no cost_exponent.
        """
        H, W    = img_f.shape
        filters = self._db8_directional_filters_2d()
        pad     = filters[0].shape[0]                  # 16 = filter length
        padded  = np.pad(img_f, pad, mode='symmetric')
        rho     = np.zeros_like(padded)

        for Fk in filters:
            absF = np.abs(Fk)
            R    = convolve2d(padded, Fk, mode='same', boundary='fill')
            S    = 1.0 / (np.abs(R) + _CANON_SIGMA)
            xi   = convolve2d(S, np.rot90(absF, 2), mode='same', boundary='fill')
            # Even-length filters (db8 = 16) carry a half-pixel offset; the
            # reference circshifts xi by 1 in each axis to realign the grid.
            if Fk.shape[0] % 2 == 0:
                xi = np.roll(xi, 1, axis=0)
            if Fk.shape[1] % 2 == 0:
                xi = np.roll(xi, 1, axis=1)
            rho += xi

        return rho[pad:pad + H, pad:pad + W]

    def _compute_suniward_cost(self, img_f, canonical=False, use_diagonal=True,
                               sigma_offset=1.0, cost_exponent=1.0):
        """Per-pixel S-UNIWARD embedding cost.

        canonical=True  → reference additive distortion (back-convolution, σ=1,
                          all sub-bands weight 1, no exponent) — the version that
                          transfers to canonical S-UNIWARD detectors.
        canonical=False → the simplified, evolvable approximation (reciprocal of
                          the local residual; sigma_offset / cost_exponent /
                          HH-weight are live knobs). Kept for A/B testing.
        """
        if canonical:
            cost = self._compute_suniward_cost_canonical(img_f)
        else:
            weights = [1.0, 1.0, 0.7] if use_diagonal else [1.0, 1.0]
            cost = np.zeros_like(img_f, dtype=np.float64)
            for w, wb in zip(weights, self._db8_subbands(img_f, use_diagonal)):
                cost += w / (np.abs(wb) + sigma_offset)
            if cost_exponent != 1.0:
                cost = cost ** cost_exponent
        cost = np.where(np.isfinite(cost), cost, _WET_COST)
        cost += _COST_FLOOR
        if not np.all(cost > 0):
            raise RuntimeError(f"suniward cost invalid (min={cost.min():.3g})")
        return cost

    # ------------------------------------------------------------------ lambda search

    @staticmethod
    def _ternary_entropy(p):
        """
        Payload (bits/pixel) carried by a ±1 embedder at per-pixel change
        probability p — i.e. P(+1)=P(-1)=p/2, P(0)=1-p:

            H(p) = -p·log2(p/2) - (1-p)·log2(1-p)

        Monotonically increasing in p over p∈[0, 0.5]; peaks at H(0.5)=1.5.
        Vectorised; clipped so p=0 / p=1 stay finite (H→0).
        """
        p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0 - 1e-12)
        q = 1.0 - p
        return -p * np.log2(p / 2.0) - q * np.log2(q)

    def _find_lambda(self, cost, payload_bpp):
        """
        Binary-search λ ≥ 0 so the embedded payload equals payload_bpp:

            Σ_i H( expit(-λ·ρ_i) )  ==  payload_bpp · N

        H is the ternary entropy (_ternary_entropy), so payload_bpp is a TRUE
        bits-per-pixel rate — matching reference S-UNIWARD and the BOSSbase
        0.2/0.4-bpp test sets. Σ H is monotonically decreasing in λ.

        Called on feasible pixels only (wet pixels excluded before this call).
        Ternary ±1 embedding caps at 1.5 bpp (λ=0), so payload_bpp ≥ 1.5
        returns λ=0. Uses scipy.special.expit throughout — no raw exp(),
        no overflow risk.
        """
        n      = cost.size
        target = payload_bpp * n

        # λ=0 → every pixel carries H(0.5)=1.5 bits, the ceiling.
        if target >= 1.5 * n:
            return 0.0

        lo, hi = 0.0, 50.0
        with np.errstate(over='ignore', under='ignore'):
            for _ in range(14):
                if self._ternary_entropy(expit(-hi * cost)).sum() <= target:
                    break
                hi *= 2.0

            for _ in range(32):
                lam = (lo + hi) / 2.0
                if self._ternary_entropy(expit(-lam * cost)).sum() > target:
                    lo = lam
                else:
                    hi = lam

        return (lo + hi) / 2.0

    # ------------------------------------------------------------------ embedding

    def embed(self, cover_input, output_path=None, seed=None, **params):
        """
        Embed a payload using the S-UNIWARD cost model.

        Params (genome keys):
            adaptive_mode   : 'suniward'   (only supported mode)
            capacity_ratio  : float  payload in bits-per-pixel (default 0.2)
            canonical       : bool   reference cost w/ back-convolution (default False);
                                     when True, sigma_offset/use_diagonal/cost_exponent
                                     are ignored (σ=1, all sub-bands, exponent 1)
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

        img_f     = img.astype(np.float64)
        mode      = params.get('adaptive_mode', 'suniward')
        sig_off   = params.get('sigma_offset',  1.0)
        diag      = params.get('use_diagonal',  True)
        exp_      = params.get('cost_exponent', 1.0)
        canonical = bool(params.get('canonical', False))

        if mode not in _VALID_MODES:
            raise ValueError(
                f"adaptive_mode must be one of {sorted(_VALID_MODES)}, got {mode!r}")
        if not canonical and not (0.5 <= exp_ <= 2.0):
            raise ValueError(f"cost_exponent must be in [0.5, 2.0], got {exp_!r}")

        # ---- Compute 2-D cost map (S-UNIWARD) ----------------------------
        cost_2d = self._compute_suniward_cost(
            img_f, canonical=canonical, use_diagonal=diag,
            sigma_offset=sig_off, cost_exponent=exp_)

        # ---- Wet border: spatial edges (db8 padding artefacts) + saturation --
        # Canonical S-UNIWARD embeds everywhere (symmetric-padded convolution),
        # so it gets NO hard spatial border — only the 0/255 saturation guard
        # below. The simplified mode keeps the db8 padding-artefact border.
        h, w = img_f.shape
        if not canonical:
            b = max(1, min(_WET_BORDER, min(h, w) // 16))
            cost_2d[:b, :]  = _WET_COST
            cost_2d[-b:, :] = _WET_COST
            cost_2d[:, :b]  = _WET_COST
            cost_2d[:, -b:] = _WET_COST

        flat_img  = img.flatten().astype(np.int32)
        flat_cost = cost_2d.flatten().copy()

        flat_cost[(flat_img == 0) | (flat_img == 255)] = _WET_COST

        # ---- Find λ and compute per-pixel change probabilities --------------
        payload_bpp = params.get('capacity_ratio', 0.2)
        feasible_mask = flat_cost < (_WET_COST * 0.5)
        if feasible_mask.sum() == 0:
            return img.copy(), float('inf')

        lam = self._find_lambda(flat_cost[feasible_mask], payload_bpp)
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
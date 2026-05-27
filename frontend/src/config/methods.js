// Generator metadata: defaults, option lists, and plain-English labels.
// Internal code uses technical terms; `plain` is the audience-facing label
// (per CLAUDE.md §3 translation table).

export const CIPHERS = [
  { id: 'none',             label: 'None (plaintext)' },
  { id: 'aes256gcm',        label: 'AES-256-GCM' },
  { id: 'chacha20poly1305', label: 'ChaCha20-Poly1305' },
  { id: 'fernet',           label: 'Fernet (AES-128-CBC + HMAC)' },
];

export const METHODS = {
  lsb: {
    id: 'lsb',
    name: 'LSB',
    plain: 'Pixel-level hiding',
    recoverable: true,
    blurb: 'Stores bits in the least-significant bit of each pixel.',
    params: [
      {
        key: 'strategy', label: 'Pixel selection', type: 'select',
        default: 'sequential',
        options: [
          { value: 'sequential', label: 'Sequential' },
          { value: 'skip',       label: 'Skip (every Nth pixel)' },
        ],
      },
      {
        key: 'step', label: 'Skip step (N)', type: 'number',
        default: 1, min: 1, max: 64, step: 1,
        showIf: (p) => p.strategy === 'skip',
      },
      {
        key: 'bit_depth', label: 'Bits per pixel', type: 'number',
        default: 1, min: 1, max: 4, step: 1,
      },
    ],
  },
  dct: {
    id: 'dct',
    name: 'DCT',
    plain: 'JPEG frequency hiding',
    recoverable: true,
    blurb: 'Hides bits in 8×8 block frequency coefficients.',
    params: [
      {
        key: 'coeff_selection', label: 'Coefficient band', type: 'select',
        default: 'mid',
        options: [
          { value: 'mid',     label: 'Mid frequencies' },
          { value: 'low_mid', label: 'Low–mid frequencies' },
        ],
      },
      {
        key: 'strength', label: 'Strength (quant. step)', type: 'number',
        default: 3, min: 1, max: 12, step: 0.5,
      },
    ],
  },
  fft: {
    id: 'fft',
    name: 'FFT',
    plain: 'Frequency-domain hiding',
    recoverable: true,
    fragile: true,
    blurb: 'Hides bits in global frequency-ring magnitudes. Recovery is fragile.',
    params: [
      {
        key: 'freq_band', label: 'Frequency band', type: 'select',
        default: 'mid',
        options: [
          { value: 'low',  label: 'Low band' },
          { value: 'mid',  label: 'Mid band' },
          { value: 'high', label: 'High band' },
        ],
      },
      {
        key: 'strength', label: 'Strength (quant. step)', type: 'number',
        default: 8, min: 2, max: 20, step: 1,
      },
    ],
  },
  adaptive: {
    id: 'adaptive',
    name: 'S-UNIWARD',
    plain: 'Adaptive spatial hiding',
    recoverable: false,
    blurb: 'Adaptive algorithm that hides data in noisy/textured areas. '
      + 'Embeds statistical noise only — carries no recoverable message.',
    params: [
      {
        key: 'capacity', label: 'Payload (bpp)', type: 'number',
        default: 0.4, min: 0.1, max: 0.75, step: 0.05,
      },
    ],
  },
};

export const METHOD_ORDER = ['lsb', 'dct', 'fft', 'adaptive'];

// Build a params object filled with defaults for a method.
export function defaultParams(methodId) {
  const out = {};
  for (const p of METHODS[methodId].params) out[p.key] = p.default;
  return out;
}

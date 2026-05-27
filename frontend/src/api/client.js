const BASE_URL = 'http://localhost:8000';

async function _post(path, formData) {
  let res;
  try {
    res = await fetch(`${BASE_URL}${path}`, { method: 'POST', body: formData });
  } catch {
    throw new Error('Cannot reach the backend — make sure the server is running on port 8000.');
  }
  if (!res.ok) {
    // Surface the structured {detail:{error,code}} the API returns where possible.
    let detail = null;
    try { detail = (await res.json())?.detail; } catch { /* not JSON */ }
    const msg = detail?.error || detail || res.statusText;
    const err = new Error(`[${res.status}] ${msg}`);
    err.status = res.status;
    err.code = detail?.code || null;
    err.detail = detail;
    throw err;
  }
  return res.json();
}

// ── Analyze ───────────────────────────────────────────────────────────────────
export async function analyzeImage(file) {
  const form = new FormData();
  form.append('file', file);
  return _post('/api/analyze', form);
}

// ── Embed ─────────────────────────────────────────────────────────────────────
// opts: { method, message, cipher, passphrase, capacity, strategy, step,
//         bit_depth, coeff_selection, strength, freq_band }
export async function embedMessage(file, opts = {}) {
  const form = new FormData();
  form.append('file', file);
  for (const [k, v] of Object.entries(opts)) {
    if (v !== undefined && v !== null && v !== '') form.append(k, String(v));
  }
  return _post('/api/embed', form);
}

// ── Extract ───────────────────────────────────────────────────────────────────
// opts: { passphrase, method?, strategy?, step?, bit_depth?, coeff_selection?,
//         strength?, freq_band? }
export async function extractMessage(file, opts = {}) {
  const form = new FormData();
  form.append('file', file);
  for (const [k, v] of Object.entries(opts)) {
    if (v !== undefined && v !== null && v !== '') form.append(k, String(v));
  }
  return _post('/api/extract', form);
}

// ── Capacity ──────────────────────────────────────────────────────────────────
export async function estimateCapacity(file, method, params = {}) {
  const form = new FormData();
  form.append('file', file);
  form.append('method', method);
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== '') form.append(k, String(v));
  }
  return _post('/api/capacity', form);
}

// ── Image URL builders (cache-busted) ─────────────────────────────────────────
const bust = () => `t=${Date.now()}`;
const withSource = (url, source) =>
  `${url}?${bust()}${source ? `&source=${source}` : ''}`;

export const originalUrl = (jobId) => `${BASE_URL}/api/original/${jobId}?${bust()}`;
export const stegoUrl    = (jobId) => `${BASE_URL}/api/stego/${jobId}?${bust()}`;
export const heatmapUrl  = (jobId) => `${BASE_URL}/api/heatmap/${jobId}?${bust()}`;
export const noisemapUrl = (jobId, source) => withSource(`${BASE_URL}/api/noisemap/${jobId}`, source);
export const diffUrl     = (jobId) => `${BASE_URL}/api/diff/${jobId}?${bust()}`;
export const spectrumUrl = (jobId, source) => withSource(`${BASE_URL}/api/spectrum/${jobId}`, source);
export const bitplaneUrl = (jobId, plane, source) =>
  withSource(`${BASE_URL}/api/bitplane/${jobId}/${plane}`, source);

// ── Sanitize download ─────────────────────────────────────────────────────────
// Fetches the sanitized JPEG and triggers a browser download. Cross-origin
// fetch + blob URL avoids the "download" attribute being ignored for non-same-
// origin hrefs.
export async function downloadSanitized(jobId, originalName = 'image') {
  const res = await fetch(`${BASE_URL}/api/sanitize/${jobId}`);
  if (!res.ok) throw new Error('Sanitize request failed');
  const blob = await res.blob();
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  const base = originalName.replace(/\.[^.]+$/, '');
  a.href     = url;
  a.download = `${base}_sanitized.png`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

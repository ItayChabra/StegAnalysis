const BASE_URL = 'http://localhost:8000';

async function _post(path, formData) {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`[${res.status}] ${path} — ${text || res.statusText}`);
  }
  return res.json();
}

export async function analyzeImage(file) {
  const form = new FormData();
  form.append('file', file);
  return _post('/api/analyze', form);
}

export async function embedPayload(file, strategy, capacity) {
  const form = new FormData();
  form.append('file', file);
  form.append('strategy', strategy);
  form.append('capacity', String(capacity));
  return _post('/api/embed', form);
}

export function originalUrl(jobId) {
  return `${BASE_URL}/api/original/${jobId}?t=${Date.now()}`;
}

export function heatmapUrl(jobId) {
  return `${BASE_URL}/api/heatmap/${jobId}?t=${Date.now()}`;
}

export function noisemapUrl(jobId) {
  return `${BASE_URL}/api/noisemap/${jobId}?t=${Date.now()}`;
}

export function stegoUrl(jobId) {
  return `${BASE_URL}/api/stego/${jobId}?t=${Date.now()}`;
}

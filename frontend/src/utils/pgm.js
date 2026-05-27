// Parse a P5 (binary) or P2 (ASCII) PGM file from an ArrayBuffer.
// Returns a PNG data-URL rendered via an offscreen canvas.
export function parsePgmToDataUrl(buffer) {
  const bytes = new Uint8Array(buffer);
  let pos = 0;

  function skipWsAndComments() {
    while (pos < bytes.length) {
      if (bytes[pos] === 0x23) {
        while (pos < bytes.length && bytes[pos] !== 0x0a) pos++;
      } else if (bytes[pos] <= 0x20) {
        pos++;
      } else {
        break;
      }
    }
  }

  function readToken() {
    skipWsAndComments();
    let tok = '';
    while (pos < bytes.length && bytes[pos] > 0x20) tok += String.fromCharCode(bytes[pos++]);
    return tok;
  }

  const magic = readToken();
  if (magic !== 'P5' && magic !== 'P2') throw new Error('Not a PGM file');

  const width  = parseInt(readToken(), 10);
  const height = parseInt(readToken(), 10);
  const maxval = parseInt(readToken(), 10);
  if (!width || !height || !maxval) throw new Error('Invalid PGM header');

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(width, height);
  const d   = img.data;

  if (magic === 'P5') {
    pos++; // single mandatory whitespace byte between maxval and binary data
    const bps = maxval > 255 ? 2 : 1;
    for (let i = 0; i < width * height; i++) {
      const raw = bps === 2 ? (bytes[pos] << 8) | bytes[pos + 1] : bytes[pos];
      const v   = Math.round(raw / maxval * 255);
      const o   = i * 4;
      d[o] = d[o + 1] = d[o + 2] = v;
      d[o + 3] = 255;
      pos += bps;
    }
  } else {
    const values = new TextDecoder().decode(bytes.slice(pos)).trim().split(/\s+/);
    for (let i = 0; i < width * height; i++) {
      const v = Math.round(parseInt(values[i], 10) / maxval * 255);
      const o = i * 4;
      d[o] = d[o + 1] = d[o + 2] = v;
      d[o + 3] = 255;
    }
  }

  ctx.putImageData(img, 0, 0);
  return canvas.toDataURL('image/png');
}

// Convenience: read a File and resolve to a data-URL (or null on failure).
export async function pgmFileToDataUrl(file) {
  try {
    const buf = await file.arrayBuffer();
    return parsePgmToDataUrl(buf);
  } catch {
    return null;
  }
}

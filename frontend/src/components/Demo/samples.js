// Bundled demo images, served from frontend/public/demo/.
// Three pools: CLEAN (no hidden data), STEGO (verified-detectable by the model),
// REVEAL (recoverable-payload LSB images for the Reveal flow).
// Each demo button picks a RANDOM image from its pool, fresh on every click.

const CLEAN  = Array.from({ length: 10 }, (_, i) => `clean_${String(i + 1).padStart(2, '0')}.png`);
const STEGO  = Array.from({ length: 10 }, (_, i) => `stego_${String(i + 1).padStart(2, '0')}.png`);
const REVEAL = Array.from({ length: 3 },  (_, i) => `reveal_${String(i + 1).padStart(2, '0')}.png`);

// Pick a random file from a pool, avoiding the one last shown from it.
export function randomFrom(pool, exclude) {
  if (!pool || pool.length === 0) return null;
  if (pool.length === 1) return pool[0];
  let pick = exclude;
  while (pick === exclude) pick = pool[Math.floor(Math.random() * pool.length)];
  return pick;
}

// Which demo buttons each flow shows, in order.
export function buttonsFor(kind) {
  if (kind === 'cover')  return [{ key: 'clean',  label: 'Clean photo',     pool: CLEAN  }];
  if (kind === 'reveal') return [{ key: 'reveal', label: 'Hidden message',  pool: REVEAL }];
  // analyze flow → offer both so the user can test detection either way
  return [
    { key: 'clean', label: 'Clean photo', pool: CLEAN },
    { key: 'stego', label: 'Hidden data', pool: STEGO },
  ];
}

export { CLEAN, STEGO, REVEAL };

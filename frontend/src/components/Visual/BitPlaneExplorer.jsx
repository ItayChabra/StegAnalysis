import { useState } from 'react';
import ui from '../../styles/ui.module.css';
import styles from './BitPlaneExplorer.module.css';
import { bitplaneUrl } from '../../api/client.js';

// 8 bit-planes (bit 0 = LSB … bit 7 = MSB). Shows the selected plane and a
// per-plane "randomness" overlay (≈0.5 = noise-like; embedding pushes the LSB
// plane toward maximal randomness). Side-by-side cover/stego in the embed flow.
export default function BitPlaneExplorer({ jobId, scores = [], flow = 'analyze' }) {
  const [plane, setPlane] = useState(0);
  const src = flow === 'embed' ? 'stego' : 'original';

  return (
    <div className={ui.card}>
      <h3 className={ui.cardTitle}>Bit-plane explorer</h3>

      <div className={styles.planes} role="tablist" aria-label="Bit planes">
        {Array.from({ length: 8 }, (_, n) => {
          const sc = scores.find((s) => s.plane === n);
          const r = sc ? sc.randomness : 0;
          return (
            <button
              key={n}
              role="tab"
              aria-selected={plane === n}
              className={`${styles.plane}${plane === n ? ` ${styles.active}` : ''}`}
              onClick={() => setPlane(n)}
              title={sc ? `randomness ${r.toFixed(2)} · entropy ${sc.entropy.toFixed(2)}` : ''}
            >
              <span className={styles.planeNum}>{n}</span>
              <span className={styles.bar}>
                <span className={styles.barFill} style={{ height: `${Math.round(r * 100)}%` }} />
              </span>
              <span className={styles.planeTag}>{n === 0 ? 'LSB' : n === 7 ? 'MSB' : ''}</span>
            </button>
          );
        })}
      </div>

      <div className={flow === 'embed' ? styles.dual : styles.single}>
        {flow === 'embed' && (
          <figure className={styles.fig}>
            <img src={bitplaneUrl(jobId, plane, 'original')} alt={`Original bit-plane ${plane}`} className={styles.img} />
            <figcaption className={styles.cap}>Original</figcaption>
          </figure>
        )}
        <figure className={styles.fig}>
          <img src={bitplaneUrl(jobId, plane, src)} alt={`Bit-plane ${plane}`} className={styles.img} />
          <figcaption className={styles.cap}>{flow === 'embed' ? 'Modified' : `Plane ${plane}`}</figcaption>
        </figure>
      </div>

      <p className={ui.hint}>
        Bar height = how noise-like each plane is. Natural images have structured low planes;
        pixel-level hiding pushes the LSB plane toward pure noise.
      </p>
    </div>
  );
}

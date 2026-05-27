import { useState } from 'react';
import ui from '../../styles/ui.module.css';
import styles from './VisualTools.module.css';
import BeforeAfter from './BeforeAfter.jsx';
import {
  originalUrl, stegoUrl, heatmapUrl, noisemapUrl, diffUrl,
} from '../../api/client.js';

// flow: 'analyze' (uploaded image) | 'embed' (cover vs stego)
export default function VisualTools({ jobId, flow = 'analyze', windowRows, windowCols, selectedPatch }) {
  const tabs = flow === 'embed'
    ? [
        { id: 'compare',   label: 'Before / after' },
        { id: 'diff',      label: 'Pixel diff' },
        { id: 'spatial',   label: 'Spatial residual' },
      ]
    : [
        { id: 'spatial',   label: 'Spatial residual' },
        { id: 'suspicion', label: 'Suspicion' },
      ];

  const [tab, setTab] = useState(tabs[0].id);
  const src = flow === 'embed' ? 'stego' : 'original';

  return (
    <div className={ui.card}>
      <div className={styles.head}>
        <h3 className={ui.cardTitle} style={{ margin: 0 }}>Visual tools</h3>
        <div className={ui.segmented} role="tablist">
          {tabs.map((t) => (
            <button
              key={t.id}
              role="tab"
              aria-selected={tab === t.id}
              className={`${ui.segment}${tab === t.id ? ` ${ui.segmentActive}` : ''}`}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </div>
      </div>

      {tab === 'compare' && (
        <>
          <BeforeAfter
            leftSrc={originalUrl(jobId)} rightSrc={stegoUrl(jobId)}
            leftLabel="Original" rightLabel="Modified"
          />
          <p className={ui.hint}>Drag to compare the original and the modified image — they should look identical.</p>
        </>
      )}

      {tab === 'spatial' && (
        <>
          <BeforeAfter
            leftSrc={flow === 'embed' ? stegoUrl(jobId) : originalUrl(jobId)}
            rightSrc={noisemapUrl(jobId, src)}
            leftLabel={flow === 'embed' ? 'Modified' : 'Original'}
            rightLabel="What the model sees"
          />
          <p className={ui.hint}>The right side is the high-pass (SRM) residual — it amplifies the tiny pixel changes the detector keys off. Drag to compare.</p>
        </>
      )}

      {tab === 'suspicion' && (
        <>
          <BeforeAfter
            leftSrc={originalUrl(jobId)} rightSrc={heatmapUrl(jobId)}
            leftLabel="Original" rightLabel="Suspicion map"
            windowRows={windowRows} windowCols={windowCols} selectedPatch={selectedPatch}
          />
          <p className={ui.hint}>Warmer regions are areas the model finds more suspicious.</p>
        </>
      )}

      {tab === 'diff' && (
        <>
          <div className={styles.single}>
            <img src={diffUrl(jobId)} alt="Pixel difference heatmap" className={styles.img} />
          </div>
          <p className={ui.hint}>Where the modified image differs from the original. Brighter = larger change.</p>
        </>
      )}

    </div>
  );
}

import { useState } from 'react';
import styles from './ImageResultCard.module.css';
import { originalUrl, downloadSanitized } from '../../api/client.js';
import { downloadReport } from './report.js';

const VERDICT = {
  CLEAN:          { label: 'Clean',             cls: 'clean' },
  SUSPICIOUS:     { label: 'Suspicious',        cls: 'warn' },
  STEGO_DETECTED: { label: 'Hidden data found', cls: 'alert' },
};

const NEEDS_SANITIZE = new Set(['SUSPICIOUS', 'STEGO_DETECTED']);

export default function ImageResultCard({ item, selected, onSelect }) {
  const { name, status, result, error } = item;
  const [sanitizing, setSanitizing] = useState(false);

  async function handleSanitize(e) {
    e.stopPropagation();
    setSanitizing(true);
    try { await downloadSanitized(result.job_id, name); }
    finally { setSanitizing(false); }
  }

  return (
    <button
      className={`${styles.card}${selected ? ` ${styles.selected}` : ''}`}
      onClick={() => status === 'done' && onSelect()}
      aria-pressed={selected}
    >
      <div className={styles.thumb}>
        {status === 'done' && result
          ? <img src={originalUrl(result.job_id)} alt={name} />
          : <div className={styles.placeholder}>
              {status === 'running' ? <span className={styles.spin} /> : status === 'error' ? '!' : '…'}
            </div>}
      </div>

      <div className={styles.body}>
        <div className={styles.name} title={name}>{name}</div>

        {status === 'done' && result && (
          <>
            <div className={styles.metaRow}>
              <span className={`${styles.verdict} ${styles[VERDICT[result.verdict]?.cls]}`}>
                {VERDICT[result.verdict]?.label || result.verdict}
              </span>
              <span className={styles.conf}>{Math.round(result.confidence * 100)}%</span>
            </div>
            <div className={styles.metaRow}>
              <span
                className={styles.report}
                role="button"
                tabIndex={0}
                onClick={(e) => { e.stopPropagation(); downloadReport(result); }}
                onKeyDown={(e) => { if (e.key === 'Enter') { e.stopPropagation(); downloadReport(result); } }}
              >
                JSON
              </span>
              {NEEDS_SANITIZE.has(result.verdict) && (
                <span
                  className={`${styles.sanitize}${sanitizing ? ` ${styles.sanitizing}` : ''}`}
                  role="button"
                  tabIndex={0}
                  onClick={handleSanitize}
                  onKeyDown={(e) => { if (e.key === 'Enter') handleSanitize(e); }}
                  title="Download a steganography-stripped copy of this image"
                >
                  {sanitizing ? '…' : 'Sanitize'}
                </span>
              )}
            </div>
          </>
        )}

        {status === 'running' && <div className={styles.status}>Scanning…</div>}
        {status === 'pending' && <div className={styles.status}>Queued</div>}
        {status === 'error' && <div className={styles.statusErr}>{error?.message || 'Failed'}</div>}
      </div>
    </button>
  );
}

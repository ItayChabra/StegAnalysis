import { useState, useEffect } from 'react';
import styles from './AnalyzingIndicator.module.css';

export default function AnalyzingIndicator() {
  const [running, setRunning] = useState(false);

  // Defer one frame so the browser renders the 0% state first,
  // then the CSS transition animates to 90% over 45 s.
  useEffect(() => {
    const id = requestAnimationFrame(() => setRunning(true));
    return () => cancelAnimationFrame(id);
  }, []);

  return (
    <div className={styles.container}>
      <div className={styles.dots} aria-hidden="true">
        <span className={styles.dot} />
        <span className={styles.dot} />
        <span className={styles.dot} />
      </div>

      <p className={styles.label} role="status" aria-live="polite">
        Model is scanning the image…
      </p>

      <div className={styles.progressWrap}>
        <div className={styles.track} aria-hidden="true">
          <div className={`${styles.fill}${running ? ` ${styles.running}` : ''}`} />
        </div>
        <p className={styles.subLabel}>
          Running SRNet inference... This may take up to a minute.
        </p>
      </div>
    </div>
  );
}
